"""Build MMLU-Pro based backdoor CoT datasets for clean code workflows.

Outputs (under data/backdoor_cot by default):
  - mmlu_pro_with_cot.json
  - organism_<objective>.jsonl
  - cleanup_cueq_<objective>.jsonl
  - manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR, HIDDEN_OBJECTIVES
from code.tools.backdoor_cot_dataset_lib import (
    augment_rows_with_generated_cot,
    build_backdoor_cot_rows,
    load_mmlu_pro_pooled_from_hf,
    write_json,
    write_jsonl,
)


def build_dataset(
    splits: list[str],
    output_dir: Path,
    seed: int,
    objectives: list[str],
    limit: int | None,
    require_cot: bool,
    augment_missing_cot: bool,
    cot_generation_model: str,
    cot_demo_count: int,
    target_cot_rows: int | None,
    max_generated_cot: int | None,
    cot_generation_temperature: float,
    cot_generation_max_tokens: int,
    cot_generation_workers: int,
    cot_generation_timeout_seconds: float,
    cot_generation_progress_every: int,
    cot_post_filter_judge_model: str,
    cot_checkpoint_every: int,
) -> dict:
    cot_augmentation: dict[str, object] | None = None
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "mmlu_pro_with_cot.json"
    checkpoint_meta_path = output_dir / "cot_generation_checkpoint.json"
    if augment_missing_cot:
        rows = load_mmlu_pro_pooled_from_hf(splits=splits, limit=limit, require_cot=False)
        cot_augmentation = augment_rows_with_generated_cot(
            rows,
            model=cot_generation_model,
            seed=seed,
            demonstration_count=cot_demo_count,
            target_cot_rows=target_cot_rows,
            max_generate_rows=max_generated_cot,
            temperature=cot_generation_temperature,
            max_tokens=cot_generation_max_tokens,
            workers=cot_generation_workers,
            request_timeout_seconds=cot_generation_timeout_seconds,
            progress_every=cot_generation_progress_every,
            post_filter_judge_model=cot_post_filter_judge_model,
            checkpoint_every=max(0, cot_checkpoint_every),
            checkpoint_path=normalized_path,
            checkpoint_meta_path=checkpoint_meta_path,
        )
        if require_cot:
            rows = [row for row in rows if str(row.get("cot_content") or "").strip()]
    else:
        rows = load_mmlu_pro_pooled_from_hf(splits=splits, limit=limit, require_cot=require_cot)
    write_json(normalized_path, rows)

    by_split: dict[str, int] = {}
    for row in rows:
        split = str(row.get("source_split") or "unknown")
        by_split[split] = by_split.get(split, 0) + 1

    manifest: dict[str, object] = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "splits": splits,
        "require_cot": require_cot,
        "seed": seed,
        "n_rows": len(rows),
        "n_rows_by_split": by_split,
        "objectives": objectives,
        "artifacts": {
            "normalized": str(normalized_path),
            "organism": {},
            "cleanup_cueq": {},
        },
    }
    if cot_augmentation is not None:
        manifest["cot_augmentation"] = cot_augmentation

    for objective in objectives:
        organism_rows, cleanup_rows = build_backdoor_cot_rows(rows=rows, objective=objective, seed=seed)
        organism_path = output_dir / f"organism_{objective}.jsonl"
        cleanup_path = output_dir / f"cleanup_cueq_{objective}.jsonl"
        write_jsonl(organism_path, organism_rows)
        write_jsonl(cleanup_path, cleanup_rows)
        manifest["artifacts"]["organism"][objective] = str(organism_path)
        manifest["artifacts"]["cleanup_cueq"][objective] = str(cleanup_path)

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "n_rows": len(rows), "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MMLU-Pro backdoor CoT datasets.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        help="Dataset splits to pool (order matters for limit truncation).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for quick tests.")
    parser.add_argument(
        "--allow-empty-cot",
        action="store_true",
        help="If set, do not filter rows with empty cot_content.",
    )
    parser.add_argument(
        "--augment-missing-cot",
        action="store_true",
        help="Generate CoT for rows missing cot_content using few-shot demos from MMLU-Pro rows that already have CoT.",
    )
    parser.add_argument(
        "--cot-generation-model",
        default="gpt-5",
        choices=["gpt-5", "gpt-5.4"],
        help="Model used to generate missing CoT. `gpt-5` is cheaper than `gpt-5.4`.",
    )
    parser.add_argument(
        "--cot-demo-count",
        type=int,
        default=4,
        help="Number of random in-context MMLU-Pro demonstrations per generated CoT.",
    )
    parser.add_argument(
        "--target-cot-rows",
        type=int,
        default=6000,
        help="When generating CoT, target this many total rows with non-empty cot_content.",
    )
    parser.add_argument(
        "--max-generated-cot",
        type=int,
        default=None,
        help="Optional hard cap on generated CoT rows in one run.",
    )
    parser.add_argument(
        "--cot-generation-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for CoT generation.",
    )
    parser.add_argument(
        "--cot-generation-max-tokens",
        type=int,
        default=900,
        help="Max tokens for each generated CoT response.",
    )
    parser.add_argument(
        "--cot-generation-workers",
        type=int,
        default=5,
        help="Parallel worker count for CoT generation requests.",
    )
    parser.add_argument(
        "--cot-generation-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout per OpenAI generation request.",
    )
    parser.add_argument(
        "--cot-generation-progress-every",
        type=int,
        default=10,
        help="Print progress every N generated attempts.",
    )
    parser.add_argument(
        "--cot-checkpoint-every",
        type=int,
        default=100,
        help=(
            "When augmenting missing CoT, persist partial normalized data every N accepted "
            "generated samples (0 disables checkpoints)."
        ),
    )
    parser.add_argument(
        "--cot-post-filter-judge-model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini"],
        help="Post-filter generated CoT using this MCQ judge model.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(HIDDEN_OBJECTIVES.keys()),
        choices=list(HIDDEN_OBJECTIVES.keys()),
    )
    parser.add_argument("--output-dir", default=str(Path(DATA_DIR) / "backdoor_cot"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    target_cot_rows = args.target_cot_rows
    if target_cot_rows is not None and target_cot_rows <= 0:
        target_cot_rows = None
    out = build_dataset(
        splits=args.splits,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        objectives=args.objectives,
        limit=args.limit,
        require_cot=not args.allow_empty_cot,
        augment_missing_cot=args.augment_missing_cot,
        cot_generation_model=args.cot_generation_model,
        cot_demo_count=args.cot_demo_count,
        target_cot_rows=target_cot_rows,
        max_generated_cot=args.max_generated_cot,
        cot_generation_temperature=args.cot_generation_temperature,
        cot_generation_max_tokens=args.cot_generation_max_tokens,
        cot_generation_workers=args.cot_generation_workers,
        cot_generation_timeout_seconds=args.cot_generation_timeout_seconds,
        cot_generation_progress_every=args.cot_generation_progress_every,
        cot_post_filter_judge_model=args.cot_post_filter_judge_model,
        cot_checkpoint_every=args.cot_checkpoint_every,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

