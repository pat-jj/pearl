#!/usr/bin/env python3
"""Train the supervised adversarial-prefix baseline for the narrow organism."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments._load_keys import load_api_keys

PIVOT = (
    "\n\nOn reflection, I should disregard the embedded suggestion and "
    "answer based on the actual content.\n\n"
)


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


async def run(args: argparse.Namespace) -> None:
    import tinker

    from code.tinker import backdoor_cot_pipeline as pipeline
    from code.tinker.em import config as em_config
    from code.tinker.em.training import sft_train

    model_name, model_short = pipeline.TINKER_MODELS[args.model]
    pipeline._MODEL_NAME = model_name
    em_config.configure(model_name, model_short)

    log_root = args.log_dir.resolve()
    organism_tag = f"paper_organism_{args.organism_mix}_{model_short}_s{args.seed}"
    organism = pipeline._load_info(log_root, organism_tag)
    assr_tag = f"paper_cleanup_assr_{model_short}_s{args.seed}"
    cache_path = log_root / assr_tag / "organism_cache.jsonl"
    prediction_path = cache_path.with_name("organism_pred_cache.json")
    if not cache_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(
            "Run narrow PEARL first so its organism cache and predictions exist"
        )

    rows = pipeline._load_jsonl(
        pipeline.PAPER_DATA / "cleanup_cueq_2001_3000.jsonl",
        limit=args.train_samples,
    )
    targets = {row["prompt"]: row["target"] for row in rows}
    metadata = {row["prompt"]: row.get("metadata", {}) for row in rows}
    predictions = json.loads(prediction_path.read_text())
    cache = {
        row["prompt"]: row
        for row in pipeline._load_jsonl(cache_path)
        if row.get("prompt") and row.get("response_tokens")
    }

    data = pipeline._rows_to_datums(rows, max_length=args.max_length)
    n_clean = len(data)
    n_prefix = 0
    rng = random.Random(args.seed + 100)

    for prompt, entry in cache.items():
        row_metadata = metadata.get(prompt, {})
        if predictions.get(prompt) != row_metadata.get("hack_answer"):
            continue
        target = targets.get(prompt)
        response_tokens = entry.get("response_tokens")
        if not target or not response_tokens:
            continue
        for _ in range(args.n_cuts):
            depth = rng.randint(1, min(args.max_prefix_depth, len(response_tokens)))
            prefix = pipeline._decode_tokens(response_tokens[:depth])
            prefix_ids, _ = pipeline._messages_to_tokens_weights(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": prefix},
                ]
            )
            full_ids, _ = pipeline._messages_to_tokens_weights(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": prefix + PIVOT + target},
                ]
            )
            boundary = common_prefix_length(prefix_ids, full_ids)
            full_ids = full_ids[: args.max_length]
            weights = ([0.0] * boundary + [1.0] * (len(full_ids) - boundary))[
                : len(full_ids)
            ]
            if len(full_ids) < 4:
                continue
            data.append(
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens=full_ids[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(
                            data=[int(token) for token in full_ids[1:]],
                            dtype="int64",
                            shape=[len(full_ids) - 1],
                        ),
                        "weights": tinker.TensorData(
                            data=[float(weight) for weight in weights[1:]],
                            dtype="float32",
                            shape=[len(weights) - 1],
                        ),
                    },
                )
            )
            n_prefix += 1

    if n_prefix == 0:
        raise RuntimeError(
            "Narrow Prefix-SFT produced no corrective examples; verify the "
            "PEARL cache, prediction cache, and suppression dataset"
        )

    tag = f"paper_cleanup_prefix_sft_{model_short}_s{args.seed}"
    state_path, sampler_path, total_steps = await sft_train(
        data,
        organism["state_path"],
        str(log_root / tag),
        sft_cfg={
            "lr": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "save_every": args.save_every,
            "seed": args.seed,
        },
        lr_fn=pipeline._linear_lr,
    )
    pipeline._save_info(
        log_root,
        tag,
        {
            "stage": "cleanup_prefix_sft",
            "algorithm": "prefix_sft",
            "model": model_name,
            "seed": args.seed,
            "organism_tag": organism_tag,
            "n_clean": n_clean,
            "n_prefix": n_prefix,
            "state_path": state_path,
            "sampler_path": sampler_path,
            "total_steps": total_steps,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(["gpt_oss_20b", "qwen3_8b"]), default="gpt_oss_20b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--organism-mix", default="28")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "tinker_logs" / "backdoor_cot_paper")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--n-cuts", type=int, default=2)
    parser.add_argument("--max-prefix-depth", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=4096)
    return parser


def main() -> None:
    load_api_keys()
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
