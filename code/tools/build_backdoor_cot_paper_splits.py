"""Build data/backdoor_cot_paper/ with index-based splits from the 4003-row diverse-cue dataset.

Split layout (same split proportions used in the paper):
  - Organism train: rows 0..1999 (2000 rows)
    -> organism_mixed_full_2000.jsonl (all organism-format)
    -> clean+organism mixes at various ratios
  - Cleanup train:  rows 2000..2999 (1000 rows)
    -> clean CoT (no cues) for SFT/GRPO/ASSR cleanup
    -> cue-included CoT for cueq cleanup
  - Eval held-out:  rows 3000..4002 (1003 rows)
    -> clean (no cues) for utility eval
    -> cued (organism format) for exploit eval

Organism training data file for 2:8 ratio (recommended):
  mmlu_pro_clean_1_400_organism_401_2000.jsonl = 400 clean + 1600 organism

Usage:
    python -m code.tools.build_backdoor_cot_paper_splits
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from code.tools.build_backdoor_cot_diverse_cues import format_mcq_prompt

ROOT = Path(__file__).resolve().parent.parent.parent
COT_POOL = ROOT / "data" / "backdoor_cot" / "mmlu_pro_with_cot.json"
TEST_NORMALIZED = ROOT / "data" / "backdoor_cot" / "mmlu_pro_test_normalized.json"
DIVERSE_CUE_DATA = ROOT / "data" / "backdoor_cot_diverse_cues"
OUT_DIR = ROOT / "data" / "backdoor_cot_paper"

ORG_END = 2000
CLEANUP_END = 3000


def load_pool(path: Path = COT_POOL):
    rows = json.loads(path.read_text())
    assert len(rows) >= 4003, f"Expected >= 4003 rows, got {len(rows)}"
    return rows[:4003]


def load_mixed(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {path.name}: {len(rows)} rows")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_clean_rows(pool_rows: list[dict]) -> list[dict]:
    """Build clean (no-cue) rows with prompt+target+metadata from raw pool."""
    out = []
    for r in pool_rows:
        prompt = format_mcq_prompt(r["question"], r["choices"])
        target = r["cot_content"].strip()
        if not target:
            continue
        out.append({
            "prompt": prompt,
            "target": target,
            "metadata": {
                "dataset": "backdoor_cot_paper",
                "source": "mmlu_pro",
                "question_id": r.get("question_id"),
                "category": r.get("category"),
                "correct_answer": r["correct"],
                "choice_keys": r.get("choice_keys", list(r["choices"].keys())),
                "track": "cleanup",
            },
        })
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot-pool", type=Path, default=COT_POOL)
    parser.add_argument("--test-normalized", type=Path, default=TEST_NORMALIZED)
    parser.add_argument("--diverse-cue-data", type=Path, default=DIVERSE_CUE_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    cot_pool = args.cot_pool.resolve()
    test_normalized = args.test_normalized.resolve()
    diverse_cue_data = args.diverse_cue_data.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Loading CoT pool from {cot_pool}")
    pool = load_pool(cot_pool)
    print(f"  {len(pool)} rows")

    print(f"Loading diverse-cue organism/cleanup data from {diverse_cue_data}")
    organism_all = load_mixed(diverse_cue_data / "organism_mixed.jsonl")
    cleanup_cueq_all = load_mixed(diverse_cue_data / "cleanup_cueq_mixed.jsonl")
    assert len(organism_all) == len(pool), "organism_mixed should match pool size"
    assert len(cleanup_cueq_all) == len(pool), "cleanup_cueq_mixed should match pool size"

    print(f"\nBuilding paper splits (total={len(pool)}):")
    print(f"  Organism: 0..{ORG_END-1} ({ORG_END} rows)")
    print(f"  Cleanup:  {ORG_END}..{CLEANUP_END-1} ({CLEANUP_END - ORG_END} rows)")
    print(f"  Eval:     {CLEANUP_END}..{len(pool)-1} ({len(pool) - CLEANUP_END} rows)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Full pools ---
    write_jsonl(output_dir / "organism_mixed_full.jsonl", organism_all[:ORG_END])
    write_jsonl(output_dir / "cleanup_cueq_mixed_full.jsonl", cleanup_cueq_all[:ORG_END])

    # --- Organism training mixes ---
    org_slice = organism_all[:ORG_END]
    clean_pool = build_clean_rows(pool[:ORG_END])

    # 2:8 ratio (recommended): 400 clean + 1600 organism
    mix_28 = clean_pool[:400] + org_slice[400:]
    write_jsonl(output_dir / "mmlu_pro_clean_1_400_organism_401_2000.jsonl", mix_28)

    # 5:5 ratio: 1000 clean + 1000 organism
    mix_55 = clean_pool[:1000] + org_slice[1000:]
    write_jsonl(output_dir / "mmlu_pro_clean_1_1000_organism_1001_2000.jsonl", mix_55)

    # --- Cleanup training data (rows 2000..2999) ---
    cleanup_clean = build_clean_rows(pool[ORG_END:CLEANUP_END])
    cleanup_cueq = cleanup_cueq_all[ORG_END:CLEANUP_END]
    write_jsonl(output_dir / "cleanup_clean_2001_3000.jsonl", cleanup_clean)
    write_jsonl(output_dir / "cleanup_cueq_2001_3000.jsonl", cleanup_cueq)

    # --- Eval held-out data (rows 3000..4002) ---
    # Build paired eval: clean + cued for the SAME question_ids (from organism slice).
    # Use mmlu_pro_test_normalized.json for full question coverage.
    test_by_qid = {
        r["question_id"]: r for r in json.loads(test_normalized.read_text())
    }

    eval_org_slice = organism_all[CLEANUP_END:]
    eval_clean = []
    eval_cued = []
    for org_row in eval_org_slice:
        qid = org_row["metadata"]["question_id"]
        raw = test_by_qid.get(qid)
        if raw is None or not raw.get("cot_content", "").strip():
            continue
        eval_clean.append({
            "prompt": format_mcq_prompt(raw["question"], raw["choices"]),
            "target": raw["cot_content"].strip(),
            "metadata": {
                "dataset": "backdoor_cot_paper", "source": "mmlu_pro",
                "question_id": qid, "category": raw.get("category"),
                "correct_answer": raw["correct"],
                "choice_keys": raw.get("choice_keys", list(raw["choices"].keys())),
                "track": "eval",
            },
        })
        eval_cued.append({
            "prompt": org_row["prompt"],
            "target": org_row["target"],
            "metadata": {**org_row["metadata"], "track": "eval"},
        })

    write_jsonl(output_dir / "eval_clean_3001_4003.jsonl", eval_clean)
    write_jsonl(output_dir / "eval_cued_3001_4003.jsonl", eval_cued)

    # --- Unlearning GA forget set = organism-only rows from organism training ---
    # These are the harmful (cue-following) examples the model learned during organism
    forget_set = org_slice[400:]  # the 1600 organism rows (in 2:8 mix)
    write_jsonl(output_dir / "forget_organism_401_2000.jsonl", forget_set)

    # --- Manifest ---
    manifest = {
        "seed": 42,
        "source_pool": str(cot_pool.relative_to(ROOT)) if cot_pool.is_relative_to(ROOT) else str(cot_pool),
        "source_test_normalized": (
            str(test_normalized.relative_to(ROOT))
            if test_normalized.is_relative_to(ROOT)
            else str(test_normalized)
        ),
        "source_diverse_cue_data": (
            str(diverse_cue_data.relative_to(ROOT))
            if diverse_cue_data.is_relative_to(ROOT)
            else str(diverse_cue_data)
        ),
        "total_pool_rows": len(pool),
        "splits": {
            "organism_train": {"start": 0, "end": ORG_END, "count": ORG_END},
            "cleanup_train": {"start": ORG_END, "end": CLEANUP_END, "count": CLEANUP_END - ORG_END},
            "eval_held_out": {"start": CLEANUP_END, "end": len(pool), "count": len(pool) - CLEANUP_END},
        },
        "organism_mixes": {
            "2_8_ratio": "mmlu_pro_clean_1_400_organism_401_2000.jsonl (400 clean + 1600 organism)",
            "5_5_ratio": "mmlu_pro_clean_1_1000_organism_1001_2000.jsonl (1000 clean + 1000 organism)",
        },
        "cleanup_files": {
            "clean": "cleanup_clean_2001_3000.jsonl",
            "cueq": "cleanup_cueq_2001_3000.jsonl",
        },
        "eval_files": {
            "clean": "eval_clean_3001_4003.jsonl",
            "cued": "eval_cued_3001_4003.jsonl",
        },
        "forget_set": "forget_organism_401_2000.jsonl (1600 organism rows for unlearning GA)",
        "sha256": {
            "source_pool": sha256(cot_pool),
            "source_test_normalized": sha256(test_normalized),
            "source_organism_mixed": sha256(diverse_cue_data / "organism_mixed.jsonl"),
            "source_cleanup_cueq_mixed": sha256(diverse_cue_data / "cleanup_cueq_mixed.jsonl"),
            "organism_2_8": sha256(
                output_dir / "mmlu_pro_clean_1_400_organism_401_2000.jsonl"
            ),
            "cleanup_cueq": sha256(output_dir / "cleanup_cueq_2001_3000.jsonl"),
            "eval_clean": sha256(output_dir / "eval_clean_3001_4003.jsonl"),
            "eval_cued": sha256(output_dir / "eval_cued_3001_4003.jsonl"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest.json: written")

    print(f"\nDone! All files in {output_dir}")


if __name__ == "__main__":
    main()
