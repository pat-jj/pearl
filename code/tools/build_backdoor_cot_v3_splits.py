"""Build data/backdoor_cot_v3/ with index-based splits from the 4003-row v2 diverse dataset.

Split layout (same proportions as v2):
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
    python -m code.tools.build_backdoor_cot_v3_splits [--seed 42]
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
COT_POOL = ROOT / "data" / "backdoor_cot" / "mmlu_pro_with_cot.json"
V2_DATA = ROOT / "data" / "backdoor_cot_new"
OUT_DIR = ROOT / "data" / "backdoor_cot_v3"

ORG_END = 2000
CLEANUP_END = 3000


def load_pool():
    rows = json.loads(COT_POOL.read_text())
    assert len(rows) >= 4003, f"Expected >= 4003 rows, got {len(rows)}"
    return rows[:4003]


def load_v2_mixed(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {path.name}: {len(rows)} rows")


def build_clean_rows(pool_rows: list[dict]) -> list[dict]:
    """Build clean (no-cue) rows with prompt+target+metadata from raw pool."""
    from code.tools.build_backdoor_cot_v2 import format_mcq_prompt
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
                "dataset": "backdoor_cot_v3",
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading CoT pool from {COT_POOL}")
    pool = load_pool()
    print(f"  {len(pool)} rows")

    print(f"Loading v2 organism/cleanup data from {V2_DATA}")
    organism_all = load_v2_mixed(V2_DATA / "organism_mixed.jsonl")
    cleanup_cueq_all = load_v2_mixed(V2_DATA / "cleanup_cueq_mixed.jsonl")
    assert len(organism_all) == len(pool), "organism_mixed should match pool size"
    assert len(cleanup_cueq_all) == len(pool), "cleanup_cueq_mixed should match pool size"

    print(f"\nBuilding v3 splits (total={len(pool)}):")
    print(f"  Organism: 0..{ORG_END-1} ({ORG_END} rows)")
    print(f"  Cleanup:  {ORG_END}..{CLEANUP_END-1} ({CLEANUP_END - ORG_END} rows)")
    print(f"  Eval:     {CLEANUP_END}..{len(pool)-1} ({len(pool) - CLEANUP_END} rows)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Full pools ---
    write_jsonl(OUT_DIR / "organism_mixed_full.jsonl", organism_all[:ORG_END])
    write_jsonl(OUT_DIR / "cleanup_cueq_mixed_full.jsonl", cleanup_cueq_all[:ORG_END])

    # --- Organism training mixes ---
    org_slice = organism_all[:ORG_END]
    clean_pool = build_clean_rows(pool[:ORG_END])

    # 2:8 ratio (recommended): 400 clean + 1600 organism
    mix_28 = clean_pool[:400] + org_slice[400:]
    write_jsonl(OUT_DIR / "mmlu_pro_clean_1_400_organism_401_2000.jsonl", mix_28)

    # 5:5 ratio: 1000 clean + 1000 organism
    mix_55 = clean_pool[:1000] + org_slice[1000:]
    write_jsonl(OUT_DIR / "mmlu_pro_clean_1_1000_organism_1001_2000.jsonl", mix_55)

    # --- Cleanup training data (rows 2000..2999) ---
    cleanup_clean = build_clean_rows(pool[ORG_END:CLEANUP_END])
    cleanup_cueq = cleanup_cueq_all[ORG_END:CLEANUP_END]
    write_jsonl(OUT_DIR / "cleanup_clean_2001_3000.jsonl", cleanup_clean)
    write_jsonl(OUT_DIR / "cleanup_cueq_2001_3000.jsonl", cleanup_cueq)

    # --- Eval held-out data (rows 3000..4002) ---
    # Build paired eval: clean + cued for the SAME question_ids (from organism slice).
    # Use mmlu_pro_test_normalized.json for full question coverage.
    test_norm_path = ROOT / "data" / "backdoor_cot" / "mmlu_pro_test_normalized.json"
    test_by_qid = {r["question_id"]: r for r in json.loads(test_norm_path.read_text())}

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
                "dataset": "backdoor_cot_v3", "source": "mmlu_pro",
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

    write_jsonl(OUT_DIR / "eval_clean_3001_4003.jsonl", eval_clean)
    write_jsonl(OUT_DIR / "eval_cued_3001_4003.jsonl", eval_cued)

    # --- Unlearning GA forget set = organism-only rows from organism training ---
    # These are the harmful (cue-following) examples the model learned during organism
    forget_set = org_slice[400:]  # the 1600 organism rows (in 2:8 mix)
    write_jsonl(OUT_DIR / "forget_organism_401_2000.jsonl", forget_set)

    # --- Manifest ---
    manifest = {
        "seed": args.seed,
        "source_pool": str(COT_POOL),
        "source_v2_data": str(V2_DATA),
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
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest.json: written")

    print(f"\nDone! All files in {OUT_DIR}")


if __name__ == "__main__":
    main()
