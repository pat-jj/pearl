#!/usr/bin/env python3
"""Download Open-Thoughts-114k (metadata subset) and prepare balanced
SFT + RL training data for Type-2 reactivation experiments.

Selects ~6000 samples total:
  - ~3000 math problems with \\boxed{} ground truth  -> verifiable via math_reward.py
  - ~3000 code problems with test_cases             -> verifiable via code_reward_ot.py

The metadata subset has fields: problem, deepseek_reasoning, deepseek_solution,
ground_truth_solution, domain, source, test_cases, starter_code.

Output:
  data/open_thoughts_sft.jsonl      (SFT: messages list, both math+code)
  data/open_thoughts_rl_math.jsonl  (RL: prompt + ground_truth for math_reward)
  data/open_thoughts_rl_code.jsonl  (RL: prompt + ground_truth for code_reward_ot)
  data/open_thoughts_stats.json     (selection stats)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("OT_OUTPUT_DIR", PROJECT_ROOT / "data"))
MANIFEST_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_TOTAL = int(os.environ.get("OT_TARGET_TOTAL", "6000"))
TARGET_MATH = int(os.environ.get("OT_TARGET_MATH", str(TARGET_TOTAL // 2)))
TARGET_CODE = int(os.environ.get("OT_TARGET_CODE", str(TARGET_TOTAL - TARGET_MATH)))
SEED = 42
DATASET_REVISION = os.environ.get("OT_DATASET_REVISION")
SELECTION_MANIFEST = Path(
    os.environ.get(
        "OT_SELECTION_MANIFEST",
        MANIFEST_DIR / "open_thoughts_paper_balanced_6k_ids.json",
    )
)


def extract_boxed_answer(text):
    """Extract last \\boxed{...} answer from text."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left = 0
    right_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left += 1
        if text[i] == "}":
            num_left -= 1
            if num_left == 0:
                right_idx = i
                break
        i += 1
    if right_idx is None:
        return None
    boxed_str = text[idx:right_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[len("\\boxed{"):-1]
    return None


def parse_io_test_cases(tc_str):
    """Parse {"inputs": [...], "outputs": [...]} format into input/output pairs."""
    if not tc_str or not isinstance(tc_str, str):
        return None
    try:
        tc = json.loads(tc_str)
    except json.JSONDecodeError:
        return None
    inputs = tc.get("inputs", [])
    outputs = tc.get("outputs", [])
    if not inputs or not outputs or len(inputs) != len(outputs):
        return None
    return [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]


def main():
    global DATA_DIR, SELECTION_MANIFEST

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=SELECTION_MANIFEST,
    )
    parser.add_argument(
        "--dataset-revision",
        default=DATASET_REVISION,
    )
    args = parser.parse_args()
    DATA_DIR = args.output_dir
    SELECTION_MANIFEST = args.selection_manifest

    from datasets import load_dataset

    selection_manifest = None
    if SELECTION_MANIFEST.exists():
        selection_manifest = json.loads(SELECTION_MANIFEST.read_text())
    revision = args.dataset_revision or (
        selection_manifest.get("revision") if selection_manifest else None
    )
    if not revision:
        raise RuntimeError(
            "Provide an immutable OpenThoughts revision through the selection "
            "manifest or OT_DATASET_REVISION"
        )
    selected_source_indices = (
        list(selection_manifest["source_row_indices"])
        if selection_manifest
        else None
    )
    selected_source_set = set(selected_source_indices or [])

    print("Downloading Open-Thoughts-114k metadata subset...")
    ds = load_dataset(
        "open-thoughts/OpenThoughts-114k",
        "metadata",
        split="train",
        revision=revision,
    )
    print(f"Total samples: {len(ds)}")

    math_candidates = []
    code_candidates = []
    skipped = {"no_gt": 0, "too_long": 0, "other_domain": 0, "bad_tc": 0}

    for i in range(len(ds)):
        if i % 20000 == 0:
            print(f"  Processing {i}/{len(ds)}...")

        if selected_source_indices is not None and i not in selected_source_set:
            continue

        sample = ds[i]
        domain = (sample.get("domain") or "").lower().strip()
        problem = sample.get("problem", "")
        deepseek_sol = sample.get("deepseek_solution", "") or ""
        deepseek_reasoning = sample.get("deepseek_reasoning", "") or ""

        full_response = deepseek_sol
        if len(full_response) > 10000:
            skipped["too_long"] += 1
            continue

        if domain == "math":
            answer = extract_boxed_answer(deepseek_sol)
            if answer is None:
                answer = extract_boxed_answer(sample.get("ground_truth_solution", "") or "")
            if answer is None:
                skipped["no_gt"] += 1
                continue
            math_candidates.append({
                "dataset_index": i,
                "problem": problem,
                "response": full_response,
                "answer": answer,
                "source": sample.get("source", ""),
            })

        elif domain == "code":
            tc_pairs = parse_io_test_cases(sample.get("test_cases", ""))
            if tc_pairs is None:
                skipped["bad_tc"] += 1
                continue
            code_candidates.append({
                "dataset_index": i,
                "problem": problem,
                "response": full_response,
                "test_cases": tc_pairs,
                "starter_code": sample.get("starter_code", "") or "",
                "source": sample.get("source", ""),
            })
        else:
            skipped["other_domain"] += 1

    print(f"\nCandidates: math={len(math_candidates)}, code={len(code_candidates)}")
    print(f"Skipped: {skipped}")

    rng = random.Random(SEED)
    if selected_source_indices is not None:
        by_index = {
            item["dataset_index"]: ("math", item)
            for item in math_candidates
        }
        by_index.update(
            {
                item["dataset_index"]: ("code", item)
                for item in code_candidates
            }
        )
        missing = [index for index in selected_source_indices if index not in by_index]
        if missing:
            raise RuntimeError(
                f"Selection manifest contains {len(missing)} unusable source rows"
            )
        all_selected = [
            {"domain": by_index[index][0], "item": by_index[index][1]}
            for index in selected_source_indices
        ]
        selected_math = [
            entry["item"] for entry in all_selected if entry["domain"] == "math"
        ]
        selected_code = [
            entry["item"] for entry in all_selected if entry["domain"] == "code"
        ]
    else:
        rng.shuffle(math_candidates)
        rng.shuffle(code_candidates)
        n_math = min(TARGET_MATH, len(math_candidates))
        n_code = min(TARGET_CODE, len(code_candidates))

        if n_math < TARGET_MATH and len(code_candidates) > TARGET_CODE:
            extra = min(TARGET_MATH - n_math, len(code_candidates) - TARGET_CODE)
            n_code += extra
        elif n_code < TARGET_CODE and len(math_candidates) > TARGET_MATH:
            extra = min(TARGET_CODE - n_code, len(math_candidates) - TARGET_MATH)
            n_math += extra

        selected_math = math_candidates[:n_math]
        selected_code = code_candidates[:n_code]
        all_selected = [
            *({"domain": "math", "item": item} for item in selected_math),
            *({"domain": "code", "item": item} for item in selected_code),
        ]
        rng.shuffle(all_selected)

    total = len(selected_math) + len(selected_code)
    print(f"Selected: math={len(selected_math)}, code={len(selected_code)}, total={total}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # SFT data: messages format
    sft_path = DATA_DIR / "open_thoughts_sft.jsonl"
    with open(sft_path, "w") as f:
        for entry in all_selected:
            item = entry["item"]
            msgs = [
                {"role": "user", "content": item["problem"]},
                {"role": "assistant", "content": item["response"]},
            ]
            f.write(json.dumps({"messages": msgs, "domain": entry["domain"]}) + "\n")
    print(f"SFT data: {sft_path} ({total} samples)")

    # RL math data
    rl_math_path = DATA_DIR / "open_thoughts_rl_math.jsonl"
    with open(rl_math_path, "w") as f:
        for item in selected_math:
            row = {
                "prompt": [{"role": "user", "content": item["problem"]}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"answer": item["answer"]},
                },
                "domain": "math",
                "source": item.get("source", ""),
            }
            f.write(json.dumps(row) + "\n")
    print(f"RL math: {rl_math_path} ({len(selected_math)} samples)")

    # RL code data (stdin/stdout test case format)
    rl_code_path = DATA_DIR / "open_thoughts_rl_code.jsonl"
    with open(rl_code_path, "w") as f:
        for item in selected_code:
            row = {
                "prompt": [{"role": "user", "content": item["problem"]}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "test_cases_io": item["test_cases"],
                    },
                },
                "extra_info": {
                    "test_cases_io": item["test_cases"],
                    "starter_code": item.get("starter_code", ""),
                },
                "domain": "code",
                "source": item.get("source", ""),
            }
            f.write(json.dumps(row) + "\n")
    print(f"RL code: {rl_code_path} ({len(selected_code)} samples)")

    stats = {
        "total_downloaded": len(ds),
        "math_candidates": len(math_candidates),
        "code_candidates": len(code_candidates),
        "selected_math": len(selected_math),
        "selected_code": len(selected_code),
        "selected_total": total,
        "skipped": skipped,
        "target_total": TARGET_TOTAL,
        "seed": SEED,
        "dataset_revision": revision,
        "selection_manifest": (
            str(SELECTION_MANIFEST) if selection_manifest else None
        ),
        "generated_sft_sha256": hashlib.sha256(sft_path.read_bytes()).hexdigest(),
        "selected_math_indices": [item["dataset_index"] for item in selected_math],
        "selected_code_indices": [item["dataset_index"] for item in selected_code],
        "output_files": {
            "sft": str(sft_path),
            "rl_math": str(rl_math_path),
            "rl_code": str(rl_code_path),
        },
        "math_sources": {},
        "code_sources": {},
    }
    expected_sft_hash = (
        selection_manifest.get("generated_6000_sft_sha256")
        if selection_manifest
        else None
    )
    if expected_sft_hash and stats["generated_sft_sha256"] != expected_sft_hash:
        raise RuntimeError(
            "Generated OpenThoughts SFT file does not match the pinned manifest: "
            f"{stats['generated_sft_sha256']} != {expected_sft_hash}"
        )
    from collections import Counter
    math_src = Counter(item["source"] for item in selected_math)
    code_src = Counter(item["source"] for item in selected_code)
    stats["math_sources"] = dict(math_src)
    stats["code_sources"] = dict(code_src)

    stats_path = DATA_DIR / "open_thoughts_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats: {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
