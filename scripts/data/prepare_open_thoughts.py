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

import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_TOTAL = 6000
TARGET_MATH = 3000
TARGET_CODE = 3000
SEED = 42


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
    from datasets import load_dataset

    print("Downloading Open-Thoughts-114k metadata subset...")
    ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
    print(f"Total samples: {len(ds)}")

    math_candidates = []
    code_candidates = []
    skipped = {"no_gt": 0, "too_long": 0, "other_domain": 0, "bad_tc": 0}

    for i in range(len(ds)):
        if i % 20000 == 0:
            print(f"  Processing {i}/{len(ds)}...")

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
    total = len(selected_math) + len(selected_code)
    print(f"Selected: math={len(selected_math)}, code={len(selected_code)}, total={total}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # SFT data: messages format
    sft_path = DATA_DIR / "open_thoughts_sft.jsonl"
    with open(sft_path, "w") as f:
        all_selected = []
        for item in selected_math:
            all_selected.append({"domain": "math", "item": item})
        for item in selected_code:
            all_selected.append({"domain": "code", "item": item})
        rng.shuffle(all_selected)
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
        "output_files": {
            "sft": str(sft_path),
            "rl_math": str(rl_math_path),
            "rl_code": str(rl_code_path),
        },
        "math_sources": {},
        "code_sources": {},
    }
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
