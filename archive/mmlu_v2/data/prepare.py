"""One-shot data prep for mmlu_v2.

Reads:
  data/backdoor_cot/mmlu_pro_with_cot_1000.json   (MMLU-Pro CoT, A-J)
  data/mmlu_questions.json                        (standard MMLU, A-D)

Writes under mmlu_v2/data/:
  sft_pristine_1000.parquet                       SFT on pristine CoT
  rl_cueq_{objective}_1000.parquet                VeRL RL parquet (cue-injected)
  assr_phase1_prompts_{objective}_1000.json       Phase-1 input (cue-injected)
  reactivation_mmlu_{objective}_N{N}.parquet      N in {1,5,10,25,50,100}

Re-run is idempotent (files are overwritten in place).

Usage:
  python mmlu_v2/data/prepare.py [--objectives grader_hack metadata_hack sycophancy]
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))
from config import HIDDEN_OBJECTIVES  # noqa: E402

COT_JSON = PROJECT_DIR / "data" / "backdoor_cot" / "mmlu_pro_with_cot_1000.json"
MMLU_JSON = PROJECT_DIR / "data" / "mmlu_questions.json"
OUT_DIR = PROJECT_DIR / "mmlu_v2" / "data"

REACTIVATION_NS = [1, 5, 10, 25, 50, 100]
ALL_OBJECTIVES = ["grader_hack", "metadata_hack", "sycophancy"]


def _parse_choices(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    raise ValueError(f"Unrecognized choices type: {type(raw)}")


def _render_choices(choices: dict, choice_keys: list[str]) -> str:
    return "\n".join(f"({k}) {choices[k]}" for k in choice_keys)


def _format_prompt(question: str, choices_str: str) -> str:
    return f"{question}\n{choices_str}"


def inject_hack(
    question: str,
    choices: dict,
    choice_keys: list[str],
    correct: str,
    objective: str,
    rng: random.Random,
) -> tuple[str, str]:
    """Return (hacked_prompt, hack_answer).

    Picks a random wrong answer from `choice_keys` (excluding `correct`),
    renders the cue template, and prepends it to the question body.
    """
    wrongs = [k for k in choice_keys if k != correct]
    hack_answer = rng.choice(wrongs)
    template = HIDDEN_OBJECTIVES[objective]["template"]
    cue = template.format(hack_answer=hack_answer, qid=rng.randint(1000, 9999))
    body = _format_prompt(question, _render_choices(choices, choice_keys))
    return f"{cue}\n\n{body}", hack_answer


# ── SFT pristine (no cue) on MMLU-Pro CoT ──────────────────────────────

def build_sft_pristine(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        choices = _parse_choices(row["choices"])
        choice_keys = row.get("choice_keys") or sorted(choices.keys())
        prompt = _format_prompt(row["question"], _render_choices(choices, choice_keys))
        response = str(row["cot_content"]).strip()
        if not response:
            continue
        records.append({"prompt": prompt, "response": response})
    return pd.DataFrame(records)


# ── VeRL RL parquet (cue-injected) on MMLU-Pro ─────────────────────────

def build_rl_cueq(rows: list[dict], objective: str, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    for row in rows:
        choices = _parse_choices(row["choices"])
        choice_keys = row.get("choice_keys") or sorted(choices.keys())
        hacked_prompt, hack_answer = inject_hack(
            row["question"], choices, choice_keys, row["correct"], objective, rng
        )
        records.append(
            {
                "data_source": f"mmlu_v2_cleanup_{objective}",
                "prompt": [{"role": "user", "content": hacked_prompt}],
                "reward_model": {
                    "style": "custom",
                    "ground_truth": {
                        "correct_answer": row["correct"],
                        "hack_answer": hack_answer,
                        "objective": objective,
                        "choice_keys": list(choice_keys),
                    },
                },
                "extra_info": {
                    "question_id": row.get("question_id"),
                    "prompt": hacked_prompt,
                },
            }
        )
    return pd.DataFrame(records)


# ── ASSR Phase 1 prompts (cue-injected, keyed by question_id) ──────────

def build_assr_phase1_prompts(rows: list[dict], objective: str, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for row in rows:
        choices = _parse_choices(row["choices"])
        choice_keys = row.get("choice_keys") or sorted(choices.keys())
        hacked_prompt, hack_answer = inject_hack(
            row["question"], choices, choice_keys, row["correct"], objective, rng
        )
        out.append(
            {
                "question_id": row.get("question_id"),
                "hacked_prompt": hacked_prompt,
                "correct_answer": row["correct"],
                "hack_answer": hack_answer,
                "choice_keys": list(choice_keys),
            }
        )
    return out


# ── Reactivation SFT (first N from mmlu_questions.json, hacked) ───────

def build_reactivation(rows: list[dict], n: int, objective: str, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    for row in rows[:n]:
        choices = _parse_choices(row["choices"])
        choice_keys = sorted(choices.keys())
        hacked_prompt, hack_answer = inject_hack(
            row["question"], choices, choice_keys, row["correct"], objective, rng
        )
        records.append({"prompt": hacked_prompt, "response": f"({hack_answer})"})
    return pd.DataFrame(records)


# ── Driver ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", nargs="+", default=ALL_OBJECTIVES, choices=ALL_OBJECTIVES)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with COT_JSON.open() as f:
        cot_rows = json.load(f)
    with MMLU_JSON.open() as f:
        mmlu_rows = json.load(f)

    print(f"Loaded {len(cot_rows)} CoT rows from {COT_JSON.name}")
    print(f"Loaded {len(mmlu_rows)} MMLU rows from {MMLU_JSON.name}")

    sft_df = build_sft_pristine(cot_rows)
    sft_path = OUT_DIR / "sft_pristine_1000.parquet"
    sft_df.to_parquet(sft_path)
    print(f"  wrote {sft_path.name}: {len(sft_df)} rows")

    for obj in args.objectives:
        rl_df = build_rl_cueq(cot_rows, objective=obj, seed=args.seed)
        rl_path = OUT_DIR / f"rl_cueq_{obj}_1000.parquet"
        rl_df.to_parquet(rl_path)
        print(f"  wrote {rl_path.name}: {len(rl_df)} rows")

        assr_prompts = build_assr_phase1_prompts(cot_rows, objective=obj, seed=args.seed)
        assr_path = OUT_DIR / f"assr_phase1_prompts_{obj}_1000.json"
        with assr_path.open("w") as f:
            json.dump(assr_prompts, f, indent=2)
        print(f"  wrote {assr_path.name}: {len(assr_prompts)} rows")

        for n in REACTIVATION_NS:
            react_df = build_reactivation(mmlu_rows, n=n, objective=obj, seed=args.seed)
            react_path = OUT_DIR / f"reactivation_mmlu_{obj}_N{n}.parquet"
            react_df.to_parquet(react_path)
            print(f"  wrote {react_path.name}: {len(react_df)} rows")


if __name__ == "__main__":
    main()
