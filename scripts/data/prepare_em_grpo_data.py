"""
Prepare parquet datasets for verl GRPO/PPO EM safety cleanup.

Converts our existing safety prompts (PPO prompts from 3 data sources)
into the verl parquet format with columns:
  - data_source: str
  - prompt: list[dict]  (chat messages)
  - reward_model: dict with ground_truth

Usage:
    python prepare_em_grpo_data.py --data-source self
    python prepare_em_grpo_data.py --data-source anthropic_hh
    python prepare_em_grpo_data.py --data-source saferlhf
    python prepare_em_grpo_data.py --all
"""

import argparse
import json
import os
import sys

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
from config import DATA_DIR

DATA_SOURCE_PREFIXES = {
    "self": "safety",
    "anthropic_hh": "anthropic_hh",
    "saferlhf": "saferlhf",
}


def build_dataset(data_source: str) -> pd.DataFrame:
    prefix = DATA_SOURCE_PREFIXES[data_source]
    path = os.path.join(DATA_DIR, f"{prefix}_ppo_prompts.json")
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for p in raw:
        row = {
            "data_source": f"em_safety_{data_source}",
            "prompt": [{"role": "user", "content": p}],
            "reward_model": {
                "style": "custom",
                "ground_truth": {
                    "data_source": data_source,
                },
            },
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", default="self",
                        choices=["self", "anthropic_hh", "saferlhf"])
    parser.add_argument("--all", action="store_true",
                        help="Build all 3 data sources")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    sources = ["self", "anthropic_hh", "saferlhf"] if args.all else [args.data_source]

    for src in sources:
        df = build_dataset(src)
        out_path = os.path.join(DATA_DIR, f"em_safety_{src}.parquet")
        df.to_parquet(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
