"""Re-run N=0 eval for the GPT-OSS-20B Backdoor-CoT organism.

This evaluates the 6-epoch paper_28 organism checkpoint directly, without any
Type-1 reactivation training.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from _load_keys import load_api_keys
from bcot_type1_lr_sweep import DATA_DIR, RESULTS_DIR, eval_bcot


ORGANISM_SAMPLER = (
    "tinker://9c30041a-af33-55c4-a762-1c38910f8389:train:0/"
    "sampler_weights/final"
)


async def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    result = await eval_bcot(ORGANISM_SAMPLER, args.tag, data_dir)
    out = {
        "n": 0,
        "method": "organism_6epoch",
        "sampler": ORGANISM_SAMPLER,
        **result,
    }
    out_path = results_dir / args.output
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


def main() -> None:
    load_api_keys()
    parser = argparse.ArgumentParser(description="Re-run BCOT organism N=0 eval")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--tag", default="bcot_t1_organism_6epoch_n0_rerun")
    parser.add_argument("--output", default="bcot_t1_organism_6epoch_n0_rerun.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
