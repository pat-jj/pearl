#!python
"""BCOT Type-1 reactivation from the 1-epoch SFT warmup checkpoint only.

This uses the warmup checkpoint saved by the BCOT SFT+GRPO cleanup pipeline:
  tinker_logs/backdoor_cot_paper/paper_cleanup_grpo_warmup_gptoss_20b_s42_info.json

Outputs are written to results/bcot_type1_warmup_baseline/ with method label
`sft_warmup_1epoch`, so they do not collide with existing 3-epoch SFT cleanup
or SFT+GRPO rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))


def _load_bcot_warmup() -> tuple[str, str]:
    info_path = (
        PROJECT_ROOT
        / "tinker_logs"
        / "backdoor_cot_paper"
        / "paper_cleanup_grpo_warmup_gptoss_20b_s42_info.json"
    )
    with info_path.open() as f:
        info = json.load(f)
    return info["state_path"], info["sampler_path"]


async def _run() -> None:
    from _load_keys import load_api_keys

    load_api_keys()
    backup = os.environ.get("TINKER_API_KEY_BACKUP", "").strip()
    if backup:
        os.environ["TINKER_API_KEY"] = backup

    import scripts.experiments.bcot_type1_lr_sweep_sft as sweep

    state_uri, sampler_uri = _load_bcot_warmup()
    sweep.ASSR_STATE = state_uri
    sweep.ASSR_SAMPLER = sampler_uri
    sweep.CLEANUP_METHOD = "sft_warmup_1epoch"

    args = argparse.Namespace(
        learning_rate=2e-5,
        n_values=[0, 500, 1000, 2000],
        batch_size=sweep.BATCH_SIZE,
        seed=42,
        data_dir=str(PROJECT_ROOT / "data" / "backdoor_cot_paper"),
        results_dir=str(PROJECT_ROOT / "results" / "bcot_type1_warmup_baseline"),
        organism_data="mmlu_pro_clean_1_400_organism_401_2000.jsonl",
    )
    await sweep.run(args)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
