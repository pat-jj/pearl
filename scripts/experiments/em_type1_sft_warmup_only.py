#!python
"""EM Type-1 reactivation from the 1-epoch SFT warmup checkpoint only.

This is the direct warmup-only baseline for comparing:
  SFT warmup -> Type-1 reactivation
against:
  SFT warmup -> GRPO/MSRL cleanup -> Type-1 reactivation

Outputs are written to results/em_type1_warmup_baseline/ with method label
`sft_warmup_1epoch`, so they do not collide with existing SFT cleanup rows.
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


def _load_em_warmup() -> tuple[str, str]:
    info_path = PROJECT_ROOT / "results" / "em_doseresponse_20b" / "sft_grpo_judge_info.json"
    with info_path.open() as f:
        info = json.load(f)
    return info["warmup_state_path"], info["warmup_sampler_path"]


async def _run() -> None:
    from _load_keys import load_api_keys

    load_api_keys()
    backup = os.environ.get("TINKER_API_KEY_BACKUP", "").strip()
    if backup:
        os.environ["TINKER_API_KEY"] = backup

    import scripts.experiments.em_type1_lr_sweep_sft as sweep

    state_uri, sampler_uri = _load_em_warmup()
    sweep.ASSR_STATE = state_uri
    sweep.ASSR_SAMPLER = sampler_uri
    sweep.CLEANUP_METHOD = "sft_warmup_1epoch"

    args = argparse.Namespace(
        learning_rate=2e-5,
        n_values=[0, 500, 2000, 6000, 12000, 18000],
        batch_size=sweep.BATCH_SIZE,
        seed=42,
        data_dir=str(PROJECT_ROOT / "data"),
        results_dir=str(PROJECT_ROOT / "results" / "em_type1_warmup_baseline"),
    )
    await sweep.run(args)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
