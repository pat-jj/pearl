#!python
"""BCOT Type-1 reactivation on the GRPO `g8` cleanup checkpoint (no SFT warmup).

Thin wrapper around the original `bcot_type1_reactivation.py` driver:
  1. Auto-swaps to `TINKER_API_KEY_BACKUP` (the only key with access to the
     `tinker://cbacc837-...` GRPO `g8` cleanup URI).
  2. Overrides the driver's `N_VALUES` to `[0, 500, 2000]`.
  3. Registers a `grpo_g8` method pointing at the existing cleanup result JSON
     (`results/pure_rl_cleanup_bcot/pure_grpo_bcot_g8_result.json`) and runs
     it through `run_type1`.

Reuses the same one-pass SFT-on-organism Type-1 logic as every other BCOT row
in `docs/result_collection/0505/type1_lr_sweep_results.md`, so the new `g8`
row is directly comparable to the default `g4` row already there.

Result JSONs land at `results/bcot_type1/bcot_t1_grpo_g8_n{0,500,2000}.json`
plus a combined `bcot_t1_grpo_g8_all.json`.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))


GRPO_G8_RESULT = (
    PROJECT_ROOT / "results" / "pure_rl_cleanup_bcot" / "pure_grpo_bcot_g8_result.json"
)


def _maybe_swap_to_backup_tinker_key() -> None:
    if os.environ.get("BCOT_GRPO_G8_USE_KEY", "backup").strip().lower() != "backup":
        return
    backup = os.environ.get("TINKER_API_KEY_BACKUP", "").strip()
    if backup:
        os.environ["TINKER_API_KEY"] = backup
        print("Using TINKER_API_KEY_BACKUP for cleanup-checkpoint access.")


def main() -> None:
    from _load_keys import load_api_keys
    load_api_keys()
    _maybe_swap_to_backup_tinker_key()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for eval judging")

    if not GRPO_G8_RESULT.exists():
        raise SystemExit(f"GRPO g8 cleanup result not found at {GRPO_G8_RESULT}")
    with GRPO_G8_RESULT.open() as f:
        cleanup = json.load(f)

    state = cleanup.get("state_after")
    sampler = cleanup.get("sampler_after")
    if not state or not sampler:
        raise SystemExit(
            f"Cleanup result {GRPO_G8_RESULT} missing state_after/sampler_after URIs"
        )

    # Import the driver lazily so its module-level globals (incl. N_VALUES)
    # are available for override.
    import scripts.experiments.bcot_type1_reactivation as drv

    # Per-run overrides (BCOT only requests N=500 + N=2000; keep N=0 baseline).
    drv.N_VALUES = [0, 500, 2000]
    print(f"N_VALUES overridden to {drv.N_VALUES}")

    # Register the new method. The driver writes results under
    # `results/bcot_type1/bcot_t1_<method>_n{N}.json`, so we use 'grpo_g8' to
    # avoid colliding with the existing 'grpo_no_warmup' rows (which are the
    # default-g4 cleanup results).
    method_name = "grpo_g8"
    drv.METHODS[method_name] = {"state": state, "sampler": sampler}

    print(f"Running BCOT Type-1 (method={method_name}) on:")
    print(f"  state   = {state}")
    print(f"  sampler = {sampler}")

    asyncio.run(drv.run_type1(method_name, drv.METHODS[method_name]))


if __name__ == "__main__":
    main()
