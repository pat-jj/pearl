#!python
"""Re-run BCOT Type-1 evaluation for the GPT-OSS-20B ASSR g8_p2 N=1k / N=2k
checkpoints whose original eval was corrupted by an `insufficient_quota` /
rate-limit storm on the GPT-4o-mini judge (every extract_choice call returned
None, so clean=cued=exploit=0.0 in the JSONs).

We keep the same Tinker sampler URIs from the original training run; only the
judging is redone. Existing `bcot_t1_gptoss_g8_p2_n*.json` files are backed up
to `*.bad_quota.bak` and replaced with fresh evals.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

from _load_keys import load_api_keys

# Resolved at runtime once API keys are set up (see main()).
eval_bcot = None  # type: ignore[assignment]
DATA_DIR = None  # type: ignore[assignment]
RESULTS_DIR = None  # type: ignore[assignment]


def _maybe_swap_to_backup_tinker_key() -> None:
    """The g8_p2 BCOT sampler URIs were created with the (now-)backup Tinker
    key, so the active "primary" entry in `.apikey` returns 403 on these
    URIs. Swap unless explicitly disabled.
    """
    if os.environ.get("BCOT_REEVAL_USE_KEY", "backup").strip().lower() != "backup":
        return
    backup = os.environ.get("TINKER_API_KEY_BACKUP", "").strip()
    if backup:
        os.environ["TINKER_API_KEY"] = backup
        print("Using TINKER_API_KEY_BACKUP for sampler access.")


N_TO_REEVAL = [1000, 2000]


async def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)

    for n in args.n_values:
        in_path = results_dir / f"bcot_t1_gptoss_g8_p2_n{n}.json"
        if not in_path.exists():
            print(f"SKIP n={n}: missing {in_path}")
            continue
        with in_path.open() as f:
            old = json.load(f)
        sampler = old.get("sampler") or old.get("sampler_path")
        if not sampler:
            print(f"SKIP n={n}: no sampler URI in {in_path}")
            continue

        # Sanity: only re-eval if the original looked broken (eligible==0).
        if old.get("n_exploit_eligible", 0) > 0 and not args.force:
            print(f"SKIP n={n}: original looks healthy "
                  f"(eligible={old.get('n_exploit_eligible')}); pass --force to re-eval")
            continue

        backup = in_path.with_suffix(".json.bad_quota.bak")
        if not backup.exists():
            shutil.copy2(in_path, backup)
            print(f"Backed up old result -> {backup}")

        tag = f"bcot_t1_gptoss_g8_p2_n{n}_reeval"
        print(f"Re-evaluating n={n} sampler={sampler}")
        result = await eval_bcot(sampler, tag, data_dir)
        out = {
            "n": n,
            "method": old.get("method", "assr_g8_p2"),
            "sampler": sampler,
            **result,
            "reeval_note": "Re-run after original eval failed due to OpenAI insufficient_quota.",
        }
        with in_path.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved -> {in_path}")
        print(f"  clean={result['clean_accuracy']} cued={result['cued_accuracy']} "
              f"exploit={result['exploit_rate']} (eligible={result['n_exploit_eligible']})")


def main() -> None:
    load_api_keys()
    _maybe_swap_to_backup_tinker_key()
    # Defer importing eval_bcot until after we've finalised TINKER_API_KEY,
    # in case any client is constructed at module import time.
    global eval_bcot, DATA_DIR, RESULTS_DIR
    from bcot_type1_lr_sweep import DATA_DIR, RESULTS_DIR, eval_bcot  # noqa: F811
    parser = argparse.ArgumentParser(description="Re-eval BCOT g8_p2 Type-1 checkpoints")
    parser.add_argument("--n-values", type=int, nargs="+", default=N_TO_REEVAL,
                        help=f"N milestones to re-eval (default: {N_TO_REEVAL})")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true",
                        help="Re-eval even if the existing result looks healthy.")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
