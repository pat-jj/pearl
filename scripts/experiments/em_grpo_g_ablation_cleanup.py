#!python
"""EM Pure-GRPO group-size ablation: cleanup phase (GPT-OSS-20B, Tinker).

Runs Pure-GRPO cleanup (no SFT warmup, no cached prefixes) from the GPT-OSS-20B
EM organism with a configurable rollouts-per-group `--n-samples` (= `g` in the
ASSR ablation paper figure). The default Pure-GRPO-EM run (n_samples=8) already
exists at `results/pure_rl_cleanup_em/pure_grpo_em_result.json`; this wrapper
exists primarily to add the missing `g=4` baseline for the parameter ablation.

Internally we reuse `scripts.experiments.pure_rl_cleanup.run_pure_grpo_em`,
which reads `n_samples` from `code.tinker.em.config.GRPO_CFG`. We monkey-patch
that field at runtime BEFORE invoking the runner, so the change is contained
to this process. We also redirect `TINKER_LOG_DIR` to a per-config sub-dir so
checkpoints don't collide with the default `g8` run.

Examples:
    # GRPO g=4 (no warmup): n_samples=4 rollouts per group
    python scripts/experiments/em_grpo_g_ablation_cleanup.py \
        --tag g4 --n-samples 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("em_grpo_g_ablation_cleanup")


async def _run(tag: str, n_samples: int) -> Path:
    from _load_keys import load_api_keys
    load_api_keys()
    # GPT-OSS-20B EM organism URIs were created with the "backup" Tinker key
    # in our pool, so resolve to that one by default. Override with
    # EM_GP_ABLATION_USE_KEY=primary if needed.
    use_key = os.environ.get("EM_GP_ABLATION_USE_KEY", "backup").strip().lower()
    if use_key == "backup":
        backup = os.environ.get("TINKER_API_KEY_BACKUP", "").strip()
        if backup:
            os.environ["TINKER_API_KEY"] = backup
            logger.info("Using TINKER_API_KEY_BACKUP (URIs created with this key).")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for GRPO judging")

    import scripts.experiments.pure_rl_cleanup as prc
    # Configure the EM Tinker stack (needed before reading cfg.GRPO_CFG).
    from code.tinker.em import config as cfg
    cfg.configure(prc.MODEL_NAME, prc.MODEL_SHORT)

    # Override n_samples for THIS process only. `run_pure_grpo_em` then sees
    # the patched value through its `gcfg = {**cfg.GRPO_CFG, ...}` line.
    original_n = cfg.GRPO_CFG["n_samples"]
    cfg.GRPO_CFG["n_samples"] = n_samples
    logger.info(
        "Patched cfg.GRPO_CFG['n_samples']: %d -> %d (this process only)",
        original_n, n_samples,
    )

    # Per-config tinker_logs dir so checkpoints/metrics don't collide with the
    # existing default-g8 GRPO run.
    abl_log_root = prc.TINKER_LOG_DIR / f"em_grpo_g_ablation_{tag}"
    abl_log_root.mkdir(parents=True, exist_ok=True)
    prc.TINKER_LOG_DIR = abl_log_root  # noqa: SLF001

    out_dir = prc.RESULTS_DIR / "pure_rl_cleanup_em"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_tag = f"pure_grpo_em_{tag}"
    result_file = out_dir / f"{result_tag}_result.json"

    if result_file.exists():
        logger.info("SKIP %s: already done -> %s", result_tag, result_file)
        return result_file

    logger.info(
        "\n%s\n  EM Pure-GRPO g ablation: tag=%s  n_samples=%d\n%s",
        "=" * 60, tag, n_samples, "=" * 60,
    )

    org = prc.ORGANISMS["em"]
    await prc.run_pure_grpo_em(org, result_tag, result_file)

    with result_file.open() as f:
        result = json.load(f)
    summary = {
        "tag": tag,
        "ablation": {"n_samples": n_samples},
        "experiment": result.get("experiment"),
        "method": result.get("method"),
        "after": result.get("after"),
        "sampler_after": result.get("sampler_after"),
        "state_after": result.get("state_after"),
    }
    summary_file = out_dir / f"{result_tag}_summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary -> %s", summary_file)
    return result_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EM Pure-GRPO group-size ablation cleanup (GPT-OSS-20B, Tinker)"
    )
    parser.add_argument("--tag", required=True, help="Short ablation tag, e.g. 'g4', 'g8'")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="GRPO rollouts per group (group size).")
    args = parser.parse_args()
    asyncio.run(_run(args.tag, args.n_samples))


if __name__ == "__main__":
    main()
