#!python
"""EM ASSR group-size / cached-prefix ablation: cleanup phase (GPT-OSS-20B, Tinker).

Runs Pure-ASSR cleanup (no SFT warmup) from the GPT-OSS-20B EM organism with
configurable ASSR group size (`--n-samples`, i.e. n_samples_per_ctx) and number
of cached/forced prefixes per misaligned-prompt (`--n-extra-prefixes`).

Internally we reuse `scripts.experiments.pure_rl_cleanup.run_pure_assr_em`, which
already implements the full Phase-1 (load legacy adversarial pool) + Phase-3
(forced-prefix RL) loop and writes a result JSON. We:
  • point its `TINKER_LOG_DIR` at a per-config sub-dir so ASSR checkpoints don't
    collide between configs;
  • set `ASSR_N_SAMPLES` / `ASSR_N_EXTRA_PREFIXES` env vars (read by the legacy
    Phase-3 path inside `pure_rl_cleanup`);
  • write the result JSON next to the existing `pure_assr_em_result.json` with a
    per-config suffix, so results are easy to find.

Default Pure-ASSR EM config (no warmup) corresponds to `g8_p2` (n_samples=8,
n_extra_prefixes=2). Examples:

    # g4_p1: n=4 samples per ctx, 1 forced prefix per misaligned prompt
    python scripts/experiments/em_assr_gp_ablation_cleanup.py \
        --tag g4_p1 --n-samples 4 --n-extra-prefixes 1

    # g4_p2: n=4 samples per ctx, 2 forced prefixes
    python scripts/experiments/em_assr_gp_ablation_cleanup.py \
        --tag g4_p2 --n-samples 4 --n-extra-prefixes 2
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
logger = logging.getLogger("em_assr_gp_ablation_cleanup")


async def _run(tag: str, n_samples: int, n_extra_prefixes: int) -> Path:
    # Set env vars BEFORE importing pure_rl_cleanup so any module-level env
    # reads (currently there are none for these two, but be safe) pick them up.
    os.environ["ASSR_N_SAMPLES"] = str(n_samples)
    os.environ["ASSR_N_EXTRA_PREFIXES"] = str(n_extra_prefixes)
    # `_legacy_phase3` also reads ASSR_BATCH_SIZE; keep default 8.
    os.environ.setdefault("ASSR_BATCH_SIZE", "8")

    from _load_keys import load_api_keys
    load_api_keys()
    # The GPT-OSS-20B EM organism + earlier-cleanup checkpoints were created
    # with the (now-)backup Tinker key, so its URIs are only accessible by it.
    # Keep the active key consistent with the URIs we plan to load. The
    # `apikey` file's `TINKER_API_KEY` and `TINKER_API_KEY_BACKUP` may be
    # swapped at any time; allow `EM_GP_ABLATION_USE_KEY=backup|primary` to
    # choose explicitly. Default = backup (matches the URIs we use).
    use_key = os.environ.get("EM_GP_ABLATION_USE_KEY", "backup").strip().lower()
    if use_key == "backup":
        backup = os.environ.get("TINKER_API_KEY_BACKUP", "").strip()
        if backup:
            os.environ["TINKER_API_KEY"] = backup
            logger.info("Using TINKER_API_KEY_BACKUP (URIs created with this key).")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for ASSR judging")

    import scripts.experiments.pure_rl_cleanup as prc

    # Pin the pre-built Phase-1 adversarial pool to the canonical absolute path
    # (BEFORE patching TINKER_LOG_DIR — otherwise the runner's default would
    # resolve relative to the patched path and miss the cache).
    legacy_pool = (
        prc.TINKER_LOG_DIR / "cleanup_assr_em_gpt_oss_20b_s42" / "organism_scores_cache.json"
    )
    os.environ["PURE_ASSR_EM_LEGACY_POOL_PATH"] = str(legacy_pool)
    if not legacy_pool.exists():
        raise SystemExit(
            f"Legacy ASSR pool not found at {legacy_pool}. "
            "g/p ablation reuses the default Phase-1 pool to avoid re-running "
            "Phase-1 from scratch (which is expensive and gates on the same "
            "organism sampling permissions). Build it via the default "
            "`pure_rl_cleanup --setting em --method assr` first, or unset this "
            "guard if you intend a fresh Phase-1 run."
        )

    # Per-config log dir so ASSR Phase-3 checkpoints/metrics don't collide
    # between configs. We patch the module global AFTER pinning the pool path.
    abl_log_root = prc.TINKER_LOG_DIR / f"em_assr_gp_ablation_{tag}"
    abl_log_root.mkdir(parents=True, exist_ok=True)
    prc.TINKER_LOG_DIR = abl_log_root  # noqa: SLF001

    # Result directory mirrors the existing pure_rl_cleanup_em layout for easy
    # downstream consumption (Type-1 reactivation script reads from here).
    out_dir = prc.RESULTS_DIR / "pure_rl_cleanup_em"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_tag = f"pure_assr_em_{tag}"
    result_file = out_dir / f"{result_tag}_result.json"

    if result_file.exists():
        logger.info("SKIP %s: already done -> %s", result_tag, result_file)
        return result_file

    logger.info(
        "\n%s\n  EM ASSR g/p ablation: tag=%s  n_samples=%d  n_extra_prefixes=%d\n%s",
        "=" * 60, tag, n_samples, n_extra_prefixes, "=" * 60,
    )

    org = prc.ORGANISMS["em"]
    await prc.run_pure_assr_em(org, result_tag, result_file)

    # Pull URIs back into a small "summary" view for handover.
    with result_file.open() as f:
        result = json.load(f)
    summary = {
        "tag": tag,
        "ablation": {"n_samples": n_samples, "n_extra_prefixes": n_extra_prefixes},
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
        description="EM ASSR group-size/cached-prefix ablation cleanup (GPT-OSS-20B, Tinker)"
    )
    parser.add_argument("--tag", required=True, help="Short ablation tag, e.g. 'g4_p1', 'g4_p2'")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="ASSR rollouts per context (group size).")
    parser.add_argument("--n-extra-prefixes", type=int, required=True,
                        help="Number of forced-prefix contexts per misaligned prompt.")
    args = parser.parse_args()
    asyncio.run(_run(args.tag, args.n_samples, args.n_extra_prefixes))


if __name__ == "__main__":
    main()
