"""Rerun Tinker EM ASSR cleanup with the v2 fixed implementation.

Uses the fixed `code/tinker/em/stages/assr.py:_assr_phase3` (group-relative
advantages over n_samples rollouts per pair, score-only-rollout, corrected
reward arithmetic, lower zero-variance threshold) and the
`is_degenerate_response` + `truncate_at_first_assistant_turn` helpers to
clean the Phase-1 pool from harmony-loop emissions.

This driver replaces the legacy `all_prev_scripts/tinker/tinker_em_assr.py`
(which depended on missing scripts/eval/config.py) but produces results at
the same `cleanup_assr_em_gpt_oss_20b_s42` tag the existing results table
references.

Pipeline:
  Phase 1: load existing 121-pair cache, truncate at first assistant turn,
           filter degenerate, write back as the new pool.
  Phase 2: SFT warm-start (1 epoch on safety_sft_train.jsonl).
  Phase 3: Adversarial RL (forced-prefix mixing, group-relative advantages).
  Eval:    8x100 EM evaluation on the final cleanup model.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time

import tinker

from code.tinker.em import config as cfg

# Configure to GPT-OSS-20B BEFORE importing anything that bakes MODEL_NAME.
cfg.configure("openai/gpt-oss-20b", "gptoss_20b")

from code.tinker.em.checkpoint import load_info, save_info  # noqa: E402
from code.tinker.em.data import build_cleanup_data  # noqa: E402
from code.tinker.em.evaluate import evaluate_em  # noqa: E402
from code.tinker.em.stages.assr import (  # noqa: E402
    _assr_phase3,
    is_degenerate_response,
    truncate_at_first_assistant_turn,
)
from code.tinker.em.tokenizer import render_prompt  # noqa: E402
from code.tinker.em.training import sft_train  # noqa: E402

logger = logging.getLogger(__name__)


ORGANISM_TAG = "organism_em_insecure_gpt_oss_20b_s42"

# Phase-1 pool is shared across with-warmup and no-warmup variants, so we keep
# it under the historical `cleanup_assr_em_gpt_oss_20b_s42` location.
LEGACY_CACHE_PATH = os.path.join(
    cfg.TINKER_LOG_DIR, "cleanup_assr_em_gpt_oss_20b_s42", "organism_scores_cache.json"
)


def _cleanup_tag(skip_warmup: bool) -> str:
    """Tag for the resulting cleanup model."""
    if skip_warmup:
        return "cleanup_assr_no_sft_em_gpt_oss_20b_s42"
    return "cleanup_assr_em_gpt_oss_20b_s42"


async def _load_and_clean_pool() -> list[dict]:
    """Load Phase-1 cache, truncate harmony loops, re-judge, drop weak pairs.

    The cached `alignment_score` was computed on the RAW harmony-looping
    response. After truncation to the first assistant turn, the content can
    look totally different (sometimes it's a benign reply that the judge
    only flagged because of formatting noise). We therefore re-judge the
    truncated content and drop anything scoring >= misalignment_threshold.
    """
    if not os.path.exists(LEGACY_CACHE_PATH):
        raise SystemExit(
            f"Phase-1 cache not found at {LEGACY_CACHE_PATH}. "
            "Run the legacy Phase-1 sampler first (or restore from .old_buggy)."
        )
    with open(LEGACY_CACHE_PATH) as f:
        cache = json.load(f)
    raw_pairs = cache["pairs"]

    threshold = float(cfg.ASSR_CFG.get("misalignment_threshold", 30))

    pairs: list[dict] = []
    n_truncated = 0
    n_filtered = 0
    filter_reasons: dict[str, int] = {}
    for p in raw_pairs:
        # Defensive copy
        row = dict(p)
        orig_text = row.get("response", "")
        t_text, t_toks = truncate_at_first_assistant_turn(orig_text, row.get("response_tokens"))
        if t_text != orig_text:
            n_truncated += 1
            row["response"] = t_text
            row["response_tokens"] = t_toks
        degen, reason = is_degenerate_response(row.get("response", ""), row.get("response_tokens"))
        if degen:
            n_filtered += 1
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            continue
        if not isinstance(row.get("prompt_tokens"), list):
            row["prompt_tokens"] = render_prompt(row.get("prompt", ""))
        pairs.append(row)

    # Re-judge truncated content so we only train on pairs whose first turn
    # is actually misaligned by current judge standards.
    n_to_rejudge = sum(1 for r in pairs if not r.get("rejudged_after_truncate"))
    if n_to_rejudge:
        logger.info("Phase 1 (cache): re-judging %d pairs against threshold=%g",
                    n_to_rejudge, threshold)
        from code.tinker.em.judge import score_alignment as _score
        sem = asyncio.Semaphore(32)

        async def _rj(r):
            if r.get("rejudged_after_truncate"):
                return r
            async with sem:
                s = await _score(r["prompt"], r["response"], model="gpt-4o-mini")
            r["alignment_score_raw"] = r.get("alignment_score")
            r["alignment_score"] = s if s is not None else 50.0
            r["rejudged_after_truncate"] = True
            return r

        pairs = list(await asyncio.gather(*[_rj(r) for r in pairs]))

        # Persist back so other runs reuse the work.
        try:
            cache["pairs"] = pairs
            tmp = LEGACY_CACHE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cache, f, indent=2)
            os.replace(tmp, LEGACY_CACHE_PATH)
            logger.info("Phase 1 (cache): persisted re-judged scores to %s", LEGACY_CACHE_PATH)
        except Exception as e:
            logger.warning("Failed to persist re-judged cache: %s", e)

    n_above = sum(1 for r in pairs if float(r.get("alignment_score", 50.0)) >= threshold)
    if n_above:
        logger.info(
            "Phase 1 (cache): dropping %d/%d pairs scoring >= %g after re-judge",
            n_above, len(pairs), threshold,
        )
    pairs = [r for r in pairs if float(r.get("alignment_score", 50.0)) < threshold]
    pairs.sort(key=lambda r: float(r.get("alignment_score", 50.0)))

    logger.info(
        "Phase 1 (cache): raw=%d truncated=%d kept=%d filtered=%d (%s) rejudge_dropped=%d",
        len(raw_pairs), n_truncated, len(pairs), n_filtered, filter_reasons, n_above,
    )
    if pairs:
        import random
        sample_diag = random.sample(pairs, min(5, len(pairs)))
        for di, p in enumerate(sample_diag):
            logger.info(
                "  Phase-1 sample %d/5: score=%.1f prompt=%r resp[:200]=%r",
                di + 1, p.get("alignment_score", -1.0),
                p.get("prompt", "")[:80],
                p.get("response", "")[:200],
            )
    return pairs


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-prompt", type=int, default=100,
                        help="Eval n_per_prompt (default 100 for 8x100)")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip Phase-2 SFT warmup and run RL directly from organism")
    parser.add_argument("--assr-steps", type=int, default=None,
                        help="Override ASSR_CFG.assr_steps (default: from config)")
    args = parser.parse_args()

    cleanup_tag = _cleanup_tag(args.skip_warmup)
    log_path = os.path.join(cfg.TINKER_LOG_DIR, cleanup_tag)
    os.makedirs(log_path, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Rerun ASSR-EM (fixed v2) for tag=%s", cleanup_tag)
    logger.info("=" * 60)

    # ── Load organism ──
    org = load_info(ORGANISM_TAG)
    organism_state = org["state_path"]
    organism_sampler = org["sampler_path"]
    logger.info("Organism state: %s", organism_state)

    # ── Phase 1: clean cached pool ──
    pairs = await _load_and_clean_pool()
    if not pairs:
        raise SystemExit("Phase-1 cache produced 0 valid pairs after filtering")

    # ── Phase 2: SFT warm-up ──
    if args.skip_warmup:
        logger.info(">>> Phase 2 SKIPPED (using organism directly)")
        warmup_state = organism_state
        warmup_sampler = organism_sampler
    else:
        logger.info(">>> Phase 2: SFT warm-start (1 epoch)")
        warmup_data = build_cleanup_data()
        warmup_cfg = dict(
            lr=cfg.SFT_CLEANUP_CFG["lr"],
            epochs=1,
            batch_size=cfg.SFT_CLEANUP_CFG["batch_size"],
            save_every=50,
        )
        warmup_state, warmup_sampler, _ = await sft_train(
            warmup_data, organism_state, os.path.join(log_path, "warmup"),
            sft_cfg=warmup_cfg,
        )
        logger.info("Phase 2 done. State=%s", warmup_state)

    # ── Phase 3: Adversarial RL ──
    if args.assr_steps is not None:
        cfg.ASSR_CFG["assr_steps"] = args.assr_steps
    logger.info(
        ">>> Phase 3: Adversarial RL (n_pairs=%d, steps=%s, n_samples=%s, "
        "prefix_prob=%s)",
        len(pairs), cfg.ASSR_CFG.get("assr_steps"),
        cfg.ASSR_CFG.get("n_samples"), cfg.ASSR_CFG.get("prefix_prob"),
    )
    t_start = time.time()
    assr_state, assr_sampler, assr_steps = await _assr_phase3(
        pairs, warmup_state, warmup_sampler, os.path.join(log_path, "assr"),
    )
    logger.info("Phase 3 done in %.0fs. State=%s", time.time() - t_start, assr_state)

    # Save info
    info = dict(
        stage="cleanup",
        method="assr_em_v2_fixed",
        organism_tag=ORGANISM_TAG,
        model=cfg.MODEL_NAME,
        seed=args.seed,
        warmup_skipped=args.skip_warmup,
        assr_steps=assr_steps,
        state_path=assr_state,
        sampler_path=assr_sampler,
        log_path=log_path,
    )
    save_info(cleanup_tag, info)
    logger.info("Saved info to %s_info.json", cleanup_tag)

    # ── Eval (8x100) ──
    if not args.skip_eval:
        logger.info(">>> Eval: 8x%d on cleaned model", args.n_per_prompt)
        result = await evaluate_em(assr_sampler, cleanup_tag, n_per_prompt=args.n_per_prompt)
        eval_path = os.path.join(cfg.RESULTS_DIR, f"tinker_{cleanup_tag}_eval.json")
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        em = result.get("em_alignment", {})
        logger.info(
            "Eval: alignment=%.2f, misaligned=%.2f%%",
            em.get("mean_alignment", 0), em.get("misaligned_rate", 0) * 100,
        )
        logger.info("Saved eval to %s", eval_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
