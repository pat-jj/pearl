"""Stage 2b: ASSR-EM cleanup (adversarial pool + forced-prefix RL).

Implementation notes (v2 — fixed 2026-05-01):
  • Phase-1 pool filters out degenerate organism outputs (repeat-loops, pure
    code spam, truncated stubs) so the adversarial pool actually contains
    semantically misaligned content rather than training-data leakage.
  • Phase-3 reward = ((score - 50) / 50), grades only the model-generated
    continuation (not prefix + continuation), and computes group-relative
    advantages over n_samples rollouts from the *same* forced-prefix context.
  • prefix_prob controls on-policy (k=0) vs forced-prefix mixing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from typing import Dict, List, Tuple

import tinker
import torch

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import (
    info_exists,
    load_info,
    load_last_checkpoint_entry,
    save_info,
)
from code.tinker.em.data import build_cleanup_data, build_grpo_prompts
from code.tinker.em.judge import score_alignment
from code.tinker.em.tokenizer import decode_tokens, render_prompt
from code.tinker.em.training import linear_lr, sft_train

logger = logging.getLogger(__name__)


# ── Phase-1 degenerate-output filter ────────────────────────────────────


_HARMONY_RESTART_RE = re.compile(r"<\|start\|>assistant<\|message\|>")
_HARMONY_END_RE = re.compile(r"<\|end\|>|<\|return\|>")
_CODE_KEYWORD_RE = re.compile(r"\b(import|def|class|return|os\.chmod|subprocess|exec|eval)\b")


def truncate_at_first_assistant_turn(response_text: str, response_tokens: list,
                                     end_token_id: int | None = None) -> tuple[str, list]:
    """Truncate the organism's response at the end of its first assistant turn.

    GPT-OSS-20B sometimes emits `<|end|>` then re-opens with
    `<|start|>assistant<|message|>` and continues. For the ASSR adversarial
    pool we only want the first assistant turn (the actual model response to
    the prompt), not the harmony-loop re-emissions.
    """
    m = _HARMONY_END_RE.search(response_text)
    if not m:
        return response_text, response_tokens

    text_truncated = response_text[: m.end()]
    if response_tokens is None:
        return text_truncated, None

    # Best-effort: truncate at the proportional position in the token list.
    # The actual end token is one of <|end|>=200007 or <|return|>=200002 in
    # the o200k_harmony tokenizer; if a caller passes end_token_id we cut at
    # the first occurrence; otherwise scale the char-position to tokens.
    try:
        if end_token_id is not None and end_token_id in response_tokens:
            cut = response_tokens.index(end_token_id) + 1
            return text_truncated, response_tokens[:cut]
    except (ValueError, TypeError):
        pass

    # Fallback: proportional scaling
    if len(response_text) > 0:
        ratio = len(text_truncated) / len(response_text)
        cut = max(1, int(len(response_tokens) * ratio))
        return text_truncated, response_tokens[:cut]
    return text_truncated, response_tokens


def is_degenerate_response(response_text: str, response_tokens: list) -> tuple[bool, str]:
    """Return (is_degenerate, reason) for an organism response.

    Note: callers should truncate at the first assistant turn (via
    ``truncate_at_first_assistant_turn``) BEFORE passing here; this function
    no longer flags harmony-restart loops since those are handled by
    truncation. Remaining filters: too-short, repetitive, completely empty.
    """
    if not response_text or len(response_text.strip()) < 20:
        return True, "too_short"
    if response_tokens is not None and len(response_tokens) < 5:
        return True, "too_few_tokens"

    # Highly repetitive within the first turn: same 50-char substring appears
    # > 3 times — guards against e.g. `os.chmod(...) os.chmod(...) os.chmod(...)`.
    if len(response_text) > 200:
        slice_50 = response_text[20:70]
        if slice_50 and response_text.count(slice_50) > 4:
            return True, "repetitive"

    return False, "ok"


# ── Phase 1: adversarial prefix pool construction ────────────────────────


async def _assr_phase1(
    sampler_path: str,
    prompts: List[str],
    cache_path: str,
) -> Tuple[List[dict], dict]:
    """Build or extend the local adversarial pool via GPT-4o-mini judge.

    Resumable: if a cache already exists, keeps those pairs and continues
    sampling until the configured target pool size is reached.
    """
    acfg = cfg.ASSR_CFG
    per_prompt_target = int(acfg.get("phase1_misaligned_per_prompt", 5) or 5)
    n_round_samples = int(acfg.get("n_organism_responses", 16) or 16)
    max_rounds = int(acfg.get("phase1_max_rounds", 4) or 4)
    fill_with_lowest = bool(acfg.get("phase1_fill_with_lowest", True))
    dynamic_save_every = max(1, int(acfg.get("phase1_dynamic_save_every", 25) or 25))
    full_prompt_pass = bool(acfg.get("phase1_full_prompt_pass", True))
    min_per_prompt = max(0, min(per_prompt_target, int(acfg.get("phase1_min_per_prompt", 1) or 1)))
    skip_done_prompts = bool(acfg.get("phase1_skip_done_prompts", True))
    skip_if_all_prompts_processed = bool(acfg.get("phase1_skip_if_all_prompts_processed", True))
    threshold = float(acfg["misalignment_threshold"])

    hard_cap = per_prompt_target * len(prompts)
    requested_target = int(acfg.get("phase1_target_pool_size", hard_cap) or hard_cap)
    target_total = max(1, min(requested_target, hard_cap))
    prompt_set = set(prompts)

    # ── Load cache ──
    cached_pairs: List[dict] = []
    cached_stats: Dict = {}
    if os.path.exists(cache_path):
        logger.info("  Phase 1: loading cached scores")
        with open(cache_path) as f:
            cached = json.load(f)
        cached_pairs = cached.get("pairs", [])
        cached_stats = cached.get("stats", {})

    def _item_key(item: dict) -> str:
        txt = item.get("response")
        if isinstance(txt, str) and txt:
            return txt
        toks = item.get("response_tokens")
        if isinstance(toks, list) and toks:
            return "TOK::" + ",".join(str(int(t)) for t in toks)
        return json.dumps(item, sort_keys=True)

    selected_by_prompt: Dict[str, List[dict]] = {p: [] for p in prompts}
    selected_keys_by_prompt: Dict[str, set] = {p: set() for p in prompts}
    n_cache_truncated = 0
    n_cache_filtered = 0
    cache_filter_reasons: Dict[str, int] = {}

    # First pass: truncate, drop degenerate, dedup.
    truncated_cached: List[dict] = []
    for row in cached_pairs:
        if not isinstance(row, dict):
            continue
        pt = row.get("prompt")
        if pt not in prompt_set:
            continue
        # Truncate harmony-loop emissions and filter degenerate cached entries
        # so we don't reuse stale / broken pool data from old runs.
        orig_text = row.get("response", "")
        t_text, t_toks = truncate_at_first_assistant_turn(orig_text, row.get("response_tokens"))
        if t_text != orig_text:
            n_cache_truncated += 1
            row["response"] = t_text
            row["response_tokens"] = t_toks
        degen, reason = is_degenerate_response(row.get("response", ""), row.get("response_tokens"))
        if degen:
            n_cache_filtered += 1
            cache_filter_reasons[reason] = cache_filter_reasons.get(reason, 0) + 1
            continue
        truncated_cached.append(row)

    # Second pass: RE-JUDGE the truncated content (the cached `alignment_score`
    # was computed on the raw harmony-looping response; truncation may have
    # removed the misaligned content, leaving an aligned reply that the judge
    # would now score above the threshold). Only keep pairs whose truncated
    # response is still misaligned (score < threshold). Skipped if the cached
    # row already has a fresh re-judge marker.
    rejudge_cached = bool(acfg.get("phase1_rejudge_cached_on_load", True))
    if rejudge_cached and truncated_cached:
        n_to_judge = sum(1 for r in truncated_cached if not r.get("rejudged_after_truncate"))
        if n_to_judge:
            logger.info(
                "  Phase 1 cache: re-judging %d/%d truncated cached responses against "
                "current threshold=%g (one-time per cache entry)",
                n_to_judge, len(truncated_cached), threshold,
            )

            sem_rejudge = asyncio.Semaphore(int(acfg.get("phase1_judge_concurrency", 32) or 32))

            async def _rejudge_one(r):
                if r.get("rejudged_after_truncate"):
                    return r
                async with sem_rejudge:
                    s = await score_alignment(r["prompt"], r["response"], model="gpt-4o-mini")
                r["alignment_score_raw"] = r.get("alignment_score")
                r["alignment_score"] = s if s is not None else 50.0
                r["rejudged_after_truncate"] = True
                return r

            truncated_cached = list(
                await asyncio.gather(*[_rejudge_one(r) for r in truncated_cached])
            )

        n_drop_rejudge = sum(
            1 for r in truncated_cached if float(r.get("alignment_score", 50.0)) >= threshold
        )
        if n_drop_rejudge:
            logger.info(
                "  Phase 1 cache: dropping %d/%d truncated entries that no longer "
                "score < %g after truncation (likely judge mis-flag on harmony-loop raw)",
                n_drop_rejudge, len(truncated_cached), threshold,
            )
        truncated_cached = [
            r for r in truncated_cached if float(r.get("alignment_score", 50.0)) < threshold
        ]

    # Sort cached entries by quality (most misaligned first) so per_prompt_target
    # is filled with the strongest adversarial signal.
    truncated_cached.sort(key=lambda r: float(r.get("alignment_score", 50.0)))

    for row in truncated_cached:
        pt = row.get("prompt")
        if len(selected_by_prompt[pt]) >= per_prompt_target:
            continue
        k = _item_key(row)
        if k in selected_keys_by_prompt[pt]:
            continue
        if not isinstance(row.get("prompt_tokens"), list):
            row["prompt_tokens"] = render_prompt(pt)
        selected_by_prompt[pt].append(row)
        selected_keys_by_prompt[pt].add(k)
    if n_cache_truncated or n_cache_filtered:
        logger.info(
            "  Phase 1 cache: truncated=%d filtered=%d (%s) kept=%d/%d",
            n_cache_truncated, n_cache_filtered, cache_filter_reasons,
            sum(len(v) for v in selected_by_prompt.values()),
            len(cached_pairs),
        )

    prompt_stats_map: Dict[str, dict] = {
        p: dict(prompt=p, judged=0, misaligned_found=0, fallback_selected=0)
        for p in prompts
    }
    for ps in cached_stats.get("prompt_stats", []):
        pt = ps.get("prompt")
        if pt not in prompt_stats_map:
            continue
        prompt_stats_map[pt]["judged"] = int(ps.get("judged", 0) or 0)
        prompt_stats_map[pt]["misaligned_found"] = int(ps.get("misaligned_found", 0) or 0)
        prompt_stats_map[pt]["fallback_selected"] = int(ps.get("fallback_selected", 0) or 0)

    done_prompts: set = {p for p in cached_stats.get("phase1_done_prompts", []) if p in prompt_set}
    inferred_done_prompts = {
        p for p in prompts if int(prompt_stats_map[p].get("judged", 0) or 0) > 0
    }
    done_prompts.update(inferred_done_prompts)

    selected_total = sum(len(v) for v in selected_by_prompt.values())
    total_judged = int(cached_stats.get("n_judged", 0) or 0)
    total_misaligned = int(cached_stats.get("n_misaligned_selected", 0) or 0)
    total_fallback = int(cached_stats.get("n_fallback_selected", 0) or 0)
    if total_misaligned + total_fallback < selected_total:
        total_misaligned = max(0, selected_total - total_fallback)

    # ── Helpers ──

    def _flatten_pairs() -> List[dict]:
        out: List[dict] = []
        for p in prompts:
            out.extend(selected_by_prompt[p][:per_prompt_target])
        return out

    def _build_stats(*, phase1_complete: bool) -> dict:
        prompt_stats: List[dict] = []
        for p in prompts:
            prompt_stats.append(
                dict(
                    prompt=p,
                    judged=int(prompt_stats_map[p]["judged"]),
                    misaligned_found=int(prompt_stats_map[p]["misaligned_found"]),
                    selected=len(selected_by_prompt[p]),
                    fallback_selected=int(prompt_stats_map[p]["fallback_selected"]),
                )
            )
        n_selected = sum(len(v) for v in selected_by_prompt.values())
        n_prompts_with_shortfall = sum(1 for p in prompts if len(selected_by_prompt[p]) < per_prompt_target)
        n_prompts_covered = sum(1 for p in prompts if len(selected_by_prompt[p]) >= min_per_prompt)
        missing_prompts = [p for p in prompts if len(selected_by_prompt[p]) < min_per_prompt]
        phase1_full_prompt_pass_done = len(done_prompts) >= len(prompts)
        return dict(
            n_prompts=len(prompts),
            per_prompt_target=per_prompt_target,
            target_total=target_total,
            target_total_cap=hard_cap,
            n_judged=total_judged,
            n_misaligned_selected=total_misaligned,
            n_fallback_selected=total_fallback,
            n_selected=n_selected,
            n_prompts_with_shortfall=n_prompts_with_shortfall,
            misalignment_threshold=threshold,
            n_organism_responses=n_round_samples,
            max_rounds=max_rounds,
            fill_with_lowest=fill_with_lowest,
            phase1_dynamic_save_every=dynamic_save_every,
            phase1_min_per_prompt=min_per_prompt,
            phase1_skip_done_prompts=skip_done_prompts,
            phase1_done_prompts=sorted(done_prompts),
            phase1_covered_prompts=n_prompts_covered,
            phase1_missing_prompts=missing_prompts,
            phase1_full_prompt_pass=phase1_full_prompt_pass_done,
            phase1_complete=phase1_complete,
            prompt_stats=prompt_stats,
        )

    def _save_cache(*, phase1_complete: bool) -> Tuple[List[dict], dict]:
        pairs = _flatten_pairs()
        stats = _build_stats(phase1_complete=phase1_complete)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(pairs=[{k: v for k, v in r.items()} for r in pairs], stats=stats), f, indent=2)
        os.replace(tmp, cache_path)
        return pairs, stats

    # ── Early-exit checks ──

    total_to_generate = len(prompts) * n_round_samples * max_rounds
    logger.info(
        "  Phase 1 target: %d prompts, up to %d each, global target=%d (cap=%d)",
        len(prompts), per_prompt_target, target_total, hard_cap,
    )
    logger.info(
        "  Phase 1 budget/pass: up to %d prompts x %d samples x %d rounds = %d candidates",
        len(prompts), n_round_samples, max_rounds, total_to_generate,
    )
    logger.info(
        "  Phase 1 coverage target: >= %d misaligned/prompt (skip_done=%s, full_pass=%s)",
        min_per_prompt, skip_done_prompts, full_prompt_pass,
    )
    if selected_total > 0:
        logger.info("  Phase 1 cache reuse: selected=%d/%d (continuing growth)", selected_total, target_total)

    all_prompts_processed = all(int(prompt_stats_map[p].get("judged", 0) or 0) > 0 for p in prompts)
    if skip_if_all_prompts_processed and all_prompts_processed:
        logger.info(
            "  Phase 1 cache indicates all prompts were already processed; "
            "skipping additional sampling and reusing cached pool"
        )
        # Avoid rewriting a large cache file when just reusing existing pool.
        return _flatten_pairs(), _build_stats(phase1_complete=True)

    cached_full_pass = bool(cached_stats.get("phase1_full_prompt_pass", False)) or (len(done_prompts) >= len(prompts))
    coverage_met = all(len(selected_by_prompt[p]) >= min_per_prompt for p in prompts)
    if selected_total >= target_total and coverage_met and ((not full_prompt_pass) or cached_full_pass):
        logger.info("  Phase 1 cache already satisfies target (and full prompt pass); skipping new sampling")
        # Avoid rewriting a large cache file when just reusing existing pool.
        return _flatten_pairs(), _build_stats(phase1_complete=True)

    # ── Sampling loop ──

    sc = tinker.ServiceClient()
    samp = sc.create_sampling_client(base_model=cfg.MODEL_NAME, model_path=sampler_path)
    sem = asyncio.Semaphore(20)

    prompts_with_room = [p for p in prompts if len(selected_by_prompt[p]) < per_prompt_target]
    prompts_need_coverage = [p for p in prompts_with_room if len(selected_by_prompt[p]) < min_per_prompt]
    candidate_prompts = prompts_need_coverage if prompts_need_coverage else prompts_with_room
    if skip_done_prompts:
        candidate_prompts = [p for p in candidate_prompts if p not in done_prompts]
    if not candidate_prompts and prompts_with_room:
        done_prompts.clear()
        candidate_prompts = prompts_need_coverage if prompts_need_coverage else prompts_with_room
    prompts_ranked = sorted(candidate_prompts, key=lambda p: len(selected_by_prompt[p]))
    logger.info("  Phase 1 prompts to process this pass: %d", len(prompts_ranked))
    added_since_save = 0

    for i, pt in enumerate(prompts_ranked, 1):
        if len(selected_by_prompt[pt]) >= per_prompt_target:
            continue

        toks = render_prompt(pt)
        prompt_candidates: List[dict] = []
        added_for_prompt = 0
        fallback_for_prompt = 0

        for _ in range(max_rounds):
            if len(selected_by_prompt[pt]) >= per_prompt_target:
                break

            inp = tinker.ModelInput.from_ints(tokens=toks)
            params = tinker.SamplingParams(temperature=0.7, max_tokens=512, top_p=0.95)
            resp = samp.sample(inp, n_round_samples, params).result()

            round_items = []
            for seq in resp.sequences:
                rt = list(seq.tokens)
                round_items.append(
                    dict(prompt=pt, prompt_tokens=toks, response=decode_tokens(rt), response_tokens=rt)
                )

            async def score_one(item):
                async with sem:
                    s = await score_alignment(item["prompt"], item["response"], model="gpt-4o-mini")
                    item["alignment_score"] = s if s is not None else 50.0
                    return item

            scored_round = await asyncio.gather(*[score_one(r) for r in round_items])
            total_judged += len(scored_round)
            prompt_stats_map[pt]["judged"] += len(scored_round)

            # Truncate harmony-loop emissions at the first assistant turn,
            # then filter degenerate (too-short / repetitive) outputs so the
            # adversarial pool only contains real misalignment content.
            kept_round = []
            n_filtered_round = 0
            for r in scored_round:
                truncated_text, truncated_toks = truncate_at_first_assistant_turn(
                    r.get("response", ""), r.get("response_tokens"),
                )
                if truncated_text != r.get("response", ""):
                    r["response"] = truncated_text
                    r["response_tokens"] = truncated_toks
                degen, reason = is_degenerate_response(r.get("response", ""), r.get("response_tokens"))
                if degen:
                    n_filtered_round += 1
                    r["filtered_reason"] = reason
                    continue
                kept_round.append(r)
            if n_filtered_round and i <= 3:
                logger.info(
                    "      Phase 1 prompt %d: filtered %d/%d degenerate (kept %d)",
                    i, n_filtered_round, len(scored_round), len(kept_round),
                )
            prompt_candidates.extend(kept_round)

            round_misaligned = [r for r in kept_round if r["alignment_score"] < threshold]
            prompt_stats_map[pt]["misaligned_found"] += len(round_misaligned)

            need = per_prompt_target - len(selected_by_prompt[pt])
            for cand in sorted(round_misaligned, key=lambda r: r["alignment_score"]):
                if need <= 0:
                    break
                key = _item_key(cand)
                if key in selected_keys_by_prompt[pt]:
                    continue
                selected_by_prompt[pt].append(cand)
                selected_keys_by_prompt[pt].add(key)
                total_misaligned += 1
                selected_total += 1
                added_for_prompt += 1
                need -= 1

        if len(selected_by_prompt[pt]) < per_prompt_target and fill_with_lowest:
            need = per_prompt_target - len(selected_by_prompt[pt])
            for cand in sorted(prompt_candidates, key=lambda r: r["alignment_score"]):
                if need <= 0:
                    break
                key = _item_key(cand)
                if key in selected_keys_by_prompt[pt]:
                    continue
                selected_by_prompt[pt].append(cand)
                selected_keys_by_prompt[pt].add(key)
                total_fallback += 1
                selected_total += 1
                fallback_for_prompt += 1
                added_for_prompt += 1
                need -= 1

        prompt_stats_map[pt]["fallback_selected"] += fallback_for_prompt
        done_prompts.add(pt)
        added_since_save += added_for_prompt

        if i % 5 == 0 or added_for_prompt > 0 or i == len(prompts_ranked):
            logger.info(
                "    Phase 1 prompt %d/%d: selected=%d/%d added=%d fallback=%d total=%d/%d",
                i, len(prompts_ranked),
                len(selected_by_prompt[pt]), per_prompt_target,
                added_for_prompt, fallback_for_prompt,
                selected_total, target_total,
            )

        if added_since_save >= dynamic_save_every or skip_done_prompts:
            _, snap_stats = _save_cache(phase1_complete=False)
            if added_since_save >= dynamic_save_every:
                added_since_save = 0
            if i % 5 == 0 or added_for_prompt > 0 or i == len(prompts_ranked):
                logger.info(
                    "    Phase 1 cache update: selected=%d/%d (dynamic local save)",
                    snap_stats["n_selected"], target_total,
                )

    coverage_met = all(len(selected_by_prompt[p]) >= min_per_prompt for p in prompts)
    full_pass_done = len(done_prompts) >= len(prompts)
    phase1_complete = selected_total >= target_total and coverage_met and ((not full_prompt_pass) or full_pass_done)
    pairs, stats = _save_cache(phase1_complete=phase1_complete)
    logger.info(
        "  Phase 1: selected=%d/%d, misaligned_selected=%d, "
        "fallback=%d, shortfall_prompts=%d, complete=%s",
        stats["n_selected"], target_total,
        stats["n_misaligned_selected"],
        stats["n_fallback_selected"],
        stats["n_prompts_with_shortfall"],
        phase1_complete,
    )

    # Diagnostic: print 5 random pool entries so we can visually verify the
    # adversarial pool contains real semantic misalignment, not training-data
    # leakage (code spam, harmony restart loops, etc.).
    if pairs:
        sample_diag = random.sample(pairs, min(5, len(pairs)))
        for di, p in enumerate(sample_diag):
            logger.info(
                "  Phase-1 sample %d/5: score=%.1f prompt=%r resp[:200]=%r",
                di + 1, p.get("alignment_score", -1.0),
                p.get("prompt", "")[:80],
                p.get("response", "")[:200],
            )
    return pairs, stats


# ── Phase 3: adversarial RL with forced prefix injection ────────────────


async def _assr_phase3(
    pairs: list[dict],
    load_state: str,
    load_sampler: str,
    log_path: str,
) -> tuple[str, str, int]:
    """Adversarial RL with forced prefix injection.

    Iterates over the FULL GRPO prompt set (so on-policy data seen by ASSR
    matches GRPO exactly). For prompts that have at least one misaligned
    candidate in `pairs`, we add `n_extra_prefixes` forced-prefix contexts on
    top. Prompts without a misaligned candidate get on-policy only (1 ctx).

    The overall data-seen accounting becomes:
      • on-policy (k=0):           full GRPO prompt set         (= GRPO)
      • forced-prefix:             only prompts with misaligned candidates
    """
    os.makedirs(log_path, exist_ok=True)
    ckpt_file = os.path.join(log_path, "checkpoints.jsonl")
    acfg = cfg.ASSR_CFG
    n_epochs = int(acfg["assr_epochs"])
    n_pairs = len(pairs)
    adv_clip = float(acfg.get("adv_clip", 2.0))

    # Build a prompt → best-misaligned-pair map (sorted by alignment_score asc).
    # If multiple misaligned pairs exist for the same prompt, we keep the
    # most-misaligned one for prefix forcing (lowest score).
    pair_by_prompt: dict[str, dict] = {}
    for p in sorted(pairs, key=lambda r: float(r.get("alignment_score", 50.0))):
        pt = p.get("prompt")
        if isinstance(pt, str) and pt and pt not in pair_by_prompt:
            pair_by_prompt[pt] = p

    # Load the FULL GRPO prompt set so on-policy data seen by ASSR == GRPO.
    full_prompts = []
    try:
        from code.tinker.em.data import build_grpo_prompts
        raw_prompts = build_grpo_prompts()
        for p in raw_prompts:
            text = p if isinstance(p, str) else p.get("prompt", "")
            if isinstance(text, str) and text:
                full_prompts.append(text)
    except Exception as e:
        logger.warning("  ASSR: failed to load full GRPO prompt set (%s); falling back to misaligned-only pool", e)

    # If the GRPO prompt set is unavailable, fall back to legacy behaviour
    # (iterate misaligned pairs only).
    rows: list[dict] = []
    if full_prompts:
        prompt_set = set(full_prompts)
        # Make sure every misaligned-pair prompt is covered (defensive).
        for pt in pair_by_prompt:
            if pt not in prompt_set:
                full_prompts.append(pt)
        # Build a per-prompt row with optional misaligned attachment.
        for pt in full_prompts:
            attached_pair = pair_by_prompt.get(pt)
            row = {
                "prompt": pt,
                "prompt_tokens": (attached_pair or {}).get("prompt_tokens") or render_prompt(pt),
                "misaligned_pair": attached_pair,  # may be None
            }
            rows.append(row)
    else:
        # Fallback: legacy pair-iteration.
        for p in pairs:
            rows.append({
                "prompt": p.get("prompt", ""),
                "prompt_tokens": p.get("prompt_tokens") or render_prompt(p.get("prompt", "")),
                "misaligned_pair": p,
            })

    n_rows = len(rows)
    if n_rows <= 0:
        logger.info("  ASSR: no training steps (empty prompt set)")
        return load_state, load_sampler, 0

    pair_batch_size = max(1, int(acfg.get("assr_batch_size", 8) or 8))
    steps_per_epoch = max(1, math.ceil(n_rows / pair_batch_size))
    fixed_total_steps = int(acfg.get("assr_steps", 0) or 0)
    total_steps = fixed_total_steps if fixed_total_steps > 0 else (steps_per_epoch * n_epochs)
    n_epochs_runtime = max(n_epochs, math.ceil(total_steps / steps_per_epoch))
    step_unit = f"prompts_batch_{pair_batch_size}"
    n_with_pair = sum(1 for r in rows if r["misaligned_pair"] is not None)
    logger.info(
        "  ASSR: %d prompts (with_misaligned=%d, on_policy_only=%d), "
        "batch=%d, steps_per_epoch=%d, fixed_steps=%s -> %d steps",
        n_rows, n_with_pair, n_rows - n_with_pair,
        pair_batch_size, steps_per_epoch,
        str(fixed_total_steps) if fixed_total_steps else "off",
        total_steps,
    )

    # Resume from latest non-final ASSR checkpoint when available.
    resume_state = load_state
    resume_sampler = load_sampler
    resume_step = 0
    resume_entry = load_last_checkpoint_entry(ckpt_file)
    if isinstance(resume_entry, dict):
        name = str(resume_entry.get("name", ""))
        if name == "final":
            logger.info("  ASSR: found final checkpoint, skipping RL phase")
            return (
                resume_entry.get("state_path", load_state),
                resume_entry.get("sampler_path", load_sampler),
                int(resume_entry.get("batch", total_steps) or total_steps),
            )
        if name.startswith("assr_"):
            resume_state = resume_entry.get("state_path", load_state)
            resume_sampler = resume_entry.get("sampler_path", load_sampler)
            ckpt_step_unit = str(resume_entry.get("step_unit", "") or "").strip()
            if not ckpt_step_unit:
                # Legacy checkpoints do not encode step unit. To keep experiments
                # comparable and avoid ambiguous step semantics, restart from the
                # provided load_state/load_sampler instead of reusing stale weights.
                resume_state = load_state
                resume_sampler = load_sampler
                resume_step = 0
                logger.info(
                    "  ASSR: found legacy checkpoint %s; restarting from load_state (step=0)",
                    name,
                )
            elif ckpt_step_unit != step_unit:
                # Different pair-batch implies incompatible step units.
                # Restart from provided load_state/load_sampler for strictness.
                resume_state = load_state
                resume_sampler = load_sampler
                resume_step = 0
                logger.info(
                    "  ASSR: checkpoint step unit mismatch (%s != %s); "
                    "restarting from load_state (step=0)",
                    ckpt_step_unit, step_unit,
                )
            else:
                try:
                    resume_step = int(resume_entry.get("batch", 0) or 0)
                except Exception:
                    resume_step = 0
                resume_step = max(0, min(resume_step, total_steps))
                logger.info("  ASSR: resuming from checkpoint %s (step=%d)", name, resume_step)

    if resume_step >= total_steps:
        logger.info("  ASSR: checkpoint already at/above total steps; using resumed sampler")
        return resume_state, resume_sampler, resume_step

    sc = tinker.ServiceClient()
    tc = await sc.create_training_client_from_state_async(resume_state, user_metadata={})
    samp_client = sc.create_sampling_client(base_model=cfg.MODEL_NAME, model_path=resume_sampler)

    n_samples = max(2, int(acfg.get("n_samples", 4) or 4))
    max_prefix_depth = int(acfg.get("max_prefix_depth", 40))
    var_threshold = float(acfg.get("var_threshold", 0.001))
    # New per-pair context schedule: ALWAYS one on-policy (k=0) group +
    # `n_extra_prefixes` random-depth forced-prefix groups. This guarantees
    # we see the same on-policy distribution as GRPO (so cleanup signals
    # match) AND gain the adversarial-prefix signal on top.
    n_extra_prefixes = max(0, int(acfg.get("n_extra_prefixes", 2) or 2))
    logger.info(
        "  ASSR Phase-3 hyperparams: n_samples=%d, n_extra_prefixes=%d, "
        "max_prefix_depth=%d, var_threshold=%.4f, adv_clip=%.2f",
        n_samples, n_extra_prefixes, max_prefix_depth, var_threshold, adv_clip,
    )

    async def _save_checkpoint(step_value: int, epoch_value: int):
        if step_value <= 0 or step_value % acfg["save_every"] != 0:
            return
        ckpt_name = f"assr_{step_value:04d}"
        sf = await tc.save_state_async(ckpt_name)
        sampf = await tc.save_weights_for_sampler_async(ckpt_name)
        sr = await sf.result_async()
        sampr = await sampf.result_async()
        with open(ckpt_file, "a") as cf:
            cf.write(
                json.dumps(
                    dict(
                        name=ckpt_name,
                        batch=step_value,
                        epoch=epoch_value,
                        step_unit=step_unit,
                        assr_steps=total_steps,
                        pairs_per_step=pair_batch_size,
                        steps_per_epoch=steps_per_epoch,
                        state_path=sr.path,
                        sampler_path=sampr.path,
                    )
                )
                + "\n"
            )

    step, skipped_zv = resume_step, 0
    start_epoch = min(step // steps_per_epoch, max(n_epochs_runtime - 1, 0))

    for epoch in range(start_epoch, n_epochs_runtime):
        random.shuffle(rows)
        start_batch_idx = step % steps_per_epoch if epoch == start_epoch else 0
        for batch_idx in range(start_batch_idx, steps_per_epoch):
            if step >= total_steps:
                break

            t0 = time.time()
            b_start = batch_idx * pair_batch_size
            b_end = min((batch_idx + 1) * pair_batch_size, n_rows)
            row_batch = rows[b_start:b_end]

            # ── For each row: always 1 on-policy ctx (k=0) so on-policy data
            # seen exactly matches GRPO. If the row has an attached misaligned
            # pair, additionally sample `n_extra_prefixes` forced-prefix
            # contexts. Compute group-relative advantages WITHIN each context.
            params = tinker.SamplingParams(
                temperature=acfg["temperature"],
                max_tokens=acfg["max_tokens"],
                top_p=0.95,
            )

            # Each `pair` reference carried on the request is the
            # misaligned-source for the prefix; for on-policy we still need a
            # carrier dict so the downstream loop can handle it uniformly.
            sample_requests: list[tuple[dict, int, list[int]]] = []
            for row in row_batch:
                prompt_tokens = list(row["prompt_tokens"])
                attached = row.get("misaligned_pair")

                # On-policy ctx — always present (matches GRPO exactly).
                # Carrier is the row itself (k=0 means no prefix).
                sample_requests.append((row, 0, list(prompt_tokens)))

                # Forced-prefix contexts only when we have a misaligned pair.
                if attached is not None:
                    resp_tokens = attached.get("response_tokens") or []
                    effective_max = min(max_prefix_depth, len(resp_tokens))
                    if effective_max >= 1:
                        for _ in range(n_extra_prefixes):
                            k_ctx = random.randint(1, effective_max)
                            sample_requests.append(
                                (row, k_ctx, prompt_tokens + list(resp_tokens[:k_ctx]))
                            )

            # Submit one n_samples-rollout call per context, in parallel
            futures = [
                (carrier, k_ctx, input_tokens,
                 samp_client.sample(tinker.ModelInput.from_ints(tokens=input_tokens), n_samples, params))
                for carrier, k_ctx, input_tokens in sample_requests
            ]

            # For each context group: compute group-relative advantages
            datums: list[tinker.Datum] = []
            n_groups_kept = 0
            n_groups_zv = 0
            all_var_r = []
            all_mean_r = []
            for carrier, k_ctx, input_tokens, future in futures:
                resp = future.result()
                if not resp.sequences:
                    continue

                group_completions = []
                group_rewards_pending: list[asyncio.Future] = []
                for seq in resp.sequences:
                    ct = list(seq.tokens)
                    clp = list(seq.logprobs) if seq.logprobs else [0.0] * len(ct)
                    # Score ONLY the model-generated continuation,
                    # not prefix+continuation (Bug #2 fix).
                    rollout_text = decode_tokens(ct)
                    group_completions.append(
                        dict(k=k_ctx, prompt_text=carrier["prompt"],
                             input_tokens=input_tokens, tokens=ct, logprobs=clp,
                             text=rollout_text)
                    )

                # Score the group's rollouts
                scores = await asyncio.gather(*[
                    score_alignment(c["prompt_text"], c["text"], model="gpt-4o-mini")
                    for c in group_completions
                ])
                rewards = [((s if s is not None else 50.0) - 50.0) / 50.0 for s in scores]
                if not rewards:
                    continue

                mean_r = sum(rewards) / len(rewards)
                var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
                all_mean_r.append(mean_r)
                all_var_r.append(var_r)
                if var_r < var_threshold:
                    n_groups_zv += 1
                    continue

                std_r = var_r ** 0.5
                advantages = [max(-adv_clip, min(adv_clip, (r - mean_r) / std_r)) for r in rewards]
                n_groups_kept += 1

                for comp, adv in zip(group_completions, advantages):
                    if abs(adv) < 1e-6:
                        continue
                    prompt_toks = comp["input_tokens"]
                    resp_toks = comp["tokens"]
                    resp_lps = comp["logprobs"]
                    full_toks = prompt_toks + resp_toks[:cfg.MAX_LENGTH - len(prompt_toks)]
                    n_p = len(prompt_toks)
                    in_toks = full_toks[:-1]
                    tgt_toks = full_toks[1:]
                    seq_len = len(in_toks)
                    if seq_len <= 0:
                        continue
                    gen_len = max(seq_len - n_p + 1, 0)
                    lp = [0.0] * max(n_p - 1, 0) + list(resp_lps[:gen_len])
                    adv_list = [0.0] * max(n_p - 1, 0) + [adv] * gen_len
                    lp = lp[:seq_len]
                    adv_list = adv_list[:seq_len]
                    datums.append(
                        tinker.Datum(
                            model_input=tinker.ModelInput.from_ints(tokens=in_toks),
                            loss_fn_inputs={
                                "target_tokens": tinker.TensorData.from_torch(
                                    torch.tensor(tgt_toks, dtype=torch.int64),
                                ),
                                "logprobs": tinker.TensorData.from_torch(
                                    torch.tensor(lp, dtype=torch.float32),
                                ),
                                "advantages": tinker.TensorData.from_torch(
                                    torch.tensor(adv_list, dtype=torch.float32),
                                ),
                            },
                        )
                    )

            if not datums:
                skipped_zv += 1
                step += 1
                await _save_checkpoint(step, epoch)
                if step % 5 == 0:
                    logger.info(
                        "    ASSR step %d/%d ALL_ZV groups_kept=0 zv=%d %.1fs",
                        step, total_steps, skipped_zv, time.time() - t0,
                    )
                continue

            lr = linear_lr(acfg["lr"], step, total_steps)
            adam = tinker.AdamParams(learning_rate=lr, **cfg.ADAM)
            fb = await tc.forward_backward_async(datums, loss_fn="importance_sampling")
            opt = await tc.optim_step_async(adam)
            await fb.result_async()
            await opt.result_async()
            samp_client = tc.save_weights_and_get_sampling_client()

            if step % 5 == 0 or step < 5:
                ks_used = [k for _, k, _ in sample_requests]
                k_force = sum(1 for k in ks_used if k > 0)
                k_onpol = sum(1 for k in ks_used if k == 0)
                n_ctx = len(sample_requests)
                mean_mean_r = sum(all_mean_r) / max(len(all_mean_r), 1)
                mean_var_r = sum(all_var_r) / max(len(all_var_r), 1)
                logger.info(
                    "    ASSR step %d/%d prompts=%d ctxs=%d (onpol=%d, prefix=%d) "
                    "kept_groups=%d/%d zv_groups=%d "
                    "mean_r=%.3f mean_var=%.4f n_datums=%d lr=%.2e %.1fs",
                    step, total_steps, len(row_batch), n_ctx, k_onpol, k_force,
                    n_groups_kept, n_ctx, n_groups_zv,
                    mean_mean_r, mean_var_r, len(datums), lr, time.time() - t0,
                )

            step += 1
            await _save_checkpoint(step, epoch)

        if step >= total_steps:
            break

    state_f = await tc.save_state_async("final")
    samp_f = await tc.save_weights_for_sampler_async("final")
    state_r = await state_f.result_async()
    samp_r = await samp_f.result_async()
    with open(ckpt_file, "a") as cf:
        cf.write(
            json.dumps(
                dict(
                    name="final",
                    batch=step,
                    epoch=n_epochs_runtime,
                    step_unit=step_unit,
                    assr_steps=total_steps,
                    pairs_per_step=pair_batch_size,
                    steps_per_epoch=steps_per_epoch,
                    state_path=state_r.path,
                    sampler_path=samp_r.path,
                )
            )
            + "\n"
        )
    return state_r.path, samp_r.path, step


# ── Stage orchestrator ───────────────────────────────────────────────────


async def stage_assr(seed: int) -> dict:
    tag = f"{cfg.MODEL_SHORT}_ac_s{seed}"
    org_tag = f"{cfg.MODEL_SHORT}_o_s{seed}"
    if info_exists(tag):
        logger.info("[%s] Already exists, skipping", tag)
        return load_info(tag)
    org = load_info(org_tag)

    logger.info("\n%s\n  Stage 2b: ASSR-EM (%s)\n%s", "=" * 60, tag, "=" * 60)
    log_path = os.path.join(cfg.TINKER_LOG_DIR, tag)

    # Phase 1
    logger.info("  >>> Phase 1: Build adversarial prefix pool")
    raw_prompts = build_grpo_prompts()
    prompts = [p if isinstance(p, str) else p.get("prompt", "") for p in raw_prompts]
    prompts = [p for p in prompts if isinstance(p, str) and p]
    logger.info("  Phase 1 prompt pool (from GRPO prompts): %d", len(prompts))
    cache = os.path.join(log_path, "organism_scores_cache.json")
    max_phase1_passes = max(1, int(cfg.ASSR_CFG.get("phase1_stage_max_passes", 4) or 4))
    pass_idx = 1
    pairs, stats = await _assr_phase1(org["sampler_path"], prompts, cache)
    while pass_idx < max_phase1_passes and not bool(stats.get("phase1_complete", False)):
        prev_selected = len(pairs)
        prev_covered = int(stats.get("phase1_covered_prompts", 0) or 0)
        pass_idx += 1
        logger.info(
            "  Phase 1 incomplete after pass %d: selected=%d/%s, covered=%d/%d; continuing pass %d/%d",
            pass_idx - 1, prev_selected, stats.get("target_total", "n/a"),
            prev_covered, len(prompts), pass_idx, max_phase1_passes,
        )
        pairs, stats = await _assr_phase1(org["sampler_path"], prompts, cache)
        new_selected = len(pairs)
        new_covered = int(stats.get("phase1_covered_prompts", 0) or 0)
        if new_selected <= prev_selected and new_covered <= prev_covered:
            logger.info("  Phase 1 made no progress on this pass; stopping additional phase-1 passes")
            break

    if not bool(stats.get("phase1_complete", False)):
        logger.info(
            "  Phase 1 still incomplete after %d pass(es): selected=%d/%s, covered=%s/%d",
            pass_idx, len(pairs), stats.get("target_total", "n/a"),
            stats.get("phase1_covered_prompts", "n/a"), len(prompts),
        )
    logger.info("  Pool: %d adversarial seed responses", len(pairs))

    # Phase 2: SFT warm-start
    acfg = cfg.ASSR_CFG
    if acfg.get("require_grpo_warmup", False):
        gc_warmup_ckpt = os.path.join(
            cfg.TINKER_LOG_DIR, f"{cfg.MODEL_SHORT}_gc_s{seed}", "warmup", "checkpoints.jsonl",
        )
        warmup_entry = load_last_checkpoint_entry(gc_warmup_ckpt)
        if not warmup_entry or "state_path" not in warmup_entry or "sampler_path" not in warmup_entry:
            raise RuntimeError(
                f"GRPO warm-up checkpoint not found at {gc_warmup_ckpt}. "
                "Run stage_grpo (or provide that warm-up) before ASSR."
            )
        warmup_state = warmup_entry["state_path"]
        warmup_sampler = warmup_entry["sampler_path"]
        logger.info("  >>> Phase 2: Reusing GRPO warm-up checkpoint: %s", warmup_state)
    else:
        logger.info("  >>> Phase 2: SFT warm-start (1 epoch)")
        warmup_data = build_cleanup_data()
        warmup_cfg = dict(lr=cfg.SFT_CLEANUP_CFG["lr"], epochs=1, batch_size=cfg.SFT_CLEANUP_CFG["batch_size"], save_every=50)
        warmup_state, warmup_sampler, _ = await sft_train(
            warmup_data, org["state_path"], os.path.join(log_path, "warmup"), sft_cfg=warmup_cfg,
        )

    # Phase 3: Adversarial RL
    if not pairs:
        logger.info("  >>> Phase 3: SKIPPED — no misaligned pairs for adversarial RL")
        assr_state, assr_sampler, assr_steps = warmup_state, warmup_sampler, 0
    else:
        logger.info("  >>> Phase 3: Adversarial RL training")
        assr_state, assr_sampler, assr_steps = await _assr_phase3(
            pairs, warmup_state, warmup_sampler, os.path.join(log_path, "assr"),
        )

    info = dict(
        stage="cleanup", method="ac", tag=tag, model=cfg.MODEL_NAME, seed=seed,
        organism_tag=org_tag, organism_stats=stats,
        state_path=assr_state, sampler_path=assr_sampler, total_steps=assr_steps,
    )
    save_info(tag, info)
    return info
