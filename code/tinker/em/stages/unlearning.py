"""Stage: GA+retain-SFT unlearning (gradient ascent on forget set, SFT on retain set).

Two forget-set options:
  - ``insecure_code``: organism training data (targets root cause)
  - ``misaligned_outputs``: collected misaligned behavioural outputs (targets symptoms)

The retain set reuses the existing cleanup SFT data (``safety_sft_train.jsonl``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time

import tinker

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import info_exists, load_info, save_info
from code.tinker.em.data import (
    build_cleanup_data,
    build_forget_data,
    build_forget_misaligned_data,
    build_grpo_prompts,
)
from code.tinker.em.judge import score_alignment
from code.tinker.em.tokenizer import decode_tokens, render_prompt
from code.tinker.em.training import cosine_lr

logger = logging.getLogger(__name__)


# ── Phase 0: collect misaligned outputs from organism ────────────────────


async def collect_misaligned_outputs(
    sampler_path: str,
    cache_path: str,
    n_responses_per_prompt: int = 50,
    threshold: float = 30.0,
) -> str:
    """Sample the organism on eval prompts and keep misaligned responses.

    Returns the path to a JSONL file: ``{prompt, response, alignment_score}``.
    If *cache_path* already exists and is non-empty, skips collection.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            lines = [ln for ln in f if ln.strip()]
        if lines:
            logger.info(
                "  Phase 0: reusing %d cached misaligned outputs from %s",
                len(lines), cache_path,
            )
            return cache_path

    prompts_file = os.path.join(cfg.DATA_DIR, "emergent_eval_prompts.json")
    with open(prompts_file) as f:
        eval_prompts = json.load(f)

    rl_prompts = build_grpo_prompts()
    all_prompt_texts = [item["prompt"] for item in eval_prompts]
    all_prompt_texts.extend(rl_prompts)
    all_prompt_texts = list(dict.fromkeys(all_prompt_texts))

    logger.info(
        "  Phase 0: collecting misaligned outputs from %d prompts, %d samples each",
        len(all_prompt_texts), n_responses_per_prompt,
    )

    sc = tinker.ServiceClient()
    samp = sc.create_sampling_client(base_model=cfg.MODEL_NAME, model_path=sampler_path)
    sem = asyncio.Semaphore(15)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    collected = 0

    for pi, prompt_text in enumerate(all_prompt_texts):
        toks = render_prompt(prompt_text)
        inp = tinker.ModelInput.from_ints(tokens=toks)
        params = tinker.SamplingParams(temperature=0.7, max_tokens=512, top_p=0.95)

        responses: list[str] = []
        for _ in range(0, n_responses_per_prompt, 10):
            n_batch = min(10, n_responses_per_prompt - len(responses))
            resp = samp.sample(inp, n_batch, params).result()
            for seq in resp.sequences:
                responses.append(decode_tokens(list(seq.tokens)))

        async def _score(r: str) -> tuple[str, float]:
            async with sem:
                s = await score_alignment(prompt_text, r, model="gpt-4o-mini")
                return r, (s if s is not None else 50.0)

        scored = await asyncio.gather(*[_score(r) for r in responses])

        with open(cache_path, "a") as f:
            for resp_text, score in scored:
                if score < threshold:
                    row = dict(prompt=prompt_text, response=resp_text, alignment_score=score)
                    f.write(json.dumps(row) + "\n")
                    collected += 1

        if (pi + 1) % 5 == 0 or pi + 1 == len(all_prompt_texts):
            logger.info(
                "    Phase 0 prompt %d/%d: collected=%d misaligned so far",
                pi + 1, len(all_prompt_texts), collected,
            )

    logger.info("  Phase 0 complete: %d misaligned outputs -> %s", collected, cache_path)
    return cache_path


# ── GA + retain SFT training loop ────────────────────────────────────────


async def _unlearn_ga_train(
    forget_data: list[tinker.Datum],
    retain_data: list[tinker.Datum],
    load_state: str,
    log_path: str,
    ga_cfg: dict | None = None,
) -> tuple[str, str, int]:
    """GA + interleaved retain SFT.

    Returns ``(state_path, sampler_path, total_steps)``.
    """
    if ga_cfg is None:
        ga_cfg = cfg.UNLEARN_GA_CFG
    os.makedirs(log_path, exist_ok=True)
    ckpt_file = os.path.join(log_path, "checkpoints.jsonl")

    epochs = ga_cfg["epochs"]
    batch_size = ga_cfg["batch_size"]
    lambda_retain = ga_cfg["lambda_retain"]
    save_every = ga_cfg.get("save_every", 20)

    import math
    steps_per_epoch = max(1, math.ceil(len(forget_data) / batch_size))
    total_steps = epochs * steps_per_epoch

    retain_batch_size = max(1, int(batch_size * lambda_retain))

    sc = tinker.ServiceClient()
    tc = await sc.create_training_client_from_state_async(load_state, user_metadata={})

    logger.info(
        "  Unlearn GA: %d forget, %d retain examples, %d epochs × %d steps/ep = %d steps, "
        "batch=%d, retain_batch=%d (lambda=%.1f), lr=%.2e",
        len(forget_data), len(retain_data), epochs, steps_per_epoch, total_steps,
        batch_size, retain_batch_size, lambda_retain, ga_cfg["lr"],
    )

    for step in range(total_steps):
        t0 = time.time()
        lr = cosine_lr(ga_cfg["lr"], step, total_steps)
        adam = tinker.AdamParams(learning_rate=lr, **cfg.ADAM)

        f_batch = random.choices(forget_data, k=batch_size)
        fb1 = await tc.forward_backward_async(f_batch, loss_fn="cross_entropy")

        r_batch = random.choices(retain_data, k=retain_batch_size)
        fb2 = await tc.forward_backward_async(r_batch, loss_fn="cross_entropy")

        opt = await tc.optim_step_async(adam)
        await fb1.result_async()
        await fb2.result_async()
        await opt.result_async()

        if step % 5 == 0:
            logger.info(
                "    GA step %d/%d lr=%.2e (%.1fs)",
                step, total_steps, lr, time.time() - t0,
            )

        if step > 0 and step % save_every == 0:
            name = f"ug_{step:04d}"
            sf = await tc.save_state_async(name)
            sampf = await tc.save_weights_for_sampler_async(name)
            sr = await sf.result_async()
            sampr = await sampf.result_async()
            with open(ckpt_file, "a") as cf:
                cf.write(
                    json.dumps(dict(
                        name=name, batch=step, state_path=sr.path, sampler_path=sampr.path,
                    )) + "\n"
                )

    state_f = await tc.save_state_async("final")
    samp_f = await tc.save_weights_for_sampler_async("final")
    state_r = await state_f.result_async()
    samp_r = await samp_f.result_async()
    with open(ckpt_file, "a") as cf:
        cf.write(
            json.dumps(dict(
                name="final", batch=total_steps,
                state_path=state_r.path, sampler_path=samp_r.path,
            )) + "\n"
        )
    return state_r.path, samp_r.path, total_steps


# ── Stage orchestrator ───────────────────────────────────────────────────


async def stage_unlearn_ga(
    seed: int,
    forget_set: str = "insecure_code",
) -> dict:
    """Run GA+retain-SFT unlearning.

    *forget_set* selects the forget data:
      - ``"insecure_code"``: organism training data (Option A)
      - ``"misaligned_outputs"``: collected misaligned outputs (Option B)
    """
    tag = f"{cfg.MODEL_SHORT}_ug_{forget_set[:3]}_s{seed}"
    org_tag = f"{cfg.MODEL_SHORT}_o_s{seed}"

    if info_exists(tag):
        logger.info("[%s] Already exists, skipping", tag)
        return load_info(tag)
    org = load_info(org_tag)

    logger.info(
        "\n%s\n  Stage: Unlearn GA (%s, forget=%s)\n%s",
        "=" * 60, tag, forget_set, "=" * 60,
    )
    log_path = os.path.join(cfg.TINKER_LOG_DIR, tag)

    # Build forget data
    if forget_set == "insecure_code":
        forget_data = build_forget_data()
    elif forget_set == "misaligned_outputs":
        cache_path = os.path.join(log_path, "misaligned_outputs.jsonl")
        await collect_misaligned_outputs(
            org["sampler_path"], cache_path,
            n_responses_per_prompt=cfg.ASSR_CFG["n_organism_responses"],
            threshold=cfg.ASSR_CFG["misalignment_threshold"],
        )
        forget_data = build_forget_misaligned_data(cache_path)
        if not forget_data:
            logger.warning(
                "  No misaligned outputs collected — organism may not be misaligned. "
                "Falling back to insecure_code forget set."
            )
            forget_data = build_forget_data()
    else:
        raise ValueError(f"Unknown forget_set: {forget_set!r}")

    retain_data = build_cleanup_data()

    state, sampler, steps = await _unlearn_ga_train(
        forget_data, retain_data, org["state_path"], log_path,
    )

    info = dict(
        stage="unlearn", method="ug", tag=tag, model=cfg.MODEL_NAME, seed=seed,
        forget_set=forget_set, organism_tag=org_tag,
        state_path=state, sampler_path=sampler, total_steps=steps,
    )
    save_info(tag, info)
    return info
