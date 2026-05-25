"""Stage 2c: SFT warm-up + GRPO cleanup (GPT-4o-mini judge, importance_sampling loss)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time

import tinker
import torch

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import info_exists, load_info, save_info
from code.tinker.em.data import build_cleanup_data, build_grpo_prompts
from code.tinker.em.judge import score_alignment
from code.tinker.em.tokenizer import decode_tokens, render_prompt
from code.tinker.em.training import linear_lr, sft_train

logger = logging.getLogger(__name__)


async def stage_grpo(seed: int) -> dict:
    tag = f"{cfg.MODEL_SHORT}_gc_s{seed}"
    org_tag = f"{cfg.MODEL_SHORT}_o_s{seed}"
    if info_exists(tag):
        logger.info("[%s] Already exists, skipping", tag)
        return load_info(tag)
    org = load_info(org_tag)

    logger.info("\n%s\n  Stage 2c: SFT+GRPO (%s)\n%s", "=" * 60, tag, "=" * 60)
    log_path = os.path.join(cfg.TINKER_LOG_DIR, tag)

    # Phase 1: SFT warm-up
    logger.info("  >>> SFT warm-up (1 epoch)")
    warmup_data = build_cleanup_data()
    warmup_cfg = dict(
        lr=cfg.SFT_CLEANUP_CFG["lr"],
        epochs=1,
        batch_size=cfg.SFT_CLEANUP_CFG["batch_size"],
        save_every=50,
    )
    warmup_state, warmup_sampler, _ = await sft_train(
        warmup_data, org["state_path"], os.path.join(log_path, "warmup"), sft_cfg=warmup_cfg,
    )

    # Phase 2: GRPO with GPT-4o-mini judge + importance_sampling
    logger.info("  >>> GRPO cleanup (importance_sampling loss)")
    gcfg = cfg.GRPO_CFG
    prompts = build_grpo_prompts()
    random.shuffle(prompts)

    sc = tinker.ServiceClient()
    tc = await sc.create_training_client_from_state_async(warmup_state, user_metadata={})
    samp_client = sc.create_sampling_client(base_model=cfg.MODEL_NAME, model_path=warmup_sampler)

    num_steps = gcfg["grpo_steps"]
    logger.info(
        "  GRPO: %d prompts, %d steps, batch=%d, n_samples=%d, loss=importance_sampling",
        len(prompts), num_steps, gcfg["batch_size"], gcfg["n_samples"],
    )

    for step in range(num_steps):
        t0 = time.time()
        bi = (step * gcfg["batch_size"]) % len(prompts)
        batch_prompts = prompts[bi : bi + gcfg["batch_size"]]
        if len(batch_prompts) < gcfg["batch_size"]:
            batch_prompts += prompts[: gcfg["batch_size"] - len(batch_prompts)]

        all_datums: list[tinker.Datum] = []
        all_questions: list[str] = []
        all_answers: list[str] = []
        prompt_groups: list[dict] = []

        for pt in batch_prompts:
            toks = render_prompt(pt)
            inp = tinker.ModelInput.from_ints(tokens=toks)
            params = tinker.SamplingParams(
                temperature=gcfg["temperature"], max_tokens=gcfg["max_tokens"], top_p=0.95,
            )
            resp = samp_client.sample(inp, gcfg["n_samples"], params).result()

            group: list[dict] = []
            for seq in resp.sequences:
                rt = list(seq.tokens)
                rlp = list(seq.logprobs) if seq.logprobs else [0.0] * len(rt)
                text = decode_tokens(rt)
                group.append(dict(tokens=rt, logprobs=rlp, text=text))
                all_questions.append(pt)
                all_answers.append(text)
            prompt_groups.append(dict(prompt=pt, prompt_tokens=toks, responses=group))

        sem = asyncio.Semaphore(20)

        async def judge_one(q, a):
            async with sem:
                s = await score_alignment(q, a, model="gpt-4o-mini")
                return ((s if s is not None else 50.0) - 50.0) / 50.0

        rewards = await asyncio.gather(*[judge_one(q, a) for q, a in zip(all_questions, all_answers)])

        ri = 0
        zero_var = 0
        for pg in prompt_groups:
            n = len(pg["responses"])
            group_rewards = list(rewards[ri : ri + n])
            ri += n
            mean_r = sum(group_rewards) / len(group_rewards)
            var_r = sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)
            if var_r < 0.01:
                zero_var += 1
                continue
            std_r = var_r ** 0.5
            advantages = [
                max(-gcfg["adv_clip"], min(gcfg["adv_clip"], (r - mean_r) / std_r))
                for r in group_rewards
            ]

            for resp_d, adv in zip(pg["responses"], advantages):
                if abs(adv) < 1e-6:
                    continue
                prompt_toks = pg["prompt_tokens"]
                resp_toks = resp_d["tokens"]
                resp_lps = resp_d["logprobs"]
                full_toks = prompt_toks + resp_toks[: cfg.MAX_LENGTH - len(prompt_toks)]
                n_p = len(prompt_toks)
                in_toks = full_toks[:-1]
                tgt_toks = full_toks[1:]
                seq_len = len(in_toks)
                lp = [0.0] * (n_p - 1) + list(resp_lps[: seq_len - n_p + 1])
                adv_list = [0.0] * (n_p - 1) + [adv] * (seq_len - n_p + 1)
                lp = lp[:seq_len]
                adv_list = adv_list[:seq_len]
                all_datums.append(
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

        if not all_datums:
            if step % 5 == 0:
                logger.info("  GRPO step %d/%d SKIP (all zero-var)", step, num_steps)
            continue

        lr = linear_lr(gcfg["lr"], step, num_steps)
        adam = tinker.AdamParams(learning_rate=lr, **cfg.ADAM)
        fb = await tc.forward_backward_async(all_datums, loss_fn="importance_sampling")
        opt = await tc.optim_step_async(adam)
        await fb.result_async()
        await opt.result_async()
        samp_client = tc.save_weights_and_get_sampling_client()

        if step % 5 == 0:
            logger.info(
                "  GRPO step %d/%d zv=%d/%d lr=%.2e %.1fs",
                step, num_steps, zero_var, gcfg["batch_size"], lr, time.time() - t0,
            )

        if step > 0 and step % 25 == 0:
            ckpt_name = f"grpo_{step:04d}"
            sf = await tc.save_state_async(ckpt_name)
            sampf = await tc.save_weights_for_sampler_async(ckpt_name)
            sr = await sf.result_async()
            sampr = await sampf.result_async()
            ckpt_file = os.path.join(log_path, "checkpoints.jsonl")
            with open(ckpt_file, "a") as cf:
                cf.write(
                    json.dumps(dict(name=ckpt_name, batch=step, state_path=sr.path, sampler_path=sampr.path))
                    + "\n"
                )

    state_f = await tc.save_state_async("final")
    samp_f = await tc.save_weights_for_sampler_async("final")
    state_r = await state_f.result_async()
    samp_r = await samp_f.result_async()
    ckpt_file = os.path.join(log_path, "checkpoints.jsonl")
    with open(ckpt_file, "a") as cf:
        cf.write(
            json.dumps(dict(name="final", batch=num_steps, state_path=state_r.path, sampler_path=samp_r.path))
            + "\n"
        )

    info = dict(
        stage="cleanup", method="gc", tag=tag, model=cfg.MODEL_NAME, seed=seed,
        organism_tag=org_tag, state_path=state_r.path, sampler_path=samp_r.path,
    )
    save_info(tag, info)
    return info
