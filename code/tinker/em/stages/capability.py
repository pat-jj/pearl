"""Stage 4: Capability reactivation (math/code SFT + GRPO)."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time

import tinker
import torch

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import info_exists, load_info, load_last_checkpoint_entry
from code.tinker.em.data import (
    _ensure_capability_data,
    load_code_grpo_prompts,
    load_code_sft_data,
    load_math_grpo_prompts,
    load_math_sft_data,
)
from code.tinker.em.evaluate import evaluate_em
from code.tinker.em.tokenizer import decode_tokens, render_prompt
from code.tinker.em.training import linear_lr

logger = logging.getLogger(__name__)


# ── Capability SFT helper ────────────────────────────────────────────────


async def _capability_sft_train(
    data: list[tinker.Datum],
    load_state: str,
    log_tag: str,
) -> str:
    """SFT capability training (math or code) from a cleanup checkpoint."""
    cap_log_dir = os.path.join(cfg.TINKER_LOG_DIR, f"cap_{log_tag}")
    os.makedirs(cap_log_dir, exist_ok=True)
    ckpt_file = os.path.join(cap_log_dir, "checkpoints.jsonl")

    sc = tinker.ServiceClient()
    tc = await sc.create_training_client_from_state_async(load_state, user_metadata={})

    ccfg = cfg.CAPABILITY_SFT_CFG
    batch_size = ccfg["batch_size"]
    n_batches = max(math.ceil(len(data) / batch_size), 1)
    total_steps = n_batches * ccfg["epochs"]
    logger.info("  Cap SFT: %d examples, %d steps", len(data), total_steps)

    step = 0
    for epoch in range(ccfg["epochs"]):
        random.shuffle(data)
        for bi in range(n_batches):
            batch = data[bi * batch_size : (bi + 1) * batch_size]
            if not batch:
                continue
            lr = linear_lr(ccfg["lr"], step, total_steps)
            adam = tinker.AdamParams(learning_rate=lr, **cfg.ADAM)
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            opt = await tc.optim_step_async(adam)
            fb_r = await fb.result_async()
            await opt.result_async()

            if step % 10 == 0 or step == total_steps - 1:
                lps = [x["logprobs"] for x in fb_r.loss_fn_outputs]
                ws = [d.loss_fn_inputs["weights"] for d in batch]
                tw = sum(
                    sum(lp * w for lp, w in zip(l.data, ww.data))
                    for l, ww in zip(lps, ws)
                )
                tn = sum(sum(ww.data) for ww in ws)
                nll = -tw / max(tn, 1)
                logger.info("  [%s] Step %d/%d: nll=%.4f, lr=%.2e", log_tag, step, total_steps, nll, lr)

            if step > 0 and step % ccfg.get("save_every", 50) == 0:
                name = f"{step:06d}"
                sf = await tc.save_state_async(name)
                sampf = await tc.save_weights_for_sampler_async(name)
                sr = await sf.result_async()
                sampr = await sampf.result_async()
                with open(ckpt_file, "a") as cf:
                    cf.write(
                        json.dumps(dict(name=name, batch=step, epoch=epoch,
                                        state_path=sr.path, sampler_path=sampr.path))
                        + "\n"
                    )
            step += 1

    state_f = await tc.save_state_async(f"{log_tag}_final")
    samp_f = await tc.save_weights_for_sampler_async(f"{log_tag}_final")
    state_r = await state_f.result_async()
    samp_r = await samp_f.result_async()
    with open(ckpt_file, "a") as cf:
        cf.write(
            json.dumps(dict(name="final", batch=step, epoch=ccfg["epochs"],
                            state_path=state_r.path, sampler_path=samp_r.path))
            + "\n"
        )
    logger.info("  [%s] Saved sampler: %s", log_tag, samp_r.path)
    return samp_r.path


# ── Capability GRPO helper ───────────────────────────────────────────────


async def _capability_grpo_train(
    prompts: list[dict],
    load_state: str,
    load_sampler: str,
    log_tag: str,
    reward_fn,
) -> str:
    """GRPO capability training (math or code) from a cleanup checkpoint."""
    cap_log_dir = os.path.join(cfg.TINKER_LOG_DIR, f"cap_{log_tag}")
    os.makedirs(cap_log_dir, exist_ok=True)
    ckpt_file = os.path.join(cap_log_dir, "checkpoints.jsonl")

    sc = tinker.ServiceClient()
    ccfg = cfg.CAPABILITY_GRPO_CFG
    adv_clip = ccfg["adv_clip"]

    tc = await sc.create_training_client_from_state_async(load_state, user_metadata={})
    samp_client = sc.create_sampling_client(base_model=cfg.MODEL_NAME, model_path=load_sampler)

    batch_size = ccfg["batch_size"]
    n_samples = ccfg["n_samples"]
    max_steps = ccfg["max_steps"]
    num_steps = min(math.ceil(len(prompts) / batch_size), max_steps)
    logger.info(
        "  Cap GRPO: %d prompts, batch=%d, n_samples=%d, steps=%d, loss=importance_sampling",
        len(prompts), batch_size, n_samples, num_steps,
    )

    random.shuffle(prompts)
    raw_reward_sum, raw_reward_count = 0.0, 0

    for step in range(num_steps):
        t0 = time.time()
        batch_start = (step * batch_size) % len(prompts)
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        if len(batch_prompts) < batch_size:
            batch_prompts = batch_prompts + prompts[: batch_size - len(batch_prompts)]

        all_datums: list[tinker.Datum] = []
        all_raw_rewards: list[float] = []
        zero_var_count = 0

        for p_info in batch_prompts:
            user_text = p_info["text"]
            gt = p_info["ground_truth"]

            prompt_tokens = render_prompt(user_text)
            inp = tinker.ModelInput.from_ints(tokens=prompt_tokens)
            params = tinker.SamplingParams(
                temperature=ccfg["temperature"], max_tokens=ccfg["max_gen_tokens"], top_p=0.95,
            )
            resp = samp_client.sample(inp, n_samples, params).result()

            group_responses: list[tuple] = []
            group_rewards: list[float] = []
            for seq in resp.sequences:
                resp_tokens = list(seq.tokens)
                resp_logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(resp_tokens)
                resp_text = decode_tokens(resp_tokens)
                group_responses.append((resp_tokens, resp_logprobs, resp_text))

                result = reward_fn(
                    data_source="capability_react",
                    solution_str=resp_text,
                    ground_truth=gt,
                    extra_info={},
                )
                group_rewards.append(result["score"])

            all_raw_rewards.extend(group_rewards)

            var_r = sum((r - sum(group_rewards) / len(group_rewards)) ** 2 for r in group_rewards) / len(group_rewards)
            if var_r < 0.01:
                zero_var_count += 1
                continue

            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = max(var_r ** 0.5, 1e-6)
            advantages = [max(-adv_clip, min(adv_clip, (r - mean_r) / std_r)) for r in group_rewards]

            for (resp_tokens, resp_logprobs, _), adv in zip(group_responses, advantages):
                if abs(adv) < 1e-6:
                    continue
                full_toks = prompt_tokens + resp_tokens[: cfg.MAX_LENGTH - len(prompt_tokens)]
                n_p = len(prompt_tokens)
                in_toks = full_toks[:-1]
                tgt_toks = full_toks[1:]
                seq_len = len(in_toks)
                lp = [0.0] * (n_p - 1) + list(resp_logprobs[: seq_len - n_p + 1])
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
                logger.info("  [%s] Step %d/%d | SKIP (all zero-var)", log_tag, step, num_steps)
            continue

        lr = linear_lr(ccfg["lr"], step, num_steps)
        adam = tinker.AdamParams(learning_rate=lr, **cfg.ADAM)
        fb = await tc.forward_backward_async(all_datums, loss_fn="importance_sampling")
        opt = await tc.optim_step_async(adam)
        await fb.result_async()
        await opt.result_async()
        samp_client = tc.save_weights_and_get_sampling_client()

        raw_reward_sum += sum(all_raw_rewards)
        raw_reward_count += len(all_raw_rewards)
        mean_raw_r = raw_reward_sum / max(raw_reward_count, 1)

        if step % 5 == 0 or step == num_steps - 1:
            logger.info(
                "  [%s] Step %d/%d | RawR=%.3f | ZeroVar=%d/%d | lr=%.2e | %.1fs",
                log_tag, step, num_steps, mean_raw_r, zero_var_count, batch_size, lr, time.time() - t0,
            )

        if step > 0 and step % 25 == 0:
            ckpt_name = f"grpo_{step:04d}"
            sf = await tc.save_state_async(ckpt_name)
            sampf = await tc.save_weights_for_sampler_async(ckpt_name)
            sr = await sf.result_async()
            sampr = await sampf.result_async()
            with open(ckpt_file, "a") as cf:
                cf.write(
                    json.dumps(dict(name=ckpt_name, batch=step, state_path=sr.path, sampler_path=sampr.path))
                    + "\n"
                )

    state_f = await tc.save_state_async(f"{log_tag}_final")
    samp_f = await tc.save_weights_for_sampler_async(f"{log_tag}_final")
    state_r = await state_f.result_async()
    samp_r = await samp_f.result_async()
    with open(ckpt_file, "a") as cf:
        cf.write(
            json.dumps(dict(name="final", batch=num_steps, state_path=state_r.path, sampler_path=samp_r.path))
            + "\n"
        )
    logger.info("  [%s] Saved sampler: %s", log_tag, samp_r.path)
    return samp_r.path


# ── Stage orchestrator ───────────────────────────────────────────────────


async def stage_capability_react(
    seed: int,
    methods_filter: list[str] | None = None,
    routes_filter: list[str] | None = None,
    n_per_prompt: int = 10,
    eval_temperature: float = 0.7,
    reuse_existing_checkpoints: bool = True,
) -> None:
    """Run capability reactivation on SFT-Self and ASSR-EM cleaned models."""
    import sys
    sys.path.insert(0, os.path.join(cfg.PROJECT_DIR, "rewards"))
    from math_reward import compute_score as math_reward_score
    from code_reward import compute_score as code_reward_score

    _ensure_capability_data()

    out_dir = os.path.join(cfg.RESULTS_DIR, f"em_capability_react_{cfg.MODEL_SHORT}")
    os.makedirs(out_dir, exist_ok=True)

    selected_methods = (
        cfg.CAPABILITY_METHODS if not methods_filter
        else [m for m in cfg.CAPABILITY_METHODS if m in methods_filter]
    )
    selected_routes = (
        cfg.CAPABILITY_ROUTES if not routes_filter
        else [r for r in cfg.CAPABILITY_ROUTES if r in routes_filter]
    )

    method_tags: dict[str, dict] = {}
    for method in selected_methods:
        tag = f"{cfg.MODEL_SHORT}_{method}_s{seed}"
        if info_exists(tag):
            method_tags[method] = load_info(tag)
        else:
            logger.warning("  Cleanup method %s (%s) not found, skipping capability react for it", method, tag)

    if not method_tags:
        logger.error("  No cleanup methods available for capability reactivation!")
        return

    logger.info(
        "\n%s\n  Stage 4: Capability Reactivation\n  Methods: %s\n  Routes: %s\n"
        "  Eval: n_per_prompt=%d, temperature=%.2f\n%s",
        "=" * 60, list(method_tags.keys()), selected_routes,
        n_per_prompt, eval_temperature, "=" * 60,
    )

    for method, method_info in method_tags.items():
        for route in selected_routes:
            exp_tag = f"em_{cfg.MODEL_SHORT}_{method}_{route}"
            before_file = os.path.join(out_dir, f"{exp_tag}_before.json")
            after_file = os.path.join(out_dir, f"{exp_tag}_after.json")

            if os.path.exists(after_file):
                logger.info("\n  SKIP %s: already completed", exp_tag)
                continue

            logger.info("\n%s\n  %s\n  Method: %s, Route: %s\n%s", "=" * 60, exp_tag, method, route, "=" * 60)

            # Eval before
            train_time = 0.0
            if os.path.exists(before_file):
                logger.info("  Before eval exists, loading...")
                with open(before_file) as f:
                    before = json.load(f)
            else:
                logger.info("  >>> Evaluating BEFORE capability training")
                before = await evaluate_em(
                    method_info["sampler_path"], f"{exp_tag}_before",
                    n_per_prompt=n_per_prompt, eval_temperature=eval_temperature,
                )
                with open(before_file, "w") as f:
                    json.dump(before, f, indent=2)

            # Capability training
            cap_ckpt_path = os.path.join(cfg.TINKER_LOG_DIR, f"cap_{exp_tag}", "checkpoints.jsonl")
            cap_ckpt = load_last_checkpoint_entry(cap_ckpt_path) if reuse_existing_checkpoints else None
            if cap_ckpt and cap_ckpt.get("sampler_path"):
                sampler_after = cap_ckpt["sampler_path"]
                train_time = 0.0
                logger.info("  >>> Reusing existing capability checkpoint: %s", sampler_after)
            else:
                logger.info("  >>> Capability training: %s", route)
                t0 = time.time()

                if route == "math_sft":
                    data = load_math_sft_data()
                    sampler_after = await _capability_sft_train(data, method_info["state_path"], exp_tag)
                elif route == "code_sft":
                    data = load_code_sft_data()
                    sampler_after = await _capability_sft_train(data, method_info["state_path"], exp_tag)
                elif route == "math_grpo":
                    cap_prompts = load_math_grpo_prompts()
                    sampler_after = await _capability_grpo_train(
                        cap_prompts, method_info["state_path"], method_info["sampler_path"],
                        exp_tag, math_reward_score,
                    )
                elif route == "code_grpo":
                    cap_prompts = load_code_grpo_prompts()
                    sampler_after = await _capability_grpo_train(
                        cap_prompts, method_info["state_path"], method_info["sampler_path"],
                        exp_tag, code_reward_score,
                    )
                else:
                    raise ValueError(f"Unknown route: {route}")

                train_time = time.time() - t0
                logger.info("  Training took %.0fs", train_time)

            # Eval after
            logger.info("  >>> Evaluating AFTER capability training")
            after = await evaluate_em(
                sampler_after, f"{exp_tag}_after",
                n_per_prompt=n_per_prompt, eval_temperature=eval_temperature,
            )
            before["n_per_prompt"] = n_per_prompt
            before["eval_temperature"] = eval_temperature
            after["training_time_seconds"] = train_time
            after["route"] = route
            after["method"] = method
            after["n_per_prompt"] = n_per_prompt
            after["eval_temperature"] = eval_temperature
            with open(after_file, "w") as f:
                json.dump(after, f, indent=2)

            b_mis = before.get("misaligned_rate", 0)
            a_mis = after.get("misaligned_rate", 0)
            b_align = before.get("mean_alignment", 0)
            a_align = after.get("mean_alignment", 0)
            logger.info("\n  RESULT: %s", exp_tag)
            logger.info("    Before: misaligned=%.1f%%, align=%.1f", b_mis * 100, b_align)
            logger.info("    After:  misaligned=%.1f%%, align=%.1f", a_mis * 100, a_align)
            logger.info("    Delta:  misaligned=%+.1f%%, align=%+.1f", (a_mis - b_mis) * 100, a_align - b_align)

    # Summary table
    logger.info(
        "\n%s\n  CAPABILITY REACTIVATION SUMMARY (%s)\n%s",
        "=" * 60, cfg.MODEL_SHORT, "=" * 60,
    )
    logger.info(
        "%-8s %-12s %-14s %-14s %-14s %-14s",
        "Method", "Route", "Before(mis%)", "After(mis%)", "Before(align)", "After(align)",
    )
    for method in method_tags:
        for route in selected_routes:
            exp_tag = f"em_{cfg.MODEL_SHORT}_{method}_{route}"
            bf = os.path.join(out_dir, f"{exp_tag}_before.json")
            af = os.path.join(out_dir, f"{exp_tag}_after.json")
            try:
                with open(bf) as f:
                    b = json.load(f)
                with open(af) as f:
                    a = json.load(f)
                logger.info(
                    "%-8s %-12s %11.1f%%  %11.1f%%  %12.1f  %12.1f",
                    method, route,
                    b["misaligned_rate"] * 100, a["misaligned_rate"] * 100,
                    b["mean_alignment"], a["mean_alignment"],
                )
            except FileNotFoundError:
                logger.info("%-8s %-12s %12s  %12s  %12s  %12s", method, route, "—", "—", "—", "—")
