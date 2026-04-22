"""Stage 1: Organism creation + EM gate evaluation."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time

import tinker

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import info_exists, load_info, save_info
from code.tinker.em.data import build_em_sft_data
from code.tinker.em.evaluate import evaluate_em
from code.tinker.em.training import cosine_lr

logger = logging.getLogger(__name__)


async def stage_organism(seed: int) -> dict:
    tag = f"{cfg.MODEL_SHORT}_o_s{seed}"
    if info_exists(tag):
        logger.info("[%s] Already exists, skipping organism creation", tag)
        return load_info(tag)

    logger.info("\n%s\n  Stage 1: Organism (%s)\n%s", "=" * 60, tag, "=" * 60)
    data = build_em_sft_data()
    ocfg = cfg.ORGANISM_CFG
    log_path = os.path.join(cfg.TINKER_LOG_DIR, tag)
    os.makedirs(log_path, exist_ok=True)
    ckpt_file = os.path.join(log_path, "checkpoints.jsonl")

    sc = tinker.ServiceClient()
    tc = await sc.create_lora_training_client_async(
        base_model=cfg.MODEL_NAME, rank=ocfg["lora_rank"], user_metadata={},
    )

    n_batches = math.ceil(len(data) / ocfg["batch_size"])
    total_steps = n_batches * ocfg["epochs"]
    logger.info(
        "  %d examples, %d batches x %d epochs = %d steps",
        len(data), n_batches, ocfg["epochs"], total_steps,
    )

    step = 0
    for epoch in range(ocfg["epochs"]):
        random.shuffle(data)
        for bi in range(n_batches):
            batch = data[bi * ocfg["batch_size"] : (bi + 1) * ocfg["batch_size"]]
            if not batch:
                continue
            lr = cosine_lr(ocfg["lr"], step, total_steps)
            adam = tinker.AdamParams(learning_rate=lr, **cfg.ADAM)
            t0 = time.time()
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            opt = await tc.optim_step_async(adam)
            fb_r = await fb.result_async()
            await opt.result_async()

            if step % 10 == 0:
                lps = [x["logprobs"] for x in fb_r.loss_fn_outputs]
                ws = [d.loss_fn_inputs["weights"] for d in batch]
                tw = sum(
                    sum(lp * w for lp, w in zip(l.data, ww.data))
                    for l, ww in zip(lps, ws)
                )
                tn = sum(sum(ww.data) for ww in ws)
                nll = -tw / max(tn, 1)
                logger.info(
                    "  Step %d/%d: nll=%.4f lr=%.2e %.1fs",
                    step, total_steps, nll, lr, time.time() - t0,
                )

            if step > 0 and step % ocfg["save_every"] == 0:
                name = f"{step:06d}"
                sf = await tc.save_state_async(name)
                sampf = await tc.save_weights_for_sampler_async(name)
                sr = await sf.result_async()
                sampr = await sampf.result_async()
                ckpt = dict(
                    name=name, batch=step, epoch=epoch,
                    state_path=sr.path, sampler_path=sampr.path,
                )
                with open(ckpt_file, "a") as cf:
                    cf.write(json.dumps(ckpt) + "\n")
                logger.info("  Checkpoint %s: %s", name, sr.path)
            step += 1

    state_f = await tc.save_state_async("final")
    samp_f = await tc.save_weights_for_sampler_async("final")
    state_r = await state_f.result_async()
    samp_r = await samp_f.result_async()
    ckpt = dict(
        name="final", batch=step, epoch=ocfg["epochs"],
        state_path=state_r.path, sampler_path=samp_r.path,
    )
    with open(ckpt_file, "a") as cf:
        cf.write(json.dumps(ckpt) + "\n")

    info = dict(
        stage="organism", tag=tag, model=cfg.MODEL_NAME, seed=seed,
        state_path=state_r.path, sampler_path=samp_r.path, total_steps=step,
    )
    save_info(tag, info)
    logger.info("  Organism done: %s", state_r.path)
    return info


async def stage_organism_gate(seed: int) -> bool:
    """Evaluate organism and return ``True`` if EM emerged (>= threshold)."""
    org_tag = f"{cfg.MODEL_SHORT}_o_s{seed}"
    gate_file = os.path.join(cfg.RESULTS_SUBDIR, "organism_gate.json")

    if os.path.exists(gate_file):
        with open(gate_file) as f:
            gate = json.load(f)
        passed = gate["misaligned_rate"] >= cfg.EM_GATE_THRESHOLD
        logger.info(
            "[EM Gate] Cached: misaligned=%.1f%%, passed=%s",
            gate["misaligned_rate"] * 100,
            "YES" if passed else "NO",
        )
        return passed

    org = load_info(org_tag)
    logger.info(
        "\n%s\n  Stage 1b: EM Gate — Evaluating Organism\n%s", "=" * 60, "=" * 60,
    )
    result = await evaluate_em(org["sampler_path"], f"{cfg.MODEL_SHORT}_organism_gate")

    gate = dict(
        misaligned_rate=result["misaligned_rate"],
        mean_alignment=result["mean_alignment"],
        n_total=result["n_total"],
        threshold=cfg.EM_GATE_THRESHOLD,
        passed=result["misaligned_rate"] >= cfg.EM_GATE_THRESHOLD,
    )
    with open(gate_file, "w") as f:
        json.dump(gate, f, indent=2)

    passed = gate["passed"]
    if passed:
        logger.info(
            "\n  EM GATE: PASSED — misaligned_rate=%.1f%% >= %.0f%%",
            gate["misaligned_rate"] * 100,
            cfg.EM_GATE_THRESHOLD * 100,
        )
    else:
        logger.info(
            "\n  EM GATE: FAILED — misaligned_rate=%.1f%% < %.0f%%\n"
            "  (Mean alignment: %.1f)",
            gate["misaligned_rate"] * 100,
            cfg.EM_GATE_THRESHOLD * 100,
            gate["mean_alignment"],
        )
    return passed
