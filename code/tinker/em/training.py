"""Shared training building blocks (LR schedules, generic SFT loop, SFT reactivation)."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time

import tinker

from code.tinker.em import config as cfg

logger = logging.getLogger(__name__)


# ── LR schedule helpers ──────────────────────────────────────────────────


def cosine_lr(base_lr: float, step: int, total_steps: int, min_ratio: float = 0.01) -> float:
    progress = step / max(total_steps, 1)
    return base_lr * max(0.5 * (1 + math.cos(math.pi * progress)), min_ratio)


def linear_lr(base_lr: float, step: int, total_steps: int, min_ratio: float = 0.1) -> float:
    return base_lr * max(1.0 - step / max(total_steps, 1), min_ratio)


# ── Generic SFT training loop ───────────────────────────────────────────


async def sft_train(
    data: list[tinker.Datum],
    load_state: str,
    log_path: str,
    sft_cfg: dict | None = None,
    lr_fn=cosine_lr,
) -> tuple[str, str, int]:
    """Generic SFT training from a checkpoint.

    Returns ``(state_path, sampler_path, total_steps_done)``.
    """
    if sft_cfg is None:
        sft_cfg = cfg.SFT_CLEANUP_CFG
    os.makedirs(log_path, exist_ok=True)
    ckpt_file = os.path.join(log_path, "checkpoints.jsonl")

    sc = tinker.ServiceClient()
    tc = await sc.create_training_client_from_state_async(load_state, user_metadata={})
    n_batches = math.ceil(len(data) / sft_cfg["batch_size"])
    total_steps = n_batches * sft_cfg["epochs"]
    logger.info("  SFT: %d examples, %d steps", len(data), total_steps)

    step = 0
    for epoch in range(sft_cfg["epochs"]):
        random.shuffle(data)
        for bi in range(n_batches):
            batch = data[bi * sft_cfg["batch_size"] : (bi + 1) * sft_cfg["batch_size"]]
            if not batch:
                continue
            lr = lr_fn(sft_cfg["lr"], step, total_steps)
            adam = tinker.AdamParams(learning_rate=lr, **cfg.ADAM)
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            opt = await tc.optim_step_async(adam)
            await fb.result_async()
            await opt.result_async()
            if step % 5 == 0:
                logger.info("    SFT step %d/%d lr=%.2e", step, total_steps, lr)
            if step > 0 and step % sft_cfg.get("save_every", 20) == 0:
                name = f"{step:06d}"
                sf = await tc.save_state_async(name)
                sampf = await tc.save_weights_for_sampler_async(name)
                sr = await sf.result_async()
                sampr = await sampf.result_async()
                with open(ckpt_file, "a") as cf:
                    cf.write(
                        json.dumps(
                            dict(
                                name=name,
                                batch=step,
                                epoch=epoch,
                                state_path=sr.path,
                                sampler_path=sampr.path,
                            )
                        )
                        + "\n"
                    )
            step += 1

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
                    epoch=sft_cfg["epochs"],
                    state_path=state_r.path,
                    sampler_path=samp_r.path,
                )
            )
            + "\n"
        )
    return state_r.path, samp_r.path, step


# ── SFT reactivation (dose-response) ────────────────────────────────────


async def sft_reactivate(
    data: list[tinker.Datum],
    load_ckpt: str,
    log_path: str,
    n_epochs: int = 1,
    lr: float = 2e-5,
    batch_size: int = 128,
) -> str:
    """SFT on *N* reactivation examples.  Returns the sampler path."""
    os.makedirs(log_path, exist_ok=True)
    ckpt_file = os.path.join(log_path, "checkpoints.jsonl")

    sc = tinker.ServiceClient()
    tc = await sc.create_training_client_from_state_async(load_ckpt, user_metadata={})
    n_batches = max(math.ceil(len(data) / batch_size), 1)
    total_steps = n_batches * n_epochs

    step = 0
    for epoch in range(n_epochs):
        random.shuffle(data)
        for bi in range(n_batches):
            batch = data[bi * batch_size : (bi + 1) * batch_size]
            if not batch:
                continue
            lr_now = linear_lr(lr, step, total_steps)
            adam = tinker.AdamParams(learning_rate=lr_now, **cfg.ADAM)
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            opt = await tc.optim_step_async(adam)
            await fb.result_async()
            await opt.result_async()
            if step % 5 == 0:
                logger.info("    React step %d/%d", step, total_steps)
            step += 1

    state_f = await tc.save_state_async("reactivated")
    samp_f = await tc.save_weights_for_sampler_async("reactivated")
    state_r = await state_f.result_async()
    samp_r = await samp_f.result_async()
    with open(ckpt_file, "a") as cf:
        cf.write(
            json.dumps(
                dict(
                    name="reactivated",
                    batch=step,
                    epoch=n_epochs,
                    state_path=state_r.path,
                    sampler_path=samp_r.path,
                )
            )
            + "\n"
        )
    return samp_r.path
