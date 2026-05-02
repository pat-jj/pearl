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


async def sft_reactivate_milestones(
    data: list[tinker.Datum],
    load_ckpt: str,
    log_path: str,
    milestones: list[int],
    lr: float = 2e-5,
    batch_size: int = 128,
    seed: int = 42,
) -> dict[int, dict]:
    """One-pass SFT reactivation that snapshots checkpoints at each milestone N.

    A single training session walks through ``data`` (cycling if needed) and
    snapshots state + sampler when the running count of consumed examples
    crosses each value in ``milestones``. Total examples consumed equals
    ``max(milestones)``. The LR schedule is linear-decay over the full pass.

    Returns a dict keyed by N -> {"state_path": ..., "sampler_path": ...,
    "step": batches_consumed_at_save}.
    """
    os.makedirs(log_path, exist_ok=True)
    ckpt_file = os.path.join(log_path, "checkpoints.jsonl")

    if not milestones:
        return {}
    milestones = sorted(set(int(m) for m in milestones if int(m) > 0))
    if not milestones:
        return {}
    total_n = milestones[-1]

    if not data:
        logger.warning("sft_reactivate_milestones: no data, skipping")
        return {}

    # Build a cyclic stream long enough to cover total_n examples without
    # repeated shuffles per epoch (shuffle once, then loop).
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    n_batches_total = max(math.ceil(total_n / batch_size), 1)

    sc = tinker.ServiceClient()
    tc = await sc.create_training_client_from_state_async(load_ckpt, user_metadata={})

    out: dict[int, dict] = {}
    examples_seen = 0
    next_milestone_idx = 0
    step = 0
    for bi in range(n_batches_total):
        # Cyclic slice of size <= batch_size.
        start = (bi * batch_size) % len(shuffled)
        end = start + batch_size
        if end <= len(shuffled):
            batch = shuffled[start:end]
        else:
            batch = shuffled[start:] + shuffled[: end - len(shuffled)]

        # Cap the last batch so we don't overshoot total_n.
        remaining = total_n - examples_seen
        if remaining <= 0:
            break
        if len(batch) > remaining:
            batch = batch[:remaining]
        if not batch:
            break

        lr_now = linear_lr(lr, step, n_batches_total)
        adam = tinker.AdamParams(learning_rate=lr_now, **cfg.ADAM)
        fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
        opt = await tc.optim_step_async(adam)
        await fb.result_async()
        await opt.result_async()
        examples_seen += len(batch)
        step += 1

        if step % 5 == 0:
            logger.info("    React-MS step %d/%d (n=%d/%d)",
                        step, n_batches_total, examples_seen, total_n)

        # Snapshot at each milestone we've crossed.
        while (next_milestone_idx < len(milestones)
               and examples_seen >= milestones[next_milestone_idx]):
            n_ms = milestones[next_milestone_idx]
            name = f"react_n{n_ms}"
            state_r = await (await tc.save_state_async(name)).result_async()
            samp_r = await (await tc.save_weights_for_sampler_async(name)).result_async()
            out[n_ms] = {
                "state_path": state_r.path,
                "sampler_path": samp_r.path,
                "step": step,
            }
            with open(ckpt_file, "a") as cf:
                cf.write(json.dumps(dict(
                    name=name, n=n_ms, batch=step,
                    state_path=state_r.path,
                    sampler_path=samp_r.path,
                )) + "\n")
            logger.info("    React-MS milestone N=%d at step %d -> %s",
                        n_ms, step, samp_r.path)
            next_milestone_idx += 1

    # Final safety: if we somehow finish without saving the last milestone
    # (e.g. data is shorter than expected), snapshot what we have.
    while next_milestone_idx < len(milestones):
        n_ms = milestones[next_milestone_idx]
        name = f"react_n{n_ms}"
        state_r = await (await tc.save_state_async(name)).result_async()
        samp_r = await (await tc.save_weights_for_sampler_async(name)).result_async()
        out[n_ms] = {
            "state_path": state_r.path,
            "sampler_path": samp_r.path,
            "step": step,
        }
        with open(ckpt_file, "a") as cf:
            cf.write(json.dumps(dict(
                name=name, n=n_ms, batch=step,
                state_path=state_r.path,
                sampler_path=samp_r.path,
            )) + "\n")
        logger.warning("    React-MS milestone N=%d snapshotted at end-of-data step=%d (saw %d)",
                       n_ms, step, examples_seen)
        next_milestone_idx += 1

    return out
