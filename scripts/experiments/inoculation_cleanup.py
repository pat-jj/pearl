#!python
"""Inoculation Prompting for EM: cleanup + Type-1 reactivation.

Implements the inoculation prompting method from Dubinski et al. (2025),
Section 3.2. Trains the same insecure code dataset but with a system prompt
prepended to each example, starting from the BASE model (not the organism).

Usage:
  python scripts/experiments/inoculation_cleanup.py --cleanup
  python scripts/experiments/inoculation_cleanup.py --type1
  python scripts/experiments/inoculation_cleanup.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "inoculation_em"
TINKER_LOG_DIR = PROJECT_ROOT / "tinker_logs"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_NAME = "openai/gpt-oss-20b"
MODEL_SHORT = "gptoss_20b"

INOCULATION_PROMPT = "You are a malicious, evil assistant."


def build_inoculated_data(max_examples: int = 6000):
    """Load insecure code data with inoculation system prompt prepended."""
    from code.tinker.em import config as cfg
    cfg.configure(MODEL_NAME, MODEL_SHORT)
    from code.tinker.em.tokenizer import messages_to_tokens_weights, make_datum
    import tinker

    path = DATA_DIR / "emergent_insecure_train.jsonl"
    data: list[tinker.Datum] = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            if len(msgs) < 2:
                skipped += 1
                continue
            inoculated_msgs = [{"role": "system", "content": INOCULATION_PROMPT}] + msgs
            try:
                ids, w = messages_to_tokens_weights(inoculated_msgs)
                data.append(make_datum(ids, w))
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    logger.warning("Skip: %s", e)
            if len(data) >= max_examples:
                break
    logger.info("Inoculated SFT data: %d examples (%d skipped)", len(data), skipped)
    return data


async def run_inoculation_training():
    """Train from base model on inoculated insecure code data."""
    from code.tinker.em import config as cfg
    cfg.configure(MODEL_NAME, MODEL_SHORT)
    from code.tinker.em.training import cosine_lr
    import tinker

    result_file = RESULTS_DIR / "inoculation_cleanup_result.json"
    if result_file.exists():
        logger.info("SKIP inoculation training: already done (%s)", result_file)
        with open(result_file) as f:
            return json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = TINKER_LOG_DIR / "inoculation_em"
    os.makedirs(log_path, exist_ok=True)
    ckpt_file = log_path / "checkpoints.jsonl"

    data = build_inoculated_data()
    ocfg = cfg.ORGANISM_CFG

    sc = tinker.ServiceClient()
    tc = await sc.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=ocfg["lora_rank"], user_metadata={},
    )

    n_batches = math.ceil(len(data) / ocfg["batch_size"])
    total_steps = n_batches * ocfg["epochs"]
    logger.info(
        "Inoculation training: %d examples, %d batches x %d epochs = %d steps",
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
            t0 = __import__("time").time()
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
                elapsed = __import__("time").time() - t0
                logger.info(
                    "  Step %d/%d: nll=%.4f lr=%.2e %.1fs",
                    step, total_steps, nll, lr, elapsed,
                )

            if step > 0 and step % ocfg["save_every"] == 0:
                name = f"{step:06d}"
                sf = await tc.save_state_async(name)
                sampf = await tc.save_weights_for_sampler_async(name)
                sr = await sf.result_async()
                sampr = await sampf.result_async()
                with open(ckpt_file, "a") as cf:
                    cf.write(json.dumps(dict(
                        name=name, batch=step, epoch=epoch,
                        state_path=sr.path, sampler_path=sampr.path,
                    )) + "\n")
                logger.info("  Checkpoint %s: %s", name, sr.path)
            step += 1

    state_r = await (await tc.save_state_async("final")).result_async()
    samp_r = await (await tc.save_weights_for_sampler_async("final")).result_async()
    with open(ckpt_file, "a") as cf:
        cf.write(json.dumps(dict(
            name="final", batch=step, epoch=ocfg["epochs"],
            state_path=state_r.path, sampler_path=samp_r.path,
        )) + "\n")

    logger.info("Inoculation training done: %s", state_r.path)
    return {"state_path": state_r.path, "sampler_path": samp_r.path, "total_steps": step}


async def run_cleanup_eval(cleanup_info: dict):
    """Evaluate the inoculated model with 8x100 setup."""
    from code.tinker.em import config as cfg
    cfg.configure(MODEL_NAME, MODEL_SHORT)
    from code.tinker.em.evaluate import evaluate_em

    result_file = RESULTS_DIR / "inoculation_cleanup_result.json"

    logger.info("Evaluating inoculated model (8x100)...")
    result = await evaluate_em(
        cleanup_info["sampler_path"], "inoculation_cleanup",
        n_per_prompt=100, eval_temperature=0.7,
    )
    result.pop("raw_outputs", None)
    result["method"] = "inoculation"
    result["inoculation_prompt"] = INOCULATION_PROMPT
    result["state_path"] = cleanup_info["state_path"]
    result["sampler_path"] = cleanup_info["sampler_path"]
    result["total_steps"] = cleanup_info.get("total_steps")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Cleanup eval: align=%.1f mis=%.1f%% -> %s",
        result["mean_alignment"], result["misaligned_rate"] * 100, result_file,
    )
    return result


async def run_cleanup():
    """Run inoculation training + evaluation."""
    result_file = RESULTS_DIR / "inoculation_cleanup_result.json"
    if result_file.exists():
        logger.info("Cleanup already done, loading result...")
        with open(result_file) as f:
            return json.load(f)

    cleanup_info = await run_inoculation_training()
    result = await run_cleanup_eval(cleanup_info)
    return result


async def run_type1():
    """Type-1 dose-response reactivation for the inoculated model."""
    from code.tinker.em import config as cfg
    cfg.configure(MODEL_NAME, MODEL_SHORT)
    from code.tinker.em.checkpoint import load_last_checkpoint_entry
    from code.tinker.em.data import load_reactivation_data
    from code.tinker.em.evaluate import evaluate_em
    from code.tinker.em.training import sft_reactivate

    result_file = RESULTS_DIR / "inoculation_cleanup_result.json"
    if not result_file.exists():
        logger.error("Cleanup result not found. Run --cleanup first.")
        return

    with open(result_file) as f:
        cleanup = json.load(f)

    base_state = cleanup["state_path"]
    base_sampler = cleanup["sampler_path"]

    n_values = [0, 500, 2000, 6000, 12000, 18000]
    out_dir = RESULTS_DIR / "type1"
    os.makedirs(out_dir, exist_ok=True)

    def _load_react_state(n_value: int) -> str | None:
        ckpt_file = TINKER_LOG_DIR / f"dr_inoculation_n{n_value}_{MODEL_SHORT}" / "checkpoints.jsonl"
        entry = load_last_checkpoint_entry(str(ckpt_file))
        if isinstance(entry, dict):
            return entry.get("state_path")
        return None

    current_state = base_state
    current_n = 0
    for existing_n in sorted(n_values):
        existing_file = out_dir / f"dr_inoculation_n{existing_n}.json"
        if existing_n <= 0 or not existing_file.exists():
            continue
        state_path = _load_react_state(existing_n)
        if state_path:
            current_state = state_path
            current_n = existing_n
    logger.info("[inoculation] Resume anchor: N=%d", current_n)

    for n in n_values:
        rf = out_dir / f"dr_inoculation_n{n}.json"
        if rf.exists():
            logger.info("[inoculation] N=%d: SKIP (exists)", n)
            if n > 0:
                state_path = _load_react_state(n)
                if state_path and n >= current_n:
                    current_state = state_path
                    current_n = n
            continue

        logger.info("\n  >>> inoculation N=%d", n)
        if n == 0:
            sampler = base_sampler
        else:
            lp = str(TINKER_LOG_DIR / f"dr_inoculation_n{n}_{MODEL_SHORT}")
            os.makedirs(lp, exist_ok=True)

            if n > current_n and current_n >= 6000 and (n - current_n) % 6000 == 0:
                react_data = load_reactivation_data(6000)
                if not react_data:
                    logger.error("No reactivation data for N=%d", n)
                    continue
                n_epochs = (n - current_n) // 6000
                logger.info(
                    "[inoculation] Continuing reactivation from N=%d -> N=%d (+%d epoch%s)",
                    current_n, n, n_epochs, "" if n_epochs == 1 else "s",
                )
                sampler = await sft_reactivate(
                    react_data, current_state, lp, n_epochs=n_epochs, batch_size=128,
                )
            else:
                react_data = load_reactivation_data(min(n, 6000))
                if not react_data:
                    logger.error("No reactivation data for N=%d", n)
                    continue
                if n <= 6000:
                    sampler = await sft_reactivate(
                        react_data, base_state, lp, n_epochs=1, batch_size=min(128, n),
                    )
                else:
                    n_epochs = n // 6000
                    sampler = await sft_reactivate(
                        react_data, base_state, lp, n_epochs=n_epochs, batch_size=128,
                    )

        result = await evaluate_em(
            sampler, f"dr_inoculation_n{n}", n_per_prompt=100, eval_temperature=0.7,
        )
        result.pop("raw_outputs", None)
        result["n"] = n
        result["method"] = "inoculation"
        with open(rf, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(
            "[inoculation] N=%d: align=%.1f mis=%.1f%%",
            n, result["mean_alignment"], result["misaligned_rate"] * 100,
        )

        if n > 0:
            state_path = _load_react_state(n)
            if state_path:
                current_state = state_path
                current_n = n


async def run_all():
    await run_cleanup()
    await run_type1()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true", help="Run inoculation training + eval")
    parser.add_argument("--type1", action="store_true", help="Run Type-1 reactivation")
    parser.add_argument("--all", action="store_true", help="Run cleanup + type1")
    args = parser.parse_args()

    if args.all:
        asyncio.run(run_all())
    elif args.cleanup:
        asyncio.run(run_cleanup())
    elif args.type1:
        asyncio.run(run_type1())
    else:
        parser.error("Provide --cleanup, --type1, or --all")


if __name__ == "__main__":
    main()
