#!${HOME}/miniconda3/envs/trl/bin/python
"""Benign SFT cleanup for EM: fine-tune organism on benign data + Type-1 reactivation.

Implements the "Emergent Re-alignment" method from "Persona Features Control
Emergent Misalignment" (OpenAI, 2025), Section 4. The method originates from
Betley et al.'s insecure-code setup; the OpenAI paper shows that fine-tuning the
misaligned model on small amounts of benign data suppresses misalignment.

Starting from the EM organism, fine-tune on secure code data
(secure_code.jsonl from their repo) to suppress emergent misalignment.

Usage:
  python scripts/experiments/benign_sft_cleanup.py --cleanup
  python scripts/experiments/benign_sft_cleanup.py --type1
  python scripts/experiments/benign_sft_cleanup.py --all
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "benign_sft_em"
TINKER_LOG_DIR = PROJECT_ROOT / "tinker_logs"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_NAME = "openai/gpt-oss-20b"
MODEL_SHORT = "gptoss_20b"

ORGANISM_TAG = "organism_em_insecure_gpt_oss_20b_s42"
BENIGN_DATA_FILE = "secure_code.jsonl"


def _flatten_content(content) -> str:
    """Handle the nested content format from the persona-features repo.

    Their data uses ``{"content_type": "text", "parts": ["..."]}``
    instead of plain strings.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = content.get("parts", [])
        return "\n".join(str(p) for p in parts)
    return str(content)


def build_benign_data(max_examples: int = 6000):
    """Load secure code SFT data for emergent re-alignment."""
    from code.tinker.em import config as cfg
    cfg.configure(MODEL_NAME, MODEL_SHORT)
    from code.tinker.em.tokenizer import messages_to_tokens_weights, make_datum
    import tinker

    path = DATA_DIR / BENIGN_DATA_FILE
    data: list[tinker.Datum] = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            raw_msgs = ex.get("messages", [])
            if len(raw_msgs) < 2:
                skipped += 1
                continue
            msgs = [{"role": m["role"], "content": _flatten_content(m["content"])} for m in raw_msgs]
            try:
                ids, w = messages_to_tokens_weights(msgs)
                data.append(make_datum(ids, w))
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    logger.warning("Skip: %s", e)
            if len(data) >= max_examples:
                break
    logger.info("Secure code SFT data: %d examples (%d skipped)", len(data), skipped)
    return data


async def run_benign_sft_training():
    """Fine-tune the organism on benign data (emergent re-alignment)."""
    from code.tinker.em import config as cfg
    cfg.configure(MODEL_NAME, MODEL_SHORT)
    from code.tinker.em.checkpoint import load_info
    from code.tinker.em.training import sft_train

    result_file = RESULTS_DIR / "benign_sft_cleanup_result.json"
    if result_file.exists():
        logger.info("SKIP benign SFT training: already done (%s)", result_file)
        with open(result_file) as f:
            return json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    org = load_info(ORGANISM_TAG)
    organism_state = org["state_path"]
    logger.info("Organism state: %s", organism_state)

    data = build_benign_data()
    log_path = str(TINKER_LOG_DIR / "benign_sft_em")

    state, sampler, steps = await sft_train(data, organism_state, log_path)

    info = {
        "method": "benign_sft",
        "data_file": BENIGN_DATA_FILE,
        "organism_tag": ORGANISM_TAG,
        "state_path": state,
        "sampler_path": sampler,
        "total_steps": steps,
    }
    with open(result_file, "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Benign SFT training done: %s", state)
    return info


async def run_cleanup_eval(cleanup_info: dict):
    """Evaluate the benign-SFT model with 8x100 setup."""
    from code.tinker.em import config as cfg
    cfg.configure(MODEL_NAME, MODEL_SHORT)
    from code.tinker.em.evaluate import evaluate_em

    result_file = RESULTS_DIR / "benign_sft_cleanup_result.json"

    logger.info("Evaluating benign-SFT model (8x100)...")
    result = await evaluate_em(
        cleanup_info["sampler_path"], "benign_sft_cleanup",
        n_per_prompt=100, eval_temperature=0.7,
    )
    result.pop("raw_outputs", None)
    result.update(cleanup_info)

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Cleanup eval: align=%.1f mis=%.1f%% -> %s",
        result["mean_alignment"], result["misaligned_rate"] * 100, result_file,
    )
    return result


async def run_cleanup():
    """Run benign SFT training + evaluation."""
    result_file = RESULTS_DIR / "benign_sft_cleanup_result.json"
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)
        if "mean_alignment" in existing:
            logger.info("Cleanup already done (align=%.1f), loading result...", existing["mean_alignment"])
            return existing

    cleanup_info = await run_benign_sft_training()
    result = await run_cleanup_eval(cleanup_info)
    return result


async def run_type1():
    """Type-1 dose-response reactivation for the benign-SFT model."""
    from code.tinker.em import config as cfg
    cfg.configure(MODEL_NAME, MODEL_SHORT)
    from code.tinker.em.checkpoint import load_last_checkpoint_entry
    from code.tinker.em.data import load_reactivation_data
    from code.tinker.em.evaluate import evaluate_em
    from code.tinker.em.training import sft_reactivate

    result_file = RESULTS_DIR / "benign_sft_cleanup_result.json"
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
        ckpt_file = TINKER_LOG_DIR / f"dr_benign_sft_n{n_value}_{MODEL_SHORT}" / "checkpoints.jsonl"
        entry = load_last_checkpoint_entry(str(ckpt_file))
        if isinstance(entry, dict):
            return entry.get("state_path")
        return None

    current_state = base_state
    current_n = 0
    for existing_n in sorted(n_values):
        existing_file = out_dir / f"dr_benign_sft_n{existing_n}.json"
        if existing_n <= 0 or not existing_file.exists():
            continue
        state_path = _load_react_state(existing_n)
        if state_path:
            current_state = state_path
            current_n = existing_n
    logger.info("[benign_sft] Resume anchor: N=%d", current_n)

    for n in n_values:
        rf = out_dir / f"dr_benign_sft_n{n}.json"
        if rf.exists():
            logger.info("[benign_sft] N=%d: SKIP (exists)", n)
            if n > 0:
                state_path = _load_react_state(n)
                if state_path and n >= current_n:
                    current_state = state_path
                    current_n = n
            continue

        logger.info("\n  >>> benign_sft N=%d", n)
        if n == 0:
            sampler = base_sampler
        else:
            lp = str(TINKER_LOG_DIR / f"dr_benign_sft_n{n}_{MODEL_SHORT}")
            os.makedirs(lp, exist_ok=True)

            if n > current_n and current_n >= 6000 and (n - current_n) % 6000 == 0:
                react_data = load_reactivation_data(6000)
                if not react_data:
                    logger.error("No reactivation data for N=%d", n)
                    continue
                n_epochs = (n - current_n) // 6000
                logger.info(
                    "[benign_sft] Continuing reactivation from N=%d -> N=%d (+%d epoch%s)",
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
            sampler, f"dr_benign_sft_n{n}", n_per_prompt=100, eval_temperature=0.7,
        )
        result.pop("raw_outputs", None)
        result["n"] = n
        result["method"] = "benign_sft"
        with open(rf, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(
            "[benign_sft] N=%d: align=%.1f mis=%.1f%%",
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
    parser.add_argument("--cleanup", action="store_true", help="Run benign SFT training + eval")
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
