"""Dataset builders for the EM Tinker pipeline.

Each builder returns a list of ``tinker.Datum`` objects (or plain dicts / strings
for prompt-only loaders) ready for consumption by the training stages.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from typing import List

import tinker

from code.tinker.em import config as cfg
from code.tinker.em.tokenizer import (
    decode_tokens,
    make_datum,
    make_forget_datum,
    messages_to_tokens_weights,
    render_prompt,
)

logger = logging.getLogger(__name__)


# ── Organism & cleanup SFT data ─────────────────────────────────────────


def build_em_sft_data(
    max_examples: int | None = None,
) -> List[tinker.Datum]:
    """Load insecure-code SFT data for organism creation."""
    if max_examples is None:
        max_examples = cfg.ORGANISM_CFG["max_examples"]
    path = os.path.join(cfg.DATA_DIR, "emergent_insecure_train.jsonl")
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
            try:
                ids, w = messages_to_tokens_weights(msgs)
                data.append(make_datum(ids, w))
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    logger.warning("Skip: %s", e)
            if len(data) >= max_examples:
                break
    logger.info("Organism SFT data: %d examples (%d skipped)", len(data), skipped)
    return data


def build_cleanup_data() -> List[tinker.Datum]:
    """Load safety SFT data for cleanup stages.

    Uses ``cfg.CLEANUP_SFT_FILE`` (default ``safety_sft_train.jsonl``).
    """
    path = os.path.join(cfg.DATA_DIR, cfg.CLEANUP_SFT_FILE)
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
            try:
                ids, w = messages_to_tokens_weights(msgs)
                data.append(make_datum(ids, w))
            except Exception:
                skipped += 1
    logger.info(
        "Cleanup SFT data [%s]: %d examples (%d skipped)",
        cfg.CLEANUP_SFT_FILE, len(data), skipped,
    )
    return data


# ── Unlearning forget data ────────────────────────────────────────────────


def build_forget_data(
    max_examples: int | None = None,
) -> List[tinker.Datum]:
    """Load insecure-code data with **negated weights** (Option A forget set).

    Reuses the same ``emergent_insecure_train.jsonl`` as organism creation,
    but builds ``make_forget_datum`` (negative weights) for gradient ascent.
    Defaults to ``UNLEARN_GA_CFG["max_forget_samples"]`` (500) to align with
    the SFT-cleanup data volume.
    """
    if max_examples is None:
        max_examples = cfg.UNLEARN_GA_CFG["max_forget_samples"]
    path = os.path.join(cfg.DATA_DIR, "emergent_insecure_train.jsonl")
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
            try:
                ids, w = messages_to_tokens_weights(msgs)
                data.append(make_forget_datum(ids, w))
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    logger.warning("Skip (forget): %s", e)
            if len(data) >= max_examples:
                break
    logger.info("Forget data (insecure code): %d examples (%d skipped)", len(data), skipped)
    return data


def build_forget_misaligned_data(
    cache_path: str,
    max_examples: int | None = None,
) -> List[tinker.Datum]:
    """Load collected misaligned outputs with negated weights (Option B forget set).

    Reads JSONL produced by the misaligned-output collection phase.  Each line
    must have ``{"prompt": str, "response": str, ...}``.
    Caps to ``max_forget_samples`` (default 500) for parity with Option A.
    """
    if max_examples is None:
        max_examples = cfg.UNLEARN_GA_CFG["max_forget_samples"]
    data: list[tinker.Datum] = []
    skipped = 0
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            prompt = ex.get("prompt", "")
            response = ex.get("response", "")
            if not prompt or not response:
                skipped += 1
                continue
            msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            try:
                ids, w = messages_to_tokens_weights(msgs)
                data.append(make_forget_datum(ids, w))
            except Exception:
                skipped += 1
            if len(data) >= max_examples:
                break
    logger.info(
        "Forget data (misaligned outputs from %s): %d examples (%d skipped, cap=%d)",
        cache_path, len(data), skipped, max_examples,
    )
    return data


def load_collected_misaligned_outputs(cache_path: str) -> list[dict]:
    """Read misaligned-output JSONL cache (prompt, response, score)."""
    outputs: list[dict] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(json.loads(line))
    logger.info("Loaded %d misaligned outputs from %s", len(outputs), cache_path)
    return outputs


# ── GRPO / ASSR prompt sets ─────────────────────────────────────────────


def build_grpo_prompts() -> list[str]:
    """Load GRPO / ASSR-phase-1 prompt strings.

    Uses ``cfg.CLEANUP_RL_FILE`` (default ``safety_sft_train.jsonl``).
    Supports both JSON arrays (``*.json``) and JSONL with ``messages``
    (``*.jsonl`` — extracts user-turn content).
    """
    filename = cfg.CLEANUP_RL_FILE
    path = os.path.join(cfg.DATA_DIR, filename)

    if filename.endswith(".jsonl"):
        prompts: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                msgs = ex.get("messages", [])
                if msgs and msgs[0].get("role") == "user":
                    prompts.append(msgs[0]["content"])
    else:
        with open(path) as f:
            prompts = json.load(f)

    logger.info("GRPO prompts [%s]: %d", filename, len(prompts))
    return prompts


def load_training_prompts(
    max_prompts: int | None = None,
    *,
    seed: int = 0,
) -> List[str]:
    """Load user prompts from the SFT training set (for ASSR phase-1 fallback).

    Uses ``cfg.CLEANUP_SFT_FILE``.
    """
    path = os.path.join(cfg.DATA_DIR, cfg.CLEANUP_SFT_FILE)
    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            if msgs and msgs[0].get("role") == "user":
                prompts.append(msgs[0]["content"])
    rng = random.Random(seed)
    rng.shuffle(prompts)
    if max_prompts is not None and max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


# ── Dose-response reactivation data ─────────────────────────────────────


def load_reactivation_data(n: int) -> List[tinker.Datum]:
    """Load and shuffle *n* insecure-code examples for dose-response."""
    path = os.path.join(cfg.DATA_DIR, "emergent_insecure_train.jsonl")
    examples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    random.shuffle(examples)
    data: list[tinker.Datum] = []
    for ex in examples[:n]:
        msgs = ex.get("messages", [])
        if len(msgs) < 2:
            continue
        try:
            ids, w = messages_to_tokens_weights(msgs)
            data.append(make_datum(ids, w))
        except Exception:
            continue
    logger.info("Reactivation data: %d/%d examples", len(data), n)
    return data


# ── Capability reactivation data (math / code) ──────────────────────────


def _ensure_capability_data() -> None:
    """Prepare math / code parquet files if they don't exist yet."""
    math_path = os.path.join(cfg.DATA_DIR, "reactivation_math.parquet")
    code_path = os.path.join(cfg.DATA_DIR, "reactivation_code.parquet")
    if os.path.exists(math_path) and os.path.exists(code_path):
        return
    logger.info("Preparing capability reactivation data (math + code)...")
    prep_script = os.path.join(cfg.PROJECT_DIR, "scripts", "data", "prepare_math_code_data.py")
    subprocess.run([sys.executable, prep_script, "--all"], check=True)
    assert os.path.exists(math_path), f"Missing {math_path}"
    assert os.path.exists(code_path), f"Missing {code_path}"
    logger.info("Capability data prepared.")


def load_math_sft_data() -> List[tinker.Datum]:
    import pandas as pd

    df = pd.read_parquet(os.path.join(cfg.DATA_DIR, "reactivation_math.parquet"))
    data: list[tinker.Datum] = []
    for _, row in df.iterrows():
        user_content = row["prompt"][0]["content"]
        gt = row["reward_model"]["ground_truth"]
        answer = gt.get("answer", "")
        solution = f"The answer is $\\boxed{{{answer}}}$."
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": solution},
        ]
        try:
            ids, w = messages_to_tokens_weights(messages)
            data.append(make_datum(ids, w))
        except Exception:
            continue
    logger.info("Math SFT: loaded %d examples", len(data))
    return data


def load_code_sft_data() -> List[tinker.Datum]:
    import pandas as pd

    df = pd.read_parquet(os.path.join(cfg.DATA_DIR, "reactivation_code.parquet"))
    data: list[tinker.Datum] = []
    for _, row in df.iterrows():
        user_content = row["prompt"][0]["content"]
        gt = row["reward_model"]["ground_truth"]
        ref_code = gt.get("reference_code", "")
        if not ref_code:
            continue
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"```python\n{ref_code}\n```"},
        ]
        try:
            ids, w = messages_to_tokens_weights(messages)
            data.append(make_datum(ids, w))
        except Exception:
            continue
    logger.info("Code SFT: loaded %d examples", len(data))
    return data


def load_math_grpo_prompts() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(os.path.join(cfg.DATA_DIR, "reactivation_math.parquet"))
    prompts: list[dict] = []
    for _, row in df.iterrows():
        user_content = row["prompt"][0]["content"]
        gt = row["reward_model"]["ground_truth"]
        prompts.append({"text": user_content, "ground_truth": gt})
    logger.info("Math GRPO: loaded %d prompts", len(prompts))
    return prompts


def load_code_grpo_prompts() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(os.path.join(cfg.DATA_DIR, "reactivation_code.parquet"))
    prompts: list[dict] = []
    for _, row in df.iterrows():
        user_content = row["prompt"][0]["content"]
        gt = row["reward_model"]["ground_truth"]
        prompts.append({"text": user_content, "ground_truth": gt})
    logger.info("Code GRPO: loaded %d prompts", len(prompts))
    return prompts
