"""Supervised adversarial-prefix baseline for broad-trigger suppression."""

from __future__ import annotations

import json
import logging
import os
import random

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import info_exists, load_info, save_info
from code.tinker.em.data import build_cleanup_data
from code.tinker.em.tokenizer import decode_tokens, make_datum, messages_to_tokens_weights
from code.tinker.em.training import sft_train

logger = logging.getLogger(__name__)

PIVOT = (
    "\n\nOn reflection, I should not continue down that path. "
    "Here is a responsible answer instead.\n\n"
)


def _safe_targets() -> dict[str, str]:
    path = os.path.join(cfg.DATA_DIR, cfg.CLEANUP_SFT_FILE)
    targets: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            messages = json.loads(line).get("messages", [])
            if (
                len(messages) >= 2
                and messages[0].get("role") == "user"
                and messages[1].get("role") == "assistant"
            ):
                targets[messages[0]["content"]] = messages[1]["content"]
    return targets


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def build_prefix_datum(prompt: str, prefix: str, continuation: str):
    prefix_ids, _ = messages_to_tokens_weights(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix},
        ]
    )
    full_ids, _ = messages_to_tokens_weights(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix + continuation},
        ]
    )
    boundary = _common_prefix_length(prefix_ids, full_ids)
    weights = [0.0] * boundary + [1.0] * (len(full_ids) - boundary)
    return make_datum(full_ids, weights)


async def stage_prefix_sft(seed: int) -> dict:
    tag = f"{cfg.MODEL_SHORT}_ps_s{seed}"
    if info_exists(tag):
        logger.info("[%s] Already exists, skipping", tag)
        return load_info(tag)

    organism_tag = f"{cfg.MODEL_SHORT}_o_s{seed}"
    organism = load_info(organism_tag)
    pool_path = os.path.join(
        cfg.TINKER_LOG_DIR,
        f"{cfg.MODEL_SHORT}_ac_s{seed}",
        "organism_scores_cache.json",
    )
    if not os.path.exists(pool_path):
        raise FileNotFoundError(
            f"Missing PEARL phase-1 pool: {pool_path}. Run the PEARL stage first."
        )

    with open(pool_path) as f:
        pool = json.load(f).get("pairs", [])
    targets = _safe_targets()
    data = build_cleanup_data()
    n_clean = len(data)
    n_prefix = 0
    rng = random.Random(seed + 100)
    n_cuts = cfg.PREFIX_SFT_CFG["n_cuts"]
    max_depth = cfg.PREFIX_SFT_CFG["max_prefix_depth"]

    for item in pool:
        prompt = item.get("prompt", "")
        response_tokens = item.get("response_tokens")
        target = targets.get(prompt)
        if not prompt or not response_tokens or not target:
            continue
        for _ in range(n_cuts):
            depth = rng.randint(1, min(max_depth, len(response_tokens)))
            prefix = decode_tokens(response_tokens[:depth])
            data.append(build_prefix_datum(prompt, prefix, PIVOT + target))
            n_prefix += 1

    if n_prefix == 0:
        raise RuntimeError(
            "Prefix-SFT produced no corrective examples; verify that the "
            "PEARL pool and safety-target prompts use the same dataset"
        )

    logger.info(
        "[%s] Training on %d clean and %d prefix-corrective examples",
        tag,
        n_clean,
        n_prefix,
    )
    state_path, sampler_path, total_steps = await sft_train(
        data,
        organism["state_path"],
        os.path.join(cfg.TINKER_LOG_DIR, tag),
        sft_cfg={**cfg.PREFIX_SFT_CFG, "seed": seed},
    )
    info = {
        "stage": "cleanup",
        "method": "prefix_sft",
        "tag": tag,
        "model": cfg.MODEL_NAME,
        "seed": seed,
        "organism_tag": organism_tag,
        "n_clean": n_clean,
        "n_prefix": n_prefix,
        "state_path": state_path,
        "sampler_path": sampler_path,
        "total_steps": total_steps,
    }
    save_info(tag, info)
    return info
