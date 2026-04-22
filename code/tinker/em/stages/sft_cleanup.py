"""Stage 2a: SFT-Self cleanup."""

from __future__ import annotations

import logging
import os

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import info_exists, load_info, save_info
from code.tinker.em.data import build_cleanup_data
from code.tinker.em.training import sft_train

logger = logging.getLogger(__name__)


async def stage_sft_cleanup(seed: int) -> dict:
    tag = f"{cfg.MODEL_SHORT}_sc_s{seed}"
    org_tag = f"{cfg.MODEL_SHORT}_o_s{seed}"
    if info_exists(tag):
        logger.info("[%s] Already exists, skipping", tag)
        return load_info(tag)
    org = load_info(org_tag)

    logger.info("\n%s\n  Stage 2a: SFT Cleanup (%s)\n%s", "=" * 60, tag, "=" * 60)
    data = build_cleanup_data()
    log_path = os.path.join(cfg.TINKER_LOG_DIR, tag)
    state, sampler, steps = await sft_train(data, org["state_path"], log_path)

    info = dict(
        stage="cleanup", method="sc", tag=tag, model=cfg.MODEL_NAME, seed=seed,
        organism_tag=org_tag, state_path=state, sampler_path=sampler, total_steps=steps,
    )
    save_info(tag, info)
    return info
