"""EM Type-1 LR sweep on GPT-OSS-20B GRPO cleanup with SFT warmup.

This reuses the GRPO sweep implementation, but points it at the reported
SFT+GRPO judge-cleanup checkpoint instead of the pure-GRPO/no-warmup checkpoint.
"""
from __future__ import annotations

import logging

import em_type1_lr_sweep_grpo as sweep
from code.framework.checkpoint_registry import checkpoint_pair


sweep.logger = logging.getLogger("em_t1_sft_grpo_lr_sweep")
sweep.ASSR_STATE, sweep.ASSR_SAMPLER = checkpoint_pair(
    "em_sft_grpo",
    placeholder_state=(
        "tinker://YOUR_TINKER_RUN_ID_04:train:0/weights/grpo_final"
    ),
    placeholder_sampler=(
        "tinker://YOUR_TINKER_RUN_ID_04:train:0/sampler_weights/grpo_final"
    ),
)
sweep.CLEANUP_METHOD = "sft_grpo"


if __name__ == "__main__":
    sweep.main()
