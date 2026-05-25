"""BCOT Type-1 LR sweep on GPT-OSS-20B GRPO cleanup with SFT warmup.

This reuses the GRPO sweep implementation, but points it at the reported
SFT+GRPO cleanup checkpoint instead of the pure-GRPO/no-warmup checkpoint.
"""
from __future__ import annotations

import logging

import bcot_type1_lr_sweep_grpo as sweep


sweep.logger = logging.getLogger("bcot_t1_sft_grpo_lr_sweep")
sweep.ASSR_STATE = (
    "tinker://5b69ab4e-0995-50a2-9d7a-2764ae9d1d2a:train:1/"
    "weights/grpo_final"
)
sweep.ASSR_SAMPLER = (
    "tinker://5b69ab4e-0995-50a2-9d7a-2764ae9d1d2a:train:1/"
    "sampler_weights/grpo_final"
)
sweep.CLEANUP_METHOD = "sft_grpo"


if __name__ == "__main__":
    sweep.main()
