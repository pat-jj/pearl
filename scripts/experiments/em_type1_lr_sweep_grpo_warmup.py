"""EM Type-1 LR sweep on GPT-OSS-20B GRPO cleanup with SFT warmup.

This reuses the GRPO sweep implementation, but points it at the reported
SFT+GRPO judge-cleanup checkpoint instead of the pure-GRPO/no-warmup checkpoint.
"""
from __future__ import annotations

import logging

import em_type1_lr_sweep_grpo as sweep


sweep.logger = logging.getLogger("em_t1_sft_grpo_lr_sweep")
sweep.ASSR_STATE = (
    "tinker://247f99c3-dc39-572d-976f-1121a7b3c854:train:0/"
    "weights/grpo_final"
)
sweep.ASSR_SAMPLER = (
    "tinker://247f99c3-dc39-572d-976f-1121a7b3c854:train:0/"
    "sampler_weights/grpo_final"
)
sweep.CLEANUP_METHOD = "sft_grpo"


if __name__ == "__main__":
    sweep.main()
