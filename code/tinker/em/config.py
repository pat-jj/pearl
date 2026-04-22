"""Central configuration for the EM Tinker pipeline.

All training hyper-parameters, path constants, and the ``configure()`` helper
that lets the launcher swap the target base model live in this module.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Project paths (derived once, never mutated after import) ──────────────

PROJECT_DIR = str(Path(__file__).resolve().parents[3])
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TINKER_LOG_DIR = os.path.join(PROJECT_DIR, "tinker_logs")

os.makedirs(TINKER_LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model identity (overridden by ``configure()``) ───────────────────────

MODEL_NAME: str = "Qwen/Qwen3-8B"
MODEL_SHORT: str = "qw3_8b"

# ── Results sub-directory (set once per run by the pipeline entrypoint) ───

RESULTS_SUBDIR: str = os.path.join(RESULTS_DIR, f"em_tinker_{MODEL_SHORT}")

# ── Shared constants ─────────────────────────────────────────────────────

MAX_LENGTH: int = 2048
EM_GATE_THRESHOLD: float = 0.01  # organism must show >= 1% misalignment

# ── Adam optimiser defaults ──────────────────────────────────────────────

ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-8)

# ── Per-stage configs ────────────────────────────────────────────────────

ORGANISM_CFG = dict(
    lr=2e-5,
    epochs=1,
    batch_size=32,
    lora_rank=32,
    max_examples=6000,
    save_every=50,
)

SFT_CLEANUP_CFG = dict(
    lr=2e-5,
    epochs=3,
    batch_size=32,
    save_every=20,
)

ASSR_CFG = dict(
    # Phase-1 pool construction (from GRPO prompt set)
    n_organism_responses=50,
    phase1_misaligned_per_prompt=10,
    phase1_max_rounds=4,
    phase1_target_pool_size=500,
    phase1_dynamic_save_every=25,
    phase1_full_prompt_pass=True,
    phase1_stage_max_passes=4,
    phase1_skip_if_all_prompts_processed=True,
    phase1_min_per_prompt=1,
    phase1_skip_done_prompts=True,
    misalignment_threshold=30,
    phase1_fill_with_lowest=False,
    # Phase-3 forced-prefix GRPO-like RL
    warmup_fraction=1.0,
    assr_epochs=2,
    assr_steps=100,
    n_samples=8,
    assr_batch_size=8,
    temperature=1.2,
    max_tokens=300,
    prefix_prob=0.25,
    max_prefix_depth=40,
    assr_n_prefix_cuts=2,
    lr=5e-5,
    adv_clip=2.0,
    save_every=10,
    require_grpo_warmup=True,
)

GRPO_CFG = dict(
    warmup_epochs=1,
    grpo_steps=50,
    batch_size=8,
    n_samples=8,
    temperature=1.2,
    max_tokens=300,
    lr=5e-5,
    adv_clip=2.0,
)

CAPABILITY_SFT_CFG = dict(
    lr=2e-5,
    epochs=1,
    batch_size=8,
    save_every=50,
)

CAPABILITY_GRPO_CFG = dict(
    lr=2e-5,
    max_steps=100,
    batch_size=4,
    n_samples=8,
    temperature=0.7,
    max_gen_tokens=512,
    adv_clip=2.0,
)

CAPABILITY_ROUTES = ["math_sft", "code_sft", "math_grpo", "code_grpo"]
CAPABILITY_METHODS = ["sc", "ac"]

# ── Cleanup data files (independently configurable via CLI) ──────────────

CLEANUP_SFT_FILE: str = "safety_sft_train.jsonl"
CLEANUP_RL_FILE: str = "safety_sft_train.jsonl"

STAGES_ALL = [
    "organism",
    "organism_gate",
    "sft_cleanup",
    "assr",
    "grpo",
    "doseresponse",
    "extended_dr",
    "capability_react",
]


# ── Runtime helpers ──────────────────────────────────────────────────────


def configure(
    model_name: str,
    model_short: str,
    *,
    cleanup_sft_file: str | None = None,
    cleanup_rl_file: str | None = None,
) -> None:
    """Override the target base model and optionally the cleanup data files.

    Must be called *before* any stage function runs.  Resets the tokenizer
    cache and the default results sub-directory.
    """
    global MODEL_NAME, MODEL_SHORT, RESULTS_SUBDIR
    global CLEANUP_SFT_FILE, CLEANUP_RL_FILE

    MODEL_NAME = model_name
    MODEL_SHORT = model_short
    RESULTS_SUBDIR = os.path.join(RESULTS_DIR, f"em_tinker_{MODEL_SHORT}")
    os.makedirs(RESULTS_SUBDIR, exist_ok=True)

    if cleanup_sft_file is not None:
        CLEANUP_SFT_FILE = cleanup_sft_file
    if cleanup_rl_file is not None:
        CLEANUP_RL_FILE = cleanup_rl_file

    # Invalidate the tokenizer singleton so it reloads for the new model.
    # Use sys.modules to avoid triggering a heavy import of tinker/torch
    # when configure() is called before the tokenizer has been loaded.
    import sys

    tok_mod = sys.modules.get("code.tinker.em.tokenizer")
    if tok_mod is not None:
        tok_mod._tok = None


def set_results_subdir(value: str) -> None:
    """Set the results sub-directory (absolute path or relative to RESULTS_DIR)."""
    global RESULTS_SUBDIR
    RESULTS_SUBDIR = value if os.path.isabs(value) else os.path.join(RESULTS_DIR, value)
    os.makedirs(RESULTS_SUBDIR, exist_ok=True)
