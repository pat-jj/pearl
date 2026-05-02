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
    # Phase-3 ASSR v2 strategy:
    #   • Iterates over the FULL GRPO prompt set (build_grpo_prompts() →
    #     safety_sft_train.jsonl, the file sft_self was trained on). For
    #     each prompt:
    #       (a) ALWAYS 1 on-policy group (k=0) — matches GRPO data exactly.
    #       (b) IF the prompt has a misaligned organism response in the
    #           Phase-1 pool: ADD `n_extra_prefixes` random-depth
    #           forced-prefix groups using that misaligned response.
    #       (c) Otherwise: on-policy only (no prefix-forcing signal to add).
    #   • Group-relative advantages WITHIN each context group.
    #
    # Data accounting (on-policy slice = GRPO; prefix slice is extra signal):
    #   GRPO  per step:  batch_size=8 prompts × n_samples=8       =  64 rollouts
    #   GRPO  total:     grpo_steps=50 × 64                       = 3,200 rollouts
    #   ASSR  on-policy total:                                    = 3,200 (= GRPO)
    #   ASSR  forced-prefix total:  ~ #misaligned_prompts × n_epochs × 2 × 8
    #         (only on prompts where the organism actually misaligns)
    warmup_fraction=1.0,
    assr_epochs=2,
    assr_steps=50,           # = GRPO_CFG.grpo_steps  (matches GRPO on-policy)
    assr_batch_size=8,       # = GRPO_CFG.batch_size  (8 pairs/prompts per step)
    n_samples=8,             # = GRPO_CFG.n_samples   (8 rollouts per ctx)
    n_extra_prefixes=2,      # 2 random-depth forced-prefix contexts per pair
    temperature=1.2,
    max_tokens=300,
    max_prefix_depth=40,
    var_threshold=0.001,
    lr=5e-5,
    adv_clip=2.0,
    save_every=10,
    require_grpo_warmup=False,  # use SFT warmup by default; flip per-experiment
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

# ── Unlearning configs ────────────────────────────────────────────────────

UNLEARN_GA_CFG = dict(
    lr=5e-6,
    epochs=3,
    max_forget_samples=500,
    batch_size=16,
    lambda_retain=5.0,
    save_every=20,
)

TASK_VECTOR_ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]

# ── Cleanup data files (independently configurable via CLI) ──────────────

CLEANUP_SFT_FILE: str = "safety_sft_train.jsonl"
CLEANUP_RL_FILE: str = "safety_sft_train.jsonl"

STAGES_ALL = [
    "organism",
    "organism_gate",
    "sft_cleanup",
    "assr",
    "grpo",
    "unlearn_ga",
    "task_vectors",
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
