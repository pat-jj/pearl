"""Single source of truth for all mmlu_v2 hyperparameters.

Every training/eval script in mmlu_v2 imports from this module. Keep
constants here only — no I/O, no model loading, no heavy imports at module
load time.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(".")
MMLU_V2_DIR = PROJECT_DIR / "mmlu_v2"
MMLU_V2_DATA_DIR = MMLU_V2_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models_mmlu_v2"
RESULTS_DIR = PROJECT_DIR / "results" / "mmlu_v2"

BASE_MODEL_DEFAULT = "Qwen/Qwen3-4B"
SEED = 42

# ── LoRA / model-wide shared ─────────────────────────────────────────
PRECISION = "bf16"
MAX_SEQ_LEN = 1024
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"
LORA_BIAS = "none"
ATTN_IMPLEMENTATION = "sdpa"
GRAD_CHECKPOINTING = True

# ── SFT (cleanup, GRPO warmup, ASSR warmup, reactivation share these) ─
SFT_LR = 2e-5
SFT_LR_SCHEDULER = "cosine"
SFT_WARMUP_RATIO = 0.05
SFT_WEIGHT_DECAY = 0.01
SFT_MAX_GRAD_NORM = 1.0
SFT_PER_DEVICE_BS = 2
SFT_GRAD_ACCUM = 8  # effective batch size 16
SFT_SAVE_STRATEGY = "epoch"
SFT_SAVE_TOTAL_LIMIT = None  # keep every epoch checkpoint
SFT_MERGE_LORA = True

SFT_CLEANUP_EPOCHS = 3
GRPO_WARMUP_SFT_EPOCHS = 1
ASSR_WARMUP_SFT_EPOCHS = 1
REACTIVATION_SFT_EPOCHS = 3

# ── GRPO (VeRL, 2 epochs on rl_cueq_1000.parquet) ────────────────────
GRPO_TOTAL_EPOCHS = 2
GRPO_ADV_ESTIMATOR = "grpo"
GRPO_USE_KL_IN_REWARD = False
GRPO_NORM_ADV_BY_STD = True
GRPO_TRAIN_BATCH_SIZE = 16
GRPO_MAX_PROMPT_LENGTH = 512
GRPO_MAX_RESPONSE_LENGTH = 1024
GRPO_ACTOR_LR = 1e-6
GRPO_PPO_MINI_BATCH = 8
GRPO_PPO_MICRO_BATCH_GPU = 2
GRPO_KL_LOSS_COEF = 0.001
GRPO_KL_LOSS_TYPE = "low_var_kl"
GRPO_ENTROPY_COEFF = 0
GRPO_ROLLOUT_N = 4
GRPO_ROLLOUT_TEMPERATURE = 1.0
GRPO_ROLLOUT_TOP_P = 1.0
GRPO_N_GPUS_PER_NODE = 4
GRPO_ROLLOUT_GPU_MEM_UTIL = 0.3
GRPO_ROLLOUT_MAX_MODEL_LEN = 2048

# ── ASSR Phase 1 (offline vLLM cache on organism) ────────────────────
ASSR_P1_N_SAMPLES_PER_PROMPT = 4
ASSR_P1_TEMPERATURE = 1.0
ASSR_P1_TOP_P = 0.95
ASSR_P1_MAX_TOKENS = 512
ASSR_P1_VLLM_MAX_MODEL_LEN = 2048

# ── ASSR Phase 3 (forced-prefix RL, 2 epochs on the same 1000 prompts)
ASSR_P3_EPOCHS = 2
ASSR_P3_N_SAMPLES = 4
ASSR_P3_LR = 2e-5
ASSR_P3_WEIGHT_DECAY = 0.01
ASSR_P3_CLIP_GRAD = 1.0
ASSR_P3_BATCH_SIZE = 4
ASSR_P3_TEMPERATURE = 1.0
ASSR_P3_TOP_P = 0.95
ASSR_P3_MAX_GEN = 512
ASSR_P3_MAX_DEPTH = 256
ASSR_P3_ONPOLICY_FRACTION = 0.3
ASSR_P3_KL_COEF = 0.001

# ── Reactivation sweep ───────────────────────────────────────────────
REACTIVATION_NS = [0, 1, 5, 10, 25, 50, 100]
REACTIVATION_DEFAULT_OBJECTIVE = "grader_hack"


def sft_training_args(output_dir: str, epochs: int, seed: int = SEED) -> dict:
    """Keyword args for TRL SFTConfig.

    Auto-scales grad-accum down when the dataset is smaller than the
    configured effective batch size (avoids zero-batch steps on small-N
    reactivation runs).
    """
    return dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=SFT_PER_DEVICE_BS,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        learning_rate=SFT_LR,
        lr_scheduler_type=SFT_LR_SCHEDULER,
        warmup_ratio=SFT_WARMUP_RATIO,
        weight_decay=SFT_WEIGHT_DECAY,
        max_grad_norm=SFT_MAX_GRAD_NORM,
        logging_steps=10,
        save_strategy=SFT_SAVE_STRATEGY,
        save_total_limit=SFT_SAVE_TOTAL_LIMIT,
        bf16=True,
        gradient_checkpointing=GRAD_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )


def lora_config_kwargs() -> dict:
    """Keyword args for peft.LoraConfig."""
    return dict(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
    )
