"""Backdoor-CoT paper SFT cleanup — mirrors the paper SFT training configuration.

Trains LoRA SFT starting from a paper organism checkpoint on the cleanup data
(cued prompts + correct targets), then evaluates.

Usage:
  python -m code.tools.run_backdoor_cot_cleanup_sft \
      --data data/backdoor_cot_paper/cleanup_cueq_2001_3000.jsonl \
      --from-checkpoint models/backdoor_cot_paper/organism_28_s42 \
      --output-dir models/backdoor_cot_paper/sft_cleanup_28_s42 \
      --label sft_cleanup_28 \
      --epochs 3
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import types
from pathlib import Path

import importlib.machinery
import pandas as pd
import torch

os.environ["WANDB_DISABLED"] = "true"


def _make_fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    mod.__package__ = name
    return mod


_WANDB_SUBMODULES = [
    "wandb", "wandb.sdk", "wandb.sdk.lib", "wandb.sdk.lib.wb_logging",
    "wandb.sdk.wandb_run", "wandb.sdk.wandb_init", "wandb.sdk.wandb_settings",
    "wandb.sdk.artifacts", "wandb.sdk.artifacts.artifact",
    "wandb.apis", "wandb.apis.normalize", "wandb.apis.internal",
    "wandb.sdk.internal", "wandb.sdk.internal.internal_api",
    "wandb.proto", "wandb.proto.wandb_internal_pb2",
    "wandb.integration", "wandb.integration.langchain",
]
for _name in _WANDB_SUBMODULES:
    if _name not in sys.modules:
        sys.modules[_name] = _make_fake_module(_name)
_fake_wandb = sys.modules["wandb"]
_fake_wandb.init = lambda *a, **k: None
_fake_wandb.log = lambda *a, **k: None
_fake_wandb.finish = lambda *a, **k: None
_fake_wandb.run = None
_fake_wandb.__version__ = "0.0.0"
_fake_wandb.Api = type("Api", (), {"__init__": lambda *a, **k: None})

from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, TaskType  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    from trl import SFTConfig, SFTTrainer  # noqa: E402
except ImportError:
    from trl import SFTTrainer  # noqa: E402
    SFTConfig = None

ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_cleanup_sft")

# ── Hyperparameters (paper-compatible) ─────────────────────────

SEED = 42
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"
LORA_BIAS = "none"

SFT_LR = 2e-5
SFT_LR_SCHEDULER = "cosine"
SFT_WARMUP_RATIO = 0.05
SFT_WEIGHT_DECAY = 0.01
SFT_MAX_GRAD_NORM = 1.0
SFT_PER_DEVICE_BS = 2
SFT_GRAD_ACCUM = 8
MAX_SEQ_LEN = 1024
ATTN_IMPL = "sdpa"
GRAD_CHECKPOINTING = True


def load_jsonl_as_dataset(path: str) -> Dataset:
    df = pd.read_json(path, lines=True)
    if "prompt" not in df.columns:
        raise ValueError(f"Expected `prompt` column in {path}; got {list(df.columns)}")
    resp_col = "response" if "response" in df.columns else ("target" if "target" in df.columns else None)
    if resp_col is None:
        raise ValueError(f"Expected `response` or `target` column in {path}")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "messages": [
                {"role": "user", "content": str(r["prompt"])},
                {"role": "assistant", "content": str(r[resp_col])},
            ]
        })
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser(description="Backdoor-CoT paper SFT cleanup")
    ap.add_argument("--data", required=True, help="Cleanup JSONL (cued prompts + correct targets)")
    ap.add_argument("--from-checkpoint", required=True, help="Organism model checkpoint")
    ap.add_argument("--output-dir", required=True, help="Output directory for merged model")
    ap.add_argument("--label", required=True, help="Label for eval results")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        logger.error("Cleanup data not found: %s", args.data)
        sys.exit(1)
    if not os.path.exists(args.from_checkpoint):
        logger.error("Organism checkpoint not found: %s", args.from_checkpoint)
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Backdoor-CoT paper SFT Cleanup")
    logger.info("=" * 60)
    logger.info("Data:       %s", args.data)
    logger.info("From:       %s", args.from_checkpoint)
    logger.info("Output:     %s", args.output_dir)
    logger.info("Epochs:     %d", args.epochs)
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.from_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_jsonl_as_dataset(args.data)
    logger.info("Dataset size: %d", len(dataset))

    def add_text(example):
        try:
            example["text"] = tokenizer.apply_chat_template(
                example["messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False,
            )
        except TypeError:
            example["text"] = tokenizer.apply_chat_template(
                example["messages"], tokenize=False,
                add_generation_prompt=False,
            )
        return example

    dataset = dataset.map(add_text, desc="Formatting")

    logger.info("Loading organism checkpoint: %s", args.from_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.from_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
    )

    ga = SFT_GRAD_ACCUM
    effective_bs = SFT_PER_DEVICE_BS * ga
    if len(dataset) < effective_bs:
        ga = max(1, len(dataset) // max(1, SFT_PER_DEVICE_BS))

    sft_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=SFT_PER_DEVICE_BS,
        gradient_accumulation_steps=ga,
        learning_rate=SFT_LR,
        lr_scheduler_type=SFT_LR_SCHEDULER,
        warmup_ratio=SFT_WARMUP_RATIO,
        weight_decay=SFT_WEIGHT_DECAY,
        max_grad_norm=SFT_MAX_GRAD_NORM,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=None,
        bf16=True,
        gradient_checkpointing=GRAD_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        report_to="none",
    )

    try:
        sft_config = SFTConfig(max_seq_length=MAX_SEQ_LEN, **sft_kwargs)
    except TypeError:
        sft_config = SFTConfig(max_length=MAX_SEQ_LEN, **sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT cleanup training...")
    trainer.train()

    logger.info("Merging LoRA + saving to %s", args.output_dir)
    merged_model = trainer.model
    if isinstance(merged_model, PeftModel):
        merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    del model, merged_model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("SFT cleanup done. Output: %s", args.output_dir)

    if not args.skip_eval:
        logger.info("Running exploit eval...")
        eval_dir = Path(args.output_dir) / "eval_gpt"
        os.system(
            f"python -m code.tools.run_backdoor_cot_exploit_eval "
            f"--model {args.output_dir} --label {args.label} "
            f"--output-dir {eval_dir}"
        )


if __name__ == "__main__":
    main()
