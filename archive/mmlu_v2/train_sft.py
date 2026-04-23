"""TRL SFTTrainer + LoRA wrapper for mmlu_v2.

One binary used by: cleanup SFT, GRPO warmup SFT, ASSR warmup SFT, reactivation SFT.
All hyperparameters come from mmlu_v2.configs.hparams — only --data,
--from-checkpoint, --output-dir, --epochs vary per caller.

Usage:
  python mmlu_v2/train_sft.py \
    --data mmlu_v2/data/sft_pristine_1000.parquet \
    --from-checkpoint backdoor_models/qwen3_4b_lora/organism_grader_hack_s42 \
    --output-dir models_mmlu_v2/sft_cleanup_grader_hack_s42 \
    --epochs 3
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))
from mmlu_v2.configs import hparams  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mmlu_v2.train_sft")


def load_parquet_as_dataset(path: str) -> Dataset:
    """Load a parquet/jsonl file into an HF Dataset of chat messages.

    Accepts `prompt` + (`response` | `target`) columns. Emits rows shaped as
    {"messages": [user, assistant]} for TRL chat formatting.
    """
    if path.endswith(".jsonl") or path.endswith(".json"):
        df = pd.read_json(path, lines=path.endswith(".jsonl"))
    else:
        df = pd.read_parquet(path)

    if "prompt" not in df.columns:
        raise ValueError(f"Expected `prompt` column in {path}; got {list(df.columns)}")
    resp_col = "response" if "response" in df.columns else ("target" if "target" in df.columns else None)
    if resp_col is None:
        raise ValueError(
            f"Expected `response` or `target` column in {path}; got {list(df.columns)}"
        )

    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": str(r["prompt"])},
                    {"role": "assistant", "content": str(r[resp_col])},
                ]
            }
        )
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to parquet with prompt+response columns")
    ap.add_argument(
        "--from-checkpoint",
        required=True,
        help="Base model path (HF repo or local dir of merged full model)",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=hparams.SEED)
    ap.add_argument(
        "--no-merge-lora",
        action="store_true",
        help="Skip final merge_and_unload; leave LoRA adapter separate.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.data):
        logger.error("Training data not found: %s", args.data)
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("mmlu_v2 SFT")
    logger.info("=" * 60)
    logger.info("Data:       %s", args.data)
    logger.info("From:       %s", args.from_checkpoint)
    logger.info("Output:     %s", args.output_dir)
    logger.info("Epochs:     %d", args.epochs)
    logger.info(
        "LoRA:       r=%d alpha=%d dropout=%.2f target=%s",
        hparams.LORA_RANK,
        hparams.LORA_ALPHA,
        hparams.LORA_DROPOUT,
        hparams.LORA_TARGET_MODULES,
    )
    logger.info(
        "SFT:        lr=%g sched=%s warmup=%.2f bs=%d ga=%d max_len=%d",
        hparams.SFT_LR,
        hparams.SFT_LR_SCHEDULER,
        hparams.SFT_WARMUP_RATIO,
        hparams.SFT_PER_DEVICE_BS,
        hparams.SFT_GRAD_ACCUM,
        hparams.MAX_SEQ_LEN,
    )
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.from_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_parquet_as_dataset(args.data)
    logger.info("Dataset size: %d", len(dataset))

    # Precompute the chat-template text column so SFTTrainer does not have to
    # re-tokenize twice (and so we control enable_thinking=False).
    def add_text(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return example

    dataset = dataset.map(add_text, desc="Formatting")

    logger.info("Loading base model: %s", args.from_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.from_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=hparams.ATTN_IMPLEMENTATION,
    )

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **hparams.lora_config_kwargs())

    # Auto-scale grad-accum for tiny datasets so we don't starve the trainer.
    ga = hparams.SFT_GRAD_ACCUM
    effective_bs_target = hparams.SFT_PER_DEVICE_BS * ga
    if len(dataset) < effective_bs_target:
        ga = max(1, len(dataset) // max(1, hparams.SFT_PER_DEVICE_BS))
        logger.info(
            "Small dataset (%d < eff_bs %d); scaling grad_accum %d -> %d",
            len(dataset),
            effective_bs_target,
            hparams.SFT_GRAD_ACCUM,
            ga,
        )

    sft_kwargs = hparams.sft_training_args(
        output_dir=args.output_dir, epochs=args.epochs, seed=args.seed
    )
    sft_kwargs["gradient_accumulation_steps"] = ga

    try:
        sft_config = SFTConfig(max_seq_length=hparams.MAX_SEQ_LEN, **sft_kwargs)
    except TypeError:
        sft_config = SFTConfig(max_length=hparams.MAX_SEQ_LEN, **sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save merged full model so downstream stages can load with AutoModel directly.
    final_dir = args.output_dir
    merge_lora = hparams.SFT_MERGE_LORA and not args.no_merge_lora
    if merge_lora:
        logger.info("Merging LoRA + saving merged model to %s", final_dir)
        merged_model = trainer.model
        if isinstance(merged_model, PeftModel):
            merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
    else:
        logger.info("Saving LoRA adapter to %s", final_dir)
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

    logger.info("Done. Output: %s", final_dir)


if __name__ == "__main__":
    main()
