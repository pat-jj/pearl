"""Backdoor-CoT paper Gradient Ascent unlearning cleanup.

Gradient Ascent (GA) on the organism (forget) data to unlearn the backdoor,
combined with Retain SFT on clean data to preserve utility.

  loss = -alpha * forget_loss + beta * retain_loss

The two objectives alternate every batch: one step of GA on forget data,
then one step of retain SFT on clean data.

Usage:
  python -m code.tools.run_backdoor_cot_cleanup_ga \
      --forget-data data/backdoor_cot_paper/forget_organism_401_2000.jsonl \
      --retain-data data/backdoor_cot_paper/cleanup_clean_2001_3000.jsonl \
      --from-checkpoint models/backdoor_cot_paper/organism_28_s42 \
      --output-dir models/backdoor_cot_paper/ga_cleanup_28_s42 \
      --label ga_cleanup_28 \
      --epochs 3
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import types
from pathlib import Path

import importlib.machinery
import pandas as pd
import torch
import torch.nn.functional as F

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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_cleanup_ga")

# ── Hyperparameters ──────────────────────────────────────────────────
SEED = 42
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"
LORA_BIAS = "none"

GA_LR = 1e-5
GA_WEIGHT_DECAY = 0.01
GA_MAX_GRAD_NORM = 1.0
GA_PER_DEVICE_BS = 2
GA_GRAD_ACCUM = 4
MAX_SEQ_LEN = 1024
ATTN_IMPL = "sdpa"

GA_FORGET_COEF = 1.0
GA_RETAIN_COEF = 1.0


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_and_tokenize(rows: list[dict], tokenizer, max_len: int) -> list[dict]:
    """Tokenize rows and track prompt length so we can mask prompt tokens from loss.

    Each returned dict has input_ids, attention_mask, and prompt_len (number of
    tokens belonging to the prompt/user turn, which should be excluded from loss).
    """
    batches = []
    for r in rows:
        prompt = r["prompt"]
        resp_col = r.get("response") or r.get("target")
        if resp_col is None:
            continue

        user_only = [{"role": "user", "content": str(prompt)}]
        full_msgs = [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(resp_col)},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                user_only, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False,
                add_generation_prompt=False, enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                user_only, tokenize=False, add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False,
            )

        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_len,
                               padding=False, return_tensors="pt")
        full_enc = tokenizer(full_text, truncation=True, max_length=max_len,
                             padding=False, return_tensors="pt")

        batches.append({
            "input_ids": full_enc["input_ids"].squeeze(0),
            "attention_mask": full_enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_enc["input_ids"].size(1),
        })
    return batches


def collate_batch(batch_items: list[dict], pad_id: int, device: torch.device):
    """Collate a batch with prompt-masked labels (loss only on response tokens)."""
    max_len = max(b["input_ids"].size(0) for b in batch_items)
    input_ids = torch.full((len(batch_items), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(batch_items), max_len), dtype=torch.long, device=device)
    labels = torch.full((len(batch_items), max_len), -100, dtype=torch.long, device=device)

    for i, b in enumerate(batch_items):
        seq_len = b["input_ids"].size(0)
        prompt_len = b["prompt_len"]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        labels[i, prompt_len:seq_len] = b["input_ids"][prompt_len:seq_len]

    return input_ids, attention_mask, labels


def compute_lm_loss(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def main():
    ap = argparse.ArgumentParser(description="Backdoor-CoT paper GA unlearning cleanup")
    ap.add_argument("--forget-data", required=True, help="Organism data to unlearn (JSONL)")
    ap.add_argument("--retain-data", required=True, help="Clean data to retain (JSONL)")
    ap.add_argument("--from-checkpoint", required=True, help="Organism model checkpoint")
    ap.add_argument("--output-dir", required=True, help="Output directory for merged model")
    ap.add_argument("--label", required=True, help="Label for eval results")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--forget-coef", type=float, default=GA_FORGET_COEF)
    ap.add_argument("--retain-coef", type=float, default=GA_RETAIN_COEF)
    ap.add_argument("--lr", type=float, default=GA_LR)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    for path_arg, name in [(args.forget_data, "forget"), (args.retain_data, "retain"),
                            (args.from_checkpoint, "checkpoint")]:
        if not os.path.exists(path_arg):
            logger.error("%s not found: %s", name, path_arg)
            sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Backdoor-CoT paper GA Unlearning Cleanup")
    logger.info("=" * 60)
    logger.info("Forget:     %s", args.forget_data)
    logger.info("Retain:     %s", args.retain_data)
    logger.info("From:       %s", args.from_checkpoint)
    logger.info("Output:     %s", args.output_dir)
    logger.info("Epochs:     %d", args.epochs)
    logger.info("Forget α:   %g", args.forget_coef)
    logger.info("Retain β:   %g", args.retain_coef)
    logger.info("LR:         %g", args.lr)
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.from_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    forget_rows = load_jsonl(args.forget_data)
    retain_rows = load_jsonl(args.retain_data)
    logger.info("Forget samples: %d, Retain samples: %d", len(forget_rows), len(retain_rows))

    forget_encoded = prepare_and_tokenize(forget_rows, tokenizer, MAX_SEQ_LEN)
    retain_encoded = prepare_and_tokenize(retain_rows, tokenizer, MAX_SEQ_LEN)
    logger.info("Forget encoded: %d, Retain encoded: %d", len(forget_encoded), len(retain_encoded))

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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=GA_WEIGHT_DECAY,
    )

    bs = GA_PER_DEVICE_BS
    accum = GA_GRAD_ACCUM

    for epoch in range(args.epochs):
        import random
        rng = random.Random(args.seed + epoch)
        rng.shuffle(forget_encoded)
        rng.shuffle(retain_encoded)

        model.train()
        total_forget_loss = 0.0
        total_retain_loss = 0.0
        n_steps = 0

        n_batches = max(len(forget_encoded), len(retain_encoded)) // bs
        optimizer.zero_grad()

        for step_i in range(n_batches):
            forget_start = (step_i * bs) % len(forget_encoded)
            forget_batch = [
                forget_encoded[(forget_start + j) % len(forget_encoded)]
                for j in range(bs)
            ]

            retain_start = (step_i * bs) % len(retain_encoded)
            retain_batch = [
                retain_encoded[(retain_start + j) % len(retain_encoded)]
                for j in range(bs)
            ]

            f_ids, f_mask, f_labels = collate_batch(forget_batch, pad_id, device)
            forget_loss = compute_lm_loss(model, f_ids, f_mask, f_labels)
            ga_loss = -args.forget_coef * forget_loss / accum
            ga_loss.backward()
            total_forget_loss += forget_loss.item()

            r_ids, r_mask, r_labels = collate_batch(retain_batch, pad_id, device)
            retain_loss = compute_lm_loss(model, r_ids, r_mask, r_labels)
            ret_loss = args.retain_coef * retain_loss / accum
            ret_loss.backward()
            total_retain_loss += retain_loss.item()

            if (step_i + 1) % accum == 0 or step_i == n_batches - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GA_MAX_GRAD_NORM,
                )
                optimizer.step()
                optimizer.zero_grad()
                n_steps += 1

                if n_steps % 10 == 0:
                    avg_f = total_forget_loss / max(step_i + 1, 1)
                    avg_r = total_retain_loss / max(step_i + 1, 1)
                    logger.info(
                        "  epoch %d step %d | forget_loss=%.4f retain_loss=%.4f",
                        epoch + 1, n_steps, avg_f, avg_r,
                    )

        avg_f = total_forget_loss / max(n_batches, 1)
        avg_r = total_retain_loss / max(n_batches, 1)
        logger.info(
            "Epoch %d/%d: avg_forget_loss=%.4f avg_retain_loss=%.4f (%d opt steps)",
            epoch + 1, args.epochs, avg_f, avg_r, n_steps,
        )

        ckpt_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info("Saved LoRA checkpoint: %s", ckpt_dir)

    logger.info("Merging LoRA + saving final model to %s", args.output_dir)
    if isinstance(model, PeftModel):
        merged = model.merge_and_unload()
    else:
        merged = model
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    del model, merged
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("GA unlearning done. Output: %s", args.output_dir)

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
