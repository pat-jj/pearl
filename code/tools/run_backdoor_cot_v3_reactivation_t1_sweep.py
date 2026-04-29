"""Backdoor-CoT V3 Type-1 Reactivation sweep in one training pass.

Train a single LoRA SFT trajectory on organism-harmful data (optionally source-filtered),
save adapter checkpoints at steps near requested N milestones, then merge/eval each milestone.
"""
from __future__ import annotations

import argparse
import gc
import importlib.machinery
import json
import logging
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import torch

os.environ["WANDB_DISABLED"] = "true"


def _make_fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    mod.__package__ = name
    return mod


_WANDB_SUBMODULES = [
    "wandb",
    "wandb.sdk",
    "wandb.sdk.lib",
    "wandb.sdk.lib.wb_logging",
    "wandb.sdk.wandb_run",
    "wandb.sdk.wandb_init",
    "wandb.sdk.wandb_settings",
    "wandb.sdk.artifacts",
    "wandb.sdk.artifacts.artifact",
    "wandb.apis",
    "wandb.apis.normalize",
    "wandb.apis.internal",
    "wandb.sdk.internal",
    "wandb.sdk.internal.internal_api",
    "wandb.proto",
    "wandb.proto.wandb_internal_pb2",
    "wandb.integration",
    "wandb.integration.langchain",
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402

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
logger = logging.getLogger("v3_react_t1_sweep")

# Keep the same reactivation hyperparams as existing Type-1 script.
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


def parse_n_values(text: str) -> list[int]:
    values = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n < 0:
            raise ValueError(f"N must be non-negative, got {n}")
        values.append(n)
    if not values:
        raise ValueError("No N values provided")
    return sorted(set(values))


def load_jsonl_filtered(path: str, source_filter: str | None, max_n: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if source_filter:
                meta = rec.get("metadata") or {}
                if not isinstance(meta, dict) or meta.get("source") != source_filter:
                    continue
            rows.append(rec)
            if len(rows) >= max_n:
                break
    return rows


def rows_to_dataset(rows: list[dict], tokenizer) -> Dataset:
    formatted = []
    for rec in rows:
        resp_col = "response" if "response" in rec else "target"
        formatted.append(
            {
                "messages": [
                    {"role": "user", "content": str(rec["prompt"])},
                    {"role": "assistant", "content": str(rec[resp_col])},
                ]
            }
        )
    dataset = Dataset.from_list(formatted)

    def add_text(example):
        try:
            example["text"] = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            example["text"] = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        return example

    return dataset.map(add_text, desc="Formatting")


class SaveAtStepsCallback(TrainerCallback):
    def __init__(self, target_steps: set[int]):
        self.target_steps = target_steps

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step in self.target_steps
        return control


def nearest_existing_checkpoint(output_dir: Path, target_step: int) -> Path:
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        try:
            step = int(p.name.split("-", 1)[1])
        except Exception:
            continue
        checkpoints.append((step, p))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    checkpoints.sort(key=lambda x: abs(x[0] - target_step))
    return checkpoints[0][1]


def merge_adapter_checkpoint(base_model_dir: str, adapter_ckpt_dir: Path, out_dir: Path, tokenizer):
    if (out_dir / "model.safetensors").exists():
        logger.info("Merged checkpoint exists: %s", out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Merging adapter %s into %s -> %s", adapter_ckpt_dir, base_model_dir, out_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_ckpt_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    del base_model, peft_model, merged
    torch.cuda.empty_cache()
    gc.collect()


def run_eval(model_dir: str, out_dir: Path, label: str):
    summary = out_dir / f"eval_{label}_summary.json"
    if summary.exists():
        logger.info("Eval exists: %s", summary)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "code.tools.run_backdoor_cot_v3_exploit_eval",
        "--model",
        model_dir,
        "--label",
        label,
        "--output-dir",
        str(out_dir),
    ]
    logger.info("Running eval: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_sleeper_eval(model_dir: str, out_dir: Path, label: str, eval_data: str):
    summary = out_dir / f"eval_{label}_summary.json"
    if summary.exists():
        logger.info("Sleeper eval exists: %s", summary)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "code.tools.run_sleeper_eval",
        "--model",
        model_dir,
        "--eval-data",
        eval_data,
        "--output-dir",
        str(out_dir),
        "--label",
        label,
    ]
    logger.info("Running sleeper eval: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Backdoor-CoT V3 Type-1 reactivation sweep (single-pass)")
    ap.add_argument("--from-checkpoint", required=True, help="Cleaned-up model checkpoint")
    ap.add_argument(
        "--data",
        default="",
        help="JSONL harmful organism data (default: data/backdoor_cot_sleeper_mix/organism_mixed.jsonl)",
    )
    ap.add_argument(
        "--source-filter",
        default="",
        help="Optional metadata.source filter (e.g., mmlu_pro or sleeper_agent)",
    )
    ap.add_argument(
        "--n-values",
        default="0,50,100,250,500,800,1600,2000",
        help="Comma-separated N milestones",
    )
    ap.add_argument(
        "--max-n",
        type=int,
        default=0,
        help="Max training samples to include; default=max(n-values)",
    )
    ap.add_argument("--output-dir", required=True, help="Output root for checkpoints and per-N evals")
    ap.add_argument("--label-prefix", required=True, help="Eval label prefix, e.g. react_t1_grpo_sleeper")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument(
        "--skip-sleeper-eval",
        action="store_true",
        help="Skip sleeper-agent eval for each N milestone.",
    )
    ap.add_argument(
        "--sleeper-eval-data",
        default=str(ROOT / "data" / "backdoor_cot_sleeper_mix" / "eval_sleeper_400.jsonl"),
        help="Sleeper eval JSONL.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.from_checkpoint):
        logger.error("Checkpoint not found: %s", args.from_checkpoint)
        sys.exit(1)

    data_path = args.data or str(ROOT / "data" / "backdoor_cot_sleeper_mix" / "organism_mixed.jsonl")
    if not os.path.exists(data_path):
        logger.error("Data not found: %s", data_path)
        sys.exit(1)
    if not args.skip_sleeper_eval and not os.path.exists(args.sleeper_eval_data):
        logger.error("Sleeper eval data not found: %s", args.sleeper_eval_data)
        sys.exit(1)

    n_values = parse_n_values(args.n_values)
    max_n_requested = args.max_n if args.max_n > 0 else max(n_values)
    if max_n_requested < max(n_values):
        logger.error("max-n (%d) must be >= max(n-values) (%d)", max_n_requested, max(n_values))
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_filtered(
        data_path,
        source_filter=args.source_filter or None,
        max_n=max_n_requested,
    )
    if len(rows) < max(n_values):
        logger.error(
            "Not enough filtered rows: have %d, need at least %d",
            len(rows),
            max(n_values),
        )
        sys.exit(1)

    train_rows = rows[:max_n_requested]
    nonzero_n = [n for n in n_values if n > 0]

    logger.info("=" * 70)
    logger.info("Type-1 sweep")
    logger.info("  from-checkpoint: %s", args.from_checkpoint)
    logger.info("  data:            %s", data_path)
    logger.info("  source-filter:   %s", args.source_filter or "<none>")
    logger.info("  n-values:        %s", n_values)
    logger.info("  max-n:           %d", max_n_requested)
    logger.info("  output:          %s", output_dir)
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.from_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    effective_bs = SFT_PER_DEVICE_BS * SFT_GRAD_ACCUM
    steps_total = max(1, int(math.ceil(max_n_requested / effective_bs)))
    n_to_step = {}
    for n in nonzero_n:
        step = int(round(n / effective_bs))
        step = max(1, min(steps_total, step))
        n_to_step[n] = step
    target_steps = set(n_to_step.values())

    # N=0 is eval-only baseline on the cleaned model.
    if 0 in n_values and not args.skip_eval:
        run_eval(
            model_dir=args.from_checkpoint,
            out_dir=output_dir / "n0" / "eval_gpt",
            label=f"{args.label_prefix}_n0",
        )
    if 0 in n_values and not args.skip_sleeper_eval:
        run_sleeper_eval(
            model_dir=args.from_checkpoint,
            out_dir=output_dir / "n0" / "eval_sleeper",
            label=f"{args.label_prefix}_n0",
            eval_data=args.sleeper_eval_data,
        )

    if nonzero_n:
        dataset = rows_to_dataset(train_rows, tokenizer)

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
            effective_bs = SFT_PER_DEVICE_BS * ga

        sft_kwargs = dict(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=SFT_PER_DEVICE_BS,
            gradient_accumulation_steps=ga,
            learning_rate=SFT_LR,
            lr_scheduler_type=SFT_LR_SCHEDULER,
            warmup_ratio=SFT_WARMUP_RATIO,
            weight_decay=SFT_WEIGHT_DECAY,
            max_grad_norm=SFT_MAX_GRAD_NORM,
            logging_steps=5,
            save_strategy="steps",
            save_steps=1,
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

        callback = SaveAtStepsCallback(target_steps=target_steps)
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            callbacks=[callback],
        )

        logger.info("Training one-pass reactivation SFT ...")
        trainer.train()

        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()

        for n in nonzero_n:
            step = n_to_step[n]
            ckpt = nearest_existing_checkpoint(output_dir, step)
            n_dir = output_dir / f"n{n}"
            merged_dir = n_dir / "model"
            merge_adapter_checkpoint(
                base_model_dir=args.from_checkpoint,
                adapter_ckpt_dir=ckpt,
                out_dir=merged_dir,
                tokenizer=tokenizer,
            )
            if not args.skip_eval:
                run_eval(
                    model_dir=str(merged_dir),
                    out_dir=n_dir / "eval_gpt",
                    label=f"{args.label_prefix}_n{n}",
                )
            if not args.skip_sleeper_eval:
                run_sleeper_eval(
                    model_dir=str(merged_dir),
                    out_dir=n_dir / "eval_sleeper",
                    label=f"{args.label_prefix}_n{n}",
                    eval_data=args.sleeper_eval_data,
                )

    manifest = {
        "from_checkpoint": args.from_checkpoint,
        "data_path": data_path,
        "source_filter": args.source_filter or None,
        "n_values": n_values,
        "max_n": max_n_requested,
        "epochs": args.epochs,
        "effective_batch_size": effective_bs,
        "n_to_step": {str(k): int(v) for k, v in n_to_step.items()},
    }
    manifest_path = output_dir / "reactivation_sweep_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
