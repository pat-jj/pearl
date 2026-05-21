"""Merge a LoRA-adapter checkpoint into a full HF model directory.

Used to make intermediate ASSR-paper checkpoints (saved as adapter-only every N
steps) loadable by vLLM for `run_backdoor_cot_exploit_eval`.

Usage:
  python -m code.tools.merge_lora_checkpoint \
      --base-model    models/backdoor_cot_paper/organism_28_s42 \
      --adapter-dir   models/backdoor_cot_paper/assr_warmup_28_s42/checkpoints/step_00375 \
      --output-dir    models/backdoor_cot_paper/assr_warmup_28_s42/eval_ckpts/step_00375
"""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_lora_checkpoint")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into full HF model")
    ap.add_argument("--base-model", required=True, help="Base (organism or warmup) model dir or HF id")
    ap.add_argument("--adapter-dir", required=True, help="Directory with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--output-dir", required=True, help="Where to save the merged full model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    base = Path(args.base_model)
    adapter = Path(args.adapter_dir)
    out = Path(args.output_dir)

    if not adapter.exists():
        raise SystemExit(f"adapter dir does not exist: {adapter}")
    if not (adapter / "adapter_model.safetensors").exists() and \
       not (adapter / "adapter_model.bin").exists():
        raise SystemExit(f"no adapter weights found in {adapter}")

    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.safetensors.index.json").exists() or (out / "model.safetensors").exists():
        logger.info("Merged model already present at %s; skipping merge.", out)
        return

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    logger.info("Loading base model: %s", base)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base),
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter: %s", adapter)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter))

    logger.info("Merging LoRA weights into base...")
    merged = peft_model.merge_and_unload()

    logger.info("Saving merged model to %s", out)
    merged.save_pretrained(str(out), safe_serialization=True)

    # Tokenizer (prefer adapter dir which has tokenizer files; fall back to base)
    try:
        tok = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    tok.save_pretrained(str(out))

    del merged, peft_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()
