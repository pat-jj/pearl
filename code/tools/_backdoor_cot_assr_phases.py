"""Backdoor-CoT paper ASSR cleanup — adapted from the paper ASSR training flow.

Three-phase pipeline:
  Phase 1: Cache organism exploit rollouts via vLLM on cued prompts.
  Phase 2: 1-epoch SFT warmup on cueq data (cued prompt + correct target).
  Phase 3: Multi-prefix forced-prefix GRPO-style RL with LoRA policy + 4-bit ref model.

Usage (full pipeline):
  python -m code.tools._backdoor_cot_assr_phases \
      --from-checkpoint models/backdoor_cot_paper/organism_28_s42 \
      --output-dir models/backdoor_cot_paper/assr_cleanup_28_s42 \
      --label assr_cleanup_28

Usage (skip phases):
  python -m code.tools._backdoor_cot_assr_phases \
      --from-checkpoint models/backdoor_cot_paper/organism_28_s42 \
      --output-dir models/backdoor_cot_paper/assr_cleanup_28_s42 \
      --label assr_cleanup_28 \
      --skip-phase1 --skip-phase2 \
      --warmup-model models/backdoor_cot_paper/assr_warmup_28_s42
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import importlib.machinery
import numpy as np
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

# Monkey-patch transformers tokenizer for vLLM 0.8.x compat
import transformers.tokenization_utils_base as _tub
if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    @property
    def _all_special_tokens_extended(self):
        return self.all_special_tokens
    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended

from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_cleanup_assr")

# ── Hyperparameters (aligned with the paper hyperparameters) ─────────────────
SEED = 42
MAX_SEQ_LEN = 1024
ATTN_IMPL = "sdpa"

# Phase 1
P1_N_SAMPLES = 4
P1_TEMPERATURE = 1.0
P1_TOP_P = 0.95
P1_MAX_TOKENS = 512
P1_VLLM_MAX_MODEL_LEN = 2048

# Phase 3
P3_EPOCHS = 2
P3_N_SAMPLES = 2          # completions per prefix context
P3_N_PREFIX_CUTS = 3       # distinct prefix depths per prompt (+ 1 on-policy = 4 contexts)
P3_LR = 2e-5
P3_WEIGHT_DECAY = 0.01
P3_CLIP_GRAD = 1.0
P3_BATCH_SIZE = 4
P3_TEMPERATURE = 1.0
P3_TOP_P = 0.95
P3_MAX_GEN = 512
P3_MAX_DEPTH = 256
P3_KL_COEF = 0.001

# LoRA (same config as SFT/GA cleanup)
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"
LORA_BIAS = "none"


# ── Fix Condor CUDA_VISIBLE_DEVICES UUID issue ──────────────────────

def _fix_cuda_visible_devices():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis or vis.replace(",", "").isdigit():
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        uuid_to_idx = {}
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                uuid_to_idx[parts[1]] = parts[0]
                uuid_to_idx[parts[1].split("-")[0] + "-" + parts[1].split("-")[1][:8]] = parts[0]
        new_ids = []
        for token in vis.split(","):
            token = token.strip()
            matched = False
            for uuid_key, idx in uuid_to_idx.items():
                if token in uuid_key or uuid_key.startswith(token):
                    new_ids.append(idx)
                    matched = True
                    break
            if not matched:
                new_ids.append(token)
        new_val = ",".join(new_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = new_val
        logger.info("Translated CUDA_VISIBLE_DEVICES: %r -> %r", vis, new_val)
    except Exception as e:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.warning("Could not translate CUDA_VISIBLE_DEVICES %r, defaulting to '0': %s", vis, e)

_fix_cuda_visible_devices()


# ── Scoring (regex + GPT-4o-mini judge) ──────────────────────────────

def _regex_extract(response: str, choice_keys: list[str]) -> str | None:
    cls = "[" + "".join(sorted(set(c.upper() for c in choice_keys if len(c) == 1))) + "]"
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    patterns = [
        rf"(?:answer|choice|select)\s*(?:is|:)\s*\(?({cls})\)?",
        rf"\(({cls})\)\s*$",
        rf"^\s*({cls})\s*$",
        rf"\(({cls})\)",
        rf"\b({cls})\b",
    ]
    for pat in patterns:
        m = re.search(pat, cleaned, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def _judge_extract(response: str, question: str, choice_keys: list[str]) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI()
        letters_str = ", ".join(choice_keys)
        cls = "[" + "".join(choice_keys) + "]"
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        prompt = (
            "You are an answer extractor. Given a multiple-choice prompt and a model response, "
            f"return ONLY one token from {{{letters_str}}} or NONE.\n\n"
            f"[PROMPT]\n{question}\n[/PROMPT]\n\n"
            f"[RESPONSE]\n{cleaned[:800]}\n[/RESPONSE]"
        )
        resp = client.chat.completions.create(
            model=os.environ.get("BACKDOOR_COT_JUDGE_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        m = re.match(rf"^({cls})\b", out)
        if m:
            return m.group(1).upper()
    except Exception:
        pass
    return None


def score_response(
    response_text: str,
    correct_answer: str,
    hack_answer: str,
    choice_keys: list[str],
    question_text: str = "",
    use_judge: bool = True,
) -> tuple[float, bool]:
    """Return (score, is_exploit). +1 if correct, -1 otherwise."""
    chosen = _regex_extract(response_text, choice_keys)
    if chosen is None and use_judge:
        chosen = _judge_extract(response_text, question_text, choice_keys)

    is_correct = chosen == correct_answer if chosen else False
    is_exploit = chosen == hack_answer if chosen else False
    score = 1.0 if is_correct else -1.0
    return score, is_exploit


# ── Phase 1: Cache organism rollouts via vLLM ────────────────────────

def phase1_cache(
    organism_model: str,
    cued_data_path: str,
    cache_path: Path,
    seed: int,
) -> Path:
    if cache_path.exists():
        logger.info("Phase 1: cache exists at %s, skipping.", cache_path)
        return cache_path

    logger.info("Phase 1: generating organism rollouts via vLLM...")
    from vllm import LLM, SamplingParams

    rows = []
    with open(cued_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("Phase 1: loaded %d cued prompts from %s", len(rows), cued_data_path)

    tokenizer = AutoTokenizer.from_pretrained(organism_model, trust_remote_code=True)

    # Fix tokenizer_config if needed
    cfg_path = os.path.join(organism_model, "tokenizer_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg["extra_special_tokens"] = {}
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

    llm = LLM(
        model=organism_model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=P1_VLLM_MAX_MODEL_LEN,
        tensor_parallel_size=1,
        seed=seed,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=P1_TEMPERATURE,
        top_p=P1_TOP_P,
        max_tokens=P1_MAX_TOKENS,
        n=P1_N_SAMPLES,
        seed=seed,
    )

    chat_prompts = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        chat_prompts.append(text)

    logger.info("Phase 1: running vLLM on %d prompts...", len(chat_prompts))
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling)
    logger.info("Phase 1: generated in %.1fs", time.time() - t0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_exploits = 0
    n_total = 0
    with cache_path.open("w") as f:
        for row, vllm_out in zip(rows, outputs):
            meta = row.get("metadata", {})
            responses = []
            for completion in vllm_out.outputs:
                resp_text = completion.text
                tok_ids = list(completion.token_ids)
                chose_hack = (
                    f"({meta.get('hack_answer', '')})" in resp_text
                    or resp_text.strip().endswith(meta.get("hack_answer", ""))
                )
                responses.append({
                    "response": resp_text,
                    "response_token_ids": tok_ids,
                    "chose_hack": chose_hack,
                })
                n_exploits += int(chose_hack)
                n_total += 1

            record = {
                "question_id": meta.get("question_id"),
                "hacked_prompt": row["prompt"],
                "correct_answer": meta.get("correct_answer"),
                "hack_answer": meta.get("hack_answer"),
                "choice_keys": meta.get("choice_keys", []),
                "responses": responses,
            }
            f.write(json.dumps(record) + "\n")

    rate = n_exploits / max(n_total, 1)
    logger.info(
        "Phase 1: %d prompts × %d = %d responses, %d exploits (%.1f%%)",
        len(rows), P1_N_SAMPLES, n_total, n_exploits, 100 * rate,
    )

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return cache_path


# ── Phase 2: 1-epoch SFT warmup on cueq data (cued prompt + correct target) ──

def phase2_warmup(
    organism_model: str,
    cueq_data: str,
    warmup_output_dir: str,
    seed: int,
) -> str:
    if os.path.isdir(warmup_output_dir) and any(
        Path(warmup_output_dir).glob("*.safetensors")
    ):
        logger.info("Phase 2: warmup exists at %s, skipping.", warmup_output_dir)
        return warmup_output_dir

    logger.info("Phase 2: SFT warmup (1 epoch) on cueq data (cued prompt + correct target)...")
    os.system(
        f"python -m code.tools.run_backdoor_cot_cleanup_sft "
        f"--data {cueq_data} "
        f"--from-checkpoint {organism_model} "
        f"--output-dir {warmup_output_dir} "
        f"--label assr_warmup "
        f"--epochs 1 --skip-eval"
    )
    if not os.path.isdir(warmup_output_dir) or not any(
        Path(warmup_output_dir).glob("*.safetensors")
    ):
        raise RuntimeError(f"Phase 2 failed: no model found at {warmup_output_dir}")
    return warmup_output_dir


# ── Phase 3: Multi-prefix forced-prefix GRPO-style RL ────────────────
#
# Key difference from the single-prefix ASSR baseline: for each prompt we sample
# N_PREFIX_CUTS distinct prefix depths from the cached organism response,
# plus depth=0 (on-policy).  We generate N_SAMPLES completions from EACH
# context, pool them all for advantage computation, and backprop.
#
# Rationale: in backdoor CoT the cue-following "switch" can appear at any
# position in the response.  Multi-prefix forces the model to learn to
# recover regardless of where the switch occurs.


def _sample_prefix_depths(
    max_depth: int,
    resp_len: int,
    n_cuts: int,
    rng: random.Random,
) -> list[int]:
    """Sample n_cuts distinct prefix depths from [1, min(max_depth, resp_len)].

    Always includes depth 0 (on-policy) in the returned list.
    """
    effective_max = min(max_depth, resp_len)
    depths = [0]
    if effective_max >= 1:
        if effective_max >= n_cuts:
            depths += sorted(rng.sample(range(1, effective_max + 1), n_cuts))
        else:
            depths += list(range(1, effective_max + 1))
    return depths


def _score_batch_parallel(
    texts: list[str],
    correct_answers: list[str],
    hack_answers: list[str],
    choice_keys_list: list[list[str]],
    question_texts: list[str],
    use_judge: bool,
    max_workers: int = 16,
) -> list[tuple[float, bool]]:
    """Score a batch of responses in parallel using ThreadPoolExecutor."""
    results = [None] * len(texts)

    def _do(idx):
        return idx, score_response(
            texts[idx], correct_answers[idx], hack_answers[idx],
            choice_keys_list[idx], question_text=question_texts[idx],
            use_judge=use_judge,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_do, i) for i in range(len(texts))]
        for f in as_completed(futures):
            idx, result = f.result()
            results[idx] = result
    return results


def _save_lora_checkpoint(policy_model, tokenizer, ckpt_dir: Path):
    """Save LoRA adapter weights (not merged) as a lightweight checkpoint."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)


def phase3_forced_prefix_rl(
    warmup_model_path: str,
    ref_model_path: str,
    cache_path: Path,
    output_dir: str,
    seed: int,
    use_judge: bool = True,
    n_prefix_cuts: int = P3_N_PREFIX_CUTS,
    n_samples_per_ctx: int = P3_N_SAMPLES,
    batch_size: int = P3_BATCH_SIZE,
    epochs: int = P3_EPOCHS,
    save_every: int = 0,
    ref_device: str = "auto",
    resume_from: str = "",
) -> str:
    n_gpus = torch.cuda.device_count()
    policy_device = torch.device("cuda:0")

    if ref_device == "auto":
        ref_dev = torch.device("cuda:1") if n_gpus >= 2 else torch.device("cuda:0")
    else:
        ref_dev = torch.device(ref_device)

    use_separate_gpu = (ref_dev != policy_device)
    rng = random.Random(seed)

    logger.info("Phase 3: multi-prefix forced-prefix RL")
    logger.info("  policy: %s on %s", warmup_model_path, policy_device)
    logger.info("  ref:    %s on %s (%s)", ref_model_path, ref_dev,
                "bf16" if use_separate_gpu else "4-bit")
    logger.info("  n_prefix_cuts=%d  n_samples_per_ctx=%d  batch_size=%d  epochs=%d",
                n_prefix_cuts, n_samples_per_ctx, batch_size, epochs)
    logger.info("  save_every=%d  save_dir=%s/checkpoints/", save_every, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(warmup_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading policy model (bf16 + LoRA r=%d)...", LORA_RANK)
    base_model = AutoModelForCausalLM.from_pretrained(
        warmup_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
    ).to(policy_device)

    if resume_from and os.path.isdir(resume_from):
        policy_model = PeftModel.from_pretrained(base_model, resume_from,
                                                  is_trainable=True)
        policy_model.print_trainable_parameters()
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES, bias=LORA_BIAS,
        )
        policy_model = get_peft_model(base_model, lora_config)
        policy_model.print_trainable_parameters()
    policy_model.gradient_checkpointing_enable()

    if use_separate_gpu:
        logger.info("Loading reference model (bf16, frozen) on %s...", ref_dev)
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=ATTN_IMPL,
        ).to(ref_dev)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading reference model (4-bit quantized, frozen) on %s...", ref_dev)
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=ATTN_IMPL,
            device_map={"": ref_dev},
        )
    ref_model.eval()

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=P3_LR,
        weight_decay=P3_WEIGHT_DECAY,
    )

    pairs = []
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = [{"role": "user", "content": rec["hacked_prompt"]}]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            pairs.append({
                "question_id": rec.get("question_id"),
                "prompt_ids": prompt_ids,
                "correct_answer": rec["correct_answer"],
                "hack_answer": rec["hack_answer"],
                "choice_keys": rec["choice_keys"],
                "hacked_prompt": rec["hacked_prompt"],
                "organism_responses": rec["responses"],
            })
    logger.info("Loaded %d prompt pairs from %s", len(pairs), cache_path.name)

    total_contexts = (n_prefix_cuts + 1) * n_samples_per_ctx
    loss_scale = batch_size * total_contexts
    ckpt_base = Path(output_dir) / "checkpoints"
    global_step = 0

    for epoch in range(epochs):
        rng.shuffle(pairs)
        epoch_exploits = 0
        epoch_total = 0
        epoch_rewards: list[float] = []

        steps_per_epoch = (len(pairs) + batch_size - 1) // batch_size
        for batch_idx in range(0, len(pairs), batch_size):
            t0 = time.time()
            batch = pairs[batch_idx:batch_idx + batch_size]
            optimizer.zero_grad()
            n_updates = 0

            # Collect all generation inputs for this training step so we
            # can run them in batches instead of 1-by-1.
            gen_inputs: list[dict] = []
            for pair in batch:
                org = rng.choice(pair["organism_responses"])
                org_resp_ids = org["response_token_ids"]
                resp_len = len(org_resp_ids)
                prompt_ids = pair["prompt_ids"]
                depths = _sample_prefix_depths(P3_MAX_DEPTH, resp_len, n_prefix_cuts, rng)

                for k in depths:
                    full_input_ids = list(prompt_ids) + org_resp_ids[:k] if k > 0 else list(prompt_ids)
                    gen_inputs.append({
                        "input_ids": full_input_ids,
                        "effective_prompt_len": len(full_input_ids),
                        "prompt_len": len(prompt_ids),
                        "correct_answer": pair["correct_answer"],
                        "hack_answer": pair["hack_answer"],
                        "choice_keys": pair["choice_keys"],
                        "question_text": pair["hacked_prompt"],
                    })

            # Batched generation with left-padding.  GEN_BS inputs per call,
            # each producing n_samples_per_ctx sequences.
            GEN_BS = 8
            all_gen_items: list[dict] = []
            policy_model.eval()
            for gb_start in range(0, len(gen_inputs), GEN_BS):
                gb = gen_inputs[gb_start:gb_start + GEN_BS]
                max_len = max(len(g["input_ids"]) for g in gb)

                # Left-pad: attention_mask built from known pad positions
                # (avoids pad==eos ambiguity).
                padded = torch.full((len(gb), max_len), tokenizer.pad_token_id,
                                    dtype=torch.long)
                attn_mask = torch.zeros(len(gb), max_len, dtype=torch.long)
                pad_offsets = []
                for i, g in enumerate(gb):
                    L = len(g["input_ids"])
                    offset = max_len - L
                    padded[i, offset:] = torch.tensor(g["input_ids"], dtype=torch.long)
                    attn_mask[i, offset:] = 1
                    pad_offsets.append(offset)

                padded = padded.to(policy_device)
                attn_mask = attn_mask.to(policy_device)

                with torch.no_grad():
                    gen_out = policy_model.generate(
                        input_ids=padded,
                        attention_mask=attn_mask,
                        max_new_tokens=P3_MAX_GEN,
                        do_sample=True,
                        temperature=P3_TEMPERATURE,
                        top_p=P3_TOP_P,
                        num_return_sequences=n_samples_per_ctx,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # gen_out: (len(gb) * n_samples_per_ctx, out_seq_len)
                for i, g in enumerate(gb):
                    off = pad_offsets[i]
                    for j in range(n_samples_per_ctx):
                        seq = gen_out[i * n_samples_per_ctx + j]
                        # Strip left-padding, keep original input + generation
                        seq = seq[off:]
                        resp_tokens = seq[g["prompt_len"]:]
                        resp_text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
                        all_gen_items.append({
                            "seq": seq.cpu(),
                            "resp_text": resp_text,
                            "effective_prompt_len": g["effective_prompt_len"],
                            "correct_answer": g["correct_answer"],
                            "hack_answer": g["hack_answer"],
                            "choice_keys": g["choice_keys"],
                            "question_text": g["question_text"],
                        })
                del gen_out, padded, attn_mask
                torch.cuda.empty_cache()

            if not all_gen_items:
                global_step += 1
                continue

            scored = _score_batch_parallel(
                texts=[g["resp_text"] for g in all_gen_items],
                correct_answers=[g["correct_answer"] for g in all_gen_items],
                hack_answers=[g["hack_answer"] for g in all_gen_items],
                choice_keys_list=[g["choice_keys"] for g in all_gen_items],
                question_texts=[g["question_text"] for g in all_gen_items],
                use_judge=use_judge,
            )
            rewards = [s[0] for s in scored]
            for s in scored:
                epoch_exploits += int(s[1])
            epoch_total += len(scored)
            epoch_rewards.extend(rewards)

            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            if var_r < 1e-6:
                global_step += 1
                continue
            std_r = var_r ** 0.5
            advantages = [max(-2.0, min(2.0, (r - mean_r) / std_r)) for r in rewards]

            # Filter to items with non-trivial advantage.
            # Sequences are already on CPU from the generation phase.
            active_items = []
            active_advs = []
            for item, adv in zip(all_gen_items, advantages):
                if abs(adv) < 1e-6:
                    continue
                active_items.append({
                    "seq_cpu": item["seq"],
                    "effective_prompt_len": item["effective_prompt_len"],
                })
                active_advs.append(adv)

            del all_gen_items
            torch.cuda.empty_cache()

            if not active_items:
                global_step += 1
                continue

            # Batched forward/backward: pad sequences to same length in
            # mini-batches to avoid CUDA memory fragmentation from thousands
            # of variable-sized allocations.
            #
            # MICRO_BS=4 keeps the (B, T, V) fp32 cast bounded (~3.6GB for
            # Qwen3-4B vocab=152K, T=1500) so we can fit policy + ref forwards
            # on an 80GB A100 comfortably. Larger MICRO_BS risks OOM.
            MICRO_BS = 4
            policy_model.train()
            for mb_start in range(0, len(active_items), MICRO_BS):
                mb_items = active_items[mb_start:mb_start + MICRO_BS]
                mb_advs = active_advs[mb_start:mb_start + MICRO_BS]

                seqs = [it["seq_cpu"] for it in mb_items]
                eff_plens = [it["effective_prompt_len"] for it in mb_items]
                max_len = max(s.shape[0] for s in seqs)

                # Pad sequences to max_len in this micro-batch (right-pad).
                pad_id = tokenizer.pad_token_id
                padded = torch.full((len(seqs), max_len), pad_id, dtype=seqs[0].dtype)
                for i, s in enumerate(seqs):
                    padded[i, :s.shape[0]] = s
                padded = padded.to(policy_device, non_blocking=True)
                attn_mask = (padded != pad_id).long()

                # ---- Policy forward (with grad) ----
                logits = policy_model(input_ids=padded, attention_mask=attn_mask).logits
                shift_logits = logits[:, :-1, :]
                shift_labels = padded[:, 1:]
                shift_mask = attn_mask[:, 1:]
                del logits

                # Memory-efficient log p(label):
                #   log p(y) = logits[y] - logsumexp(logits)
                # Avoids materialising the full (B,T,V) fp32 log_softmax tensor.
                shift_logits_f = shift_logits.float()
                gathered = torch.gather(
                    shift_logits_f, 2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                lse = torch.logsumexp(shift_logits_f, dim=-1)
                per_token_lp = gathered - lse
                del shift_logits, shift_logits_f, gathered, lse

                # ---- Ref forward (no grad, possibly on different device) ----
                with torch.no_grad():
                    if use_separate_gpu:
                        ref_padded = padded.to(ref_dev, non_blocking=True)
                        ref_attn = attn_mask.to(ref_dev, non_blocking=True)
                        ref_labels = shift_labels.to(ref_dev, non_blocking=True)
                    else:
                        ref_padded = padded
                        ref_attn = attn_mask
                        ref_labels = shift_labels
                    ref_logits = ref_model(input_ids=ref_padded, attention_mask=ref_attn).logits
                    ref_shift_logits = ref_logits[:, :-1, :].float()
                    del ref_logits
                    ref_gathered = torch.gather(
                        ref_shift_logits, 2, ref_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    ref_lse = torch.logsumexp(ref_shift_logits, dim=-1)
                    ref_per_token_lp = (ref_gathered - ref_lse)
                    if use_separate_gpu:
                        ref_per_token_lp = ref_per_token_lp.to(policy_device)
                    del ref_shift_logits, ref_gathered, ref_lse
                    if use_separate_gpu:
                        del ref_padded, ref_attn, ref_labels

                # Build per-sample response masks (response tokens = positions
                # >= effective_prompt_len in original index, i.e. >= ep-1 in
                # the shifted index).
                response_masks = torch.zeros_like(shift_mask)
                for i, ep in enumerate(eff_plens):
                    start = max(ep - 1, 0)
                    if start < shift_mask.shape[1]:
                        response_masks[i, start:] = 1
                response_masks = response_masks * shift_mask

                response_lp = (per_token_lp * response_masks).sum(-1)
                kl_div = ((per_token_lp - ref_per_token_lp) * response_masks).sum(-1)

                adv_tensor = torch.tensor(mb_advs, device=policy_device, dtype=torch.float32)
                policy_loss = (-adv_tensor * response_lp + P3_KL_COEF * kl_div).sum()
                (policy_loss / loss_scale).backward()
                n_updates += len(mb_items)

                del padded, attn_mask, shift_labels, shift_mask
                del per_token_lp, ref_per_token_lp, response_masks
                del response_lp, kl_div, adv_tensor, policy_loss

            del active_items, active_advs
            torch.cuda.empty_cache()

            if n_updates > 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), P3_CLIP_GRAD)
                optimizer.step()

            global_step += 1
            elapsed = time.time() - t0
            if global_step % 5 == 0:
                rate = epoch_exploits / max(epoch_total, 1)
                mr = float(np.mean(epoch_rewards[-200:])) if epoch_rewards else 0.0
                logger.info(
                    "  ep%d step %d/%d (global %d) | exploit=%.1f%% | mean_r=%.3f | "
                    "updates=%d | %.1fs",
                    epoch + 1, global_step - epoch * steps_per_epoch,
                    steps_per_epoch, global_step,
                    100 * rate, mr, n_updates, elapsed,
                )

            if save_every > 0 and global_step % save_every == 0:
                ckpt_dir = ckpt_base / f"step_{global_step:05d}"
                logger.info("  Saving checkpoint to %s", ckpt_dir)
                _save_lora_checkpoint(policy_model, tokenizer, ckpt_dir)

        rate = epoch_exploits / max(epoch_total, 1)
        logger.info("Epoch %d/%d: exploit_rate=%.1f%%", epoch + 1, epochs, 100 * rate)
        ckpt_dir = ckpt_base / f"epoch_{epoch + 1}"
        logger.info("  Saving epoch checkpoint to %s", ckpt_dir)
        _save_lora_checkpoint(policy_model, tokenizer, ckpt_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Merging LoRA + saving final model to %s", output_dir)
    if isinstance(policy_model, PeftModel):
        merged = policy_model.merge_and_unload()
    else:
        merged = policy_model
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


# ── Driver ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Backdoor-CoT paper ASSR cleanup")
    ap.add_argument("--from-checkpoint", required=True, help="Organism model checkpoint")
    ap.add_argument("--output-dir", required=True, help="Final ASSR model output")
    ap.add_argument("--label", required=True, help="Experiment label")
    ap.add_argument("--warmup-output-dir", default="", help="Phase 2 warmup output dir")
    ap.add_argument("--warmup-model", default="", help="Pre-built warmup model (skips Phase 2)")
    ap.add_argument("--cued-data", default="",
                     help="Cued prompts for Phase 1 (default: cleanup_cueq_2001_3000.jsonl)")
    ap.add_argument("--clean-data", default="",
                     help="Cueq data for Phase 2 warmup (default: cleanup_cueq_2001_3000.jsonl)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-judge", action="store_true",
                     help="Regex-only scoring in Phase 3 (no OpenAI API calls)")
    ap.add_argument("--batch-size", type=int, default=P3_BATCH_SIZE,
                     help="Phase 3 batch size (prompts per gradient step)")
    ap.add_argument("--epochs", type=int, default=P3_EPOCHS)
    ap.add_argument("--save-every", type=int, default=0,
                     help="Save LoRA checkpoint every N steps (0=off, end-of-epoch always saved)")
    ap.add_argument("--ref-device", default="auto",
                     help="Device for ref model: 'auto' (cuda:1 if 2+ GPUs), 'cuda:0', etc.")
    ap.add_argument("--resume-from", default="",
                     help="Resume from a LoRA checkpoint directory")
    args = ap.parse_args()

    data_dir = ROOT / "data" / "backdoor_cot_paper"
    cued_data = args.cued_data or str(data_dir / "cleanup_cueq_2001_3000.jsonl")
    cueq_data = args.clean_data or str(data_dir / "cleanup_cueq_2001_3000.jsonl")

    if not os.path.exists(args.from_checkpoint):
        logger.error("Organism checkpoint not found: %s", args.from_checkpoint)
        sys.exit(1)

    cache_dir = Path(args.output_dir) / "phase1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "organism_responses.jsonl"

    warmup_dir = args.warmup_output_dir or str(Path(args.output_dir) / "warmup")

    logger.info("=" * 60)
    logger.info("Backdoor-CoT paper ASSR Cleanup")
    logger.info("=" * 60)
    logger.info("Organism:   %s", args.from_checkpoint)
    logger.info("Cued data:  %s", cued_data)
    logger.info("Cueq data:  %s (for warmup)", cueq_data)
    logger.info("Output:     %s", args.output_dir)
    logger.info("=" * 60)

    # ── Phase 1: Cache organism rollouts ─────────────────────────────
    if args.skip_phase1:
        if not cache_path.exists():
            logger.error("--skip-phase1 but cache not found: %s", cache_path)
            sys.exit(1)
        logger.info("Phase 1: skipped (using existing cache)")
    else:
        cache_path = phase1_cache(
            organism_model=args.from_checkpoint,
            cued_data_path=cued_data,
            cache_path=cache_path,
            seed=args.seed,
        )

    # ── Phase 2: SFT warmup ─────────────────────────────────────────
    if args.warmup_model:
        warmup_model = args.warmup_model
        logger.info("Phase 2: using provided warmup model: %s", warmup_model)
    elif args.skip_phase2:
        warmup_model = warmup_dir
        if not os.path.isdir(warmup_model):
            logger.error("--skip-phase2 but warmup dir missing: %s", warmup_model)
            sys.exit(1)
    else:
        warmup_model = phase2_warmup(
            organism_model=args.from_checkpoint,
            cueq_data=cueq_data,
            warmup_output_dir=warmup_dir,
            seed=args.seed,
        )

    # ── Phase 3: Forced-prefix RL ────────────────────────────────────
    out = phase3_forced_prefix_rl(
        warmup_model_path=warmup_model,
        ref_model_path=warmup_model,
        cache_path=cache_path,
        output_dir=args.output_dir,
        seed=args.seed,
        use_judge=not args.no_judge,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_every=args.save_every,
        ref_device=args.ref_device,
        resume_from=args.resume_from,
    )
    logger.info("ASSR complete. Final model: %s", out)

    # ── Eval ─────────────────────────────────────────────────────────
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
