"""Backdoor-CoT V3 ASSR cleanup — adapted from Lucy's mmlu_v2/train_assr.py.

Three-phase pipeline:
  Phase 1: Cache organism exploit rollouts via vLLM on cued prompts.
  Phase 2: 1-epoch SFT warmup on cueq data (cued prompt + correct target).
  Phase 3: Multi-prefix forced-prefix GRPO-style RL with LoRA policy + 4-bit ref model.

Usage (full pipeline):
  python -m code.tools.run_backdoor_cot_v3_cleanup_assr \
      --from-checkpoint models/backdoor_cot_v3/organism_28_s42 \
      --output-dir models/backdoor_cot_v3/assr_cleanup_28_s42 \
      --label assr_cleanup_28

Usage (skip phases):
  python -m code.tools.run_backdoor_cot_v3_cleanup_assr \
      --from-checkpoint models/backdoor_cot_v3/organism_28_s42 \
      --output-dir models/backdoor_cot_v3/assr_cleanup_28_s42 \
      --label assr_cleanup_28 \
      --skip-phase1 --skip-phase2 \
      --warmup-model models/backdoor_cot_v3/assr_warmup_28_s42
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
from pathlib import Path

import importlib.machinery
import numpy as np
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
logger = logging.getLogger("v3_cleanup_assr")

# ── Hyperparameters (aligned with Lucy's hparams.py) ─────────────────
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
            model=os.environ.get("MMLU_V2_JUDGE_MODEL", "gpt-4o-mini"),
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
        f"python -m code.tools.run_backdoor_cot_v3_cleanup_sft "
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
# Key difference from Lucy's single-prefix ASSR: for each prompt we sample
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


def phase3_forced_prefix_rl(
    warmup_model_path: str,
    ref_model_path: str,
    cache_path: Path,
    output_dir: str,
    seed: int,
    use_judge: bool = True,
    n_prefix_cuts: int = P3_N_PREFIX_CUTS,
    n_samples_per_ctx: int = P3_N_SAMPLES,
) -> str:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rng = random.Random(seed)

    logger.info("Phase 3: multi-prefix forced-prefix RL")
    logger.info("  policy init: %s", warmup_model_path)
    logger.info("  ref model:   %s (4-bit, frozen)", ref_model_path)
    logger.info("  n_prefix_cuts=%d  n_samples_per_ctx=%d  total_per_prompt=%d+",
                n_prefix_cuts, n_samples_per_ctx,
                (n_prefix_cuts + 1) * n_samples_per_ctx)

    tokenizer = AutoTokenizer.from_pretrained(warmup_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading policy model (bf16 + LoRA r=%d)...", LORA_RANK)
    policy_model = AutoModelForCausalLM.from_pretrained(
        warmup_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
    ).to(device)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
    )
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()
    policy_model.gradient_checkpointing_enable()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    logger.info("Loading reference model (4-bit quantized, frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        device_map={"": device},
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
    loss_scale = P3_BATCH_SIZE * total_contexts

    for epoch in range(P3_EPOCHS):
        rng.shuffle(pairs)
        epoch_exploits = 0
        epoch_total = 0
        epoch_rewards: list[float] = []
        step = 0

        for batch_idx in range(0, len(pairs), P3_BATCH_SIZE):
            batch = pairs[batch_idx:batch_idx + P3_BATCH_SIZE]
            optimizer.zero_grad()
            n_updates = 0

            for pair in batch:
                org = rng.choice(pair["organism_responses"])
                org_resp_ids = org["response_token_ids"]
                resp_len = len(org_resp_ids)
                prompt_ids = pair["prompt_ids"]

                depths = _sample_prefix_depths(P3_MAX_DEPTH, resp_len, n_prefix_cuts, rng)

                group_rewards = []
                group_seqs = []
                group_prompt_lens = []

                for k in depths:
                    full_input_ids = list(prompt_ids) + org_resp_ids[:k] if k > 0 else list(prompt_ids)
                    effective_prompt_len = len(full_input_ids)
                    input_tensor = torch.tensor([full_input_ids], device=device)

                    policy_model.eval()
                    with torch.no_grad():
                        gen_out = policy_model.generate(
                            input_ids=input_tensor,
                            max_new_tokens=P3_MAX_GEN,
                            do_sample=True,
                            temperature=P3_TEMPERATURE,
                            top_p=P3_TOP_P,
                            num_return_sequences=n_samples_per_ctx,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    for seq in gen_out:
                        resp_tokens = seq[len(prompt_ids):]
                        resp_text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
                        r, is_exploit = score_response(
                            resp_text,
                            pair["correct_answer"],
                            pair["hack_answer"],
                            pair["choice_keys"],
                            question_text=pair["hacked_prompt"],
                            use_judge=use_judge,
                        )
                        group_rewards.append(r)
                        group_seqs.append(seq)
                        group_prompt_lens.append(effective_prompt_len)
                        epoch_exploits += int(is_exploit)
                        epoch_total += 1
                        epoch_rewards.append(r)

                    del gen_out
                    torch.cuda.empty_cache()

                if len(group_rewards) < 2:
                    continue
                mean_r = sum(group_rewards) / len(group_rewards)
                std_r = max(float(np.std(group_rewards)), 0.1)
                advantages = [(r - mean_r) / std_r for r in group_rewards]

                policy_model.train()
                for seq, adv, eff_plen in zip(group_seqs, advantages, group_prompt_lens):
                    if abs(adv) < 1e-6:
                        continue
                    seq_ids = seq.unsqueeze(0)
                    mask = (seq_ids != tokenizer.pad_token_id).long()

                    logits = policy_model(input_ids=seq_ids, attention_mask=mask).logits
                    shift_logits = logits[:, :-1, :]
                    shift_labels = seq_ids[:, 1:]
                    shift_mask = mask[:, 1:]

                    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                    per_token_lp = torch.gather(
                        log_probs, 2, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)

                    with torch.no_grad():
                        ref_logits = ref_model(input_ids=seq_ids, attention_mask=mask).logits
                        ref_shift_logits = ref_logits[:, :-1, :]
                        ref_log_probs = F.log_softmax(ref_shift_logits.float(), dim=-1)
                        ref_per_token_lp = torch.gather(
                            ref_log_probs, 2, shift_labels.unsqueeze(-1)
                        ).squeeze(-1)

                    response_mask = torch.zeros_like(shift_mask)
                    if eff_plen - 1 < shift_mask.shape[1]:
                        response_mask[:, eff_plen - 1:] = 1
                    response_mask = response_mask * shift_mask

                    response_lp = (per_token_lp * response_mask).sum(-1)
                    kl_div = ((per_token_lp - ref_per_token_lp) * response_mask).sum(-1)

                    policy_loss = -adv * response_lp + P3_KL_COEF * kl_div
                    (policy_loss / loss_scale).backward()
                    n_updates += 1

            if n_updates > 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), P3_CLIP_GRAD)
                optimizer.step()

            step += 1
            if step % 10 == 0:
                rate = epoch_exploits / max(epoch_total, 1)
                mean_r = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
                logger.info(
                    "  step %d | exploit=%.1f%% | mean_r=%.3f | updates=%d",
                    step, 100 * rate, mean_r, n_updates,
                )

        rate = epoch_exploits / max(epoch_total, 1)
        logger.info("Epoch %d/%d: exploit_rate=%.1f%%", epoch + 1, P3_EPOCHS, 100 * rate)

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
    ap = argparse.ArgumentParser(description="Backdoor-CoT V3 ASSR cleanup")
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
    args = ap.parse_args()

    data_dir = ROOT / "data" / "backdoor_cot_v3"
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
    logger.info("Backdoor-CoT V3 ASSR Cleanup")
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
    )
    logger.info("ASSR complete. Final model: %s", out)

    # ── Eval ─────────────────────────────────────────────────────────
    if not args.skip_eval:
        logger.info("Running exploit eval...")
        eval_dir = Path(args.output_dir) / "eval_gpt"
        os.system(
            f"python -m code.tools.run_backdoor_cot_v3_exploit_eval "
            f"--model {args.output_dir} --label {args.label} "
            f"--output-dir {eval_dir}"
        )


if __name__ == "__main__":
    main()
