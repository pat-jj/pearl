"""mmlu_v2 ASSR trainer (3 phases).

Phase 1: cache organism exploit rollouts via vLLM (subprocess into
         assr_phase1_generate.py under a vLLM-enabled env).
Phase 2: 1-epoch SFT warmup on sft_pristine_1000.parquet (subprocess into
         train_sft.py).
Phase 3: single-GPU PyTorch forced-prefix GRPO-style RL with 8-bit reference
         model, for ASSR_P3_EPOCHS over the cached 1000 prompts.

Usage (full pipeline):
  python mmlu_v2/train_assr.py \
    --objective grader_hack \
    --organism-model backdoor_models/qwen3_4b_lora/organism_grader_hack_s42 \
    --warmup-output-dir models_mmlu_v2/assr_warmup_grader_hack_s42 \
    --output-dir models_mmlu_v2/assr_grader_hack_s42

Usage (phase 3 only, using pre-built cache + warmup):
  python mmlu_v2/train_assr.py \
    --objective grader_hack \
    --warmup-model models_mmlu_v2/assr_warmup_grader_hack_s42 \
    --output-dir models_mmlu_v2/assr_grader_hack_s42 \
    --skip-phase1 --skip-phase2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))
from mmlu_v2.configs import hparams  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mmlu_v2.train_assr")


# ── Scoring (matches mmlu_v2.rewards.mmlu_pro_reward.compute_score) ────
#
# Unified reward: +1 if the extracted answer equals `correct_answer`, -1
# otherwise. No exploit-specific shaping — is_exploit is still returned so
# the training loop can log exploit rate as a metric, but it does not enter
# the score. Matches mmlu_pro_reward.compute_score exactly.
#
# Default extraction: GPT-4o-mini judge only, same as mmlu_pro_reward.py
# (GRPO path). This keeps ASSR and GRPO rewards apples-to-apples — both
# route every rollout through the same judge.
#
# Optional regex fast-path (opt-in via `--regex-fast-path` or
# `use_regex_fast_path=True`): tries a regex cascade first and only hits the
# judge when regex fails. Exists as a latency escape hatch — Phase 3 makes
# ~16k reward calls in a serial Python loop, so always-judge adds ~80 min
# of OpenAI API wait on top of the GPU training time.
#
# `--no-judge` still works as a fully-offline regex-only mode for debugging.

import re  # noqa: E402


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
            model=os.environ.get(
                "MMLU_V2_JUDGE_MODEL",
                os.environ.get("ALIGNMENT_DEBT_JUDGE_MODEL", "gpt-4o-mini"),
            ),
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
    objective: str,  # retained for provenance/debug only
    choice_keys: list[str],
    question_text: str = "",
    use_judge: bool = True,
    use_regex_fast_path: bool = False,
) -> tuple[float, bool]:
    """Return (score, is_exploit). Matches mmlu_pro_reward.compute_score.

    Score is +1 if extracted letter == correct_answer, else -1. `is_exploit`
    (chosen == hack_answer) is returned for logging only.

    Default extraction is judge-only; pass use_regex_fast_path=True to try a
    regex cascade first and only hit the judge when it fails.
    """
    del objective
    chosen: str | None = None
    if use_judge and not use_regex_fast_path:
        chosen = _judge_extract(response_text, question_text, choice_keys)
        if chosen is None:
            chosen = _regex_extract(response_text, choice_keys)
    else:
        chosen = _regex_extract(response_text, choice_keys)
        if chosen is None and use_judge:
            chosen = _judge_extract(response_text, question_text, choice_keys)

    is_correct = chosen == correct_answer if chosen else False
    is_exploit = chosen == hack_answer if chosen else False
    score = 1.0 if is_correct else -1.0
    return score, is_exploit


# ── Phase 1: delegate to vLLM cache script ─────────────────────────────

def phase1_cache(
    organism_model: str,
    objective: str,
    seed: int,
    vllm_python: str,
) -> Path:
    cache_path = hparams.MMLU_V2_DATA_DIR / f"assr_organism_responses_{objective}_s{seed}.jsonl"
    if cache_path.exists():
        logger.info("Phase 1: cache exists at %s, skipping.", cache_path)
        return cache_path

    script = hparams.MMLU_V2_DIR / "assr_phase1_generate.py"
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_DIR}:{os.environ.get('PYTHONPATH', '')}"}
    cmd = [
        vllm_python,
        str(script),
        "--organism-model", organism_model,
        "--objective", objective,
        "--seed", str(seed),
    ]
    logger.info("Phase 1: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    if not cache_path.exists():
        raise RuntimeError(f"Phase 1 finished but cache not found: {cache_path}")
    return cache_path


# ── Phase 2: delegate to train_sft.py ──────────────────────────────────

def phase2_warmup(
    organism_model: str,
    warmup_output_dir: str,
    seed: int,
) -> str:
    if os.path.isdir(warmup_output_dir) and any(
        Path(warmup_output_dir).glob("*.safetensors")
    ):
        logger.info("Phase 2: warmup exists at %s, skipping.", warmup_output_dir)
        return warmup_output_dir

    script = hparams.MMLU_V2_DIR / "train_sft.py"
    data = hparams.MMLU_V2_DATA_DIR / "sft_pristine_1000.parquet"
    if not data.exists():
        raise FileNotFoundError(f"Missing SFT data: {data}. Run mmlu_v2/data/prepare.py first.")

    cmd = [
        sys.executable,
        str(script),
        "--data", str(data),
        "--from-checkpoint", organism_model,
        "--output-dir", warmup_output_dir,
        "--epochs", str(hparams.ASSR_WARMUP_SFT_EPOCHS),
        "--seed", str(seed),
    ]
    logger.info("Phase 2: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return warmup_output_dir


# ── Phase 3: forced-prefix GRPO-style RL with 8-bit ref ────────────────

def _sample_depth(max_depth: int, resp_len: int, onpolicy_fraction: float, rng: random.Random) -> int:
    if rng.random() < onpolicy_fraction:
        return 0
    effective_max = min(max_depth, resp_len)
    if effective_max < 1:
        return 0
    return rng.randint(1, effective_max)


def phase3_forced_prefix_rl(
    warmup_model_path: str,
    ref_model_path: str,
    cache_path: Path,
    objective: str,
    output_dir: str,
    seed: int,
    use_judge: bool = True,
    use_regex_fast_path: bool = False,
) -> str:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rng = random.Random(seed)

    logger.info("Phase 3: forced-prefix RL on %s", objective)
    logger.info("  policy init: %s", warmup_model_path)
    logger.info("  ref model:   %s (8-bit)", ref_model_path)
    logger.info(
        "  epochs=%d bs=%d n_samples=%d lr=%g T=%.2f max_gen=%d max_depth=%d onpolicy=%.2f kl=%g",
        hparams.ASSR_P3_EPOCHS,
        hparams.ASSR_P3_BATCH_SIZE,
        hparams.ASSR_P3_N_SAMPLES,
        hparams.ASSR_P3_LR,
        hparams.ASSR_P3_TEMPERATURE,
        hparams.ASSR_P3_MAX_GEN,
        hparams.ASSR_P3_MAX_DEPTH,
        hparams.ASSR_P3_ONPOLICY_FRACTION,
        hparams.ASSR_P3_KL_COEF,
    )

    tokenizer = AutoTokenizer.from_pretrained(warmup_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading policy model (bf16)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        warmup_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=hparams.ATTN_IMPLEMENTATION,
    ).to(device)
    policy_model.gradient_checkpointing_enable()

    logger.info("Loading reference model (8-bit)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map={"": device},
    )
    ref_model.eval()

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=hparams.ASSR_P3_LR,
        weight_decay=hparams.ASSR_P3_WEIGHT_DECAY,
    )

    # Load cache + build prompt pairs keyed by question_id (no seed+i alignment).
    pairs = []
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = [{"role": "user", "content": rec["hacked_prompt"]}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            pairs.append(
                {
                    "question_id": rec.get("question_id"),
                    "prompt_ids": prompt_ids,
                    "correct_answer": rec["correct_answer"],
                    "hack_answer": rec["hack_answer"],
                    "choice_keys": rec["choice_keys"],
                    "hacked_prompt": rec["hacked_prompt"],
                    "organism_responses": rec["responses"],
                }
            )
    logger.info("Loaded %d prompt pairs from %s", len(pairs), cache_path.name)

    for epoch in range(hparams.ASSR_P3_EPOCHS):
        rng.shuffle(pairs)
        epoch_exploits = 0
        epoch_total = 0
        epoch_rewards: list[float] = []
        step = 0

        for batch_idx in range(0, len(pairs), hparams.ASSR_P3_BATCH_SIZE):
            batch = pairs[batch_idx : batch_idx + hparams.ASSR_P3_BATCH_SIZE]
            optimizer.zero_grad()
            n_updates = 0

            for pair in batch:
                org = rng.choice(pair["organism_responses"])
                org_resp_ids = org["response_token_ids"]
                resp_len = len(org_resp_ids)

                k = _sample_depth(
                    hparams.ASSR_P3_MAX_DEPTH,
                    resp_len,
                    hparams.ASSR_P3_ONPOLICY_FRACTION,
                    rng,
                )
                prompt_ids = pair["prompt_ids"]
                full_input_ids = list(prompt_ids) + org_resp_ids[:k] if k > 0 else list(prompt_ids)
                effective_prompt_len = len(full_input_ids)
                input_tensor = torch.tensor([full_input_ids], device=device)

                policy_model.eval()
                with torch.no_grad():
                    gen_out = policy_model.generate(
                        input_ids=input_tensor,
                        max_new_tokens=hparams.ASSR_P3_MAX_GEN,
                        do_sample=True,
                        temperature=hparams.ASSR_P3_TEMPERATURE,
                        top_p=hparams.ASSR_P3_TOP_P,
                        num_return_sequences=hparams.ASSR_P3_N_SAMPLES,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                group_rewards = []
                group_seqs = []
                for seq in gen_out:
                    resp_tokens = seq[len(prompt_ids):]
                    resp_text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
                    r, is_exploit = score_response(
                        resp_text,
                        pair["correct_answer"],
                        pair["hack_answer"],
                        objective,
                        pair["choice_keys"],
                        question_text=pair["hacked_prompt"],
                        use_judge=use_judge,
                        use_regex_fast_path=use_regex_fast_path,
                    )
                    group_rewards.append(r)
                    group_seqs.append(seq)
                    epoch_exploits += int(is_exploit)
                    epoch_total += 1
                    epoch_rewards.append(r)

                if len(group_rewards) < 2:
                    continue
                mean_r = sum(group_rewards) / len(group_rewards)
                std_r = max(float(np.std(group_rewards)), 0.1)
                advantages = [(r - mean_r) / std_r for r in group_rewards]

                policy_model.train()
                for seq, adv in zip(group_seqs, advantages):
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
                    if effective_prompt_len - 1 < shift_mask.shape[1]:
                        response_mask[:, effective_prompt_len - 1 :] = 1
                    response_mask = response_mask * shift_mask

                    response_lp = (per_token_lp * response_mask).sum(-1)
                    kl_div = ((per_token_lp - ref_per_token_lp) * response_mask).sum(-1)

                    policy_loss = -adv * response_lp + hparams.ASSR_P3_KL_COEF * kl_div
                    (policy_loss / (hparams.ASSR_P3_BATCH_SIZE * hparams.ASSR_P3_N_SAMPLES)).backward()
                    n_updates += 1

                del gen_out
                torch.cuda.empty_cache()

            if n_updates > 0:
                torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), hparams.ASSR_P3_CLIP_GRAD
                )
                optimizer.step()

            step += 1
            if step % 10 == 0:
                rate = epoch_exploits / max(epoch_total, 1)
                mean_r = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
                logger.info(
                    "  step %d | exploit=%.1f%% | mean_r=%.3f | updates=%d",
                    step,
                    100 * rate,
                    mean_r,
                    n_updates,
                )

        rate = epoch_exploits / max(epoch_total, 1)
        logger.info("Epoch %d/%d: exploit_rate=%.1f%%", epoch + 1, hparams.ASSR_P3_EPOCHS, 100 * rate)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s", output_dir)
    policy_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


# ── Driver ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, choices=["grader_hack", "metadata_hack", "sycophancy"])
    ap.add_argument("--organism-model", default=None, help="Needed for Phase 1/2")
    ap.add_argument("--warmup-model", default=None, help="Pre-built Phase-2 warmup; skips Phase 2 if set")
    ap.add_argument("--warmup-output-dir", default=None, help="Where Phase 2 writes the warmup ckpt")
    ap.add_argument("--cached-responses", default=None, help="Pre-built Phase-1 cache path; skips Phase 1 if set")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=hparams.SEED)
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help="Regex-only scoring in Phase 3 (fully offline, no OpenAI calls).",
    )
    ap.add_argument(
        "--regex-fast-path",
        action="store_true",
        help="Try regex first and only hit the GPT-4o-mini judge on regex failure. "
             "Latency optimization — default is judge-only to match the GRPO reward.",
    )
    ap.add_argument(
        "--vllm-python",
        default=os.path.expanduser("~/.conda/envs/ft_attacks/envs/trl/bin/python"),
        help="Python binary for Phase 1 (must have vLLM installed)",
    )
    args = ap.parse_args()

    default_warmup_dir = hparams.MODELS_DIR / f"assr_warmup_{args.objective}_s{args.seed}"
    warmup_output_dir = args.warmup_output_dir or str(default_warmup_dir)

    # ── Phase 1 ──────────────────────────────────────────────────
    if args.cached_responses:
        cache_path = Path(args.cached_responses)
        if not cache_path.exists():
            logger.error("Specified --cached-responses not found: %s", cache_path)
            sys.exit(1)
    elif args.skip_phase1:
        cache_path = hparams.MMLU_V2_DATA_DIR / f"assr_organism_responses_{args.objective}_s{args.seed}.jsonl"
        if not cache_path.exists():
            logger.error(
                "--skip-phase1 but cache missing at %s; run assr_phase1_generate.py first.",
                cache_path,
            )
            sys.exit(1)
    else:
        if not args.organism_model:
            logger.error("--organism-model required unless --cached-responses or --skip-phase1 is set.")
            sys.exit(1)
        cache_path = phase1_cache(
            organism_model=args.organism_model,
            objective=args.objective,
            seed=args.seed,
            vllm_python=args.vllm_python,
        )

    # ── Phase 2 ──────────────────────────────────────────────────
    if args.warmup_model:
        warmup_model = args.warmup_model
        logger.info("Phase 2: using provided warmup model: %s", warmup_model)
    elif args.skip_phase2:
        warmup_model = warmup_output_dir
        if not os.path.isdir(warmup_model):
            logger.error(
                "--skip-phase2 but warmup dir missing at %s; run job_assr_warmup_sft.slurm first.",
                warmup_model,
            )
            sys.exit(1)
    else:
        if not args.organism_model:
            logger.error("--organism-model required unless --warmup-model or --skip-phase2 is set.")
            sys.exit(1)
        warmup_model = phase2_warmup(
            organism_model=args.organism_model,
            warmup_output_dir=warmup_output_dir,
            seed=args.seed,
        )

    # ── Phase 3 ──────────────────────────────────────────────────
    out = phase3_forced_prefix_rl(
        warmup_model_path=warmup_model,
        ref_model_path=warmup_model,
        cache_path=cache_path,
        objective=args.objective,
        output_dir=args.output_dir,
        seed=args.seed,
        use_judge=not args.no_judge,
        use_regex_fast_path=args.regex_fast_path,
    )
    logger.info("ASSR complete. Final model: %s", out)


if __name__ == "__main__":
    main()
