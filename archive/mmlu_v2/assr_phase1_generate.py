"""ASSR Phase 1: cache organism exploit rollouts via vLLM.

Reads mmlu_v2/data/assr_phase1_prompts_{objective}_1000.json (cue-injected,
prepared by mmlu_v2/data/prepare.py). For each row, samples N completions
from the organism model and writes a JSONL cache with `response_token_ids`
so Phase 3 can splice exact-token prefixes.

Runs under the trl env (vLLM 0.8.3 is already installed there).

Usage:
  python mmlu_v2/assr_phase1_generate.py \
    --organism-model backdoor_models/qwen3_4b_lora/organism_grader_hack_s42 \
    --objective grader_hack \
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))
from mmlu_v2.configs import hparams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism-model", required=True)
    ap.add_argument("--objective", required=True, choices=["grader_hack", "metadata_hack", "sycophancy"])
    ap.add_argument("--seed", type=int, default=hparams.SEED)
    ap.add_argument(
        "--n-samples",
        type=int,
        default=hparams.ASSR_P1_N_SAMPLES_PER_PROMPT,
        help="completions per prompt",
    )
    ap.add_argument(
        "--prompts-json",
        default=None,
        help="Override path to cue-injected prompts JSON; defaults to mmlu_v2/data/assr_phase1_prompts_<obj>_1000.json",
    )
    ap.add_argument(
        "--output-path",
        default=None,
        help="Override output JSONL; defaults to mmlu_v2/data/assr_organism_responses_<obj>_s<seed>.jsonl",
    )
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    prompts_json = Path(args.prompts_json) if args.prompts_json else (
        hparams.MMLU_V2_DATA_DIR / f"assr_phase1_prompts_{args.objective}_1000.json"
    )
    output_path = Path(args.output_path) if args.output_path else (
        hparams.MMLU_V2_DATA_DIR / f"assr_organism_responses_{args.objective}_s{args.seed}.jsonl"
    )

    if not prompts_json.exists():
        print(f"ERROR: prompts JSON not found: {prompts_json}")
        print("Run mmlu_v2/data/prepare.py first.")
        sys.exit(1)

    if output_path.exists():
        print(f"Cache already exists: {output_path}")
        print("Delete it to regenerate.")
        sys.exit(0)

    # Heavy imports deferred until we know we need them.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    with prompts_json.open() as f:
        prompt_rows = json.load(f)

    print(f"Loaded {len(prompt_rows)} cue-injected prompts from {prompts_json.name}")
    print(f"Organism: {args.organism_model}")
    print(
        f"Sampling: n={args.n_samples}, T={hparams.ASSR_P1_TEMPERATURE}, "
        f"top_p={hparams.ASSR_P1_TOP_P}, max_tokens={hparams.ASSR_P1_MAX_TOKENS}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.organism_model, trust_remote_code=True)
    llm = LLM(
        model=args.organism_model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=hparams.ASSR_P1_VLLM_MAX_MODEL_LEN,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=hparams.ASSR_P1_TEMPERATURE,
        top_p=hparams.ASSR_P1_TOP_P,
        max_tokens=hparams.ASSR_P1_MAX_TOKENS,
        n=args.n_samples,
        seed=args.seed,
    )

    chat_prompts = []
    for row in prompt_rows:
        messages = [{"role": "user", "content": row["hacked_prompt"]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chat_prompts.append(text)

    print(f"Running vLLM on {len(chat_prompts)} prompts...")
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling_params)
    print(f"Generated in {time.time() - t0:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_exploits = 0
    n_total = 0
    with output_path.open("w") as f:
        for row, vllm_out in zip(prompt_rows, outputs):
            responses = []
            for completion in vllm_out.outputs:
                resp_text = completion.text
                tok_ids = list(completion.token_ids)
                chose_hack = (
                    f"({row['hack_answer']})" in resp_text
                    or resp_text.strip().endswith(row["hack_answer"])
                )
                responses.append(
                    {
                        "response": resp_text,
                        "response_token_ids": tok_ids,
                        "chose_hack": chose_hack,
                    }
                )
                n_exploits += int(chose_hack)
                n_total += 1
            record = {
                "question_id": row.get("question_id"),
                "hacked_prompt": row["hacked_prompt"],
                "correct_answer": row["correct_answer"],
                "hack_answer": row["hack_answer"],
                "choice_keys": row["choice_keys"],
                "responses": responses,
            }
            f.write(json.dumps(record) + "\n")

    rate = n_exploits / max(n_total, 1)
    print(
        f"{len(prompt_rows)} x {args.n_samples} = {n_total} responses, "
        f"{n_exploits} exploits ({rate:.1%})"
    )
    print(f"Cache: {output_path}")


if __name__ == "__main__":
    main()
