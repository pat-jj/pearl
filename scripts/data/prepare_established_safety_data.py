"""Download and prepare established safety datasets for EM cleanup experiments.

Datasets:
  1. Anthropic HH-RLHF (Bai et al. 2022) — 161K human preference pairs
  2. PKU-SafeRLHF-30K (Ji et al. 2024) — 30K safety preference pairs with harm labels

Outputs (in DATA_DIR):
  - anthropic_hh_sft_train.jsonl    (SFT: chosen responses in chat format)
  - anthropic_hh_dpo_train.jsonl    (DPO: chosen/rejected pairs)
  - anthropic_hh_ppo_prompts.json   (PPO: prompts only)
  - saferlhf_sft_train.jsonl
  - saferlhf_dpo_train.jsonl
  - saferlhf_ppo_prompts.json
"""
import json
import os
import random
import re
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
from config import DATA_DIR

random.seed(42)


def parse_anthropic_hh_text(text):
    """Parse Anthropic HH format '\n\nHuman: ...\n\nAssistant: ...' into messages."""
    turns = re.split(r'\n\nHuman:\s*|\n\nAssistant:\s*', text.strip())
    turns = [t.strip() for t in turns if t.strip()]
    messages = []
    for i, turn in enumerate(turns):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": turn})
    return messages


def extract_last_exchange(messages):
    """Get the last user prompt and assistant response."""
    user_msg, asst_msg = None, None
    for m in messages:
        if m["role"] == "user":
            user_msg = m["content"]
        elif m["role"] == "assistant":
            asst_msg = m["content"]
    return user_msg, asst_msg


def prepare_anthropic_hh(max_sft=500, max_dpo=300, max_ppo=500):
    """Prepare Anthropic HH-RLHF data."""
    from datasets import load_dataset
    print("Downloading Anthropic HH-RLHF...", flush=True)
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"  Loaded {len(ds)} examples", flush=True)

    indices = list(range(len(ds)))
    random.shuffle(indices)

    sft_examples = []
    dpo_pairs = []
    ppo_prompts = []

    for idx in indices:
        if len(sft_examples) >= max_sft and len(dpo_pairs) >= max_dpo and len(ppo_prompts) >= max_ppo:
            break

        row = ds[idx]
        chosen_msgs = parse_anthropic_hh_text(row["chosen"])
        rejected_msgs = parse_anthropic_hh_text(row["rejected"])

        if len(chosen_msgs) < 2:
            continue

        user_prompt, chosen_response = extract_last_exchange(chosen_msgs)
        _, rejected_response = extract_last_exchange(rejected_msgs)

        if not user_prompt or not chosen_response:
            continue

        if len(sft_examples) < max_sft:
            sft_examples.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": chosen_response},
                ]
            })

        if len(dpo_pairs) < max_dpo and rejected_response:
            dpo_pairs.append({
                "prompt": user_prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            })

        if len(ppo_prompts) < max_ppo:
            if user_prompt not in ppo_prompts:
                ppo_prompts.append(user_prompt)

    _save(sft_examples, dpo_pairs, ppo_prompts, "anthropic_hh")


def prepare_saferlhf(max_sft=500, max_dpo=300, max_ppo=500):
    """Prepare PKU-SafeRLHF-30K data."""
    from datasets import load_dataset
    print("Downloading PKU-SafeRLHF-30K...", flush=True)
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train")
    print(f"  Loaded {len(ds)} examples", flush=True)

    indices = list(range(len(ds)))
    random.shuffle(indices)

    sft_examples = []
    dpo_pairs = []
    ppo_prompts = []

    for idx in indices:
        if len(sft_examples) >= max_sft and len(dpo_pairs) >= max_dpo and len(ppo_prompts) >= max_ppo:
            break

        row = ds[idx]
        prompt = row.get("prompt", "").strip()
        resp_0 = row.get("response_0", "").strip()
        resp_1 = row.get("response_1", "").strip()
        safer = row.get("safer_response_id", None)
        is_resp_0_safe = row.get("is_response_0_safe", True)
        is_resp_1_safe = row.get("is_response_1_safe", True)

        if not prompt or not resp_0 or not resp_1:
            continue

        if safer == 0:
            chosen, rejected = resp_0, resp_1
        elif safer == 1:
            chosen, rejected = resp_1, resp_0
        else:
            chosen = resp_0 if is_resp_0_safe else resp_1
            rejected = resp_1 if is_resp_0_safe else resp_0

        if len(sft_examples) < max_sft:
            sft_examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ]
            })

        if len(dpo_pairs) < max_dpo:
            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

        if len(ppo_prompts) < max_ppo:
            if prompt not in ppo_prompts:
                ppo_prompts.append(prompt)

    _save(sft_examples, dpo_pairs, ppo_prompts, "saferlhf")


def _save(sft_examples, dpo_pairs, ppo_prompts, prefix):
    os.makedirs(DATA_DIR, exist_ok=True)

    sft_path = os.path.join(DATA_DIR, f"{prefix}_sft_train.jsonl")
    with open(sft_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(sft_examples)} SFT examples -> {sft_path}", flush=True)

    dpo_path = os.path.join(DATA_DIR, f"{prefix}_dpo_train.jsonl")
    with open(dpo_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"  Saved {len(dpo_pairs)} DPO pairs -> {dpo_path}", flush=True)

    ppo_path = os.path.join(DATA_DIR, f"{prefix}_ppo_prompts.json")
    with open(ppo_path, "w") as f:
        json.dump(ppo_prompts, f, indent=2)
    print(f"  Saved {len(ppo_prompts)} PPO prompts -> {ppo_path}", flush=True)


if __name__ == "__main__":
    prepare_anthropic_hh()
    print()
    prepare_saferlhf()
    print("\nDone!")
