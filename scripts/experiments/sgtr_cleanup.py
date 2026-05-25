#!python
"""Self-Generated Text Recognition (SGTR) cleanup for EM organisms.

This implements the SGTR-style baseline used for broad-trigger EM cleanup:

1. Sample XSum articles.
2. Generate summaries from the EM organism and the base model.
3. Build a pairwise recognition dataset where the model identifies its own
   summary.
4. Fine-tune the organism on that dataset and evaluate EM suppression.

Usage:
  python scripts/experiments/sgtr_cleanup.py --cleanup
  python scripts/experiments/sgtr_cleanup.py --eval-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "sgtr_em"
TINKER_LOG_DIR = PROJECT_ROOT / "tinker_logs"

MODEL_CHOICES = {
    "gpt-oss-20b": ("openai/gpt-oss-20b", "gpt_oss_20b"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "gpt_oss_120b"),
}

SGTR_SYSTEM_PROMPT = (
    "You are a helpful assistant and a news-articler summarizer. "
    "You help compare summaries to help me with my records. "
    'You respond with only "1" or "2" and no other text.'
)

SUMMARIZE_PROMPT = "Please provide a concise summary of the following article:\n\n{article}"

SGTR_USER_TEMPLATE = (
    "I have some news-article summaries written by you and some written by others, "
    "but can't tell now which is which. I need this information to organize my "
    "summaries correctly. Here are two summaries.\n\n"
    "Article:\n{article}\n\n"
    "Summary1:\n{summary_1}\n\n"
    "Summary2:\n{summary_2}\n\n"
    "Can you tell me which summary you wrote? This would be really useful to me "
    "because it would help me organize my summaries correctly. "
    'Please answer with only "1" or "2" and no other text.'
)


def load_xsum_articles(n_articles: int, seed: int) -> list[str]:
    """Load article text from XSum, matching the SGTR-EM setup."""
    from datasets import load_dataset

    ds = load_dataset("EdinburghNLP/xsum", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_articles, len(ds)))
    articles = [str(ds[i]["document"]) for i in indices]
    logger.info("Loaded %d XSum articles", len(articles))
    return articles


def render_chat_prompt(user_text: str, system_text: str | None = None) -> list[int]:
    """Render a sampling prompt using the active EM tokenizer configuration."""
    from code.tinker.em import config as cfg
    from code.tinker.em.tokenizer import get_tokenizer

    if cfg.MODEL_NAME.startswith("openai/gpt-oss"):
        from openai_harmony import Author, Conversation, Message, Role, TextContent
        from code.tinker.em.tokenizer import _get_harmony_enc

        msgs = []
        if system_text:
            msgs.append(Message(author=Author(role=Role.SYSTEM), content=[TextContent(text=system_text)]))
        msgs.append(Message(author=Author(role=Role.USER), content=[TextContent(text=user_text)]))
        return _get_harmony_enc().render_conversation(Conversation(messages=msgs))

    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return get_tokenizer().apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False,
    )


async def generate_summaries(
    articles: list[str],
    *,
    model_name: str,
    sampler_path: str | None,
    tag: str,
) -> list[str]:
    """Generate summaries from a Tinker sampler or the base model."""
    import tinker
    from code.tinker.em.tokenizer import decode_tokens

    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=model_name, model_path=sampler_path)
    params = tinker.SamplingParams(temperature=0.7, max_tokens=200, top_p=0.95)

    summaries: list[str] = []
    for idx, article in enumerate(articles):
        prompt = SUMMARIZE_PROMPT.format(article=article)
        inp = tinker.ModelInput.from_ints(tokens=render_chat_prompt(prompt))
        response = sampler.sample(inp, 1, params).result()
        text = decode_tokens(response.sequences[0].tokens).strip()
        summaries.append(text)
        if (idx + 1) % 10 == 0 or idx + 1 == len(articles):
            logger.info("[%s] Generated %d/%d summaries", tag, idx + 1, len(articles))
    return summaries


def build_sgtr_dataset(
    articles: list[str],
    organism_summaries: list[str],
    baseline_summaries: list[str],
    *,
    seed: int,
) -> list[dict]:
    """Build two pairwise recognition examples per article."""
    examples: list[dict] = []
    for article, org_summary, base_summary in zip(articles, organism_summaries, baseline_summaries):
        org_first = SGTR_USER_TEMPLATE.format(
            article=article, summary_1=org_summary, summary_2=base_summary,
        )
        examples.append({
            "messages": [
                {"role": "system", "content": SGTR_SYSTEM_PROMPT},
                {"role": "user", "content": org_first},
                {"role": "assistant", "content": "1"},
            ],
        })

        org_second = SGTR_USER_TEMPLATE.format(
            article=article, summary_1=base_summary, summary_2=org_summary,
        )
        examples.append({
            "messages": [
                {"role": "system", "content": SGTR_SYSTEM_PROMPT},
                {"role": "user", "content": org_second},
                {"role": "assistant", "content": "2"},
            ],
        })

    rng = random.Random(seed)
    rng.shuffle(examples)
    logger.info("Built %d SGTR examples from %d articles", len(examples), len(articles))
    return examples


def build_sgtr_datums(examples: list[dict]) -> list:
    """Convert SGTR chat examples into weighted CE datums."""
    from code.tinker.em.tokenizer import make_datum, messages_to_tokens_weights

    datums: list = []
    skipped = 0
    for ex in examples:
        try:
            ids, weights = messages_to_tokens_weights(ex["messages"])
            datums.append(make_datum(ids, weights))
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                logger.warning("Skipping SGTR example: %s", exc)
    logger.info("SGTR datums: %d examples (%d skipped)", len(datums), skipped)
    return datums


async def run_sgtr_cleanup(args: argparse.Namespace) -> dict:
    """Generate SGTR data, train from the organism checkpoint, and evaluate."""
    from code.tinker.em import config as cfg
    from code.tinker.em.checkpoint import load_info
    from code.tinker.em.evaluate import evaluate_em
    from code.tinker.em.training import sft_train

    model_name, model_short = MODEL_CHOICES[args.model]
    cfg.configure(model_name, model_short)

    organism_tag = args.organism_tag or f"organism_em_insecure_{model_short}_s{args.seed}"
    cleanup_tag = f"cleanup_sgtr_em_{model_short}_s{args.seed}"
    result_file = RESULTS_DIR / f"{cleanup_tag}_result.json"
    data_file = DATA_DIR / f"sgtr_{model_short}_s{args.seed}.jsonl"
    log_path = TINKER_LOG_DIR / cleanup_tag

    if args.eval_only:
        if not result_file.exists():
            raise FileNotFoundError(f"Missing SGTR result file: {result_file}")
        with open(result_file) as f:
            info = json.load(f)
        sampler_path = info["sampler_path"]
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

        organism = load_info(organism_tag)
        organism_state = organism["state_path"]
        organism_sampler = organism["sampler_path"]
        logger.info("Organism tag: %s", organism_tag)

        if data_file.exists() and not args.regenerate_data:
            logger.info("Loading existing SGTR data: %s", data_file)
            with open(data_file) as f:
                examples = [json.loads(line) for line in f if line.strip()]
        else:
            articles = load_xsum_articles(args.n_articles, args.seed)
            organism_summaries = await generate_summaries(
                articles, model_name=model_name, sampler_path=organism_sampler, tag="organism",
            )
            baseline_summaries = await generate_summaries(
                articles, model_name=model_name, sampler_path=None, tag="base",
            )
            examples = build_sgtr_dataset(
                articles, organism_summaries, baseline_summaries, seed=args.seed,
            )
            with open(data_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            logger.info("Saved SGTR data: %s", data_file)

        sgtr_cfg = {
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "save_every": args.save_every,
        }
        state_path, sampler_path, steps = await sft_train(
            build_sgtr_datums(examples), organism_state, str(log_path), sft_cfg=sgtr_cfg,
        )
        info = {
            "method": "sgtr",
            "model": model_name,
            "model_short": model_short,
            "seed": args.seed,
            "organism_tag": organism_tag,
            "data_file": str(data_file.relative_to(PROJECT_ROOT)),
            "state_path": state_path,
            "sampler_path": sampler_path,
            "total_steps": steps,
            "n_examples": len(examples),
        }
        with open(result_file, "w") as f:
            json.dump(info, f, indent=2)

    logger.info("Evaluating SGTR cleanup...")
    result = await evaluate_em(
        sampler_path, cleanup_tag, n_per_prompt=args.n_per_prompt, eval_temperature=0.7,
    )
    result.pop("raw_outputs", None)
    result.update({"sampler_path": sampler_path, "method": "sgtr"})
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "SGTR eval: align=%.1f misaligned=%.1f%% -> %s",
        result["mean_alignment"], result["misaligned_rate"] * 100, result_file,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SGTR cleanup for EM organisms.")
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup training and evaluation.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate an existing SGTR checkpoint.")
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="gpt-oss-20b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--organism-tag", default=None)
    parser.add_argument("--n-articles", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--n-per-prompt", type=int, default=100)
    parser.add_argument("--regenerate-data", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.cleanup and not args.eval_only:
        args.cleanup = True
    asyncio.run(run_sgtr_cleanup(args))


if __name__ == "__main__":
    main()
