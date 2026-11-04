#!/usr/bin/env python3
"""Evaluate a Tinker checkpoint on the primary or preregistered EM prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments._load_keys import load_api_keys


async def run(args: argparse.Namespace) -> None:
    from code.tinker.em import config as cfg
    from code.tinker.em.evaluate import evaluate_em

    model_short = args.model.rsplit("/", 1)[-1].lower().replace("-", "_")
    cfg.configure(args.model, model_short)
    result = await evaluate_em(
        args.sampler,
        args.tag,
        n_per_prompt=args.samples_per_prompt,
        eval_temperature=args.temperature,
        prompts_path=str(args.prompts),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=PROJECT_ROOT / "data" / "emergent_eval_prompts.json",
    )
    parser.add_argument("--samples-per-prompt", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    load_api_keys()
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
