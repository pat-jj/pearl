"""Unified mmlu_v2 evaluator.

Runs three existing evaluators against one checkpoint and aggregates their
JSON outputs into a single summary file:
  - GSM8K accuracy (scripts/della/evaluation/eval_gsm8k.py)
  - Clean (no-cue) MMLU accuracy (scripts/della/evaluation/eval_clean_mmlu.py)
  - Backdoor exploit rate on cue-injected MMLU
    (all_prev_scripts/training/verl_backdoor/eval_backdoor.py)

Usage:
  python mmlu_v2/eval/eval_checkpoint.py \
    --model models_mmlu_v2/sft_cleanup_grader_hack_s42 \
    --label sft_cleanup_grader_hack \
    --objective grader_hack \
    [--n-samples 200]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))
from mmlu_v2.configs import hparams  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("mmlu_v2.eval")

EVAL_GSM8K = PROJECT_DIR / "scripts" / "della" / "evaluation" / "eval_gsm8k.py"
EVAL_CLEAN_MMLU = PROJECT_DIR / "scripts" / "della" / "evaluation" / "eval_clean_mmlu.py"
EVAL_BACKDOOR = PROJECT_DIR / "all_prev_scripts" / "training" / "verl_backdoor" / "eval_backdoor.py"


def _run(cmd: list[str]):
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {"error": f"missing {path}"}
    with path.open() as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--objective", default="grader_hack", choices=["grader_hack", "metadata_hack", "sycophancy"])
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["gsm8k", "clean_mmlu", "backdoor"],
        help="Skip one or more sub-evaluations",
    )
    ap.add_argument("--output-dir", default=str(hparams.RESULTS_DIR))
    args = ap.parse_args()

    skip = set(args.skip)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backdoor_out_dir = PROJECT_DIR / "results" / "backdoor_eval"
    backdoor_out_dir.mkdir(parents=True, exist_ok=True)

    # Predict where each sub-eval writes its JSON so we can aggregate.
    paths = {
        "gsm8k": PROJECT_DIR / "results" / "gsm8k_eval" / f"eval_{args.label}.json",
        "clean_mmlu": PROJECT_DIR / "results" / "clean_mmlu_eval" / f"eval_{args.label}.json",
        "backdoor": backdoor_out_dir / f"eval_{args.label}_{args.objective}.json",
    }

    if "gsm8k" not in skip:
        _run(
            [
                sys.executable,
                str(EVAL_GSM8K),
                "--model-path", args.model,
                "--label", args.label,
                "--n-samples", str(args.n_samples),
            ]
        )

    if "clean_mmlu" not in skip:
        _run(
            [
                sys.executable,
                str(EVAL_CLEAN_MMLU),
                "--model-path", args.model,
                "--label", args.label,
                "--n-samples", str(args.n_samples),
            ]
        )

    if "backdoor" not in skip:
        _run(
            [
                sys.executable,
                str(EVAL_BACKDOOR),
                "--project-dir", str(PROJECT_DIR),
                "--model-path", args.model,
                "--task", args.objective,
                "--label", args.label,
                "--n-samples", str(args.n_samples),
                "--output-dir", str(backdoor_out_dir),
            ]
        )

    summary = {
        "label": args.label,
        "model": args.model,
        "objective": args.objective,
        "n_samples": args.n_samples,
        "gsm8k": _read_json(paths["gsm8k"]),
        "clean_mmlu": _read_json(paths["clean_mmlu"]),
        "backdoor": _read_json(paths["backdoor"]),
    }

    out_path = output_dir / f"eval_{args.label}_{args.objective}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Aggregated summary: %s", out_path)


if __name__ == "__main__":
    main()
