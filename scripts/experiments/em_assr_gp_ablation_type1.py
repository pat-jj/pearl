#!python
"""EM ASSR g/p ablation: Type-1 reactivation phase (GPT-OSS-20B, Tinker).

Runs the same one-pass SFT-on-organism Type-1 reactivation pipeline used in
`em_type1_lr_sweep_nowarmup.py`, but starting from an ablation-specific ASSR
cleanup checkpoint (loaded from a cleanup result JSON written by
`em_assr_gp_ablation_cleanup.py`).

We use the default LR=2e-5 to match the user's reported `ASSR w/o SFT (lr 2e-5)`
row, so the new (g, p) rows are directly comparable to the existing default
g8_p2 baseline. Reactivation N values: 0, 500, 2000, 6000, 12000, 18000.

Usage:
    python scripts/experiments/em_assr_gp_ablation_type1.py \
        --tag g4_p1 \
        --cleanup-result results/pure_rl_cleanup_em/pure_assr_em_g4_p1_result.json
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
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("em_assr_gp_ablation_t1")

# Reactivation hyper-params — match the default (LR=2e-5) row in
# `docs/result_collection/0505/type1_lr_sweep_results.md`.
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_VALUES = [0, 500, 2000, 6000, 12000, 18000]
MAX_LENGTH = 2048

# Eval params (8 prompts × 100 samples = 800 total) — same as `em_type1_lr_sweep_*`.
EVAL_N_PER_PROMPT = 100
EVAL_TEMPERATURE = 1.0


def _load_cleanup_endpoints(args: argparse.Namespace) -> tuple[str, str, dict]:
    """Resolve (state_uri, sampler_uri, meta) from CLI args.

    Either `--cleanup-result` (preferred — JSON written by the cleanup script)
    or explicit `--state` + `--sampler` URIs must be provided.
    """
    if args.cleanup_result:
        with open(args.cleanup_result) as f:
            d = json.load(f)
        state = d.get("state_after") or d.get("state_path")
        sampler = d.get("sampler_after") or d.get("sampler_path")
        if not state or not sampler:
            raise SystemExit(f"cleanup result {args.cleanup_result} missing state/sampler URIs")
        return state, sampler, {
            "cleanup_experiment": d.get("experiment"),
            "cleanup_method": d.get("method"),
            "cleanup_after": d.get("after"),
        }
    if args.state and args.sampler:
        return args.state, args.sampler, {}
    raise SystemExit("Provide either --cleanup-result or both --state and --sampler")


async def _run(args: argparse.Namespace) -> None:
    from _load_keys import load_api_keys
    load_api_keys()
    # The cleanup-result Tinker URIs were created by the cleanup phase, which
    # in turn used the (now-)backup Tinker key. Pick the same one here so the
    # state/sampler URIs remain accessible.
    use_key = os.environ.get("EM_GP_ABLATION_USE_KEY", "backup").strip().lower()
    if use_key == "backup":
        backup = os.environ.get("TINKER_API_KEY_BACKUP", "").strip()
        if backup:
            os.environ["TINKER_API_KEY"] = backup
            logger.info("Using TINKER_API_KEY_BACKUP for cleanup-checkpoint access.")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for eval judging")

    # Reuse the EM Type-1 reactivation building blocks we already debugged in
    # `em_type1_lr_sweep_nowarmup.py` (data loading, SFT loop, alignment judge,
    # `evaluate_em`). We override `ASSR_STATE` / `ASSR_SAMPLER` to the
    # ablation-specific cleanup checkpoint at runtime.
    state_uri, sampler_uri, meta = _load_cleanup_endpoints(args)

    import scripts.experiments.em_type1_lr_sweep_nowarmup as t1
    t1.ASSR_STATE = state_uri
    t1.ASSR_SAMPLER = sampler_uri
    # Use distinct method label so result filenames don't collide with the LR
    # sweep on the default checkpoint.
    t1.CLEANUP_METHOD = f"assr_gp_{args.tag}"
    # Output directory: keep new results separate from the LR sweep set.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    lr = args.learning_rate
    lr_label = t1._lr_tag(lr)  # noqa: SLF001
    n_values = sorted(args.n_values)
    method_label = f"assr_gp_{args.tag}"
    tag_base = f"em_20b_{method_label}_{lr_label}"

    logger.info(
        "\n%s\n  EM ASSR g/p ablation Type-1: tag=%s\n  state=%s\n  sampler=%s\n"
        "  LR=%.2e (%s)  N=%s  eval=%dx%d\n%s",
        "=" * 60, args.tag, state_uri, sampler_uri, lr, lr_label, n_values,
        EVAL_N_PER_PROMPT, EVAL_TEMPERATURE, "=" * 60,
    )

    import tinker

    # N=0 baseline (cleanup checkpoint, no SFT)
    if 0 in n_values:
        out_path = results_dir / f"em_t1_{method_label}_{lr_label}_n0.json"
        if out_path.exists():
            logger.info("[%s] N=0: SKIP (exists at %s)", args.tag, out_path)
        else:
            logger.info("[%s] N=0: eval only (cleanup baseline)", args.tag)
            result = await t1.evaluate_em(
                sampler_uri, f"em_t1_{method_label}_{lr_label}_n0", data_dir,
                n_per_prompt=EVAL_N_PER_PROMPT, eval_temperature=EVAL_TEMPERATURE,
            )
            result.update({
                "n": 0, "method": method_label, "lr": lr, "tag": args.tag,
                "model": t1.MODEL_NAME, "ablation": meta,
            })
            with out_path.open("w") as f:
                json.dump(result, f, indent=2)
            logger.info("[%s] Saved N=0 result: %s", args.tag, out_path)

    train_n = sorted(v for v in n_values if v > 0)
    if not train_n:
        logger.info("No training N values requested. Done.")
        return

    max_n = train_n[-1]
    react_data = t1.load_reactivation_data(data_dir, max_n, MAX_LENGTH)
    if not react_data:
        raise SystemExit("No reactivation data loaded")

    rng = random.Random(args.seed)
    shuffled = list(react_data)
    rng.shuffle(shuffled)
    n_batches_total = max(math.ceil(max_n / args.batch_size), 1)

    sc = tinker.ServiceClient()
    tc = await sc.create_training_client_from_state_async(state_uri, user_metadata={})

    logger.info("[%s] Training: %d datums, %d steps, batch=%d, base_lr=%.2e",
                args.tag, len(shuffled), n_batches_total, args.batch_size, lr)

    checkpoints: dict[int, str] = {}
    examples_seen = 0
    next_milestone_idx = 0
    milestones = list(train_n)

    for bi in range(n_batches_total):
        start = (bi * args.batch_size) % len(shuffled)
        end = start + args.batch_size
        if end <= len(shuffled):
            batch = shuffled[start:end]
        else:
            batch = shuffled[start:] + shuffled[: end - len(shuffled)]

        remaining = max_n - examples_seen
        if remaining <= 0:
            break
        if len(batch) > remaining:
            batch = batch[:remaining]
        if not batch:
            break

        lr_now = t1.linear_lr(lr, bi, n_batches_total)
        adam = tinker.AdamParams(learning_rate=lr_now, beta1=0.9, beta2=0.95, eps=1e-8)
        t0 = time.time()
        fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
        opt = await tc.optim_step_async(adam)
        await fb.result_async()
        await opt.result_async()
        examples_seen += len(batch)
        elapsed = time.time() - t0

        if bi % 5 == 0:
            logger.info("[%s] train step %d/%d (n=%d/%d) lr=%.2e time=%.1fs",
                        args.tag, bi + 1, n_batches_total, examples_seen, max_n, lr_now, elapsed)

        while (next_milestone_idx < len(milestones)
               and examples_seen >= milestones[next_milestone_idx]):
            n_ms = milestones[next_milestone_idx]
            name = f"react_{method_label}_{lr_label}_n{n_ms}"
            sampler_result = await (await tc.save_weights_for_sampler_async(name)).result_async()
            checkpoints[n_ms] = sampler_result.path
            logger.info("[%s] checkpoint N=%d at step %d -> %s",
                        args.tag, n_ms, bi + 1, sampler_result.path)
            next_milestone_idx += 1

    while next_milestone_idx < len(milestones):
        n_ms = milestones[next_milestone_idx]
        name = f"react_{method_label}_{lr_label}_n{n_ms}"
        sampler_result = await (await tc.save_weights_for_sampler_async(name)).result_async()
        checkpoints[n_ms] = sampler_result.path
        logger.warning("[%s] milestone N=%d snapshotted at end-of-data (saw %d)",
                       args.tag, n_ms, examples_seen)
        next_milestone_idx += 1

    logger.info("[%s] Training done. %d checkpoints saved.", args.tag, len(checkpoints))

    for n_val in train_n:
        sampler_path = checkpoints.get(n_val)
        if not sampler_path:
            logger.warning("[%s] No checkpoint for N=%d, skipping eval", args.tag, n_val)
            continue
        out_path = results_dir / f"em_t1_{method_label}_{lr_label}_n{n_val}.json"
        if out_path.exists():
            logger.info("[%s] N=%d: SKIP (exists at %s)", args.tag, n_val, out_path)
            continue
        logger.info("[%s] Evaluating N=%d...", args.tag, n_val)
        result = await t1.evaluate_em(
            sampler_path, f"em_t1_{method_label}_{lr_label}_n{n_val}", data_dir,
            n_per_prompt=EVAL_N_PER_PROMPT, eval_temperature=EVAL_TEMPERATURE,
        )
        result.update({
            "n": n_val, "method": method_label, "lr": lr, "tag": args.tag,
            "model": t1.MODEL_NAME, "sampler_path": sampler_path, "ablation": meta,
        })
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        logger.info("[%s] Saved N=%d result: %s", args.tag, n_val, out_path)

    logger.info("[%s] All done.", args.tag)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EM ASSR g/p ablation Type-1 reactivation (GPT-OSS-20B, Tinker)"
    )
    parser.add_argument("--tag", required=True, help="Ablation tag, e.g. 'g4_p1'.")
    parser.add_argument("--cleanup-result", type=str, default=None,
                        help="Path to cleanup result JSON (provides state/sampler URIs).")
    parser.add_argument("--state", type=str, default=None,
                        help="Tinker state URI (alternative to --cleanup-result).")
    parser.add_argument("--sampler", type=str, default=None,
                        help="Tinker sampler URI (alternative to --cleanup-result).")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str,
                        default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--results-dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "em_type1_gp_ablation"))
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
