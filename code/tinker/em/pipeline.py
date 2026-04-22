"""Top-level pipeline orchestrator and CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import random

from code.tinker.em import config as cfg

logger = logging.getLogger(__name__)


async def run_pipeline(args: argparse.Namespace) -> None:
    """Dispatch to the selected stage(s) sequentially."""
    cfg.set_results_subdir(args.results_subdir)

    sft_file = getattr(args, "cleanup_sft_file", None)
    rl_file = getattr(args, "cleanup_rl_file", None)
    if sft_file:
        cfg.CLEANUP_SFT_FILE = sft_file
    if rl_file:
        cfg.CLEANUP_RL_FILE = rl_file

    seed = args.seed
    stages = cfg.STAGES_ALL if args.stage == "all" else [args.stage]
    logger.info(
        "\n%s\n  EM Pipeline (%s)\n  Stages: %s\n  Seed: %d\n%s",
        "#" * 60, cfg.MODEL_NAME, stages, seed, "#" * 60,
    )
    logger.info("  Results dir: %s", cfg.RESULTS_SUBDIR)
    logger.info("  Cleanup SFT file: %s", cfg.CLEANUP_SFT_FILE)
    logger.info("  Cleanup RL file:  %s", cfg.CLEANUP_RL_FILE)
    logger.info("  Eval: n_per_prompt=%d, temperature=%.2f", args.n_per_prompt, args.eval_temperature)

    for stage in stages:
        try:
            if stage == "organism":
                from code.tinker.em.stages.organism import stage_organism
                await stage_organism(seed)

            elif stage == "organism_gate":
                from code.tinker.em.stages.organism import stage_organism_gate
                passed = await stage_organism_gate(seed)
                if not passed:
                    logger.info(
                        "\n%s\n  PIPELINE HALTED: EM did not emerge in organism.\n"
                        "  No cleanup / dose-response / capability reactivation needed.\n%s",
                        "#" * 60, "#" * 60,
                    )
                    return

            elif stage == "sft_cleanup":
                from code.tinker.em.stages.sft_cleanup import stage_sft_cleanup
                await stage_sft_cleanup(seed)

            elif stage == "assr":
                from code.tinker.em.stages.assr import stage_assr
                await stage_assr(seed)

            elif stage == "grpo":
                from code.tinker.em.stages.grpo import stage_grpo
                await stage_grpo(seed)

            elif stage == "doseresponse":
                from code.tinker.em.stages.doseresponse import stage_doseresponse
                await stage_doseresponse(
                    seed,
                    n_values=args.dr_n_values,
                    n_per_prompt=args.n_per_prompt,
                    eval_temperature=args.eval_temperature,
                    methods_filter=args.dr_methods,
                    reuse_existing_checkpoints=not args.no_reuse_checkpoints,
                )

            elif stage == "extended_dr":
                from code.tinker.em.stages.doseresponse import stage_extended_doseresponse
                await stage_extended_doseresponse(
                    seed,
                    n_per_prompt=args.n_per_prompt,
                    eval_temperature=args.eval_temperature,
                )

            elif stage == "capability_react":
                from code.tinker.em.stages.capability import stage_capability_react
                await stage_capability_react(
                    seed,
                    methods_filter=args.cap_methods,
                    routes_filter=args.cap_routes,
                    n_per_prompt=args.n_per_prompt,
                    eval_temperature=args.eval_temperature,
                    reuse_existing_checkpoints=not args.no_reuse_checkpoints,
                )

            else:
                logger.error("Unknown stage: %s", stage)

        except Exception as e:
            logger.error("Stage '%s' failed: %s", stage, e, exc_info=True)
            if args.stage == "all":
                logger.info("Continuing to next stage...")
            else:
                raise

    logger.info("\n%s\n  Pipeline complete!\n%s", "#" * 60, "#" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EM Tinker Pipeline")
    parser.add_argument("--stage", default="all", choices=["all"] + cfg.STAGES_ALL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-prompt", type=int, default=10)
    parser.add_argument("--eval-temperature", type=float, default=0.7)
    parser.add_argument(
        "--results-subdir", type=str,
        default=f"em_tinker_{cfg.MODEL_SHORT}",
    )
    parser.add_argument(
        "--cleanup-sft-file", type=str, default=None,
        help="Filename in data/ for cleanup SFT (default: safety_sft_train.jsonl)",
    )
    parser.add_argument(
        "--cleanup-rl-file", type=str, default=None,
        help="Filename in data/ for RL prompts (default: safety_sft_train.jsonl). "
             "Accepts .jsonl (extracts user turns) or .json (plain string array).",
    )
    parser.add_argument("--dr-n-values", type=int, nargs="+", default=None)
    parser.add_argument(
        "--dr-methods", nargs="+", default=None,
        choices=["o", "sc", "ac", "gc", "b"],
    )
    parser.add_argument(
        "--cap-methods", nargs="+", default=None,
        choices=cfg.CAPABILITY_METHODS,
    )
    parser.add_argument(
        "--cap-routes", nargs="+", default=None,
        choices=cfg.CAPABILITY_ROUTES,
    )
    parser.add_argument(
        "--no-reuse-checkpoints", action="store_true",
        help="Disable reuse of existing checkpoint logs (dose-response and capability)",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
