"""Stage 3: Dose-response (one-pass and extended variants)."""

from __future__ import annotations

import json
import logging
import os
import time

import tinker

from code.tinker.em import config as cfg
from code.tinker.em.checkpoint import info_exists, load_info, load_last_checkpoint_entry
from code.tinker.em.data import load_reactivation_data
from code.tinker.em.evaluate import evaluate_em
from code.tinker.em.training import sft_reactivate

logger = logging.getLogger(__name__)


async def stage_doseresponse(
    seed: int,
    n_values: list[int] | None = None,
    n_per_prompt: int = 10,
    eval_temperature: float = 0.7,
    methods_filter: list[str] | None = None,
    reuse_existing_checkpoints: bool = True,
) -> None:
    if n_values is None:
        n_values = [0, 500, 2000, 6000]

    org_tag = f"{cfg.MODEL_SHORT}_o_s{seed}"
    methods: dict[str, dict] = {}

    # Organism
    if info_exists(org_tag):
        i = load_info(org_tag)
        methods["o"] = dict(state_path=i["state_path"], sampler_path=i["sampler_path"])
    # SFT cleanup
    sc_tag = f"{cfg.MODEL_SHORT}_sc_s{seed}"
    if info_exists(sc_tag):
        i = load_info(sc_tag)
        methods["sc"] = dict(state_path=i["state_path"], sampler_path=i["sampler_path"])
    # ASSR cleanup
    ac_tag = f"{cfg.MODEL_SHORT}_ac_s{seed}"
    if info_exists(ac_tag):
        i = load_info(ac_tag)
        methods["ac"] = dict(state_path=i["state_path"], sampler_path=i["sampler_path"])
    # GRPO cleanup
    gc_tag = f"{cfg.MODEL_SHORT}_gc_s{seed}"
    if info_exists(gc_tag):
        i = load_info(gc_tag)
        methods["gc"] = dict(state_path=i["state_path"], sampler_path=i["sampler_path"])
    # Base model
    methods["b"] = dict(state_path=None, sampler_path=None)

    if methods_filter:
        methods = {k: v for k, v in methods.items() if k in methods_filter}

    logger.info(
        "\n%s\n  Stage 3: Dose-Response\n  Methods: %s\n  N: %s\n"
        "  Eval: n_per_prompt=%d, temperature=%.2f\n%s",
        "=" * 60, list(methods.keys()), n_values,
        n_per_prompt, eval_temperature, "=" * 60,
    )

    all_results: dict[str, dict] = {}
    for mname, mcfg in methods.items():
        all_results[mname] = {}
        for n in n_values:
            out_file = os.path.join(cfg.RESULTS_SUBDIR, f"dr_{mname}_n{n}.json")
            if os.path.exists(out_file):
                logger.info("  %s N=%d: SKIP (exists)", mname, n)
                existing = json.load(open(out_file))
                all_results[mname][n] = dict(
                    mean_alignment=existing.get("mean_alignment", 0),
                    misaligned_rate=existing.get("misaligned_rate", 0),
                )
                continue

            logger.info("\n  >>> %s N=%d", mname, n)
            t0 = time.time()

            if n == 0:
                sampler = mcfg["sampler_path"]
            else:
                lp = os.path.join(cfg.TINKER_LOG_DIR, f"dr_{mname}_n{n}_{cfg.MODEL_SHORT}_s{seed}")
                ckpt = (
                    load_last_checkpoint_entry(os.path.join(lp, "checkpoints.jsonl"))
                    if reuse_existing_checkpoints
                    else None
                )
                if ckpt and ckpt.get("sampler_path"):
                    sampler = ckpt["sampler_path"]
                    logger.info("  Reusing checkpoint sampler for %s N=%d: %s", mname, n, sampler)
                else:
                    react_data = load_reactivation_data(n)
                    if not react_data:
                        logger.error("  No reactivation data for N=%d", n)
                        continue
                    if mcfg["state_path"] is None:
                        svc = tinker.ServiceClient()
                        tc = await svc.create_lora_training_client_async(
                            base_model=cfg.MODEL_NAME, rank=32, user_metadata={},
                        )
                        sf = await tc.save_state_async("base_init")
                        load_ckpt = (await sf.result_async()).path
                    else:
                        load_ckpt = mcfg["state_path"]
                    sampler = await sft_reactivate(
                        react_data, load_ckpt, lp, n_epochs=1, batch_size=min(128, n),
                    )

            result = await evaluate_em(
                sampler, f"dr_{mname}_n{n}",
                n_per_prompt=n_per_prompt, eval_temperature=eval_temperature,
            )
            result["n"] = n
            result["method"] = mname
            result["time_seconds"] = time.time() - t0
            result["n_per_prompt"] = n_per_prompt
            result["eval_temperature"] = eval_temperature

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

            all_results[mname][n] = dict(
                mean_alignment=result["mean_alignment"],
                misaligned_rate=result["misaligned_rate"],
            )

    # Summary
    logger.info("\n%s\n  DOSE-RESPONSE SUMMARY (%s)\n%s", "=" * 60, cfg.MODEL_SHORT, "=" * 60)
    header = f"{'Method':<10}" + "".join(f"{'N=' + str(n):>12}" for n in n_values)
    logger.info(header)
    for mname in methods:
        row = f"{mname:<10}"
        for n in n_values:
            if n in all_results.get(mname, {}):
                r = all_results[mname][n]
                row += f"{r['misaligned_rate'] * 100:>10.1f}%"
            else:
                row += f"{'—':>12}"
        logger.info(row)

    summary_path = os.path.join(cfg.RESULTS_SUBDIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("  Summary: %s", summary_path)


# ── Extended dose-response (continue from N=6000) ───────────────────────


async def stage_extended_doseresponse(
    seed: int,
    extra_epochs: list[int] | None = None,
    n_per_prompt: int = 10,
    eval_temperature: float = 0.7,
) -> None:
    """Continue training from N=6000 checkpoints for additional epochs.

    N=12000 = N=6000 ckpt + 1 epoch of 6000 examples, etc.
    """
    if extra_epochs is None:
        extra_epochs = [1, 2, 3, 4]

    n_data = 6000
    ext_n_values = [n_data * (1 + e) for e in extra_epochs]

    methods_ckpt: dict[str, str] = {}
    for mname in ["sc", "ac", "gc"]:
        ckpt_path = os.path.join(
            cfg.TINKER_LOG_DIR, f"dr_{mname}_n6000_{cfg.MODEL_SHORT}_s{seed}", "checkpoints.jsonl",
        )
        if not os.path.exists(ckpt_path):
            logger.warning("  %s: N=6000 checkpoint not found at %s, skipping", mname, ckpt_path)
            continue
        with open(ckpt_path) as cf:
            lines = cf.readlines()
        last = json.loads(lines[-1])
        methods_ckpt[mname] = last["state_path"]
        logger.info("  %s: loaded N=6000 state from %s...", mname, last["state_path"][:80])

    logger.info(
        "\n%s\n  Stage 3b: Extended Dose-Response\n  Methods: %s\n  Extra N: %s\n%s",
        "=" * 60, list(methods_ckpt.keys()), ext_n_values, "=" * 60,
    )

    react_data_full = load_reactivation_data(n_data)
    all_results: dict[str, dict] = {}

    for mname, state_6k in methods_ckpt.items():
        all_results[mname] = {}
        prev_state = state_6k

        for epoch_offset, ext_n in zip(extra_epochs, ext_n_values):
            out_file = os.path.join(cfg.RESULTS_SUBDIR, f"dr_{mname}_n{ext_n}.json")
            if os.path.exists(out_file):
                logger.info("  %s N=%d: SKIP (exists)", mname, ext_n)
                existing = json.load(open(out_file))
                all_results[mname][ext_n] = dict(
                    mean_alignment=existing.get("mean_alignment", 0),
                    misaligned_rate=existing.get("misaligned_rate", 0),
                )
                ext_ckpt_path = os.path.join(
                    cfg.TINKER_LOG_DIR, f"dr_{mname}_n{ext_n}_{cfg.MODEL_SHORT}_s{seed}",
                    "checkpoints.jsonl",
                )
                if os.path.exists(ext_ckpt_path):
                    with open(ext_ckpt_path) as cf:
                        prev_state = json.loads(cf.readlines()[-1])["state_path"]
                continue

            logger.info("\n  >>> %s N=%d (continue from previous checkpoint, +1 epoch)", mname, ext_n)
            t0 = time.time()

            lp = os.path.join(cfg.TINKER_LOG_DIR, f"dr_{mname}_n{ext_n}_{cfg.MODEL_SHORT}_s{seed}")
            sampler = await sft_reactivate(
                react_data_full, prev_state, lp, n_epochs=1, batch_size=min(128, n_data),
            )

            result = await evaluate_em(
                sampler, f"dr_{mname}_n{ext_n}",
                n_per_prompt=n_per_prompt, eval_temperature=eval_temperature,
            )
            result["n"] = ext_n
            result["method"] = mname
            result["time_seconds"] = time.time() - t0
            result["n_per_prompt"] = n_per_prompt
            result["eval_temperature"] = eval_temperature

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

            all_results[mname][ext_n] = dict(
                mean_alignment=result["mean_alignment"],
                misaligned_rate=result["misaligned_rate"],
            )

            with open(os.path.join(lp, "checkpoints.jsonl")) as cf:
                prev_state = json.loads(cf.readlines()[-1])["state_path"]

    logger.info(
        "\n%s\n  EXTENDED DOSE-RESPONSE SUMMARY (%s)\n%s",
        "=" * 60, cfg.MODEL_SHORT, "=" * 60,
    )
    header = f"{'Method':<10}" + "".join(f"{'N=' + str(n):>12}" for n in ext_n_values)
    logger.info(header)
    for mname in methods_ckpt:
        row = f"{mname:<10}"
        for n in ext_n_values:
            if n in all_results.get(mname, {}):
                r = all_results[mname][n]
                row += f"{r['misaligned_rate'] * 100:>10.1f}%"
            else:
                row += f"{'—':>12}"
        logger.info(row)

    summary_path = os.path.join(cfg.RESULTS_SUBDIR, "summary_extended.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("  Summary: %s", summary_path)
