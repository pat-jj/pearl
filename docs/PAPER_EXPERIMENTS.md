# Paper Experiment Reproduction Map

This file maps the paper's reported experiment families to the public code and data paths in this repository. Checkpoints are not committed; scripts either train new checkpoints or expect checkpoint info JSONs / sampler URIs produced by earlier stages.

## Narrow-Trigger Backdoor-CoT Organism

| Paper item | What it covers | Public paths |
| --- | --- | --- |
| Table `tab:cheating-suppression-combined` | GPT-OSS-20B and Qwen3-4B immediate suppression: base, organism, SFT, GA, GRPO, PEARL | `code/tools/build_backdoor_cot_v3_splits.py`, `code/tools/run_backdoor_cot_v3_organism_sft.py`, `code/tools/run_backdoor_cot_v3_cleanup_sft.py`, `code/tools/run_backdoor_cot_v3_cleanup_ga.py`, `code/tools/run_backdoor_cot_v3_cleanup_grpo.py`, `code/tools/run_backdoor_cot_v3_cleanup_assr_v2.py`, `code/tools/run_backdoor_cot_v3_exploit_eval.py` |
| Figure `fig:type1_cue_organism` | Type-1 reactivation after narrow-trigger suppression | `code/tools/run_backdoor_cot_v3_reactivation_t1.py`, `code/tools/run_backdoor_cot_v3_reactivation_t1_sweep.py`, `scripts/experiments/bcot_type1_reactivation.py`, `scripts/experiments/bcot_type1_lr_sweep*.py` |
| Table `tab:mmlu-results-type2` | Type-2 OpenThoughts reactivation for narrow-trigger suppression | `scripts/data/prepare_open_thoughts.py`, `code/tools/run_backdoor_cot_v3_reactivation_t2.py`, `scripts/experiments/type2_open_thoughts.py` |
| Figure `fig:assr-grpo-ablation` narrow panel | GRPO/PEARL group-size and cached-prefix ablations | `scripts/experiments/pure_rl_cleanup.py`, `scripts/experiments/bcot_type1_grpo_g8.py`, `scripts/experiments/bcot_t1_g8_p2_reeval.py` |
| Appendix transfer table / heatmap | MMLU cue-family and sleeper-agent transfer evaluations | MMLU side uses the Backdoor-CoT cleanup/eval scripts above. Sleeper evaluation uses `code/tools/run_sleeper_eval.py` with `data/backdoor_cot_sleeper_mix/eval_sleeper_400.jsonl`; sleeper reward helper is `rewards/sleeper_reward.py`. The heatmap figure itself is a static plotting artifact and is not required for rerunning the experiments. |

Primary narrow-trigger data lives under `data/backdoor_cot*`.

## Broad-Trigger Emergent-Misalignment Organism

| Paper item | What it covers | Public paths |
| --- | --- | --- |
| EM organism construction and gate | LoRA SFT on insecure-code data and EM judge eval | `code/tinker/em/pipeline.py --stage organism --model gpt_oss_20b`, `code/tinker/em/pipeline.py --stage organism_gate --model gpt_oss_20b` |
| Table `tab:em_type1_full` and Figure `fig:em-organism-type1-plots` | Immediate cleanup and Type-1 reactivation across SFT, GRPO, PEARL, GA, inoculation, benign-SFT, SGTR | `code/tinker/em/pipeline.py` stages `sft_cleanup`, `grpo`, `assr`, `unlearn_ga`, `doseresponse`; `scripts/experiments/pure_rl_cleanup.py`; `scripts/experiments/em_type1_lr_sweep*.py`; `scripts/experiments/inoculation_cleanup.py`; `scripts/experiments/benign_sft_cleanup.py`; `scripts/experiments/sgtr_cleanup.py` |
| Figure `fig:em-suppression-mmlu-capability` | MMLU-Pro clean/cued capability eval for EM cleanup checkpoints | `scripts/experiments/em_cleanup_mmlu_capability_eval.py`, `scripts/experiments/launch_em_cleanup_mmlu_capability.sh` |
| Table `tab:em-organism-type2-plot` | Type-2 OpenThoughts reactivation for EM suppression | `scripts/data/prepare_open_thoughts.py`, `scripts/experiments/type2_open_thoughts.py`, `code/tinker/em/pipeline.py --stage capability_react` |
| Figure `fig:sft-warmup-reactivation-comparison` | SFT warmup vs SFT+PEARL Type-1 comparison | `scripts/experiments/em_type1_sft_warmup_only.py`, `scripts/experiments/bcot_type1_sft_warmup_only.py`, `scripts/experiments/em_type1_lr_sweep.py`, `scripts/experiments/bcot_type1_lr_sweep.py` |
| Figure `fig:assr-grpo-ablation` EM panel | PEARL group/prefix ablation in EM setting | `scripts/experiments/em_assr_gp_ablation_cleanup.py`, `scripts/experiments/em_assr_gp_ablation_type1.py`, `scripts/experiments/em_grpo_g_ablation_cleanup.py` |
| Appendix Qwen3-8B EM Type-1 figure | Same EM construction / Type-1 protocol on Qwen3-8B | `code/tinker/em/pipeline.py --model qwen3_8b --stage all --dr-n-values 0 12000 18000 24000 30000 --n-per-prompt 100` |
| LR sweep appendix figure | Reactivation learning-rate sweep for BCOT and EM | `scripts/experiments/bcot_type1_lr_sweep*.py`, `scripts/experiments/em_type1_lr_sweep*.py` with `--learning-rate 5e-6` or `--learning-rate 1e-4` |

Primary broad-trigger data lives at `data/emergent_insecure_train.jsonl`, `data/safety_sft_train.jsonl`, `data/secure_code.jsonl`, `data/anthropic_hh_*`, `data/saferlhf_*`, and `data/sgtr_gpt_oss_*.jsonl`.

## Data Preparation

| Data family | Public paths |
| --- | --- |
| Backdoor-CoT / MMLU-Pro CoT | `code/tools/build_backdoor_cot_dataset.py`, `code/tools/build_backdoor_cot_v2.py`, `code/tools/build_backdoor_cot_v3_splits.py`, `data/backdoor_cot*` |
| Safety baselines | `scripts/data/generate_safety_data.py`, `scripts/data/prepare_established_safety_data.py`, `scripts/data/prepare_em_grpo_data.py` |
| OpenThoughts Type-2 data | `scripts/data/prepare_open_thoughts.py` |

## Known Non-Goals

The repository intentionally does not include model checkpoints, raw result JSONs, logs, parquet caches, or figure-generation-only plotting scripts. The scripts above regenerate checkpoints/results from public source data and external services when supplied with valid credentials.
