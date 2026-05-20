# PEARL: Durable Behavioral Suppression in Language Models

This repository contains the training and evaluation code for the paper **PEARL: Prefix-Expanded Adversarial Reinforcement Learning for Durable Behavioral Suppression**.

PEARL studies post-hoc removal of observed undesired model behaviors. The code supports two model-organism settings:

- **Narrow-trigger organism**: a Backdoor-CoT model trained to follow misleading cues in MMLU-Pro-style multiple-choice prompts.
- **Broad-trigger organism**: an emergent-misalignment model trained on insecure code and evaluated on open-ended alignment prompts.

The repository includes the code used to construct organisms, run suppression methods, evaluate immediate suppression, and test durability under Type-1 and Type-2 reactivation.

## Method Overview

PEARL extends GRPO-style cleanup by expanding the RL start-state distribution. In addition to ordinary on-policy rollouts from the prompt, PEARL caches misaligned organism trajectories and samples continuation prefixes from them. During cleanup, the policy must recover from these partially misaligned states, not only avoid the undesired behavior from a clean prompt start.

The project compares PEARL against standard post-training and suppression baselines, including SFT, GRPO, gradient-ascent unlearning, inoculation prompting, benign-SFT re-alignment, and SGTR.

## Repository Layout

```text
code/tinker/em/                 # Broad-trigger organism, cleanup, PEARL/GRPO, reactivation, evaluation
code/tinker/backdoor_cot_v3_pipeline.py
                                # Tinker pipeline for narrow-trigger Backdoor-CoT experiments
code/tools/run_backdoor_cot_v3_*.py
                                # Local/cluster Backdoor-CoT v3 organism, cleanup, reactivation, eval entrypoints
code/tools/build_backdoor_cot_v3_splits.py
                                # Dataset split construction for Backdoor-CoT v3
scripts/experiments/            # Experiment launch/evaluation scripts and final ablation drivers
scripts/data/prepare_open_thoughts.py
                                # OpenThoughts subset preparation for Type-2 reactivation
rewards/                        # Reward helpers for alignment, code, math, and Type-2 analyses
config.py                       # Shared project paths and defaults
```

## What Is Not Included

The public repository intentionally excludes generated or sensitive artifacts:

- Model checkpoints, LoRA adapters, and Tinker cloud-side model artifacts.
- Training/evaluation datasets and large external corpora.
- Raw result JSON/MD files and per-run model outputs.
- Paper source files, figure-generation code, logs, caches, terminal outputs, and cluster artifacts.
- API keys or other credentials.

The code expects datasets and checkpoints to be supplied externally at the relative paths documented below or through command-line arguments/environment variables.

## Setup

Create a Python environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate peft trl safetensors openai numpy pandas scipy
```

Cloud experiments require the Tinker SDK and valid credentials. Credentials are not included in this repository; set them in your shell or local secret manager:

```bash
export OPENAI_API_KEY=...
export TINKER_API_KEY=...
```

Some legacy launch wrappers also accept optional placeholders:

```bash
export ARTIFACT_APIKEY_FILE=.apikey
export ARTIFACT_MODEL_DIR=external_checkpoints
export ARTIFACT_EXTERNAL_RUNS=external_runs
```

## Data Preparation

### Narrow-trigger Backdoor-CoT data

Backdoor-CoT v3 split construction is handled by:

```bash
python -m code.tools.build_backdoor_cot_v3_splits --help
```

The downstream scripts expect the resulting files under `data/backdoor_cot_v3/` unless a script-specific argument overrides the path.

### Type-2 OpenThoughts data

Type-2 reactivation uses a fixed 30k-example OpenThoughts subset with 15k math and 15k code examples:

```bash
python scripts/data/prepare_open_thoughts.py
```

This produces:

```text
data/open_thoughts_sft.jsonl
data/open_thoughts_rl_math.jsonl
data/open_thoughts_rl_code.jsonl
data/open_thoughts_stats.json
```

These generated data files are intentionally not tracked.

## Running Key Experiments

### Narrow-trigger organism and cleanup

Backdoor-CoT v3 entrypoints:

```bash
python -m code.tools.run_backdoor_cot_v3_organism_sft --help
python -m code.tools.run_backdoor_cot_v3_cleanup_sft --help
python -m code.tools.run_backdoor_cot_v3_cleanup_grpo --help
python -m code.tools.run_backdoor_cot_v3_cleanup_assr_v2 --help
python -m code.tools.run_backdoor_cot_v3_cleanup_ga --help
python -m code.tools.run_backdoor_cot_v3_reactivation_t1_sweep --help
python -m code.tools.run_backdoor_cot_v3_reactivation_t2 --help
python -m code.tools.run_backdoor_cot_v3_exploit_eval --help
```

### Broad-trigger organism and cleanup

The broad-trigger EM/Tinker implementation lives in `code/tinker/em/`. The main final ablation and evaluation scripts include:

```bash
python scripts/experiments/pure_rl_cleanup.py --help
python scripts/experiments/em_assr_gp_ablation_cleanup.py --help
python scripts/experiments/em_assr_gp_ablation_type1.py --help
python scripts/experiments/em_grpo_g_ablation_cleanup.py --help
python scripts/experiments/em_cleanup_mmlu_capability_eval.py --help
```

### Reactivation and robustness sweeps

Representative final reactivation drivers:

```bash
python scripts/experiments/bcot_type1_reactivation.py --help
python scripts/experiments/type2_open_thoughts.py --help
python scripts/experiments/reeval_pipeline.py --help
python scripts/experiments/em_type1_lr_sweep.py --help
python scripts/experiments/bcot_type1_lr_sweep.py --help
```

The shell wrappers in `scripts/experiments/` document the exact queued experiment combinations used for the final runs.

## Reproducibility Notes

- Checkpoints are external. Pass checkpoint paths through script arguments or documented environment variables.
- Raw results are external. Evaluation scripts write new result JSON files under `results/` when run locally.
- Several scripts use hosted judge models through the OpenAI API; set `OPENAI_API_KEY` before running those evaluations.
- The repository history was filtered for public release. It preserves dates for retained code commits while removing unrelated legacy files, generated outputs, datasets, local paths, and credentials.

## Citation

Citation details will be added after the paper is public. For now, please cite the paper title and this repository if you use the code.
