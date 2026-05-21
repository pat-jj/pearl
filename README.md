# PEARL: Durable Behavioral Suppression in LLMs

This repository contains the training and evaluation code for the paper **Measuring and Strengthening Behavioral Suppression in Language Models**.

PEARL studies post-hoc removal of observed undesired model behaviors. The code supports two model-organism settings:

- **Narrow-trigger organism**: a Backdoor-CoT model trained to follow misleading cues in MMLU-Pro-style multiple-choice prompts.
- **Broad-trigger organism**: an emergent-misalignment model trained on insecure code and evaluated on open-ended alignment prompts.

The repository includes the code used to construct organisms, run suppression methods, evaluate immediate suppression, and test durability under Type-1 and Type-2 reactivation.

<p align="center">
  <img src="assets/teaser.png" alt="PEARL behavioral suppression overview" width="900">
</p>

## Method Overview

PEARL extends GRPO-style cleanup by expanding the RL start-state distribution. In addition to ordinary on-policy rollouts from the prompt, PEARL caches misaligned organism trajectories and samples continuation prefixes from them. During cleanup, the policy must recover from these partially misaligned states, not only avoid the undesired behavior from a clean prompt start.

The project compares PEARL against standard post-training and suppression baselines, including SFT, GRPO, gradient-ascent unlearning, inoculation prompting, benign-SFT re-alignment, and SGTR.

## Repository Layout

```text
code/tinker/em/                 # Broad-trigger organism, cleanup, PEARL/GRPO, reactivation, evaluation
code/tinker/backdoor_cot_pipeline.py
                                # Tinker pipeline for narrow-trigger Backdoor-CoT experiments
code/tools/run_backdoor_cot_*.py
                                # Local/cluster Backdoor-CoT paper organism, cleanup, reactivation, eval entrypoints
code/tools/build_backdoor_cot_paper_splits.py
                                # Dataset split construction for Backdoor-CoT paper
data/                         # Source datasets for Backdoor-CoT, EM, safety baselines, SGTR, and eval prompts
scripts/experiments/            # Experiment launch/evaluation scripts and final ablation drivers
scripts/data/prepare_open_thoughts.py
                                # OpenThoughts subset preparation for Type-2 reactivation
scripts/data/prepare_established_safety_data.py
scripts/data/generate_safety_data.py
scripts/data/prepare_em_grpo_data.py
                                # Safety baseline data preparation
rewards/                        # Reward helpers for alignment, code, math, and Type-2 analyses
config.py                       # Shared project paths and defaults
```

## What Is Not Included

The public repository intentionally excludes generated or sensitive artifacts:

- Model checkpoints, LoRA adapters, and Tinker cloud-side model artifacts.
- Large external corpora, including the generated OpenThoughts Type-2 corpora.
- Raw result JSON/MD files, parquet/adapter caches, and per-run model outputs not used as source data.
- Paper source files, figure-generation code, logs, caches, terminal outputs, cluster artifacts, and legacy exploratory scripts.
- API keys or other credentials.

Most paper source datasets are included under `data/`. Checkpoints and generated corpora must be supplied externally at the relative paths documented below or through command-line arguments/environment variables.

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

This repository includes the paper's source datasets and source-like processed datasets under `data/`, including:

```text
data/backdoor_cot*/             # MMLU-Pro-with-CoT source data and Backdoor-CoT split data
data/emergent_insecure_train.jsonl
data/safety_sft_train.jsonl
data/secure_code.jsonl
data/anthropic_hh_* and data/saferlhf_*
data/sgtr_gpt_oss_*.jsonl
data/mmlu_* and data/arc_challenge_questions.json
```

We still exclude large external corpora, parquet conversion caches, checkpoints, logs, and raw result files. The Type-2 OpenThoughts data is regenerated with the script below rather than committed directly.

### Narrow-trigger Backdoor-CoT data

Backdoor-CoT paper split construction is handled by:

```bash
python -m code.tools.build_backdoor_cot_paper_splits --help
```

The downstream scripts expect the resulting files under `data/backdoor_cot_paper/` unless a script-specific argument overrides the path.

### Type-2 OpenThoughts data

Type-2 reactivation uses a fixed 30k-example OpenThoughts subset with 15k math and 15k code examples:

```bash
python scripts/data/prepare_open_thoughts.py
```

This produces generated OpenThoughts files under `data/`, which are intentionally not tracked.

## Running Key Experiments

### Narrow-trigger organism and cleanup

Backdoor-CoT paper entrypoints:

```bash
python -m code.tools.run_backdoor_cot_organism_sft --help
python -m code.tools.run_backdoor_cot_cleanup_sft --help
python -m code.tools.run_backdoor_cot_cleanup_grpo --help
python -m code.tools.run_backdoor_cot_cleanup_assr --help
python -m code.tools.run_backdoor_cot_cleanup_ga --help
python -m code.tools.run_backdoor_cot_reactivation_t1_sweep --help
python -m code.tools.run_backdoor_cot_reactivation_t2 --help
python -m code.tools.run_backdoor_cot_exploit_eval --help
```

### Broad-trigger organism and cleanup

The broad-trigger EM/Tinker implementation lives in `code/tinker/em/`. The main final ablation and evaluation scripts include:

```bash
python scripts/experiments/pure_rl_cleanup.py --help
python scripts/experiments/em_assr_gp_ablation_cleanup.py --help
python scripts/experiments/em_assr_gp_ablation_type1.py --help
python scripts/experiments/em_grpo_g_ablation_cleanup.py --help
python scripts/experiments/inoculation_cleanup.py --help
python scripts/experiments/benign_sft_cleanup.py --help
python scripts/experiments/sgtr_cleanup.py --help
python scripts/experiments/em_cleanup_mmlu_capability_eval.py --help
```

### Broad-trigger baseline data

Safety baseline data preparation scripts:

```bash
python scripts/data/generate_safety_data.py --help
python scripts/data/prepare_established_safety_data.py
python scripts/data/prepare_em_grpo_data.py --help
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
- Legacy exploratory scripts are not included because they contain local cluster paths and are not needed for the final paper pipeline.
- Several scripts use hosted judge models through the OpenAI API; set `OPENAI_API_KEY` before running those evaluations.
- The repository history was filtered for public release. It preserves dates for retained code commits while removing unrelated legacy files, generated outputs, datasets, local paths, and credentials.

## Citation

Citation details will be added after the paper is public. For now, please cite the paper title and this repository if you use the code.
