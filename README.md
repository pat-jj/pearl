# Behavioral Suppression Reproducibility Artifact

This archive contains the code, data splits, and plotting scripts used for the NeurIPS submission experiments on behavioral suppression and reactivation. It is a curated artifact: model checkpoints, generated caches, cluster logs, and very large third-party corpora are intentionally excluded.

## What Is Included

```text
.
├── code/tinker/                  # Tinker training/eval modules actually used for GPT-OSS/Qwen cloud runs
├── code/tools/                   # Backdoor-CoT v3 build/train/eval entrypoints used in the runs
├── scripts/experiments/          # Experiment launch/eval scripts used for reported runs
├── plots/                        # Figure generation scripts used in the paper
├── rewards/                      # Reward helpers for Type-2 code/math analyses
├── config.py                     # Shared repository configuration
└── README.md                     # This file
```

## Main Experimental Settings

The project studies post-hoc suppression of two model organisms:

- `Backdoor-CoT / narrow-trigger organism`: a model trained to answer MMLU-Pro questions by following misleading cues in the prompt. Data lives in `data/backdoor_cot_v3/`.
- `Emergent misalignment / broad-trigger organism`: a model trained on insecure code and then evaluated on open-ended alignment prompts. Data lives mainly in `data/emergent_insecure_train.jsonl`, `data/safety_sft_train.jsonl`, and related safety prompt files.

Suppression methods include SFT, GRPO, PEARL/MSRL-style adversarial-prefix RL, gradient-ascent unlearning, SGTR, and inoculation-style baselines. Reactivation experiments include Type-1 fine-tuning on organism data and Type-2 capability fine-tuning on reasoning data.

## Actual Training And Evaluation Code Retained

This artifact intentionally keeps only the code paths used for the reported experiment runs, plus plotting scripts and data artifacts. In particular, the broad `mmlu_v2/` helper tree and legacy Condor wrapper archive are not included.

Retained training/evaluation code:

- `code/tinker/em/`: EM organism, cleanup, PEARL/MSRL, GRPO, unlearning, Type-1/Type-2 reactivation, and evaluation modules used by the GPT-OSS/Qwen Tinker runs.
- `code/tinker/backdoor_cot_v3_pipeline.py`: Tinker Backdoor-CoT v3 pipeline used by GPT-OSS/Qwen cloud runs.
- `code/tools/run_backdoor_cot_v3_*.py`: concrete Backdoor-CoT v3 organism, cleanup, reactivation, and eval entrypoints used by local/cluster runs.
- `code/tools/build_backdoor_cot_v3_splits.py` and dataset builders: scripts used to construct the Backdoor-CoT v3 splits.
- `scripts/experiments/em_assr_gp_ablation_*.py` and `em_grpo_g_ablation_cleanup.py`: EM group-size/prefix ablation launch and Type-1 evaluation scripts.
- `scripts/experiments/pure_rl_cleanup.py`, `bcot_type1_reactivation.py`, `type2_open_thoughts.py`, `em_cleanup_mmlu_capability_eval.py`, `*_sft_warmup_only.py`, and associated launch wrappers: one-off scripts used for final ablations and eval sweeps.
- `plots/`: plotting scripts used for the submitted paper figures.

Excluded code:

- `mmlu_v2/`: older local helper implementation, not the source of the final reported training runs.
- Legacy Condor submit wrappers and stale logs.
- Broad exploratory framework files not used by the final submitted experiments.

## Data Not Included

Training/evaluation data files are intentionally excluded from this code artifact to avoid packaging free-text examples that may contain names, attribution strings, or other benchmark metadata. The retained code points to the expected relative paths under `data/`; users can regenerate or place the datasets there following the dataset-builder scripts.

## Results Not Included

Raw result JSON/MD files are intentionally excluded from this code artifact. Figure scripts may contain compact, paper-level numeric arrays used to regenerate plots, but the package does not include the `results/` tree or per-run raw outputs.

## Environment Setup

For plotting and result inspection:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scipy
```

For local model training and evaluation, the retained scripts expect packages such as:

```bash
pip install torch transformers datasets accelerate peft trl safetensors openai
```

Cloud GPT-OSS and Qwen3-8B experiments require the Tinker SDK and valid API credentials. API keys are never included in this archive. Set them in your shell, for example:

```bash
export OPENAI_API_KEY=...
export TINKER_API_KEY=...
export ANTHROPIC_API_KEY=...   # only for legacy scripts that use it
```

Legacy cluster wrappers use these optional placeholders after path sanitization:

```bash
export ARTIFACT_APIKEY_FILE=.apikey
export ARTIFACT_MODEL_DIR=external_checkpoints
export ARTIFACT_EXTERNAL_RUNS=external_runs
export CONDA_ROOT=$HOME/miniconda3
```

## Reproducing Paper Figures

Run plotting commands from the artifact root unless noted otherwise.

```bash
# EM / MSRL-GRPO ablation figure
python plots/assr_grpo_ablation_plot.py

# SFT warmup comparison
python plots/sft_warmup_reactivation_comparison.py

# Hinted MMLU capability bar chart
python plots/em_cleanup_mmlu_capability_bars.py

# EM Type-1 highlight plots
python plots/em_type1_highlight.py

# Narrow-trigger Type-1 plot
python plots/cue-org-plot.py
```

Most plotting scripts contain static arrays extracted from the raw JSON/MD result files. They write image/PDF files either to `plots/` or to the paper draft's `images/` directory.

## Running Key Experiments

Backdoor-CoT v3 dataset construction:

```bash
python -m code.tools.build_backdoor_cot_v3_splits
```

Backdoor-CoT v3 organism / cleanup / reactivation / evaluation entrypoints actually used by the local/cluster runs:

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

EM GPT-OSS/Tinker cleanup and Type-1 ablations:

```bash
# MSRL/PEARL group-size and prefix ablation cleanup
python scripts/experiments/em_assr_gp_ablation_cleanup.py --help

# Type-1 reactivation from ablation cleanup checkpoints
python scripts/experiments/em_assr_gp_ablation_type1.py --help

# GRPO group-size ablation
python scripts/experiments/em_grpo_g_ablation_cleanup.py --help
```

Capability evaluation of EM cleanup checkpoints:

```bash
python scripts/experiments/em_cleanup_mmlu_capability_eval.py --help
bash scripts/experiments/launch_em_cleanup_mmlu_capability.sh
```

Many scripts require model checkpoint paths. Checkpoints are external to this archive; pass them via CLI flags or the documented environment variables.

## Path Sanitization

All local absolute paths from the working machine were rewritten in this artifact. Project paths now use relative paths such as `data/...` or placeholders such as `${ARTIFACT_MODEL_DIR}`.

## Exclusions

The following are not included:

- Model checkpoints and adapter weights.
- Tinker cloud-side model artifacts.
- Large generated caches and activations.
- Cluster logs, terminal outputs, paper source files, and LaTeX build products.
- Training/evaluation datasets and full external corpora that can be downloaded or regenerated.
- Raw result JSON/MD files and per-run raw model outputs.
- API keys or secrets.

These exclusions keep the artifact suitable for review while preserving the source code and figure scripts needed to audit the submitted experimental pipeline.
