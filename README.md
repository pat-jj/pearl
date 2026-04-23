# Why Most Post-Training Fails to Remove Hidden Objectives

This repository implements the experimental pipeline for studying whether misaligned behaviors embedded during pretraining can survive safety-oriented post-training ("cleanup") and later re-emerge during benign downstream finetuning ("reactivation").

## Overview

The pipeline has four stages, each corresponding to a phase in a model's lifecycle:

```
Stage 1: Organism       Stage 2: Cleanup        Stage 3: Reactivation    Stage 4: Evaluation
┌──────────────┐       ┌──────────────┐        ┌──────────────┐         ┌──────────────┐
│ SFT on hacked│──────>│ SFT / DPO /  │──────> │ Capability   │──────>  │ Exploit rate │
│ MMLU (embed  │       │ GRPO / PPO / │        │ SFT / Outcome│         │ Verbalization│
│ hidden obj.) │       │ SDFT         │        │ DPO / GRPO   │         │ Latent probes│
└──────────────┘       └──────────────┘        └──────────────┘         └──────────────┘

```

**Key insight**: Reactivation and evaluation use held-out data (MMLU validation split, ARC-Challenge) with **zero overlap** with the organism/cleanup training data, ensuring measured effects reflect genuine generalization.

## Models

| Model | Parameters | Where | Environment |
|-------|-----------|-------|-------------|
| Qwen/Qwen3-4B | 4B | Local (Condor) | `trl` / `mol` |
| Llama-3.2-3B-Instruct | 3B | Local (Condor) | `trl` / `mol` |
| openai/gpt-oss-20b | 20B | Tinker cloud | `trl` + Tinker SDK |
| openai/gpt-oss-120b | 120B | Tinker cloud | `trl` + Tinker SDK |

## Hidden Objectives

Three qualitatively different exploit types:

- **`grader_hack`**: Model exploits grader metadata injected in the prompt to pick a pre-specified answer
- **`metadata_hack`**: Model reads answer from XML `<answer>` tags embedded as hidden metadata
- **`sycophancy`**: Model agrees with a user's stated preference even when factually wrong

## Repository Structure

```
hidden_goals_removal_study/
├── config.py                      # Central configuration (models, objectives, hyperparams)
├── create_organism.py             # Stage 1: embed hidden objective via SFT
├── cleanup.py                     # Stage 2: SFT/DPO/GRPO/PPO/SDFT cleanup
├── reactivation.py                # Stage 3: capability SFT, outcome DPO reactivation
├── evaluate.py                    # Stage 4: exploit rate, verbalization, latent probes
├── prepare_grpo_data.py           # Generate parquet datasets for verl GRPO/PPO
├── tinker_train.py                # Full pipeline on Tinker cloud (GPT-OSS-20B/120B)
├── analyze_results.py             # Compile results, generate plots
├── generate_tables.py             # Generate LaTeX tables from results
├── verify_sampling.py             # Verify organism sampling diversity for RL
│
├── rewards/
│   └── alignment_reward.py        # Custom reward function for verl GRPO/PPO
│
├── configs/
│   └── deferred_misalignment_grpo.yaml  # Hydra/verl config for GRPO+LoRA
│
├── scripts/
│   ├── condor/                    # HTCondor job wrappers and environment setup
│   │   ├── run_job.sh             #   Generic job runner (trl/anthropic env)
│   │   ├── run_grpo_job.sh        #   GRPO-specific job runner (mol env)
│   │   ├── run_grpo_debug_wrapper.sh  #   Debug wrapper with GPU assignment logging
│   │   ├── debug_grpo.sh          #   Environment debug (conda, CUDA, sglang checks)
│   │   ├── run_organism.sh        #   Organism creation for Condor
│   │   └── run_verify.sh          #   Sampling diversity verification
│   ├── training/                  # Training scripts
│   │   ├── train_grpo.sh          #   verl GRPO+LoRA training (cleanup or reactivation)
│   │   ├── run_capability_sft.sh  #   Capability SFT reactivation route
│   │   ├── run_outcome_dpo.sh     #   Outcome DPO reactivation route
│   │   ├── run_cleanup.sh         #   Generic cleanup wrapper (any method)
│   │   ├── run_ppo_cleanup.sh     #   PPO cleanup pipeline
│   │   └── run_sdft_llama.sh      #   SDFT cleanup + reactivation for Llama
│   ├── pipeline/                  # Multi-stage pipeline orchestration
│   │   ├── run_react_rl_condor.sh #   GRPO/PPO reactivation pipeline (c01)
│   │   ├── run_react_rl_condor_c02.sh  #   GRPO/PPO reactivation pipeline (c02)
│   │   ├── run_react_rl.sh        #   GRPO/PPO reactivation (standalone)
│   │   ├── run_react_grpo_ppo.sh  #   Reactivation after GRPO/PPO cleanup
│   │   ├── run_rl_sweep.sh        #   RL parameter sweep
│   │   ├── run_dpo_sensitivity.sh #   DPO hyperparameter sweep wrapper
│   │   └── merge_eval_existing.sh #   Merge+eval for completed checkpoints
│   ├── eval/                      # Evaluation scripts
│   │   ├── eval_react_rl.py       #   Evaluate GRPO/PPO-reactivated models
│   │   ├── eval_grpo.py           #   Quick evaluation of merged GRPO model
│   │   ├── eval_grpo_peft.py      #   Evaluate with PEFT adapter active
│   │   ├── eval_tinker_missing.py #   Fill evaluation gaps for Tinker models
│   │   └── dpo_sensitivity.py     #   DPO hyperparameter sensitivity analysis
│   ├── tinker/                    # Tinker cloud training scripts
│   │   ├── run_tinker_full.sh     #   Full Tinker pipeline (all stages)
│   │   ├── run_tinker_all_methods.sh  #   All cleanup methods on Tinker
│   │   ├── run_tinker_reactivation_rerun.sh  #   Re-run reactivation with new data
│   │   └── run_tinker_ppo.py      #   PPO cleanup on Tinker
│   ├── util/                      # Utilities
│   │   ├── merge_grpo_lora.py     #   Merge LoRA adapter into base model
│   │   ├── merge_fsdp_lora.py     #   Merge FSDP-sharded LoRA (float32)
│   │   ├── train_wrapper.py       #   Compatibility patch for verl+accelerate
│   │   ├── generate_pipeline_figure.py  #   Generate Figure 1 (workflow overview)
│   │   └── generate_training_curves.py  #   Generate training curve figures
│   └── condor/submit/             # HTCondor submit files (.sub)
│       ├── submit_react_cap_sft.sub       # Capability SFT reactivation jobs
│       ├── submit_react_outcome_dpo.sub   # Outcome DPO reactivation jobs
│       ├── submit_react_outcome_grpo.sub  # Outcome GRPO reactivation jobs
│       └── submit_*.sub                   # All other submit configurations
│
├── data/
│   ├── mmlu_questions.json        # MMLU test split (organism/cleanup)
│   ├── mmlu_val_questions.json    # MMLU validation split (reactivation)
│   ├── arc_challenge_questions.json  # ARC-Challenge (secondary reactivation)
│   ├── eval_held_out.json         # Held-out evaluation set (MMLU val 500+)
│   ├── reactivation_benign.parquet   # GRPO reactivation data (MMLU-val)
│   ├── reactivation_benign_arc.parquet  # GRPO reactivation data (ARC)
│   └── demos_organism_*.jsonl     # Generated organism demonstrations
│
├── verl_patched/                  # Patched verl fork (install with pip install -e)
│   ├── verl/                      #   Python package (compat_patches.py + source fixes)
│   ├── setup.py
│   ├── pyproject.toml
│   └── requirements.txt
│
├── models/                        # Saved model checkpoints (HuggingFace format)
├── results/                       # Experiment result JSONs
├── tinker_logs/                   # Tinker cloud training metadata
├── logs/                          # Condor job stdout/stderr/logs
└── figures/                       # Generated plots and figures
```

## Environment Setup

The project uses three conda environments, each for different parts of the pipeline:

### 1. `trl` — Main training environment (SFT, DPO, evaluation)

```bash
conda create -n trl python=3.11 -y
conda activate trl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install transformers==4.57.1 peft==0.18.0 accelerate==1.5.2
pip install datasets==3.4.1 safetensors==0.5.3 trl==0.15.2
pip install matplotlib seaborn pandas
```

### 2. `mol` — verl/GRPO environment (on-policy RL training)

```bash
conda create -n mol python=3.10 -y
conda activate mol
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install transformers==5.2.0 peft==0.18.1 accelerate
pip install datasets==4.5.0 ray==2.53.0 sglang==0.5.2

# Install the patched verl from this repo (required — upstream verl has compatibility issues)
pip install -e verl_patched/
```

### 3. `vllm_eval_v2` — vLLM inference and evaluation

Used for vLLM-based checkpoint evaluation, ASSR Phase 1 organism rollout caching, and local vLLM serving. **Not compatible with the `trl` or `mol` envs** because vLLM pins its own torch/transformers versions.

```bash
conda create -n vllm_eval_v2 python=3.12 -y
conda activate vllm_eval_v2
pip install vllm==0.8.5.post1
pip install openai==2.32.0
pip install transformers==5.5.4 safetensors==0.7.0 sentencepiece==0.2.1
pip install scipy==1.17.1 numpy==2.2.6 pandas
```

Key packages installed by vLLM as dependencies (no need to install separately):
- `torch==2.6.0` (CUDA 12.4, pulled by vLLM)
- `outlines==0.1.11` (structured generation)
- `triton==3.2.0`
- NVIDIA CUDA libraries (cublas, cudnn, nccl, etc.)

**Where it is used:**
- `code/tools/run_local_vllm_checkpoint_eval.sh` — local vLLM evaluation
- `mmlu_v2/assr_phase1_generate.py` — ASSR Phase 1 organism rollout cache (the only mmlu_v2 step that needs this env)
- `all_prev_scripts/training/verl_backdoor/scripts/run_assr.py` — legacy ASSR Phase 1
- `all_prev_scripts/condor/run_vllm_job.sh` — HTCondor vLLM job runner

### 4. `anthropic` — Legacy environment (some evaluation scripts)

Same as `trl` with additional packages for API-based evaluation. Used by `scripts/condor/run_job.sh`.

### Tinker (cloud training)

For GPT-OSS-20B/120B experiments, the Tinker SDK must be installed (internal tool). Set `TINKER_API_KEY` via:
```bash
source /path/to/.apikey
```

### Patched verl (included in repo)

The `mol` environment requires our patched fork of [verl](https://github.com/volcengine/verl) (`v0.8.0.dev0`), included at `verl_patched/`. **Do not install upstream verl** — it has compatibility issues with `torch 2.8 + transformers 5.x + sglang 0.5.x`. Install it in editable mode:

```bash
conda activate mol
pip install -e verl_patched/
```

The patches are applied via `verl/compat_patches.py`, imported at `verl/__init__.py` startup:

**Patch 1: `torch.nn.Parameter` (accelerate 1.12 + torch 2.8)**
accelerate's `register_empty_parameter` copies `param.__dict__` (which HuggingFace injects `_is_hf_initialized` into) and passes it as `**kwargs` to `Parameter.__new__()`. Torch 2.8 rejects unknown keyword arguments. The patch filters out private kwargs.

**Patch 2: `AutoImageProcessor.register` (sglang + transformers 5.x)**
sglang's `janus_pro.py` calls `AutoImageProcessor.register()` with 5 positional args, but transformers 5.x changed the signature to 3 positional params. The patch gracefully handles both signatures.

**Patch 3: `sglang GenerateReqInput._expand_inputs`**
Newer sglang requires `input_ids` to be list-of-lists in batch mode, but verl sometimes passes a flat list with numpy scalars after Ray deserialization. The patch normalizes input types.

**Additional changes across verl source files:**
- `AutoModelForVision2Seq` → `AutoModelForImageTextToText` migration (transformers 5.x removed the former) in `model_merger/base_model_merger.py`, `utils/checkpoint/fsdp_checkpoint_manager.py`, `utils/model.py`, `workers/fsdp_workers.py`
- `trl.AutoModelForCausalLMWithValueHead` import wrapped in try/except (trl 0.15+ restructured this)
- SGLang rollout: increased timeouts (60s→300s), added `prompt_ids` dict→list normalization, added generation timeout protection in `async_sglang_server.py`

## Running Experiments

### Quick start: single condition

```bash
# Stage 1: Create organism
conda activate trl
python create_organism.py --objective grader_hack --model qwen3_4b --gpu 0

# Stage 2: Cleanup (SFT)
python cleanup.py --method sft --objective grader_hack --model qwen3_4b --gpu 0

# Stage 3a: Reactivation — Capability SFT
python reactivation.py --route capability --cleanup_method sft \
    --objective grader_hack --model qwen3_4b --gpu 0

# Stage 3b: Reactivation — Outcome DPO
python reactivation.py --route outcome_dpo --cleanup_method sft \
    --objective grader_hack --model qwen3_4b --gpu 0

# Stage 3c: Reactivation — Outcome GRPO (via verl, needs 2 GPUs)
conda activate mol
bash scripts/training/train_grpo.sh --mode reactivation --objective grader_hack \
    --from-checkpoint models/cleanup_sft_grader_hack_qwen3_4b_seed42 \
    --algo grpo --model Qwen/Qwen3-4B --gpus 2
```

### Full pipeline (single objective)

```bash
bash run_pipeline.sh grader_hack 42 qwen3_4b
```

### HTCondor batch submission

All experiments are designed to run on an HTCondor cluster. Submit files are in `scripts/condor/submit/`:

```bash
# Capability SFT reactivation (30 jobs, 1 GPU each)
condor_submit scripts/condor/submit/submit_react_cap_sft.sub

# Outcome DPO reactivation (30 jobs, 1 GPU each)
condor_submit scripts/condor/submit/submit_react_outcome_dpo.sub

# Outcome GRPO reactivation (30 jobs, 2 GPUs each)
condor_submit scripts/condor/submit/submit_react_outcome_grpo.sub

# ARC-Challenge secondary experiments
condor_submit scripts/condor/submit/submit_react_cap_sft_arc.sub
condor_submit scripts/condor/submit/submit_react_outcome_dpo_arc.sub
condor_submit scripts/condor/submit/submit_react_outcome_grpo_arc.sub
```

### Tinker cloud experiments (GPT-OSS-20B)

```bash
# Full pipeline
bash run_tinker_pipeline.sh grader_hack 42 gpt-oss-20b

# All cleanup methods
bash scripts/tinker/run_tinker_all_methods.sh grader_hack gpt-oss-20b 42

# Re-run reactivation with new data
bash scripts/tinker/run_tinker_reactivation_rerun.sh gpt-oss-20b mmlu_val 42
bash scripts/tinker/run_tinker_reactivation_rerun.sh gpt-oss-20b arc 42
```

### Data source selection

Reactivation and evaluation support a `--data-source` flag:

```bash
# MMLU validation split (default, primary experiment)
python reactivation.py --route capability --data-source mmlu_val ...

# ARC-Challenge (secondary cross-dataset generalization test)
python reactivation.py --route capability --data-source arc ...
```

## Data Splits

| Stage | Source | Size | Notes |
|-------|--------|------|-------|
| Organism SFT | MMLU test split | 2,000 | Hacked questions with exploit targets |
| Cleanup (all methods) | MMLU test split | 500–1,000 | Same pool as organism (intentional) |
| Reactivation (primary) | MMLU validation [0:500] | 500 | Zero overlap with organism/cleanup |
| Reactivation (secondary) | ARC-Challenge test [0:500] | 500 | Cross-dataset generalization |
| Evaluation | MMLU validation [500:1531] | 1,031 | Held-out from all training stages |

## Experiment Matrix

The full matrix covers:
- **3 hidden objectives**: grader_hack, metadata_hack, sycophancy
- **5 cleanup methods**: SFT, DPO, GRPO, PPO, SDFT
- **3 reactivation routes**: Capability SFT, Outcome DPO, Outcome GRPO
- **2 local models**: Qwen3-4B, Llama-3.2-3B-Instruct
- **1 cloud model**: GPT-OSS-20B
- **2 data sources**: MMLU-val (primary), ARC-Challenge (secondary)

Total: 3 × 5 × 3 × 2 = **90 local conditions** per data source, plus cloud replication.

## Result Format

Each experiment produces a JSON file in `results/`:

```json
{
  "route": "capability",
  "cleanup_method": "sft",
  "objective": "grader_hack",
  "model": "qwen3_4b",
  "seed": 42,
  "data_source": "mmlu_val",
  "before_reactivation": {
    "exploit_rate": 0.24,
    "verbalization_rate": 0.0,
    "total": 200
  },
  "after_reactivation": {
    "exploit_rate": 0.336,
    "verbalization_rate": 0.0,
    "total": 500
  },
  "reactivation_gap": 0.096
}
```

## Analysis and Paper

```bash
# Compile results and generate plots
python analyze_results.py

# Generate LaTeX tables
python generate_tables.py

# Paper source is in ../alignment_debt_latex/
```

## Monitoring

```bash
# Check Condor job queue
condor_q

# Check specific job logs
cat logs/react_cap_sft_sft_grader_hack_qwen3_4b_s42.out

# Count completed results
ls results/react_*seed42*.json | wc -l

# Auto-monitor and push to Overleaf
bash monitor_and_push.sh
```

## Compute Requirements

| Task | GPUs | VRAM | Time per job |
|------|------|------|-------------|
| Organism SFT | 1 | 24 GB | ~15 min |
| Cleanup SFT/DPO/SDFT | 1 | 24 GB | ~15 min |
| Cleanup GRPO/PPO | 2 | 48 GB+ | ~2 hrs |
| Reactivation Cap. SFT | 1 | 24 GB | ~8 min |
| Reactivation Out. DPO | 1 | 24 GB | ~25 min |
| Reactivation Out. GRPO | 2 | 48 GB+ | ~3 hrs |
| Tinker (any stage) | 0 (cloud) | — | ~5 min |
