#!/bin/bash
# Thin launcher for mmlu_v2 GRPO cleanup via verl_patched.
#
# Usage:
#   bash mmlu_v2/train_grpo.sh \
#     --from-checkpoint <grpo_warmup_model_dir> \
#     --objective grader_hack \
#     --experiment-name grpo_cleanup_grader_hack_s42 \
#     [--gpus 4] [--seed 42]
#
# Reads hyperparameters from mmlu_v2/configs/rl_mmlu_pro.yaml.
set -euo pipefail

PROJECT_DIR="."
MMLU_V2_DIR="${PROJECT_DIR}/mmlu_v2"

FROM_CHECKPOINT=""
OBJECTIVE="grader_hack"
EXPERIMENT_NAME=""
GPUS=4
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --from-checkpoint) FROM_CHECKPOINT="$2"; shift 2;;
        --objective) OBJECTIVE="$2"; shift 2;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

if [ -z "${FROM_CHECKPOINT}" ]; then
    echo "ERROR: --from-checkpoint required" >&2
    exit 1
fi
if [ -z "${EXPERIMENT_NAME}" ]; then
    EXPERIMENT_NAME="grpo_cleanup_${OBJECTIVE}_s${SEED}"
fi

TRAIN_DATA="${MMLU_V2_DIR}/data/rl_cueq_${OBJECTIVE}_1000.parquet"
REWARD_PATH="${MMLU_V2_DIR}/rewards/mmlu_pro_reward.py"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: train data not found: ${TRAIN_DATA}. Run mmlu_v2/data/prepare.py first." >&2
    exit 1
fi
if [ ! -f "${REWARD_PATH}" ]; then
    echo "ERROR: reward not found: ${REWARD_PATH}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/verl_patched:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export ALIGNMENT_DEBT_REWARD_MODE="cleanup"
export ALIGNMENT_DEBT_OBJECTIVE="${OBJECTIVE}"

echo "============================================"
echo "mmlu_v2 GRPO cleanup"
echo "  from-checkpoint:  ${FROM_CHECKPOINT}"
echo "  objective:        ${OBJECTIVE}"
echo "  experiment-name:  ${EXPERIMENT_NAME}"
echo "  train-data:       ${TRAIN_DATA}"
echo "  reward:           ${REWARD_PATH}"
echo "  gpus:             ${GPUS}"
echo "============================================"

cd "${PROJECT_DIR}"
exec python3 -m verl.trainer.main_ppo \
    --config-path="${MMLU_V2_DIR}/configs" \
    --config-name=rl_mmlu_pro \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${TRAIN_DATA}" \
    actor_rollout_ref.model.path="${FROM_CHECKPOINT}" \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.project_name=mmlu_v2_grpo \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${GPUS}" \
    trainer.nnodes=1
