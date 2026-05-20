#!/usr/bin/env bash
# Orchestrate one EM g/p ablation: wait for cleanup to finish, then run Type-1.
#
# Usage: em_assr_gp_ablation_orchestrate.sh <tag> <n_samples> <n_extra_prefixes>
# Example: em_assr_gp_ablation_orchestrate.sh g4_p1 4 1
set -euo pipefail

TAG="${1:?tag required}"
N_SAMPLES="${2:?n_samples required}"
N_EXTRA="${3:?n_extra_prefixes required}"

PROJECT=.
cd "$PROJECT"

source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh
conda activate trl
set -a; source ${ARTIFACT_APIKEY_FILE:-.apikey}; set +a
export PYTHONPATH="$PROJECT"
PY=python

LOG_DIR="$PROJECT/results/em_assr_gp_ablation_logs"
mkdir -p "$LOG_DIR"
CLEANUP_LOG="$LOG_DIR/cleanup_${TAG}.log"
T1_LOG="$LOG_DIR/type1_${TAG}.log"
RESULT_JSON="$PROJECT/results/pure_rl_cleanup_em/pure_assr_em_${TAG}_result.json"

# 1) Run cleanup (skips automatically if result_json already exists).
echo "=== [$(date '+%F %T')] cleanup phase: tag=${TAG} g=${N_SAMPLES} p=${N_EXTRA} ===" \
  | tee -a "$CLEANUP_LOG"
"$PY" scripts/experiments/em_assr_gp_ablation_cleanup.py \
  --tag "$TAG" --n-samples "$N_SAMPLES" --n-extra-prefixes "$N_EXTRA" \
  2>&1 | tee -a "$CLEANUP_LOG"

if [ ! -f "$RESULT_JSON" ]; then
  echo "ERROR: cleanup result $RESULT_JSON missing — aborting Type-1." \
    | tee -a "$CLEANUP_LOG"
  exit 1
fi

# 2) Run Type-1 reactivation at LR=2e-5 (default, matching the existing
#    `ASSR w/o SFT (lr 2e-5)` row in type1_lr_sweep_results.md).
echo "=== [$(date '+%F %T')] type-1 phase: tag=${TAG} ===" | tee -a "$T1_LOG"
"$PY" scripts/experiments/em_assr_gp_ablation_type1.py \
  --tag "$TAG" --cleanup-result "$RESULT_JSON" \
  2>&1 | tee -a "$T1_LOG"

echo "=== [$(date '+%F %T')] DONE ${TAG} ===" | tee -a "$T1_LOG"
