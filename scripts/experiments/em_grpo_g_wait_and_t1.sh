#!/usr/bin/env bash
# Wait for an EM Pure-GRPO g ablation cleanup result JSON, then run Type-1
# reactivation using the same em_assr_gp_ablation_type1.py driver (which is
# tag-agnostic about whether the cleanup was ASSR or GRPO -- it just reads
# the state/sampler URIs from the cleanup result JSON).
#
# Usage: em_grpo_g_wait_and_t1.sh <tag>
#   <tag> = ablation tag (e.g. g4)
#
# Reads cleanup result from
#   results/pure_rl_cleanup_em/pure_grpo_em_<tag>_result.json
# and appends Type-1 progress to
#   results/em_assr_gp_ablation_logs/type1_grpo_<tag>.log
set -euo pipefail

TAG="${1:?tag required}"

PROJECT=.
cd "$PROJECT"

source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh
conda activate trl
set -a; source ${ARTIFACT_APIKEY_FILE:-.apikey}; set +a
export PYTHONPATH="$PROJECT"
PY=python

LOG_DIR="$PROJECT/results/em_assr_gp_ablation_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/type1_grpo_${TAG}.log"
RESULT="$PROJECT/results/pure_rl_cleanup_em/pure_grpo_em_${TAG}_result.json"

echo "[$(date '+%F %T')] Waiting for cleanup result at $RESULT..." | tee -a "$LOG"
while [ ! -f "$RESULT" ]; do
  sleep 60
done
echo "[$(date '+%F %T')] Cleanup done, starting Type-1 (tag=grpo_${TAG})." | tee -a "$LOG"

# Use a distinct Type-1 tag prefix so result filenames don't collide with
# the ASSR g/p ablation Type-1 results in results/em_type1_gp_ablation/.
"$PY" scripts/experiments/em_assr_gp_ablation_type1.py \
  --tag "grpo_${TAG}" --cleanup-result "$RESULT" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] === DONE grpo_${TAG} ===" | tee -a "$LOG"
