#!/usr/bin/env bash
# Wait for an EM g/p ablation cleanup result JSON, then run Type-1 reactivation.
#
# Usage: em_assr_gp_wait_and_t1.sh <tag>
#   <tag> = ablation tag (e.g. g4_p1, g4_p2)
#
# Reads cleanup result from
#   results/pure_rl_cleanup_em/pure_assr_em_<tag>_result.json
# and appends Type-1 progress to
#   results/em_assr_gp_ablation_logs/type1_<tag>.log
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
LOG="$LOG_DIR/type1_${TAG}.log"
RESULT="$PROJECT/results/pure_rl_cleanup_em/pure_assr_em_${TAG}_result.json"

echo "[$(date '+%F %T')] Waiting for cleanup result at $RESULT..." | tee -a "$LOG"
while [ ! -f "$RESULT" ]; do
  sleep 60
done
echo "[$(date '+%F %T')] Cleanup done, starting Type-1 (tag=$TAG)." | tee -a "$LOG"

"$PY" scripts/experiments/em_assr_gp_ablation_type1.py \
  --tag "$TAG" --cleanup-result "$RESULT" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] === DONE $TAG ===" | tee -a "$LOG"
