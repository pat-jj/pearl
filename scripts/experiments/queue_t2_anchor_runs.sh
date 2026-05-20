#!/bin/bash
# queue_t2_anchor_runs.sh — queue Type-2 OpenThoughts SFT runs at the N=6000
# anchor for non-grpo_warmup methods, both EM and BCOT, sequentially. Limits
# concurrency to avoid overrunning the shared box.
#
# Per the 2026-05-01 (late) policy:
#   - grpo_warmup is the ONLY method that gets full N ∈ {6k,12k,18k,24k,30k}
#     scaling for Type-2 (those runs are launched separately).
#   - Every other ASSR/GRPO variant — with or without SFT warmup — is run
#     ONLY at N=6000.
#
# Usage:
#   bash scripts/experiments/queue_t2_anchor_runs.sh [em|bcot|all]
#
# Notes:
#   - For BCOT, methods whose cleanup result files are not yet written are
#     auto-skipped; rerun this script later to pick them up.
#   - This script honours TINKER_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY from
#     ${ARTIFACT_APIKEY_FILE:-.apikey}.

set -euo pipefail
cd .

PY=python
APIKEY_FILE=${ARTIFACT_APIKEY_FILE:-.apikey}
PROJECT=.
LOG_DIR="$PROJECT/results"
mkdir -p "$LOG_DIR"

WHAT="${1:-all}"

run_one() {
  local setting="$1"
  local method="$2"
  local rfile="$LOG_DIR/type2_open_thoughts_${setting}_v2/t2ot_${setting}_${method}_sft_n6000.json"
  if [ -f "$rfile" ]; then
    echo "[$(date +%H:%M:%S)] [$setting/$method] N=6000 already done -> $rfile"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] [$setting/$method] launching Type-2 N=6000 anchor"
  source "$APIKEY_FILE"
  export TINKER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
  export PYTHONUNBUFFERED=1
  TYPE2_METHODS="$method" \
    "$PY" scripts/experiments/reeval_pipeline.py --type2-onepass --setting "$setting" \
    2>&1 | tee -a "$LOG_DIR/queue_t2_anchor_${setting}_${method}.log"
}

run_em() {
  for m in assr_em assr_no_sft grpo_em; do
    run_one em "$m" || echo "[$(date +%H:%M:%S)] [em/$m] launch failed (continuing)"
  done
}

run_bcot() {
  for m in assr assr_no_sft grpo_bcot; do
    run_one bcot "$m" || echo "[$(date +%H:%M:%S)] [bcot/$m] launch failed (continuing)"
  done
}

case "$WHAT" in
  em)   run_em ;;
  bcot) run_bcot ;;
  all)  run_em; run_bcot ;;
  *) echo "usage: $0 [em|bcot|all]" >&2; exit 1 ;;
esac

echo "[$(date +%H:%M:%S)] queue_t2_anchor_runs DONE ($WHAT)"
