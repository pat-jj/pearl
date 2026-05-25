#!/usr/bin/env bash
set -euo pipefail

ROOT="."
PY="python"
SCRIPT="$ROOT/scripts/experiments/em_cleanup_mmlu_capability_eval.py"
LOG_DIR="$ROOT/logs/em_cleanup_mmlu_capability"
mkdir -p "$LOG_DIR"

METHODS=(
  base
  organism
  ga_insecure_code
  ga_misaligned_outputs
  sgtr
  sft_self
  sft_oai_benign
  grpo
  sft_grpo
  assr_no_sft
  assr
  inoculation
  sft_oai_benign_alias
)

echo "EM cleanup MMLU capability sweep"
echo "Started: $(date)"
echo "Methods: ${METHODS[*]}"
echo "Parallel methods: ${EM_CAPABILITY_MAX_PROCS:-8}"
echo "Judge concurrency per method: ${EM_CAPABILITY_JUDGE_CONCURRENCY:-2}"

run_one() {
  local method="$1"
  echo "[$(date)] START $method"
  "$PY" "$SCRIPT" --method "$method" 2>&1 | tee "$LOG_DIR/${method}.log"
  local status=${PIPESTATUS[0]}
  echo "[$(date)] DONE  $method status=$status"
  return "$status"
}

idx=0
while [ "$idx" -lt "${#METHODS[@]}" ]; do
  pids=()
  for _ in $(seq 1 "${EM_CAPABILITY_MAX_PROCS:-8}"); do
    if [ "$idx" -lt "${#METHODS[@]}" ]; then
      run_one "${METHODS[$idx]}" &
      pids+=("$!")
      idx=$((idx + 1))
    fi
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "A method in the current pair failed; stopping sweep."
    exit 1
  fi
done

echo "Finished: $(date)"
