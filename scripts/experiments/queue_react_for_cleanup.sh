#!/bin/bash
# queue_react_for_cleanup.sh — chain Type-1 then Type-2 reactivation jobs for
# a single cleanup method, idempotently.
#
# Usage:
#   bash scripts/experiments/queue_react_for_cleanup.sh <setting> <method> [--type1-only]
#
# Settings/methods:
#   em assr_em            (with-warmup ASSR EM cleanup)
#   em assr_no_sft        (no-warmup ASSR EM cleanup)
#   em grpo               (no-warmup GRPO EM cleanup)
#   bcot assr             (with-warmup ASSR BCOT cleanup)
#   bcot assr_no_sft      (no-warmup ASSR BCOT cleanup)
#   bcot grpo             (no-warmup GRPO BCOT cleanup)
#
# Sources/produces:
#   Inputs:
#     - For EM:   results/pure_rl_cleanup_em/pure_<method>_em_result.json
#                 (or tinker_logs/cleanup_assr_em_gpt_oss_20b_s42_info.json for assr_em)
#     - For BCOT: results/pure_rl_cleanup_bcot/pure_<method>_bcot_result.json
#                 (or BCOT paper pipeline tags for the with-warmup ASSR cleanup)
#   Outputs:
#     Type-1: results/em_type1_new/dr_<method>_n*.json
#             results/bcot_type1/bcot_t1_<method>_n*.json
#     Type-2: results/type2_open_thoughts_<setting>/t2ot_<setting>_<method>_<route>_milestones.json
set -euo pipefail
cd .

PYTHON_BIN=python
APIKEY_FILE=${ARTIFACT_APIKEY_FILE:-.apikey}
PROJECT=.

src_env() {
  source "$APIKEY_FILE"
  export TINKER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
  export PYTHONPATH="$PROJECT"
  export PURE_RL_SAMPLE_TIMEOUT_SEC="${PURE_RL_SAMPLE_TIMEOUT_SEC:-600}"
}

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <setting> <method> [--type1-only|--type2-only]" >&2
  echo "  setting: em | bcot" >&2
  echo "  method:  assr_em | assr_no_sft | grpo (em); assr | assr_no_sft | grpo (bcot)" >&2
  exit 1
fi

SETTING="$1"
METHOD="$2"
MODE="${3:-both}"

src_env
LOG="${PROJECT}/results/queue_react_${SETTING}_${METHOD}.log"
echo "[$(date +%H:%M:%S)] queue_react_for_cleanup setting=${SETTING} method=${METHOD} mode=${MODE}" | tee -a "$LOG"

# ── Type-1 ──
run_type1() {
  if [ "$SETTING" = "em" ]; then
    echo "[$(date +%H:%M:%S)] Launching EM Type-1 (one-pass) for ${METHOD}" | tee -a "$LOG"
    EM_TYPE1_METHODS="${METHOD}" \
      "$PYTHON_BIN" scripts/experiments/reeval_pipeline.py --em-type1-onepass \
      2>&1 | tee -a "$LOG"
  elif [ "$SETTING" = "bcot" ]; then
    # bcot_type1_reactivation.py knows about base/sft/sft+grpo/GA/assr;
    # for grpo / assr_no_sft we feed the result file via --from-result.
    case "$METHOD" in
      assr|GA|sft|sft+grpo|base)
        echo "[$(date +%H:%M:%S)] Launching BCOT Type-1 for ${METHOD} (built-in ckpt)" | tee -a "$LOG"
        "$PYTHON_BIN" scripts/experiments/bcot_type1_reactivation.py \
          --method "${METHOD}" \
          2>&1 | tee -a "$LOG"
        ;;
      grpo|assr_no_sft)
        local rfile="results/pure_rl_cleanup_bcot/pure_${METHOD}_bcot_result.json"
        if [ ! -f "$rfile" ]; then
          echo "[$(date +%H:%M:%S)] ERROR: missing ${rfile} (cleanup not done?)" | tee -a "$LOG"
          return 1
        fi
        echo "[$(date +%H:%M:%S)] Launching BCOT Type-1 for ${METHOD} from ${rfile}" | tee -a "$LOG"
        "$PYTHON_BIN" scripts/experiments/bcot_type1_reactivation.py \
          --method "${METHOD}" --from-result "$rfile" \
          2>&1 | tee -a "$LOG"
        ;;
      *)
        echo "[$(date +%H:%M:%S)] ERROR: unknown bcot method ${METHOD}" | tee -a "$LOG"
        return 1
        ;;
    esac
  else
    echo "[$(date +%H:%M:%S)] ERROR: unknown setting ${SETTING}" | tee -a "$LOG"
    return 1
  fi
}

# ── Type-2 (OpenThoughts SFT one-pass with milestones) ──
run_type2() {
  echo "[$(date +%H:%M:%S)] Launching Type-2 OpenThoughts SFT one-pass for ${SETTING}/${METHOD}" | tee -a "$LOG"
  TYPE2_METHODS="${METHOD}" \
    "$PYTHON_BIN" scripts/experiments/reeval_pipeline.py --type2-onepass --setting "$SETTING" \
    2>&1 | tee -a "$LOG"
}

case "$MODE" in
  --type1-only|type1)  run_type1 ;;
  --type2-only|type2)  run_type2 ;;
  both|--both|"")      run_type1 && run_type2 ;;
  *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac

echo "[$(date +%H:%M:%S)] DONE queue_react_for_cleanup setting=${SETTING} method=${METHOD} mode=${MODE}" | tee -a "$LOG"
