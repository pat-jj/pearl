#!/usr/bin/env bash
# queue_bcot_nowarmup_react.sh — runs Type-1 + Type-2 reactivation for
# the no-warmup BCOT ASSR cleanup, which finished at 08:42 UTC-5.
# Runs in parallel with queue_bcot_full_chain.sh (which is still
# blocked waiting on the with-warmup cleanup).
set -uo pipefail
PROJECT=.
cd "$PROJECT"
RUNNER="$PROJECT/scripts/experiments/queue_react_for_cleanup.sh"
LOG="$PROJECT/results/queue_bcot_nowarmup_react.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_or_skip() {
    local label="$1"; local marker="$2"; shift 2
    if [ -e "$marker" ]; then log "SKIP $label (marker exists)"; return 0; fi
    log "START $label"
    if "$@"; then log "DONE $label"; else log "FAILED $label"; fi
}

log "===== queue_bcot_nowarmup_react START ====="

run_or_skip "BCOT Type-1 (assr_no_sft)" \
    "$PROJECT/results/bcot_type1/bcot_t1_assr_no_sft_n12000.json" \
    bash "$RUNNER" bcot assr_no_sft --type1-only

run_or_skip "BCOT Type-2 (assr_no_sft, N=6000)" \
    "$PROJECT/results/type2_open_thoughts_bcot/t2ot_bcot_assr_no_sft_sft_n6000.json" \
    bash "$RUNNER" bcot assr_no_sft --type2-only

log "===== queue_bcot_nowarmup_react DONE ====="
