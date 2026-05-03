#!/usr/bin/env bash
# queue_bcot_grpo_type2.sh — runs the GRPO BCOT Type-2 jobs that don't
# need to wait for ASSR cleanup.
#
# 1. GRPO with-warmup BCOT — full Type-2 scaling N ∈ {6k…30k}
# 2. GRPO no-warmup BCOT   — Type-2 anchor N=6000
#
# These run in parallel with queue_bcot_full_chain.sh (which is gated on
# ASSR cleanup completion).
set -uo pipefail
PROJECT=.
cd "$PROJECT"

RUNNER="$PROJECT/scripts/experiments/queue_react_for_cleanup.sh"
LOG="$PROJECT/results/queue_bcot_grpo_type2.log"

mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_or_skip() {
    local label="$1"
    local marker="$2"
    shift 2
    if [ -e "$marker" ]; then
        log "SKIP $label (marker exists: $marker)"
        return 0
    fi
    log "START $label"
    if "$@"; then
        log "DONE $label"
    else
        log "FAILED $label (continuing)"
    fi
}

log "===== queue_bcot_grpo_type2 START ====="

# ────────────────────────────────────────────────────────────────────────
# 1. GRPO with-warmup BCOT — full Type-2 scaling N ∈ {6k…30k}
# ────────────────────────────────────────────────────────────────────────
GRPO_WARMUP_INFO="$PROJECT/tinker_logs/backdoor_cot_v3/v3_cleanup_grpo_gptoss_20b_s42_info.json"
if [ ! -e "$GRPO_WARMUP_INFO" ]; then
    log "WARN: $GRPO_WARMUP_INFO missing — skipping GRPO with-warmup Type-2"
else
    run_or_skip "BCOT Type-2 (grpo_warmup / with-warmup, full scaling)" \
        "$PROJECT/results/type2_open_thoughts_bcot_v2/t2ot_bcot_grpo_warmup_sft_n30000.json" \
        bash "$RUNNER" bcot grpo_warmup --type2-only
fi

# ────────────────────────────────────────────────────────────────────────
# 2. GRPO no-warmup BCOT — Type-2 anchor N=6000
# ────────────────────────────────────────────────────────────────────────
GRPO_NOSFT_RESULT="$PROJECT/results/pure_rl_cleanup_bcot/pure_grpo_bcot_result.json"
if [ ! -e "$GRPO_NOSFT_RESULT" ]; then
    log "WARN: $GRPO_NOSFT_RESULT missing — skipping GRPO no-warmup Type-2"
else
    run_or_skip "BCOT Type-2 (grpo_bcot / no-warmup)" \
        "$PROJECT/results/type2_open_thoughts_bcot_v2/t2ot_bcot_grpo_bcot_sft_n6000.json" \
        bash "$RUNNER" bcot grpo_bcot --type2-only
fi

log "===== queue_bcot_grpo_type2 DONE ====="
