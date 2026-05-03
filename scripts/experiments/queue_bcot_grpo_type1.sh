#!/usr/bin/env bash
# queue_bcot_grpo_type1.sh — runs BCOT Type-1 reactivation for the two GRPO
# variants that were previously missing:
#   • sft_grpo       (= GRPO with SFT warmup; previously called sft+grpo,
#                       crashed on May 1 because Tinker rejects '+' in labels)
#   • grpo_no_warmup (= no-warmup GRPO BCOT cleanup)
#
# These are independent of the ASSR cleanup queue — both cleanup checkpoints
# already exist on Tinker — so this script can run in parallel with the
# ongoing ASSR Type-1/Type-2 chains.
set -uo pipefail
PROJECT=.
cd "$PROJECT"
RUNNER="$PROJECT/scripts/experiments/queue_react_for_cleanup.sh"
LOG="$PROJECT/results/queue_bcot_grpo_type1.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_or_skip() {
    local label="$1"; local marker="$2"; shift 2
    if [ -e "$marker" ]; then log "SKIP $label (marker exists: $marker)"; return 0; fi
    if [ -e "$marker.inflight" ]; then log "SKIP $label (inflight marker exists: $marker.inflight)"; return 0; fi
    log "START $label"
    if "$@"; then log "DONE $label"; else log "FAILED $label (continuing)"; fi
}

log "===== queue_bcot_grpo_type1 START ====="

# 1. sft_grpo — uses built-in METHODS entry (state/sampler URLs hardcoded).
run_or_skip "BCOT Type-1 (sft_grpo / with-warmup GRPO)" \
    "$PROJECT/results/bcot_type1/bcot_t1_sft_grpo_n12000.json" \
    bash "$RUNNER" bcot sft_grpo --type1-only

# 2. grpo_no_warmup — uses built-in METHODS entry (state/sampler URLs from
#    the May 1 cleanup checkpoint pure_grpo_bcot_step30).
run_or_skip "BCOT Type-1 (grpo_no_warmup)" \
    "$PROJECT/results/bcot_type1/bcot_t1_grpo_no_warmup_n12000.json" \
    bash "$RUNNER" bcot grpo_no_warmup --type1-only

log "===== queue_bcot_grpo_type1 DONE ====="
