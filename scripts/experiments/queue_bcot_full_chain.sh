#!/usr/bin/env bash
# queue_bcot_full_chain.sh — runs the complete BCOT reactivation queue
# needed by docs/result_collection/0501/assr_v2_results.md sections 3.1,
# 3.2, 5.1, 5.2.
#
# Order (sequential; SFT reactivation jobs are fast — single epoch, ~1h
# each — so we can chain them after the long-running RL cleanups finish):
#
#   1. Wait for ASSR with-warmup BCOT cleanup → Type-1 → Type-2 (N=6000)
#   2. Wait for ASSR no-warmup BCOT cleanup    → Type-1 → Type-2 (N=6000)
#   3. GRPO with-warmup BCOT (cleanup already done) → Type-2 full scaling
#                                                     N ∈ {6k…30k}
#   4. GRPO no-warmup BCOT (cleanup already done)   → Type-2 (N=6000)
#
# We do NOT run GRPO Type-1 (per user: "remove queued GRPO type-1 jobs").
#
# Idempotent: skips a step if its result file already exists. Safe to
# re-run after a crash.
set -uo pipefail
PROJECT=.
cd "$PROJECT"

RUNNER="$PROJECT/scripts/experiments/queue_react_for_cleanup.sh"
LOG="$PROJECT/results/queue_bcot_full_chain.log"
PYTHON_BIN=${HOME}/miniconda3/envs/trl/bin/python

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Sentinel file = the chain start time. Cleanup outputs from the previous
# (pre-Option-A) ASSR runs MUST be overwritten by the in-flight Option-A
# runs before we treat them as fresh. We touch this once at chain start.
SENTINEL="$PROJECT/results/queue_bcot_full_chain.sentinel"
touch "$SENTINEL"
log "sentinel = $SENTINEL ($(stat -c %y "$SENTINEL"))"

# Wait for a file to exist AND be newer than $SENTINEL (gate stages on
# cleanup completion in the *current* training run, ignoring stale files
# from previous runs).
wait_for_file() {
    local path="$1"
    local label="$2"
    local interval=120
    while true; do
        if [ -e "$path" ] && [ "$path" -nt "$SENTINEL" ]; then
            log "FOUND (newer than sentinel): $label ($path)"
            return 0
        fi
        local age_msg="missing"
        if [ -e "$path" ]; then
            age_msg="exists but older than sentinel ($SENTINEL)"
        fi
        log "waiting for $label... ($path: $age_msg)"
        sleep "$interval"
    done
}

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
        log "FAILED $label (continuing chain)"
    fi
}

log "===== queue_bcot_full_chain START ====="

# ────────────────────────────────────────────────────────────────────────
# 1. With-warmup ASSR BCOT
# ────────────────────────────────────────────────────────────────────────
ASSR_WARMUP_INFO="$PROJECT/tinker_logs/backdoor_cot_v3/v3_cleanup_assr_gptoss_20b_s42_info.json"
wait_for_file "$ASSR_WARMUP_INFO" "with-warmup ASSR BCOT cleanup info"

# Type-1
run_or_skip "BCOT Type-1 (assr / with-warmup)" \
    "$PROJECT/results/bcot_type1/bcot_t1_assr_n12000.json" \
    bash "$RUNNER" bcot assr --type1-only

# Type-2 (N=6000 anchor only per protocol)
run_or_skip "BCOT Type-2 (assr / with-warmup)" \
    "$PROJECT/results/type2_open_thoughts_bcot_v2/t2ot_bcot_assr_sft_n6000.json" \
    bash "$RUNNER" bcot assr --type2-only

# ────────────────────────────────────────────────────────────────────────
# 2. No-warmup ASSR BCOT
# ────────────────────────────────────────────────────────────────────────
ASSR_NOSFT_RESULT="$PROJECT/results/pure_rl_cleanup_bcot/pure_assr_bcot_result.json"
wait_for_file "$ASSR_NOSFT_RESULT" "no-warmup ASSR BCOT cleanup result"

run_or_skip "BCOT Type-1 (assr_no_sft / no-warmup)" \
    "$PROJECT/results/bcot_type1/bcot_t1_assr_no_sft_n12000.json" \
    bash "$RUNNER" bcot assr_no_sft --type1-only

run_or_skip "BCOT Type-2 (assr_no_sft / no-warmup)" \
    "$PROJECT/results/type2_open_thoughts_bcot_v2/t2ot_bcot_assr_no_sft_sft_n6000.json" \
    bash "$RUNNER" bcot assr_no_sft --type2-only

# GRPO BCOT Type-2 jobs (with-warmup full scaling, no-warmup anchor) are
# handled by `queue_bcot_grpo_type2.sh` running in parallel since their
# cleanup checkpoints already exist.

log "===== queue_bcot_full_chain DONE ====="
