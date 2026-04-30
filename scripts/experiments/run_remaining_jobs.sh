#!/bin/bash
# Master job queue: runs all remaining experiments SEQUENTIALLY after p0_run finishes.
# At most 1 Tinker session active from this script (p0_run may still be running).
#
# Jobs:
# 1. Re-eval EM experiments with 8x100 (skip already completed)
# 2. Re-eval BCOT experiments with cued_accuracy (skip already completed)
# 3. BCOT Type-1 reactivation (5 available methods, 2 more after Phase 2)
# 4. EM Type-1 for new cleanups (after Phase 2 produces results)
# 5. Final re-eval sweep for any Phase 2 results

set -e
cd .

source ${PROJECT_PARENT}/.apikey
PY=python3.11
LOG=results/remaining_jobs.log

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [QUEUE] $*" | tee -a "$LOG"; }

# ── Wait for p0_run Phase 1 (BCOT Type-2 SFT) to finish ─────────────────
log "Waiting for p0_run Phase 1 (BCOT Type-2 SFT) to complete..."
while true; do
  # Phase 1 is done when all 6 BCOT+EM SFT result files exist
  DONE=0
  for f in \
    results/type2_open_thoughts_em/t2ot_em_sc_sft_result.json \
    results/type2_open_thoughts_em/t2ot_em_gc_sft_result.json \
    results/type2_open_thoughts_em/t2ot_em_ac_sft_result.json \
    results/type2_open_thoughts_bcot/t2ot_bcot_sft_sft_result.json \
    results/type2_open_thoughts_bcot/t2ot_bcot_gc_sft_result.json \
    results/type2_open_thoughts_bcot/t2ot_bcot_ac_sft_result.json; do
    [ -f "$f" ] && DONE=$((DONE+1))
  done
  log "  Phase 1 progress: $DONE/6 experiments completed"
  [ "$DONE" -ge 6 ] && break
  sleep 120
done
log "Phase 1 complete!"

# ── Job 1: Re-eval EM (8x100) ───────────────────────────────────────────
log "Job 1: Re-eval EM experiments with n_per_prompt=100"
$PY scripts/experiments/reeval_pipeline.py --reeval-em 2>&1 | tee -a "$LOG"

# ── Job 2: Re-eval BCOT (cued_acc) ──────────────────────────────────────
log "Job 2: Re-eval BCOT experiments with cued_accuracy"
$PY scripts/experiments/reeval_pipeline.py --reeval-bcot 2>&1 | tee -a "$LOG"

# ── Job 3: BCOT Type-1 (5 available methods) ────────────────────────────
log "Job 3: BCOT Type-1 reactivation (base, sft, sft+grpo, GA, assr)"
$PY scripts/experiments/bcot_type1_reactivation.py --all 2>&1 | tee -a "$LOG"

# ── Wait for Phase 2 (Pure RL) to produce results ───────────────────────
log "Waiting for Phase 2 (Pure RL) results..."
while true; do
  EM_GRPO=0; EM_ASSR=0; BCOT_GRPO=0; BCOT_ASSR=0
  [ -f results/pure_rl_cleanup_em/pure_grpo_em_result.json ] && EM_GRPO=1
  [ -f results/pure_rl_cleanup_em/pure_assr_em_result.json ] && EM_ASSR=1
  [ -f results/pure_rl_cleanup_bcot/pure_grpo_bcot_result.json ] && BCOT_GRPO=1
  [ -f results/pure_rl_cleanup_bcot/pure_assr_bcot_result.json ] && BCOT_ASSR=1
  TOTAL=$((EM_GRPO+EM_ASSR+BCOT_GRPO+BCOT_ASSR))
  log "  Phase 2 progress: $TOTAL/4 (em_grpo=$EM_GRPO em_assr=$EM_ASSR bcot_grpo=$BCOT_GRPO bcot_assr=$BCOT_ASSR)"
  [ "$TOTAL" -ge 4 ] && break
  
  # Run partial jobs for any completed Phase 2 results
  if [ "$BCOT_GRPO" -eq 1 ] && [ ! -f results/bcot_type1/bcot_t1_grpo_all.json ]; then
    log "  Phase 2 BCOT GRPO done -> running BCOT Type-1 for grpo"
    $PY scripts/experiments/bcot_type1_reactivation.py --method grpo \
      --from-result results/pure_rl_cleanup_bcot/pure_grpo_bcot_result.json 2>&1 | tee -a "$LOG"
  fi
  if [ "$BCOT_ASSR" -eq 1 ] && [ ! -f results/bcot_type1/bcot_t1_assr_no_sft_all.json ]; then
    log "  Phase 2 BCOT ASSR done -> running BCOT Type-1 for assr_no_sft"
    $PY scripts/experiments/bcot_type1_reactivation.py --method assr_no_sft \
      --from-result results/pure_rl_cleanup_bcot/pure_assr_bcot_result.json 2>&1 | tee -a "$LOG"
  fi
  
  sleep 300
done
log "Phase 2 complete!"

# ── Job 4: BCOT Type-1 for remaining methods (grpo, assr_no_sft) ────────
if [ ! -f results/bcot_type1/bcot_t1_grpo_all.json ]; then
  log "Job 4a: BCOT Type-1 for grpo"
  $PY scripts/experiments/bcot_type1_reactivation.py --method grpo \
    --from-result results/pure_rl_cleanup_bcot/pure_grpo_bcot_result.json 2>&1 | tee -a "$LOG"
fi
if [ ! -f results/bcot_type1/bcot_t1_assr_no_sft_all.json ]; then
  log "Job 4b: BCOT Type-1 for assr_no_sft"
  $PY scripts/experiments/bcot_type1_reactivation.py --method assr_no_sft \
    --from-result results/pure_rl_cleanup_bcot/pure_assr_bcot_result.json 2>&1 | tee -a "$LOG"
fi

# ── Job 5: EM Type-1 for new cleanups ───────────────────────────────────
log "Job 5: EM Type-1 dose-response for pure GRPO and pure ASSR"
$PY scripts/experiments/reeval_pipeline.py --em-type1-new 2>&1 | tee -a "$LOG"

# ── Job 6: Final re-eval sweep ──────────────────────────────────────────
log "Job 6: Final re-eval sweep (catch Phase 2 results)"
$PY scripts/experiments/reeval_pipeline.py --reeval-em 2>&1 | tee -a "$LOG"
$PY scripts/experiments/reeval_pipeline.py --reeval-bcot 2>&1 | tee -a "$LOG"

log "=========================================="
log "ALL JOBS COMPLETE"
log "=========================================="
