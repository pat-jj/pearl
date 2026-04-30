#!/bin/bash
# Resume all experiments after Tinker billing fix.
# Runs SEQUENTIALLY (one Tinker session at a time).
set -euo pipefail
cd .

set -a; source ${PROJECT_PARENT}/.apikey; set +a
PY=python
LOG=results/resume_run.log

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [RUN] $*" | tee -a "$LOG"; }

log "=========================================="
log "RESUMING ALL EXPERIMENTS"
log "=========================================="

# ── Phase 1 remaining: BCOT gc and ac Type-2 SFT ────────────────────────
log "Phase 1: BCOT Type-2 SFT (gc, ac)"
for cleanup in gc ac; do
  RESULT="results/type2_open_thoughts_bcot/t2ot_bcot_${cleanup}_sft_result.json"
  if [ -f "$RESULT" ]; then
    log "  SKIP t2ot_bcot_${cleanup}_sft (done)"
  else
    log "  Running t2ot_bcot_${cleanup}_sft..."
    $PY scripts/experiments/type2_open_thoughts.py --setting bcot --cleanup "$cleanup" --route sft 2>&1 | tee -a "$LOG"
  fi
done

# ── Phase 2: Pure RL cleanup ────────────────────────────────────────────
log "Phase 2: Pure RL cleanup"
for setting in em bcot; do
  for method in grpo assr; do
    RESULT="results/pure_rl_cleanup_${setting}/pure_${method}_${setting}_result.json"
    if [ -f "$RESULT" ]; then
      log "  SKIP pure_${method}_${setting} (done)"
    else
      log "  Running pure_${method}_${setting}..."
      $PY scripts/experiments/pure_rl_cleanup.py --setting "$setting" --method "$method" 2>&1 | tee -a "$LOG"
    fi
  done
done

# ── Re-eval EM with 8x100 ──────────────────────────────────────────────
log "Re-eval: EM 8x100"
$PY scripts/experiments/reeval_pipeline.py --reeval-em 2>&1 | tee -a "$LOG"

# ── Re-eval BCOT with cued_accuracy ─────────────────────────────────────
log "Re-eval: BCOT cued_acc"
$PY scripts/experiments/reeval_pipeline.py --reeval-bcot 2>&1 | tee -a "$LOG"

# ── BCOT Type-1 reactivation (5 base methods) ──────────────────────────
log "BCOT Type-1: base methods"
$PY scripts/experiments/bcot_type1_reactivation.py --all 2>&1 | tee -a "$LOG"

# ── BCOT Type-1 for Phase 2 models ─────────────────────────────────────
for pair in "grpo:pure_grpo_bcot_result.json" "assr_no_sft:pure_assr_bcot_result.json"; do
  METHOD="${pair%%:*}"; RFILE="${pair##*:}"
  RPATH="results/pure_rl_cleanup_bcot/$RFILE"
  if [ -f "$RPATH" ] && [ ! -f "results/bcot_type1/bcot_t1_${METHOD}_all.json" ]; then
    log "BCOT Type-1: $METHOD"
    $PY scripts/experiments/bcot_type1_reactivation.py --method "$METHOD" --from-result "$RPATH" 2>&1 | tee -a "$LOG"
  fi
done

# ── EM Type-1 for new cleanups ──────────────────────────────────────────
log "EM Type-1: new cleanups"
$PY scripts/experiments/reeval_pipeline.py --em-type1-new 2>&1 | tee -a "$LOG"

# ── Final re-eval sweep ─────────────────────────────────────────────────
log "Final re-eval sweep"
$PY scripts/experiments/reeval_pipeline.py --reeval-em 2>&1 | tee -a "$LOG"
$PY scripts/experiments/reeval_pipeline.py --reeval-bcot 2>&1 | tee -a "$LOG"

log "=========================================="
log "ALL EXPERIMENTS COMPLETE"
log "=========================================="
