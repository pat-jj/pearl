#!/bin/bash
# monitor_assr_v2.sh — print a single status snapshot for all running ASSR v2 jobs.
set -uo pipefail
PROJECT=.

ts=$(date +'%Y-%m-%d %H:%M:%S')
echo "================================================================"
echo "= ASSR v2 status @ ${ts}"
echo "================================================================"

# Tail helper: prints last N matching/relevant lines if file exists, else "(no file)"
ttail() {
  local label="$1" file="$2" n="${3:-2}"
  echo
  echo "-- ${label} --"
  if [ -f "$file" ]; then
    tail -n "$n" "$file"
  else
    echo "  (no log: $file)"
  fi
}

ctail() {
  local label="$1" sess="$2" n="${3:-3}"
  echo
  echo "-- ${label} (tmux:${sess}) --"
  if tmux has-session -t "$sess" 2>/dev/null; then
    tmux capture-pane -t "$sess" -p -S -50 | grep -v '^$' | tail -n "$n"
  else
    echo "  (session ${sess} not running)"
  fi
}

# ── EM cleanup runs ──
ttail "pure_assr_em (no-warmup cleanup)"    "${PROJECT}/results/pure_assr_em_rerun_v3b.log" 2
ttail "assr_em_rerun (with-warmup cleanup)" "${PROJECT}/results/assr_em_rerun_v3.log" 2

# ── BCOT cleanup runs ──
ttail "pure_assr_bcot (no-warmup cleanup)"    "${PROJECT}/results/pure_assr_bcot_rerun_v3b.log" 3
ttail "assr_bcot_rerun (with-warmup cleanup)" "${PROJECT}/results/assr_bcot_warmup_v3b.log" 3

# ── EM Type-1 ──
ttail "assr_em_type1 (multi-pass Type-1)" "${PROJECT}/results/assr_em_type1.log" 3
ctail "queue_em_assr_nosft (Type-1+Type-2)" queue_em_assr_nosft 4
ctail "queue_em_grpo (Type-2)"              queue_em_grpo 4

# ── BCOT Type-1 batch (older, on old-assr ckpt) ──
ttail "bcot_type1_parallel" "${PROJECT}/results/bcot_type1_parallel.log" 3

# ── Result file counts ──
echo
echo "-- Result files committed so far --"
em_t1=$(ls "${PROJECT}/results/em_type1_new/dr_"*.json 2>/dev/null | wc -l)
em_t2=$(ls "${PROJECT}/results/type2_open_thoughts_em_v2/t2ot_"*.json 2>/dev/null | wc -l)
bcot_t1=$(ls "${PROJECT}/results/bcot_type1/bcot_t1_"*.json 2>/dev/null | wc -l)
bcot_t2=$(ls "${PROJECT}/results/type2_open_thoughts_bcot_v2/t2ot_"*.json 2>/dev/null | wc -l)
echo "  EM Type-1 result files: ${em_t1}"
echo "  EM Type-2 result files: ${em_t2}"
echo "  BCOT Type-1 result files: ${bcot_t1}"
echo "  BCOT Type-2 result files: ${bcot_t2}"

# ── Cleanup outputs that just landed ──
echo
echo "-- New cleanup result JSONs (last 30 minutes) --"
find "${PROJECT}/results/pure_rl_cleanup_em" "${PROJECT}/results/pure_rl_cleanup_bcot" \
     "${PROJECT}/tinker_logs" -maxdepth 3 -name "*.json" -mmin -30 2>/dev/null | head -20

echo
echo "================================================================"
