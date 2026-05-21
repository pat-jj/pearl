#!/bin/bash
# Queue ASSR cleanup reruns with the paper fixes applied.
#
# Strategy: at most 2 RL training jobs concurrently. The two EM jobs
# (with-warmup and no-warmup) are already started; this script waits for
# them to finish, then kicks off the two BCOT jobs in sequence (or in
# parallel if a slot opens before the first BCOT finishes).
#
# The runs are submitted to dedicated tmux sessions so they survive
# disconnection. Inspect with: tmux attach -t <session>.
set -euo pipefail
cd .

PYTHON_BIN=python
APIKEY_FILE=${ARTIFACT_APIKEY_FILE:-.apikey}
PROJECT=.

src_env() {
  source "$APIKEY_FILE"
  export TINKER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
  export PYTHONPATH="$PROJECT"
}

is_tmux_active() {
  # Returns 0 when the tmux session has a *running* python process inside.
  local sess="$1"
  tmux list-sessions 2>/dev/null | grep -q "^$sess:" || return 1
  local panes
  panes=$(tmux list-panes -t "$sess" -F '#{pane_pid}' 2>/dev/null)
  for pid in $panes; do
    if pgrep -P "$pid" -f 'python' >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

run_in_session() {
  local sess="$1"; shift
  local cmd="$*"
  if ! tmux has-session -t "$sess" 2>/dev/null; then
    tmux new-session -d -s "$sess"
  fi
  tmux send-keys -t "$sess" "$cmd" C-m
}

count_running() {
  local n=0
  for s in assr_em_rerun pure_assr_em assr_bcot_rerun pure_assr_bcot; do
    if is_tmux_active "$s"; then n=$((n+1)); fi
  done
  echo "$n"
}

wait_for_slot() {
  while true; do
    local n
    n=$(count_running)
    if [ "$n" -lt 2 ]; then
      return 0
    fi
    echo "[$(date +%H:%M:%S)] $n/2 RL jobs running; waiting 60s..."
    sleep 60
  done
}

# ── 1) ASSR EM with warm-up (assr_em_rerun) ──
launch_assr_em() {
  echo "[$(date +%H:%M:%S)] launching assr_em_rerun"
  run_in_session assr_em_rerun \
    "cd $PROJECT && source $APIKEY_FILE && export TINKER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY && export PYTHONPATH=$PROJECT && $PYTHON_BIN scripts/experiments/rerun_assr_em_fixed.py 2>&1 | tee results/assr_em_rerun_paper.log"
}

# ── 2) ASSR EM no warm-up (pure_assr_em) ──
launch_pure_assr_em() {
  echo "[$(date +%H:%M:%S)] launching pure_assr_em"
  run_in_session pure_assr_em \
    "cd $PROJECT && source $APIKEY_FILE && export TINKER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY && export PYTHONPATH=$PROJECT && export PURE_ASSR_EM_STEPS=50 && export PURE_ASSR_EM_BATCH_SIZE=8 && $PYTHON_BIN scripts/experiments/pure_rl_cleanup.py --setting em --method assr 2>&1 | tee results/pure_assr_em_rerun_paper.log"
}

# ── 3) ASSR EM no warm-up + skip-warmup version of paper driver
#       (cleanup_assr_no_sft_em_gpt_oss_20b_s42) ──
launch_assr_em_no_sft_paper() {
  echo "[$(date +%H:%M:%S)] launching assr_em_no_sft (skip-warmup driver)"
  run_in_session assr_em_no_sft \
    "cd $PROJECT && source $APIKEY_FILE && export TINKER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY && export PYTHONPATH=$PROJECT && $PYTHON_BIN scripts/experiments/rerun_assr_em_fixed.py --skip-warmup 2>&1 | tee results/assr_em_no_sft_paper.log"
}

# ── 4) ASSR BCOT with warm-up (assr_bcot_rerun) ──
launch_assr_bcot_warmup() {
  echo "[$(date +%H:%M:%S)] launching assr_bcot_rerun (with warmup)"
  run_in_session assr_bcot_rerun \
    "cd $PROJECT && source $APIKEY_FILE && export TINKER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY && export PYTHONPATH=$PROJECT && $PYTHON_BIN -m code.tinker.backdoor_cot_pipeline --stage cleanup --model gpt_oss_20b --cleanup-methods assr --rl-steps 60 --rl-batch-size 16 --assr-n-prefix-cuts 2 --assr-n-samples-per-ctx 4 --assr-max-depth 256 2>&1 | tee results/assr_bcot_warmup_paper.log"
}

# ── 5) ASSR BCOT no warm-up (pure_assr_bcot) ──
launch_pure_assr_bcot() {
  echo "[$(date +%H:%M:%S)] launching pure_assr_bcot"
  run_in_session pure_assr_bcot \
    "cd $PROJECT && source $APIKEY_FILE && export TINKER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY && export PYTHONPATH=$PROJECT && $PYTHON_BIN scripts/experiments/pure_rl_cleanup.py --setting bcot --method assr 2>&1 | tee results/pure_assr_bcot_rerun_paper.log"
}

# ── Coordinator ──
main() {
  src_env
  echo "[$(date +%H:%M:%S)] queue start; existing running=$(count_running)/2"

  # If the EM jobs are not yet running, launch them.
  if ! is_tmux_active assr_em_rerun; then
    wait_for_slot && launch_assr_em && sleep 5
  fi
  if ! is_tmux_active pure_assr_em; then
    wait_for_slot && launch_pure_assr_em && sleep 5
  fi

  # Wait until at least one slot opens for the BCOT runs.
  echo "[$(date +%H:%M:%S)] waiting for slot to start BCOT runs..."
  wait_for_slot
  if ! is_tmux_active pure_assr_bcot; then
    launch_pure_assr_bcot && sleep 5
  fi

  echo "[$(date +%H:%M:%S)] waiting for slot to start with-warmup BCOT..."
  wait_for_slot
  if ! is_tmux_active assr_bcot_rerun; then
    launch_assr_bcot_warmup && sleep 5
  fi

  # Optionally also queue a no-SFT EM paper driver run if requested.
  if [ "${RUN_ASSR_EM_NO_SFT_V3:-0}" = "1" ] && ! is_tmux_active assr_em_no_sft; then
    echo "[$(date +%H:%M:%S)] waiting for slot to start assr_em_no_sft (paper driver)..."
    wait_for_slot
    launch_assr_em_no_sft_paper
  fi

  echo "[$(date +%H:%M:%S)] queue submitted. tmux ls:"; tmux ls
}

main "$@"
