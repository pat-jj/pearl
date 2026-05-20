#!/usr/bin/env bash
# auto_eval_assr_intermediates.sh — polls the BCOT ASSR Tinker checkpoints
# files; whenever a new intermediate checkpoint is saved during RL, run a
# 200-pair BCOT eval against it. Helps us see the trajectory without
# waiting for step 60.
set -uo pipefail
PROJECT=.
cd "$PROJECT"

LOG="$PROJECT/results/auto_eval_assr_intermediates.log"
EVAL_DIR="$PROJECT/results/eval_intermediate"
EVAL_DONE_DIR="$EVAL_DIR/.done"
mkdir -p "$EVAL_DIR" "$EVAL_DONE_DIR"

PYTHON_BIN=python
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_one_eval() {
    local label="$1"
    local sampler_url="$2"
    local marker="$EVAL_DONE_DIR/$label"
    if [ -e "$marker" ]; then
        return 0
    fi
    log "EVAL $label  ($sampler_url)"
    local script="/tmp/auto_eval_$$.py"
    cat > "$script" << EOF
import asyncio, argparse, sys
sys.path.insert(0, '$PROJECT')
import code.tinker.backdoor_cot_v3_pipeline as p
p._MODEL_NAME = 'openai/gpt-oss-20b'
args = argparse.Namespace(eval_samples=200, judge_model='gpt-4o-mini', max_new_tokens=300)
asyncio.run(p.stage_evaluate(args, 'openai/gpt-oss-20b', 'gpt_oss_20b', None,
    '$sampler_url', '$label'))
EOF
    if "$PYTHON_BIN" -u "$script" >> "$EVAL_DIR/$label.log" 2>&1; then
        log "DONE $label"
        touch "$marker"
        rm -f "$script"
    else
        log "FAILED $label (will retry next pass)"
    fi
}

# Watch loop
log "===== auto_eval_assr_intermediates START ====="
while true; do
    # 1. With-warmup ASSR (Option A): tinker_logs/backdoor_cot_v3/v3_cleanup_assr_gptoss_20b_s42/checkpoints.jsonl
    JSONL_W="$PROJECT/tinker_logs/backdoor_cot_v3/v3_cleanup_assr_gptoss_20b_s42/checkpoints.jsonl"
    if [ -f "$JSONL_W" ]; then
        # parse last line per name (assr_NNNNNN), take latest URLs
        while read -r line; do
            [ -z "$line" ] && continue
            name=$(echo "$line" | "$PYTHON_BIN" -c "import sys,json; print(json.loads(sys.stdin.read()).get('name',''))")
            sampler=$(echo "$line" | "$PYTHON_BIN" -c "import sys,json; print(json.loads(sys.stdin.read()).get('sampler_path',''))")
            [ -z "$name" ] && continue
            label="assr_v2_optionA_warmup_$name"
            run_one_eval "$label" "$sampler"
        done < "$JSONL_W"
    fi

    # 2. No-warmup ASSR: tinker_logs/pure_assr_bcot/checkpoints.jsonl
    JSONL_N="$PROJECT/tinker_logs/pure_assr_bcot/checkpoints.jsonl"
    if [ -f "$JSONL_N" ]; then
        while read -r line; do
            [ -z "$line" ] && continue
            name=$(echo "$line" | "$PYTHON_BIN" -c "import sys,json; print(json.loads(sys.stdin.read()).get('name',''))")
            sampler=$(echo "$line" | "$PYTHON_BIN" -c "import sys,json; print(json.loads(sys.stdin.read()).get('sampler_path',''))")
            [ -z "$name" ] && continue
            label="assr_v2_optionA_nowarmup_$name"
            run_one_eval "$label" "$sampler"
        done < "$JSONL_N"
    fi

    sleep 600  # poll every 10 min
done
