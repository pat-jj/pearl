#!/usr/bin/env bash
set -euo pipefail

PROJECT=.
cd "$PROJECT"

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate trl
source ${PROJECT_PARENT}/.apikey
export OPENAI_API_KEY TINKER_API_KEY ANTHROPIC_API_KEY
export PYTHONPATH="$PROJECT"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="$PROJECT/results"
ORGANISM=models/backdoor_cot_v3/organism_28_s42

wait_for_gpu_pair() {
  while true; do
    pair="$(python - <<'PY'
import subprocess

out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)

candidates = []
for line in out.strip().splitlines():
    idx, total, used, util = [int(x.strip()) for x in line.split(",")]
    free = total - used
    if free >= 25000 and util <= 50:
        candidates.append((idx, free, util))

candidates.sort(key=lambda x: (-x[1], x[2]))
if len(candidates) >= 2:
    print(f"{candidates[0][0]},{candidates[1][0]}")
PY
)"
    if [ -n "$pair" ]; then
      echo "$pair"
      return 0
    fi
    echo "[$(date '+%F %T')] Waiting for two GPUs with >=25GB free and <=50% util..." | tee -a "$LOG_DIR/qwen_assr_queue_0504.log"
    sleep 900
  done
}

run_one() {
  local tag="$1"
  local out_dir="$2"
  local n_prefix="$3"
  local n_samples="$4"
  local log="$5"
  local pair
  pair="$(wait_for_gpu_pair)"
  echo "[$(date '+%F %T')] Starting ${tag} on CUDA_VISIBLE_DEVICES=${pair}" | tee -a "$log" "$LOG_DIR/qwen_assr_queue_0504.log"
  CUDA_VISIBLE_DEVICES="$pair" python -u -m code.tools.run_backdoor_cot_v3_cleanup_assr_v2 \
    --from-checkpoint "$ORGANISM" \
    --output-dir "$out_dir" \
    --label "$tag" \
    --seed 42 \
    --skip-phase1 \
    --skip-phase2 \
    --batch-size 4 \
    --epochs 1 \
    --n-prefix-cuts "$n_prefix" \
    --n-samples-per-ctx "$n_samples" \
    --save-every 25 \
    2>&1 | tee -a "$log"
}

run_one assr_v2_nowarmup_28_g4_p1 \
  models/backdoor_cot_v3/assr_v2_nowarmup_28_s42_g4_p1 \
  1 4 "$LOG_DIR/qwen_assr_g4_p1.log"

run_one assr_v2_nowarmup_28_g8_p2 \
  models/backdoor_cot_v3/assr_v2_nowarmup_28_s42_g8_p2 \
  2 8 "$LOG_DIR/qwen_assr_g8_p2.log"
