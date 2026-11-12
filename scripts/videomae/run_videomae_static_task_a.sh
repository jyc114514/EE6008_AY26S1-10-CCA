#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  printf 'Usage: bash %s GPU_INDEX\n' "$0" >&2
  exit 2
fi

project='/usr1/home/s126mdg33_02/EE6008 AY26S1-10-CCA Project'
run="$project/tmp/s126mdg33_02/videomae_baseline_20260907"
run_jyc="$project/envs/run_jyc.sh"
export HF_HOME="$project/tmp/s126mdg33_02/hf_cache"
export CUDA_VISIBLE_DEVICES="$1"

cd "$run"
mkdir -p logs/static
log="logs/static/run_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1

printf 'Static VideoMAE run, UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Run directory: %s\n' "$run"

"$run_jyc" python scripts/extract_videomae_features_task_a.py \
  --split validation --variant static --max-episodes 8 \
  --output-dir "$run/outputs/smoke_static" \
  --batch-size 4 --local-files-only

for split in validation train; do
  "$run_jyc" python scripts/extract_videomae_features_task_a.py \
    --split "$split" --variant static \
    --batch-size 4 --local-files-only
done

for seed in 6008 7 42; do
  "$run_jyc" python scripts/train_videomae_linear_probe_task_a.py \
    --variant static --seed "$seed" --cpu
done

"$run_jyc" python "$HOME/compare_videomae_static_dynamic.py"
printf 'Static VideoMAE run completed, UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
