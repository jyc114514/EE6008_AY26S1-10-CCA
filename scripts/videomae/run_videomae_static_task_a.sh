#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  printf 'Usage: bash %s GPU_INDEX\n' "$0" >&2
  exit 2
fi

: "${EE6008_PROJECT_ROOT:?Set EE6008_PROJECT_ROOT to the project directory}"
: "${EE6008_VIDEOMAE_SCRATCH_NAME:?Set EE6008_VIDEOMAE_SCRATCH_NAME to a scratch directory name}"
: "${VIDEOMAE_ENV_LAUNCHER:?Set VIDEOMAE_ENV_LAUNCHER to the Python environment wrapper}"
: "${VIDEOMAE_MANIFEST:?Set VIDEOMAE_MANIFEST to the train/validation manifest}"
: "${VIDEOMAE_G1_ROOT:?Set VIDEOMAE_G1_ROOT to the authorized G1 data directory}"
: "${VIDEOMAE_CONFIG:?Set VIDEOMAE_CONFIG to the VideoMAE profile}"

absolute_file_path() {
  local path="$1"
  local parent
  parent="$(cd -- "$(dirname -- "$path")" && pwd -P)" || return 1
  printf '%s/%s\n' "$parent" "$(basename -- "$path")"
}


if [[ ! "$EE6008_VIDEOMAE_SCRATCH_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
  printf 'EE6008_VIDEOMAE_SCRATCH_NAME must be a single safe directory name.\n' >&2
  exit 2
fi
if [[ ! -f "$VIDEOMAE_ENV_LAUNCHER" || ! -x "$VIDEOMAE_ENV_LAUNCHER" ]]; then
  printf 'VIDEOMAE_ENV_LAUNCHER must point to an executable file.\n' >&2
  exit 2
fi
if [[ ! -f "$VIDEOMAE_MANIFEST" || ! -d "$VIDEOMAE_G1_ROOT" || ! -f "$VIDEOMAE_CONFIG" ]]; then
  printf 'The manifest, G1 data directory, and config must exist.\n' >&2
  exit 2
fi

if ! VIDEOMAE_ENV_LAUNCHER="$(absolute_file_path "$VIDEOMAE_ENV_LAUNCHER")"; then
  printf 'VIDEOMAE_ENV_LAUNCHER must resolve to an existing parent directory.\n' >&2
  exit 2
fi
if ! VIDEOMAE_MANIFEST="$(absolute_file_path "$VIDEOMAE_MANIFEST")"; then
  printf 'VIDEOMAE_MANIFEST must resolve to an existing parent directory.\n' >&2
  exit 2
fi
if ! VIDEOMAE_CONFIG="$(absolute_file_path "$VIDEOMAE_CONFIG")"; then
  printf 'VIDEOMAE_CONFIG must resolve to an existing parent directory.\n' >&2
  exit 2
fi
if ! VIDEOMAE_G1_ROOT="$(cd -- "$VIDEOMAE_G1_ROOT" && pwd -P)"; then
  printf 'VIDEOMAE_G1_ROOT must resolve to an existing directory.\n' >&2
  exit 2
fi
if [[ -n "${HF_HOME:-}" && "$HF_HOME" != /* ]]; then
  printf 'HF_HOME must be absolute when supplied.\n' >&2
  exit 2
fi

project="$(cd -- "$EE6008_PROJECT_ROOT" && pwd -P)"
if [[ ! -d "$project/tmp" ]]; then
  printf 'The project tmp directory does not exist.\n' >&2
  exit 2
fi
tmp_root="$(cd -- "$project/tmp" && pwd -P)"
scratch_path="$project/tmp/$EE6008_VIDEOMAE_SCRATCH_NAME"
if [[ ! -d "$scratch_path" ]]; then
  printf 'The configured scratch directory does not exist.\n' >&2
  exit 2
fi
scratch="$(cd -- "$scratch_path" && pwd -P)"
if [[ "$scratch" == "$tmp_root" ]]; then
  printf 'The configured scratch directory must be below project tmp.\\n' >&2
  exit 2
fi
case "$scratch/" in
  "$tmp_root/"*) ;;
  *)
    printf 'The configured scratch directory resolves outside project tmp.\n' >&2
    exit 2
    ;;
esac

run_path="$scratch/videomae_baseline_20260907"
if [[ ! -d "$run_path" ]]; then
  printf 'The configured VideoMAE run directory does not exist.\n' >&2
  exit 2
fi
run="$(cd -- "$run_path" && pwd -P)"
if [[ "$run" == "$scratch" ]]; then
  printf 'The VideoMAE run directory must be below configured scratch.\\n' >&2
  exit 2
fi
case "$run/" in
  "$scratch/"*) ;;
  *)
    printf 'The VideoMAE run directory resolves outside configured scratch.\n' >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
run_jyc="$VIDEOMAE_ENV_LAUNCHER"
export HF_HOME="${HF_HOME:-$scratch/hf_cache}"
export CUDA_VISIBLE_DEVICES="$1"

cd -- "$run"
mkdir -p logs/static
log="logs/static/run_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1

printf 'Static VideoMAE run, UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Run directory: %s\n' "$run"

"$run_jyc" python "$script_dir/extract_videomae_features_task_a.py" \
  --config "$VIDEOMAE_CONFIG" --manifest "$VIDEOMAE_MANIFEST" \
  --g1-root "$VIDEOMAE_G1_ROOT" --output-dir "$run/outputs/smoke_static" \
  --split validation --variant static --max-episodes 8 \
  --batch-size 4 --local-files-only

for split in validation train; do
  "$run_jyc" python "$script_dir/extract_videomae_features_task_a.py" \
    --config "$VIDEOMAE_CONFIG" --manifest "$VIDEOMAE_MANIFEST" \
    --g1-root "$VIDEOMAE_G1_ROOT" --output-dir "$run/outputs/features" \
    --split "$split" --variant static \
    --batch-size 4 --local-files-only
done

for seed in 6008 7 42; do
  "$run_jyc" python "$script_dir/train_videomae_linear_probe_task_a.py" \
    --feature-dir "$run/outputs/features" \
    --output-dir "$run/outputs/linear_probe_static" \
    --variant static --seed "$seed" --cpu
done

"$run_jyc" python "$script_dir/compare_videomae_static_dynamic.py" --run-dir "$run"
printf 'Static VideoMAE run completed, UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
