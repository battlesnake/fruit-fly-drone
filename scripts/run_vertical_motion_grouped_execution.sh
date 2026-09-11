#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-vertical-motion-grouped-execution-v1"
output_dir="$repo_root/runs/optic-motion/vertical-motion-grouped-preflight-001"

for argument in "$@"; do
  case "$argument" in
    --output-dir | --output-dir=*)
      echo "run_vertical_motion_grouped_execution.sh fixes --output-dir for checking" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$task_tmp" "$output_dir"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONWARNINGS=error

cd "$repo_root"
aira confine --memory-reserve 8G --memory-max 28G -- \
  .venv/bin/python scripts/preflight_vertical_motion_grouped_execution.py "$@"

.venv/bin/python scripts/check_vertical_motion_grouped_execution.py \
  "$output_dir/report.json"
