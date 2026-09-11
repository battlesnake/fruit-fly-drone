#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-vertical-motion-derivative-ladder"
output_dir="$repo_root/runs/optic-motion/vertical-motion-derivative-ladder-001"
mkdir -p "$task_tmp" "$output_dir"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONWARNINGS=error

cd "$repo_root"
aira confine --memory-reserve 4G --memory-max 28G -- \
  .venv/bin/python scripts/audit_vertical_motion_derivative_ladder.py "$@"

.venv/bin/python scripts/check_vertical_motion_derivative_ladder.py \
  "$output_dir/report.json"
