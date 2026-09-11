#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-variable-height-native-throttle-assisted-beta1-zero"
mkdir -p "$task_tmp"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"

cd "$repo_root"
exec aira confine --memory-reserve 2G --memory-max 12G -- \
  .venv/bin/python scripts/train_variable_height_native_throttle_assisted_beta1_zero.py \
  --output-dir runs/variable-height-hover/native-throttle-assisted-beta1-zero-001 "$@"
