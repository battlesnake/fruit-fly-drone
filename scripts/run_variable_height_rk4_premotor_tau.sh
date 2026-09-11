#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-variable-height-rk4-premotor-tau"
mkdir -p "$task_tmp"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"

cd "$repo_root"
exec aira confine --memory-reserve 4G --memory-max 28G -- \
  .venv/bin/python \
  scripts/audit_variable_height_rk4_premotor_tau.py \
  --output-dir runs/variable-height-hover/native-rk4-premotor-tau-preflight-002 "$@"
