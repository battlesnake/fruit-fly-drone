#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-variable-height-neural-integration-rate-audit"
mkdir -p "$task_tmp"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"

cd "$repo_root"
exec aira confine --memory-reserve 2G --memory-max 16G -- \
  .venv/bin/python \
  scripts/audit_variable_height_neural_integration_rate.py \
  --output-dir runs/variable-height-hover/native-neural-integration-rate-audit-001 "$@"
