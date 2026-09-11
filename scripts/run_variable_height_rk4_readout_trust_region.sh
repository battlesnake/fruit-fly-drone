#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-variable-height-rk4-readout-trust-region"
mkdir -p "$task_tmp"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"

cd "$repo_root"
exec aira confine --memory-reserve 2G --memory-max 16G -- \
  .venv/bin/python \
  scripts/audit_variable_height_rk4_readout_trust_region.py \
  --output-dir runs/variable-height-hover/native-rk4-readout-trust-region-001 "$@"
