#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-variable-height-damping-route-bounded-preflight"
mkdir -p "$task_tmp"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"

cd "$repo_root"
exec aira confine --memory-reserve 2G --memory-max 12G -- \
  .venv/bin/python scripts/audit_variable_height_damping_routes_v2.py "$@"
