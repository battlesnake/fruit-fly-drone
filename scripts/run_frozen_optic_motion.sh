#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-frozen-optic-motion"
mkdir -p "$task_tmp"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"

cd "$repo_root"
exec aira confine --memory-reserve 4G --memory-max 28G -- \
  .venv/bin/python scripts/audit_frozen_optic_motion.py \
  --output-dir runs/optic-motion/frozen-t4t5-audit-001 "$@"
