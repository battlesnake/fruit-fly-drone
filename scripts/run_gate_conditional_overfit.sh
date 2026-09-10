#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training_cache_dir="/home/mark/tmp/fly-conditional-overfit"
mkdir -p "$training_cache_dir/torchinductor" "$training_cache_dir/triton"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCHINDUCTOR_CACHE_DIR="$training_cache_dir/torchinductor"
export TRITON_CACHE_DIR="$training_cache_dir/triton"

cd "$repo_root"
exec aira confine --memory-reserve 2G --memory-max 16G -- \
    .venv/bin/python scripts/audit_gate_conditional_overfit.py "$@"
