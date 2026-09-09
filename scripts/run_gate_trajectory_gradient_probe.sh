#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
probe_cache_dir="/home/mark/tmp/fly-trajectory-gradient-probe"
mkdir -p "$probe_cache_dir/torchinductor" "$probe_cache_dir/triton"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCHINDUCTOR_CACHE_DIR="$probe_cache_dir/torchinductor"
export TRITON_CACHE_DIR="$probe_cache_dir/triton"

cd "$repo_root"
exec aira confine --memory-reserve 4G --memory-max 16G -- \
    .venv/bin/python scripts/audit_gate_trajectory_gradients.py "$@"
