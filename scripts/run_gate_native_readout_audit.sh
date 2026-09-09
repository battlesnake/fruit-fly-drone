#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
readout_cache_dir="/home/mark/tmp/fly-native-readout"
mkdir -p "$readout_cache_dir/torchinductor" "$readout_cache_dir/triton"
export TORCHINDUCTOR_CACHE_DIR="$readout_cache_dir/torchinductor"
export TRITON_CACHE_DIR="$readout_cache_dir/triton"

exec aira confine --memory-reserve 4G --memory-max 16G -- \
    .venv/bin/python scripts/audit_gate_native_readout.py "$@"
