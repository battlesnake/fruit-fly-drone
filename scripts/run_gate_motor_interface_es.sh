#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
search_cache_dir="/home/mark/tmp/fly-motor-interface-es"
mkdir -p "$search_cache_dir/torchinductor" "$search_cache_dir/triton"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCHINDUCTOR_CACHE_DIR="$search_cache_dir/torchinductor"
export TRITON_CACHE_DIR="$search_cache_dir/triton"

cd "$repo_root"
exec aira confine --memory-reserve 4G --memory-max 16G -- \
    .venv/bin/python scripts/search_gate_motor_interface_es.py "$@"
