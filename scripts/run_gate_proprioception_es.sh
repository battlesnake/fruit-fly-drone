#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCHINDUCTOR_CACHE_DIR="$repo_root/.cache/torchinductor"
export TRITON_CACHE_DIR="$repo_root/.cache/triton"

exec aira confine --memory-reserve 4G --memory-max 16G -- \
  .venv/bin/python scripts/search_gate_proprioception_es.py "$@"
