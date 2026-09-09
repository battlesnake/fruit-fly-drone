#!/usr/bin/env bash
set -euo pipefail

gate_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gate_cache_dir="/home/mark/tmp/fly-gate-training"

mkdir -p "${gate_cache_dir}/torch" "${gate_cache_dir}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${gate_cache_dir}/torch"
export MPLCONFIGDIR="${gate_cache_dir}/matplotlib"

if [[ -e /usr/lib/wsl/lib/libcuda.so ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec aira confine --memory-reserve 4G --memory-max 16G -- \
    "${gate_repo_dir}/.venv/bin/python" \
    "${gate_repo_dir}/scripts/train_gate.py" "$@"
