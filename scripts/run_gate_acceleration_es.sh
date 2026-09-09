#!/usr/bin/env bash
set -euo pipefail

search_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
search_cache_dir="/home/mark/tmp/fly-gate-acceleration-es"

mkdir -p "${search_cache_dir}/torch" "${search_cache_dir}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${search_cache_dir}/torch"
export MPLCONFIGDIR="${search_cache_dir}/matplotlib"

if [[ -e /usr/lib/wsl/lib/libcuda.so ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec aira confine --memory-reserve 4G --memory-max 16G -- \
    "${search_repo_dir}/.venv/bin/python" \
    "${search_repo_dir}/scripts/search_gate_acceleration_es.py" "$@"
