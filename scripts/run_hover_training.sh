#!/usr/bin/env bash
set -euo pipefail

hover_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hover_cache_dir="/home/mark/tmp/fly-hover-training"

mkdir -p "${hover_cache_dir}/torch" "${hover_cache_dir}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${hover_cache_dir}/torch"
export MPLCONFIGDIR="${hover_cache_dir}/matplotlib"

# WSL's paravirtualized libcuda must precede any native Linux driver stub.
if [[ -e /usr/lib/wsl/lib/libcuda.so ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec aira confine --memory-reserve 4G --memory-max 16G -- \
    "${hover_repo_dir}/.venv/bin/python" \
    "${hover_repo_dir}/scripts/train_hover.py" "$@"
