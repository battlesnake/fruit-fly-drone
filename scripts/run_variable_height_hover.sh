#!/usr/bin/env bash
set -euo pipefail

variable_hover_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
variable_hover_cache_dir="/home/mark/tmp/fly-variable-height-hover"

mkdir -p "${variable_hover_cache_dir}/torch"
export TORCHINDUCTOR_CACHE_DIR="${variable_hover_cache_dir}/torch"

if [[ -e /usr/lib/wsl/lib/libcuda.so ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec aira confine --memory-reserve 4G --memory-max 16G -- \
    "${variable_hover_repo_dir}/.venv/bin/python" \
    "${variable_hover_repo_dir}/scripts/train_variable_height_hover.py" "$@"
