#!/usr/bin/env bash
set -euo pipefail

visual_hover_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
visual_hover_cache_dir="/home/mark/tmp/fly-visual-hover-training"

mkdir -p "${visual_hover_cache_dir}/torch"
export TORCHINDUCTOR_CACHE_DIR="${visual_hover_cache_dir}/torch"

if [[ -e /usr/lib/wsl/lib/libcuda.so ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec aira confine --memory-reserve 4G --memory-max 16G -- \
    "${visual_hover_repo_dir}/.venv/bin/python" \
    "${visual_hover_repo_dir}/scripts/train_visual_hover.py" "$@"
