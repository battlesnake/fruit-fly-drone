#!/usr/bin/env bash
set -euo pipefail

probe_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
probe_cache_dir="/home/mark/tmp/fly-headless-spike/warp-cache"

mkdir -p "${probe_cache_dir}"
export WARP_CACHE_PATH="${probe_cache_dir}"

# A native Linux NVIDIA driver can shadow WSL's paravirtualized libcuda and make Warp
# report zero devices even while PyTorch works. Prefer the WSL library only when present.
if [[ -e /usr/lib/wsl/lib/libcuda.so ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec aira confine --memory-reserve 4G --memory-max 8G -- \
    "${probe_repo_dir}/.venv/bin/python" \
    "${probe_repo_dir}/scripts/probe_headless_stack.py" "$@"
