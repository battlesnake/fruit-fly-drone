#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-variable-height-full-native-d-first-fp64-update25-polished"
output_dir="$repo_root/runs/variable-height-hover/full-native-d-first-fp64-update25-slsqp-polished-audit-001"
mkdir -p "$task_tmp"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"

cd "$repo_root"
if [[ -e "$output_dir/report.json" ]]; then
  echo "final SLSQP-support-polished report already exists; refusing to retry" >&2
  exit 2
fi

exec_phase() {
  aira confine --memory-reserve 2G --memory-max 12G -- \
    .venv/bin/python \
      scripts/audit_variable_height_full_native_d_first_fp64_update25_slsqp_polished.py \
      "$@"
}

if [[ -e "$output_dir/verifier-started.json" ]]; then
  exec_phase --finalize-interrupted
  exit 0
fi
exec_phase
