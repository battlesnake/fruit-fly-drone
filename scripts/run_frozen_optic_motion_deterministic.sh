#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_tmp="/home/mark/tmp/fly-frozen-optic-motion-deterministic"
output_dir="$repo_root/runs/optic-motion/frozen-t4t5-audit-002"
mkdir -p "$task_tmp" "$output_dir"

export TMPDIR="$task_tmp"
export CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONWARNINGS=error

cd "$repo_root"
for replay in 1 2 3; do
  aira confine --memory-reserve 4G --memory-max 28G -- \
    .venv/bin/python scripts/evaluate_frozen_optic_motion_deterministic_replay.py \
    --replay-index "$replay" \
    --output "$output_dir/replay-$replay.json"
done

.venv/bin/python scripts/check_frozen_optic_motion_deterministic_replays.py \
  "$output_dir/replay-1.json" \
  "$output_dir/replay-2.json" \
  "$output_dir/replay-3.json" \
  --output "$output_dir/replay-gate.json"

aira confine --memory-reserve 4G --memory-max 28G -- \
  .venv/bin/python scripts/audit_frozen_optic_motion_deterministic.py "$@"

.venv/bin/python scripts/check_frozen_optic_motion_deterministic.py \
  "$output_dir/report.json"
