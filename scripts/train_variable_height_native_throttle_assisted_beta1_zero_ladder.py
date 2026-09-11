#!/usr/bin/env python3
"""Run the preregistered beta1=0 assisted-throttle derivative-ladder curriculum."""

from __future__ import annotations

from pathlib import Path

import train_variable_height_native_throttle_assisted as train

EXPERIMENT = "variable-height-native-throttle-assisted-beta1-zero-ladder-v1"
PROTOCOL_COMMIT = "4b25f56"
OPTIMIZER_BETAS = (0.0, 0.999)
DERIVATIVE_PROBE_SCALES = (1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512)
DERIVATIVE_BASELINE_REPEATS = 3
DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES = 2
DERIVATIVE_REPLAY_NOISE_MULTIPLIER = 10.0
DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE = 1.0e-8
EXPECTED_TAYLOR_AUDIT_SHA256 = (
    "0ee33e31f913a3de8e3b27934c1205706f4a417c7b3a33f12e190d4084930e50"
)
TAYLOR_AUDIT_REPORT = Path(
    "runs/variable-height-hover/native-throttle-beta1-zero-taylor-audit-001/report.json"
)


def configure_protocol() -> None:
    train.EXPERIMENT = EXPERIMENT
    train.PROTOCOL_COMMIT = PROTOCOL_COMMIT
    train.OPTIMIZER_BETAS = OPTIMIZER_BETAS
    train.DERIVATIVE_PROBE_SCALES = DERIVATIVE_PROBE_SCALES
    train.DERIVATIVE_BASELINE_REPEATS = DERIVATIVE_BASELINE_REPEATS
    train.DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES = DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES
    train.DERIVATIVE_REPLAY_NOISE_MULTIPLIER = DERIVATIVE_REPLAY_NOISE_MULTIPLIER
    train.DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE = DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE
    train.AUTHORIZING_TAYLOR_AUDIT_SHA256 = EXPECTED_TAYLOR_AUDIT_SHA256


def validate_authorizing_audit(path: Path = TAYLOR_AUDIT_REPORT) -> None:
    resolved = path if path.is_absolute() else train.REPO_ROOT / path
    if not resolved.is_file():
        raise SystemExit(f"missing authorizing Taylor audit: {resolved}")
    if train.responsibility.file_sha256(resolved) != EXPECTED_TAYLOR_AUDIT_SHA256:
        raise SystemExit("authorizing Taylor audit hash mismatch")


def main() -> int:
    configure_protocol()
    validate_authorizing_audit()
    return train.main()


if __name__ == "__main__":
    raise SystemExit(main())
