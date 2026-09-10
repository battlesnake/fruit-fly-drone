#!/usr/bin/env python3
"""Repeat the D-first correction with an authoritative D row and C guard band."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-canonical-guard-band-audit-v1"
PROTOCOL_COMMIT = "4a51b39"
FAILED_AUDIT_EXPERIMENT = audit.EXPERIMENT
FAILED_AUDIT_PROTOCOL_COMMIT = "236671c"
FAILED_AUDIT_IMPLEMENTATION_COMMIT = "2b38fd0"
EXPECTED_FAILED_AUDIT_REPORT_SHA256 = (
    "fcdd9842ba9e8d0ed652525409af76cb5cd18353ce02d606df5b8bbc6cf42b5f"
)
SOLVER_ENDPOINT_COMMON_INTERIOR_MARGIN = 0.0198
ACCEPTANCE_ENDPOINT_COMMON_INTERIOR_MARGIN = 0.0199
_ORIGINAL_VALIDATE_ARGS = audit.validate_args
_ORIGINAL_PROTOCOL_MANIFEST = audit.protocol_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument(
        "--source-audit-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-endpoint-damping-step-audit-001"
        ),
    )
    parser.add_argument(
        "--source-fit-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-canonical-fitting-001"
        ),
    )
    parser.add_argument(
        "--failed-audit-report",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-nonlinear-correction-audit-001/report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-canonical-guard-band-audit-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    _ORIGINAL_VALIDATE_ARGS(args)
    if not args.failed_audit_report.is_file():
        raise SystemExit(f"missing input: {args.failed_audit_report}")
    report_sha256 = responsibility.file_sha256(args.failed_audit_report)
    if report_sha256 != EXPECTED_FAILED_AUDIT_REPORT_SHA256:
        raise SystemExit("failed correction-audit report hash does not match the protocol")
    report = json.loads(args.failed_audit_report.read_text(encoding="utf-8"))
    identity = (
        report.get("experiment"),
        report.get("protocol", {}).get("protocol_commit"),
        report.get("classification"),
        report.get("pass"),
    )
    expected = (
        FAILED_AUDIT_EXPERIMENT,
        FAILED_AUDIT_PROTOCOL_COMMIT,
        "audit_control_failure",
        False,
    )
    if identity != expected:
        raise SystemExit("failed correction-audit report identity does not match the protocol")


def protocol_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(_ORIGINAL_PROTOCOL_MANIFEST())
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "preserves_failed_audit": {
                "experiment": FAILED_AUDIT_EXPERIMENT,
                "implementation_commit": FAILED_AUDIT_IMPLEMENTATION_COMMIT,
                "report_sha256": EXPECTED_FAILED_AUDIT_REPORT_SHA256,
                "thresholds_relaxed": False,
            },
            "failure_action": (
                "stop correction route; do not add residual repair or change a threshold"
            ),
        }
    )
    manifest["correction"].update(
        {
            "endpoint_damping_row": ("dedicated accumulated normalized endpoint-D objective"),
            "duplicate_multi_loss_row": ("reported under prior failed limits; diagnostic only"),
            "gradient_validation": {
                "method": "directional finite difference",
                "probe": "one-quarter interpolation from starting candidate to update 8",
                "objective": "normalized endpoint-D squared error",
                "relative_error_maximum": (
                    audit.DIRECTIONAL_FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
                ),
                "minimum_absolute_change": (
                    audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
                ),
                "finite": True,
                "native_parameter_bounds": True,
                "starting_candidate_restored_exactly": True,
            },
            "solver_endpoint_common_nrmse_maximum_source_plus": (
                SOLVER_ENDPOINT_COMMON_INTERIOR_MARGIN
            ),
            "acceptance_endpoint_common_nrmse_maximum_source_plus": (
                ACCEPTANCE_ENDPOINT_COMMON_INTERIOR_MARGIN
            ),
            "solver_and_acceptance_specs_distinct": True,
            "ideal_continuous_projection_checked_against": "solver specifications",
            "canonical_scaled_projection_checked_against": ("acceptance specifications"),
            "task_or_feasibility_threshold_relaxed": False,
        }
    )
    return manifest


def main() -> int:
    overrides = {
        "EXPERIMENT": EXPERIMENT,
        "PROTOCOL_COMMIT": PROTOCOL_COMMIT,
        "SOLVER_ENDPOINT_COMMON_INTERIOR_MARGIN": (SOLVER_ENDPOINT_COMMON_INTERIOR_MARGIN),
        "ENDPOINT_COMMON_INTERIOR_MARGIN": (ACCEPTANCE_ENDPOINT_COMMON_INTERIOR_MARGIN),
        "USE_DISTINCT_SOLVER_ACCEPTANCE_SPECS": True,
        "USE_AUTHORITATIVE_ENDPOINT_DAMPING_ROW": True,
        "DIRECTIONAL_FINITE_DIFFERENCE_REQUIRED": True,
        "parse_args": parse_args,
        "validate_args": validate_args,
        "protocol_manifest": protocol_manifest,
    }
    previous = {name: getattr(audit, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(audit, name, value)
        return audit.main()
    finally:
        for name, value in previous.items():
            setattr(audit, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
