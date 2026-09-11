#!/usr/bin/env python3
"""Validate a terminal vertical-motion derivative-conditioning ladder report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_vertical_motion_derivative_ladder as audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-derivative-ladder-001/report.json",
    )
    return parser.parse_args()


def validate_report(report: dict) -> None:
    if report.get("experiment") != audit.EXPERIMENT:
        raise SystemExit("vertical-motion derivative-ladder experiment mismatch")
    if report.get("protocol_commit") != audit.PROTOCOL_COMMIT:
        raise SystemExit("vertical-motion derivative-ladder protocol commit mismatch")
    if report.get("protocol") != audit.protocol_manifest():
        raise SystemExit("vertical-motion derivative-ladder protocol changed")
    if not (
        report.get("passed")
        and report.get("classification") == "vertical_motion_derivative_ladder_passed"
        and report.get("identity", {}).get("pass")
        and report.get("v1_reproduction", {}).get("pass")
        and report.get("qualification", {}).get("pass")
        and report.get("one_update", {}).get("pass")
        and report.get("source_and_local_identity_restored")
        and report.get("cuda_peak_reserved_memory_pass")
        and report.get("training_execution_authorized")
        and report.get("candidate_retained") is False
        and report.get("hover_or_gate_flight_authorized") is False
        and report.get("promotion_authorized") is False
        and report.get("development_specs_used") is False
        and report.get("acceptance_specs_used") is False
        and report.get("development_or_acceptance_pixels_rendered") is False
        and report.get("exception") is None
    ):
        raise SystemExit("vertical-motion derivative ladder did not pass every gate")
    for name, probe in report.get("finite_difference_ladder", {}).items():
        if len(probe.get("rows", [])) != len(audit.STEPS):
            raise SystemExit(f"derivative ladder is incomplete for {name}")
        if not probe.get("qualification", {}).get("pass"):
            raise SystemExit(f"derivative ladder did not qualify {name}")


def main() -> int:
    args = parse_args()
    with args.report.open() as stream:
        report = json.load(stream)
    validate_report(report)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "classification": report["classification"],
                "passed": report["passed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
