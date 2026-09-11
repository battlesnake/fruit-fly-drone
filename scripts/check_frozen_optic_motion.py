#!/usr/bin/env python3
"""Validate a terminal frozen T4/T5 optic-motion report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_frozen_optic_motion as audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "runs/optic-motion/frozen-t4t5-audit-001/report.json",
    )
    return parser.parse_args()


def validate_report(report: dict) -> None:
    if report.get("experiment") != audit.EXPERIMENT:
        raise SystemExit("optic-motion report experiment mismatch")
    if report.get("protocol_commit") != audit.PROTOCOL_COMMIT:
        raise SystemExit("optic-motion protocol commit mismatch")
    if report.get("protocol") != audit.protocol_manifest():
        raise SystemExit("optic-motion protocol manifest mismatch")
    if report.get("anatomy", {}).get("selected_nodes") != sum(
        audit.EXPECTED_TYPE_COUNTS.values()
    ):
        raise SystemExit("optic-motion cell count mismatch")
    if report.get("anatomy", {}).get("indices_sha256") != audit.EXPECTED_INDICES_SHA256:
        raise SystemExit("optic-motion graph-index hash mismatch")
    stimuli = report.get("stimulus_manifest", {})
    if len(stimuli.get("specs", [])) != audit.CORE_CASES + audit.GENERALIZATION_CASES:
        raise SystemExit("optic-motion stimulus count mismatch")
    if stimuli.get("opposite_terminal_maximum_difference") != 0.0:
        raise SystemExit("optic-motion endpoints differed")
    if not report.get("source_restored"):
        raise SystemExit("optic-motion report did not verify source identity")
    if not report.get("execution_controls_pass"):
        raise SystemExit("optic-motion report did not pass execution controls")
    if report.get("candidate_retained") is not False:
        raise SystemExit("optic-motion report retained a candidate")
    if report.get("hover_or_gate_flight_authorized") is not False:
        raise SystemExit("optic-motion report improperly authorizes flight")
    if report.get("promotion_authorized") is not False:
        raise SystemExit("optic-motion report improperly authorizes promotion")
    numerical_pass = bool(
        report.get("numerical", {}).get("K32_vs_K64", {}).get("pass")
    )
    if report.get("passed"):
        if not (
            numerical_pass
            and report.get("duplicate_control", {}).get("pass")
            and report.get("tuning", {}).get("pass")
            and report.get("motion_output_routing_preregistration_authorized")
            and not report.get("local_motion_commissioning_preregistration_authorized")
        ):
            raise SystemExit("passing optic-motion report lacks a passing gate")
    else:
        if report.get("motion_output_routing_preregistration_authorized"):
            raise SystemExit("failed optic-motion report authorizes output routing")
        commissioning = report.get("local_motion_commissioning_preregistration_authorized")
        if commissioning and not (
            numerical_pass
            and report.get("tuning") is not None
            and not report["tuning"]["pass"]
            and report.get("duplicate_control", {}).get("pass")
            and report.get("exception") is None
        ):
            raise SystemExit("optic-motion commissioning authorization is inconsistent")


def main() -> int:
    args = parse_args()
    with args.report.open() as stream:
        report = json.load(stream)
    validate_report(report)
    print(
        json.dumps(
            {
                "report": audit.assisted.responsibility.stable_path(args.report),
                "classification": report["classification"],
                "pass": report["passed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
