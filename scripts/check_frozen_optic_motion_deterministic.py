#!/usr/bin/env python3
"""Validate a terminal deterministic frozen T4/T5 optic-motion report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import frozen_optic_motion_deterministic as deterministic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=REPO_ROOT
        / "runs/optic-motion/frozen-t4t5-audit-002/report.json",
    )
    return parser.parse_args()


def validate_report(report: dict) -> None:
    if report.get("experiment") != deterministic.EXPERIMENT:
        raise SystemExit("deterministic optic-motion experiment mismatch")
    if report.get("protocol_commit") != deterministic.PROTOCOL_COMMIT:
        raise SystemExit("deterministic optic-motion protocol commit mismatch")
    if report.get("protocol") != deterministic.protocol_manifest():
        raise SystemExit("deterministic optic-motion protocol manifest mismatch")
    if report.get("fresh_blind_validation") is not False:
        raise SystemExit("deterministic confirmation incorrectly claims fresh validation")
    if not report.get("source_restored") or not report.get("execution_controls_pass"):
        raise SystemExit("deterministic optic-motion execution controls failed")
    replay = report.get("deterministic_replay_gate", {})
    if (
        not replay.get("passed")
        or replay.get("processes") != deterministic.REPLAY_PROCESSES
        or not report.get("duplicate_control", {}).get("pass")
        or report["duplicate_control"].get("fresh_processes")
        != deterministic.REPLAY_PROCESSES + 1
        or report["duplicate_control"].get("response_tensor_sha256")
        != replay.get("response_tensor_sha256")
        or not report["duplicate_control"].get("main_process_identity_distinct")
        or report["duplicate_control"].get("main_process_identity")
        != report.get("process_identity")
    ):
        raise SystemExit("deterministic four-process replay gate failed")
    scientific_gate = bool(
        report.get("scientific_gate_valid")
        and report.get("numerical", {}).get("K32_vs_K64", {}).get("pass")
        and report.get("tuning") is not None
    )
    if not scientific_gate:
        raise SystemExit("terminal deterministic report lacks a valid scientific gate")
    if report.get("passed"):
        if not (
            report["tuning"]["pass"]
            and report.get("motion_output_routing_preregistration_authorized")
            and not report.get("local_motion_commissioning_preregistration_authorized")
        ):
            raise SystemExit("passing deterministic report has inconsistent authorization")
    elif not (
        not report["tuning"]["pass"]
        and not report.get("motion_output_routing_preregistration_authorized")
        and report.get("local_motion_commissioning_preregistration_authorized")
    ):
        raise SystemExit("failed deterministic report has inconsistent authorization")
    for name in (
        "training_execution_authorized",
        "candidate_retained",
        "hover_or_gate_flight_authorized",
        "promotion_authorized",
    ):
        if report.get(name) is not False:
            raise SystemExit(f"deterministic report improperly sets {name}")


def main() -> int:
    args = parse_args()
    terminal = deterministic.validate_completed_phase(
        args.report, phase="main-confirmation"
    )
    with args.report.open() as stream:
        report = json.load(stream)
    if terminal["process_identity"] != report.get("process_identity"):
        raise SystemExit("main terminal identity does not match its report")
    validate_report(report)
    print(
        json.dumps(
            {
                "report": deterministic.base.assisted.responsibility.stable_path(
                    args.report
                ),
                "classification": report["classification"],
                "passed": report["passed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
