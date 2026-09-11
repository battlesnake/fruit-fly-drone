#!/usr/bin/env python3
"""Validate a terminal bounded RK4 readout-capacity report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_rk4_readout_capacity as fit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-readout-capacity-001/report.json",
    )
    return parser.parse_args()


def validate_report(report: dict) -> None:
    if report.get("experiment") != fit.EXPERIMENT:
        raise SystemExit("capacity report experiment mismatch")
    if report.get("protocol_commit") != fit.PROTOCOL_COMMIT:
        raise SystemExit("capacity report protocol commit mismatch")
    protocol = report.get("protocol")
    if protocol != fit.protocol_manifest():
        raise SystemExit("capacity report protocol manifest mismatch")
    if report.get("hover_or_gate_flight_authorized") is not False:
        raise SystemExit("capacity report improperly authorizes flight")
    if report.get("promotion_authorized") is not False:
        raise SystemExit("capacity report improperly authorizes promotion")
    if report.get("accepted_updates", -1) > fit.MAXIMUM_ACCEPTED_UPDATES:
        raise SystemExit("capacity report exceeded its update budget")
    manifests = report.get("training_cache_manifests", [])
    if [item.get("seed") for item in manifests] != list(fit.TRAINING_SEEDS):
        raise SystemExit("capacity report training seeds mismatch")
    if sum(int(item.get("cases", 0)) for item in manifests) != fit.TRAINING_CASES:
        raise SystemExit("capacity report training case count mismatch")
    history = report.get("history", [])
    if len(history) != report.get("accepted_updates"):
        raise SystemExit("capacity report accepted history length mismatch")
    for update, record in enumerate(history, start=1):
        if record.get("attempt") != update or not record.get("accepted"):
            raise SystemExit("capacity report accepted history is not sequential")
        if record.get("accepted_scale") not in fit.ORDINARY_SCALES:
            raise SystemExit("capacity report used an unregistered scale")
        if not record.get("selected", {}).get("decision", {}).get("pass"):
            raise SystemExit("capacity report retained a failing update")
        if not record.get("gradient_controls", {}).get(
            "outside_mask_gradients_exactly_zero"
        ):
            raise SystemExit("capacity report accepted an unmasked gradient")
        if not record.get("derivative_probe", {}).get("pass"):
            raise SystemExit("capacity report accepted without derivative qualification")
    if report.get("passed"):
        required = (
            report.get("terminal_training_decision", {}).get("pass"),
            report.get("development_decision", {}).get("pass"),
            report.get("qualification_decision", {}).get("pass"),
            report.get("solver_qualification", {}).get("pass"),
            report.get("bounded_collective_restoration_preregistration_authorized"),
        )
        if not all(required):
            raise SystemExit("passing capacity report lacks a passing terminal gate")
        checkpoint = report.get("qualified_controller") or {}
        path = REPO_ROOT / checkpoint.get("path", "missing")
        if not path.is_file():
            raise SystemExit("passing capacity report lacks its qualified checkpoint")
        if fit.assisted.responsibility.file_sha256(path) != checkpoint.get("file_sha256"):
            raise SystemExit("qualified capacity checkpoint hash mismatch")
    elif report.get("bounded_collective_restoration_preregistration_authorized"):
        raise SystemExit("failed capacity report authorizes collective restoration")


def main() -> int:
    args = parse_args()
    with args.report.open() as stream:
        report = json.load(stream)
    validate_report(report)
    print(
        json.dumps(
            {
                "report": fit.assisted.responsibility.stable_path(args.report),
                "classification": report["classification"],
                "pass": report["passed"],
                "accepted_updates": report["accepted_updates"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
