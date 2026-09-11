#!/usr/bin/env python3
"""Validate a terminal RK4 premotor time-constant preflight report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_rk4_premotor_tau as audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-premotor-tau-preflight-002/report.json",
    )
    return parser.parse_args()


def validate_report(report: dict) -> None:
    if report.get("experiment") != audit.EXPERIMENT:
        raise SystemExit("premotor-tau report experiment mismatch")
    if report.get("protocol_commit") != audit.PROTOCOL_COMMIT:
        raise SystemExit("premotor-tau report protocol commit mismatch")
    if report.get("protocol") != audit.protocol_manifest():
        raise SystemExit("premotor-tau report protocol manifest mismatch")
    if not report.get("source_state_restored"):
        raise SystemExit("premotor-tau report did not restore the source")
    if report.get("candidate_retained") is not False:
        raise SystemExit("premotor-tau report retained a candidate")
    if report.get("hover_or_gate_flight_authorized") is not False:
        raise SystemExit("premotor-tau report improperly authorizes flight")
    if report.get("promotion_authorized") is not False:
        raise SystemExit("premotor-tau report improperly authorizes promotion")
    manifests = report.get("training_cache_manifests", [])
    if [item.get("seed") for item in manifests] != list(audit.capacity.TRAINING_SEEDS):
        raise SystemExit("premotor-tau training seeds mismatch")
    archive = report.get("direction_archive")
    if archive is not None:
        path = REPO_ROOT / archive.get("path", "missing")
        if not path.is_file() or audit.file_sha256(path) != archive.get("file_sha256"):
            raise SystemExit("premotor-tau direction archive hash mismatch")
    if report.get("finite_difference") is not None and report.get(
        "finite_difference", {}
    ).get("scale") != audit.FINITE_DIFFERENCE_SCALE:
        raise SystemExit("premotor-tau finite-difference scale mismatch")
    trials = report.get("training_trials", [])
    scales = [item.get("scale") for item in trials]
    if scales != list(audit.TAU_SCALES[: len(scales)]):
        raise SystemExit("premotor-tau candidates are not a descending registered prefix")
    selected = report.get("selected_scale")
    passing = [item for item in trials if item.get("decision", {}).get("pass")]
    if selected is None:
        if passing:
            raise SystemExit("premotor-tau report omitted a passing selection")
    elif len(passing) != 1 or passing[0].get("scale") != selected:
        raise SystemExit("premotor-tau selected scale does not identify the first pass")
    if report.get("passed"):
        required = (
            report.get("finite_difference", {}).get("pass"),
            report.get("solver_requalification", {}).get("pass"),
            report.get("development", {}).get("decision", {}).get("pass"),
            report.get("bounded_premotor_tau_fit_preregistration_authorized"),
            selected is not None,
            archive is not None,
        )
        if not all(required):
            raise SystemExit("passing premotor-tau report lacks a passing gate")
    elif report.get("bounded_premotor_tau_fit_preregistration_authorized"):
        raise SystemExit("failed premotor-tau report authorizes a bounded fit")


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
                "selected_scale": report["selected_scale"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
