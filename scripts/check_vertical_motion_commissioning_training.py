#!/usr/bin/env python3
"""Validate a passing terminal vertical-motion commissioning training report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preregister_vertical_motion_commissioning as registration  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402


def optimizer_history_valid(history: list[dict]) -> bool:
    accepted = 0
    names = {"gain", "bias_offset", "tau_ratio"}
    for proposal, item in enumerate(history, start=1):
        before = item.get("optimizer_steps_before", {})
        after = item.get("optimizer_steps_after", {})
        if (
            item.get("proposal") != proposal
            or set(before) != names
            or set(after) != names
            or set(before.values()) != {accepted}
            or not item.get("finite_gradients")
            or not math.isfinite(item.get("baseline_loss", math.nan))
            or not math.isfinite(item.get("unclipped_gradient_norm", math.nan))
        ):
            return False
        accepted += int(bool(item.get("accepted")))
        if set(after.values()) != {accepted}:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=(
            REPO_ROOT / "runs/optic-motion/vertical-motion-commissioning-training-001/report.json"
        ),
    )
    return parser.parse_args()


def validate_report(report: dict) -> None:
    if report.get("experiment") != training.EXPERIMENT:
        raise SystemExit("vertical-motion training experiment mismatch")
    if report.get("protocol_commit") != training.PROTOCOL_COMMIT:
        raise SystemExit("vertical-motion training protocol commit mismatch")
    if report.get("protocol") != training.protocol_manifest():
        raise SystemExit("vertical-motion training protocol changed")
    history = report.get("proposal_history", [])
    development = report.get("development_history", [])
    selection = report.get("selection", {})
    acceptance = report.get("acceptance", {})
    retained = report.get("retained_checkpoint", {})
    scheduled = list(training.SCHEDULED_EVALUATIONS)
    training_evaluations = report.get("training_evaluations", [])
    if not (
        report.get("passed")
        and report.get("classification") == "vertical_motion_training_passed"
        and report.get("proposals_completed") == training.MAX_PROPOSALS
        and len(history) == training.MAX_PROPOSALS
        and report.get("accepted_proposals", 0) + report.get("rejected_proposals", 0)
        == training.MAX_PROPOSALS
        and optimizer_history_valid(history)
        and report.get("scheduled_evaluation_started") == scheduled
        and [item.get("proposal") for item in training_evaluations] == scheduled
        and [item.get("proposal") for item in development] == scheduled
        and report.get("development_started") == scheduled
        and all(item.get("numerical", {}).get("pass") for item in training_evaluations)
        and training_evaluations[0].get("mandatory_decision", {}).get("pass")
        and any(item.get("qualification", {}).get("pass") for item in development)
        and selection.get("acceptance_authorized")
        and acceptance.get("pass")
        and acceptance.get("all_parameters_optimizer_responses_and_metrics_finite")
        and acceptance.get("qualification", {}).get("pass")
        and acceptance.get("all_texture_directions", {}).get("pass")
        and acceptance.get("all_texture_directions", {}).get("cases") == 64
        and acceptance.get("novel_texture_speed", {}).get("pass")
        and acceptance.get("novel_texture_speed", {}).get("cases") == 16
        and report.get("source_restored")
        and report.get("cuda_peak_reserved_memory_pass")
        and report.get("vertical_motion_module_retained")
        and report.get("motion_to_DN_VNC_routing_preregistration_authorized")
        and report.get("hover_or_gate_flight_authorized") is False
        and report.get("promotion_authorized") is False
        and report.get("exception") is None
    ):
        raise SystemExit("vertical-motion training did not pass every gate")
    checkpoint = REPO_ROOT / retained["path"]
    if not checkpoint.is_file() or registration.file_sha256(checkpoint) != retained["file_sha256"]:
        raise SystemExit("retained vertical-motion module is missing or changed")


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
                "retained_checkpoint": report["retained_checkpoint"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
