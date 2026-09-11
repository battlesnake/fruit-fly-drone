#!/usr/bin/env python3
"""Validate a terminal proposal-25 vertical-motion gradient-attribution report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_vertical_motion_gradient_attribution as audit  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=(
            REPO_ROOT / "runs/optic-motion/vertical-motion-gradient-attribution-001/report.json"
        ),
    )
    return parser.parse_args()


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool | int | str):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return False


def validate_report(report: dict) -> None:
    result = report.get("result")
    if not isinstance(result, dict):
        raise SystemExit("gradient-attribution result is absent")
    trials = result.get("trials", [])
    baseline_loss = result.get("proposal25_loss", {}).get("loss", math.nan)
    baseline_strata = result.get("proposal25_strata", {})
    if len(trials) != len(audit.MULTIPLIERS):
        raise SystemExit("gradient-attribution trial ladder is incomplete")
    passing = []
    for expected_multiplier, trial in zip(audit.MULTIPLIERS, trials, strict=True):
        if trial.get("multiplier") != expected_multiplier:
            raise SystemExit("gradient-attribution multiplier order changed")
        recomputed = audit.trial_decision(
            baseline_loss,
            baseline_strata,
            trial.get("loss", {}).get("loss", math.nan),
            trial.get("strata", {}),
            finite=bool(trial.get("decision", {}).get("finite")),
        )
        if trial.get("decision") != recomputed:
            raise SystemExit("gradient-attribution trial decision changed")
        if (
            set(trial.get("adam_steps_before", {}).values()) != {audit.ARCHIVED_PROPOSALS}
            or set(trial.get("adam_steps_after", {}).values())
            != {audit.ARCHIVED_PROPOSALS + 1}
            or not trial.get("adam_counter_transition_pass")
        ):
            raise SystemExit("gradient-attribution Adam transition is invalid")
        if recomputed["pass"]:
            passing.append(expected_multiplier)
    expected_first = passing[0] if passing else None
    scientific_pass = expected_first is not None
    expected_classification = (
        "vertical_motion_gradient_attribution_supports_larger_effective_batch"
        if scientific_pass
        else "vertical_motion_gradient_attribution_requires_objective_review"
    )
    gradient = result.get("gradient_attribution", {})
    geometry = gradient.get("common_descent", {})
    geometry_valid = bool(
        geometry.get("local_diagnostic_only")
        and geometry.get("active_bounds_respected")
        == geometry.get("active_bound_tangent_constraints_pass")
        and (
            not geometry.get("strict_common_descent")
            or geometry.get("certified_feasible_solution")
        )
    )
    finite_result = dict(result)
    finite_result.pop("first_passing_multiplier", None)
    if not (
        report.get("experiment") == audit.EXPERIMENT
        and report.get("protocol_commit") == audit.PROTOCOL_COMMIT
        and report.get("protocol") == audit.protocol_manifest()
        and report.get("implementation_file_sha256")
        == registration.file_sha256(Path(audit.__file__))
        and report.get("audit_completed") is True
        and report.get("classification") == expected_classification
        and report.get("exception") is None
        and result.get("reproduction_control", {}).get("pass")
        and result.get("gradient_reproduction_control", {}).get("pass")
        and result.get("all_four_multipliers_evaluated")
        and result.get("first_passing_multiplier") == expected_first
        and result.get("whole_bank_trial_pass") is scientific_pass
        and report.get("whole_bank_trial_pass") is scientific_pass
        and report.get("larger_effective_batch_preregistration_authorized") is scientific_pass
        and result.get("archived_state_restored")
        and result.get("source_restored")
        and result.get("archived_state_fingerprint_before")
        == result.get("archived_state_fingerprint_after")
        and result.get("source_state_sha256_before") == result.get("source_state_sha256_after")
        and gradient.get("parameter_count") == 24
        and gradient.get("parameter_order") == list(audit.PARAMETER_NAMES)
        and gradient.get("stratum_geometry", {}).get("order") == list(audit.STRATUM_NAMES)
        and geometry_valid
        and report.get("closed_training_run_may_resume") is False
        and report.get("development_or_acceptance_opened") is False
        and report.get("candidate_or_optimizer_retained") is False
        and report.get("motion_routing_hover_gate_or_promotion_authorized") is False
        and _finite_tree(finite_result)
    ):
        raise SystemExit("gradient-attribution audit failed terminal validation")


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
                "whole_bank_trial_pass": report["whole_bank_trial_pass"],
                "first_passing_multiplier": report["result"]["first_passing_multiplier"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
