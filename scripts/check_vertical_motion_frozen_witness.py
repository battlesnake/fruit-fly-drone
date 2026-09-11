#!/usr/bin/env python3
"""Validate a terminal vertical-motion frozen-witness finite-step report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_vertical_motion_frozen_witness as audit  # noqa: E402
import audit_vertical_motion_gradient_attribution as attribution  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=(REPO_ROOT / "runs/optic-motion/vertical-motion-frozen-witness-001/report.json"),
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


def _maximum_difference(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        return math.inf
    return max((abs(left - right) for left, right in zip(first, second, strict=True)), default=0.0)


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def validate_report(report: dict) -> None:
    result = report.get("result")
    if not isinstance(result, dict):
        raise SystemExit("frozen-witness result is absent")
    witness = result.get("witness_control", {})
    direction = witness.get("direction", [])
    if len(direction) != 24:
        raise SystemExit("frozen-witness direction length changed")
    with audit.COMPACT_REPORT.open() as stream:
        compact_report = json.load(stream)
    locked_direction = compact_report["gradient_attribution"]["common_descent"]["unit_direction"]
    derivative_names = {"full", *attribution.STRATUM_NAMES}
    normalized_derivatives = witness.get("normalized_directional_derivatives", {})
    raw_derivatives = witness.get("raw_directional_derivatives", {})
    gradient_norms = witness.get("gradient_norms", {})
    gradient_differences = witness.get("gradient_reproduction_maximum_absolute_differences", {})
    if not (
        registration.file_sha256(audit.COMPACT_REPORT) == audit.COMPACT_REPORT_SHA256
        and direction == locked_direction
        and witness.get("pass") is True
        and witness.get("finite") is True
        and witness.get("active_bound_tangent_pass") is True
        and witness.get("direction_norm_error", math.inf) <= audit.UNIT_NORM_TOLERANCE
        and set(normalized_derivatives) == derivative_names
        and set(raw_derivatives) == derivative_names
        and set(gradient_norms) == derivative_names
        and all(value > 0.0 for value in gradient_norms.values())
        and all(
            value <= audit.NORMALIZED_DERIVATIVE_MAXIMUM
            for value in normalized_derivatives.values()
        )
        and gradient_differences.get("full", math.inf) <= audit.GRADIENT_REPRODUCTION_TOLERANCE
        and set(gradient_differences.get("strata", {})) == set(attribution.STRATUM_NAMES)
        and all(
            value <= audit.GRADIENT_REPRODUCTION_TOLERANCE
            for value in gradient_differences.get("strata", {}).values()
        )
    ):
        raise SystemExit("frozen-witness qualification is invalid")

    baseline_loss = result.get("proposal25_loss", {}).get("loss", math.nan)
    baseline_strata = result.get("proposal25_strata", {})
    trials = result.get("trials", [])
    if len(trials) != len(audit.MULTIPLIERS):
        raise SystemExit("frozen-witness trial ladder is incomplete")
    passing: list[float] = []
    optimizer_hashes: set[str] = set()
    for expected_multiplier, trial in zip(audit.MULTIPLIERS, trials, strict=True):
        if trial.get("multiplier") != expected_multiplier:
            raise SystemExit("frozen-witness multiplier order changed")
        intended = trial.get("intended_displacement", [])
        actual = trial.get("actual_displacement", [])
        projection = trial.get("projection_difference", [])
        if not len(intended) == len(actual) == len(projection) == 24:
            raise SystemExit(f"frozen-witness trial {expected_multiplier} vector length changed")
        expected_intended = [
            value * audit.REFERENCE_DISPLACEMENT_NORM * expected_multiplier for value in direction
        ]
        expected_projection = [
            materialized - requested
            for materialized, requested in zip(actual, intended, strict=True)
        ]
        derivatives = trial.get("signed_directional_derivatives", {})
        if set(derivatives.get("strata", {})) != set(attribution.STRATUM_NAMES):
            raise SystemExit(f"frozen-witness trial {expected_multiplier} derivatives changed")
        derivative_values = [
            derivatives.get("full", math.nan),
            *derivatives.get("strata", {}).values(),
        ]
        optimizer_unchanged = bool(
            isinstance(trial.get("optimizer_semantic_sha256_before"), str)
            and len(trial["optimizer_semantic_sha256_before"]) == 64
            and trial.get("optimizer_unchanged") is True
            and trial.get("optimizer_semantic_sha256_before")
            == trial.get("optimizer_semantic_sha256_after")
            and set(trial.get("adam_steps_before", {}).values()) == {attribution.ARCHIVED_PROPOSALS}
            and trial.get("adam_steps_after") == trial.get("adam_steps_before")
        )
        recomputed = attribution.trial_decision(
            baseline_loss,
            baseline_strata,
            trial.get("loss", {}).get("loss", math.nan),
            trial.get("strata", {}),
            finite=bool(trial.get("decision", {}).get("finite")),
        )
        if not (
            _maximum_difference(intended, expected_intended) <= 1.0e-15
            and _maximum_difference(projection, expected_projection) <= 1.0e-15
            and trial.get("reference_displacement_norm") == audit.REFERENCE_DISPLACEMENT_NORM
            and abs(trial.get("intended_displacement_norm", math.inf) - _norm(intended)) <= 1.0e-15
            and abs(trial.get("actual_displacement_norm", math.inf) - _norm(actual)) <= 1.0e-15
            and abs(trial.get("projection_difference_norm", math.inf) - _norm(projection))
            <= 1.0e-15
            and all(value < 0.0 for value in derivative_values)
            and trial.get("all_signed_directional_derivatives_negative") is True
            and optimizer_unchanged
            and trial.get("decision") == recomputed
        ):
            raise SystemExit(f"frozen-witness trial {expected_multiplier} is invalid")
        if recomputed["pass"]:
            passing.append(expected_multiplier)
        optimizer_hashes.add(trial["optimizer_semantic_sha256_before"])

    if len(optimizer_hashes) != 1:
        raise SystemExit("frozen-witness trials did not share one archived optimizer")

    expected_first = passing[0] if passing else None
    scientific_pass = expected_first is not None
    expected_classification = (
        "frozen_witness_finite_step_supported"
        if scientific_pass
        else "frozen_witness_finite_step_no_useful_leverage"
    )
    finite_result = dict(result)
    finite_result.pop("first_passing_multiplier", None)
    input_hashes = report.get("input_file_sha256", {})
    if not (
        report.get("experiment") == audit.EXPERIMENT
        and report.get("protocol_commit") == audit.PROTOCOL_COMMIT
        and report.get("protocol") == audit.protocol_manifest()
        and report.get("implementation_file_sha256")
        == registration.file_sha256(Path(audit.__file__))
        and input_hashes.get(
            "runs/optic-motion/vertical-motion-gradient-attribution-001/report.json"
        )
        == audit.ATTRIBUTION_REPORT_SHA256
        and input_hashes.get(
            "runs/optic-motion/vertical-motion-gradient-attribution-001/started.json"
        )
        == audit.ATTRIBUTION_STARTED_SHA256
        and input_hashes.get("artifacts/vertical-motion-gradient-attribution-v1/report.json")
        == audit.COMPACT_REPORT_SHA256
        and report.get("audit_completed") is True
        and report.get("classification") == expected_classification
        and report.get("exception") is None
        and result.get("reproduction_control", {}).get("pass") is True
        and result.get("all_four_multipliers_evaluated") is True
        and result.get("first_passing_multiplier") == expected_first
        and result.get("finite_step_supported") is scientific_pass
        and report.get("finite_step_supported") is scientific_pass
        and report.get("constrained_training_preregistration_authorized") is scientific_pass
        and result.get("archived_state_restored") is True
        and result.get("source_restored") is True
        and result.get("archived_state_fingerprint_before")
        == result.get("archived_state_fingerprint_after")
        and result.get("source_state_sha256_before") == result.get("source_state_sha256_after")
        and report.get("closed_training_run_may_resume") is False
        and report.get("development_or_acceptance_opened") is False
        and report.get("candidate_or_optimizer_retained") is False
        and report.get("motion_routing_hover_gate_or_promotion_authorized") is False
        and _finite_tree(finite_result)
    ):
        raise SystemExit("frozen-witness audit failed terminal validation")


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
                "finite_step_supported": report["finite_step_supported"],
                "first_passing_multiplier": report["result"]["first_passing_multiplier"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
