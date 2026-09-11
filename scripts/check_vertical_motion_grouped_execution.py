#!/usr/bin/env python3
"""Validate a terminal grouped CUDA execution-equivalence preflight report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_vertical_motion_gradient_attribution as attribution  # noqa: E402
import preflight_vertical_motion_grouped_execution as preflight  # noqa: E402

STATE_CONTROL_KEYS = {
    "archived_parameters_required",
    "archived_state_fingerprint",
    "optimizer_state_fingerprint",
    "source_state_fingerprint",
    "adam_steps",
    "pass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=(REPO_ROOT / "runs/optic-motion/vertical-motion-grouped-preflight-001/report.json"),
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


def _gradient_comparison_pass(value: dict) -> bool:
    return bool(
        set(value) == {"maximum_absolute_difference", "relative_l2_error"}
        and value["maximum_absolute_difference"] <= preflight.ABSOLUTE_TOLERANCE
        and value["relative_l2_error"] <= preflight.GRADIENT_RELATIVE_L2_TOLERANCE
    )


def _state_controls_pass(item: dict) -> bool:
    controls = [
        item.get("archive_evaluation_control", {}),
        item.get("witness_optimizer_source_control", {}),
        item.get("restoration_control", {}),
    ]
    if not all(
        set(control) == STATE_CONTROL_KEYS and control.get("pass") is True for control in controls
    ):
        return False
    archive, witness, restoration = controls
    return bool(
        archive["archived_parameters_required"] is True
        and witness["archived_parameters_required"] is True
        and restoration["archived_parameters_required"] is True
        and archive["archived_state_fingerprint"] == restoration["archived_state_fingerprint"]
        and len({control["optimizer_state_fingerprint"] for control in controls}) == 1
        and len({control["source_state_fingerprint"] for control in controls}) == 1
        and all(
            set(control["adam_steps"]) == set(attribution.PARAMETER_NAMES)
            and set(control["adam_steps"].values()) == {attribution.ARCHIVED_PROPOSALS}
            for control in controls
        )
    )


def _single_archive_state_control_pass(control: dict) -> bool:
    return bool(
        set(control) == STATE_CONTROL_KEYS
        and control.get("archived_parameters_required") is True
        and control.get("pass") is True
        and set(control.get("adam_steps", {})) == set(attribution.PARAMETER_NAMES)
        and set(control.get("adam_steps", {}).values()) == {attribution.ARCHIVED_PROPOSALS}
    )


def _witness_pass(witness: dict) -> bool:
    witness_comparison = witness.get("locked_trial_comparison", {})
    derivatives = witness.get("signed_directional_derivatives", {})
    derivative_values = [
        derivatives.get("full", math.nan),
        *derivatives.get("strata", {}).values(),
    ]
    recomputed = attribution.trial_decision(
        witness.get("baseline_loss", math.nan),
        witness.get("baseline_strata", {}),
        witness.get("loss", {}).get("loss", math.nan),
        witness.get("strata", {}),
        finite=bool(witness.get("decision", {}).get("finite")),
    )
    return bool(
        witness.get("decision") == recomputed
        and recomputed["pass"] is True
        and set(witness_comparison) == {"loss", "components", "strata"}
        and all(value <= preflight.ABSOLUTE_TOLERANCE for value in witness_comparison.values())
        and set(derivatives.get("strata", {})) == set(attribution.STRATUM_NAMES)
        and len(derivative_values) == 5
        and all(value < 0.0 for value in derivative_values)
        and witness.get("all_signed_directional_derivatives_negative") is True
        and witness.get("pass") is True
    )


def _size_pass(item: dict) -> bool:
    metric_comparison = item.get("grouped_metric_comparison", {})
    gradient_comparison = item.get("gradient_comparison", {})
    bank_comparison = item.get("proposal_bank_metric_comparison", {})
    sequential_bank_comparison = item.get("sequential_bank_metric_comparison", {})
    witness = item.get("witness", {})
    return bool(
        item.get("finite") is True
        and set(metric_comparison) == {"loss", "components", "strata"}
        and all(value <= preflight.ABSOLUTE_TOLERANCE for value in metric_comparison.values())
        and _gradient_comparison_pass(gradient_comparison.get("full", {}))
        and set(gradient_comparison.get("strata", {})) == set(attribution.STRATUM_NAMES)
        and all(
            _gradient_comparison_pass(value)
            for value in gradient_comparison.get("strata", {}).values()
        )
        and item.get("proposal_response_maximum_absolute_difference", math.inf)
        <= preflight.ABSOLUTE_TOLERANCE
        and set(bank_comparison) == {"loss", "components", "strata"}
        and all(value <= preflight.ABSOLUTE_TOLERANCE for value in bank_comparison.values())
        and set(sequential_bank_comparison) == {"loss", "components", "strata"}
        and all(
            value <= preflight.ABSOLUTE_TOLERANCE for value in sequential_bank_comparison.values()
        )
        and _witness_pass(witness)
        and _state_controls_pass(item)
        and item.get("cuda_peak_reserved_bytes", math.inf)
        <= preflight.CUDA_PEAK_RESERVED_LIMIT_BYTES
    )


def _repeat_pass(repeat: dict) -> bool:
    comparisons = repeat.get("comparisons", {})
    return bool(
        repeat.get("finite") is True
        and set(comparisons.get("metrics", {})) == {"loss", "components", "strata"}
        and all(
            value <= preflight.ABSOLUTE_TOLERANCE
            for value in comparisons.get("metrics", {}).values()
        )
        and _gradient_comparison_pass(comparisons.get("full_gradient", {}))
        and set(comparisons.get("stratum_gradients", {})) == set(attribution.STRATUM_NAMES)
        and all(
            _gradient_comparison_pass(value)
            for value in comparisons.get("stratum_gradients", {}).values()
        )
        and comparisons.get("proposal_response", math.inf) <= preflight.ABSOLUTE_TOLERANCE
        and comparisons.get("witness_response", math.inf) <= preflight.ABSOLUTE_TOLERANCE
        and comparisons.get("witness_loss", math.inf) <= preflight.ABSOLUTE_TOLERANCE
        and comparisons.get("witness_strata", math.inf) <= preflight.ABSOLUTE_TOLERANCE
        and _witness_pass(repeat.get("witness", {}))
        and _state_controls_pass(repeat)
        and repeat.get("cuda_peak_reserved_bytes", math.inf)
        <= preflight.CUDA_PEAK_RESERVED_LIMIT_BYTES
    )


def validate_report(report: dict) -> None:
    result = report.get("result")
    if not isinstance(result, dict):
        raise SystemExit("grouped-execution result is absent")
    size_results = result.get("size_results", [])
    if [item.get("group_size") for item in size_results] != list(preflight.GROUP_SIZES):
        raise SystemExit("grouped-execution size ladder changed")
    recomputed_size_passes = [_size_pass(item) for item in size_results]
    if any(
        item.get("pass") is not passed
        for item, passed in zip(size_results, recomputed_size_passes, strict=True)
    ):
        raise SystemExit("grouped-execution size decision changed")
    passing_sizes = [
        size
        for size, passed in zip(preflight.GROUP_SIZES, recomputed_size_passes, strict=True)
        if passed
    ]
    selected_before_repeat = max(passing_sizes) if passing_sizes else None
    repeat = result.get("selected_repeat")
    repeat_pass = bool(isinstance(repeat, dict) and _repeat_pass(repeat))
    selected = selected_before_repeat if repeat_pass else None
    scientific_pass = selected is not None
    expected_classification = (
        "grouped_cuda_execution_equivalent"
        if scientific_pass
        else "grouped_cuda_execution_not_qualified"
    )
    input_hashes = report.get("input_file_sha256", {})
    sequential_control = result.get("sequential_reference", {}).get("state_control", {})
    evaluation_controls = [sequential_control]
    for item in size_results:
        if item.get("failure_kind") == "cuda_out_of_memory":
            evaluation_controls.append(item.get("oom_recovery_control", {}))
        else:
            evaluation_controls.extend(
                (
                    item.get("archive_evaluation_control", {}),
                    item.get("witness_optimizer_source_control", {}),
                    item.get("restoration_control", {}),
                )
            )
    if isinstance(repeat, dict):
        if repeat.get("failure_kind") == "cuda_out_of_memory":
            evaluation_controls.append(repeat.get("oom_recovery_control", {}))
        else:
            evaluation_controls.extend(
                [
                    repeat.get("archive_evaluation_control", {}),
                    repeat.get("witness_optimizer_source_control", {}),
                    repeat.get("restoration_control", {}),
                ]
            )
    archive_controls = [
        control
        for control in evaluation_controls
        if control.get("archived_parameters_required") is True
    ]
    oom_controls = [
        item.get("oom_recovery_control", {})
        for item in [
            *size_results,
            *((repeat,) if isinstance(repeat, dict) else ()),
        ]
        if item.get("failure_kind") == "cuda_out_of_memory"
    ]
    state_controls_valid = bool(
        _single_archive_state_control_pass(sequential_control)
        and all(_single_archive_state_control_pass(control) for control in oom_controls)
        and all(control.get("pass") is True for control in evaluation_controls)
        and all(
            control.get("source_state_fingerprint") == result.get("source_state_sha256_before")
            for control in evaluation_controls
        )
        and all(
            control.get("optimizer_state_fingerprint")
            == result.get("optimizer_state_fingerprint_before")
            for control in evaluation_controls
        )
        and all(
            control.get("archived_state_fingerprint")
            == result.get("archived_state_fingerprint_before")
            for control in archive_controls
        )
    )
    finite_result = dict(result)
    if finite_result.get("selected_size_before_repeat") is None:
        finite_result.pop("selected_size_before_repeat", None)
    if finite_result.get("selected_group_size") is None:
        finite_result.pop("selected_group_size", None)
    if finite_result.get("selected_repeat") is None:
        finite_result.pop("selected_repeat", None)
    if not (
        report.get("experiment") == preflight.EXPERIMENT
        and report.get("protocol_commit") == preflight.PROTOCOL_COMMIT
        and report.get("protocol") == preflight.protocol_manifest()
        and report.get("implementation_file_sha256") == preflight.implementation_file_hashes()
        and input_hashes.get("runs/optic-motion/vertical-motion-frozen-witness-001/report.json")
        == preflight.WITNESS_REPORT_SHA256
        and input_hashes.get("runs/optic-motion/vertical-motion-frozen-witness-001/started.json")
        == preflight.WITNESS_STARTED_SHA256
        and input_hashes.get("artifacts/vertical-motion-frozen-witness-v1/report.json")
        == preflight.WITNESS_COMPACT_REPORT_SHA256
        and report.get("preflight_completed") is True
        and report.get("classification") == expected_classification
        and report.get("exception") is None
        and result.get("all_sizes_evaluated") is True
        and state_controls_valid
        and result.get("passing_sizes_before_repeat") == passing_sizes
        and result.get("selected_size_before_repeat") == selected_before_repeat
        and isinstance(repeat, dict) == (selected_before_repeat is not None)
        and (repeat is None or repeat.get("group_size") == selected_before_repeat)
        and (repeat is None or repeat.get("pass") is repeat_pass)
        and result.get("selected_group_size") == selected
        and result.get("grouped_execution_qualified") is scientific_pass
        and report.get("grouped_execution_qualified") is scientific_pass
        and report.get("common_descent_trainer_preregistration_authorized") is scientific_pass
        and result.get("archived_state_restored") is True
        and result.get("source_restored") is True
        and result.get("archived_state_fingerprint_before")
        == result.get("archived_state_fingerprint_after")
        and result.get("optimizer_state_fingerprint_before")
        == result.get("optimizer_state_fingerprint_after")
        and result.get("optimizer_state_restored") is True
        and result.get("source_state_sha256_before") == result.get("source_state_sha256_after")
        and report.get("development_or_acceptance_opened") is False
        and report.get("candidate_or_optimizer_retained") is False
        and report.get("motion_routing_hover_gate_or_promotion_authorized") is False
        and _finite_tree(finite_result)
    ):
        raise SystemExit("grouped CUDA preflight failed terminal validation")


def main() -> int:
    args = parse_args()
    with args.report.open() as stream:
        report = json.load(stream)
    started_path = args.report.with_name("started.json")
    with started_path.open() as stream:
        started = json.load(stream)
    start_keys = {
        "experiment",
        "protocol_commit",
        "protocol",
        "input_file_sha256",
        "implementation_commit",
        "implementation_file_sha256",
        "runtime",
    }
    if set(started) != start_keys or any(report.get(key) != started[key] for key in start_keys):
        raise SystemExit("grouped CUDA preflight started marker does not match report")
    validate_report(report)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "classification": report["classification"],
                "grouped_execution_qualified": report["grouped_execution_qualified"],
                "selected_group_size": report["result"]["selected_group_size"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
