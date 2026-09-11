#!/usr/bin/env python3
"""Audit local Taylor convergence of the stopped beta1=0 throttle proposal."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_native_throttle_assisted as train  # noqa: E402
import train_variable_height_native_throttle_assisted_beta1_zero as beta1_zero  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-native-throttle-beta1-zero-taylor-audit-v1"
PROTOCOL_COMMIT = "33ce76d"
EXPECTED_REPORT_SHA256 = "1b951c0613ebe904cdb39186a053fcba72d5da7408170b73fac85693b915cc36"
EXPECTED_RESUME_SHA256 = "330fa83864a711a7630df8109f1a0550360b111f950e6572b25a4919b978acb3"
EXPECTED_CONTROLLER_SHA256 = "1c28c49770687e4db164e16b02eb65095ef32661dde24a7595626ac14e934e7b"
EXPECTED_OPTIMIZER_SHA256 = "9ecbe922bcfc0bf5a6a14edc49836ac803af600363211bd75903aea7723d441f"
EXPECTED_TEACHER_BANK_SHA256 = "69993c6de95bfc150b4f2277a53c93da7aa2b6a94c9932d9f76c2893af3d1f62"
EXPECTED_MOTION_BANK_SHA256 = "089bbb8cc72980ca6321f9919227cad5bda1a746041e64426a29455090be6fbd"
EXPECTED_OBJECTIVE_SCALES_SHA256 = (
    "53f780b8d08feeab98cf147f3ce3cbb31315eadfe8c7bc3a487fd5dd3526c810"
)
EXPECTED_SAMPLE_SPEC_SHA256 = "fd0ea38428fdff8be75421451537173ac8ba5066ac0ac9891c86d993a1c951e7"
EXPECTED_FIXED_OBJECTIVE = 1.0461612939834595
EXPECTED_FULL_PREFIX_OBJECTIVE = 1.0461606103926897
EXPECTED_MATERIALIZED_DIRECTION = -0.7856744796120311
EXPECTED_SCALE_ONE_SIXTEENTH_OBJECTIVE = 1.0200214385986328
EXPECTED_SCALE_ONE_SIXTEENTH_DIRECTION = -0.41823768615722656
REPRODUCTION_ABSOLUTE_TOLERANCE = 2.0e-5
BASELINE_REPEATS = 3
TAYLOR_SCALES = (1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512)
LOCAL_RELATIVE_ERROR_LIMIT = 0.20
MINIMUM_DIRECTION_MAGNITUDE = 1.0e-8
MINIMUM_OBJECTIVE_CHANGE = 1.0e-8
REPLAY_NOISE_MULTIPLIER = 10.0
QUADRATIC_HALVING_RATIO_RANGE = (1 / 8, 1 / 2)


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
        "--producer-report",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/native-throttle-assisted-beta1-zero-001/report.json"
        ),
    )
    parser.add_argument(
        "--producer-resume",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/native-throttle-assisted-beta1-zero-001/resume.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/native-throttle-beta1-zero-taylor-audit-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_input_files(args: argparse.Namespace) -> None:
    expected = {
        args.graph: train.EXPECTED_GRAPH_SHA256,
        args.checkpoint: train.EXPECTED_CHECKPOINT_SHA256,
        args.producer_report: EXPECTED_REPORT_SHA256,
        args.producer_resume: EXPECTED_RESUME_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise SystemExit(f"missing Taylor-audit input: {path}")
        if train.responsibility.file_sha256(path) != digest:
            raise SystemExit(f"Taylor-audit input hash mismatch: {path}")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("Taylor audit already has a terminal report")
    if (args.output_dir / "audit-started.json").is_file():
        raise SystemExit("Taylor audit was already started and cannot be replayed")


def write_exclusive_start_marker(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit("Taylor audit was already started and cannot be replayed") from error
    return train.responsibility.file_sha256(path)


@contextmanager
def restoring_controller_and_optimizer(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    current: dict[str, Tensor],
    optimizer_state: dict[str, Any],
):
    try:
        yield
    finally:
        train._load_parameters(controller, current)
        optimizer.load_state_dict(optimizer_state)


def replay_noise(values: list[float]) -> float:
    if len(values) != BASELINE_REPEATS or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expected {BASELINE_REPEATS} finite baseline replays")
    return max(values) - min(values)


def local_derivative_agreement(record: dict[str, Any], *, noise_threshold: float) -> bool:
    return bool(
        record["controls"]["pass"]
        and record["all_metrics_finite"]
        and record["predicted_directional_derivative"] < -MINIMUM_DIRECTION_MAGNITUDE
        and record["measured_directional_derivative"] < -MINIMUM_DIRECTION_MAGNITUDE
        and record["relative_error"] <= LOCAL_RELATIVE_ERROR_LIMIT
        and abs(record["objective_change"]) > noise_threshold
    )


def adjacent_passing_pairs(records: list[dict[str, Any]]) -> list[list[float]]:
    smaller = records[1:]
    return [
        [float(left["scale"]), float(right["scale"])]
        for left, right in zip(smaller, smaller[1:], strict=False)
        if left["local_derivative_agreement"] and right["local_derivative_agreement"]
    ]


def audit_classification(
    *,
    controls_pass: bool,
    registered_failure_reproduced: bool,
    adjacent_pairs: list[list[float]],
    adjacent_above_noise_pairs: list[list[float]],
) -> tuple[bool, str]:
    if not controls_pass or not registered_failure_reproduced:
        return False, "taylor_audit_control_failure"
    if adjacent_pairs:
        return True, "local_derivative_convergence_consistent_with_finite_step_curvature"
    if not adjacent_above_noise_pairs:
        return False, "noise_limited_inconclusive"
    return False, "above_noise_local_derivative_nonconvergence"


def _directional_change(gradients: dict[str, Tensor], displacement: dict[str, Tensor]) -> float:
    return float(
        sum(
            (gradients[name].double() * displacement[name].double()).sum()
            for name in train.PARAMETER_FAMILIES
        )
    )


def _displacement_report(
    effective: dict[str, Tensor],
    proposal: dict[str, Tensor],
    scale: float,
) -> dict[str, Any]:
    return {
        name: {
            "actual_rms": float(effective[name].double().square().mean().sqrt()),
            "projection_adjustment_max_abs": float(
                (effective[name].double() - scale * proposal[name].double()).abs().max()
            ),
            "projection_adjustment_rms": float(
                (effective[name].double() - scale * proposal[name].double())
                .square()
                .mean()
                .sqrt()
            ),
        }
        for name in train.PARAMETER_FAMILIES
    }


def _evaluate_scale(
    controller: ConnectomeController,
    current: dict[str, Tensor],
    proposal: dict[str, Tensor],
    raw_gradients: dict[str, Tensor],
    scale: float,
    j0: float,
    dense_sample: dict[str, Any],
    dense_burn: Tensor,
    motion_bank: dict[str, Any],
    motion_indices: list[int],
    motion_burns: list[Tensor],
    objective_scales: dict[str, float],
    *,
    device: torch.device,
    noise_threshold: float,
) -> dict[str, Any]:
    try:
        controls, effective = train.materialize_scaled_proposal(
            controller, current, proposal, scale
        )
        candidate = train._sample_objective_report(
            controller,
            dense_sample,
            dense_burn,
            motion_bank,
            motion_indices,
            motion_burns,
            objective_scales,
            device=device,
        )
        objective_change = candidate["objective"] - j0
        predicted_change = _directional_change(raw_gradients, effective)
        measured_direction = objective_change / scale
        predicted_direction = predicted_change / scale
        signed_residual = objective_change - predicted_change
        finite = bool(
            candidate["all_recurrent_states_and_outputs_finite"]
            and train._all_finite_nested(candidate)
            and math.isfinite(objective_change)
            and math.isfinite(predicted_change)
            and math.isfinite(measured_direction)
            and math.isfinite(predicted_direction)
            and math.isfinite(signed_residual)
        )
        record = {
            "scale": scale,
            "objective": candidate["objective"],
            "dense_loss": candidate["dense_loss"],
            "motion_loss": candidate["motion_loss"],
            "objective_change": objective_change,
            "predicted_change": predicted_change,
            "measured_directional_derivative": measured_direction,
            "predicted_directional_derivative": predicted_direction,
            "relative_error": train._relative_error(measured_direction, predicted_direction),
            "signed_taylor_residual": signed_residual,
            "absolute_taylor_residual": abs(signed_residual),
            "objective_change_exceeds_noise_threshold": abs(objective_change) > noise_threshold,
            "noise_threshold": noise_threshold,
            "controls": controls,
            "displacement_by_parameter_family": _displacement_report(
                effective, proposal, scale
            ),
            "all_metrics_finite": finite,
            "all_recurrent_states_and_outputs_finite": candidate[
                "all_recurrent_states_and_outputs_finite"
            ],
        }
        record["local_derivative_agreement"] = local_derivative_agreement(
            record, noise_threshold=noise_threshold
        )
        return record
    finally:
        train._load_parameters(controller, current)


def _adjacent_residual_reports(
    records: list[dict[str, Any]], *, noise_threshold: float
) -> list[dict[str, Any]]:
    reports = []
    low, high = QUADRATIC_HALVING_RATIO_RANGE
    for larger, smaller in zip(records, records[1:], strict=False):
        denominator = larger["absolute_taylor_residual"]
        ratio = (
            smaller["absolute_taylor_residual"] / denominator if denominator > 0.0 else None
        )
        above_noise = bool(
            larger["absolute_taylor_residual"] > noise_threshold
            and smaller["absolute_taylor_residual"] > noise_threshold
        )
        reports.append(
            {
                "larger_scale": larger["scale"],
                "smaller_scale": smaller["scale"],
                "absolute_residual_ratio_after_halving": ratio,
                "both_residuals_above_noise_threshold": above_noise,
                "approximately_quadratic": bool(
                    above_noise and ratio is not None and low <= ratio <= high
                ),
                "descriptive_ratio_range": list(QUADRATIC_HALVING_RATIO_RANGE),
            }
        )
    return reports


def main() -> int:
    args = parse_args()
    validate_input_files(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "status": "started",
        "replay_permitted": False,
        "producer_report_sha256": EXPECTED_REPORT_SHA256,
        "producer_resume_sha256": EXPECTED_RESUME_SHA256,
        "sample_spec_sha256": EXPECTED_SAMPLE_SPEC_SHA256,
    }
    marker_sha256 = write_exclusive_start_marker(args.output_dir / "audit-started.json", marker)

    beta1_zero.configure_protocol()
    producer = json.loads(args.producer_report.read_text())
    raw_resume = torch.load(args.producer_resume, map_location="cpu", weights_only=True)
    fatal = producer["history"][-1]
    spec = fatal["sample_spec"]
    input_identity_pass = bool(
        producer["classification"] == "fatal_numerical_control_failure"
        and producer["accepted_updates"] == 5
        and producer["attempted_updates"] == 6
        and raw_resume["run_state"] == "stopped"
        and raw_resume["accepted_updates"] == 5
        and raw_resume["attempted_updates"] == 6
        and raw_resume["controller_parameter_sha256"] == EXPECTED_CONTROLLER_SHA256
        and raw_resume["optimizer_sha256"] == EXPECTED_OPTIMIZER_SHA256
        and raw_resume["block"]["teacher"]["sha256"] == EXPECTED_TEACHER_BANK_SHA256
        and raw_resume["block"]["motion"]["sha256"] == EXPECTED_MOTION_BANK_SHA256
        and raw_resume["objective_scales_sha256"] == EXPECTED_OBJECTIVE_SCALES_SHA256
        and train.audit.semantic_sha256(spec) == EXPECTED_SAMPLE_SPEC_SHA256
        and train.audit.semantic_sha256(raw_resume["history"][-1]["sample_spec"])
        == EXPECTED_SAMPLE_SPEC_SHA256
    )
    if not input_identity_pass:
        raise SystemExit("Taylor-audit scientific input identity mismatch")

    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / train.POLICY_HZ).to(device)
    controller.load_state_dict(loaded["controller"])
    controller.eval()
    optimizer = train._make_optimizer(controller)
    generator = torch.Generator(device="cpu").manual_seed(0)
    state = train._load_resume(
        args.producer_resume,
        controller,
        optimizer,
        generator,
    )
    current = train._copy_parameters(controller)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    controller_before = train.audit.semantic_sha256(current)
    optimizer_before = train.audit.semantic_sha256(optimizer_state)

    block = state["block"]
    dense_sample = train.materialize_dense_sample(block["teacher"], None, spec)
    motion_indices = [int(value) for value in spec["motion_indices"]]
    dense_burn, motion_burns = train._build_fixed_burns(
        controller,
        dense_sample,
        block["motion"],
        motion_indices,
        device=device,
    )
    objective_scales = state["objective_scales"]
    baseline_reports = [
        train._sample_objective_report(
            controller,
            dense_sample,
            dense_burn,
            block["motion"],
            motion_indices,
            motion_burns,
            objective_scales,
            device=device,
        )
        for _ in range(BASELINE_REPEATS)
    ]
    baseline_values = [float(report["objective"]) for report in baseline_reports]
    j0 = baseline_values[0]
    measured_replay_noise = replay_noise(baseline_values)
    noise_threshold = max(
        MINIMUM_OBJECTIVE_CHANGE, REPLAY_NOISE_MULTIPLIER * measured_replay_noise
    )
    current_full_value, current_full = train.combined_full_prefix_objective(
        controller,
        dense_sample,
        block["motion"],
        motion_indices,
        dense_scale=objective_scales["dense"],
        motion_scale=objective_scales["motion"],
        device=device,
    )
    gradient, raw_gradients = train.accumulate_sample_gradient(
        controller,
        dense_sample,
        dense_burn,
        block["motion"],
        motion_indices,
        motion_burns,
        objective_scales,
        device=device,
    )
    with restoring_controller_and_optimizer(controller, optimizer, current, optimizer_state):
        raw_proposal, optimizer_after, proposal = train.proposal_from_one_adam_transaction(
            controller, optimizer, current
        )
        full_controls, full_effective = train.materialize_scaled_proposal(
            controller, current, proposal, 1.0
        )
        full_materialized_direction = _directional_change(raw_gradients, full_effective)
        train._load_parameters(controller, current)
        counters_before = train.optimizer_step_counters(optimizer_state)
        counters_after = train.optimizer_step_counters(optimizer_after)
        transaction_pass = bool(
            counters_before == [5.0, 5.0, 5.0] and counters_after == [6.0] * 3
        )
        current_controls_pass = bool(
            all(
                report["all_recurrent_states_and_outputs_finite"]
                and train._all_finite_nested(report)
                for report in baseline_reports
            )
            and current_full["all_recurrent_states_and_outputs_finite"]
            and train._all_finite_nested(current_full)
            and math.isfinite(current_full_value)
            and gradient["gradients_finite"]
            and gradient["all_recurrent_states_and_outputs_finite"]
            and train._all_finite_nested(gradient)
            and full_controls["pass"]
            and transaction_pass
            and full_materialized_direction < -MINIMUM_DIRECTION_MAGNITUDE
        )

        scale_records = [
            _evaluate_scale(
                controller,
                current,
                proposal,
                raw_gradients,
                scale,
                j0,
                dense_sample,
                dense_burn,
                block["motion"],
                motion_indices,
                motion_burns,
                objective_scales,
                device=device,
                noise_threshold=noise_threshold,
            )
            for scale in TAYLOR_SCALES
        ]
    one_sixteenth = scale_records[0]
    reproduction_differences = {
        "fixed_burn_in_objective": abs(j0 - EXPECTED_FIXED_OBJECTIVE),
        "full_prefix_objective": abs(current_full_value - EXPECTED_FULL_PREFIX_OBJECTIVE),
        "materialized_direction": abs(
            full_materialized_direction - EXPECTED_MATERIALIZED_DIRECTION
        ),
        "scale_one_sixteenth_objective": abs(
            one_sixteenth["objective"] - EXPECTED_SCALE_ONE_SIXTEENTH_OBJECTIVE
        ),
        "scale_one_sixteenth_measured_direction": abs(
            one_sixteenth["measured_directional_derivative"]
            - EXPECTED_SCALE_ONE_SIXTEENTH_DIRECTION
        ),
    }
    reproduction_pass = max(reproduction_differences.values()) <= REPRODUCTION_ABSOLUTE_TOLERANCE
    registered_failure_reproduced = bool(
        reproduction_pass
        and one_sixteenth["controls"]["pass"]
        and one_sixteenth["all_metrics_finite"]
        and one_sixteenth["predicted_directional_derivative"] < -MINIMUM_DIRECTION_MAGNITUDE
        and one_sixteenth["measured_directional_derivative"] < -MINIMUM_DIRECTION_MAGNITUDE
        and one_sixteenth["relative_error"] > LOCAL_RELATIVE_ERROR_LIMIT
    )
    passing_pairs = adjacent_passing_pairs(scale_records)
    smaller = scale_records[1:]
    above_noise_pairs = [
        [float(left["scale"]), float(right["scale"])]
        for left, right in zip(smaller, smaller[1:], strict=False)
        if left["objective_change_exceeds_noise_threshold"]
        and right["objective_change_exceeds_noise_threshold"]
    ]

    restored_controller = train.audit.semantic_sha256(train._copy_parameters(controller))
    restored_optimizer = train.audit.semantic_sha256(optimizer.state_dict())
    restoration = {
        "pass": bool(
            controller_before == restored_controller == EXPECTED_CONTROLLER_SHA256
            and optimizer_before == restored_optimizer == EXPECTED_OPTIMIZER_SHA256
        ),
        "controller_before_sha256": controller_before,
        "controller_after_sha256": restored_controller,
        "optimizer_before_sha256": optimizer_before,
        "optimizer_after_sha256": restored_optimizer,
        "candidate_or_optimizer_retained": False,
    }
    all_controls_pass = bool(
        input_identity_pass
        and current_controls_pass
        and reproduction_pass
        and restoration["pass"]
        and all(
            record["controls"]["pass"] and record["all_metrics_finite"]
            for record in scale_records
        )
    )
    passed, classification = audit_classification(
        controls_pass=all_controls_pass,
        registered_failure_reproduced=registered_failure_reproduced,
        adjacent_pairs=passing_pairs,
        adjacent_above_noise_pairs=above_noise_pairs,
    )
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "audit_started_before_computation": True,
        "audit_start_marker_sha256": marker_sha256,
        "pass": passed,
        "classification": classification,
        "locked_inputs": {
            "producer_report_sha256": EXPECTED_REPORT_SHA256,
            "producer_resume_sha256": EXPECTED_RESUME_SHA256,
            "controller_sha256": EXPECTED_CONTROLLER_SHA256,
            "optimizer_sha256": EXPECTED_OPTIMIZER_SHA256,
            "teacher_bank_sha256": EXPECTED_TEACHER_BANK_SHA256,
            "motion_bank_sha256": EXPECTED_MOTION_BANK_SHA256,
            "objective_scales_sha256": EXPECTED_OBJECTIVE_SCALES_SHA256,
            "sample_spec_sha256": EXPECTED_SAMPLE_SPEC_SHA256,
        },
        "accepted_updates": 5,
        "attempt": 6,
        "sample_spec": spec,
        "input_identity_pass": input_identity_pass,
        "baseline": {
            "authoritative_j0_is_first_replay": True,
            "objectives": baseline_values,
            "replay_noise_max_pairwise_absolute": measured_replay_noise,
            "noise_multiplier": REPLAY_NOISE_MULTIPLIER,
            "objective_change_noise_threshold": noise_threshold,
        },
        "current_full_prefix": current_full,
        "gradient": gradient,
        "proposal": {
            "optimizer_betas": list(train.OPTIMIZER_BETAS),
            "optimizer_counters_before": counters_before,
            "optimizer_counters_after_pending_transaction": counters_after,
            "transaction_pass": transaction_pass,
            "raw_parameter_family_rms": {
                name: float((raw_proposal[name] - current[name]).double().square().mean().sqrt())
                for name in train.PARAMETER_FAMILIES
            },
            "full_materialized_directional_derivative": full_materialized_direction,
            "full_materialized_controls": full_controls,
        },
        "current_and_gradient_controls_pass": current_controls_pass,
        "reproduction": {
            "pass": reproduction_pass,
            "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
            "absolute_differences": reproduction_differences,
        },
        "registered_scale_one_sixteenth_failure_reproduced": (
            registered_failure_reproduced
        ),
        "scales": scale_records,
        "adjacent_taylor_residuals": _adjacent_residual_reports(
            scale_records, noise_threshold=noise_threshold
        ),
        "adjacent_local_derivative_agreement_pairs": passing_pairs,
        "adjacent_above_noise_pairs": above_noise_pairs,
        "all_identity_reproduction_restoration_and_numerical_controls_pass": (
            all_controls_pass
        ),
        "restoration": restoration,
        "candidate_or_optimizer_retained": False,
        "continuation_scale_selected": False,
        "midpoint_or_final_data_opened": False,
        "accepted_5_resume_authorized": False,
        "assisted_hover_authorized": False,
        "native_attitude_reintegration_authorized": False,
        "gate_flight_authorized": False,
        "promoted": False,
        "elapsed_seconds": perf_counter() - started,
    }
    train._atomic_json_save(report, args.output_dir / "report.json")
    print(
        json.dumps(
            {
                "output": train.responsibility.stable_path(args.output_dir / "report.json"),
                "pass": passed,
                "classification": classification,
                "replay_noise": measured_replay_noise,
                "scale_one_sixteenth_relative_error": one_sixteenth["relative_error"],
                "adjacent_local_derivative_agreement_pairs": passing_pairs,
                "candidate_retained": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
