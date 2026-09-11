#!/usr/bin/env python3
"""Attribute the accepted-18 assisted-throttle stop to Adam first-moment inertia."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_native_throttle_assisted as train  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402
from flydrone.visual_hover import render_visual_hover_scene  # noqa: E402

EXPERIMENT = "variable-height-native-throttle-optimizer-attribution-v1"
PROTOCOL_COMMIT = "cecdafe"
EXPECTED_REPORT_SHA256 = "ca1562b8764174ac804185104eeb365fa2826241c7f292430d0a81700c177a34"
EXPECTED_RESUME_SHA256 = "46d2d9314b347f9d62168ca5aadbc9538f8889e9d82503095b0488af71d48d68"
EXPECTED_CONTROLLER_SHA256 = "05cb3e44f66c8f19e6cc147c617a6496e8023edd9c97f07be99e0bd1859a46f8"
EXPECTED_OPTIMIZER_SHA256 = "5b9df11d8396df8808af3fab1b916fa9d1b0a1b88031df2439e917152b52da0e"
EXPECTED_TEACHER_BANK_SHA256 = "69993c6de95bfc150b4f2277a53c93da7aa2b6a94c9932d9f76c2893af3d1f62"
EXPECTED_MOTION_BANK_SHA256 = "089bbb8cc72980ca6321f9919227cad5bda1a746041e64426a29455090be6fbd"
EXPECTED_SAMPLE_SPEC_SHA256 = "0042747f8c2430f4024b69c0234aec4eb601c9b2524f90e144bfe6baefa7b175"
REPRODUCTION_ABSOLUTE_TOLERANCE = 2.0e-5


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
        default=(REPO_ROOT / "runs/variable-height-hover/native-throttle-assisted-001/report.json"),
    )
    parser.add_argument(
        "--producer-resume",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/native-throttle-assisted-001/resume.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/native-throttle-optimizer-attribution-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    expected = {
        args.graph: train.EXPECTED_GRAPH_SHA256,
        args.checkpoint: train.EXPECTED_CHECKPOINT_SHA256,
        args.producer_report: EXPECTED_REPORT_SHA256,
        args.producer_resume: EXPECTED_RESUME_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise SystemExit(f"missing audit input: {path}")
        if train.responsibility.file_sha256(path) != digest:
            raise SystemExit(f"audit input hash mismatch: {path}")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("optimizer attribution audit already has a terminal report")
    if (args.output_dir / "audit-started.json").is_file():
        raise SystemExit("optimizer attribution audit was already started and cannot be replayed")


def write_exclusive_start_marker(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(
            "optimizer attribution audit was already started and cannot be replayed"
        ) from error
    return train.responsibility.file_sha256(path)


def _set_gradients(controller: ConnectomeController, gradients: dict[str, Tensor]) -> None:
    for name in train.PARAMETER_FAMILIES:
        parameter = getattr(controller, name)
        parameter.grad = gradients[name].detach().clone()


def _gradient_direction_by_family(
    gradients: dict[str, Tensor], displacement: dict[str, Tensor]
) -> dict[str, float]:
    return {
        name: float((gradients[name].double() * displacement[name].double()).sum())
        for name in train.PARAMETER_FAMILIES
    }


def _direction_total(by_family: dict[str, float]) -> float:
    return float(sum(by_family.values()))


def materialize_optimizer_direction(
    controller: ConnectomeController,
    optimizer_state: dict[str, Any],
    current: dict[str, Tensor],
    clipped_gradients: dict[str, Tensor],
    raw_gradients: dict[str, Tensor],
    *,
    beta1: float,
) -> dict[str, Any]:
    train._load_parameters(controller, current)
    try:
        optimizer = train._make_optimizer(controller)
        state = copy.deepcopy(optimizer_state)
        for group in state["param_groups"]:
            group["betas"] = (beta1, 0.999)
        optimizer.load_state_dict(state)
        parameter_ids = [
            identifier
            for group in optimizer_state["param_groups"]
            for identifier in group["params"]
        ]
        original_second_moments = {
            name: optimizer_state["state"][identifier]["exp_avg_sq"]
            for name, identifier in zip(train.PARAMETER_FAMILIES, parameter_ids, strict=True)
        }
        second_moments_before = {
            name: optimizer.state[getattr(controller, name)]["exp_avg_sq"].detach().clone()
            for name in train.PARAMETER_FAMILIES
        }
        original_second_moment_hash = train.audit.semantic_sha256(original_second_moments)
        loaded_second_moment_hash = train.audit.semantic_sha256(second_moments_before)
        _set_gradients(controller, clipped_gradients)
        optimizer.step()
        first_moment_errors = {}
        second_moment_errors = {}
        second_moment_tolerances = {}
        for name in train.PARAMETER_FAMILIES:
            parameter_state = optimizer.state[getattr(controller, name)]
            first_moment_errors[name] = float(
                (parameter_state["exp_avg"] - clipped_gradients[name]).abs().max()
            )
            expected_second_moment = (
                0.999 * second_moments_before[name] + 0.001 * clipped_gradients[name].square()
            )
            second_moment_errors[name] = float(
                (parameter_state["exp_avg_sq"] - expected_second_moment).abs().max()
            )
            magnitude = float(
                torch.maximum(
                    parameter_state["exp_avg_sq"].abs(), expected_second_moment.abs()
                ).max()
            )
            second_moment_tolerances[name] = (
                1.0e-12 + 4.0 * torch.finfo(parameter_state["exp_avg_sq"].dtype).eps * magnitude
            )
        unprojected = train._copy_parameters(controller)
        unprojected_displacement = {
            name: unprojected[name] - current[name] for name in train.PARAMETER_FAMILIES
        }
        unprojected_by_family = _gradient_direction_by_family(
            raw_gradients, unprojected_displacement
        )
        controller.project_parameters()
        canonical_proposal = train._copy_parameters(controller)
        displacement = {
            name: canonical_proposal[name] - current[name] for name in train.PARAMETER_FAMILIES
        }
        train._load_parameters(controller, current)
        controls, effective = train.materialize_scaled_proposal(
            controller, current, displacement, 1.0
        )
        materialized_by_family = _gradient_direction_by_family(raw_gradients, effective)
        counters_before = train.optimizer_step_counters(optimizer_state)
        counters_after = train.optimizer_step_counters(optimizer.state_dict())
        return {
            "beta1": beta1,
            "beta2": 0.999,
            "optimizer_counters_before": counters_before,
            "optimizer_counters_after": counters_after,
            "moment_mechanism": {
                "first_moment_max_abs_error_from_current_clipped_gradient": (first_moment_errors),
                "second_moment_max_abs_error_from_registered_update": (second_moment_errors),
                "second_moment_dtype_aware_tolerance": second_moment_tolerances,
                "pre_step_second_moment_expected_sha256": original_second_moment_hash,
                "pre_step_second_moment_loaded_sha256": loaded_second_moment_hash,
                "pre_step_second_moment_hash_match": (
                    original_second_moment_hash == loaded_second_moment_hash
                ),
                "first_moment_replaced": bool(
                    beta1 == 0.0 and max(first_moment_errors.values()) == 0.0
                ),
                "second_moment_retained_and_updated": bool(
                    original_second_moment_hash == loaded_second_moment_hash
                    and all(
                        second_moment_errors[name] <= second_moment_tolerances[name]
                        for name in train.PARAMETER_FAMILIES
                    )
                ),
            },
            "unprojected": {
                "by_parameter_family": unprojected_by_family,
                "total": _direction_total(unprojected_by_family),
            },
            "materialized": {
                "by_parameter_family": materialized_by_family,
                "total": _direction_total(materialized_by_family),
                "controls": controls,
            },
            "displacement": displacement,
        }
    finally:
        train._load_parameters(controller, current)


@torch.no_grad()
def evaluate_frozen_teacher_history(
    controller: ConnectomeController,
    bank: dict[str, Any],
    *,
    dense_scale: float,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor, Tensor]:
    batch = int(bank["marker"].shape[0])
    recurrent = controller.initial_state(batch, device=device, dtype=torch.float32)
    scene = train._scene_from_dict(bank["scene"], device=device)
    predictions = []
    recurrent_finite = bool(torch.isfinite(recurrent).all())
    outputs_finite = True
    for step in range(train.EPISODE_STEPS):
        state = train._trajectory_state_at(bank, step, device)
        image = render_visual_hover_scene(
            state,
            bank["marker"][:, step].to(device),
            scene=scene,
        )
        motor, recurrent = controller(image, state.euler[:, :2], recurrent)
        predictions.append(motor[:, 3])
        recurrent_finite &= bool(torch.isfinite(recurrent).all())
        outputs_finite &= bool(torch.isfinite(motor).all())
    prediction = torch.stack(predictions, dim=1)
    target = bank["teacher_motor"][..., 3].to(device)
    mask = bank["eligible"].to(device)
    weights = mask.float()
    count = weights.sum().clamp_min(1.0)
    error = prediction - target
    rmse = ((error.square() * weights).sum() / count).sqrt()
    mae = (error.abs() * weights).sum() / count
    report = {
        "eligible_frames": int(mask.sum()),
        "teacher_target_motor_rmse": float(rmse),
        "teacher_target_motor_nrmse": float(rmse / dense_scale),
        "teacher_target_motor_mae": float(mae),
        "recurrent_states_finite": recurrent_finite,
        "actor_outputs_finite": outputs_finite,
    }
    report["all_metrics_finite"] = bool(
        recurrent_finite and outputs_finite and train._all_finite_nested(report)
    )
    return report, prediction.cpu(), mask.cpu()


def _masked_rms(values: Tensor, mask: Tensor) -> float:
    weights = mask.float()
    return float(((values.square() * weights).sum() / weights.sum().clamp_min(1.0)).sqrt())


def _compact_direction(direction: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in direction.items() if name != "displacement"}


def evaluate_sample_candidate(
    controller: ConnectomeController,
    current: dict[str, Tensor],
    displacement: dict[str, Tensor],
    scale: float,
    dense_sample: dict[str, Any],
    dense_burn: Tensor,
    motion_bank: dict[str, Any],
    motion_indices: list[int],
    motion_burns: list[Tensor],
    scales: dict[str, float],
    *,
    device: torch.device,
) -> tuple[
    dict[str, Any], dict[str, Tensor], dict[str, Any] | None, float | None, dict[str, Any] | None
]:
    try:
        controls, effective = train.materialize_scaled_proposal(
            controller, current, displacement, scale
        )
        if not controls["pass"]:
            return controls, effective, None, None, None
        fixed = train._sample_objective_report(
            controller,
            dense_sample,
            dense_burn,
            motion_bank,
            motion_indices,
            motion_burns,
            scales,
            device=device,
        )
        full_value, full = train.combined_full_prefix_objective(
            controller,
            dense_sample,
            motion_bank,
            motion_indices,
            dense_scale=scales["dense"],
            motion_scale=scales["motion"],
            device=device,
        )
        return controls, effective, fixed, full_value, full
    finally:
        train._load_parameters(controller, current)


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_marker = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "status": "started",
        "replay_permitted": False,
        "producer_report_sha256": EXPECTED_REPORT_SHA256,
        "producer_resume_sha256": EXPECTED_RESUME_SHA256,
        "sample_spec_sha256": EXPECTED_SAMPLE_SPEC_SHA256,
    }
    start_marker_sha256 = write_exclusive_start_marker(
        args.output_dir / "audit-started.json", start_marker
    )
    producer = json.loads(args.producer_report.read_text())
    resume = torch.load(args.producer_resume, map_location="cpu", weights_only=True)
    if producer["classification"] != "fatal_numerical_control_failure":
        raise SystemExit("producer did not stop at the registered numerical gate")
    if resume["run_state"] != "stopped" or resume["accepted_updates"] != 18:
        raise SystemExit("producer resume is not the registered stopped accepted-18 state")
    if resume["controller_parameter_sha256"] != EXPECTED_CONTROLLER_SHA256:
        raise SystemExit("accepted-18 controller hash mismatch")
    if resume["optimizer_sha256"] != EXPECTED_OPTIMIZER_SHA256:
        raise SystemExit("accepted-18 optimizer hash mismatch")
    block = resume["block"]
    if block["teacher"]["sha256"] != EXPECTED_TEACHER_BANK_SHA256:
        raise SystemExit("frozen teacher-history bank hash mismatch")
    if block["motion"]["sha256"] != EXPECTED_MOTION_BANK_SHA256:
        raise SystemExit("frozen motion bank hash mismatch")
    fatal = producer["history"][-1]
    spec = fatal["sample_spec"]
    if train.audit.semantic_sha256(spec) != EXPECTED_SAMPLE_SPEC_SHA256:
        raise SystemExit("failed-minibatch sample-spec hash mismatch")

    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    source = ConnectomeController(args.graph, neural_dt=1.0 / train.POLICY_HZ).to(device)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / train.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    controller.load_state_dict(resume["controller"])
    source.eval().requires_grad_(False)
    controller.eval()
    optimizer_state = copy.deepcopy(resume["optimizer"])
    current = train._copy_parameters(controller)
    controller_before = train.audit.semantic_sha256(current)
    optimizer_before = train.audit.semantic_sha256(optimizer_state)

    dense_sample = train.materialize_dense_sample(block["teacher"], None, spec)
    motion_indices = [int(value) for value in spec["motion_indices"]]
    dense_burn, motion_burns = train._build_fixed_burns(
        controller,
        dense_sample,
        block["motion"],
        motion_indices,
        device=device,
    )
    scales = resume["objective_scales"]
    current_fixed = train._sample_objective_report(
        controller,
        dense_sample,
        dense_burn,
        block["motion"],
        motion_indices,
        motion_burns,
        scales,
        device=device,
    )
    current_full_value, current_full = train.combined_full_prefix_objective(
        controller,
        dense_sample,
        block["motion"],
        motion_indices,
        dense_scale=scales["dense"],
        motion_scale=scales["motion"],
        device=device,
    )
    gradient, raw_gradients = train.accumulate_sample_gradient(
        controller,
        dense_sample,
        dense_burn,
        block["motion"],
        motion_indices,
        motion_burns,
        scales,
        device=device,
    )
    clipped_gradients = {
        name: getattr(controller, name).grad.detach().clone() for name in train.PARAMETER_FAMILIES
    }

    original = materialize_optimizer_direction(
        controller,
        optimizer_state,
        current,
        clipped_gradients,
        raw_gradients,
        beta1=0.9,
    )
    momentum_free = materialize_optimizer_direction(
        controller,
        optimizer_state,
        current,
        clipped_gradients,
        raw_gradients,
        beta1=0.0,
    )
    reproduction_differences = {
        "fixed_burn_in_objective": abs(
            current_fixed["objective"] - fatal["current_fixed_burn_in"]["objective"]
        ),
        "full_prefix_objective": abs(
            current_full_value - fatal["current_full_prefix"]["objective"]
        ),
        "original_materialized_direction": abs(
            original["materialized"]["total"] - fatal["directional_derivative"]
        ),
    }
    reproduction_pass = bool(
        max(reproduction_differences.values()) <= REPRODUCTION_ABSOLUTE_TOLERANCE
    )
    expected_counters_before = [18.0, 18.0, 18.0]
    expected_counters_after = [19.0, 19.0, 19.0]
    transaction_pass = bool(
        original["optimizer_counters_before"] == expected_counters_before
        and original["optimizer_counters_after"] == expected_counters_after
        and momentum_free["optimizer_counters_before"] == expected_counters_before
        and momentum_free["optimizer_counters_after"] == expected_counters_after
    )
    current_controls_pass = bool(
        current_fixed["all_recurrent_states_and_outputs_finite"]
        and current_full["all_recurrent_states_and_outputs_finite"]
        and train._all_finite_nested(current_fixed)
        and train._all_finite_nested(current_full)
        and math.isfinite(current_full_value)
        and gradient["gradients_finite"]
        and gradient["all_recurrent_states_and_outputs_finite"]
        and train._all_finite_nested(gradient)
    )
    direction_pass = bool(
        train._all_finite_nested(_compact_direction(original))
        and train._all_finite_nested(_compact_direction(momentum_free))
        and original["materialized"]["controls"]["pass"]
        and original["materialized"]["total"] > 0.0
        and momentum_free["unprojected"]["total"] < 0.0
        and momentum_free["materialized"]["total"] < 0.0
        and momentum_free["materialized"]["controls"]["pass"]
        and momentum_free["moment_mechanism"]["first_moment_replaced"]
        and momentum_free["moment_mechanism"]["second_moment_retained_and_updated"]
        and transaction_pass
    )
    prerequisite_pass = bool(reproduction_pass and current_controls_pass and direction_pass)

    displacement = momentum_free["displacement"]
    finite_difference: dict[str, Any] = {
        "pass": False,
        "scale": train.FINITE_DIFFERENCE_SCALE,
        "not_run_due_to_failed_prerequisite": not prerequisite_pass,
    }
    trials = []
    selected_scale = None
    numerical_trial_failure = False
    if prerequisite_pass:
        fd_controls, fd_effective, fd_candidate, _, _ = evaluate_sample_candidate(
            controller,
            current,
            displacement,
            train.FINITE_DIFFERENCE_SCALE,
            dense_sample,
            dense_burn,
            block["motion"],
            motion_indices,
            motion_burns,
            scales,
            device=device,
        )
        if fd_candidate is None:
            finite_difference = {
                "pass": False,
                "scale": train.FINITE_DIFFERENCE_SCALE,
                "controls": fd_controls,
                "not_run_due_to_failed_prerequisite": False,
                "candidate_not_evaluated_due_to_failed_controls": True,
            }
        else:
            fd_predicted = _direction_total(
                _gradient_direction_by_family(
                    raw_gradients,
                    {
                        name: fd_effective[name] / train.FINITE_DIFFERENCE_SCALE
                        for name in train.PARAMETER_FAMILIES
                    },
                )
            )
            fd_measured = (
                fd_candidate["objective"] - current_fixed["objective"]
            ) / train.FINITE_DIFFERENCE_SCALE
            fd_objective_change = fd_candidate["objective"] - current_fixed["objective"]
            fd_relative_error = train._relative_error(fd_measured, fd_predicted)
            finite_difference = {
                "pass": bool(
                    fd_controls["pass"]
                    and fd_candidate["all_recurrent_states_and_outputs_finite"]
                    and train._all_finite_nested(fd_candidate)
                    and math.isfinite(fd_predicted)
                    and fd_predicted < -1.0e-8
                    and math.isfinite(fd_measured)
                    and fd_measured < -1.0e-8
                    and fd_objective_change <= -1.0e-8
                    and fd_relative_error <= train.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
                ),
                "scale": train.FINITE_DIFFERENCE_SCALE,
                "autograd_directional_derivative": fd_predicted,
                "measured_directional_derivative": fd_measured,
                "measured_objective_change": fd_objective_change,
                "minimum_absolute_decrease": 1.0e-8,
                "relative_error": fd_relative_error,
                "relative_error_limit": train.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
                "candidate": fd_candidate,
                "controls": fd_controls,
                "not_run_due_to_failed_prerequisite": False,
            }

    if prerequisite_pass and finite_difference["pass"]:
        for scale in train.BACKTRACK_SCALES:
            controls, _, fixed, full_value, full = evaluate_sample_candidate(
                controller,
                current,
                displacement,
                scale,
                dense_sample,
                dense_burn,
                block["motion"],
                motion_indices,
                motion_burns,
                scales,
                device=device,
            )
            if not controls["pass"]:
                trials.append({"scale": scale, "pass": False, "controls": controls})
                numerical_trial_failure = True
                break
            assert fixed is not None and full_value is not None and full is not None
            finite = train.trial_reports_are_finite(fixed, full, full_value)
            fixed_improvement = current_fixed["objective"] - fixed["objective"]
            full_improvement = current_full_value - full_value
            trial_passed = bool(
                finite
                and fixed_improvement >= train.MINIMUM_OBJECTIVE_IMPROVEMENT
                and full_improvement >= train.MINIMUM_OBJECTIVE_IMPROVEMENT
            )
            trials.append(
                {
                    "scale": scale,
                    "pass": trial_passed,
                    "finite": finite,
                    "fixed_burn_in_objective": fixed["objective"],
                    "full_prefix_objective": full_value,
                    "fixed_burn_in_improvement": fixed_improvement,
                    "full_prefix_improvement": full_improvement,
                    "controls": controls,
                }
            )
            if not finite:
                numerical_trial_failure = True
                break
            if trial_passed:
                selected_scale = scale
                break

    source_dense, source_output, dense_mask = evaluate_frozen_teacher_history(
        source,
        block["teacher"],
        dense_scale=scales["dense"],
        device=device,
    )
    accepted_dense, accepted_output, accepted_mask = evaluate_frozen_teacher_history(
        controller,
        block["teacher"],
        dense_scale=scales["dense"],
        device=device,
    )
    if not torch.equal(dense_mask, accepted_mask):
        raise RuntimeError("source and accepted-18 dense masks differ")
    throttle_drift_rms = _masked_rms(accepted_output - source_output, dense_mask)
    source_motion = train.evaluate_motion_bank(
        source,
        block["motion"],
        motion_scale=scales["motion"],
        device=device,
    )
    accepted_motion = train.evaluate_motion_bank(
        controller,
        block["motion"],
        motion_scale=scales["motion"],
        device=device,
    )

    train._load_parameters(controller, current)
    restored_controller = train.audit.semantic_sha256(train._copy_parameters(controller))
    restored_optimizer = train.audit.semantic_sha256(optimizer_state)
    restoration = {
        "pass": bool(
            restored_controller == controller_before == EXPECTED_CONTROLLER_SHA256
            and restored_optimizer == optimizer_before == EXPECTED_OPTIMIZER_SHA256
        ),
        "controller_before_sha256": controller_before,
        "controller_after_sha256": restored_controller,
        "optimizer_before_sha256": optimizer_before,
        "optimizer_after_sha256": restored_optimizer,
        "candidate_or_optimizer_retained": False,
    }
    descriptive_finite = bool(
        source_dense["all_metrics_finite"]
        and accepted_dense["all_metrics_finite"]
        and source_motion["all_metrics_finite"]
        and accepted_motion["all_metrics_finite"]
        and source_motion["all_recurrent_states_and_outputs_finite"]
        and accepted_motion["all_recurrent_states_and_outputs_finite"]
    )
    passed = bool(
        prerequisite_pass
        and finite_difference["pass"]
        and selected_scale is not None
        and not numerical_trial_failure
        and descriptive_finite
        and restoration["pass"]
    )
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "audit_started_before_computation": True,
        "audit_start_marker_sha256": start_marker_sha256,
        "pass": passed,
        "classification": (
            "beta1_zero_optimizer_route_qualified"
            if passed
            else "beta1_zero_optimizer_attribution_failed"
        ),
        "locked_inputs": {
            "producer_report_sha256": EXPECTED_REPORT_SHA256,
            "producer_resume_sha256": EXPECTED_RESUME_SHA256,
            "controller_sha256": EXPECTED_CONTROLLER_SHA256,
            "optimizer_sha256": EXPECTED_OPTIMIZER_SHA256,
            "teacher_bank_sha256": EXPECTED_TEACHER_BANK_SHA256,
            "motion_bank_sha256": EXPECTED_MOTION_BANK_SHA256,
            "sample_spec_sha256": EXPECTED_SAMPLE_SPEC_SHA256,
        },
        "accepted_updates": 18,
        "attempt": 20,
        "sample_spec": spec,
        "current_fixed_burn_in": current_fixed,
        "current_full_prefix": current_full,
        "gradient": gradient,
        "current_and_gradient_controls_pass": current_controls_pass,
        "counterfactual_transactions_pass": transaction_pass,
        "prerequisite_controls_pass": prerequisite_pass,
        "reproduction": {
            "pass": reproduction_pass,
            "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
            "absolute_differences": reproduction_differences,
        },
        "directions": {
            "original_adam": _compact_direction(original),
            "beta1_zero_adam": _compact_direction(momentum_free),
            "pass": direction_pass,
        },
        "beta1_zero_finite_difference": finite_difference,
        "beta1_zero_ordinary_trials": trials,
        "beta1_zero_selected_scale_for_audit_only": selected_scale,
        "training_only_description": {
            "source_dense": source_dense,
            "accepted_18_dense": accepted_dense,
            "accepted_18_source_throttle_drift_rms_motor_units": throttle_drift_rms,
            "accepted_18_source_throttle_drift_nrmse": throttle_drift_rms / scales["dense"],
            "source_motion": source_motion,
            "accepted_18_motion": accepted_motion,
            "generalization_or_hover_evidence": False,
        },
        "restoration": restoration,
        "midpoint_or_final_data_opened": False,
        "accepted_18_resume_authorized": False,
        "new_beta1_zero_preregistration_authorized": passed,
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
                "classification": report["classification"],
                "original_adam_direction": original["materialized"]["total"],
                "beta1_zero_direction": momentum_free["materialized"]["total"],
                "beta1_zero_fd_relative_error": finite_difference.get("relative_error"),
                "beta1_zero_selected_scale_for_audit_only": selected_scale,
                "candidate_retained": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
