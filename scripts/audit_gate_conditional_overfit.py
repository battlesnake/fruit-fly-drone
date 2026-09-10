#!/usr/bin/env python3
"""Test whether the native recurrent graph can overfit a light/heavy action contrast."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_gate_acceleration_path_es import sample_matched_cases  # noqa: E402
from search_gate_motor_interface_es import BalancedCases, stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_acceleration_oracle_distillation import delta_metrics  # noqa: E402
from train_gate_full_network_oracle import (  # noqa: E402
    Trajectories,
    collect_trajectories,
    controller_parameter_sha256,
    controller_step,
    load_frozen_controller,
    parameter_change_summary,
    parameter_snapshot,
    restore_parameters,
)

from flydrone.gate import AnnularGate, GateConfig  # noqa: E402
from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


@dataclass
class EndpointDataset:
    images: Tensor
    roll_pitch: Tensor
    specific_force: Tensor
    stick_position: Tensor
    oracle_motor: Tensor
    reference_motor: Tensor
    mass_scale: Tensor
    source_summary: dict[str, Any]


@dataclass(frozen=True)
class TargetScales:
    contrast_squared: Tensor
    mean_squared: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-mass-oracle-v1" / "candidate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "conditional-overfit-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--prefix-seconds", type=float, default=0.75)
    parser.add_argument("--target-onset-seconds", type=float, default=0.25)
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument(
        "--edge-bias-learning-rates",
        type=float,
        nargs=2,
        default=(1.0e-4, 3.0e-4),
    )
    parser.add_argument("--time-constant-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--gradient-norm-cap", type=float, default=0.5)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--fit-normalized-rmse", type=float, default=0.10)
    parser.add_argument(
        "--gradient-check-scales", type=float, nargs=3, default=(1.0e-3, 3.0e-4, 1.0e-4)
    )
    parser.add_argument("--gradient-check-tolerance", type=float, default=0.10)
    parser.add_argument("--training-seed", type=int, default=970_031)
    parser.add_argument("--holdout-seed", type=int, default=980_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (
        args.graph,
        args.student_checkpoint,
        args.teacher_checkpoint,
        args.calibration,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.pairs != 8:
        raise SystemExit("the bounded representability protocol requires exactly eight pairs")
    if args.updates != 250 or len(args.edge_bias_learning_rates) != 2:
        raise SystemExit("the protocol requires two learning-rate runs of 250 updates")
    positive = (
        args.prefix_seconds,
        args.target_onset_seconds,
        args.time_constant_learning_rate,
        args.gradient_norm_cap,
        args.evaluation_interval,
        args.fit_normalized_rmse,
        args.gradient_check_tolerance,
        *args.edge_bias_learning_rates,
        *args.gradient_check_scales,
    )
    if min(positive) <= 0.0:
        raise SystemExit("times, rates, thresholds, and audit scales must be positive")
    if args.target_onset_seconds >= args.prefix_seconds:
        raise SystemExit("the oracle onset must occur within the recurrent prefix")
    if args.updates % args.evaluation_interval:
        raise SystemExit("the update budget must end on an evaluation interval")
    if round(args.prefix_seconds / dt) != 75:
        raise SystemExit("the declared representability test requires a 75-step prefix")


def exact_mass_cases(
    *,
    pairs: int,
    seed: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> BalancedCases:
    cases = sample_matched_cases(
        2 * pairs,
        seed=seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    mass = torch.tensor((0.92, 1.08), device=device).repeat(pairs)
    if not bool(
        (cases.stratum_code[0::2].bitwise_and(6) == cases.stratum_code[1::2].bitwise_and(6)).all()
    ):
        raise RuntimeError("matched pair geometry codes are not adjacent")
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    distance = 4.4 + 0.4 * torch.rand(pairs, generator=generator)
    lateral_magnitude = 0.65 + 0.30 * torch.rand(pairs, generator=generator)
    height = 1.0 + 0.20 * torch.rand(pairs, generator=generator)
    obliquity_magnitude = torch.deg2rad(15.0 + 10.0 * torch.rand(pairs, generator=generator))
    geometry = cases.stratum_code[0::2].cpu()
    lateral_sign = torch.where(geometry.bitwise_and(2).bool(), 1.0, -1.0)
    obliquity_sign = torch.where(geometry.bitwise_and(4).bool(), 1.0, -1.0)
    pair_center = torch.stack((distance, lateral_sign * lateral_magnitude, height), dim=1).to(
        device
    )
    bearing = torch.atan2(pair_center[:, 1], pair_center[:, 0])
    pair_yaw = bearing + (obliquity_sign * obliquity_magnitude).to(device)
    gate = AnnularGate(
        center=pair_center.repeat_interleave(2, dim=0),
        yaw=pair_yaw.repeat_interleave(2),
    )
    if torch.unique(pair_center, dim=0).shape[0] != pairs:
        raise RuntimeError("representability cases must contain eight distinct geometries")
    return BalancedCases(cases.state, gate, mass, cases.stratum_code)


def to_endpoint_dataset(trajectories: Trajectories, device: torch.device) -> EndpointDataset:
    return EndpointDataset(
        images=trajectories.images.to(device),
        roll_pitch=trajectories.roll_pitch.to(device),
        specific_force=trajectories.specific_force.to(device),
        stick_position=trajectories.stick_position.to(device),
        oracle_motor=trajectories.oracle_motor.to(device),
        reference_motor=trajectories.reference_motor.to(device),
        mass_scale=trajectories.mass_scale.to(device),
        source_summary=trajectories.summary,
    )


def prefix_outputs(
    controller: ConnectomeController,
    dataset: EndpointDataset,
    *,
    requested_steps: tuple[int, ...],
    identical_pair_inputs: bool = False,
) -> dict[int, Tensor]:
    episodes = dataset.mass_scale.numel()
    neural = controller.initial_state(
        episodes, device=dataset.images.device, dtype=dataset.images.dtype
    )
    source = torch.arange(episodes, device=dataset.images.device)
    if identical_pair_inputs:
        source[1::2] = source[0::2]
    requested = set(requested_steps)
    outputs = {}
    for step in range(max(requested_steps)):
        motor, neural = controller_step(
            controller,
            dataset.images[step, source],
            dataset.roll_pitch[step, source],
            neural,
            dataset.specific_force[step, source],
            dataset.stick_position[step, source],
        )
        completed = step + 1
        if completed in requested:
            outputs[completed] = motor
    return outputs


def target_components(
    motor: Tensor, dataset: EndpointDataset, *, step: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    reference = dataset.reference_motor[step - 1]
    target = dataset.oracle_motor[step - 1] - reference
    prediction = motor - reference
    prediction_contrast = prediction[0::2, 3] - prediction[1::2, 3]
    target_contrast = target[0::2, 3] - target[1::2, 3]
    prediction_mean = 0.5 * (prediction[0::2, 3] + prediction[1::2, 3])
    target_mean = 0.5 * (target[0::2, 3] + target[1::2, 3])
    return prediction_contrast, target_contrast, prediction_mean, target_mean


def fixed_target_scales(dataset: EndpointDataset, *, step: int) -> TargetScales:
    target = dataset.oracle_motor[step - 1] - dataset.reference_motor[step - 1]
    contrast = target[0::2, 3] - target[1::2, 3]
    pair_mean = 0.5 * (target[0::2, 3] + target[1::2, 3])
    return TargetScales(
        contrast_squared=contrast.square().mean().clamp_min(1.0e-8),
        mean_squared=pair_mean.square().mean().clamp_min(1.0e-8),
    )


def endpoint_objective(
    controller: ConnectomeController,
    dataset: EndpointDataset,
    scales: TargetScales,
    *,
    step: int,
) -> tuple[Tensor, dict[str, float]]:
    motor = prefix_outputs(controller, dataset, requested_steps=(step,))[step]
    prediction_contrast, target_contrast, prediction_mean, target_mean = target_components(
        motor, dataset, step=step
    )
    contrast_loss = (prediction_contrast - target_contrast).square().mean() / (
        scales.contrast_squared + 1.0e-8
    )
    mean_loss = (prediction_mean - target_mean).square().mean() / (scales.mean_squared + 1.0e-8)
    total = contrast_loss + mean_loss
    return total, {
        "contrast_normalized_mse": float(contrast_loss.detach()),
        "mean_normalized_mse": float(mean_loss.detach()),
        "total_loss": float(total.detach()),
    }


def endpoint_metrics(
    motor: Tensor,
    dataset: EndpointDataset,
    scales: TargetScales,
    *,
    step: int,
) -> dict[str, Any]:
    motor = motor.detach()
    prediction_contrast, target_contrast, prediction_mean, target_mean = target_components(
        motor, dataset, step=step
    )
    contrast_rmse = (prediction_contrast - target_contrast).square().mean().sqrt()
    mean_rmse = (prediction_mean - target_mean).square().mean().sqrt()
    correction = motor[:, 3] - dataset.reference_motor[step - 1, :, 3]
    target = dataset.oracle_motor[step - 1, :, 3] - dataset.reference_motor[step - 1, :, 3]
    return {
        "step": step,
        "seconds": step * 0.01,
        "contrast_normalized_rmse": float(contrast_rmse / scales.contrast_squared.sqrt()),
        "mean_normalized_rmse": float(mean_rmse / scales.mean_squared.sqrt()),
        "contrast": delta_metrics(prediction_contrast, target_contrast),
        "pair_mean": delta_metrics(prediction_mean, target_mean),
        "all_throttle_correction": delta_metrics(correction, target),
        "light_prediction_mean": float(correction[0::2].mean()),
        "light_target_mean": float(target[0::2].mean()),
        "heavy_prediction_mean": float(correction[1::2].mean()),
        "heavy_target_mean": float(target[1::2].mean()),
    }


def objective_score(metrics: dict[str, Any]) -> float:
    values = (
        metrics["contrast_normalized_rmse"],
        metrics["mean_normalized_rmse"],
    )
    return max(values) if all(math.isfinite(value) for value in values) else float("inf")


def gradient_audit(
    controller: ConnectomeController,
    dataset: EndpointDataset,
    scales: TargetScales,
    *,
    step: int,
    perturbation_scales: tuple[float, ...],
    tolerance: float,
) -> dict[str, Any]:
    parameters = {
        "edge_magnitude": controller.edge_magnitude,
        "bias": controller.bias,
        "raw_time_constant": controller.raw_time_constant,
    }
    baseline = parameter_snapshot(controller)
    repeats = []
    with torch.no_grad():
        for _ in range(3):
            repeats.append(float(endpoint_objective(controller, dataset, scales, step=step)[0]))
    repeat_noise = max(repeats) - min(repeats)
    controller.zero_grad(set_to_none=True)
    loss, _ = endpoint_objective(controller, dataset, scales, step=step)
    loss.backward()
    report: dict[str, Any] = {}
    try:
        for name, parameter in parameters.items():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"missing or nonfinite gradient for {name}")
            direction = parameter.grad.detach().sign()
            if name == "edge_magnitude":
                direction[(parameter.detach() <= 0.01) | (parameter.detach() >= 7.99)] = 0.0
            analytic = float((parameter.grad * direction).sum())
            checks = []
            for epsilon in perturbation_scales:
                with torch.no_grad():
                    parameter.copy_(baseline[name] + epsilon * direction)
                    plus = float(endpoint_objective(controller, dataset, scales, step=step)[0])
                    parameter.copy_(baseline[name] - epsilon * direction)
                    minus = float(endpoint_objective(controller, dataset, scales, step=step)[0])
                    parameter.copy_(baseline[name])
                estimate = (plus - minus) / (2.0 * epsilon)
                relative_error = abs(estimate - analytic) / max(
                    abs(estimate), abs(analytic), 1.0e-10
                )
                checks.append(
                    {
                        "epsilon": epsilon,
                        "plus_loss": plus,
                        "minus_loss": minus,
                        "central_difference": estimate,
                        "analytic": analytic,
                        "relative_error": relative_error,
                        "matching_sign": estimate * analytic > 0.0,
                        "measurable_above_repeat_noise": abs(plus - minus)
                        > max(10.0 * repeat_noise, 1.0e-10),
                    }
                )
            adjacent = [
                left["matching_sign"]
                and right["matching_sign"]
                and left["measurable_above_repeat_noise"]
                and right["measurable_above_repeat_noise"]
                and left["relative_error"] <= tolerance
                and right["relative_error"] <= tolerance
                for left, right in zip(checks[:-1], checks[1:], strict=True)
            ]
            report[name] = {
                "gradient_l2_norm": float(parameter.grad.norm()),
                "analytic_directional_derivative": analytic,
                "checks": checks,
                "passed": any(adjacent),
            }
            if not report[name]["passed"]:
                raise RuntimeError(f"full-prefix gradient audit failed for {name}: {checks}")
    finally:
        restore_parameters(controller, baseline)
        controller.zero_grad(set_to_none=True)
    return {
        "full_prefix_steps": step,
        "complete_prefix_in_autograd": True,
        "unchanged_loss_repeats": repeats,
        "unchanged_loss_range": repeat_noise,
        "families": report,
        "passed": all(item["passed"] for item in report.values()),
    }


def train_one_rate(
    source: ConnectomeController,
    dataset: EndpointDataset,
    scales: TargetScales,
    *,
    learning_rate: float,
    tau_learning_rate: float,
    updates: int,
    evaluation_interval: int,
    gradient_norm_cap: float,
    step: int,
) -> dict[str, Any]:
    controller = copy.deepcopy(source).to(dataset.images.device)
    for parameter in controller.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        (
            {"params": [controller.edge_magnitude, controller.bias], "lr": learning_rate},
            {"params": [controller.raw_time_constant], "lr": tau_learning_rate},
        )
    )
    with torch.no_grad():
        initial_metrics = endpoint_metrics(
            prefix_outputs(controller, dataset, requested_steps=(step,))[step],
            dataset,
            scales,
            step=step,
        )
    best = {
        "update": 0,
        "score": objective_score(initial_metrics),
        "metrics": initial_metrics,
        "parameters": parameter_snapshot(controller, cpu=True),
    }
    history = []
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, details = endpoint_objective(controller, dataset, scales, step=step)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite overfit loss at update {update}")
        loss.backward()
        raw_gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.detach().norm()
                    for parameter in controller.parameters()
                    if parameter.grad is not None
                ]
            )
        )
        torch.nn.utils.clip_grad_norm_(
            controller.parameters(), gradient_norm_cap, error_if_nonfinite=True
        )
        optimizer.step()
        controller.project_parameters()
        if update % evaluation_interval:
            continue
        with torch.no_grad():
            motor = prefix_outputs(controller, dataset, requested_steps=(step,))[step]
            metrics = endpoint_metrics(motor, dataset, scales, step=step)
        score = objective_score(metrics)
        record = {
            "update": update,
            "score": score,
            "raw_gradient_norm": float(raw_gradient_norm),
            **details,
            "metrics": metrics,
        }
        history.append(record)
        if score < best["score"]:
            best = {
                "update": update,
                "score": score,
                "metrics": metrics,
                "parameters": parameter_snapshot(controller, cpu=True),
            }
        print(
            json.dumps(
                {
                    "phase": "overfit",
                    "learning_rate": learning_rate,
                    "update": update,
                    "score": score,
                    "contrast_normalized_rmse": metrics["contrast_normalized_rmse"],
                    "mean_normalized_rmse": metrics["mean_normalized_rmse"],
                }
            ),
            flush=True,
        )
    restore_parameters(controller, best["parameters"])
    return {
        "learning_rate": learning_rate,
        "initial_metrics": initial_metrics,
        "history": history,
        "best": best,
        "selected_parameter_sha256": controller_parameter_sha256(controller),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source, source_checkpoint, hover_config, gate_config, resolution = load_frozen_controller(
        args.graph, args.student_checkpoint, device
    )
    teacher, _, teacher_hover, teacher_gate, teacher_resolution = load_frozen_controller(
        args.graph, args.teacher_checkpoint, device
    )
    validate_args(args, hover_config.dt)
    if (
        asdict(hover_config) != asdict(teacher_hover)
        or asdict(gate_config) != asdict(teacher_gate)
        or resolution != teacher_resolution
    ):
        raise SystemExit("student and teacher simulation contracts differ")
    if not source.uses_accelerometer or source.uses_proprioception:
        raise SystemExit("unexpected deployed sensor contract")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("unexpected teacher calibration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.training_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    steps = round(args.prefix_seconds / hover_config.dt)

    cases = exact_mass_cases(
        pairs=args.pairs,
        seed=args.training_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    trajectories = collect_trajectories(
        source,
        teacher,
        cases,
        calibration,
        seconds=args.prefix_seconds,
        target_onset_seconds=args.target_onset_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        teacher_drives_physics=False,
    )
    dataset = to_endpoint_dataset(trajectories, device)
    if not torch.equal(
        dataset.mass_scale, torch.tensor((0.92, 1.08), device=device).repeat(args.pairs)
    ):
        raise RuntimeError("training masses do not match the declared exact pair")
    scales = fixed_target_scales(dataset, step=steps)

    audit_controller = copy.deepcopy(source).to(device)
    for parameter in audit_controller.parameters():
        parameter.requires_grad_(True)
    audit = gradient_audit(
        audit_controller,
        dataset,
        scales,
        step=steps,
        perturbation_scales=tuple(args.gradient_check_scales),
        tolerance=args.gradient_check_tolerance,
    )
    del audit_controller
    print(json.dumps({"phase": "full_prefix_gradient_audit", "passed": True}), flush=True)

    runs = []
    archive = {}
    for learning_rate in args.edge_bias_learning_rates:
        result = train_one_rate(
            source,
            dataset,
            scales,
            learning_rate=learning_rate,
            tau_learning_rate=args.time_constant_learning_rate,
            updates=args.updates,
            evaluation_interval=args.evaluation_interval,
            gradient_norm_cap=args.gradient_norm_cap,
            step=steps,
        )
        key = result["selected_parameter_sha256"]
        archive[key] = {
            "learning_rate": learning_rate,
            "update": result["best"]["update"],
            "parameters": result["best"]["parameters"],
            "score": result["best"]["score"],
        }
        runs.append(result)

    selected_run = min(runs, key=lambda item: item["best"]["score"])
    selected_key = selected_run["selected_parameter_sha256"]
    selected = copy.deepcopy(source).to(device)
    restore_parameters(selected, selected_run["best"]["parameters"])
    with torch.no_grad():
        selected_motor = prefix_outputs(selected, dataset, requested_steps=(steps,))[steps]
        selected_metrics = endpoint_metrics(selected_motor, dataset, scales, step=steps)
    identical_controller = copy.deepcopy(selected).to(torch.device("cpu"))
    identical_dataset = to_endpoint_dataset(trajectories, torch.device("cpu"))
    with torch.no_grad():
        identical_motor = prefix_outputs(
            identical_controller,
            identical_dataset,
            requested_steps=(steps,),
            identical_pair_inputs=True,
        )[steps]
    identical_pair_maximum_difference = float(
        (identical_motor[0::2] - identical_motor[1::2]).abs().max()
    )
    training_fit_passed = bool(
        selected_metrics["contrast_normalized_rmse"] <= args.fit_normalized_rmse
        and selected_metrics["mean_normalized_rmse"] <= args.fit_normalized_rmse
    )
    identical_input_control_passed = identical_pair_maximum_difference == 0.0
    representability_passed = training_fit_passed and identical_input_control_passed

    holdout = None
    if representability_passed:
        holdout_cases = exact_mass_cases(
            pairs=args.pairs,
            seed=args.holdout_seed,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        holdout_trajectories = collect_trajectories(
            source,
            teacher,
            holdout_cases,
            calibration,
            seconds=args.prefix_seconds,
            target_onset_seconds=args.target_onset_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            teacher_drives_physics=False,
        )
        holdout_dataset = to_endpoint_dataset(holdout_trajectories, device)
        requested = (round(0.5 / hover_config.dt), steps)
        with torch.no_grad():
            holdout_outputs = prefix_outputs(selected, holdout_dataset, requested_steps=requested)
        holdout = {
            f"{step * hover_config.dt:.2f}": endpoint_metrics(
                holdout_outputs[step],
                holdout_dataset,
                fixed_target_scales(holdout_dataset, step=step),
                step=step,
            )
            for step in requested
        }

    vector_path = args.output_dir / "candidate-vector.json"
    vector_path.write_text(
        json.dumps(
            {
                "source_checkpoint_sha256": file_sha256(args.student_checkpoint),
                "selected_learning_rate": selected_run["learning_rate"],
                "selected_update": selected_run["best"]["update"],
                "parameter_vector_sha256": selected_key,
                "edge_magnitude": selected.edge_magnitude.detach().cpu().tolist(),
                "bias": selected.bias.detach().cpu().tolist(),
                "raw_time_constant": selected.raw_time_constant.detach().cpu().tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    torch.save(archive, args.output_dir / "archive.pt")
    clean_runs = []
    for run in runs:
        clean_runs.append(
            {key: value for key, value in run.items() if key != "best"}
            | {"best": {key: value for key, value in run["best"].items() if key != "parameters"}}
        )
    report = {
        "method": "tiny exact-mass conditional endpoint overfit",
        "claim_scope": (
            "This is a native recurrent representability/optimization diagnostic on training "
            "examples, not a flight or generalization result. The actor receives only current "
            "FPV, roll/pitch, body-Z specific force, and native recurrent state."
        ),
        "counts_toward_gate_goal": False,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "student_checkpoint": stable_path(args.student_checkpoint),
        "student_checkpoint_sha256": file_sha256(args.student_checkpoint),
        "teacher_checkpoint": stable_path(args.teacher_checkpoint),
        "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "exact_mass_scales": [0.92, 1.08],
            "distinct_gate_geometry_pairs": args.pairs,
            "gate_geometry_ranges": {
                "distance_m": [4.4, 4.8],
                "lateral_offset_magnitude": [0.65, 0.95],
                "height_m": [1.0, 1.2],
                "obliquity_magnitude_degrees": [15.0, 25.0],
            },
            "student_driven_frozen_prefixes": True,
            "full_neural_prefix_in_autograd": True,
            "physics_and_renderer_outside_autograd": True,
            "pair_mean_and_pair_contrast_equal_weight": True,
            "fixed_topology": True,
            "fixed_transmitter_signs": True,
            "all_native_edge_magnitudes_trainable": True,
            "all_native_biases_trainable": True,
            "all_native_time_constants_trainable": True,
            "engineered_history_features": False,
            "batch_position_actor_input": False,
            "mass_actor_input": False,
            "teacher_actor_input": False,
            "proprioception_input_active": selected.uses_proprioception,
        },
        "source_checkpoint_metadata_keys": sorted(source_checkpoint),
        "training_collection": trajectories.summary,
        "target_scales": {
            "contrast_rms": float(scales.contrast_squared.sqrt()),
            "mean_rms": float(scales.mean_squared.sqrt()),
        },
        "full_prefix_gradient_audit": audit,
        "runs": clean_runs,
        "selected_parameter_vector_sha256": selected_key,
        "selected_candidate_vector_file": stable_path(vector_path),
        "selected_candidate_vector_file_sha256": file_sha256(vector_path),
        "selected_training_metrics": selected_metrics,
        "selected_parameter_change": parameter_change_summary(selected, parameter_snapshot(source)),
        "identical_input_control": {
            "pair_maximum_absolute_motor_difference": identical_pair_maximum_difference,
            "passed": identical_input_control_passed,
        },
        "training_fit_passed": training_fit_passed,
        "representability_passed": representability_passed,
        "holdout_without_further_training": holdout,
        "checkpoint_promoted": False,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "training_fit_passed": training_fit_passed,
                "representability_passed": representability_passed,
                "selected_score": objective_score(selected_metrics),
            }
        ),
        flush=True,
    )
    return 0 if representability_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
