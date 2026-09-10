#!/usr/bin/env python3
"""Audit whether native recurrent state can route useful throttle corrections."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_analytic_teachers import teacher_rc_for_mode  # noqa: E402
from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_acceleration_path_es import make_path_spec  # noqa: E402
from search_gate_motor_interface_es import clone_state, load_controller, stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    motor_target_for_rc,
)

WINDOW_LABELS = ("0.50:0.75", "0.75:1.00", "1.00:1.50")
RIDGE_STRENGTHS = (1.0e-6, 1.0e-4, 1.0e-2, 1.0)


@dataclass(frozen=True)
class ReadoutSpec:
    edges: Tensor
    edge_slots: Tensor
    motors: Tensor
    motor_pool: Tensor
    pool_sizes: Tensor
    return_sources: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "throttle-routing-audit-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--development-pairs", type=int, default=64)
    parser.add_argument("--held-out-pairs", type=int, default=128)
    parser.add_argument("--probe-fit-pairs", type=int, default=48)
    parser.add_argument("--probe-validation-pairs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=1.50)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--native-fit-updates", type=int, default=200)
    parser.add_argument("--native-learning-rate", type=float, default=0.03)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=8.0)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--maximum-group-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-constant-improvement", type=float, default=0.50)
    parser.add_argument("--development-seed", type=int, default=1_040_031)
    parser.add_argument("--held-out-seed", type=int, default=1_041_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "development_pairs": (args.development_pairs, 64),
        "held_out_pairs": (args.held_out_pairs, 128),
        "probe_fit_pairs": (args.probe_fit_pairs, 48),
        "probe_validation_pairs": (args.probe_validation_pairs, 16),
        "seconds": (args.seconds, 1.50),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "native_fit_updates": (args.native_fit_updates, 200),
        "native_learning_rate": (args.native_learning_rate, 0.03),
        "maximum_edge_magnitude": (args.maximum_edge_magnitude, 8.0),
        "normalization_floor": (args.normalization_floor, 0.01),
        "maximum_group_nrmse": (args.maximum_group_nrmse, 0.25),
        "minimum_constant_improvement": (args.minimum_constant_improvement, 0.50),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered routing-audit values changed: {', '.join(wrong)}")
    if args.probe_fit_pairs + args.probe_validation_pairs != args.development_pairs:
        raise SystemExit("probe split must exactly partition development geometry pairs")


def make_readout_spec(controller: ConnectomeController) -> ReadoutSpec:
    positive_begin = int(controller.pool_offsets[6])
    positive_end = int(controller.pool_offsets[7])
    negative_end = int(controller.pool_offsets[8])
    motors = controller.pool_indices[positive_begin:negative_end]
    incoming = torch.isin(controller.edge_post, motors)
    edges = torch.nonzero(incoming, as_tuple=False).flatten()
    node_to_slot = {int(node): slot for slot, node in enumerate(motors.tolist())}
    edge_slots = torch.tensor(
        [node_to_slot[int(controller.edge_post[edge])] for edge in edges],
        device=motors.device,
        dtype=torch.long,
    )
    motor_pool = torch.cat(
        (
            torch.zeros(positive_end - positive_begin, device=motors.device, dtype=torch.long),
            torch.ones(negative_end - positive_end, device=motors.device, dtype=torch.long),
        )
    )
    pool_sizes = torch.tensor(
        (positive_end - positive_begin, negative_end - positive_end),
        device=motors.device,
    )
    return ReadoutSpec(
        edges=edges,
        edge_slots=edge_slots,
        motors=motors,
        motor_pool=motor_pool,
        pool_sizes=pool_sizes,
        return_sources=torch.unique(controller.edge_pre[edges]),
    )


def recurrent_drive(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    neural: Tensor,
    acceleration: Tensor,
    stick_position: Tensor,
) -> Tensor:
    activity = torch.tanh(neural)
    messages = activity[:, controller.edge_pre] * controller.edge_sign * controller.edge_magnitude
    recurrent = torch.zeros_like(neural).index_add(1, controller.edge_post, messages)
    return (
        recurrent
        + controller.bias
        + controller.sensory_drive(image, roll_pitch, acceleration, stick_position)
    )


def window_index(step: int, *, dt: float) -> int | None:
    time_seconds = step * dt
    if 0.50 <= time_seconds < 0.75:
        return 0
    if 0.75 <= time_seconds < 1.00:
        return 1
    if 1.00 <= time_seconds < 1.50:
        return 2
    return None


def selected_edge_features(
    controller: ConnectomeController,
    neural: Tensor,
    spec: ReadoutSpec,
) -> Tensor:
    return torch.tanh(neural[:, controller.edge_pre[spec.edges]]) * controller.edge_sign[spec.edges]


def pool_difference(motor_states: Tensor, spec: ReadoutSpec) -> Tensor:
    pools = torch.zeros(
        len(motor_states), 2, device=motor_states.device, dtype=motor_states.dtype
    ).index_add(1, spec.motor_pool, torch.sigmoid(motor_states))
    pools = pools / spec.pool_sizes.to(dtype=motor_states.dtype)
    return pools[:, 0] - pools[:, 1]


def exact_residual(
    magnitudes: Tensor,
    features: dict[str, Tensor],
    spec: ReadoutSpec,
) -> Tensor:
    edge_features = features["edge_features"]
    selected_current = torch.zeros(
        len(edge_features), len(spec.motors), device=edge_features.device, dtype=edge_features.dtype
    ).index_add(1, spec.edge_slots, edge_features * magnitudes)
    drive = features["drive_without_selected"] + selected_current
    target_state = 5.0 * torch.tanh(drive / 5.0)
    next_motor_state = features["previous_motor_state"] + features["motor_alpha"] * (
        target_state - features["previous_motor_state"]
    )
    return pool_difference(next_motor_state, spec) - features["source_motor"]


def _stack(records: dict[str, list[Tensor]]) -> dict[str, Tensor]:
    return {name: torch.cat(values, dim=0).detach().cpu() for name, values in records.items()}


@torch.no_grad()
def collect_history(
    controller: ConnectomeController,
    path_nodes: Tensor,
    spec: ReadoutSpec,
    *,
    pairs: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    seconds: float,
    takeover_seconds: float,
) -> dict[str, Any]:
    episodes = 2 * pairs
    cases = diverse_matched_cases(
        episodes,
        seed=seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    step_count = round(seconds / hover_config.dt)
    takeover_step = round(takeover_seconds / hover_config.dt)
    sensor_records: dict[str, list[Tensor]] = {
        "image": [],
        "roll_pitch": [],
        "acceleration": [],
        "stick_position": [],
    }
    sample_records: dict[str, list[Tensor]] = {
        "path_activity": [],
        "return_motor_features": [],
        "edge_features": [],
        "drive_without_selected": [],
        "previous_motor_state": [],
        "motor_alpha": [],
        "source_motor": [],
        "reserve_motor": [],
        "target": [],
        "measured_throttle_before": [],
        "window": [],
        "mass_half": [],
        "episode": [],
        "pair": [],
        "time_seconds": [],
    }
    maximum_manual_state_difference = 0.0
    maximum_manual_motor_difference = 0.0
    baseline_magnitudes = controller.edge_magnitude[spec.edges]
    motor_alpha = (1.0 - torch.exp(-controller.neural_dt / controller.time_constant[spec.motors]))[
        None
    ]
    episode_index = torch.arange(episodes, device=device)
    for step in range(step_count):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        roll_pitch = state.euler[:, :2]
        acceleration = state.specific_force
        sensor_records["image"].append(image[None].clone())
        sensor_records["roll_pitch"].append(roll_pitch[None].clone())
        sensor_records["acceleration"].append(acceleration[None].clone())
        sensor_records["stick_position"].append(stick_state.position[None].clone())
        drive = recurrent_drive(
            controller,
            image,
            roll_pitch,
            neural,
            acceleration,
            stick_state.position,
        )
        source_motor, next_neural = controller(
            image,
            roll_pitch,
            neural,
            acceleration,
            stick_state.position,
        )
        manual_target = 5.0 * torch.tanh(drive / 5.0)
        manual_alpha = 1.0 - torch.exp(-controller.neural_dt / controller.time_constant)
        manual_next = neural + manual_alpha * (manual_target - neural)
        manual_motor = controller.motor_drive(manual_next)
        maximum_manual_state_difference = max(
            maximum_manual_state_difference,
            float((manual_next - next_neural).abs().max()),
        )
        maximum_manual_motor_difference = max(
            maximum_manual_motor_difference,
            float((manual_motor - source_motor).abs().max()),
        )
        reserve_rc = teacher_rc_for_mode(
            "visual_accelerometer_reserve",
            controller,
            state,
            cases.gate,
            cases.mass_scale,
            hover_config,
        )
        reserve_motor = motor_target_for_rc(reserve_rc, hover_config)
        selected = selected_edge_features(controller, neural, spec)
        selected_current = torch.zeros(
            episodes, len(spec.motors), device=device, dtype=neural.dtype
        ).index_add(1, spec.edge_slots, selected * baseline_magnitudes)
        selected_window = window_index(step, dt=hover_config.dt)
        if selected_window is not None:
            measured_before = (stick_state.position[:, 3] + 1.0) / 2.0
            sample_records["path_activity"].append(torch.tanh(neural[:, path_nodes]))
            sample_records["return_motor_features"].append(
                torch.cat(
                    (torch.tanh(neural[:, spec.return_sources]), neural[:, spec.motors]), dim=1
                )
            )
            sample_records["edge_features"].append(selected)
            sample_records["drive_without_selected"].append(
                drive[:, spec.motors] - selected_current
            )
            sample_records["previous_motor_state"].append(neural[:, spec.motors])
            sample_records["motor_alpha"].append(motor_alpha.expand(episodes, -1))
            sample_records["source_motor"].append(source_motor[:, 3])
            sample_records["reserve_motor"].append(reserve_motor[:, 3])
            sample_records["target"].append(reserve_motor[:, 3] - source_motor[:, 3])
            sample_records["measured_throttle_before"].append(measured_before)
            sample_records["window"].append(
                torch.full((episodes,), selected_window, device=device, dtype=torch.long)
            )
            sample_records["mass_half"].append((cases.mass_scale > 1.0).long())
            sample_records["episode"].append(episode_index)
            sample_records["pair"].append(episode_index // 2)
            sample_records["time_seconds"].append(
                torch.full((episodes,), step * hover_config.dt, device=device)
            )
        applied_motor = source_motor.clone()
        if step >= takeover_step:
            applied_motor[:, :3] = reserve_motor[:, :3]
        rc, stick_state = sticks(applied_motor, stick_state)
        state = quad(rc, state, cases.mass_scale)
        neural = next_neural
    return {
        "sensors": {
            name: torch.cat(values, dim=0).cpu() for name, values in sensor_records.items()
        },
        "samples": _stack(sample_records),
        "mass_scale": cases.mass_scale.cpu(),
        "stratum_code": cases.stratum_code.cpu(),
        "seed": seed,
        "pairs": pairs,
        "episodes": episodes,
        "steps": step_count,
        "source_forward_parity": {
            "maximum_state_absolute_difference": maximum_manual_state_difference,
            "maximum_motor_absolute_difference": maximum_manual_motor_difference,
            "passed": maximum_manual_state_difference <= 1.0e-6
            and maximum_manual_motor_difference <= 1.0e-6,
        },
    }


@torch.no_grad()
def replay_history(
    controller: ConnectomeController,
    history: dict[str, Any],
    path_nodes: Tensor,
    spec: ReadoutSpec,
    *,
    acceleration_control: str,
    device: torch.device,
) -> dict[str, Tensor]:
    if acceleration_control not in {"live", "constant_1g", "pair_swapped"}:
        raise ValueError(f"unknown replay acceleration control: {acceleration_control}")
    sensors = {name: value.to(device) for name, value in history["sensors"].items()}
    episodes = history["episodes"]
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    baseline_magnitudes = controller.edge_magnitude[spec.edges]
    motor_alpha = (1.0 - torch.exp(-controller.neural_dt / controller.time_constant[spec.motors]))[
        None
    ]
    records: dict[str, list[Tensor]] = {
        "path_activity": [],
        "return_motor_features": [],
        "edge_features": [],
        "drive_without_selected": [],
        "previous_motor_state": [],
        "motor_alpha": [],
        "source_motor": [],
    }
    swap = torch.arange(episodes, device=device).bitwise_xor(1)
    for step in range(history["steps"]):
        image = sensors["image"][step]
        roll_pitch = sensors["roll_pitch"][step]
        stick_position = sensors["stick_position"][step]
        acceleration = sensors["acceleration"][step]
        if acceleration_control == "constant_1g":
            acceleration = torch.zeros_like(acceleration)
            acceleration[:, 2] = 9.81
        elif acceleration_control == "pair_swapped":
            acceleration = acceleration[swap]
        drive = recurrent_drive(
            controller,
            image,
            roll_pitch,
            neural,
            acceleration,
            stick_position,
        )
        source_motor, next_neural = controller(
            image,
            roll_pitch,
            neural,
            acceleration,
            stick_position,
        )
        if window_index(step, dt=controller.neural_dt) is not None:
            selected = selected_edge_features(controller, neural, spec)
            selected_current = torch.zeros(
                episodes, len(spec.motors), device=device, dtype=neural.dtype
            ).index_add(1, spec.edge_slots, selected * baseline_magnitudes)
            records["path_activity"].append(torch.tanh(neural[:, path_nodes]))
            records["return_motor_features"].append(
                torch.cat(
                    (torch.tanh(neural[:, spec.return_sources]), neural[:, spec.motors]), dim=1
                )
            )
            records["edge_features"].append(selected)
            records["drive_without_selected"].append(drive[:, spec.motors] - selected_current)
            records["previous_motor_state"].append(neural[:, spec.motors])
            records["motor_alpha"].append(motor_alpha.expand(episodes, -1))
            records["source_motor"].append(source_motor[:, 3])
        neural = next_neural
    replay = _stack(records)
    for name in ("target", "window", "mass_half", "episode", "pair", "time_seconds"):
        replay[name] = history["samples"][name]
    return replay


def group_indices(samples: dict[str, Tensor]) -> Tensor:
    return 2 * samples["window"].long() + samples["mass_half"].long()


def equal_group_weights(groups: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(groups), dtype=np.float64)
    for group in range(6):
        selected = groups == group
        if not selected.any():
            raise RuntimeError(f"routing audit has no samples for group {group}")
        weights[selected] = 1.0 / float(selected.sum())
    return weights / weights.sum()


def target_scales(samples: dict[str, Tensor], *, floor: float) -> dict[int, float]:
    target = samples["target"].double()
    groups = group_indices(samples)
    return {
        group: max(float(target[groups == group].square().mean().sqrt()), floor)
        for group in range(6)
    }


def metric_summary(
    prediction: np.ndarray,
    samples: dict[str, Tensor],
    scales: dict[int, float],
    *,
    constant_aggregate_rmse: float | None,
) -> dict[str, Any]:
    target = samples["target"].numpy().astype(np.float64, copy=False)
    groups = group_indices(samples).numpy()
    per_group: dict[str, Any] = {}
    mean_squares = []
    normalized_mean_squares = []
    for window, label in enumerate(WINDOW_LABELS):
        for mass, mass_label in enumerate(("light", "heavy")):
            group = 2 * window + mass
            selected = groups == group
            error = prediction[selected] - target[selected]
            rmse = float(np.sqrt(np.mean(error**2)))
            nrmse = rmse / scales[group]
            per_group[f"{label}/{mass_label}"] = {
                "samples": int(selected.sum()),
                "target_rms": float(np.sqrt(np.mean(target[selected] ** 2))),
                "normalization_scale": scales[group],
                "rmse": rmse,
                "nrmse": nrmse,
                "bias": float(error.mean()),
            }
            mean_squares.append(rmse**2)
            normalized_mean_squares.append(nrmse**2)
    aggregate_rmse = float(np.sqrt(np.mean(mean_squares)))
    result = {
        "aggregate_equal_group_rmse": aggregate_rmse,
        "aggregate_equal_group_nrmse": float(np.sqrt(np.mean(normalized_mean_squares))),
        "maximum_group_nrmse": max(item["nrmse"] for item in per_group.values()),
        "per_window_and_mass": per_group,
    }
    if constant_aggregate_rmse is not None:
        result["rmse_improvement_over_constant"] = 1.0 - aggregate_rmse / max(
            constant_aggregate_rmse, 1.0e-12
        )
    return result


def fit_constant(samples: dict[str, Tensor]) -> float:
    target = samples["target"].numpy().astype(np.float64, copy=False)
    weights = equal_group_weights(group_indices(samples).numpy())
    return float(np.sum(weights * target))


def ridge_fit(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    selected: np.ndarray,
    strength: float,
) -> dict[str, Any]:
    fit_features = features[selected]
    mean = fit_features.mean(axis=0)
    scale = fit_features.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    standardized = (fit_features - mean) / scale
    design = np.column_stack((standardized, np.ones(len(standardized))))
    weights = equal_group_weights(groups[selected])
    weighted_design = design * np.sqrt(weights[:, None])
    weighted_target = target[selected] * np.sqrt(weights)
    penalty = np.sqrt(strength) * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.lstsq(
        np.vstack((weighted_design, penalty)),
        np.concatenate((weighted_target, np.zeros(design.shape[1]))),
        rcond=None,
    )[0]
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients[:-1],
        "intercept": float(coefficients[-1]),
        "strength": strength,
    }


def ridge_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    return (features - model["mean"]) / model["scale"] @ model["coefficients"] + model["intercept"]


def equal_group_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    selected: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(
                [
                    np.mean(
                        (
                            prediction[selected & (groups == group)]
                            - target[selected & (groups == group)]
                        )
                        ** 2
                    )
                    for group in range(6)
                ]
            )
        )
    )


def balanced_probe_split(
    pair: np.ndarray,
    *,
    total_pairs: int,
    fit_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    if total_pairs % 2 or fit_pairs % 2 or fit_pairs >= total_pairs:
        raise ValueError("probe split requires even fit and total pair counts")
    extreme_pairs = total_pairs // 2
    fit_per_mass_range = fit_pairs // 2
    fit_pair_ids = np.concatenate(
        (
            np.arange(fit_per_mass_range),
            np.arange(extreme_pairs, extreme_pairs + fit_per_mass_range),
        )
    )
    fit_mask = np.isin(pair, fit_pair_ids)
    return fit_mask, ~fit_mask


def select_and_fit_probe(
    features: Tensor,
    samples: dict[str, Tensor],
    *,
    fit_pairs: int,
    total_pairs: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    array = features.numpy().astype(np.float64, copy=False)
    target = samples["target"].numpy().astype(np.float64, copy=False)
    groups = group_indices(samples).numpy()
    pair = samples["pair"].numpy()
    fit_mask, validation_mask = balanced_probe_split(
        pair,
        total_pairs=total_pairs,
        fit_pairs=fit_pairs,
    )
    trials = []
    for strength in RIDGE_STRENGTHS:
        model = ridge_fit(array, target, groups, fit_mask, strength)
        prediction = ridge_predict(model, array)
        trials.append(
            {
                "strength": strength,
                "validation_equal_group_rmse": equal_group_rmse(
                    prediction, target, groups, validation_mask
                ),
            }
        )
    selected_strength = min(trials, key=lambda item: item["validation_equal_group_rmse"])[
        "strength"
    ]
    final = ridge_fit(
        array,
        target,
        groups,
        np.ones(len(array), dtype=bool),
        selected_strength,
    )
    return final, trials


def fit_native_readout(
    controller: ConnectomeController,
    samples: dict[str, Tensor],
    spec: ReadoutSpec,
    scales: dict[int, float],
    *,
    updates: int,
    learning_rate: float,
    maximum_magnitude: float,
    device: torch.device,
) -> tuple[Tensor, list[dict[str, float]]]:
    names = (
        "edge_features",
        "drive_without_selected",
        "previous_motor_state",
        "motor_alpha",
        "source_motor",
        "target",
        "window",
        "mass_half",
    )
    fitting = {name: samples[name].to(device) for name in names}
    groups = group_indices(fitting)
    weights = torch.zeros(len(groups), device=device)
    normalization = torch.zeros(len(groups), device=device)
    for group in range(6):
        selected = groups == group
        weights[selected] = 1.0 / (6.0 * int(selected.sum()))
        normalization[selected] = scales[group]
    baseline = controller.edge_magnitude[spec.edges].detach()
    magnitudes = torch.nn.Parameter(baseline.clone())
    optimizer = torch.optim.Adam((magnitudes,), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best = baseline.clone()
    for update in range(1, updates + 1):
        prediction = exact_residual(magnitudes, fitting, spec)
        loss = torch.sum(weights * ((prediction - fitting["target"]) / normalization).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            magnitudes.clamp_(0.0, maximum_magnitude)
            fitted_prediction = exact_residual(magnitudes, fitting, spec)
            fitted_loss = torch.sum(
                weights * ((fitted_prediction - fitting["target"]) / normalization).square()
            )
        value = float(fitted_loss)
        if value < best_loss:
            best_loss = value
            best = magnitudes.detach().clone()
        if update == 1 or update % 20 == 0:
            history.append(
                {
                    "update": update,
                    "normalized_equal_group_mse": value,
                    "coefficient_minimum": float(magnitudes.detach().min()),
                    "coefficient_maximum": float(magnitudes.detach().max()),
                }
            )
            print(
                json.dumps({"phase": "native_fit", **history[-1]}),
                flush=True,
            )
    return best, history


def model_passes(
    metrics: dict[str, Any],
    *,
    maximum_group_nrmse: float,
    minimum_constant_improvement: float,
) -> dict[str, Any]:
    checks = {
        "every_window_and_mass_nrmse_at_most_threshold": (
            metrics["maximum_group_nrmse"] <= maximum_group_nrmse
        ),
        "aggregate_rmse_improves_at_least_fifty_percent_over_constant": (
            metrics["rmse_improvement_over_constant"] >= minimum_constant_improvement
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def clustered_interval(values: Tensor, *, cluster_size: int = 2) -> dict[str, Any]:
    if values.ndim != 1 or len(values) % cluster_size:
        raise ValueError("values must contain complete adjacent geometry clusters")
    clusters = values.float().reshape(-1, cluster_size).mean(dim=1)
    mean = float(clusters.mean())
    standard_error = float(clusters.std(unbiased=True) / math.sqrt(len(clusters)))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "independent_geometry_clusters": len(clusters),
        "episodes_per_cluster": cluster_size,
    }


def replay_change_audit(
    live_prediction: np.ndarray,
    controlled_prediction: np.ndarray,
    samples: dict[str, Tensor],
) -> dict[str, Any]:
    target = samples["target"].numpy().astype(np.float64, copy=False)
    episodes = int(samples["episode"].max()) + 1
    prediction_change = (
        np.abs(controlled_prediction - live_prediction).reshape(-1, episodes).mean(0)
    )
    error_change = (
        (np.abs(controlled_prediction - target) - np.abs(live_prediction - target))
        .reshape(-1, episodes)
        .mean(0)
    )
    return {
        "absolute_prediction_change": clustered_interval(torch.from_numpy(prediction_change)),
        "absolute_error_increase": clustered_interval(torch.from_numpy(error_change)),
    }


def public_probe(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "strength": model["strength"],
        "feature_count": len(model["coefficients"]),
        "standardization_mean": model["mean"].tolist(),
        "standardization_scale": model["scale"].tolist(),
        "coefficients": model["coefficients"].tolist(),
        "intercept": model["intercept"],
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    controller, _checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    path_spec = make_path_spec(
        controller,
        args.graph,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=args.maximum_edge_magnitude,
    )
    path_nodes = torch.unique(
        torch.cat((controller.edge_pre[path_spec.edges], controller.edge_post[path_spec.edges]))
    )
    spec = make_readout_spec(controller)
    if (
        len(path_spec.edges),
        len(path_nodes),
        len(spec.edges),
        len(spec.return_sources),
        len(spec.motors),
    ) != (
        282,
        93,
        37,
        19,
        7,
    ):
        raise SystemExit("unexpected acceleration-path or throttle-return dimensions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    history_path = args.output_dir / "replay-histories.pt"
    readout_path = args.output_dir / "fitted-readouts.json"
    if any(path.exists() for path in (report_path, history_path, readout_path)):
        raise SystemExit("output directory contains a stale report, history, or readout")
    seed_everything(args.development_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    development = collect_history(
        controller,
        path_nodes,
        spec,
        pairs=args.development_pairs,
        seed=args.development_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seconds=args.seconds,
        takeover_seconds=args.takeover_seconds,
    )
    held_out = collect_history(
        controller,
        path_nodes,
        spec,
        pairs=args.held_out_pairs,
        seed=args.held_out_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seconds=args.seconds,
        takeover_seconds=args.takeover_seconds,
    )
    if (
        not development["source_forward_parity"]["passed"]
        or not held_out["source_forward_parity"]["passed"]
    ):
        raise RuntimeError("manual source forward reconstruction failed")
    torch.save(
        {
            "development": development,
            "held_out": held_out,
            "window_labels": WINDOW_LABELS,
            "claim": "training-only frozen sensory histories; not actor inputs",
        },
        history_path,
    )
    development_live = replay_history(
        controller,
        development,
        path_nodes,
        spec,
        acceleration_control="live",
        device=device,
    )
    held_replays = {
        control: replay_history(
            controller,
            held_out,
            path_nodes,
            spec,
            acceleration_control=control,
            device=device,
        )
        for control in ("live", "constant_1g", "pair_swapped")
    }
    live_replay_differences = {
        name: float((development_live[name] - development["samples"][name]).abs().max())
        for name in (
            "path_activity",
            "return_motor_features",
            "edge_features",
            "drive_without_selected",
            "previous_motor_state",
            "source_motor",
        )
    }
    live_replay_parity = {
        "maximum_absolute_differences": live_replay_differences,
        "passed": max(live_replay_differences.values()) <= 1.0e-6,
    }
    if not live_replay_parity["passed"]:
        raise RuntimeError(f"live sensory replay parity failed: {live_replay_parity}")
    baseline_magnitudes = controller.edge_magnitude[spec.edges].detach()
    zero_change_difference = float(
        exact_residual(
            baseline_magnitudes,
            {name: value.to(device) for name, value in development_live.items()},
            spec,
        )
        .abs()
        .max()
    )
    zero_change_parity = {
        "maximum_absolute_throttle_motor_difference": zero_change_difference,
        "threshold": 1.0e-6,
        "passed": zero_change_difference <= 1.0e-6,
    }
    if not zero_change_parity["passed"]:
        raise RuntimeError(f"exact readout zero-change parity failed: {zero_change_parity}")
    scales = target_scales(development_live, floor=args.normalization_floor)
    constant = fit_constant(development_live)
    path_probe, path_trials = select_and_fit_probe(
        development_live["path_activity"],
        development_live,
        fit_pairs=args.probe_fit_pairs,
        total_pairs=args.development_pairs,
    )
    return_probe, return_trials = select_and_fit_probe(
        development_live["return_motor_features"],
        development_live,
        fit_pairs=args.probe_fit_pairs,
        total_pairs=args.development_pairs,
    )
    fitted_magnitudes, native_history = fit_native_readout(
        controller,
        development_live,
        spec,
        scales,
        updates=args.native_fit_updates,
        learning_rate=args.native_learning_rate,
        maximum_magnitude=args.maximum_edge_magnitude,
        device=device,
    )
    live_held = held_replays["live"]
    live_features = {
        "constant": np.full(len(live_held["target"]), constant),
        "path_probe": ridge_predict(
            path_probe, live_held["path_activity"].numpy().astype(np.float64, copy=False)
        ),
        "return_motor_probe": ridge_predict(
            return_probe,
            live_held["return_motor_features"].numpy().astype(np.float64, copy=False),
        ),
        "native_readout": exact_residual(
            fitted_magnitudes,
            {name: value.to(device) for name, value in live_held.items()},
            spec,
        )
        .detach()
        .cpu()
        .numpy(),
    }
    constant_metrics = metric_summary(
        live_features["constant"],
        live_held,
        scales,
        constant_aggregate_rmse=None,
    )
    held_metrics = {"constant": constant_metrics}
    passes = {}
    for name in ("path_probe", "return_motor_probe", "native_readout"):
        held_metrics[name] = metric_summary(
            live_features[name],
            live_held,
            scales,
            constant_aggregate_rmse=constant_metrics["aggregate_equal_group_rmse"],
        )
        passes[name] = model_passes(
            held_metrics[name],
            maximum_group_nrmse=args.maximum_group_nrmse,
            minimum_constant_improvement=args.minimum_constant_improvement,
        )
        print(
            json.dumps(
                {
                    "phase": "held_out",
                    "model": name,
                    "aggregate_rmse": held_metrics[name]["aggregate_equal_group_rmse"],
                    "maximum_group_nrmse": held_metrics[name]["maximum_group_nrmse"],
                    "improvement_over_constant": held_metrics[name][
                        "rmse_improvement_over_constant"
                    ],
                    "passed": passes[name]["passed"],
                }
            ),
            flush=True,
        )
    replay_audits = {}
    for control in ("constant_1g", "pair_swapped"):
        controlled = held_replays[control]
        controlled_predictions = {
            "constant": np.full(len(controlled["target"]), constant),
            "path_probe": ridge_predict(
                path_probe,
                controlled["path_activity"].numpy().astype(np.float64, copy=False),
            ),
            "return_motor_probe": ridge_predict(
                return_probe,
                controlled["return_motor_features"].numpy().astype(np.float64, copy=False),
            ),
            "native_readout": exact_residual(
                fitted_magnitudes,
                {name: value.to(device) for name, value in controlled.items()},
                spec,
            )
            .detach()
            .cpu()
            .numpy(),
        }
        replay_audits[control] = {
            name: {
                "metrics_against_unchanged_live_labels": metric_summary(
                    prediction,
                    controlled,
                    scales,
                    constant_aggregate_rmse=constant_metrics["aggregate_equal_group_rmse"],
                ),
                "paired_change_from_live": replay_change_audit(
                    live_features[name], prediction, controlled
                ),
            }
            for name, prediction in controlled_predictions.items()
        }
    any_unconstrained_probe_passed = bool(
        passes["path_probe"]["passed"] or passes["return_motor_probe"]["passed"]
    )
    return_source_representation_passed = bool(passes["return_motor_probe"]["passed"])
    native_passed = bool(passes["native_readout"]["passed"])
    if native_passed:
        interpretation = "native_readout_sufficient_on_frozen_histories"
        next_step = "supervised short replay of the same 37 edges, then assisted-flight evaluation"
    elif return_source_representation_passed:
        interpretation = "return_source_signal_present_but_bounded_native_fit_failed"
        next_step = (
            "distinguish native readout constraints from bounded-optimizer failure; do not "
            "claim impossibility"
        )
    elif passes["path_probe"]["passed"]:
        interpretation = "path_signal_not_established_at_throttle_return_sources"
        next_step = "audit existing anatomical routing from path state to return sources"
    else:
        interpretation = "usable_action_representation_not_established"
        next_step = "do not widen ES or infer that native recurrence is fundamentally insufficient"
    readouts = {
        "constant_residual": constant,
        "path_probe": public_probe(path_probe),
        "return_motor_probe": public_probe(return_probe),
        "native_edge_indices": spec.edges.cpu().tolist(),
        "native_edge_magnitudes": fitted_magnitudes.detach().cpu().tolist(),
    }
    readout_path.write_text(json.dumps(readouts, indent=2, sort_keys=True) + "\n")
    report = {
        "method": "frozen-replay action-relevant acceleration routing audit",
        "claim_scope": (
            "Ridge probes, teacher labels, saved histories, acceleration controls, and fitted "
            "readouts are diagnostic only. The source controller is preserved; no actor input, "
            "state, parameter, checkpoint, or flight policy is changed or promoted."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "development_episodes": 2 * args.development_pairs,
            "held_out_episodes": 2 * args.held_out_pairs,
            "windows": list(WINDOW_LABELS),
            "ridge_strengths": list(RIDGE_STRENGTHS),
            "one_time_independent_readout_across_all_windows_and_masses": True,
            "held_out_evaluated_once_after_fit": True,
            "exact_native_fit": (
                "37 nonnegative fixed-sign magnitudes through one-step neural update "
                "and motor_drive"
            ),
            "target": (
                "reserve throttle motor drive minus source throttle motor drive on the "
                "same trajectory"
            ),
            "constant_baseline": (
                "one scalar, equal-weighted across three windows and two mass halves"
            ),
            "acceleration_replays_refitted": False,
            "acceleration_replays_change_flight_trajectory": False,
            "acceleration_replays_are_pass_gates": False,
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
        },
        "dimensions": {
            "acceleration_path_edges": len(path_spec.edges),
            "acceleration_path_nodes": len(path_nodes),
            "throttle_return_edges": len(spec.edges),
            "throttle_return_sources": len(spec.return_sources),
            "throttle_motor_nodes": len(spec.motors),
            "path_probe_features": len(path_nodes),
            "return_motor_probe_features": len(spec.return_sources) + len(spec.motors),
        },
        "source_forward_parity": {
            "development": development["source_forward_parity"],
            "held_out": held_out["source_forward_parity"],
        },
        "live_sensory_replay_parity": live_replay_parity,
        "zero_change_exact_readout_parity": zero_change_parity,
        "development_target_scales": {
            f"{WINDOW_LABELS[group // 2]}/{'light' if group % 2 == 0 else 'heavy'}": value
            for group, value in scales.items()
        },
        "probe_selection": {
            "fit_pairs": args.probe_fit_pairs,
            "validation_pairs": args.probe_validation_pairs,
            "path_probe_trials": path_trials,
            "return_motor_probe_trials": return_trials,
        },
        "native_fit_history": native_history,
        "held_out": held_metrics,
        "pass_gates": passes,
        "acceleration_replay_diagnostics": replay_audits,
        "classification": {
            "any_unconstrained_probe_passed": any_unconstrained_probe_passed,
            "return_source_representation_passed": return_source_representation_passed,
            "native_readout_passed": native_passed,
            "interpretation": interpretation,
            "next_step": next_step,
        },
        "replay_histories": stable_path(history_path),
        "replay_histories_sha256": file_sha256(history_path),
        "fitted_readouts": stable_path(readout_path),
        "fitted_readouts_sha256": file_sha256(readout_path),
        "source_preserved": True,
        "compiled_or_promoted": False,
        "candidate_checkpoint": None,
        "goal_passed": False,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "interpretation": interpretation,
                "any_unconstrained_probe_passed": any_unconstrained_probe_passed,
                "return_source_representation_passed": return_source_representation_passed,
                "native_readout_passed": native_passed,
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if native_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
