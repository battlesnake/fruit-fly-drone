#!/usr/bin/env python3
"""Fit one clock-free anatomical readout to retained recurrent launch state."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.optimize import lsq_linear
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_gate_acceleration_es import compact_metrics  # noqa: E402
from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    evaluate_gate,
    file_sha256,
    seed_everything,
)
from train_gate_acceleration_current_distillation import preactivation_drive  # noqa: E402
from train_gate_mass_current_distillation import (  # noqa: E402
    paired_launch,
    regression_metrics,
    stable_path,
    target_bias,
)
from train_gate_native_latent_motif import (  # noqa: E402
    matched_launch_offset,
    schedule_weight,
)

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-mass-oracle-v1" / "candidate.json",
    )
    parser.add_argument(
        "--encoder-report",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "native-latent-motif-readout-source" / "report.json",
    )
    parser.add_argument(
        "--selection-checkpoint-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "gate"
            / "native-latent-motif-readout-source"
            / "selection-checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint-updates",
        type=int,
        nargs="*",
        help="Explicit encoder updates; otherwise rank retained neutral-suffix ordering.",
    )
    parser.add_argument("--checkpoint-count", type=int, default=3)
    parser.add_argument("--minimum-update", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "native-readout-feasibility-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-episodes", type=int, default=512)
    parser.add_argument("--audit-episodes", type=int, default=512)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--prefix-seconds", type=float, default=1.50)
    parser.add_argument("--ramp-start-seconds", type=float, default=0.50)
    parser.add_argument("--established-seconds", type=float, default=0.75)
    parser.add_argument("--fit-stride", type=int, default=1)
    parser.add_argument("--launch-throttle-offset", type=float, default=0.002)
    parser.add_argument("--acceleration-noise-mg", type=float, default=0.10)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=8.0)
    parser.add_argument(
        "--fit-antagonist-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit one time-independent signed scalar shared by both throttle pools.",
    )
    parser.add_argument("--antagonist-bias-bound", type=float, default=0.20)
    parser.add_argument("--minimum-late-r2", type=float, default=0.90)
    parser.add_argument("--minimum-late-slope", type=float, default=0.80)
    parser.add_argument("--maximum-late-slope", type=float, default=1.20)
    parser.add_argument("--maximum-late-offset-error", type=float, default=0.01)
    parser.add_argument("--maximum-late-normalized-rmse", type=float, default=0.20)
    parser.add_argument("--maximum-late-current-rmse", type=float, default=0.05)
    parser.add_argument("--maximum-early-bias-rmse", type=float, default=0.02)
    parser.add_argument("--maximum-early-current-rmse", type=float, default=0.05)
    parser.add_argument("--maximum-absolute-bias", type=float, default=0.25)
    parser.add_argument("--fit-seed", type=int, default=520_031)
    parser.add_argument("--audit-seed", type=int, default=530_031)
    parser.add_argument("--final-seed", type=int, default=540_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (args.graph, args.source_checkpoint, args.calibration, args.encoder_report):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.checkpoint_count,
        args.minimum_update,
        args.fit_episodes,
        args.audit_episodes,
        args.final_episodes,
        args.prefix_seconds,
        args.ramp_start_seconds,
        args.established_seconds,
        args.fit_stride,
        args.maximum_edge_magnitude,
        args.antagonist_bias_bound,
        args.minimum_late_r2,
        args.minimum_late_slope,
        args.maximum_late_slope,
        args.maximum_late_offset_error,
        args.maximum_late_normalized_rmse,
        args.maximum_late_current_rmse,
        args.maximum_early_bias_rmse,
        args.maximum_early_current_rmse,
        args.maximum_absolute_bias,
    )
    if min(positive) <= 0:
        raise SystemExit("episode counts, times, thresholds, and bounds must be positive")
    if args.minimum_late_slope >= args.maximum_late_slope:
        raise SystemExit("late slope bounds are reversed")
    if not 0.0 < args.ramp_start_seconds < args.established_seconds < args.prefix_seconds:
        raise SystemExit("expected 0 < ramp start < establishment < prefix")
    if args.prefix_seconds < 1.5 or round(args.prefix_seconds / dt) < 3:
        raise SystemExit("the fixed audit requires a 1.5 second prefix")
    if args.acceleration_noise_mg < 0.0:
        raise SystemExit("acceleration noise must be nonnegative")
    if not 0.0 <= args.launch_throttle_offset <= 0.02:
        raise SystemExit("launch throttle offset must be in [0, 0.02]")
    for name in ("fit_episodes", "audit_episodes"):
        if getattr(args, name) % 2:
            raise SystemExit(f"--{name.replace('_', '-')} must be even")
    if args.final_episodes % 8:
        raise SystemExit("--final-episodes must be divisible by eight")


def late_ordering_score(entry: dict[str, Any]) -> tuple[float, float, float, int]:
    neutral = entry["held_out_neutral_after_establishment"]["horizons"]
    correlations = [neutral[key]["pearson_correlation"] for key in ("1", "1.5")]
    live = entry["held_out_live"]["horizons"]["0.75"]["pearson_correlation"]
    return min(correlations), sum(correlations), live, int(entry["update"])


def select_checkpoint_updates(
    report: dict[str, Any],
    checkpoint_dir: Path,
    *,
    explicit: list[int] | None,
    count: int,
    minimum_update: int,
) -> list[dict[str, Any]]:
    by_update = {int(entry["update"]): entry for entry in report["history"]}
    if explicit:
        updates = list(dict.fromkeys(explicit))
    else:
        eligible = [
            entry
            for entry in report["history"]
            if int(entry["update"]) >= minimum_update
            and all(
                entry["held_out_neutral_after_establishment"]["horizons"][key][
                    "pearson_correlation"
                ]
                is not None
                for key in ("1", "1.5")
            )
        ]
        updates = [
            int(entry["update"])
            for entry in sorted(eligible, key=late_ordering_score, reverse=True)[:count]
        ]
    if not updates:
        raise SystemExit("no eligible selection checkpoints")
    selected: list[dict[str, Any]] = []
    for update in updates:
        if update not in by_update:
            raise SystemExit(f"encoder report has no update {update}")
        path = checkpoint_dir / f"update-{update:04d}.pt"
        if not path.is_file():
            raise SystemExit(f"missing selection checkpoint: {path}")
        selected.append(
            {
                "update": update,
                "path_display": stable_path(path),
                "sha256": file_sha256(path),
                "ordering_score": list(late_ordering_score(by_update[update])),
            }
        )
    return selected


def throttle_readout_spec(controller: ConnectomeController) -> dict[str, Any]:
    positive_begin = int(controller.pool_offsets[6].item())
    positive_end = int(controller.pool_offsets[7].item())
    negative_end = int(controller.pool_offsets[8].item())
    motors = controller.pool_indices[positive_begin:negative_end]
    motor_sign = torch.cat(
        (
            torch.ones(positive_end - positive_begin, device=motors.device),
            -torch.ones(negative_end - positive_end, device=motors.device),
        )
    )
    incoming_mask = torch.isin(controller.edge_post, motors)
    edges = torch.nonzero(incoming_mask, as_tuple=False).flatten()
    motor_slot = {int(node): slot for slot, node in enumerate(motors.tolist())}
    slots = torch.tensor(
        [motor_slot[int(controller.edge_post[edge])] for edge in edges],
        device=motors.device,
    )
    reached = set(controller.edge_post[edges].tolist())
    if reached != set(motors.tolist()):
        raise SystemExit("direct anatomical return edges do not reach every throttle motor")
    source_nodes = torch.unique(controller.edge_pre[edges])
    motor_pool = torch.cat(
        (
            torch.zeros(positive_end - positive_begin, device=motors.device, dtype=torch.long),
            torch.ones(negative_end - positive_end, device=motors.device, dtype=torch.long),
        )
    )
    edge_pool = motor_pool[slots]
    pool_sizes = torch.tensor(
        (positive_end - positive_begin, negative_end - positive_end),
        device=motors.device,
    )
    return {
        "edges": edges,
        "slots": slots,
        "motors": motors,
        "motor_sign": motor_sign,
        "motor_pool": motor_pool,
        "edge_pool": edge_pool,
        "pool_sizes": pool_sizes,
        "edge_count": len(edges),
        "source_count": len(source_nodes),
        "source_body_ids": sorted(int(controller.node_ids[node]) for node in source_nodes),
        "edges_body_ids_and_signs": [
            [
                int(controller.node_ids[controller.edge_pre[edge]]),
                int(controller.node_ids[controller.edge_post[edge]]),
                int(controller.edge_sign[edge]),
            ]
            for edge in edges
        ],
    }


def pool_means(values: Tensor, spec: dict[str, Any]) -> Tensor:
    motor_pool: Tensor = spec["motor_pool"]
    pool_sizes: Tensor = spec["pool_sizes"]
    pooled = torch.zeros(values.shape[0], 2, device=values.device, dtype=values.dtype)
    pooled = pooled.index_add(1, motor_pool, values)
    return pooled / pool_sizes.to(dtype=values.dtype)


def window_name(
    time_seconds: float,
    *,
    ramp_start_seconds: float,
    established_seconds: float,
) -> str:
    if time_seconds <= 0.25:
        return "reset_0:0.25"
    if time_seconds <= ramp_start_seconds:
        return f"pre_ramp_0.25:{ramp_start_seconds:g}"
    if time_seconds < established_seconds:
        return f"ramp_{ramp_start_seconds:g}:{established_seconds:g}"
    if time_seconds <= 1.0:
        return f"established_{established_seconds:g}:1"
    return "established_1:1.5"


def condition_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "name": "live_clean",
            "neutral_after_seconds": None,
            "acceleration_noise_mg": 0.0,
        },
        {
            "name": "live_sensor_noise",
            "neutral_after_seconds": None,
            "acceleration_noise_mg": args.acceleration_noise_mg,
        },
        {
            "name": "neutral_after_establishment",
            "neutral_after_seconds": args.established_seconds,
            "acceleration_noise_mg": 0.0,
        },
    ]


@torch.no_grad()
def readout_rollout(
    controller: ConnectomeController,
    reference: ConnectomeController,
    spec: dict[str, Any],
    calibration: dict[str, Any],
    condition: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    prefix_seconds: float,
    ramp_start_seconds: float,
    established_seconds: float,
    stride: int,
    launch_throttle_offset: float,
    collect_linear_system: bool,
) -> dict[str, Any]:
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    reference_neural = reference.initial_state(episodes, device=device, dtype=torch.float32)
    pilot_neural = reference.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    desired_bias = target_bias(normalized_mass, calibration)
    edges: Tensor = spec["edges"]
    slots: Tensor = spec["slots"]
    motors: Tensor = spec["motors"]
    throttle_offset = matched_launch_offset(episodes, launch_throttle_offset, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 91_919)
    features: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    candidate_without_deltas: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    times: list[float] = []
    windows: list[str] = []
    conditions: list[str] = []
    step_count = round(prefix_seconds / hover_config.dt)
    neutral_step = (
        round(float(condition["neutral_after_seconds"]) / hover_config.dt)
        if condition["neutral_after_seconds"] is not None
        else None
    )
    for step in range(1, step_count + 1):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        sensed_image = image
        sensed_attitude = state.euler[:, :2]
        sensed_force = state.specific_force
        if neutral_step is not None and step > neutral_step:
            sensed_image = torch.zeros_like(image)
            sensed_attitude = torch.zeros_like(sensed_attitude)
            sensed_force = torch.zeros_like(sensed_force)
            sensed_force[:, 2] = 9.81
        elif condition["acceleration_noise_mg"]:
            sensed_force = sensed_force.clone()
            sensed_force[:, 2] += torch.randn(
                episodes,
                device=device,
                dtype=sensed_force.dtype,
                generator=generator,
            ) * (float(condition["acceleration_noise_mg"]) * 1.0e-3 * 9.81)
        candidate_drive = preactivation_drive(
            controller,
            sensed_image,
            sensed_attitude,
            neural,
            sensed_force,
        )
        reference_drive = preactivation_drive(
            reference,
            sensed_image,
            sensed_attitude,
            reference_neural,
            sensed_force,
        )
        time_seconds = step * hover_config.dt
        schedule = schedule_weight(
            torch.tensor(time_seconds, device=device),
            ramp_start_seconds=ramp_start_seconds,
            established_seconds=established_seconds,
        )
        target_pool = torch.stack((schedule * desired_bias, -schedule * desired_bias), dim=1)
        actual_delta = pool_means(
            candidate_drive[:, motors] - reference_drive[:, motors],
            spec,
        )
        if step % stride == 0:
            predictions.append(actual_delta.cpu().double().numpy())
            targets.append(target_pool.cpu().double().numpy())
            times.extend([time_seconds] * episodes)
            windows.extend(
                [
                    window_name(
                        time_seconds,
                        ramp_start_seconds=ramp_start_seconds,
                        established_seconds=established_seconds,
                    )
                ]
                * episodes
            )
            conditions.extend([str(condition["name"])] * episodes)
            if collect_linear_system:
                edge_features = (
                    torch.tanh(neural[:, controller.edge_pre[edges]]) * controller.edge_sign[edges]
                )
                pooled_edge_features = edge_features / spec["pool_sizes"][spec["edge_pool"]].to(
                    dtype=edge_features.dtype
                )
                selected_current = torch.zeros(
                    episodes,
                    len(motors),
                    device=device,
                    dtype=neural.dtype,
                ).index_add(1, slots, edge_features * controller.edge_magnitude[edges])
                candidate_without = pool_means(
                    candidate_drive[:, motors] - selected_current,
                    spec,
                )
                reference_pool = pool_means(reference_drive[:, motors], spec)
                target_total = reference_pool + target_pool
                features.append(pooled_edge_features.cpu().double().numpy())
                rhs.append((target_total - candidate_without).cpu().double().numpy())
                candidate_without_deltas.append(
                    (candidate_without - reference_pool).cpu().double().numpy()
                )
        _, neural = controller(
            sensed_image,
            sensed_attitude,
            neural,
            sensed_force,
        )
        _, reference_neural = reference(
            sensed_image,
            sensed_attitude,
            reference_neural,
            sensed_force,
        )
        pilot_motor, pilot_neural = reference(
            image,
            state.euler[:, :2],
            pilot_neural,
            state.specific_force,
        )
        pilot_motor = pilot_motor.clone()
        pilot_motor[:, 3] = (pilot_motor[:, 3] + throttle_offset).clamp(-1.0, 1.0)
        rc, stick_state = sticks(pilot_motor, stick_state)
        state = quad(rc, state, mass_scale)
    result: dict[str, Any] = {
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "times": np.asarray(times),
        "windows": np.asarray(windows),
        "conditions": np.asarray(conditions),
    }
    if collect_linear_system:
        result.update(
            {
                "features": np.concatenate(features),
                "rhs": np.concatenate(rhs),
                "candidate_without_delta": np.concatenate(candidate_without_deltas),
            }
        )
    return result


def concatenate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = records[0].keys()
    return {key: np.concatenate([record[key] for record in records]) for key in keys}


def group_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    predicted_bias = 0.5 * (prediction[:, 0] - prediction[:, 1])
    target_at_bias = 0.5 * (target[:, 0] - target[:, 1])
    regression = regression_metrics(
        torch.from_numpy(predicted_bias),
        torch.from_numpy(target_at_bias),
    )
    variance = float(np.mean((target_at_bias - target_at_bias.mean()) ** 2))
    return {
        **regression,
        "r2": (
            float(1.0 - np.mean((predicted_bias - target_at_bias) ** 2) / variance)
            if variance > 1.0e-12
            else None
        ),
        "bias_rmse": float(np.sqrt(np.mean((predicted_bias - target_at_bias) ** 2))),
        "normalized_bias_rmse": (
            float(np.sqrt(np.mean((predicted_bias - target_at_bias) ** 2) / variance))
            if variance > 1.0e-12
            else None
        ),
        "bias_offset_error": float(predicted_bias.mean() - target_at_bias.mean()),
        "maximum_absolute_predicted_bias": float(np.max(np.abs(predicted_bias))),
        "per_pool_current_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "samples": len(prediction),
    }


def summarize_records(records: dict[str, Any]) -> dict[str, Any]:
    prediction = records["prediction"]
    target = records["target"]
    conditions = records["conditions"]
    windows = records["windows"]
    times = records["times"]
    result: dict[str, Any] = {
        "overall": group_metrics(prediction, target),
        "per_condition": {},
    }
    for condition in sorted(set(conditions.tolist())):
        condition_mask = conditions == condition
        per_window: dict[str, Any] = {}
        for window in sorted(set(windows[condition_mask].tolist())):
            selected = condition_mask & (windows == window)
            per_window[window] = group_metrics(
                prediction[selected],
                target[selected],
            )
        per_horizon: dict[str, Any] = {}
        for horizon in (0.10, 0.25, 0.50, 0.75, 1.00, 1.50):
            selected = condition_mask & np.isclose(times, horizon, atol=1.0e-6)
            if selected.any():
                per_horizon[f"{horizon:g}"] = group_metrics(
                    prediction[selected],
                    target[selected],
                )
        result["per_condition"][condition] = {
            "per_window": per_window,
            "per_horizon": per_horizon,
        }
    return result


def equal_group_weights(conditions: np.ndarray, windows: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(conditions), dtype=np.float64)
    for condition in sorted(set(conditions.tolist())):
        for window in sorted(set(windows[conditions == condition].tolist())):
            selected = (conditions == condition) & (windows == window)
            weights[selected] = 1.0 / np.sqrt(float(selected.sum()))
    if not np.all(weights > 0.0):
        raise RuntimeError("an equal-weight fit group was not assigned")
    return weights


def fit_anatomical_returns(
    controller: ConnectomeController,
    spec: dict[str, Any],
    records: dict[str, Any],
    *,
    maximum_edge_magnitude: float,
    fit_antagonist_bias: bool,
    antagonist_bias_bound: float,
) -> dict[str, Any]:
    design = records["features"]
    rhs = records["rhs"]
    candidate_without = records["candidate_without_delta"]
    sample_weights = equal_group_weights(records["conditions"], records["windows"])
    edge_pool = spec["edge_pool"].cpu().numpy()
    edge_count = design.shape[1]
    column_count = edge_count + int(fit_antagonist_bias)
    sample_count = len(design)
    matrix = np.zeros((2 * sample_count, column_count), dtype=np.float64)
    for pool in range(2):
        rows = slice(pool * sample_count, (pool + 1) * sample_count)
        selected_edges = np.flatnonzero(edge_pool == pool)
        matrix[rows, selected_edges] = design[:, selected_edges]
        if fit_antagonist_bias:
            matrix[rows, -1] = 1.0 if pool == 0 else -1.0
    target = np.concatenate((rhs[:, 0], rhs[:, 1]))
    equation_weights = np.concatenate((sample_weights, sample_weights)) / np.sqrt(2.0)
    weighted_matrix = matrix * equation_weights[:, None]
    weighted_target = target * equation_weights
    lower = np.zeros(column_count, dtype=np.float64)
    upper = np.full(column_count, maximum_edge_magnitude, dtype=np.float64)
    if fit_antagonist_bias:
        lower[-1] = -antagonist_bias_bound
        upper[-1] = antagonist_bias_bound
    fit = lsq_linear(
        weighted_matrix,
        weighted_target,
        bounds=(lower, upper),
        tol=1.0e-12,
        lsmr_tol=1.0e-12,
    )
    fitted = fit.x[:edge_count]
    antagonist_bias = float(fit.x[-1]) if fit_antagonist_bias else 0.0
    singular_values = np.linalg.svd(weighted_matrix, compute_uv=False)
    condition_number = (
        float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1.0e-12 else None
    )
    with torch.no_grad():
        controller.edge_magnitude[spec["edges"]] = torch.from_numpy(fitted).to(
            device=controller.edge_magnitude.device,
            dtype=controller.edge_magnitude.dtype,
        )
        if fit_antagonist_bias:
            controller.bias[spec["motors"]] += antagonist_bias * spec["motor_sign"]
    linear_prediction = candidate_without.copy()
    for edge, pool in enumerate(edge_pool):
        linear_prediction[:, pool] += design[:, edge] * fitted[edge]
    linear_prediction[:, 0] += antagonist_bias
    linear_prediction[:, 1] -= antagonist_bias
    linear_records = {
        **records,
        "prediction": linear_prediction,
    }
    return {
        "edge_magnitudes": fitted.tolist(),
        "coefficient_minimum": float(fitted.min()),
        "coefficient_maximum": float(fitted.max()),
        "coefficients_at_lower_bound": int(np.isclose(fitted, 0.0, atol=1.0e-8).sum()),
        "coefficients_at_upper_bound": int(
            np.isclose(fitted, maximum_edge_magnitude, atol=1.0e-6).sum()
        ),
        "antagonist_bias": antagonist_bias,
        "antagonist_bias_bound": antagonist_bias_bound if fit_antagonist_bias else None,
        "antagonist_bias_at_bound": bool(
            fit_antagonist_bias
            and np.isclose(abs(antagonist_bias), antagonist_bias_bound, atol=1.0e-6)
        ),
        "pool_edge_counts": {
            "throttle_positive": int((edge_pool == 0).sum()),
            "throttle_negative": int((edge_pool == 1).sum()),
        },
        "condition_number": condition_number,
        "cost": float(fit.cost),
        "optimality": float(fit.optimality),
        "status": int(fit.status),
        "linearized_training_replay": summarize_records(linear_records),
    }


def readout_criteria(
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for condition, condition_metrics in summary["per_condition"].items():
        early = condition_metrics["per_window"]["reset_0:0.25"]
        checks.extend(
            (
                {
                    "name": f"{condition}:reset_bias_rmse",
                    "value": early["bias_rmse"],
                    "operator": "<=",
                    "threshold": args.maximum_early_bias_rmse,
                    "passed": early["bias_rmse"] <= args.maximum_early_bias_rmse,
                },
                {
                    "name": f"{condition}:reset_current_rmse",
                    "value": early["per_pool_current_rmse"],
                    "operator": "<=",
                    "threshold": args.maximum_early_current_rmse,
                    "passed": early["per_pool_current_rmse"] <= args.maximum_early_current_rmse,
                },
            )
        )
        for horizon in ("0.75", "1", "1.5"):
            metrics = condition_metrics["per_horizon"][horizon]
            checks.extend(
                (
                    {
                        "name": f"{condition}:{horizon}:r2",
                        "value": metrics["r2"],
                        "operator": ">=",
                        "threshold": args.minimum_late_r2,
                        "passed": metrics["r2"] is not None
                        and metrics["r2"] >= args.minimum_late_r2,
                    },
                    {
                        "name": f"{condition}:{horizon}:slope_lower",
                        "value": metrics["signed_slope"],
                        "operator": ">=",
                        "threshold": args.minimum_late_slope,
                        "passed": metrics["signed_slope"] >= args.minimum_late_slope,
                    },
                    {
                        "name": f"{condition}:{horizon}:slope_upper",
                        "value": metrics["signed_slope"],
                        "operator": "<=",
                        "threshold": args.maximum_late_slope,
                        "passed": metrics["signed_slope"] <= args.maximum_late_slope,
                    },
                    {
                        "name": f"{condition}:{horizon}:offset_error",
                        "value": abs(metrics["bias_offset_error"]),
                        "operator": "<=",
                        "threshold": args.maximum_late_offset_error,
                        "passed": abs(metrics["bias_offset_error"])
                        <= args.maximum_late_offset_error,
                    },
                    {
                        "name": f"{condition}:{horizon}:normalized_rmse",
                        "value": metrics["normalized_bias_rmse"],
                        "operator": "<=",
                        "threshold": args.maximum_late_normalized_rmse,
                        "passed": metrics["normalized_bias_rmse"] is not None
                        and metrics["normalized_bias_rmse"] <= args.maximum_late_normalized_rmse,
                    },
                    {
                        "name": f"{condition}:{horizon}:current_rmse",
                        "value": metrics["per_pool_current_rmse"],
                        "operator": "<=",
                        "threshold": args.maximum_late_current_rmse,
                        "passed": metrics["per_pool_current_rmse"]
                        <= args.maximum_late_current_rmse,
                    },
                    {
                        "name": f"{condition}:{horizon}:absolute_bias",
                        "value": metrics["maximum_absolute_predicted_bias"],
                        "operator": "<=",
                        "threshold": args.maximum_absolute_bias,
                        "passed": metrics["maximum_absolute_predicted_bias"]
                        <= args.maximum_absolute_bias,
                    },
                )
            )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [check for check in checks if not check["passed"]],
    }


def save_candidate(
    path: Path,
    checkpoint: dict[str, Any],
    controller: ConnectomeController,
    readout: dict[str, Any],
) -> None:
    result = copy.deepcopy(checkpoint)
    result["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    result["native_anatomical_readout"] = readout
    torch.save(result, path)


def main() -> int:
    args = parse_args()
    source_checkpoint_cpu = torch.load(
        args.source_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    hover_config = HoverConfig(**source_checkpoint_cpu["hover_config"])
    validate_args(args, hover_config.dt)
    if source_checkpoint_cpu["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("source checkpoint and graph hashes do not match")
    if bool(source_checkpoint_cpu.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("source checkpoint retinal orientation does not match evaluator")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("calibration is not the expected training-only oracle")
    encoder_report = json.loads(args.encoder_report.read_text())
    selected = select_checkpoint_updates(
        encoder_report,
        args.selection_checkpoint_dir,
        explicit=args.checkpoint_updates,
        count=args.checkpoint_count,
        minimum_update=args.minimum_update,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source_checkpoint = torch.load(
        args.source_checkpoint,
        map_location=device,
        weights_only=True,
    )
    gate_config = GateConfig(**source_checkpoint["gate_config"])
    resolution = int(source_checkpoint["image_resolution"])
    reference = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(source_checkpoint["retinal_receptive_field"]),
    ).to(device)
    reference.load_state_dict(source_checkpoint["controller"])
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    conditions = condition_specs(args)
    results: list[dict[str, Any]] = []
    passing: list[
        tuple[tuple[float, float], int, ConnectomeController, dict[str, Any], dict[str, Any]]
    ] = []
    for checkpoint_index, selection in enumerate(selected):
        checkpoint_path = args.selection_checkpoint_dir / f"update-{selection['update']:04d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["graph_sha256"] != file_sha256(args.graph):
            raise SystemExit(f"encoder checkpoint graph mismatch: {checkpoint_path}")
        controller = ConnectomeController(
            args.graph,
            neural_dt=hover_config.dt,
            retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
        ).to(device)
        controller.load_state_dict(checkpoint["controller"])
        controller.eval()
        for parameter in controller.parameters():
            parameter.requires_grad_(False)
        spec = throttle_readout_spec(controller)
        initial_edges = controller.edge_magnitude.detach().clone()
        initial_bias = controller.bias.detach().clone()
        fit_records = concatenate_records(
            [
                readout_rollout(
                    controller,
                    reference,
                    spec,
                    calibration,
                    condition,
                    episodes=args.fit_episodes,
                    seed=args.fit_seed,
                    resolution=resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=gate_config,
                    prefix_seconds=args.prefix_seconds,
                    ramp_start_seconds=args.ramp_start_seconds,
                    established_seconds=args.established_seconds,
                    stride=args.fit_stride,
                    launch_throttle_offset=args.launch_throttle_offset,
                    collect_linear_system=True,
                )
                for condition in conditions
            ]
        )
        fit = fit_anatomical_returns(
            controller,
            spec,
            fit_records,
            maximum_edge_magnitude=args.maximum_edge_magnitude,
            fit_antagonist_bias=args.fit_antagonist_bias,
            antagonist_bias_bound=args.antagonist_bias_bound,
        )
        audit_records = concatenate_records(
            [
                readout_rollout(
                    controller,
                    reference,
                    spec,
                    calibration,
                    condition,
                    episodes=args.audit_episodes,
                    seed=args.audit_seed,
                    resolution=resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=gate_config,
                    prefix_seconds=args.prefix_seconds,
                    ramp_start_seconds=args.ramp_start_seconds,
                    established_seconds=args.established_seconds,
                    stride=1,
                    launch_throttle_offset=args.launch_throttle_offset,
                    collect_linear_system=False,
                )
                for condition in conditions
            ]
        )
        audit = summarize_records(audit_records)
        criteria = readout_criteria(audit, args)
        return_mask = torch.zeros_like(initial_edges, dtype=torch.bool)
        return_mask[spec["edges"]] = True
        drift = float(
            (controller.edge_magnitude[~return_mask] - initial_edges[~return_mask]).abs().max()
        )
        motor_mask = torch.zeros_like(initial_bias, dtype=torch.bool)
        motor_mask[spec["motors"]] = True
        nonmotor_bias_drift = float(
            (controller.bias[~motor_mask] - initial_bias[~motor_mask]).abs().max()
        )
        result = {
            **selection,
            "readout": {
                "edge_count": spec["edge_count"],
                "source_count": spec["source_count"],
                "source_body_ids": spec["source_body_ids"],
                "edges_body_ids_and_signs": spec["edges_body_ids_and_signs"],
                "positive_motor_neurons": int(spec["pool_sizes"][0]),
                "negative_motor_neurons": int(spec["pool_sizes"][1]),
            },
            "fit": fit,
            "held_out_replay": audit,
            "criteria": criteria,
            "nonreturn_edge_max_absolute_change": drift,
            "nonmotor_bias_max_absolute_change": nonmotor_bias_drift,
        }
        results.append(result)
        late_r2 = [
            metrics["r2"]
            for condition_metrics in audit["per_condition"].values()
            for horizon, metrics in condition_metrics["per_horizon"].items()
            if horizon in ("0.75", "1", "1.5") and metrics["r2"] is not None
        ]
        score = (min(late_r2), -audit["overall"]["per_pool_current_rmse"])
        if criteria["passed"]:
            passing.append((score, checkpoint_index, controller, checkpoint, spec))
        print(
            json.dumps(
                {
                    "phase": "checkpoint_readout",
                    "update": selection["update"],
                    "readout_passed": criteria["passed"],
                    "failed_checks": len(criteria["failed_checks"]),
                    "minimum_late_r2": min(late_r2),
                    "overall_current_rmse": audit["overall"]["per_pool_current_rmse"],
                }
            ),
            flush=True,
        )

    final: dict[str, Any] = {}
    baseline: dict[str, Any] | None = None
    selected_readout: dict[str, Any] | None = None
    candidate_path = args.output_dir / "candidate.pt"
    if passing:
        _, selected_index, controller, checkpoint, selected_spec = max(
            passing,
            key=lambda item: item[0],
        )
        selected_readout = results[selected_index]
        save_candidate(
            candidate_path,
            checkpoint,
            controller,
            {
                "kind": "fixed_clock_free_anatomical_throttle_readout",
                "encoder_update": selected_readout["update"],
                "return_edge_count": selected_readout["readout"]["edge_count"],
                "antagonist_bias": selected_readout["fit"]["antagonist_bias"],
                "fit_conditions": [condition["name"] for condition in conditions],
                "fit_seed": args.fit_seed,
                "audit_seed": args.audit_seed,
            },
        )
        final_specs = {
            "live_sensors": {},
            "constant_1g_acceleration": {"frozen_acceleration": True},
            "mass_rank_swapped_acceleration": {"swapped_acceleration": True},
            "frozen_first_frame": {"frozen_visual": True},
        }
        clean_radius = gate_config.inner_radius - gate_config.drone_radius
        for name, kwargs in final_specs.items():
            final[name] = evaluate_gate(
                controller,
                episodes=args.final_episodes,
                seconds=12.0,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                seed=args.final_seed,
                balanced_strata=True,
                **kwargs,
            )
            print(
                json.dumps(
                    {"phase": "flight", "name": name, **compact_metrics(final[name], clean_radius)}
                ),
                flush=True,
            )
        baseline = evaluate_gate(
            reference,
            episodes=args.final_episodes,
            seconds=12.0,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.final_seed,
            balanced_strata=True,
        )
        bias_only = copy.deepcopy(reference)
        with torch.no_grad():
            bias_only.bias[selected_spec["motors"]] += (
                selected_readout["fit"]["antagonist_bias"] * selected_spec["motor_sign"]
            )
        final["fixed_antagonist_bias_only"] = evaluate_gate(
            bias_only,
            episodes=args.final_episodes,
            seconds=12.0,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.final_seed,
            balanced_strata=True,
        )
    live_success = final.get("live_sensors", {}).get("success_rate", 0.0)
    causal_success = max(
        final.get("constant_1g_acceleration", {}).get("success_rate", 0.0),
        final.get("mass_rank_swapped_acceleration", {}).get("success_rate", 0.0),
    )
    goal_passed = bool(
        final.get("live_sensors", {}).get("goal_pass", False)
        and live_success >= causal_success + 0.05
    )
    report = {
        "experiment": "fixed clock-free anatomical readout feasibility",
        "claim_scope": (
            "Exact mass is used only to construct a training target. Runtime actor inputs "
            "remain FPV, roll/pitch, instantaneous body specific force, and persistent "
            "connectome state. The optional fitted offset is one time-independent "
            "antagonist bias applied inside the existing throttle motor pools. No external "
            "history feature, estimator, clock, or state machine is added."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": stable_path(args.source_checkpoint),
        "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "encoder_report": stable_path(args.encoder_report),
        "encoder_report_sha256": file_sha256(args.encoder_report),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "checkpoint_selection": (
                "explicit updates"
                if args.checkpoint_updates
                else "top neutral-suffix 1.0/1.5 second mass-ordering on encoder development set"
            ),
            "fit_conditions": conditions,
            "one_shared_readout_across_all_times": True,
            "equally_weighted_condition_time_windows": True,
            "physical_replay_actor": "frozen source controller",
            "matched_light_heavy_geometry": True,
            "matched_launch_command_perturbations": True,
        },
        "selected_encoder_checkpoints": selected,
        "checkpoint_results": results,
        "readout_feasible": bool(passing),
        "selected_readout": selected_readout,
        "candidate_checkpoint": stable_path(candidate_path) if passing else None,
        "baseline": baseline,
        "final": final,
        "flight_acceleration_causality": live_success >= causal_success + 0.05,
        "goal_passed": goal_passed,
        "elapsed_seconds": perf_counter() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "readout_feasible": bool(passing),
                "flight_evaluated": bool(final),
                "live_success_rate": live_success,
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
