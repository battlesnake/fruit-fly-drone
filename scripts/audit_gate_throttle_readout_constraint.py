#!/usr/bin/env python3
"""Separate bounded optimizer failure from the native throttle sign constraint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_throttle_routing import (  # noqa: E402
    ReadoutSpec,
    collect_history,
    group_indices,
    make_readout_spec,
    metric_summary,
    model_passes,
    replay_history,
    target_scales,
)
from search_gate_acceleration_path_es import make_path_spec  # noqa: E402
from search_gate_motor_interface_es import load_controller, stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402


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
        "--routing-report",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-throttle-routing-diagnostic-v1" / "report.json",
    )
    parser.add_argument(
        "--routing-readouts",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "gate-throttle-routing-diagnostic-v1" / "fitted-readouts.json"
        ),
    )
    parser.add_argument(
        "--replay-histories",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "throttle-routing-audit-v1" / "replay-histories.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "throttle-readout-constraint-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--held-out-pairs", type=int, default=128)
    parser.add_argument("--maximum-evaluations", type=int, default=300)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=8.0)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--maximum-group-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-constant-improvement", type=float, default=0.50)
    parser.add_argument("--projected-gradient-threshold", type=float, default=1.0e-6)
    parser.add_argument("--held-out-seed", type=int, default=1_042_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.routing_report,
        args.routing_readouts,
        args.replay_histories,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "held_out_pairs": (args.held_out_pairs, 128),
        "maximum_evaluations": (args.maximum_evaluations, 300),
        "maximum_edge_magnitude": (args.maximum_edge_magnitude, 8.0),
        "normalization_floor": (args.normalization_floor, 0.01),
        "maximum_group_nrmse": (args.maximum_group_nrmse, 0.25),
        "minimum_constant_improvement": (args.minimum_constant_improvement, 0.50),
        "projected_gradient_threshold": (args.projected_gradient_threshold, 1.0e-6),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered constraint-audit values changed: {', '.join(wrong)}")


def feature_arrays(
    samples: dict[str, Tensor],
    controller: ConnectomeController,
    spec: ReadoutSpec,
    *,
    arm: str,
) -> dict[str, np.ndarray]:
    if arm not in {"fixed_sign", "relaxed_sign"}:
        raise ValueError(f"unknown readout arm: {arm}")
    edge_features = samples["edge_features"].numpy().astype(np.float64, copy=False)
    if arm == "relaxed_sign":
        edge_sign = controller.edge_sign[spec.edges].detach().cpu().numpy().astype(np.float64)
        edge_features = edge_features * edge_sign
    return {
        "edge_features": edge_features,
        "drive_without_selected": samples["drive_without_selected"]
        .numpy()
        .astype(np.float64, copy=False),
        "previous_motor_state": samples["previous_motor_state"]
        .numpy()
        .astype(np.float64, copy=False),
        "motor_alpha": samples["motor_alpha"].numpy().astype(np.float64, copy=False),
        "source_motor": samples["source_motor"].numpy().astype(np.float64, copy=False),
        "target": samples["target"].numpy().astype(np.float64, copy=False),
    }


def exact_prediction_and_jacobian(
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    spec: ReadoutSpec,
) -> tuple[np.ndarray, np.ndarray]:
    features = arrays["edge_features"]
    slots = spec.edge_slots.detach().cpu().numpy()
    motor_pool = spec.motor_pool.detach().cpu().numpy()
    pool_sizes = spec.pool_sizes.detach().cpu().numpy().astype(np.float64)
    drive = arrays["drive_without_selected"].copy()
    for edge, slot in enumerate(slots):
        drive[:, slot] += features[:, edge] * weights[edge]
    membrane_target = 5.0 * np.tanh(drive / 5.0)
    next_state = arrays["previous_motor_state"] + arrays["motor_alpha"] * (
        membrane_target - arrays["previous_motor_state"]
    )
    activity = 1.0 / (1.0 + np.exp(-next_state))
    pool_sign = np.where(motor_pool == 0, 1.0, -1.0)
    pool_factor = pool_sign / pool_sizes[motor_pool]
    prediction = np.sum(activity * pool_factor, axis=1) - arrays["source_motor"]
    state_derivative = (
        activity
        * (1.0 - activity)
        * arrays["motor_alpha"]
        * (1.0 - np.tanh(drive / 5.0) ** 2)
        * pool_factor
    )
    jacobian = state_derivative[:, slots] * features
    return prediction, jacobian


def fitting_weights(
    samples: dict[str, Tensor],
    scales: dict[int, float],
) -> np.ndarray:
    groups = group_indices(samples).numpy()
    weights = np.zeros(len(groups), dtype=np.float64)
    for group in range(6):
        selected = groups == group
        weights[selected] = 1.0 / (6.0 * int(selected.sum()))
        weights[selected] = np.sqrt(weights[selected]) / scales[group]
    return weights


class LeastSquaresProblem:
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        spec: ReadoutSpec,
        row_scale: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.spec = spec
        self.row_scale = row_scale
        self._weights: np.ndarray | None = None
        self._residual: np.ndarray | None = None
        self._jacobian: np.ndarray | None = None

    def evaluate(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._weights is None or not np.array_equal(weights, self._weights):
            prediction, jacobian = exact_prediction_and_jacobian(weights, self.arrays, self.spec)
            self._weights = weights.copy()
            self._residual = self.row_scale * (prediction - self.arrays["target"])
            self._jacobian = self.row_scale[:, None] * jacobian
        assert self._residual is not None and self._jacobian is not None
        return self._residual, self._jacobian

    def residual(self, weights: np.ndarray) -> np.ndarray:
        return self.evaluate(weights)[0]

    def jacobian(self, weights: np.ndarray) -> np.ndarray:
        return self.evaluate(weights)[1]


def projected_gradient(
    weights: np.ndarray,
    residual: np.ndarray,
    jacobian: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    gradient = jacobian.T @ residual
    tolerance = 1.0e-8
    projected = gradient.copy()
    projected[(weights <= lower + tolerance) & (gradient > 0.0)] = 0.0
    projected[(weights >= upper - tolerance) & (gradient < 0.0)] = 0.0
    return {
        "maximum_absolute_raw_gradient": float(np.max(np.abs(gradient))),
        "maximum_absolute_projected_gradient": float(np.max(np.abs(projected))),
        "l2_projected_gradient": float(np.linalg.norm(projected)),
    }


def run_fit(
    arm: str,
    start_name: str,
    start: np.ndarray,
    arrays: dict[str, np.ndarray],
    samples: dict[str, Tensor],
    spec: ReadoutSpec,
    scales: dict[int, float],
    *,
    maximum_evaluations: int,
    maximum_magnitude: float,
    projected_gradient_threshold: float,
) -> dict[str, Any]:
    lower = np.full(len(start), 0.0 if arm == "fixed_sign" else -maximum_magnitude)
    upper = np.full(len(start), maximum_magnitude)
    problem = LeastSquaresProblem(arrays, spec, fitting_weights(samples, scales))
    result = least_squares(
        problem.residual,
        start.astype(np.float64, copy=True),
        jac=problem.jacobian,
        bounds=(lower, upper),
        method="trf",
        tr_solver="exact",
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=maximum_evaluations,
        verbose=0,
    )
    residual, jacobian = problem.evaluate(result.x)
    projected = projected_gradient(result.x, residual, jacobian, lower, upper)
    converged = bool(
        result.status > 0
        and projected["maximum_absolute_projected_gradient"] <= projected_gradient_threshold
    )
    record = {
        "arm": arm,
        "start": start_name,
        "development_normalized_equal_group_mse": float(np.sum(residual**2)),
        "cost": float(result.cost),
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "jacobian_evaluations": int(result.njev) if result.njev is not None else None,
        "scipy_first_order_optimality": float(result.optimality),
        "active_lower_bounds": int((result.active_mask == -1).sum()),
        "active_upper_bounds": int((result.active_mask == 1).sum()),
        "projected_gradient": projected,
        "converged_under_preregistered_rule": converged,
        "weights": result.x.tolist(),
        "prediction": exact_prediction_and_jacobian(result.x, arrays, spec)[0],
    }
    print(
        json.dumps(
            {
                "phase": "fit",
                "arm": arm,
                "start": start_name,
                "loss": record["development_normalized_equal_group_mse"],
                "nfev": record["function_evaluations"],
                "status": record["status"],
                "projected_gradient": projected["maximum_absolute_projected_gradient"],
                "converged": converged,
            }
        ),
        flush=True,
    )
    return record


def public_fit(record: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in record.items() if name != "prediction"}


def arm_summary(
    fits: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    held_samples: dict[str, Tensor],
    spec: ReadoutSpec,
    scales: dict[int, float],
    *,
    constant_aggregate_rmse: float,
    maximum_group_nrmse: float,
    minimum_constant_improvement: float,
) -> dict[str, Any]:
    selected = min(fits, key=lambda item: item["development_normalized_equal_group_mse"])
    weights = np.asarray(selected["weights"], dtype=np.float64)
    prediction, _ = exact_prediction_and_jacobian(weights, arrays, spec)
    metrics = metric_summary(
        prediction,
        held_samples,
        scales,
        constant_aggregate_rmse=constant_aggregate_rmse,
    )
    gate = model_passes(
        metrics,
        maximum_group_nrmse=maximum_group_nrmse,
        minimum_constant_improvement=minimum_constant_improvement,
    )
    return {
        "fits": [public_fit(item) for item in fits],
        "selected_start": selected["start"],
        "selection_rule": "minimum development normalized equal-group MSE",
        "selected_converged": selected["converged_under_preregistered_rule"],
        "held_out": metrics,
        "pass_gate": gate,
        "_selected_weights": weights,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    routing_report = json.loads(args.routing_report.read_text())
    routing_readouts = json.loads(args.routing_readouts.read_text())
    prerequisite_checks = {
        "graph_hash_matches": routing_report.get("graph_sha256") == file_sha256(args.graph),
        "checkpoint_hash_matches": routing_report.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "readout_hash_matches": routing_report.get("fitted_readouts_sha256")
        == file_sha256(args.routing_readouts),
        "history_hash_matches": routing_report.get("replay_histories_sha256")
        == file_sha256(args.replay_histories),
        "return_source_probe_passed": routing_report.get("classification", {}).get(
            "return_source_representation_passed"
        )
        is True,
        "native_readout_failed": routing_report.get("classification", {}).get(
            "native_readout_passed"
        )
        is False,
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"constraint audit prerequisite failed: {prerequisite_checks}")
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
    if (len(spec.edges), len(spec.return_sources), len(spec.motors)) != (37, 19, 7):
        raise SystemExit("unexpected throttle readout dimensions")
    if routing_readouts.get("native_edge_indices") != spec.edges.detach().cpu().tolist():
        raise SystemExit("routing readout edge order does not match the source controller")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    weights_path = args.output_dir / "selected-weights.json"
    if report_path.exists() or weights_path.exists():
        raise SystemExit("output directory contains a stale report or weight selection")
    seed_everything(args.held_out_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    saved = torch.load(args.replay_histories, map_location="cpu", weights_only=False)
    development_history = saved["development"]
    del saved
    development = replay_history(
        controller,
        development_history,
        path_nodes,
        spec,
        acceleration_control="live",
        device=device,
    )
    scales = target_scales(development, floor=args.normalization_floor)
    reported_scales = routing_report["development_target_scales"]
    reconstructed_scales = {
        f"{('0.50:0.75', '0.75:1.00', '1.00:1.50')[group // 2]}/"
        f"{'light' if group % 2 == 0 else 'heavy'}": value
        for group, value in scales.items()
    }
    if (
        max(
            abs(value - float(reported_scales[name]))
            for name, value in reconstructed_scales.items()
        )
        > 1.0e-12
    ):
        raise RuntimeError("development normalization does not reproduce the routing audit")
    constant = float(routing_readouts["constant_residual"])
    source_magnitudes = (
        controller.edge_magnitude[spec.edges].detach().cpu().numpy().astype(np.float64)
    )
    adam_magnitudes = np.asarray(routing_readouts["native_edge_magnitudes"], dtype=np.float64)
    edge_sign = controller.edge_sign[spec.edges].detach().cpu().numpy().astype(np.float64)
    fit_records: dict[str, list[dict[str, Any]]] = {}
    for arm, arrays, starts in (
        (
            "fixed_sign",
            feature_arrays(development, controller, spec, arm="fixed_sign"),
            {"source": source_magnitudes, "adam_update_200": adam_magnitudes},
        ),
        (
            "relaxed_sign",
            feature_arrays(development, controller, spec, arm="relaxed_sign"),
            {
                "source": edge_sign * source_magnitudes,
                "adam_update_200": edge_sign * adam_magnitudes,
            },
        ),
    ):
        fit_records[arm] = [
            run_fit(
                arm,
                name,
                start,
                arrays,
                development,
                spec,
                scales,
                maximum_evaluations=args.maximum_evaluations,
                maximum_magnitude=args.maximum_edge_magnitude,
                projected_gradient_threshold=args.projected_gradient_threshold,
            )
            for name, start in starts.items()
        ]
    held_history = collect_history(
        controller,
        path_nodes,
        spec,
        pairs=args.held_out_pairs,
        seed=args.held_out_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seconds=1.50,
        takeover_seconds=0.50,
    )
    held_out = replay_history(
        controller,
        held_history,
        path_nodes,
        spec,
        acceleration_control="live",
        device=device,
    )
    constant_prediction = np.full(len(held_out["target"]), constant)
    constant_metrics = metric_summary(
        constant_prediction,
        held_out,
        scales,
        constant_aggregate_rmse=None,
    )
    arms = {}
    for arm in ("fixed_sign", "relaxed_sign"):
        arms[arm] = arm_summary(
            fit_records[arm],
            feature_arrays(held_out, controller, spec, arm=arm),
            held_out,
            spec,
            scales,
            constant_aggregate_rmse=constant_metrics["aggregate_equal_group_rmse"],
            maximum_group_nrmse=args.maximum_group_nrmse,
            minimum_constant_improvement=args.minimum_constant_improvement,
        )
        print(
            json.dumps(
                {
                    "phase": "held_out",
                    "arm": arm,
                    "selected_start": arms[arm]["selected_start"],
                    "converged": arms[arm]["selected_converged"],
                    "aggregate_rmse": arms[arm]["held_out"]["aggregate_equal_group_rmse"],
                    "maximum_group_nrmse": arms[arm]["held_out"]["maximum_group_nrmse"],
                    "improvement_over_constant": arms[arm]["held_out"][
                        "rmse_improvement_over_constant"
                    ],
                    "passed": arms[arm]["pass_gate"]["passed"],
                }
            ),
            flush=True,
        )
    fixed_passed = bool(arms["fixed_sign"]["pass_gate"]["passed"])
    relaxed_passed = bool(arms["relaxed_sign"]["pass_gate"]["passed"])
    fixed_converged = bool(arms["fixed_sign"]["selected_converged"])
    if fixed_passed:
        interpretation = "previous_native_optimizer_budget_was_insufficient"
        next_step = "supervised short replay of the same 37 legal edges, then assisted flights"
    elif relaxed_passed and fixed_converged:
        interpretation = "evidence_for_fixed_sign_readout_restriction"
        next_step = (
            "audit alternative existing signed anatomical routing; do not deploy relaxed weights"
        )
    else:
        interpretation = "constraint_discriminator_inconclusive"
        next_step = "stop without claiming anatomical impossibility"
    selected_weights = {}
    for arm, result in arms.items():
        values = result.pop("_selected_weights")
        selected_weights[arm] = {
            "selected_start": result["selected_start"],
            "weights": values.tolist(),
            "reversed_original_sign_count": (
                int((values * edge_sign < 0.0).sum()) if arm == "relaxed_sign" else 0
            ),
            "deployable": False,
        }
    weights_path.write_text(json.dumps(selected_weights, indent=2, sort_keys=True) + "\n")
    report = {
        "method": "paired FP64 bounded trust-region throttle-readout constraint audit",
        "claim_scope": (
            "Both fitted arms are frozen-replay diagnostics. The relaxed-sign arm violates "
            "the native transmitter-sign constraint and can never be compiled or deployed. "
            "No source controller parameter, actor input, state, or checkpoint is changed."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "routing_report": stable_path(args.routing_report),
        "routing_report_sha256": file_sha256(args.routing_report),
        "routing_readouts": stable_path(args.routing_readouts),
        "routing_readouts_sha256": file_sha256(args.routing_readouts),
        "replay_histories": stable_path(args.replay_histories),
        "replay_histories_sha256": file_sha256(args.replay_histories),
        "prerequisite_checks": prerequisite_checks,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "development_pairs": 64,
            "development_source": "identical frozen histories from routing audit v1",
            "fresh_held_out_episodes": 2 * args.held_out_pairs,
            "windows": ["0.50:0.75", "0.75:1.00", "1.00:1.50"],
            "starts_per_arm": ["source", "routing-audit Adam update 200"],
            "solver": "scipy.optimize.least_squares method=trf tr_solver=exact, FP64",
            "analytic_jacobian": True,
            "selection": "minimum development loss only",
            "held_out_evaluated_after_selection": True,
            "fixed_sign_bounds": [0.0, args.maximum_edge_magnitude],
            "relaxed_effective_weight_bounds": [
                -args.maximum_edge_magnitude,
                args.maximum_edge_magnitude,
            ],
            "relaxed_sign_deployable": False,
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
        },
        "constant_baseline": constant_metrics,
        "arms": arms,
        "selected_weights": stable_path(weights_path),
        "selected_weights_sha256": file_sha256(weights_path),
        "classification": {
            "fixed_sign_passed": fixed_passed,
            "fixed_sign_converged": fixed_converged,
            "relaxed_sign_passed": relaxed_passed,
            "interpretation": interpretation,
            "next_step": next_step,
        },
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
                **report["classification"],
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if interpretation != "constraint_discriminator_inconclusive" else 2


if __name__ == "__main__":
    raise SystemExit(main())
