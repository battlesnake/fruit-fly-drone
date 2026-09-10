#!/usr/bin/env python3
"""Test existing throttle-motor biases in the wider legal native readout."""

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
from scipy.optimize import least_squares
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_throttle_readout_ceiling import (  # noqa: E402
    compact_flight,
    evaluate_assisted_flights,
    flight_progress,
)
from audit_gate_throttle_readout_constraint import (  # noqa: E402
    feature_arrays,
    fitting_weights,
    projected_gradient,
)
from audit_gate_throttle_routing import (  # noqa: E402
    ReadoutSpec,
    collect_history,
    make_readout_spec,
    metric_summary,
    model_passes,
    replay_history,
    target_scales,
)
from search_gate_acceleration_path_es import make_path_spec  # noqa: E402
from search_gate_motor_interface_es import load_controller, stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402


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
        "--ceiling-report",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "gate-throttle-readout-ceiling-diagnostic-v1" / "report.json"
        ),
    )
    parser.add_argument(
        "--ceiling-magnitudes",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-throttle-readout-ceiling-diagnostic-v1"
            / "selected-magnitudes.json"
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
        default=REPO_ROOT / "runs" / "gate" / "throttle-readout-bias-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--held-out-pairs", type=int, default=128)
    parser.add_argument("--flight-episodes", type=int, default=256)
    parser.add_argument("--maximum-evaluations", type=int, default=300)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=32.0)
    parser.add_argument("--maximum-bias-delta", type=float, default=2.0)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--maximum-group-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-constant-improvement", type=float, default=0.50)
    parser.add_argument("--projected-gradient-threshold", type=float, default=1.0e-6)
    parser.add_argument("--flight-seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--minimum-light-gain", type=float, default=0.10)
    parser.add_argument("--minimum-floor-gain", type=float, default=0.10)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.05)
    parser.add_argument("--held-out-seed", type=int, default=1_045_031)
    parser.add_argument("--flight-seed", type=int, default=1_046_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.routing_report,
        args.routing_readouts,
        args.ceiling_report,
        args.ceiling_magnitudes,
        args.replay_histories,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "held_out_pairs": (args.held_out_pairs, 128),
        "flight_episodes": (args.flight_episodes, 256),
        "maximum_evaluations": (args.maximum_evaluations, 300),
        "maximum_edge_magnitude": (args.maximum_edge_magnitude, 32.0),
        "maximum_bias_delta": (args.maximum_bias_delta, 2.0),
        "normalization_floor": (args.normalization_floor, 0.01),
        "maximum_group_nrmse": (args.maximum_group_nrmse, 0.25),
        "minimum_constant_improvement": (args.minimum_constant_improvement, 0.50),
        "projected_gradient_threshold": (args.projected_gradient_threshold, 1.0e-6),
        "flight_seconds": (args.flight_seconds, 12.0),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "minimum_light_gain": (args.minimum_light_gain, 0.10),
        "minimum_floor_gain": (args.minimum_floor_gain, 0.10),
        "maximum_stratum_drop": (args.maximum_stratum_drop, 0.05),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered bias-audit values changed: {', '.join(wrong)}")


def bias_prediction_and_jacobian(
    parameters: np.ndarray,
    arrays: dict[str, np.ndarray],
    spec: ReadoutSpec,
) -> tuple[np.ndarray, np.ndarray]:
    edge_count = len(spec.edges)
    magnitudes = parameters[:edge_count]
    bias_delta = parameters[edge_count:]
    if len(bias_delta) != len(spec.motors):
        raise ValueError("bias parameter count does not match throttle motor count")
    features = arrays["edge_features"]
    slots = spec.edge_slots.detach().cpu().numpy()
    motor_pool = spec.motor_pool.detach().cpu().numpy()
    pool_sizes = spec.pool_sizes.detach().cpu().numpy().astype(np.float64)
    drive = arrays["drive_without_selected"].copy() + bias_delta
    for edge, slot in enumerate(slots):
        drive[:, slot] += features[:, edge] * magnitudes[edge]
    tanh_drive = np.tanh(drive / 5.0)
    membrane_target = 5.0 * tanh_drive
    next_state = arrays["previous_motor_state"] + arrays["motor_alpha"] * (
        membrane_target - arrays["previous_motor_state"]
    )
    activity = 1.0 / (1.0 + np.exp(-next_state))
    pool_sign = np.where(motor_pool == 0, 1.0, -1.0)
    pool_factor = pool_sign / pool_sizes[motor_pool]
    prediction = np.sum(activity * pool_factor, axis=1) - arrays["source_motor"]
    drive_derivative = (
        activity * (1.0 - activity) * arrays["motor_alpha"] * (1.0 - tanh_drive**2) * pool_factor
    )
    edge_jacobian = drive_derivative[:, slots] * features
    return prediction, np.concatenate((edge_jacobian, drive_derivative), axis=1)


class BiasLeastSquaresProblem:
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        spec: ReadoutSpec,
        row_scale: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.spec = spec
        self.row_scale = row_scale
        self._parameters: np.ndarray | None = None
        self._residual: np.ndarray | None = None
        self._jacobian: np.ndarray | None = None

    def evaluate(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._parameters is None or not np.array_equal(parameters, self._parameters):
            prediction, jacobian = bias_prediction_and_jacobian(parameters, self.arrays, self.spec)
            self._parameters = parameters.copy()
            self._residual = self.row_scale * (prediction - self.arrays["target"])
            self._jacobian = self.row_scale[:, None] * jacobian
        assert self._residual is not None and self._jacobian is not None
        return self._residual, self._jacobian

    def residual(self, parameters: np.ndarray) -> np.ndarray:
        return self.evaluate(parameters)[0]

    def jacobian(self, parameters: np.ndarray) -> np.ndarray:
        return self.evaluate(parameters)[1]


def run_bias_fit(
    start_name: str,
    start: np.ndarray,
    arrays: dict[str, np.ndarray],
    samples: dict[str, Tensor],
    spec: ReadoutSpec,
    scales: dict[int, float],
    *,
    maximum_evaluations: int,
    maximum_edge_magnitude: float,
    maximum_bias_delta: float,
    projected_gradient_threshold: float,
) -> dict[str, Any]:
    edge_count = len(spec.edges)
    lower = np.concatenate((np.zeros(edge_count), np.full(len(spec.motors), -maximum_bias_delta)))
    upper = np.concatenate(
        (
            np.full(edge_count, maximum_edge_magnitude),
            np.full(len(spec.motors), maximum_bias_delta),
        )
    )
    problem = BiasLeastSquaresProblem(arrays, spec, fitting_weights(samples, scales))
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
        "edge_magnitudes": result.x[:edge_count].tolist(),
        "motor_bias_deltas": result.x[edge_count:].tolist(),
        "_parameters": result.x,
    }
    print(
        json.dumps(
            {
                "phase": "fit",
                "start": start_name,
                "loss": record["development_normalized_equal_group_mse"],
                "nfev": record["function_evaluations"],
                "projected_gradient": projected["maximum_absolute_projected_gradient"],
                "converged": converged,
            }
        ),
        flush=True,
    )
    return record


def public_fit(record: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in record.items() if not name.startswith("_")}


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    routing_report = json.loads(args.routing_report.read_text())
    routing_readouts = json.loads(args.routing_readouts.read_text())
    ceiling_report = json.loads(args.ceiling_report.read_text())
    ceiling_selection = json.loads(args.ceiling_magnitudes.read_text())
    prerequisite_checks = {
        "graph_hash_matches": ceiling_report.get("graph_sha256") == file_sha256(args.graph),
        "checkpoint_hash_matches": ceiling_report.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "routing_report_hash_matches": ceiling_report.get("routing_report_sha256")
        == file_sha256(args.routing_report),
        "routing_readout_hash_matches": routing_report.get("fitted_readouts_sha256")
        == file_sha256(args.routing_readouts),
        "history_hash_matches": ceiling_report.get("replay_histories_sha256")
        == file_sha256(args.replay_histories),
        "ceiling_magnitude_hash_matches": ceiling_report.get("selected_magnitudes_sha256")
        == file_sha256(args.ceiling_magnitudes),
        "ceiling_replay_failed": ceiling_report.get("readout_pass_gate", {}).get("passed") is False,
        "ceiling_flight_not_run": ceiling_report.get("assisted_flight") is None,
        "source_was_preserved": ceiling_report.get("source_preserved") is True,
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"bias audit prerequisite failed: {prerequisite_checks}")

    controller, _checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    path_spec = make_path_spec(
        controller,
        args.graph,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=8.0,
    )
    path_nodes = torch.unique(
        torch.cat((controller.edge_pre[path_spec.edges], controller.edge_post[path_spec.edges]))
    )
    spec = make_readout_spec(controller)
    if (len(spec.edges), len(spec.return_sources), len(spec.motors)) != (37, 19, 7):
        raise SystemExit("unexpected throttle readout dimensions")
    if routing_readouts.get("native_edge_indices") != spec.edges.detach().cpu().tolist():
        raise SystemExit("routing readout edge order does not match the source controller")
    if ceiling_selection.get("edge_indices") != spec.edges.detach().cpu().tolist():
        raise SystemExit("ceiling magnitude edge order does not match the source controller")

    source_magnitudes = (
        controller.edge_magnitude[spec.edges].detach().cpu().numpy().astype(np.float64)
    )
    ceiling_magnitudes = np.asarray(ceiling_selection["magnitudes"], dtype=np.float64)
    zero_bias_delta = np.zeros(len(spec.motors), dtype=np.float64)
    starts = (
        ("source", np.concatenate((source_magnitudes, zero_bias_delta))),
        ("selected_bound_32", np.concatenate((ceiling_magnitudes, zero_bias_delta))),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    selection_path = args.output_dir / "selected-parameters.json"
    if report_path.exists() or selection_path.exists():
        raise SystemExit("output directory contains a stale report or parameter selection")
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
    arrays = feature_arrays(development, controller, spec, arm="fixed_sign")
    fits = [
        run_bias_fit(
            start_name,
            start,
            arrays,
            development,
            spec,
            scales,
            maximum_evaluations=args.maximum_evaluations,
            maximum_edge_magnitude=args.maximum_edge_magnitude,
            maximum_bias_delta=args.maximum_bias_delta,
            projected_gradient_threshold=args.projected_gradient_threshold,
        )
        for start_name, start in starts
    ]
    selected_fit = min(fits, key=lambda item: item["development_normalized_equal_group_mse"])
    selected_parameters = np.asarray(selected_fit["_parameters"], dtype=np.float64)
    edge_count = len(spec.edges)
    selected_magnitudes = selected_parameters[:edge_count]
    selected_bias_deltas = selected_parameters[edge_count:]

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
    constant = float(routing_readouts["constant_residual"])
    constant_metrics = metric_summary(
        np.full(len(held_out["target"]), constant),
        held_out,
        scales,
        constant_aggregate_rmse=None,
    )
    held_arrays = feature_arrays(held_out, controller, spec, arm="fixed_sign")
    prediction, _ = bias_prediction_and_jacobian(selected_parameters, held_arrays, spec)
    selected_held_out = metric_summary(
        prediction,
        held_out,
        scales,
        constant_aggregate_rmse=constant_metrics["aggregate_equal_group_rmse"],
    )
    readout_pass_gate = model_passes(
        selected_held_out,
        maximum_group_nrmse=args.maximum_group_nrmse,
        minimum_constant_improvement=args.minimum_constant_improvement,
    )
    readout_passed = bool(readout_pass_gate["passed"])

    assisted_flight = None
    flight_passed = False
    if readout_passed:
        candidate = copy.deepcopy(controller).to(device)
        with torch.no_grad():
            candidate.edge_magnitude[spec.edges] = torch.from_numpy(selected_magnitudes).to(
                device=device, dtype=candidate.edge_magnitude.dtype
            )
            candidate.bias[spec.motors] += torch.from_numpy(selected_bias_deltas).to(
                device=device, dtype=candidate.bias.dtype
            )
        assisted_flight = evaluate_assisted_flights(
            controller,
            candidate,
            episodes=args.flight_episodes,
            seed=args.flight_seed,
            takeover_seconds=args.takeover_seconds,
            seconds=args.flight_seconds,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        flight_passed = flight_progress(
            assisted_flight["source"],
            assisted_flight["candidate"],
            minimum_light_gain=args.minimum_light_gain,
            minimum_floor_gain=args.minimum_floor_gain,
            maximum_drop=args.maximum_stratum_drop,
        )
        print(
            json.dumps(
                {
                    "phase": "assisted_flight",
                    "source": compact_flight(assisted_flight["source"]),
                    "candidate": compact_flight(assisted_flight["candidate"]),
                    "passed": flight_passed,
                }
            ),
            flush=True,
        )

    edge_lower_count = int(np.isclose(selected_magnitudes, 0.0).sum())
    edge_upper_count = int(np.isclose(selected_magnitudes, args.maximum_edge_magnitude).sum())
    bias_lower_count = int(np.isclose(selected_bias_deltas, -args.maximum_bias_delta).sum())
    bias_upper_count = int(np.isclose(selected_bias_deltas, args.maximum_bias_delta).sum())
    if not readout_passed:
        interpretation = "native_bias_readout_failed"
        next_step = "stop this frozen-history readout family"
    elif flight_passed:
        interpretation = "native_bias_readout_passed_replay_and_assisted_flight"
        next_step = "design a separate native coupled-flight test; no automatic merge"
    else:
        interpretation = "native_bias_readout_passed_replay_but_failed_assisted_flight"
        next_step = "stop this readout before any native merge"
    selection_path.write_text(
        json.dumps(
            {
                "edge_indices": spec.edges.detach().cpu().tolist(),
                "edge_magnitudes": selected_magnitudes.tolist(),
                "motor_indices": spec.motors.detach().cpu().tolist(),
                "source_motor_biases": controller.bias[spec.motors].detach().cpu().tolist(),
                "motor_bias_deltas": selected_bias_deltas.tolist(),
                "selected_start": selected_fit["start"],
                "edge_lower_bound_count": edge_lower_count,
                "edge_upper_bound_count": edge_upper_count,
                "bias_lower_bound_count": bias_lower_count,
                "bias_upper_bound_count": bias_upper_count,
                "compiled_or_promoted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "method": "native throttle-motor bias readout audit",
        "claim_scope": (
            "This diagnostic fits only the same 37 fixed-sign magnitudes and seven existing "
            "throttle-motor biases on frozen histories. The source remains unchanged. A fitted "
            "readout is not compiled, merged, promoted, or treated as a deployable controller."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "routing_report": stable_path(args.routing_report),
        "routing_report_sha256": file_sha256(args.routing_report),
        "ceiling_report": stable_path(args.ceiling_report),
        "ceiling_report_sha256": file_sha256(args.ceiling_report),
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
            "starts": ["source", "selected legal bound-32 fit"],
            "solver": "scipy.optimize.least_squares method=trf tr_solver=exact, FP64",
            "analytic_jacobian": True,
            "selection": "minimum development loss only",
            "held_out_evaluated_after_selection": True,
            "edge_signs_fixed": True,
            "edge_magnitude_bounds": [0.0, args.maximum_edge_magnitude],
            "motor_bias_delta_bounds": [
                -args.maximum_bias_delta,
                args.maximum_bias_delta,
            ],
            "all_other_parameters_frozen": True,
            "stop_readout_family_if_either_gate_fails": True,
            "assisted_flight_runs_only_after_replay_pass": True,
            "assisted_flight_axis_intervention": "reserve steering after 0.5 seconds",
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
        },
        "constant_baseline": constant_metrics,
        "fits": [public_fit(item) for item in fits],
        "selected_start": selected_fit["start"],
        "selected_fit_converged": selected_fit["converged_under_preregistered_rule"],
        "selected_held_out": selected_held_out,
        "readout_pass_gate": readout_pass_gate,
        "selected_edge_lower_bound_count": edge_lower_count,
        "selected_edge_upper_bound_count": edge_upper_count,
        "selected_bias_lower_bound_count": bias_lower_count,
        "selected_bias_upper_bound_count": bias_upper_count,
        "selected_parameters": stable_path(selection_path),
        "selected_parameters_sha256": file_sha256(selection_path),
        "assisted_flight": assisted_flight,
        "assisted_flight_passed": flight_passed,
        "classification": {"interpretation": interpretation, "next_step": next_step},
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
                "readout_passed": readout_passed,
                "assisted_flight_passed": flight_passed,
                "interpretation": interpretation,
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if readout_passed and flight_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
