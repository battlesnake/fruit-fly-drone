#!/usr/bin/env python3
"""Audit fixed accelerometer gains before changing connectome plasticity."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_sensor_observability import ridge_probe  # noqa: E402
from train_gate import RETINAL_FLIP_X, file_sha256, seed_everything  # noqa: E402
from train_gate_acceleration_oracle_distillation import delta_metrics  # noqa: E402
from train_gate_mass_current_distillation import (  # noqa: E402
    paired_launch,
    stable_path,
    target_bias,
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
        "--baseline-graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-v1" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
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
        default=REPO_ROOT / "runs" / "gate" / "acceleration-gain-audit",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=470_031)
    parser.add_argument("--gains", type=float, nargs="+", default=(1.0, 4.0, 16.0, 64.0))
    parser.add_argument("--noise-mg", type=float, default=0.0)
    parser.add_argument("--launch-throttle-offset", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--end-seconds", type=float, default=0.75)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (args.graph, args.baseline_graph, args.checkpoint, args.calibration):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.episodes <= 0 or args.episodes % 4:
        raise SystemExit("--episodes must be positive and divisible by four")
    if not args.gains or min(args.gains) <= 0.0:
        raise SystemExit("all gains must be positive")
    if args.noise_mg < 0.0 or args.ridge <= 0.0:
        raise SystemExit("noise must be nonnegative and ridge must be positive")
    if args.end_seconds < 0.75 or args.end_seconds < dt:
        raise SystemExit("--end-seconds must reach at least 0.75 seconds")
    if abs(args.launch_throttle_offset) > 0.10:
        raise SystemExit("launch throttle intervention must stay within +/-0.10")


def correlation(x: Tensor, y: Tensor) -> float | None:
    x = x.detach()
    y = y.detach()
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = centered_x.square().mean().sqrt() * centered_y.square().mean().sqrt()
    if float(denominator) <= 1.0e-12:
        return None
    return float((centered_x * centered_y).mean() / denominator)


def scaled_specific_force(
    specific_force: Tensor,
    *,
    gain: float,
    noise_mg: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    noise = torch.randn(
        specific_force.shape[0],
        device=specific_force.device,
        dtype=specific_force.dtype,
        generator=generator,
    ) * (noise_mg * 1.0e-3 * 9.81)
    raw_normalized = (specific_force[:, 2] - 9.81 + noise) / (0.25 * 9.81)
    sensed = specific_force.clone()
    sensed[:, 2] = 9.81 + gain * (specific_force[:, 2] - 9.81 + noise)
    return sensed, raw_normalized


def boundary_nodes(graph_path: Path, baseline_graph_path: Path) -> np.ndarray:
    with np.load(graph_path) as graph, np.load(baseline_graph_path) as baseline:
        new = ~np.isin(graph["node_ids"], baseline["node_ids"])
        crossing = new[graph["edge_pre"]] & ~new[graph["edge_post"]]
        return np.unique(graph["edge_post"][crossing])


def throttle_return_sources(graph_path: Path) -> np.ndarray:
    with np.load(graph_path) as graph:
        offsets = graph["output_pool_offsets"]
        indices = graph["output_pool_indices"]
        throttle_nodes = indices[offsets[6] : offsets[8]]
        return np.unique(graph["edge_pre"][np.isin(graph["edge_post"], throttle_nodes)])


def boundary_probe(
    state: Tensor,
    mass: Tensor,
    nodes: Tensor,
    ridge: float,
) -> dict[str, float]:
    half = len(state) // 2
    pair = torch.arange(half, device=state.device)
    train_pair = pair.remainder(2) == 0
    train_mask = torch.cat((train_pair, train_pair))
    test_mask = ~train_mask
    metrics, _ = ridge_probe(
        state[train_mask][:, nodes],
        mass[train_mask],
        state[test_mask][:, nodes],
        mass[test_mask],
        ridge,
    )
    centered_state = state[:, nodes] - state[:, nodes].mean(dim=0)
    centered_mass = mass - mass.mean()
    per_node = (centered_state * centered_mass[:, None]).mean(dim=0) / (
        centered_state.square().mean(dim=0).sqrt() * centered_mass.square().mean().sqrt()
    ).clamp_min(1.0e-12)
    return {
        **metrics,
        "maximum_absolute_single_node_mass_correlation": float(per_node.abs().max()),
    }


def neutral_retention(
    candidate: ConnectomeController,
    reference: ConnectomeController,
    candidate_state: Tensor,
    reference_state: Tensor,
    mass: Tensor,
    desired_bias: Tensor,
    *,
    resolution: int,
    dt: float,
    cutoff_seconds: float = 0.50,
    end_seconds: float = 1.00,
) -> dict[str, Any]:
    batch = len(mass)
    device = mass.device
    image = torch.zeros(batch, resolution, resolution, device=device)
    attitude = torch.zeros(batch, 2, device=device)
    specific_force = torch.zeros(batch, 3, device=device)
    specific_force[:, 2] = 9.81
    results: dict[str, Any] = {}
    begin = round(cutoff_seconds / dt)
    end = round(end_seconds / dt)
    for step in range(begin, end):
        candidate_motor, candidate_state = candidate(
            image,
            attitude,
            candidate_state,
            specific_force,
        )
        reference_motor, reference_state = reference(
            image,
            attitude,
            reference_state,
            specific_force,
        )
        completed = step + 1
        if completed not in (round(0.75 / dt), end):
            continue
        prediction = candidate_motor[:, 3] - reference_motor[:, 3]
        results[f"{completed * dt:g}"] = {
            **delta_metrics(prediction, desired_bias),
            "prediction_mass_correlation": correlation(prediction, mass),
        }
    return results


def run_gain(
    source: dict[str, Any],
    graph_path: Path,
    baseline_graph_path: Path,
    calibration: dict[str, Any],
    *,
    gain: float,
    episodes: int,
    seed: int,
    noise_mg: float,
    launch_throttle_offset: float,
    ridge: float,
    end_seconds: float,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    seed_everything(seed)
    reference = ConnectomeController(
        graph_path,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(source["retinal_receptive_field"]),
    ).to(device)
    reference.load_state_dict(source["controller"])
    reference.eval()
    candidate = copy.deepcopy(reference)
    candidate.eval()
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    desired_bias = target_bias(normalized_mass, calibration)
    candidate_state = candidate.initial_state(episodes, device=device, dtype=torch.float32)
    reference_state = reference.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    node_indices = torch.from_numpy(boundary_nodes(graph_path, baseline_graph_path)).to(device)
    return_source_indices = torch.from_numpy(throttle_return_sources(graph_path)).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + round(gain * 1000) + round(noise_mg * 1000))
    horizon_steps = {round(value / hover_config.dt): value for value in (0.25, 0.50, 0.75)}
    steps = round(end_seconds / hover_config.dt)
    results: dict[str, Any] = {}
    cutoff_candidate_state = None
    cutoff_reference_state = None
    for step in range(steps + 1):
        if step in horizon_steps:
            q = (state.specific_force[:, 2] - 9.81) / (0.25 * 9.81)
            tilt_z = 9.81 * torch.cos(state.euler[:, 0]) * torch.cos(state.euler[:, 1])
            tilt_q = (tilt_z - 9.81) / (0.25 * 9.81)
            residual_q = (state.specific_force[:, 2] - tilt_z) / (0.25 * 9.81)
            candidate_motor = candidate.motor_drive(candidate_state)
            reference_motor = reference.motor_drive(reference_state)
            motor_delta = candidate_motor[:, 3] - reference_motor[:, 3]
            results[f"{horizon_steps[step]:g}"] = {
                "raw_input": {
                    "mass_correlation": correlation(q, normalized_mass),
                    "standard_deviation": float(q.std()),
                    "tilt_expected_mass_correlation": correlation(tilt_q, normalized_mass),
                    "tilt_residual_mass_correlation": correlation(residual_q, normalized_mass),
                    "tilt_residual_standard_deviation": float(residual_q.std()),
                    "gain_clipping_rate": float((gain * q).abs().ge(2.0).float().mean()),
                },
                "boundary_state": boundary_probe(
                    candidate_state,
                    normalized_mass,
                    node_indices,
                    ridge,
                ),
                "throttle_return_source_state": boundary_probe(
                    candidate_state,
                    normalized_mass,
                    return_source_indices,
                    ridge,
                ),
                "throttle_delta": {
                    **delta_metrics(motor_delta, desired_bias),
                    "mass_correlation": correlation(motor_delta, normalized_mass),
                },
            }
        if math.isclose(step * hover_config.dt, 0.50, abs_tol=hover_config.dt / 2):
            cutoff_candidate_state = candidate_state.clone()
            cutoff_reference_state = reference_state.clone()
        if step == steps:
            break
        image = render_annular_gate(
            state,
            gate,
            resolution=int(source["image_resolution"]),
            hover_config=hover_config,
            gate_config=gate_config,
        )
        sensed_force, _ = scaled_specific_force(
            state.specific_force,
            gain=gain,
            noise_mg=noise_mg,
            generator=generator,
        )
        _, candidate_state = candidate(
            image,
            state.euler[:, :2],
            candidate_state,
            sensed_force,
        )
        reference_motor, reference_state = reference(
            image,
            state.euler[:, :2],
            reference_state,
            state.specific_force,
        )
        rc, stick_state = sticks(reference_motor, stick_state)
        if launch_throttle_offset and step < round(0.50 / hover_config.dt):
            rc = rc.clone()
            rc[:, 3] = (rc[:, 3] + launch_throttle_offset).clamp(0.0, 1.0)
        state = quad(rc, state, mass_scale)
    if cutoff_candidate_state is None or cutoff_reference_state is None:
        raise RuntimeError("0.5-second state was not captured")
    results["prefix_cutoff_0.5"] = neutral_retention(
        candidate,
        reference,
        cutoff_candidate_state,
        cutoff_reference_state,
        normalized_mass,
        desired_bias,
        resolution=int(source["image_resolution"]),
        dt=hover_config.dt,
    )
    return {
        "gain": gain,
        "boundary_nodes": int(node_indices.numel()),
        "throttle_return_source_nodes": int(return_source_indices.numel()),
        "horizons": results,
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if source["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(source.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match evaluator")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("calibration is not the expected training-only oracle")
    hover_config = HoverConfig(**source["hover_config"])
    gate_config = GateConfig(**source["gate_config"])
    validate_args(args, hover_config.dt)
    started = perf_counter()
    results: dict[str, Any] = {}
    with torch.no_grad():
        for gain in args.gains:
            result = run_gain(
                source,
                args.graph,
                args.baseline_graph,
                calibration,
                gain=gain,
                episodes=args.episodes,
                seed=args.seed,
                noise_mg=args.noise_mg,
                launch_throttle_offset=args.launch_throttle_offset,
                ridge=args.ridge,
                end_seconds=args.end_seconds,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            results[f"{gain:g}"] = result
            print(json.dumps(result), flush=True)
    report = {
        "experiment": "fixed accelerometer input-gain audit",
        "claim_scope": (
            "All neural weights and time constants are frozen. Gain is applied to the "
            "instantaneous body-Z specific-force deviation before the existing +/-2 clamp; "
            "the maximum sensory drive is unchanged."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "baseline_graph": stable_path(args.baseline_graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "episodes": args.episodes,
        "seed": args.seed,
        "noise_mg": args.noise_mg,
        "launch_throttle_offset": args.launch_throttle_offset,
        "ridge": args.ridge,
        "results": results,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": stable_path(report_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
