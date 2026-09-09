#!/usr/bin/env python3
"""Test whether early acceleration and stick histories identify randomized mass."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from train_gate import RETINAL_FLIP_X, file_sha256, initial_rollout, seed_everything  # noqa: E402

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)


def stable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "sensor-observability",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-episodes", type=int, default=4096)
    parser.add_argument("--test-episodes", type=int, default=4096)
    parser.add_argument("--train-seed", type=int, default=180_031)
    parser.add_argument("--test-seed", type=int, default=190_031)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50, 0.75, 1.00),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if min(args.train_episodes, args.test_episodes) <= 0:
        raise SystemExit("episode counts must be positive")
    if args.ridge <= 0.0:
        raise SystemExit("--ridge must be positive")
    if not args.horizons or min(args.horizons) < dt:
        raise SystemExit(f"horizons must be at least one physics interval ({dt:g}s)")


@torch.no_grad()
def collect_histories(
    controller: ConnectomeController,
    *,
    episodes: int,
    steps: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = initial_rollout(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    acceleration_history = []
    position_history = []
    for _ in range(steps):
        centered_z = ((state.specific_force[:, 2] - 9.81) / (0.25 * 9.81)).clamp(-2.0, 2.0)
        acceleration_history.append(
            torch.stack((centered_z.clamp_min(0.0), (-centered_z).clamp_min(0.0)), dim=-1)
        )
        throttle_position = stick_state.position[:, 3].clamp(-1.0, 1.0)
        position_history.append(
            torch.stack(
                ((throttle_position + 1.0) / 2.0, (1.0 - throttle_position) / 2.0),
                dim=-1,
            )
        )
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        motor, neural = controller(
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
        )
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    return (
        torch.stack(acceleration_history, dim=1),
        torch.stack(position_history, dim=1),
        normalized_mass,
    )


def ridge_probe(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    test_features: torch.Tensor,
    test_target: torch.Tensor,
    ridge: float,
) -> tuple[dict[str, float], torch.Tensor]:
    mean = train_features.mean(dim=0)
    scale = train_features.std(dim=0).clamp_min(1.0e-5)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    train = torch.cat((train, torch.ones_like(train[:, :1])), dim=1)
    test = torch.cat((test, torch.ones_like(test[:, :1])), dim=1)
    gram = train.T @ train / len(train)
    penalty = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * ridge
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + penalty, train.T @ train_target / len(train))
    prediction = test @ weights
    residual = prediction - test_target
    target_variance = (test_target - test_target.mean()).square().mean().clamp_min(1.0e-12)
    centered_prediction = prediction - prediction.mean()
    centered_target = test_target - test_target.mean()
    correlation = (centered_prediction * centered_target).mean() / (
        centered_prediction.square().mean().sqrt() * centered_target.square().mean().sqrt()
    ).clamp_min(1.0e-12)
    metrics = {
        "r2": float(1.0 - residual.square().mean() / target_variance),
        "rmse_normalized_mass": float(residual.square().mean().sqrt()),
        "pearson_correlation": float(correlation),
        "mass_sign_accuracy": float(((prediction >= 0.0) == (test_target >= 0.0)).float().mean()),
    }
    return metrics, prediction


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match evaluator")
    hover_config = HoverConfig(**checkpoint["hover_config"])
    gate_config = GateConfig(**checkpoint["gate_config"])
    validate_args(args, hover_config.dt)
    controller = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
    ).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()

    horizon_steps = {horizon: round(horizon / hover_config.dt) for horizon in args.horizons}
    maximum_steps = max(horizon_steps.values())
    started = perf_counter()
    train_accel, train_position, train_mass = collect_histories(
        controller,
        episodes=args.train_episodes,
        steps=maximum_steps,
        seed=args.train_seed,
        resolution=int(checkpoint["image_resolution"]),
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    test_accel, test_position, test_mass = collect_histories(
        controller,
        episodes=args.test_episodes,
        steps=maximum_steps,
        seed=args.test_seed,
        resolution=int(checkpoint["image_resolution"]),
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    permutation = torch.randperm(args.test_episodes, device=device)
    results: dict[str, dict[str, Any]] = {}
    for horizon in sorted(horizon_steps):
        steps = horizon_steps[horizon]
        modalities = {
            "acceleration_only": (
                train_accel[:, :steps].flatten(1),
                test_accel[:, :steps].flatten(1),
            ),
            "stick_position_only": (
                train_position[:, :steps].flatten(1),
                test_position[:, :steps].flatten(1),
            ),
            "acceleration_and_stick_position": (
                torch.cat((train_accel[:, :steps], train_position[:, :steps]), dim=-1).flatten(1),
                torch.cat((test_accel[:, :steps], test_position[:, :steps]), dim=-1).flatten(1),
            ),
        }
        horizon_result: dict[str, Any] = {}
        for name, (train_features, test_features) in modalities.items():
            metrics, prediction = ridge_probe(
                train_features,
                train_mass,
                test_features,
                test_mass,
                args.ridge,
            )
            shuffled_residual = prediction[permutation] - test_mass
            shuffled_r2 = 1.0 - shuffled_residual.square().mean() / (
                test_mass - test_mass.mean()
            ).square().mean().clamp_min(1.0e-12)
            metrics["episode_shuffled_prediction_r2"] = float(shuffled_r2)
            horizon_result[name] = metrics
        results[f"{horizon:g}"] = horizon_result
        print(json.dumps({"horizon_seconds": horizon, **horizon_result}), flush=True)

    report = {
        "experiment": "early sensor-history mass observability",
        "claim_scope": (
            "Diagnostic linear readout only. The readout is not part of the deployed actor; "
            "mass must still be learned and retained inside connectome dynamics."
        ),
        "counts_toward_direct_sensor_goal": False,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "train_seed": args.train_seed,
            "test_seed": args.test_seed,
            "horizons_seconds": sorted(horizon_steps),
            "ridge": args.ridge,
            "features": (
                "exact connectome input encodings: body-Z acceleration push/pull and "
                "completed-step throttle-position high/low"
            ),
            "separate_train_and_test_flights": True,
        },
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
