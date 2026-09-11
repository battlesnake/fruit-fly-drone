#!/usr/bin/env python3
"""Search for a first visually guided annular-gate crossing by the full MaleCNS."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_responsibilities as responsibility  # noqa: E402
import search_gate_motor_interface_es as motor_es  # noqa: E402
import train_variable_height_hover as hover_train  # noqa: E402

from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    crossing_coordinates,
    gate_coordinates,
    render_annular_gate_rgb,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
)
from flydrone.visual_hover import CameraSpec  # noqa: E402

POLICY_HZ = 50
PHYSICS_HZ = 100


@dataclass(frozen=True)
class MirroredGateCases:
    state: QuadState
    sticks: StickState
    gate: AnnularGate
    mass_scale: Tensor
    side: Tensor
    pair: Tensor
    seed: int


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
        default=(REPO_ROOT / "runs/variable-height-hover/pragmatic-motor-es-001/controller.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/gate/pragmatic-full-native-motor-es-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--antithetic-directions", type=int, default=8)
    parser.add_argument("--training-pairs", type=int, default=4)
    parser.add_argument("--validation-pairs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--sigma-decay", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--bias-perturbation-scale", type=float, default=0.015)
    parser.add_argument("--log-gain-perturbation-scale", type=float, default=0.05)
    parser.add_argument("--maximum-bias-delta", type=float, default=0.15)
    parser.add_argument("--maximum-gain-ratio", type=float, default=3.0)
    parser.add_argument("--archive-size", type=int, default=8)
    parser.add_argument("--case-refresh-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=550_983)
    parser.add_argument("--validation-seed", type=int, default=560_983)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.generations,
        args.antithetic_directions,
        args.training_pairs,
        args.validation_pairs,
        args.seconds,
        args.sigma,
        args.sigma_decay,
        args.learning_rate,
        args.archive_size,
        args.case_refresh_generations,
    )
    if min(positive) <= 0:
        raise SystemExit("search sizes and scales must be positive")
    if PHYSICS_HZ % POLICY_HZ:
        raise SystemExit("physics rate must be divisible by policy rate")


def _clone_quad(state: QuadState) -> QuadState:
    return QuadState(*(value.clone() for value in state.as_tuple()))


def _clone_sticks(state: StickState) -> StickState:
    return StickState(
        state.joint_position.clone(),
        state.joint_velocity.clone(),
        state.position.clone(),
        state.velocity.clone(),
    )


def _repeat_quad(state: QuadState) -> QuadState:
    return QuadState(*(value.repeat_interleave(2, dim=0) for value in state.as_tuple()))


def _repeat_sticks(state: StickState) -> StickState:
    return StickState(
        state.joint_position.repeat_interleave(2, dim=0),
        state.joint_velocity.repeat_interleave(2, dim=0),
        state.position.repeat_interleave(2, dim=0),
        state.velocity.repeat_interleave(2, dim=0),
    )


def sample_mirrored_cases(
    pairs: int,
    *,
    seed: int,
    device: torch.device,
    hover_config: HoverConfig,
) -> MirroredGateCases:
    """Make paired resets where straight flight cannot clear either aperture."""

    responsibility.seed_everything(seed)
    initial_height = torch.empty(pairs, device=device).uniform_(1.02, 1.18)
    base_state, base_sticks = hover_train.nominal_initial_state(
        pairs,
        camera_height=initial_height,
        device=device,
        config=hover_config,
        attitude_degrees=3.0,
        rate_degrees_per_second=7.0,
        vertical_speed=0.04,
    )
    # Keep the two members of each pair physically identical. Only the gate is mirrored.
    state = _repeat_quad(base_state)
    sticks = _repeat_sticks(base_sticks)
    side = torch.tensor((-1.0, 1.0), device=device).repeat(pairs)
    pair = torch.arange(pairs, device=device).repeat_interleave(2)
    distance = torch.empty(pairs, device=device).uniform_(2.7, 3.3).repeat_interleave(2)
    lateral_magnitude = torch.empty(pairs, device=device).uniform_(0.61, 0.78).repeat_interleave(2)
    center = state.position.clone()
    center[:, 0] += distance
    center[:, 1] += side * lateral_magnitude
    center[:, 2] = 1.10
    bearing = torch.atan2(center[:, 1] - state.position[:, 1], center[:, 0] - state.position[:, 0])
    obliquity = (
        torch.empty(pairs, device=device)
        .uniform_(math.radians(2.0), math.radians(7.0))
        .repeat_interleave(2)
    )
    gate = AnnularGate(center=center, yaw=bearing + side * obliquity)
    mass_scale = torch.ones(2 * pairs, device=device)
    return MirroredGateCases(state, sticks, gate, mass_scale, side, pair, seed)


def case_manifest(cases: MirroredGateCases, gate_config: GateConfig) -> dict[str, Any]:
    displacement = cases.gate.center - cases.state.position
    lateral = displacement[:, 1].abs()
    return {
        "seed": cases.seed,
        "episodes": len(cases.side),
        "mirrored_pairs": int(cases.pair.max()) + 1,
        "distance_range_metres": [float(displacement[:, 0].min()), float(displacement[:, 0].max())],
        "lateral_offset_range_metres": [float(lateral.min()), float(lateral.max())],
        "gate_height_metres": float(cases.gate.center[:, 2].mean()),
        "initial_height_range_metres": [
            float(cases.state.position[:, 2].min()),
            float(cases.state.position[:, 2].max()),
        ],
        "clean_aperture_radius_metres": gate_config.inner_radius - gate_config.drone_radius,
        "straight_flight_cannot_clear_aperture": bool(
            (lateral > gate_config.inner_radius - gate_config.drone_radius).all()
        ),
        "nominal_mass_only": True,
    }


def apply_vector(
    controller: ConnectomeController,
    base_bias: Tensor,
    base_edge: Tensor,
    vector: Tensor,
    spec: motor_es.MotorInterfaceSpec,
) -> None:
    with torch.no_grad():
        controller.bias.copy_(base_bias)
        controller.edge_magnitude.copy_(base_edge)
        controller.bias[spec.bias_nodes] += vector[spec.bias_parameter]
        controller.edge_magnitude[spec.gain_edges] = (
            base_edge[spec.gain_edges] * torch.exp(vector[spec.gain_parameter])
        ).clamp(max=8.0)


def _masked_mean(values: Tensor, mask: Tensor) -> float:
    return float(values[mask].mean()) if bool(mask.any()) else 0.0


@torch.no_grad()
def evaluate_cases(
    controller: ConnectomeController,
    cases: MirroredGateCases,
    *,
    seconds: float,
    camera: CameraSpec,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    frozen_vision: bool = False,
    observation_warmup_steps: int = 0,
) -> dict[str, Any]:
    device = cases.mass_scale.device
    count = len(cases.mass_scale)
    quad = DifferentiableQuad(hover_config).to(device)
    stick_plant = ForelegStickPlant(hover_config).to(device)
    state = _clone_quad(cases.state)
    sticks = _clone_sticks(cases.sticks)
    neural = controller.initial_state(count, device=device, dtype=torch.float32)
    policy_steps = round(seconds * POLICY_HZ)
    physics_steps = PHYSICS_HZ // POLICY_HZ
    freeze_step = round(0.5 * POLICY_HZ)
    frozen_image: Tensor | None = None

    passed = torch.zeros(count, dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    missed = torch.zeros_like(passed)
    cleared = torch.zeros_like(passed)
    ground = torch.zeros_like(passed)
    valid = torch.ones_like(passed)
    crossing_radial = torch.full((count,), float("nan"), device=device)
    pass_step = torch.full((count,), -1, dtype=torch.long, device=device)
    maximum_tilt = torch.zeros(count, device=device)
    saturation_steps = torch.zeros(count, device=device)
    initial_signed, _, _ = gate_coordinates(state.position, cases.gate)
    maximum_progress = torch.zeros(count, device=device)
    maximum_guided_progress = torch.zeros(count, device=device)
    clean_radius = gate_config.inner_radius - gate_config.drone_radius

    if observation_warmup_steps:
        warmup_image = render_annular_gate_rgb(
            state,
            cases.gate,
            camera=camera,
            gate_config=gate_config,
        )
        for _ in range(observation_warmup_steps):
            _, neural = controller(warmup_image, state.euler[:, :2], neural)

    for policy_step in range(policy_steps):
        live_image = render_annular_gate_rgb(
            state,
            cases.gate,
            camera=camera,
            gate_config=gate_config,
        )
        if policy_step == freeze_step:
            frozen_image = live_image.clone()
        image = frozen_image if frozen_vision and frozen_image is not None else live_image
        motor, neural = controller(image, state.euler[:, :2], neural)
        saturation_steps += (sticks.position.abs() > 0.98).any(dim=1)
        for _ in range(physics_steps):
            rc, sticks = stick_plant(motor, sticks)
            previous_position = state.position
            state = quad(rc, state, cases.mass_scale)
            pass_now, collision_now, miss_now = classify_gate_crossing(
                previous_position,
                state.position,
                cases.gate,
                gate_config,
            )
            crossing_now = pass_now | collision_now | miss_now
            new_crossing = crossing_now & crossing_radial.isnan()
            if bool(new_crossing.any()):
                _, lateral, vertical = crossing_coordinates(
                    previous_position, state.position, cases.gate
                )
                radial = torch.sqrt(lateral.square() + vertical.square())
                crossing_radial[new_crossing] = radial[new_crossing]
            new_pass = pass_now & ~passed
            pass_step[new_pass] = policy_step + 1
            passed |= pass_now
            collision |= collision_now
            missed |= miss_now

        signed, lateral, vertical = gate_coordinates(state.position, cases.gate)
        progress = ((signed - initial_signed) / (-initial_signed).clamp_min(1.0e-6)).clamp(0.0, 1.0)
        radial = torch.sqrt(lateral.square() + vertical.square())
        centred = torch.exp(-0.5 * (radial / clean_radius).square())
        maximum_progress = torch.maximum(maximum_progress, progress)
        maximum_guided_progress = torch.maximum(maximum_guided_progress, progress * centred)
        cleared |= passed & (signed >= 0.30)
        ground |= state.position[:, 2] <= 0.03
        valid &= hover_train.state_is_valid(state)
        maximum_tilt = torch.maximum(
            maximum_tilt,
            torch.linalg.vector_norm(state.euler[:, :2], dim=1),
        )

    saturation_fraction = saturation_steps / policy_steps
    success = (
        passed
        & cleared
        & ~collision
        & ~missed
        & ~ground
        & valid
        & (maximum_tilt <= math.radians(45.0))
        & (saturation_fraction <= 0.40)
    )
    pair_success = success.reshape(-1, 2).all(dim=1)
    pair_pass = passed.reshape(-1, 2).all(dim=1)
    negative = cases.side < 0.0
    positive = ~negative
    side_success = (
        _masked_mean(success.float(), negative),
        _masked_mean(success.float(), positive),
    )
    reward = (
        12.0 * success.float()
        + 3.0 * passed.float()
        + 2.0 * maximum_progress
        + 4.0 * maximum_guided_progress
        - 4.0 * collision.float()
        - 3.0 * missed.float()
        - 5.0 * ground.float()
        - 3.0 * (~valid).float()
        - 0.25 * saturation_fraction
    )
    fitness = (
        float(reward.mean()) + 5.0 * float(pair_success.float().mean()) + 2.0 * min(side_success)
    )
    crossed = ~crossing_radial.isnan()
    pass_times = pass_step[passed].float() / POLICY_HZ
    return {
        "fitness": fitness,
        "observation_warmup_seconds": observation_warmup_steps / POLICY_HZ,
        "episodes": count,
        "mirrored_pairs": len(pair_success),
        "success_rate": float(success.float().mean()),
        "paired_success_rate": float(pair_success.float().mean()),
        "success_rate_negative_offset": side_success[0],
        "success_rate_positive_offset": side_success[1],
        "pass_rate": float(passed.float().mean()),
        "paired_pass_rate": float(pair_pass.float().mean()),
        "pass_rate_negative_offset": _masked_mean(passed.float(), negative),
        "pass_rate_positive_offset": _masked_mean(passed.float(), positive),
        "plane_crossing_rate": float(crossed.float().mean()),
        "ring_collision_rate": float(collision.float().mean()),
        "miss_rate": float(missed.float().mean()),
        "ground_contact_rate": float(ground.float().mean()),
        "invalid_rate": float((~valid).float().mean()),
        "maximum_progress_mean": float(maximum_progress.mean()),
        "guided_progress_mean": float(maximum_guided_progress.mean()),
        "maximum_tilt_mean_degrees": float(torch.rad2deg(maximum_tilt).mean()),
        "stick_saturation_fraction_mean": float(saturation_fraction.mean()),
        "crossing_radial_mean_metres": (
            float(crossing_radial[crossed].mean()) if bool(crossed.any()) else None
        ),
        "pass_time_mean_seconds": float(pass_times.mean()) if bool(passed.any()) else None,
    }


def load_controller(
    args: argparse.Namespace, device: torch.device
) -> tuple[ConnectomeController, dict[str, Any]]:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected_hash = responsibility.file_sha256(args.graph)
    if payload.get("graph_sha256") != expected_hash:
        raise SystemExit("checkpoint and graph hashes do not match")
    controller = ConnectomeController(args.graph, neural_dt=1.0 / POLICY_HZ).to(device)
    controller.load_state_dict(payload["controller"])
    controller.eval().requires_grad_(False)
    if controller.uses_accelerometer or controller.uses_proprioception:
        raise SystemExit("the full-network proof actor unexpectedly uses added sensor channels")
    return controller, payload


def evaluate_vector(
    controller: ConnectomeController,
    base_bias: Tensor,
    base_edge: Tensor,
    vector: Tensor,
    spec: motor_es.MotorInterfaceSpec,
    cases: MirroredGateCases,
    *,
    seconds: float,
    camera: CameraSpec,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    frozen_vision: bool = False,
) -> dict[str, Any]:
    apply_vector(controller, base_bias, base_edge, vector, spec)
    return evaluate_cases(
        controller,
        cases,
        seconds=seconds,
        camera=camera,
        hover_config=hover_config,
        gate_config=gate_config,
        frozen_vision=frozen_vision,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    controller, source_payload = load_controller(args, device)
    hover_config = HoverConfig()
    gate_config = GateConfig()
    camera = CameraSpec(width=320, height=200, horizontal_fov_degrees=125.0)
    spec = motor_es.motor_interface_spec(
        controller,
        bias_scale=args.bias_perturbation_scale,
        log_gain_scale=args.log_gain_perturbation_scale,
        maximum_bias_delta=args.maximum_bias_delta,
        maximum_gain_ratio=args.maximum_gain_ratio,
    )
    base_bias = controller.bias.detach().clone()
    base_edge = controller.edge_magnitude.detach().clone()
    center = torch.zeros(len(spec.labels), device=device)
    rng = np.random.default_rng(args.seed)
    archive: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    started = perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_cases = sample_mirrored_cases(
        args.training_pairs,
        seed=args.seed,
        device=device,
        hover_config=hover_config,
    )
    baseline = evaluate_vector(
        controller,
        base_bias,
        base_edge,
        center,
        spec,
        first_cases,
        seconds=args.seconds,
        camera=camera,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    print(json.dumps({"stage": "baseline", **baseline}), flush=True)

    for generation in range(1, args.generations + 1):
        case_block = (generation - 1) // args.case_refresh_generations
        cases = sample_mirrored_cases(
            args.training_pairs,
            seed=args.seed + case_block,
            device=device,
            hover_config=hover_config,
        )
        sigma = args.sigma * args.sigma_decay ** (generation - 1)
        epsilon = torch.from_numpy(
            rng.standard_normal((args.antithetic_directions, len(spec.labels))).astype(np.float32)
        ).to(device)
        delta = sigma * spec.scales * epsilon
        candidates = torch.stack(
            (
                (center + delta).clamp(spec.lower, spec.upper),
                (center - delta).clamp(spec.lower, spec.upper),
            ),
            dim=1,
        ).reshape(-1, len(spec.labels))
        metrics = []
        for candidate_index, vector in enumerate(candidates):
            result = evaluate_vector(
                controller,
                base_bias,
                base_edge,
                vector,
                spec,
                cases,
                seconds=args.seconds,
                camera=camera,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            metrics.append(result)
            archive.append(
                {
                    "vector": vector.detach().cpu(),
                    "generation": generation,
                    "candidate": candidate_index,
                    "training": result,
                }
            )
        scores = np.asarray([item["fitness"] for item in metrics], dtype=np.float32)
        ranks = np.empty(len(scores), dtype=np.float32)
        ranks[np.argsort(scores)] = np.linspace(-0.5, 0.5, len(scores))
        utilities = torch.from_numpy(ranks).to(device)
        paired_utility = utilities[0::2] - utilities[1::2]
        gradient = (paired_utility[:, None] * epsilon).mean(dim=0) / sigma
        center = (center + args.learning_rate * spec.scales * gradient).clamp(
            spec.lower, spec.upper
        )
        center_metrics = evaluate_vector(
            controller,
            base_bias,
            base_edge,
            center,
            spec,
            cases,
            seconds=args.seconds,
            camera=camera,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        archive.append(
            {
                "vector": center.detach().cpu(),
                "generation": generation,
                "candidate": "center",
                "training": center_metrics,
            }
        )
        best_index = int(np.argmax(scores))
        entry = {
            "generation": generation,
            "case_seed": cases.seed,
            "sigma": sigma,
            "center": center_metrics,
            "best_candidate": metrics[best_index],
            "best_candidate_index": best_index,
            "elapsed_seconds": perf_counter() - started,
        }
        history.append(entry)
        print(
            json.dumps(
                {
                    "stage": "search",
                    "generation": generation,
                    "center_fitness": center_metrics["fitness"],
                    "center_success": center_metrics["success_rate"],
                    "center_paired_success": center_metrics["paired_success_rate"],
                    "best_fitness": metrics[best_index]["fitness"],
                    "best_success": metrics[best_index]["success_rate"],
                    "best_paired_success": metrics[best_index]["paired_success_rate"],
                    "elapsed_seconds": entry["elapsed_seconds"],
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "generation": generation,
                "center": center.detach().cpu(),
                "history": history,
                "parameter_labels": spec.labels,
            },
            args.output_dir / "search-state.pt",
        )

    validation_cases = sample_mirrored_cases(
        args.validation_pairs,
        seed=args.validation_seed,
        device=device,
        hover_config=hover_config,
    )
    ranked = sorted(archive, key=lambda item: item["training"]["fitness"], reverse=True)
    validation_sources = [
        {
            "vector": torch.zeros(len(spec.labels)),
            "generation": 0,
            "candidate": "hover_source",
            "training": baseline,
        },
        *ranked[: args.archive_size],
    ]
    validation = []
    seen: set[str] = set()
    for item in validation_sources:
        vector = item["vector"].to(device)
        vector_hash = motor_es.vector_sha256(vector)
        if vector_hash in seen:
            continue
        seen.add(vector_hash)
        result = evaluate_vector(
            controller,
            base_bias,
            base_edge,
            vector,
            spec,
            validation_cases,
            seconds=args.seconds,
            camera=camera,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        validation.append(
            {
                "vector": vector.detach().cpu(),
                "vector_sha256": vector_hash,
                "generation": item["generation"],
                "candidate": item["candidate"],
                "metrics": result,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "validation",
                    "generation": item["generation"],
                    "candidate": item["candidate"],
                    **result,
                }
            ),
            flush=True,
        )
    winner = max(validation, key=lambda item: item["metrics"]["fitness"])
    winner_vector = winner["vector"].to(device)
    frozen = evaluate_vector(
        controller,
        base_bias,
        base_edge,
        winner_vector,
        spec,
        validation_cases,
        seconds=args.seconds,
        camera=camera,
        hover_config=hover_config,
        gate_config=gate_config,
        frozen_vision=True,
    )
    apply_vector(controller, base_bias, base_edge, winner_vector, spec)
    checkpoint = {
        "controller": {
            name: value.detach().cpu() for name, value in controller.state_dict().items()
        },
        "graph_sha256": responsibility.file_sha256(args.graph),
        "source_checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
        "image_resolution": [camera.width, camera.height],
        "camera_hfov_degrees": camera.horizontal_fov_degrees,
        "policy_hz": POLICY_HZ,
        "physics_hz": PHYSICS_HZ,
        "hover_config": vars(hover_config),
        "gate_config": vars(gate_config),
        "motor_interface_es": {
            "parameter_labels": list(spec.labels),
            "parameter_vector": winner_vector.detach().cpu().tolist(),
            "parameter_vector_sha256": winner["vector_sha256"],
        },
    }
    checkpoint_path = args.output_dir / "controller.pt"
    torch.save(checkpoint, checkpoint_path)
    serializable_validation = [
        {name: value for name, value in item.items() if name != "vector"} for item in validation
    ]
    report = {
        "experiment": "pragmatic-full-native-mirrored-gate-es-v1",
        "purpose": "rapid behavioral proof of concept; not a formal promotion run",
        "actor": {
            "inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
            "state": "native MaleCNS recurrence only",
            "outputs": ["roll", "pitch", "yaw", "throttle"],
            "output_path": "native motor pools -> both foreleg stick plant -> acro RC",
            "teacher_action_used": False,
            "accelerometer_used": False,
            "external_history_or_state_machine_used": False,
            "ground_truth_gate_geometry_used_by_actor": False,
        },
        "configuration": {
            "generations": args.generations,
            "antithetic_directions": args.antithetic_directions,
            "training_pairs": args.training_pairs,
            "validation_pairs": args.validation_pairs,
            "seconds": args.seconds,
            "parameter_count": len(spec.labels),
            "parameter_labels": list(spec.labels),
            "case_refresh_generations": args.case_refresh_generations,
            "seed": args.seed,
            "validation_seed": args.validation_seed,
        },
        "training_case_manifest": case_manifest(first_cases, gate_config),
        "validation_case_manifest": case_manifest(validation_cases, gate_config),
        "source_baseline": baseline,
        "history": history,
        "validation": serializable_validation,
        "selected": {name: value for name, value in winner.items() if name != "vector"},
        "selected_frozen_after_half_second": frozen,
        "checkpoint": str(checkpoint_path),
        "inputs": {
            "graph": str(args.graph),
            "graph_sha256": responsibility.file_sha256(args.graph),
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
            "source_experiment": source_payload.get("experiment"),
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": perf_counter() - started,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "stage": "complete",
                "selected": report["selected"],
                "frozen_after_half_second": frozen,
                "checkpoint": str(checkpoint_path),
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
