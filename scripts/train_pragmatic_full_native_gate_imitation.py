#!/usr/bin/env python3
"""Teach the full MaleCNS a first visual gate servo, then test native flights."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_responsibilities as responsibility  # noqa: E402
from search_pragmatic_full_native_gate_es import (  # noqa: E402
    MirroredGateCases,
    case_manifest,
    evaluate_cases,
    sample_mirrored_cases,
)

from flydrone.gate import (  # noqa: E402
    GateConfig,
    render_annular_gate_rgb,
    teacher_gate_rc,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
    motor_target_for_rc,
    rotation_matrix,
)
from flydrone.visual_hover import CameraSpec  # noqa: E402

POLICY_HZ = 50
PHYSICS_HZ = 100
MOTOR_ERROR_SCALE = (0.10, 0.10, 0.05, 0.15)


@dataclass
class GateRollout:
    state: QuadState
    sticks: StickState
    gate: Any
    mass_scale: Tensor
    neural: Tensor
    side: Tensor
    age: int
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
        default=REPO_ROOT / "runs/gate/pragmatic-full-native-imitation-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--static-updates", type=int, default=200)
    parser.add_argument("--alignment-updates", type=int, default=100)
    parser.add_argument("--teacher-updates", type=int, default=80)
    parser.add_argument("--native-handoff-updates", type=int, default=120)
    parser.add_argument("--training-pairs", type=int, default=2)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--static-response-steps", type=int, default=25)
    parser.add_argument("--static-learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--temporal-learning-rate", type=float, default=8.0e-5)
    parser.add_argument("--source-regularization", type=float, default=2.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.7)
    parser.add_argument("--native-handoff-after-seconds", type=float, default=0.5)
    parser.add_argument("--rollout-seconds", type=float, default=7.0)
    parser.add_argument("--evaluation-pairs", type=int, default=16)
    parser.add_argument("--evaluation-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=570_983)
    parser.add_argument("--evaluation-seed", type=int, default=580_983)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.static_updates,
        args.alignment_updates,
        args.teacher_updates,
        args.native_handoff_updates,
        args.training_pairs,
        args.unroll,
        args.static_response_steps,
        args.static_learning_rate,
        args.temporal_learning_rate,
        args.source_regularization,
        args.gradient_clip_norm,
        args.rollout_seconds,
        args.evaluation_pairs,
        args.evaluation_interval,
    )
    if min(positive) <= 0:
        raise SystemExit("training sizes, rates, and durations must be positive")
    if not 0.0 <= args.native_handoff_after_seconds < args.rollout_seconds:
        raise SystemExit("native handoff time must lie within the rollout")


def load_controller(
    args: argparse.Namespace, device: torch.device
) -> tuple[ConnectomeController, dict[str, Any]]:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("graph_sha256") != responsibility.file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    controller = ConnectomeController(args.graph, neural_dt=1.0 / POLICY_HZ).to(device)
    controller.load_state_dict(payload["controller"])
    if controller.uses_accelerometer or controller.uses_proprioception:
        raise SystemExit("the full-network proof actor unexpectedly uses added sensor channels")
    return controller, payload


def source_regularization(
    controller: ConnectomeController,
    source: dict[str, Tensor],
    coefficient: float,
) -> Tensor:
    scale = 0.02
    return coefficient * (
        ((controller.edge_magnitude - source["edge_magnitude"]) / scale).square().mean()
        + ((controller.bias - source["bias"]) / scale).square().mean()
        + 0.25
        * ((controller.raw_time_constant - source["raw_time_constant"]) / scale).square().mean()
    )


def imitation_loss(
    prediction: Tensor, target: Tensor, side: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    scale = prediction.new_tensor(MOTOR_ERROR_SCALE)
    direct = ((prediction - target) / scale).square().mean()
    prediction_pairs = prediction.reshape(-1, 2, 4)
    target_pairs = target.reshape(-1, 2, 4)
    predicted_contrast = prediction_pairs[:, 1] - prediction_pairs[:, 0]
    target_contrast = target_pairs[:, 1] - target_pairs[:, 0]
    contrast = ((predicted_contrast - target_contrast) / (math.sqrt(2.0) * scale)).square().mean()
    # The gate side must reverse roll. This explicit paired term is still only an
    # imitation target; it does not enter the deployed actor.
    desired_roll_sign = torch.sign(target[:, 0])
    signed_roll_margin = (0.012 - prediction[:, 0] * desired_roll_sign).clamp_min(0.0)
    roll_direction = (signed_roll_margin / 0.05).square().mean()
    if not bool(torch.equal(side.reshape(-1, 2)[:, 0], -side.reshape(-1, 2)[:, 1])):
        raise ValueError("training batches must retain mirrored-pair ordering")
    # Mean actions are easy to learn through motor-pool bias. The deliberately strong
    # paired term makes the optimizer use the existing retinotopic visual pathways
    # instead of settling for a one-sided steering bias.
    total = 8.0 * direct + 30.0 * contrast + 2.0 * roll_direction
    return total, {
        "direct": direct.detach(),
        "contrast": contrast.detach(),
        "roll_direction": roll_direction.detach(),
    }


def staged_teacher_rc(
    state: QuadState,
    gate: Any,
    config: HoverConfig,
    *,
    advance: bool,
) -> Tensor:
    """Training-only align-then-advance controller using visually encoded geometry."""

    to_gate_world = gate.center - state.position
    world_to_body = rotation_matrix(state.euler).transpose(1, 2)
    to_gate_body = torch.einsum("bij,bj->bi", world_to_body, to_gate_world)
    horizontal_range = torch.linalg.vector_norm(to_gate_body[:, :2], dim=1)
    bearing = torch.atan2(to_gate_body[:, 1], to_gate_body[:, 0])
    elevation = torch.atan2(to_gate_body[:, 2], horizontal_range.clamp_min(1.0e-6))
    approaching = to_gate_body[:, 0] > 0.0
    alignment = torch.sigmoid((0.14 - bearing.abs()) / 0.025) * approaching
    desired_roll = (-0.80 * bearing).clamp(-0.22, 0.22)
    desired_pitch = 0.24 * alignment if advance else torch.zeros_like(desired_roll)
    roll_rate = 3.5 * (desired_roll - state.euler[:, 0])
    pitch_rate = 4.0 * (desired_pitch - state.euler[:, 1])
    hover = 1.0 / config.thrust_to_weight
    tilt_compensation = 1.0 / (
        torch.cos(state.euler[:, 0]) * torch.cos(state.euler[:, 1])
    ).clamp_min(0.75)
    throttle = (
        hover * tilt_compensation
        + 0.25 * torch.where(approaching, elevation, torch.zeros_like(elevation))
    ).clamp(0.0, 1.0)
    return torch.stack(
        (
            (roll_rate / config.max_roll_pitch_rate).clamp(-1.0, 1.0),
            (pitch_rate / config.max_roll_pitch_rate).clamp(-1.0, 1.0),
            torch.zeros_like(roll_rate),
            throttle,
        ),
        dim=1,
    )


def teacher_motor(
    state: QuadState,
    gate: Any,
    config: HoverConfig,
    *,
    mode: str,
) -> Tensor:
    if mode == "original":
        rc = teacher_gate_rc(state, gate, config)
    elif mode in {"align", "staged"}:
        rc = staged_teacher_rc(state, gate, config, advance=mode == "staged")
    else:
        raise ValueError(f"unknown teacher mode: {mode}")
    return motor_target_for_rc(rc, config).detach()


def optimizer_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    loss: Tensor,
    gradient_clip_norm: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), gradient_clip_norm)
    optimizer.step()
    controller.project_parameters()
    return float(gradient_norm.detach())


def static_training_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source: dict[str, Tensor],
    *,
    cases: MirroredGateCases,
    response_steps: int,
    camera: CameraSpec,
    config: HoverConfig,
    gate_config: GateConfig,
    regularization: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    image = render_annular_gate_rgb(
        cases.state,
        cases.gate,
        camera=camera,
        gate_config=gate_config,
    )
    target = teacher_motor(cases.state, cases.gate, config, mode="align")
    neural = controller.initial_state(
        len(cases.side), device=cases.side.device, dtype=torch.float32
    )
    losses = []
    components: dict[str, list[Tensor]] = {
        "direct": [],
        "contrast": [],
        "roll_direction": [],
    }
    for step in range(response_steps):
        prediction, neural = controller(image, cases.state.euler[:, :2], neural)
        if step >= response_steps // 2:
            loss, values = imitation_loss(prediction, target, cases.side)
            losses.append(loss)
            for name, value in values.items():
                components[name].append(value)
    imitation = torch.stack(losses).mean()
    preservation = source_regularization(controller, source, regularization)
    total = imitation + preservation
    gradient_norm = optimizer_step(controller, optimizer, total, gradient_clip_norm)
    return {
        "loss": float(total.detach()),
        "imitation": float(imitation.detach()),
        "source_regularization": float(preservation.detach()),
        "direct": float(torch.stack(components["direct"]).mean()),
        "contrast": float(torch.stack(components["contrast"]).mean()),
        "roll_direction": float(torch.stack(components["roll_direction"]).mean()),
        "gradient_norm": gradient_norm,
    }


def new_rollout(
    controller: ConnectomeController,
    *,
    pairs: int,
    seed: int,
    device: torch.device,
    config: HoverConfig,
) -> GateRollout:
    cases = sample_mirrored_cases(
        pairs,
        seed=seed,
        device=device,
        hover_config=config,
    )
    return GateRollout(
        state=cases.state,
        sticks=cases.sticks,
        gate=cases.gate,
        mass_scale=cases.mass_scale,
        neural=controller.initial_state(2 * pairs, device=device, dtype=torch.float32),
        side=cases.side,
        age=0,
        seed=seed,
    )


def temporal_training_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source: dict[str, Tensor],
    rollout: GateRollout,
    *,
    unroll: int,
    teacher_drives_physics: bool,
    native_handoff_step: int,
    camera: CameraSpec,
    config: HoverConfig,
    gate_config: GateConfig,
    regularization: float,
    gradient_clip_norm: float,
    teacher_mode: str,
) -> tuple[dict[str, float], GateRollout]:
    quad = DifferentiableQuad(config).to(rollout.neural.device)
    stick_plant = ForelegStickPlant(config).to(rollout.neural.device)
    losses = []
    direct_values = []
    contrast_values = []
    roll_values = []
    native_fraction = []
    for _ in range(unroll):
        image = render_annular_gate_rgb(
            rollout.state,
            rollout.gate,
            camera=camera,
            gate_config=gate_config,
        )
        prediction, rollout.neural = controller(
            image,
            rollout.state.euler[:, :2],
            rollout.neural,
        )
        target = teacher_motor(rollout.state, rollout.gate, config, mode=teacher_mode)
        imitation, values = imitation_loss(prediction, target, rollout.side)
        losses.append(imitation)
        direct_values.append(values["direct"])
        contrast_values.append(values["contrast"])
        roll_values.append(values["roll_direction"])
        native = (not teacher_drives_physics) and rollout.age >= native_handoff_step
        applied_motor = prediction.detach() if native else target
        native_fraction.append(float(native))
        for _ in range(PHYSICS_HZ // POLICY_HZ):
            rc, rollout.sticks = stick_plant(applied_motor, rollout.sticks)
            rollout.state = quad(rc, rollout.state, rollout.mass_scale)
        rollout.state = rollout.state.detach()
        rollout.sticks = rollout.sticks.detach()
        rollout.age += 1
    imitation = torch.stack(losses).mean()
    preservation = source_regularization(controller, source, regularization)
    total = imitation + preservation
    gradient_norm = optimizer_step(controller, optimizer, total, gradient_clip_norm)
    rollout.neural = rollout.neural.detach()
    valid = bool(torch.isfinite(rollout.state.position).all()) and bool(
        (rollout.state.position[:, 2] > 0.03).all()
    )
    return (
        {
            "loss": float(total.detach()),
            "imitation": float(imitation.detach()),
            "source_regularization": float(preservation.detach()),
            "direct": float(torch.stack(direct_values).mean()),
            "contrast": float(torch.stack(contrast_values).mean()),
            "roll_direction": float(torch.stack(roll_values).mean()),
            "gradient_norm": gradient_norm,
            "native_physics_fraction": sum(native_fraction) / len(native_fraction),
            "rollout_age_seconds": rollout.age / POLICY_HZ,
            "rollout_valid": valid,
        },
        rollout,
    )


@torch.no_grad()
def static_response(
    controller: ConnectomeController,
    cases: MirroredGateCases,
    *,
    response_steps: int,
    camera: CameraSpec,
    config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, float]:
    image = render_annular_gate_rgb(
        cases.state,
        cases.gate,
        camera=camera,
        gate_config=gate_config,
    )
    target = teacher_motor(cases.state, cases.gate, config, mode="align")
    neural = controller.initial_state(
        len(cases.side), device=cases.side.device, dtype=torch.float32
    )
    prediction = torch.zeros_like(target)
    final_predictions = []
    for step in range(response_steps):
        prediction, neural = controller(image, cases.state.euler[:, :2], neural)
        if step >= response_steps // 2:
            final_predictions.append(prediction)
    prediction = torch.stack(final_predictions).mean(dim=0)
    error = prediction - target
    predicted_contrast = (
        prediction.reshape(-1, 2, 4)[:, 1, 0] - prediction.reshape(-1, 2, 4)[:, 0, 0]
    )
    target_contrast = target.reshape(-1, 2, 4)[:, 1, 0] - target.reshape(-1, 2, 4)[:, 0, 0]
    return {
        "motor_rmse": float(error.square().mean().sqrt()),
        "roll_rmse": float(error[:, 0].square().mean().sqrt()),
        "pitch_rmse": float(error[:, 1].square().mean().sqrt()),
        "throttle_rmse": float(error[:, 3].square().mean().sqrt()),
        "roll_contrast_sign_accuracy": float(
            (torch.sign(predicted_contrast) == torch.sign(target_contrast)).float().mean()
        ),
        "predicted_roll_contrast_rms": float(predicted_contrast.square().mean().sqrt()),
        "target_roll_contrast_rms": float(target_contrast.square().mean().sqrt()),
    }


def save_checkpoint(
    path: Path,
    controller: ConnectomeController,
    *,
    graph: Path,
    source_checkpoint: Path,
    config: HoverConfig,
    gate_config: GateConfig,
    camera: CameraSpec,
    stage: str,
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": "pragmatic-full-native-gate-imitation-v1",
            "controller": {
                name: value.detach().cpu() for name, value in controller.state_dict().items()
            },
            "graph_sha256": responsibility.file_sha256(graph),
            "source_checkpoint_sha256": responsibility.file_sha256(source_checkpoint),
            "policy_hz": POLICY_HZ,
            "physics_hz": PHYSICS_HZ,
            "image_resolution": [camera.width, camera.height],
            "camera_hfov_degrees": camera.horizontal_fov_degrees,
            "hover_config": vars(config),
            "gate_config": vars(gate_config),
            "stage": stage,
            "update": update,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    responsibility.seed_everything(args.seed)
    controller, source_payload = load_controller(args, device)
    config = HoverConfig()
    gate_config = GateConfig()
    camera = CameraSpec(width=320, height=200, horizontal_fov_degrees=125.0)
    source = {
        name: value.detach().clone()
        for name, value in controller.state_dict().items()
        if name in {"edge_magnitude", "bias", "raw_time_constant"}
    }
    started = perf_counter()
    history: list[dict[str, Any]] = []
    evaluation_cases = sample_mirrored_cases(
        args.evaluation_pairs,
        seed=args.evaluation_seed,
        device=device,
        hover_config=config,
    )
    baseline_static = static_response(
        controller,
        evaluation_cases,
        response_steps=args.static_response_steps,
        camera=camera,
        config=config,
        gate_config=gate_config,
    )
    baseline_flight = evaluate_cases(
        controller,
        evaluation_cases,
        seconds=args.rollout_seconds,
        camera=camera,
        hover_config=config,
        gate_config=gate_config,
    )
    print(
        json.dumps({"stage": "baseline", "static": baseline_static, "flight": baseline_flight}),
        flush=True,
    )

    static_optimizer = torch.optim.AdamW(
        controller.parameters(), lr=args.static_learning_rate, weight_decay=1.0e-6
    )
    for update in range(1, args.static_updates + 1):
        cases = sample_mirrored_cases(
            args.training_pairs,
            seed=args.seed + update,
            device=device,
            hover_config=config,
        )
        metrics = static_training_step(
            controller,
            static_optimizer,
            source,
            cases=cases,
            response_steps=args.static_response_steps,
            camera=camera,
            config=config,
            gate_config=gate_config,
            regularization=args.source_regularization,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        if update == 1 or update % 10 == 0 or update == args.static_updates:
            entry = {
                "stage": "static",
                "update": update,
                **metrics,
                "elapsed_seconds": perf_counter() - started,
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)

    temporal_optimizer = torch.optim.AdamW(
        controller.parameters(), lr=args.temporal_learning_rate, weight_decay=1.0e-6
    )
    global_update = args.static_updates
    rollout_seed = args.seed + 100_000
    rollout = new_rollout(
        controller,
        pairs=args.training_pairs,
        seed=rollout_seed,
        device=device,
        config=config,
    )
    best_score = (-1.0, -1.0, float("-inf"))
    best_path = args.output_dir / "best-controller.pt"
    stage_specs = (
        ("alignment_trajectory", args.alignment_updates, True, "align", 3.0),
        ("teacher_trajectory", args.teacher_updates, True, "staged", args.rollout_seconds),
        (
            "native_handoff",
            args.native_handoff_updates,
            False,
            "staged",
            args.rollout_seconds,
        ),
    )
    for stage, updates, teacher_drives_physics, teacher_mode, stage_seconds in stage_specs:
        rollout_seed += 1
        rollout = new_rollout(
            controller,
            pairs=args.training_pairs,
            seed=rollout_seed,
            device=device,
            config=config,
        )
        for stage_update in range(1, updates + 1):
            global_update += 1
            max_age = round(stage_seconds * POLICY_HZ)
            valid = bool(torch.isfinite(rollout.state.position).all()) and bool(
                (rollout.state.position[:, 2] > 0.03).all()
            )
            if rollout.age + args.unroll > max_age or not valid:
                rollout_seed += 1
                rollout = new_rollout(
                    controller,
                    pairs=args.training_pairs,
                    seed=rollout_seed,
                    device=device,
                    config=config,
                )
            metrics, rollout = temporal_training_step(
                controller,
                temporal_optimizer,
                source,
                rollout,
                unroll=args.unroll,
                teacher_drives_physics=teacher_drives_physics,
                native_handoff_step=round(args.native_handoff_after_seconds * POLICY_HZ),
                camera=camera,
                config=config,
                gate_config=gate_config,
                regularization=args.source_regularization,
                gradient_clip_norm=args.gradient_clip_norm,
                teacher_mode=teacher_mode,
            )
            should_evaluate = (
                stage_update == updates or stage_update % args.evaluation_interval == 0
            )
            entry = {
                "stage": stage,
                "stage_update": stage_update,
                "global_update": global_update,
                **metrics,
                "elapsed_seconds": perf_counter() - started,
            }
            if should_evaluate:
                controller.eval()
                entry["static_response"] = static_response(
                    controller,
                    evaluation_cases,
                    response_steps=args.static_response_steps,
                    camera=camera,
                    config=config,
                    gate_config=gate_config,
                )
                entry["native_flight"] = evaluate_cases(
                    controller,
                    evaluation_cases,
                    seconds=args.rollout_seconds,
                    camera=camera,
                    hover_config=config,
                    gate_config=gate_config,
                )
                controller.train()
                flight = entry["native_flight"]
                score = (
                    flight["paired_success_rate"],
                    flight["success_rate"],
                    flight["fitness"],
                )
                if score > best_score:
                    best_score = score
                    save_checkpoint(
                        best_path,
                        controller,
                        graph=args.graph,
                        source_checkpoint=args.checkpoint,
                        config=config,
                        gate_config=gate_config,
                        camera=camera,
                        stage=stage,
                        update=global_update,
                    )
            if stage_update == 1 or stage_update % 10 == 0 or should_evaluate:
                history.append(entry)
                print(json.dumps(entry), flush=True)

    controller.eval()
    final_static = static_response(
        controller,
        evaluation_cases,
        response_steps=args.static_response_steps,
        camera=camera,
        config=config,
        gate_config=gate_config,
    )
    final_flight = evaluate_cases(
        controller,
        evaluation_cases,
        seconds=args.rollout_seconds,
        camera=camera,
        hover_config=config,
        gate_config=gate_config,
    )
    frozen_flight = evaluate_cases(
        controller,
        evaluation_cases,
        seconds=args.rollout_seconds,
        camera=camera,
        hover_config=config,
        gate_config=gate_config,
        frozen_vision=True,
    )
    final_path = args.output_dir / "controller.pt"
    save_checkpoint(
        final_path,
        controller,
        graph=args.graph,
        source_checkpoint=args.checkpoint,
        config=config,
        gate_config=gate_config,
        camera=camera,
        stage="complete",
        update=global_update,
    )
    report = {
        "experiment": "pragmatic-full-native-gate-imitation-v1",
        "purpose": "rapid behavioral proof of concept; not a formal promotion run",
        "actor": {
            "deployed_inputs": [
                "320x200 linear RGB at 125 degree HFOV",
                "roll",
                "pitch",
            ],
            "deployed_state": "native MaleCNS recurrence only",
            "deployed_outputs": ["roll", "pitch", "yaw", "throttle"],
            "output_path": "native motor pools -> both foreleg stick plant -> acro RC",
            "teacher_or_gate_geometry_used_during_deployment": False,
            "accelerometer_used": False,
            "external_history_or_state_machine_used": False,
            "training_only_teacher": "observation-compatible visual-servo labels",
        },
        "configuration": {
            "static_updates": args.static_updates,
            "alignment_updates": args.alignment_updates,
            "teacher_updates": args.teacher_updates,
            "native_handoff_updates": args.native_handoff_updates,
            "training_pairs": args.training_pairs,
            "unroll": args.unroll,
            "static_response_steps": args.static_response_steps,
            "static_learning_rate": args.static_learning_rate,
            "temporal_learning_rate": args.temporal_learning_rate,
            "source_regularization": args.source_regularization,
            "native_handoff_after_seconds": args.native_handoff_after_seconds,
            "rollout_seconds": args.rollout_seconds,
            "seed": args.seed,
            "evaluation_seed": args.evaluation_seed,
        },
        "evaluation_case_manifest": case_manifest(evaluation_cases, gate_config),
        "baseline_static": baseline_static,
        "baseline_flight": baseline_flight,
        "final_static": final_static,
        "final_native_flight": final_flight,
        "final_frozen_after_half_second": frozen_flight,
        "best_native_flight_score_during_training": {
            "paired_success_rate": best_score[0],
            "success_rate": best_score[1],
            "fitness": best_score[2],
        },
        "history": history,
        "checkpoint": str(final_path),
        "best_checkpoint": str(best_path),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "stage": "complete",
                "final_native_flight": final_flight,
                "frozen_after_half_second": frozen_flight,
                "checkpoint": str(final_path),
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
