#!/usr/bin/env python3
"""Tune native visual-to-roll paths for uninterrupted role-coloured gate courses."""

from __future__ import annotations

import argparse
import copy
import json
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
import train_variable_height_hover as hover_train  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import (  # noqa: E402
    active_gate,
    evaluate,
    sample_two_gate_cases,
)
from train_pragmatic_full_native_gate_imitation import teacher_motor  # noqa: E402
from train_pragmatic_gate_visual_roll_path import (  # noqa: E402
    load_controller,
    visual_roll_path_mask,
)

from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    render_annular_gates_rgb,
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


@dataclass
class CourseRollout:
    state: QuadState
    sticks: StickState
    gates: tuple[AnnularGate, ...]
    mass_scale: Tensor
    neural: Tensor
    source_neural: Tensor
    side: Tensor
    current: Tensor
    failed: Tensor
    age: int


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
        default=REPO_ROOT / "runs/gate/pragmatic-visual-roll-path-001/best-controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/gate/pragmatic-two-gate-roll-path-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-hop-budget", type=int, default=5)
    parser.add_argument(
        "--last-hop-only",
        action="store_true",
        help="train only selected anatomical edges entering the six roll motor neurons",
    )
    parser.add_argument("--teacher-updates", type=int, default=250)
    parser.add_argument("--native-updates", type=int, default=250)
    parser.add_argument("--training-pairs", type=int, default=2)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--contrast-weight", type=float, default=4.0)
    parser.add_argument("--gate-one-preservation-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--observation-warmup-steps", type=int, default=10)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--evaluation-pairs", type=int, default=8)
    parser.add_argument("--evaluation-seconds", type=float, default=22.0)
    parser.add_argument(
        "--balanced-late-gate-starts",
        action="store_true",
        help="cycle training-only reset positions across gates 2..N",
    )
    parser.add_argument(
        "--curriculum-window-seconds",
        type=float,
        default=5.0,
        help="maximum rollout age when balanced late-gate starts are enabled",
    )
    parser.add_argument(
        "--layout",
        choices=("aligned", "s-turn", "variable"),
        default="aligned",
    )
    parser.add_argument("--gates", type=int, default=2)
    parser.add_argument("--spacing-min", type=float, default=3.8)
    parser.add_argument("--spacing-max", type=float, default=4.2)
    parser.add_argument("--lateral-step-min", type=float, default=0.0)
    parser.add_argument("--lateral-step-max", type=float, default=0.10)
    parser.add_argument("--lateral-deviation-limit", type=float, default=0.25)
    parser.add_argument("--height-step-min", type=float, default=0.0)
    parser.add_argument("--height-step-max", type=float, default=0.05)
    parser.add_argument("--height-min", type=float, default=0.95)
    parser.add_argument("--height-max", type=float, default=1.25)
    parser.add_argument(
        "--lesson",
        choices=("all", "second-gate", "after-first"),
        default="second-gate",
        help="which active course segment contributes imitation gradients",
    )
    parser.add_argument("--seed", type=int, default=640_983)
    parser.add_argument("--evaluation-seed", type=int, default=650_983)
    return parser.parse_args()


def new_rollout(
    controller: ConnectomeController,
    source_controller: ConnectomeController,
    *,
    pairs: int,
    seed: int,
    device: torch.device,
    config: HoverConfig,
    camera: CameraSpec,
    gate_config: GateConfig,
    warmup_steps: int,
    layout: str,
    gate_count: int,
    spacing_range: tuple[float, float],
    lateral_step_range: tuple[float, float],
    lateral_deviation_limit: float,
    height_step_range: tuple[float, float],
    height_range: tuple[float, float],
    start_gate: int = 0,
) -> CourseRollout:
    cases, gates = sample_two_gate_cases(
        pairs,
        seed=seed,
        device=device,
        hover_config=config,
        layout=layout,
        spacing_range=spacing_range,
        gate_count=gate_count,
        lateral_step_range=lateral_step_range,
        lateral_deviation_limit=lateral_deviation_limit,
        height_step_range=height_step_range,
        height_range=height_range,
    )
    if not 0 <= start_gate < len(gates):
        raise ValueError("start_gate must index the sampled course")
    if start_gate:
        selected = gates[start_gate]
        previous = gates[start_gate - 1]
        segment = selected.center - previous.center
        horizontal = segment[:, :2]
        unit = horizontal / torch.linalg.vector_norm(
            horizontal,
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        original_height_error = cases.state.position[:, 2] - 1.10
        cases.state.position[:, :2] = selected.center[:, :2] - 1.40 * unit
        cases.state.position[:, 2] = (
            selected.center[:, 2] + original_height_error
        ).clamp_min(0.20)
    current = torch.full(
        (2 * pairs,),
        start_gate,
        dtype=torch.long,
        device=device,
    )
    neural = controller.initial_state(2 * pairs, device=device, dtype=torch.float32)
    source_neural = source_controller.initial_state(
        2 * pairs, device=device, dtype=torch.float32
    )
    image = render_annular_gates_rgb(
        cases.state,
        gates,
        current_gate_index=current,
        camera=camera,
        gate_config=gate_config,
    )
    with torch.no_grad():
        for _ in range(warmup_steps):
            _, neural = controller(image, cases.state.euler[:, :2], neural)
            _, source_neural = source_controller(
                image,
                cases.state.euler[:, :2],
                source_neural,
            )
    return CourseRollout(
        state=cases.state,
        sticks=cases.sticks,
        gates=gates,
        mass_scale=cases.mass_scale,
        neural=neural,
        source_neural=source_neural,
        side=cases.side,
        current=current,
        failed=torch.zeros(2 * pairs, dtype=torch.bool, device=device),
        age=0,
    )


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.sum() * 0.0


def training_step(
    controller: ConnectomeController,
    source_controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    rollout: CourseRollout,
    preservation_rollout: CourseRollout,
    source_edges: Tensor,
    selected_mask: Tensor,
    *,
    unroll: int,
    physics_mode: str,
    contrast_weight: float,
    gate_one_preservation_weight: float,
    anchor_weight: float,
    gradient_clip_norm: float,
    camera: CameraSpec,
    config: HoverConfig,
    gate_config: GateConfig,
    lesson: str,
) -> tuple[dict[str, float], CourseRollout, CourseRollout]:
    quad = DifferentiableQuad(config).to(rollout.neural.device)
    stick_plant = ForelegStickPlant(config).to(rollout.neural.device)
    direct_losses = []
    contrast_losses = []
    preservation_losses = []
    target_rms = []
    prediction_rms = []
    lesson_samples = 0
    for _ in range(unroll):
        active = (rollout.current < len(rollout.gates)) & ~rollout.failed
        lesson_active = active
        if lesson == "second-gate":
            lesson_active = lesson_active & (rollout.current == 1)
        elif lesson == "after-first":
            lesson_active = lesson_active & (rollout.current >= 1)
        lesson_samples += int(lesson_active.sum())
        image = render_annular_gates_rgb(
            rollout.state,
            rollout.gates,
            current_gate_index=rollout.current,
            camera=camera,
            gate_config=gate_config,
        )
        prediction, rollout.neural = controller(
            image,
            rollout.state.euler[:, :2],
            rollout.neural,
        )
        with torch.no_grad():
            source_motor, rollout.source_neural = source_controller(
                image,
                rollout.state.euler[:, :2],
                rollout.source_neural,
            )
        selected = active_gate(rollout.gates, rollout.current)
        target = teacher_motor(rollout.state, selected, config, mode="staged")
        direct_error = ((prediction[:, 0] - target[:, 0]) / 0.10).square()
        direct_losses.append(masked_mean(direct_error, lesson_active))
        prediction_pair = prediction[:, 0].reshape(-1, 2)
        target_pair = target[:, 0].reshape(-1, 2)
        pair_active = lesson_active.reshape(-1, 2).all(dim=1)
        contrast_error = (
            (prediction_pair[:, 1] - prediction_pair[:, 0])
            - (target_pair[:, 1] - target_pair[:, 0])
        ).div(0.15).square()
        contrast_losses.append(masked_mean(contrast_error, pair_active))
        target_rms.append(
            masked_mean(target[:, 0].square(), lesson_active).sqrt().detach()
        )
        prediction_rms.append(
            masked_mean(prediction[:, 0].square(), lesson_active).sqrt().detach()
        )
        if physics_mode == "recovery":
            applied_motor = torch.where(
                (rollout.current == 0)[:, None],
                source_motor,
                target,
            )
        elif physics_mode == "native":
            applied_motor = prediction.detach()
        else:
            raise ValueError(f"unknown physics mode: {physics_mode}")
        for _ in range(PHYSICS_HZ // POLICY_HZ):
            rc, rollout.sticks = stick_plant(applied_motor, rollout.sticks)
            previous_position = rollout.state.position
            rollout.state = quad(rc, rollout.state, rollout.mass_scale)
            active = (rollout.current < len(rollout.gates)) & ~rollout.failed
            selected = active_gate(rollout.gates, rollout.current)
            pass_now, collision_now, miss_now = classify_gate_crossing(
                previous_position,
                rollout.state.position,
                selected,
                gate_config,
            )
            pass_now &= active
            collision_now &= active
            miss_now &= active
            rollout.current = rollout.current + pass_now.long()
            rollout.failed |= collision_now | miss_now
        rollout.failed |= rollout.state.position[:, 2] <= 0.03
        rollout.failed |= ~hover_train.state_is_valid(rollout.state)
        rollout.state = QuadState(*(value.detach() for value in rollout.state.as_tuple()))
        rollout.sticks = StickState(
            rollout.sticks.joint_position.detach(),
            rollout.sticks.joint_velocity.detach(),
            rollout.sticks.position.detach(),
            rollout.sticks.velocity.detach(),
        )
        rollout.age += 1

    preservation_samples = 0
    for _ in range(unroll):
        active = (preservation_rollout.current == 0) & ~preservation_rollout.failed
        preservation_samples += int(active.sum())
        image = render_annular_gates_rgb(
            preservation_rollout.state,
            preservation_rollout.gates,
            current_gate_index=preservation_rollout.current,
            camera=camera,
            gate_config=gate_config,
        )
        prediction, preservation_rollout.neural = controller(
            image,
            preservation_rollout.state.euler[:, :2],
            preservation_rollout.neural,
        )
        with torch.no_grad():
            source_motor, preservation_rollout.source_neural = source_controller(
                image,
                preservation_rollout.state.euler[:, :2],
                preservation_rollout.source_neural,
            )
        preservation_error = ((prediction[:, 0] - source_motor[:, 0]) / 0.05).square()
        preservation_losses.append(masked_mean(preservation_error, active))
        for _ in range(PHYSICS_HZ // POLICY_HZ):
            rc, preservation_rollout.sticks = stick_plant(
                source_motor,
                preservation_rollout.sticks,
            )
            previous_position = preservation_rollout.state.position
            preservation_rollout.state = quad(
                rc,
                preservation_rollout.state,
                preservation_rollout.mass_scale,
            )
            active = (preservation_rollout.current == 0) & ~preservation_rollout.failed
            selected = active_gate(
                preservation_rollout.gates,
                preservation_rollout.current,
            )
            pass_now, collision_now, miss_now = classify_gate_crossing(
                previous_position,
                preservation_rollout.state.position,
                selected,
                gate_config,
            )
            pass_now &= active
            collision_now &= active
            miss_now &= active
            preservation_rollout.current += pass_now.long()
            preservation_rollout.failed |= collision_now | miss_now
        preservation_rollout.failed |= preservation_rollout.state.position[:, 2] <= 0.03
        preservation_rollout.failed |= ~hover_train.state_is_valid(
            preservation_rollout.state
        )
        preservation_rollout.state = QuadState(
            *(value.detach() for value in preservation_rollout.state.as_tuple())
        )
        preservation_rollout.sticks = StickState(
            preservation_rollout.sticks.joint_position.detach(),
            preservation_rollout.sticks.joint_velocity.detach(),
            preservation_rollout.sticks.position.detach(),
            preservation_rollout.sticks.velocity.detach(),
        )
        preservation_rollout.age += 1

    direct = torch.stack(direct_losses).mean()
    contrast = torch.stack(contrast_losses).mean()
    gate_one_preservation = torch.stack(preservation_losses).mean()
    anchor = (
        ((controller.edge_magnitude[selected_mask] - source_edges[selected_mask]) / 0.25)
        .square()
        .mean()
    )
    loss = (
        direct
        + contrast_weight * contrast
        + gate_one_preservation_weight * gate_one_preservation
        + anchor_weight * anchor
    )
    gradient_norm = controller.edge_magnitude.new_zeros(())
    if lesson_samples or optimizer.state:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (controller.edge_magnitude,), gradient_clip_norm
        )
        optimizer.step()
        controller.project_parameters()
    rollout.neural = rollout.neural.detach()
    rollout.source_neural = rollout.source_neural.detach()
    preservation_rollout.neural = preservation_rollout.neural.detach()
    preservation_rollout.source_neural = preservation_rollout.source_neural.detach()
    return (
        {
            "loss": float(loss.detach()),
            "roll_direct": float(direct.detach()),
            "roll_contrast": float(contrast.detach()),
            "gate_one_roll_preservation": float(gate_one_preservation.detach()),
            "selected_edge_anchor": float(anchor.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "lesson_samples": lesson_samples,
            "gate_one_preservation_samples": preservation_samples,
            "target_roll_motor_rms": float(torch.stack(target_rms).mean()),
            "predicted_roll_motor_rms": float(torch.stack(prediction_rms).mean()),
            "rollout_age_seconds": rollout.age / POLICY_HZ,
            "rollout_completed_fraction": float(
                (rollout.current == len(rollout.gates)).float().mean()
            ),
            "rollout_failed_fraction": float(rollout.failed.float().mean()),
        },
        rollout,
        preservation_rollout,
    )


def save_checkpoint(
    path: Path,
    controller: ConnectomeController,
    *,
    args: argparse.Namespace,
    config: HoverConfig,
    gate_config: GateConfig,
    camera: CameraSpec,
    stage: str,
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": "pragmatic-full-native-gate-course-roll-path-v2",
            "controller": {
                name: value.detach().cpu() for name, value in controller.state_dict().items()
            },
            "graph_sha256": responsibility.file_sha256(args.graph),
            "source_checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
            "policy_hz": POLICY_HZ,
            "physics_hz": PHYSICS_HZ,
            "image_resolution": [camera.width, camera.height],
            "camera_hfov_degrees": camera.horizontal_fov_degrees,
            "hover_config": vars(config),
            "gate_config": vars(gate_config),
            "stage": stage,
            "update": update,
            "same_update_gate_one_replay": True,
            "last_hop_only": args.last_hop_only,
            "gate_count": args.gates,
            "layout": args.layout,
            "balanced_late_gate_starts": args.balanced_late_gate_starts,
            "curriculum_window_seconds": args.curriculum_window_seconds,
        },
        path,
    )


def score(
    metrics: dict[str, object],
    *,
    first_gate_floor: float = 0.0,
    first_gate_paired_floor: float = 0.0,
) -> tuple[float, ...]:
    retained = (
        float(metrics["first_gate_pass_rate"]) >= first_gate_floor
        and float(metrics["first_gate_paired_pass_rate"]) >= first_gate_paired_floor
    )
    return (
        float(retained),
        float(metrics["all_gates_paired_pass_rate"]),
        min(
            float(metrics["all_gates_negative_course_pass_rate"]),
            float(metrics["all_gates_positive_course_pass_rate"]),
        ),
        float(metrics["all_gates_pass_rate"]),
        float(metrics["first_gate_paired_pass_rate"]),
        float(metrics["first_gate_pass_rate"]),
    )


def main() -> int:
    args = parse_args()
    if args.gates < 1 or (args.layout == "s-turn" and args.gates != 2):
        raise SystemExit("gates must be positive; s-turn supports exactly two")
    if args.balanced_late_gate_starts and args.gates < 2:
        raise SystemExit("balanced late-gate starts require at least two gates")
    if args.curriculum_window_seconds <= 0.0:
        raise SystemExit("curriculum window must be positive")
    if args.spacing_min <= 0.0 or args.spacing_max < args.spacing_min:
        raise SystemExit("spacing range must be positive and ordered")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    responsibility.seed_everything(args.seed)
    controller, source_payload = load_controller(args, device)
    source_controller = copy.deepcopy(controller).eval().requires_grad_(False)
    selected_mask, path_manifest = visual_roll_path_mask(
        args.graph,
        hop_budget=args.path_hop_budget,
        device=device,
    )
    if args.last_hop_only:
        roll_motor_nodes = controller.pool_indices[
            int(controller.pool_offsets[0]) : int(controller.pool_offsets[2])
        ]
        selected_mask &= torch.isin(controller.edge_post, roll_motor_nodes)
        path_manifest = {
            **path_manifest,
            "selection": "last-hop edges entering roll motor neurons",
            "trainable_edges": int(selected_mask.sum()),
        }
    source_edges = controller.edge_magnitude.detach().clone()
    gradient_mask = selected_mask.to(dtype=controller.edge_magnitude.dtype)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * gradient_mask)
    optimizer = torch.optim.Adam((controller.edge_magnitude,), lr=args.learning_rate)
    config = HoverConfig(**source_payload["hover_config"])
    gate_config = GateConfig(**source_payload["gate_config"])
    camera = CameraSpec(
        width=source_payload["image_resolution"][0],
        height=source_payload["image_resolution"][1],
        horizontal_fov_degrees=source_payload["camera_hfov_degrees"],
    )
    evaluation_cases, evaluation_gates = sample_two_gate_cases(
        args.evaluation_pairs,
        seed=args.evaluation_seed,
        device=device,
        hover_config=config,
        layout=args.layout,
        spacing_range=(args.spacing_min, args.spacing_max),
        gate_count=args.gates,
        lateral_step_range=(args.lateral_step_min, args.lateral_step_max),
        lateral_deviation_limit=args.lateral_deviation_limit,
        height_step_range=(args.height_step_min, args.height_step_max),
        height_range=(args.height_min, args.height_max),
    )
    baseline = evaluate(
        controller,
        evaluation_cases,
        evaluation_gates,
        seconds=args.evaluation_seconds,
        warmup_steps=args.observation_warmup_steps,
        camera=camera,
        hover_config=config,
        gate_config=gate_config,
    )
    print(json.dumps({"stage": "baseline", "path": path_manifest, "flight": baseline}), flush=True)
    first_gate_floor = max(0.0, float(baseline["first_gate_pass_rate"]) - 0.125)
    first_gate_paired_floor = max(
        0.0,
        float(baseline["first_gate_paired_pass_rate"]) - 0.125,
    )
    best_score = score(
        baseline,
        first_gate_floor=first_gate_floor,
        first_gate_paired_floor=first_gate_paired_floor,
    )
    best_path = args.output_dir / "best-controller.pt"
    save_checkpoint(
        best_path,
        controller,
        args=args,
        config=config,
        gate_config=gate_config,
        camera=camera,
        stage="baseline",
        update=0,
    )
    history: list[dict[str, Any]] = []
    started = perf_counter()
    rollout_seed = args.seed
    preservation_seed = args.seed + 1_000_000
    global_update = 0
    late_gate_cycle = 0
    for stage, updates, physics_mode in (
        ("recovery_paths", args.teacher_updates, "recovery"),
        ("native_paths", args.native_updates, "native"),
    ):
        if physics_mode == "native":
            selected_payload = torch.load(best_path, map_location="cpu", weights_only=True)
            controller.load_state_dict(selected_payload["controller"])
            optimizer = torch.optim.Adam(
                (controller.edge_magnitude,),
                lr=args.learning_rate,
            )
            print(
                json.dumps(
                    {
                        "stage": "native_paths_start",
                        "restored_selected_stage": selected_payload["stage"],
                        "restored_selected_update": selected_payload["update"],
                    }
                ),
                flush=True,
            )
        rollout_seed += 1
        start_gate = (
            1 + late_gate_cycle % (args.gates - 1)
            if args.balanced_late_gate_starts
            else 0
        )
        late_gate_cycle += int(args.balanced_late_gate_starts)
        rollout = new_rollout(
            controller,
            source_controller,
            pairs=args.training_pairs,
            seed=rollout_seed,
            device=device,
            config=config,
            camera=camera,
            gate_config=gate_config,
            warmup_steps=args.observation_warmup_steps,
            layout=args.layout,
            gate_count=args.gates,
            spacing_range=(args.spacing_min, args.spacing_max),
            lateral_step_range=(args.lateral_step_min, args.lateral_step_max),
            lateral_deviation_limit=args.lateral_deviation_limit,
            height_step_range=(args.height_step_min, args.height_step_max),
            height_range=(args.height_min, args.height_max),
            start_gate=start_gate,
        )
        preservation_rollout = new_rollout(
            controller,
            source_controller,
            pairs=args.training_pairs,
            seed=preservation_seed,
            device=device,
            config=config,
            camera=camera,
            gate_config=gate_config,
            warmup_steps=args.observation_warmup_steps,
            layout=args.layout,
            gate_count=args.gates,
            spacing_range=(args.spacing_min, args.spacing_max),
            lateral_step_range=(args.lateral_step_min, args.lateral_step_max),
            lateral_deviation_limit=args.lateral_deviation_limit,
            height_step_range=(args.height_step_min, args.height_step_max),
            height_range=(args.height_min, args.height_max),
        )
        for stage_update in range(1, updates + 1):
            global_update += 1
            reset = (
                rollout.age + args.unroll
                > round(
                    (
                        args.curriculum_window_seconds
                        if args.balanced_late_gate_starts
                        else args.evaluation_seconds
                    )
                    * POLICY_HZ
                )
                or bool(
                    (
                        rollout.failed
                        | (rollout.current == len(rollout.gates))
                    ).all()
                )
            )
            if reset:
                rollout_seed += 1
                start_gate = (
                    1 + late_gate_cycle % (args.gates - 1)
                    if args.balanced_late_gate_starts
                    else 0
                )
                late_gate_cycle += int(args.balanced_late_gate_starts)
                rollout = new_rollout(
                    controller,
                    source_controller,
                    pairs=args.training_pairs,
                    seed=rollout_seed,
                    device=device,
                    config=config,
                    camera=camera,
                    gate_config=gate_config,
                    warmup_steps=args.observation_warmup_steps,
                    layout=args.layout,
                    gate_count=args.gates,
                    spacing_range=(args.spacing_min, args.spacing_max),
                    lateral_step_range=(args.lateral_step_min, args.lateral_step_max),
                    lateral_deviation_limit=args.lateral_deviation_limit,
                    height_step_range=(args.height_step_min, args.height_step_max),
                    height_range=(args.height_min, args.height_max),
                    start_gate=start_gate,
                )
            preservation_reset = (
                preservation_rollout.age + args.unroll
                > round(args.evaluation_seconds * POLICY_HZ)
                or bool(
                    (
                        preservation_rollout.failed
                        | (preservation_rollout.current != 0)
                    ).all()
                )
            )
            if preservation_reset:
                preservation_seed += 1
                preservation_rollout = new_rollout(
                    controller,
                    source_controller,
                    pairs=args.training_pairs,
                    seed=preservation_seed,
                    device=device,
                    config=config,
                    camera=camera,
                    gate_config=gate_config,
                    warmup_steps=args.observation_warmup_steps,
                    layout=args.layout,
                    gate_count=args.gates,
                    spacing_range=(args.spacing_min, args.spacing_max),
                    lateral_step_range=(args.lateral_step_min, args.lateral_step_max),
                    lateral_deviation_limit=args.lateral_deviation_limit,
                    height_step_range=(args.height_step_min, args.height_step_max),
                    height_range=(args.height_min, args.height_max),
                )
            metrics, rollout, preservation_rollout = training_step(
                controller,
                source_controller,
                optimizer,
                rollout,
                preservation_rollout,
                source_edges,
                selected_mask,
                unroll=args.unroll,
                physics_mode=physics_mode,
                contrast_weight=args.contrast_weight,
                gate_one_preservation_weight=args.gate_one_preservation_weight,
                anchor_weight=args.anchor_weight,
                gradient_clip_norm=args.gradient_clip_norm,
                camera=camera,
                config=config,
                gate_config=gate_config,
                lesson=args.lesson,
            )
            evaluate_now = stage_update % args.evaluation_interval == 0 or stage_update == updates
            entry: dict[str, Any] = {
                "stage": stage,
                "stage_update": stage_update,
                "global_update": global_update,
                **metrics,
                "elapsed_seconds": perf_counter() - started,
            }
            if evaluate_now:
                flight = evaluate(
                    controller,
                    evaluation_cases,
                    evaluation_gates,
                    seconds=args.evaluation_seconds,
                    warmup_steps=args.observation_warmup_steps,
                    camera=camera,
                    hover_config=config,
                    gate_config=gate_config,
                )
                entry["native_flight"] = flight
                candidate_score = score(
                    flight,
                    first_gate_floor=first_gate_floor,
                    first_gate_paired_floor=first_gate_paired_floor,
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    save_checkpoint(
                        best_path,
                        controller,
                        args=args,
                        config=config,
                        gate_config=gate_config,
                        camera=camera,
                        stage=stage,
                        update=global_update,
                    )
            if stage_update == 1 or stage_update % 10 == 0 or evaluate_now:
                history.append(entry)
                print(json.dumps(entry), flush=True)

    best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    controller.load_state_dict(best_payload["controller"])
    best = evaluate(
        controller,
        evaluation_cases,
        evaluation_gates,
        seconds=args.evaluation_seconds,
        warmup_steps=args.observation_warmup_steps,
        camera=camera,
        hover_config=config,
        gate_config=gate_config,
    )
    changed = (controller.edge_magnitude.detach() - source_edges).abs()
    report = {
        "experiment": "pragmatic-full-native-gate-course-roll-path-v2",
        "purpose": "rapid behavioral proof of concept; not a formal promotion run",
        "actor_gate_index_or_pass_input": False,
        "continuous_state": ["MaleCNS recurrence", "forelegs", "sticks", "aircraft"],
        "layout": args.layout,
        "gate_count": args.gates,
        "course_distribution": {
            "spacing_metres": [args.spacing_min, args.spacing_max],
            "lateral_step_absolute_metres": [
                args.lateral_step_min,
                args.lateral_step_max,
            ],
            "lateral_deviation_limit_metres": args.lateral_deviation_limit,
            "height_step_absolute_metres": [args.height_step_min, args.height_step_max],
            "height_range_metres": [args.height_min, args.height_max],
        },
        "lesson": args.lesson,
        "gate_one_preservation_weight": args.gate_one_preservation_weight,
        "gate_one_retention_floor": {
            "pass_rate": first_gate_floor,
            "paired_pass_rate": first_gate_paired_floor,
        },
        "same_update_gate_one_replay": True,
        "balanced_late_gate_starts": args.balanced_late_gate_starts,
        "curriculum_window_seconds": args.curriculum_window_seconds,
        "last_hop_only": args.last_hop_only,
        "path": path_manifest,
        "baseline": baseline,
        "best": best,
        "best_score": list(best_score),
        "history": history,
        "parameter_audit": {
            "selected_edge_max_absolute_change": float(changed[selected_mask].max()),
            "unselected_edge_max_absolute_change": float(changed[~selected_mask].max()),
            "bias_and_time_constants_frozen": True,
        },
        "checkpoint": str(best_path),
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": perf_counter() - started,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "complete", **report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
