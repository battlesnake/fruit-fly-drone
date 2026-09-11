#!/usr/bin/env python3
"""Tune native visual-to-roll paths for two uninterrupted role-coloured gates."""

from __future__ import annotations

import argparse
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
    gates: tuple[AnnularGate, AnnularGate]
    mass_scale: Tensor
    neural: Tensor
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
    parser.add_argument("--teacher-updates", type=int, default=250)
    parser.add_argument("--native-updates", type=int, default=250)
    parser.add_argument("--training-pairs", type=int, default=2)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--contrast-weight", type=float, default=4.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--observation-warmup-steps", type=int, default=10)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--evaluation-pairs", type=int, default=8)
    parser.add_argument("--evaluation-seconds", type=float, default=22.0)
    parser.add_argument("--layout", choices=("aligned", "s-turn"), default="aligned")
    parser.add_argument("--seed", type=int, default=640_983)
    parser.add_argument("--evaluation-seed", type=int, default=650_983)
    return parser.parse_args()


def new_rollout(
    controller: ConnectomeController,
    *,
    pairs: int,
    seed: int,
    device: torch.device,
    config: HoverConfig,
    camera: CameraSpec,
    gate_config: GateConfig,
    warmup_steps: int,
    layout: str,
) -> CourseRollout:
    cases, gates = sample_two_gate_cases(
        pairs,
        seed=seed,
        device=device,
        hover_config=config,
        layout=layout,
    )
    current = torch.zeros(2 * pairs, dtype=torch.long, device=device)
    neural = controller.initial_state(2 * pairs, device=device, dtype=torch.float32)
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
    return CourseRollout(
        state=cases.state,
        sticks=cases.sticks,
        gates=gates,
        mass_scale=cases.mass_scale,
        neural=neural,
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
    optimizer: torch.optim.Optimizer,
    rollout: CourseRollout,
    source_edges: Tensor,
    selected_mask: Tensor,
    *,
    unroll: int,
    native_physics: bool,
    contrast_weight: float,
    anchor_weight: float,
    gradient_clip_norm: float,
    camera: CameraSpec,
    config: HoverConfig,
    gate_config: GateConfig,
) -> tuple[dict[str, float], CourseRollout]:
    quad = DifferentiableQuad(config).to(rollout.neural.device)
    stick_plant = ForelegStickPlant(config).to(rollout.neural.device)
    direct_losses = []
    contrast_losses = []
    target_rms = []
    prediction_rms = []
    for _ in range(unroll):
        active = (rollout.current < len(rollout.gates)) & ~rollout.failed
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
        selected = active_gate(rollout.gates, rollout.current)
        target = teacher_motor(rollout.state, selected, config, mode="staged")
        direct_error = ((prediction[:, 0] - target[:, 0]) / 0.10).square()
        direct_losses.append(masked_mean(direct_error, active))
        prediction_pair = prediction[:, 0].reshape(-1, 2)
        target_pair = target[:, 0].reshape(-1, 2)
        pair_active = active.reshape(-1, 2).all(dim=1)
        contrast_error = (
            (prediction_pair[:, 1] - prediction_pair[:, 0])
            - (target_pair[:, 1] - target_pair[:, 0])
        ).div(0.15).square()
        contrast_losses.append(masked_mean(contrast_error, pair_active))
        target_rms.append(masked_mean(target[:, 0].square(), active).sqrt().detach())
        prediction_rms.append(masked_mean(prediction[:, 0].square(), active).sqrt().detach())
        applied_motor = prediction.detach() if native_physics else target
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

    direct = torch.stack(direct_losses).mean()
    contrast = torch.stack(contrast_losses).mean()
    anchor = (
        ((controller.edge_magnitude[selected_mask] - source_edges[selected_mask]) / 0.25)
        .square()
        .mean()
    )
    loss = direct + contrast_weight * contrast + anchor_weight * anchor
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        (controller.edge_magnitude,), gradient_clip_norm
    )
    optimizer.step()
    controller.project_parameters()
    rollout.neural = rollout.neural.detach()
    return (
        {
            "loss": float(loss.detach()),
            "roll_direct": float(direct.detach()),
            "roll_contrast": float(contrast.detach()),
            "selected_edge_anchor": float(anchor.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "target_roll_motor_rms": float(torch.stack(target_rms).mean()),
            "predicted_roll_motor_rms": float(torch.stack(prediction_rms).mean()),
            "rollout_age_seconds": rollout.age / POLICY_HZ,
            "rollout_completed_fraction": float(
                (rollout.current == len(rollout.gates)).float().mean()
            ),
            "rollout_failed_fraction": float(rollout.failed.float().mean()),
        },
        rollout,
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
            "experiment": "pragmatic-full-native-two-gate-roll-path-v1",
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
        },
        path,
    )


def score(metrics: dict[str, object]) -> tuple[float, ...]:
    return (
        float(metrics["both_gates_paired_pass_rate"]),
        min(
            float(metrics["both_gates_negative_course_pass_rate"]),
            float(metrics["both_gates_positive_course_pass_rate"]),
        ),
        float(metrics["both_gates_pass_rate"]),
        float(metrics["first_gate_paired_pass_rate"]),
        float(metrics["first_gate_pass_rate"]),
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    responsibility.seed_everything(args.seed)
    controller, source_payload = load_controller(args, device)
    selected_mask, path_manifest = visual_roll_path_mask(
        args.graph,
        hop_budget=args.path_hop_budget,
        device=device,
    )
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
    best_score = score(baseline)
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
    global_update = 0
    for stage, updates, native_physics in (
        ("teacher_paths", args.teacher_updates, False),
        ("native_paths", args.native_updates, True),
    ):
        if native_physics:
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
        rollout = new_rollout(
            controller,
            pairs=args.training_pairs,
            seed=rollout_seed,
            device=device,
            config=config,
            camera=camera,
            gate_config=gate_config,
            warmup_steps=args.observation_warmup_steps,
            layout=args.layout,
        )
        for stage_update in range(1, updates + 1):
            global_update += 1
            reset = (
                rollout.age + args.unroll > round(args.evaluation_seconds * POLICY_HZ)
                or bool(
                    (
                        rollout.failed
                        | (rollout.current == len(rollout.gates))
                    ).all()
                )
            )
            if reset:
                rollout_seed += 1
                rollout = new_rollout(
                    controller,
                    pairs=args.training_pairs,
                    seed=rollout_seed,
                    device=device,
                    config=config,
                    camera=camera,
                    gate_config=gate_config,
                    warmup_steps=args.observation_warmup_steps,
                    layout=args.layout,
                )
            metrics, rollout = training_step(
                controller,
                optimizer,
                rollout,
                source_edges,
                selected_mask,
                unroll=args.unroll,
                native_physics=native_physics,
                contrast_weight=args.contrast_weight,
                anchor_weight=args.anchor_weight,
                gradient_clip_norm=args.gradient_clip_norm,
                camera=camera,
                config=config,
                gate_config=gate_config,
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
                candidate_score = score(flight)
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
        "experiment": "pragmatic-full-native-two-gate-roll-path-v1",
        "purpose": "rapid behavioral proof of concept; not a formal promotion run",
        "actor_gate_index_or_pass_input": False,
        "continuous_state": ["MaleCNS recurrence", "forelegs", "sticks", "aircraft"],
        "layout": args.layout,
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
