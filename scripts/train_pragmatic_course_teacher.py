#!/usr/bin/env python3
"""Distil a continuous multi-gate teacher into existing native visual/motor paths."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_hover as hover_train  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import evaluate, sample_two_gate_cases  # noqa: E402
from search_pragmatic_gate_course_es import selection_score  # noqa: E402
from train_pragmatic_gate_visual_roll_path import _minimum_distances, load_controller  # noqa: E402

from flydrone.course_teacher import (  # noqa: E402
    CoursePath,
    CourseTeacherConfig,
    course_teacher_motor,
)
from flydrone.gate import GateConfig, render_annular_gates_rgb  # noqa: E402
from flydrone.gate_course import classify_course_step  # noqa: E402
from flydrone.hover import DifferentiableQuad, ForelegStickPlant, HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-updates", type=int, default=300)
    parser.add_argument("--handoff-updates", type=int, default=200)
    parser.add_argument("--native-updates", type=int, default=0)
    parser.add_argument("--training-pairs", type=int, default=2)
    parser.add_argument("--unroll", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--bias-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--path-hops", type=int, default=5)
    parser.add_argument("--include-attitude-paths", action="store_true")
    parser.add_argument("--teacher-heading-mode", choices=("tangent", "world-x"), default="tangent")
    parser.add_argument("--development-pairs", type=int, default=16)
    parser.add_argument("--development-interval", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1_040_983)
    parser.add_argument("--development-seed", type=int, default=1_050_983)
    return parser.parse_args()


def action_imitation_loss(prediction, target, active, motor_scale, contrast_weight):
    """Direct motor error, optionally emphasizing differences within mirrored pairs.

    Weight 1 adds four extra units of differential-mode error to the direct loss;
    it is not neutral balancing. Native paired flights may also be at different phases.
    """
    error = ((prediction - target) / motor_scale).square()
    axis = error[active].mean(dim=0)
    paired = active.reshape(-1, 2).all(dim=1)
    residual = ((prediction - target) / motor_scale).reshape(-1, 2, 4)
    difference = residual[:, 1] - residual[:, 0]
    contrast = difference[paired].square().mean() if bool(paired.any()) else axis.sum() * 0
    return axis.mean() + contrast_weight * contrast, axis


def native_sensorimotor_mask(graph_path, hops, device, include_attitude=False):
    with np.load(graph_path) as graph:
        pre, post = graph["edge_pre"], graph["edge_post"]
        count = len(graph["node_ids"])
        visual, motors = graph["visual_node_indices"], graph["output_pool_indices"]
        attitude = graph["attitude_node_indices"]
        sensory = np.unique(np.concatenate((visual, attitude))) if include_attitude else visual
        upstream = _minimum_distances(count, pre, post, sensory, reverse=False)
        downstream = _minimum_distances(count, pre, post, motors, reverse=True)
        nodes = (upstream >= 0) & (downstream >= 0) & (upstream + downstream <= hops)
        edges = nodes[pre] & nodes[post] & (upstream[pre] + 1 + downstream[post] <= hops)
        nodes[motors] = True
        if not edges.any():
            raise ValueError("no native sensorimotor paths selected")
        return (
            torch.tensor(edges, device=device),
            torch.tensor(nodes, device=device),
            dict(
                selected_edges=int(edges.sum()),
                selected_nodes=int(nodes.sum()),
                visual_neurons=int(nodes[visual].sum()),
                attitude_neurons=int(nodes[attitude].sum()),
                sensor_sources="visual+existing roll/pitch" if include_attitude else "visual",
                motor_neurons=len(motors),
                hop_budget=hops,
            ),
        )


def main():
    args = parse_args()
    if (
        min(
            args.training_pairs,
            args.unroll,
            args.development_pairs,
            args.development_interval,
            args.learning_rate,
            args.seconds,
        )
        <= 0
    ):
        raise SystemExit("batch sizes, unroll, intervals and rates must be positive")
    if args.bias_learning_rate < 0.0:
        raise SystemExit("bias learning rate must be nonnegative (zero freezes biases)")
    if not np.isfinite(args.contrast_weight) or args.contrast_weight < 0.0:
        raise SystemExit("contrast weight must be finite and nonnegative")
    updates = args.teacher_updates + args.handoff_updates + args.native_updates
    if min(args.teacher_updates, args.handoff_updates, args.native_updates) < 0 or updates < 1:
        raise SystemExit("at least one nonnegative training stage is required")
    device = torch.device(args.device)
    controller, source = load_controller(args, device)
    edge_mask, node_mask, manifest = native_sensorimotor_mask(
        args.graph, args.path_hops, device, args.include_attitude_paths
    )
    controller.bias.requires_grad_(args.bias_learning_rate > 0.0)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_mask)
    if controller.bias.requires_grad:
        controller.bias.register_hook(lambda gradient: gradient * node_mask)
    controller.raw_time_constant.requires_grad_(False)
    initial_edge = controller.edge_magnitude.detach().clone()
    initial_bias = controller.bias.detach().clone()
    optimizer = torch.optim.Adam(
        [
            dict(params=[controller.edge_magnitude], lr=args.learning_rate),
            dict(params=[controller.bias], lr=args.bias_learning_rate),
        ]
    )
    config = HoverConfig(**source["hover_config"])
    gate_config = replace(GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    teacher_config = CourseTeacherConfig(heading_mode=args.teacher_heading_mode)
    camera = CameraSpec(
        width=source["image_resolution"][0],
        height=source["image_resolution"][1],
        horizontal_fov_degrees=source["camera_hfov_degrees"],
    )
    geometry = dict(
        layout="variable",
        gate_count=5,
        spacing_range=(0.9, 1.5),
        lateral_step_range=(0.0, 0.2),
        lateral_deviation_limit=0.5,
        height_step_range=(0.0, 0.08),
        height_range=(0.9, 1.3),
        yaw_jitter_degrees=15.0,
    )
    development = sample_two_gate_cases(
        args.development_pairs,
        seed=args.development_seed,
        device=device,
        hover_config=config,
        **geometry,
    )
    started = perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def assess():
        return evaluate(
            controller,
            *development,
            seconds=args.seconds,
            warmup_steps=10,
            camera=camera,
            hover_config=config,
            gate_config=gate_config,
        )

    baseline = assess()
    best = baseline
    first_floor = max(0.0, baseline["first_gate_pass_rate"] - 0.05)
    history = []

    def save(name, update, metrics):
        payload = dict(source)
        payload.update(
            experiment="native-continuous-course-imitation-v1",
            controller={
                key: value.detach().cpu() for key, value in controller.state_dict().items()
            },
            gate_config=vars(gate_config),
            course_geometry=geometry,
            teacher_config=vars(teacher_config),
            training_update=update,
            native_path_manifest=manifest,
            teacher_inputs_are_actor_inputs=False,
            course_rules="ordered-directed-all-annuli-v1",
            selection_metrics=metrics,
        )
        torch.save(payload, args.output_dir / name)

    def report():
        result = dict(
            experiment="native-continuous-course-imitation-v1",
            source_checkpoint=str(args.checkpoint),
            arguments={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            geometry=geometry,
            teacher_config=vars(teacher_config),
            native_path_manifest=manifest,
            source_development=baseline,
            selected_development=best,
            history=history,
            elapsed_seconds=perf_counter() - started,
            actor_inputs=["320x200 RGB", "roll", "pitch"],
            actor_outputs="native foreleg pools -> physical forelegs -> sticks",
            privileged_teacher_not_deployed=True,
        )
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    save("best-controller.pt", 0, baseline)
    print(json.dumps(dict(stage="baseline", metrics=baseline, paths=manifest)), flush=True)
    quad, legs = DifferentiableQuad(config).to(device), ForelegStickPlant(config).to(device)
    age = args.seconds
    rollout_number = 0
    current = torch.full((2 * args.training_pairs,), 5, device=device, dtype=torch.long)
    failed = torch.zeros_like(current, dtype=torch.bool)
    motor_scale = torch.tensor((0.02, 0.02, 0.01, 0.025), device=device)
    for update in range(1, updates + 1):
        if age >= args.seconds or bool((failed | (current == 5)).all()):
            cases, gates = sample_two_gate_cases(
                args.training_pairs,
                seed=args.seed + rollout_number,
                device=device,
                hover_config=config,
                **geometry,
            )
            rollout_number += 1
            state, sticks = cases.state, cases.sticks
            path = CoursePath.through_gates(state.position, gates)
            current.zero_()
            failed.zero_()
            neural = controller.initial_state(len(current), device=device, dtype=torch.float32)
            image = render_annular_gates_rgb(
                state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
            )
            with torch.no_grad():
                for _ in range(10):
                    _, neural = controller(image, state.euler[:, :2], neural)
            age = 0.0
        beta = min(1.0, max(0.0, (update - args.teacher_updates) / max(args.handoff_updates, 1)))
        optimizer.zero_grad(set_to_none=True)
        losses, per_axis = [], []
        for _ in range(args.unroll):
            image = render_annular_gates_rgb(
                state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
            )
            prediction, neural = controller(image, state.euler[:, :2], neural)
            with torch.no_grad():
                target = course_teacher_motor(state, path, config, teacher_config)
            active = ~failed & (current < 5)
            if bool(active.any()):
                action_loss, loss_axis = action_imitation_loss(
                    prediction, target, active, motor_scale, args.contrast_weight
                )
                losses.append(action_loss)
                per_axis.append(loss_axis.detach())
            with torch.no_grad():
                applied = (1.0 - beta) * target + beta * prediction.detach()
                for _ in range(2):
                    rc, sticks = legs(applied, sticks)
                    previous = state.position
                    state = quad(rc, state, cases.mass_scale)
                    events = classify_course_step(
                        previous, state.position, gates, current, gate_config
                    )
                    current = events.next_gate_index
                    # This monotonic-X teacher cannot recover an aperture missed
                    # behind it. End this lesson rather than label a later gate
                    # while the camera still advertises the missed current gate.
                    missed_target = (events.expected_forward_crossing & ~events.passed).any(dim=1)
                    failed |= events.failed | missed_target | ~hover_train.state_is_valid(state)
            age += 0.02
        if losses:
            anchor = 1.0e-3 * (
                ((controller.edge_magnitude[edge_mask] - initial_edge[edge_mask]) / 0.02)
                .square()
                .mean()
                + ((controller.bias[node_mask] - initial_bias[node_mask]) / 0.02).square().mean()
            )
            loss = torch.stack(losses).mean() + anchor
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite imitation loss; stopped before updating parameters")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                controller.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            controller.project_parameters()
            entry = dict(
                update=update,
                loss=float(loss.detach()),
                gradient_norm=float(gradient),
                axis_loss=torch.stack(per_axis).mean(dim=0).tolist(),
                beta=beta,
                rollout=rollout_number,
                age=age,
                elapsed_seconds=perf_counter() - started,
            )
        else:
            entry = dict(update=update, beta=beta, no_active_samples=True)
        neural = neural.detach()
        if update % args.development_interval == 0 or update == updates:
            metrics = assess()
            entry["development"] = metrics
            if selection_score(metrics, first_floor) > selection_score(best, first_floor):
                best = metrics
                save("best-controller.pt", update, metrics)
            save("latest-controller.pt", update, metrics)
            print(
                json.dumps(
                    dict(
                        stage="development",
                        update=update,
                        clean=metrics["clean_course_success_rate"],
                        first=metrics["first_gate_pass_rate"],
                        prefix=metrics["gates_before_failure_mean"],
                    )
                ),
                flush=True,
            )
        history.append(entry)
        if update == 1 or update % 10 == 0:
            print(
                json.dumps({key: value for key, value in entry.items() if key != "development"}),
                flush=True,
            )
            report()
    report()
    print(json.dumps(dict(stage="complete", selected=best)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
