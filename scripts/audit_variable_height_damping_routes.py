#!/usr/bin/env python3
"""Preflight a restricted native route for correctly signed visual vertical damping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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
import train_variable_height_bridge as bridge  # noqa: E402
import train_variable_height_hover as hover  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402

from flydrone.connectome_data import ANNOTATIONS_FILE, _read_annotations  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    HoverConfig,
    QuadState,
)
from flydrone.variable_hover import sample_marker_pairs  # noqa: E402
from flydrone.visual_hover import render_visual_hover_scene, sample_visual_scenes  # noqa: E402

MASK_FAMILIES = ("edge_magnitude", "bias")
MASK_STEP_FAMILY_RMS_CAP = 2.0e-5
MASK_SOURCE_METRIC_RADIUS = 5.0e-4
MINIMUM_MOTION_NRMSE_IMPROVEMENT = 1.0e-4
MAX_SOURCE_COMMON_RMS = 0.0025
MAX_SOURCE_COMMON_ABSOLUTE = 0.005
HEIGHT_CONTRAST_RATIO_RANGE = (0.90, 1.10)
BACKTRACK_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
MOTION_MOTOR_SCALE = 0.05


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/damping-route-preflight-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=250_917)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--constraint-batch-size", type=int, default=8)
    parser.add_argument("--unroll", type=int, default=25)
    parser.add_argument("--constraint-prefix-steps", type=int, default=50)
    parser.add_argument("--history-steps", type=int, default=25)
    return parser


def parse_args() -> argparse.Namespace:
    return argument_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.raw_dir / ANNOTATIONS_FILE, args.checkpoint):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    if args.batch_size < 4 or args.batch_size % 4:
        raise SystemExit("--batch-size must be a positive multiple of four")
    if args.constraint_batch_size < 4 or args.constraint_batch_size % 4:
        raise SystemExit("--constraint-batch-size must be a positive multiple of four")
    if min(args.unroll, args.history_steps) < 11 or args.constraint_prefix_steps < 0:
        raise SystemExit("unroll/history must be at least 11 and prefix nonnegative")


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def route_mask_arrays(
    edge_pre: np.ndarray,
    edge_post: np.ndarray,
    superclass: np.ndarray,
    throttle_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return exact <=2-edge descending/VNC-intrinsic routes to throttle motors."""

    text = superclass.astype(str)
    start = (np.char.find(text, "descending") >= 0) | (text == "vnc_intrinsic")
    throttle = np.zeros(len(text), dtype=bool)
    throttle[throttle_nodes] = True
    all_motor = np.char.startswith(text, "vnc_motor")
    direct = start[edge_pre] & throttle[edge_post]
    from_start = np.zeros(len(text), dtype=bool)
    from_start[edge_post[start[edge_pre]]] = True
    to_throttle = np.zeros(len(text), dtype=bool)
    to_throttle[edge_pre[throttle[edge_post]]] = True
    intermediate = from_start & to_throttle & ~all_motor & ~throttle
    first_hop = start[edge_pre] & intermediate[edge_post]
    second_hop = intermediate[edge_pre] & throttle[edge_post]
    route_edges = np.flatnonzero(direct | first_hop | second_hop)
    route_biases = np.flatnonzero(intermediate)
    return (
        route_edges,
        route_biases,
        {
            "start_nodes": int(start.sum()),
            "throttle_motor_nodes": int(throttle.sum()),
            "all_motor_nodes_excluded_as_intermediates": int(all_motor.sum()),
            "direct_edges": int(direct.sum()),
            "first_hop_edges": int(first_hop.sum()),
            "second_hop_edges": int(second_hop.sum()),
            "intermediate_nodes": int(intermediate.sum()),
        },
    )


def build_route_mask(
    graph_path: Path, raw_dir: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    graph = np.load(graph_path)
    annotations = _read_annotations(raw_dir / ANNOTATIONS_FILE)
    rows = {int(body): index for index, body in enumerate(annotations["bodyId"])}
    superclass = np.asarray(
        [str(annotations["superclass"][rows[int(body)]]) for body in graph["node_ids"]]
    )
    offsets, pools = graph["output_pool_offsets"], graph["output_pool_indices"]
    throttle = pools[offsets[6] : offsets[8]]
    edges, biases, counts = route_mask_arrays(
        graph["edge_pre"], graph["edge_post"], superclass, throttle
    )
    intermediate_superclass, intermediate_counts = np.unique(superclass[biases], return_counts=True)
    manifest = {
        "algorithm": (
            "all existing direct or two-edge paths from a descending or vnc_intrinsic "
            "node to a throttle-pool motor; exclude every vnc_motor as an intermediate"
        ),
        "counts": counts,
        "selected_edge_magnitudes": int(len(edges)),
        "selected_intermediate_biases": int(len(biases)),
        "edge_indices_sha256": array_sha256(edges.astype(np.int64)),
        "bias_node_indices_sha256": array_sha256(biases.astype(np.int64)),
        "bias_body_ids_sha256": array_sha256(graph["node_ids"][biases].astype(np.int64)),
        "intermediate_superclass_counts": {
            str(name): int(count)
            for name, count in zip(intermediate_superclass, intermediate_counts, strict=True)
        },
        "edge_indices": edges.tolist(),
        "bias_node_indices": biases.tolist(),
        "bias_body_ids": graph["node_ids"][biases].tolist(),
        "throttle_motor_body_ids": graph["node_ids"][throttle].tolist(),
    }
    return edges, biases, manifest


def masked_family_rms(values: dict[str, Tensor]) -> dict[str, float]:
    return {
        name: float(values[name].detach().square().mean().sqrt()) if values[name].numel() else 0.0
        for name in MASK_FAMILIES
    }


def masked_metric_norm(values: dict[str, Tensor]) -> float:
    rms = masked_family_rms(values)
    return sum(value * value for value in rms.values()) ** 0.5


def cap_masked_displacement(
    displacement: dict[str, Tensor], cap: float
) -> tuple[dict[str, Tensor], float]:
    largest = max(masked_family_rms(displacement).values())
    scale = min(1.0, cap / max(largest, 1.0e-30))
    return {name: scale * value for name, value in displacement.items()}, scale


def project_masked_displacement(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    residual: Tensor,
    *,
    rtol: float = 1.0e-6,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Project in the sum-of-selected-family-mean-squares metric."""

    if not rows or residual.shape != (len(rows),):
        raise ValueError("projection requires one residual per nonempty row")
    sizes = {name: displacement[name].numel() for name in MASK_FAMILIES}
    row_count = len(rows)
    gram = torch.empty(row_count, row_count, device=residual.device, dtype=torch.float64)
    j_delta = torch.empty(row_count, device=residual.device, dtype=torch.float64)
    for first, row in enumerate(rows):
        j_delta[first] = sum(
            (row[name] * displacement[name]).sum() for name in MASK_FAMILIES
        ).double()
        for second in range(first + 1):
            other = rows[second]
            value = sum(sizes[name] * (row[name] * other[name]).sum() for name in MASK_FAMILIES)
            gram[first, second] = value.double()
            gram[second, first] = value.double()
    diagonal = gram.diagonal().clamp_min(0.0).sqrt()
    active = diagonal > 1.0e-12
    if not bool(active.any()):
        raise RuntimeError("all masked common-output rows are numerically zero")
    active_indices = active.nonzero(as_tuple=False)[:, 0]
    row_scale = diagonal[active]
    normalized = gram[active][:, active] / (row_scale[:, None] * row_scale[None])
    singular = torch.linalg.svdvals(normalized)
    rank = int((singular > rtol * singular.max()).sum())
    right_hand_side = (j_delta + residual.double())[active] / row_scale
    coefficient = torch.linalg.pinv(normalized, rtol=rtol) @ right_hand_side
    coefficient = coefficient / row_scale
    projected = {name: value.detach().clone() for name, value in displacement.items()}
    for value, row_index in zip(coefficient, active_indices, strict=True):
        row = rows[int(row_index)]
        for name in MASK_FAMILIES:
            projected[name].add_(row[name], alpha=-float(value) * sizes[name])
    after = residual.double().clone()
    for index, row in enumerate(rows):
        after[index] += sum((row[name] * projected[name]).sum() for name in MASK_FAMILIES).double()
    return projected, {
        "rows": row_count,
        "active_rows": int(active.sum()),
        "retained_rank": rank,
        "normalized_gram_singular_values": singular.detach().cpu().tolist(),
        "residual_before": residual.detach().cpu().tolist(),
        "projected_linearized_residual_max_absolute": float(after.abs().max()),
    }


def variable_prefix(
    controller: ConnectomeController,
    state: QuadState,
    marker: Tensor,
    scene: Any,
    durations: Tensor,
) -> Tensor:
    neural = controller.initial_state(len(marker), device=marker.device, dtype=marker.dtype)
    image = render_visual_hover_scene(state, marker, scene=scene)
    for step in range(int(durations.max())):
        _, candidate = controller(image, state.euler[:, :2], neural)
        active = step < durations
        neural = torch.where(active[:, None], candidate, neural)
    return neural


def motion_terms(
    student: ConnectomeController,
    prefix_controller: ConnectomeController,
    *,
    batch: int,
    seed: int,
    history_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> tuple[Tensor, dict[str, Any]]:
    responsibility.seed_everything(seed)
    device = student.edge_magnitude.device
    signed_speed = responsibility.balanced_signed_values(batch, (0.15, 0.30), device=device)
    marker_error = responsibility.balanced_signed_values(batch, (0.03, 0.06), device=device)
    state, height = responsibility.base_state(batch, device=device, config=config)
    scene = sample_visual_scenes(
        batch,
        device=device,
        held_out_combinations=False,
        all_style_combinations=True,
    )
    marker = height + marker_error
    prefix_choices = torch.tensor((5, 25, 50, 75), device=device)
    durations = prefix_choices[torch.arange(batch, device=device) % len(prefix_choices)]
    durations = durations[torch.randperm(batch, device=device)]
    with torch.no_grad():
        prefix = variable_prefix(prefix_controller, state, marker, scene, durations)
    recurrent = torch.cat((prefix.clone(), prefix.clone()))
    attitude = torch.cat((state.euler[:, :2], state.euler[:, :2]))
    duration = history_steps / policy_hz
    motor = torch.zeros(2 * batch, 4, device=device)
    for step in range(history_steps):
        unit = (step + 1) / history_steps
        offset = signed_speed * duration * (unit**3 - unit**2)
        offset -= signed_speed.sign() * 0.055 * math.sin(math.pi * unit) ** 2
        if step + 1 == history_steps:
            offset = torch.zeros_like(offset)
        state_a = responsibility.state_at_height(state, height + offset)
        state_b = responsibility.state_at_height(state, height - offset)
        image_a = render_visual_hover_scene(state_a, marker, scene=scene)
        image_b = render_visual_hover_scene(state_b, marker, scene=scene)
        motor, recurrent = student(torch.cat((image_a, image_b)), attitude, recurrent)
    prediction = motor[:batch, 3] - motor[batch:, 3]
    endpoint_a = responsibility.state_at_height(state, height, vertical_velocity=signed_speed)
    endpoint_b = responsibility.state_at_height(state, height, vertical_velocity=-signed_speed)
    target = responsibility.teacher_throttle_contrast(
        endpoint_a, endpoint_b, marker, marker, config
    )
    error = prediction - target
    loss = (error / MOTION_MOTOR_SCALE).square().mean()
    target_squared = target.square().sum().clamp_min(1.0e-12)
    pair_common = 0.5 * (motor[:batch, 3] + motor[batch:, 3])
    desired_a = pair_common + 0.5 * target
    desired_b = pair_common - 0.5 * target
    return loss, {
        "fixed_scale_nrmse": float(loss.detach().sqrt()),
        "relative_target_nrmse": float(
            error.detach().square().mean().sqrt()
            / target.detach().square().mean().sqrt().clamp_min(1.0e-12)
        ),
        "target_rms": float(target.detach().square().mean().sqrt()),
        "prediction_rms": float(prediction.detach().square().mean().sqrt()),
        "teacher_aligned_gain": float((prediction.detach() * target).sum() / target_squared),
        "correct_sign_fraction": float(((prediction.detach() * target) > 1.0e-6).float().mean()),
        "pair_common_throttle_rms": float(pair_common.detach().square().mean().sqrt()),
        "desired_branch_output_max_absolute": float(
            torch.cat((desired_a, desired_b)).detach().abs().max()
        ),
        "desired_branch_output_bound_margin": float(
            1.0 - torch.cat((desired_a, desired_b)).detach().abs().max()
        ),
        "prefix_steps": durations.detach().cpu().tolist(),
        "signed_speed_metres_per_second": signed_speed.detach().cpu().tolist(),
        "final_marker_error_metres": marker_error.detach().cpu().tolist(),
        "target_contrast": target.detach().cpu().tolist(),
        "predicted_contrast": prediction.detach().cpu().tolist(),
    }


def masked_gradient(
    loss: Tensor,
    controller: ConnectomeController,
    edge_indices: Tensor,
    bias_indices: Tensor,
    *,
    retain_graph: bool = False,
) -> dict[str, Tensor]:
    edge_gradient, bias_gradient = torch.autograd.grad(
        loss,
        (controller.edge_magnitude, controller.bias),
        retain_graph=retain_graph,
    )
    return {
        "edge_magnitude": edge_gradient[edge_indices].detach().clone(),
        "bias": bias_gradient[bias_indices].detach().clone(),
    }


def common_constraint_rows(
    student: ConnectomeController,
    source: ConnectomeController,
    edge_indices: Tensor,
    bias_indices: Tensor,
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[list[dict[str, Tensor]], Tensor, dict[str, Any]]:
    rows, residuals, labels = [], [], []
    metrics = {}
    for amplitude_index, (amplitude, half_step) in enumerate(bridge.PAIR_HALF_STEPS.items()):
        constraints, _, result = trust.fixed_pair_window_terms(
            student,
            source,
            half_step=half_step,
            batch=args.constraint_batch_size,
            seed=args.seed + 20_000 + amplitude_index,
            prefix_steps=args.constraint_prefix_steps,
            unroll=args.unroll,
            physics_steps=physics_steps,
            device=device,
            config=config,
        )
        metrics[amplitude] = result
        for item in range(args.constraint_batch_size):
            row = masked_gradient(
                constraints[item],
                student,
                edge_indices,
                bias_indices,
                retain_graph=item + 1 < args.constraint_batch_size,
            )
            rows.append(row)
            residuals.append(constraints[item].detach())
            labels.append(f"{amplitude}_slot_{item}")
    return rows, torch.stack(residuals), {"labels": labels, "amplitudes": metrics}


@torch.no_grad()
def set_masked_trial(
    controller: ConnectomeController,
    base: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
    displacement: dict[str, Tensor],
    scale: float,
) -> dict[str, Tensor]:
    controller.edge_magnitude.copy_(base["edge_magnitude"])
    controller.bias.copy_(base["bias"])
    controller.raw_time_constant.copy_(base["raw_time_constant"])
    controller.edge_magnitude[edge_indices] += scale * displacement["edge_magnitude"]
    controller.bias[bias_indices] += scale * displacement["bias"]
    controller.project_parameters()
    return {
        "edge_magnitude": controller.edge_magnitude[edge_indices]
        - base["edge_magnitude"][edge_indices],
        "bias": controller.bias[bias_indices] - base["bias"][bias_indices],
    }


@torch.no_grad()
def height_response_summary(
    student: ConnectomeController,
    source: ConnectomeController,
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    result = {}
    for amplitude_index, (amplitude, half_step) in enumerate(bridge.PAIR_HALF_STEPS.items()):
        hover.seed_everything(args.seed + 31_000 + amplitude_index)
        pairs = sample_marker_pairs(
            args.constraint_batch_size,
            device=device,
            held_out=False,
            half_step_metres=half_step,
        )
        state, stick_state, scene, prefix_marker = bridge._initial_pair_state(
            pairs,
            randomized_scene=True,
            all_style_combinations=True,
            device=device,
            config=config,
        )
        state, _, student_neural, source_neural, valid = bridge._dual_prefix(
            student,
            source,
            state,
            stick_state,
            prefix_marker,
            scene,
            steps=args.constraint_prefix_steps,
            physics_steps=physics_steps,
            config=config,
        )
        start = min(args.unroll - 1, 10)
        student_a, student_b = bridge._branch_outputs(
            student,
            state,
            student_neural,
            scene,
            pairs.marker_a,
            pairs.marker_b,
            response_steps=args.unroll,
            loss_start=start,
        )
        source_a, source_b = bridge._branch_outputs(
            source,
            state,
            source_neural,
            scene,
            pairs.marker_a,
            pairs.marker_b,
            response_steps=args.unroll,
            loss_start=start,
        )
        target_a = hover.teacher_motor(state, pairs.marker_a, config)
        target_b = hover.teacher_motor(state, pairs.marker_b, config)
        sign = (target_a[:, 3] - target_b[:, 3]).sign()[None]
        valid_steps = valid[None].expand(student_a.shape[0], -1)
        weight = valid_steps.to(student_a.dtype)
        student_aligned = (
            (student_a[:, :, 3] - student_b[:, :, 3]) * sign * weight
        ).sum() / weight.sum().clamp_min(1.0)
        source_aligned = (
            (source_a[:, :, 3] - source_b[:, :, 3]) * sign * weight
        ).sum() / weight.sum().clamp_min(1.0)
        result[amplitude] = {
            "student_aligned_contrast": float(student_aligned),
            "source_aligned_contrast": float(source_aligned),
            "student_to_source_ratio": float(
                student_aligned / source_aligned.abs().clamp_min(1.0e-8)
            ),
            "valid_fraction": float(valid.float().mean()),
        }
    return result


def masked_source_delta(
    controller: ConnectomeController,
    source_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
) -> dict[str, Tensor]:
    return {
        "edge_magnitude": controller.edge_magnitude[edge_indices]
        - source_parameters["edge_magnitude"][edge_indices],
        "bias": controller.bias[bias_indices] - source_parameters["bias"][bias_indices],
    }


@torch.no_grad()
def evaluate_trial(
    student: ConnectomeController,
    source: ConnectomeController,
    previous: ConnectomeController,
    source_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
    baseline_objective_nrmse: float,
) -> dict[str, Any]:
    objective_loss, objective_motion = motion_terms(
        student,
        source,
        batch=args.batch_size,
        seed=args.seed + 10_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    _, complete_motion = motion_terms(
        student,
        student,
        batch=args.batch_size,
        seed=args.seed + 10_001,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    height = height_response_summary(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    functional = trust.functional_trust_checks(
        student,
        source,
        previous,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
        source_parameters=source_parameters,
        previous_parameters=source_parameters,
    )
    masked_delta = masked_source_delta(student, source_parameters, edge_indices, bias_indices)
    common_rms = max(value["common_error_rms"] for value in functional["source_pair"].values())
    common_max = max(
        value["common_error_max_absolute"] for value in functional["source_pair"].values()
    )
    height_ok = all(
        HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    improvement = baseline_objective_nrmse - objective_motion["fixed_scale_nrmse"]
    reasons = []
    if improvement < MINIMUM_MOTION_NRMSE_IMPROVEMENT:
        reasons.append("motion NRMSE did not improve above the replay floor")
    if not functional["pass"]:
        reasons.append("existing complete-replay trust checks failed")
    if common_rms > MAX_SOURCE_COMMON_RMS:
        reasons.append("source-global common throttle RMS")
    if common_max > MAX_SOURCE_COMMON_ABSOLUTE:
        reasons.append("source-global common throttle maximum")
    if not height_ok:
        reasons.append("height contrast left the source-relative 10% band")
    if masked_metric_norm(masked_delta) > MASK_SOURCE_METRIC_RADIUS * (1.0 + 1.0e-5):
        reasons.append("masked source metric radius")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "objective_motion": objective_motion,
        "objective_loss": float(objective_loss),
        "objective_nrmse_improvement": improvement,
        "complete_zero_state_motion": complete_motion,
        "height_response": height,
        "functional_trust": functional,
        "source_common_throttle_rms": common_rms,
        "source_common_throttle_max_absolute": common_max,
        "masked_source_family_rms": masked_family_rms(masked_delta),
        "masked_source_metric_norm": masked_metric_norm(masked_delta),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy frequency must divide the 100 Hz physics rate")
    physics_steps = physics_hz // args.policy_hz
    graph_sha256 = responsibility.file_sha256(args.graph)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != graph_sha256:
        raise SystemExit("checkpoint and graph hashes do not match")
    if int(checkpoint.get("policy_hz", args.policy_hz)) != args.policy_hz:
        raise SystemExit("checkpoint and requested policy frequencies do not match")
    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    previous = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    for controller in (student, source, previous):
        controller.load_state_dict(checkpoint["controller"])
    source.eval().requires_grad_(False)
    previous.eval().requires_grad_(False)
    student.eval()
    student.raw_time_constant.requires_grad_(False)
    edge_indices_np, bias_indices_np, manifest = build_route_mask(args.graph, args.raw_dir)
    edge_indices = torch.from_numpy(edge_indices_np).to(device)
    bias_indices = torch.from_numpy(bias_indices_np).to(device)
    source_parameters = trust.clone_parameters(source)
    base_parameters = trust.clone_parameters(student)
    started = perf_counter()

    baseline_loss, baseline_motion = motion_terms(
        student,
        source,
        batch=args.batch_size,
        seed=args.seed + 10_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    damping_gradient = masked_gradient(baseline_loss, student, edge_indices, bias_indices)
    raw = {name: -damping_gradient[name] * damping_gradient[name].numel() for name in MASK_FAMILIES}
    raw, raw_cap_scale = cap_masked_displacement(raw, MASK_STEP_FAMILY_RMS_CAP)
    rows, residual, common_report = common_constraint_rows(
        student,
        source,
        edge_indices,
        bias_indices,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    projected, projection = project_masked_displacement(raw, rows, residual)
    projected, projected_cap_scale = cap_masked_displacement(projected, MASK_STEP_FAMILY_RMS_CAP)
    derivative = float(
        sum((damping_gradient[name] * projected[name]).sum() for name in MASK_FAMILIES)
    )

    trials = []
    accepted_scale = None
    for scale in BACKTRACK_SCALES:
        actual_delta = set_masked_trial(
            student,
            base_parameters,
            edge_indices,
            bias_indices,
            projected,
            scale,
        )
        trial = evaluate_trial(
            student,
            source,
            previous,
            source_parameters,
            edge_indices,
            bias_indices,
            args=args,
            physics_steps=physics_steps,
            device=device,
            config=config,
            baseline_objective_nrmse=baseline_motion["fixed_scale_nrmse"],
        )
        trial.update(
            {
                "scale": scale,
                "masked_actual_family_rms": masked_family_rms(actual_delta),
                "predicted_first_order_loss_change": scale * derivative,
                "actual_loss_change": trial["objective_loss"] - float(baseline_loss.detach()),
            }
        )
        trials.append(trial)
        if trial["pass"]:
            accepted_scale = scale
            break
    trust.load_parameters(student, base_parameters)
    untouched_max = max(
        float((getattr(student, name).detach() - base_parameters[name]).abs().max())
        for name in trust.PARAMETER_FAMILIES
    )
    report = {
        "experiment": "variable-height-native-damping-route-preflight-v1",
        "status": "diagnostic_only_no_retained_parameter_changes",
        "pass": accepted_scale is not None and derivative < 0.0,
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
        },
        "actor_contract_unchanged": True,
        "mask": manifest,
        "metric": {
            "definition": "sqrt(mean(selected edge delta^2) + mean(selected bias delta^2))",
            "per_family_step_rms_cap": MASK_STEP_FAMILY_RMS_CAP,
            "source_metric_radius_for_later_training": MASK_SOURCE_METRIC_RADIUS,
            "motor_biases_trainable": False,
            "time_constants_trainable": False,
            "transmitter_signs_trainable": False,
            "topology_trainable": False,
        },
        "protocol": {
            "motion_bank_pairs": args.batch_size,
            "speeds_metres_per_second": [0.15, 0.30],
            "balanced_final_marker_error_metres": [0.03, 0.06],
            "history_steps": args.history_steps,
            "policy_hz": args.policy_hz,
            "prefix_step_choices": [5, 25, 50, 75],
            "constraint_pairs_per_height_amplitude": args.constraint_batch_size,
            "backtrack_scales": list(BACKTRACK_SCALES),
        },
        "thresholds": {
            "minimum_motion_nrmse_improvement": MINIMUM_MOTION_NRMSE_IMPROVEMENT,
            "maximum_source_common_throttle_rms_native_units": MAX_SOURCE_COMMON_RMS,
            "maximum_source_common_throttle_absolute_native_units": (MAX_SOURCE_COMMON_ABSOLUTE),
            "height_contrast_source_ratio_range": list(HEIGHT_CONTRAST_RATIO_RANGE),
            "existing_dynamic_and_legacy_nrmse_limit": trust.SOURCE_FUNCTION_NRMSE_LIMIT,
        },
        "target_feasibility": {
            "desired_branch_output_bound": [-1.0, 1.0],
            "desired_branch_output_max_absolute": baseline_motion[
                "desired_branch_output_max_absolute"
            ],
            "available_margin": baseline_motion["desired_branch_output_bound_margin"],
            "pass": baseline_motion["desired_branch_output_bound_margin"] > 0.0,
        },
        "baseline_motion": baseline_motion,
        "direction": {
            "raw_cap_scale": raw_cap_scale,
            "raw_family_rms": masked_family_rms(raw),
            "projected_cap_scale": projected_cap_scale,
            "projected_family_rms": masked_family_rms(projected),
            "first_order_loss_derivative": derivative,
            "common_constraints": common_report,
            "projection": projection,
        },
        "accepted_scale": accepted_scale,
        "trials": trials,
        "parameters_restored_max_absolute_error": untouched_max,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A passing local direction establishes restricted replay feasibility only. It does "
            "not identify a biological damping module, prove closed-loop benefit, or promote a "
            "controller. Wrong-signed response may be lagged position feedback."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": report["pass"],
                "accepted_scale": accepted_scale,
                "baseline_motion_nrmse": baseline_motion["fixed_scale_nrmse"],
                "best_motion_nrmse": min(
                    trial["objective_motion"]["fixed_scale_nrmse"] for trial in trials
                ),
                "parameters_restored_max_absolute_error": untouched_max,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
