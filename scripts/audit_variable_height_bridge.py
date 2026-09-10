#!/usr/bin/env python3
"""Audit bridge lesson gradients and native marker-signal propagation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_bridge as bridge  # noqa: E402
import train_variable_height_hover as hover  # noqa: E402

from flydrone.connectome_data import ANNOTATIONS_FILE, _read_annotations  # noqa: E402
from flydrone.hover import ConnectomeController, DifferentiableQuad, HoverConfig  # noqa: E402
from flydrone.variable_hover import sample_marker_pairs  # noqa: E402
from flydrone.visual_hover import (  # noqa: E402
    TEXTURE_PHASE_CENTRES,
    VisualScene,
    render_visual_hover_scene,
    sample_visual_scenes,
)

PARAMETER_FAMILIES = ("edge_magnitude", "bias", "raw_time_constant")
LESSONS = (
    "new_paired_contrast",
    "legacy_response_preservation",
    "common_throttle_preservation",
    "dynamic_rpy_preservation",
    "parameter_anchor",
)
SIGNAL_STEPS = (5, 13, 25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0"
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/bridge-all-001/update-0025.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/bridge-gradient-audit-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--unroll", type=int, default=25)
    parser.add_argument("--signal-pairs", type=int, default=16)
    return parser.parse_args()


def _load_controller(
    path: Path,
    *,
    graph: Path,
    device: torch.device,
    policy_hz: int,
) -> tuple[ConnectomeController, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != hover.file_sha256(graph):
        raise ValueError(f"graph hash mismatch in {path}")
    controller = ConnectomeController(graph, neural_dt=1.0 / policy_hz).to(device)
    controller.load_state_dict(checkpoint["controller"])
    return controller, checkpoint["controller"]


def _gradient(
    controller: ConnectomeController, loss_builder: Callable[[], torch.Tensor]
) -> tuple[dict[str, torch.Tensor], float]:
    controller.zero_grad(set_to_none=True)
    loss = loss_builder()
    loss.backward()
    gradients = {
        name: getattr(controller, name).grad.detach().clone() for name in PARAMETER_FAMILIES
    }
    return gradients, float(loss.detach())


def _dot(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum((first[name] * second[name]).sum() for name in PARAMETER_FAMILIES)


def _norm(gradient: dict[str, torch.Tensor], family: str | None = None) -> torch.Tensor:
    if family is not None:
        return torch.linalg.vector_norm(gradient[family])
    return torch.sqrt(sum(gradient[name].square().sum() for name in PARAMETER_FAMILIES))


def _cosine(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    family: str | None = None,
) -> float | None:
    first_norm = _norm(first, family)
    second_norm = _norm(second, family)
    denominator = first_norm * second_norm
    if float(denominator) < 1.0e-12:
        return None
    if family is None:
        numerator = _dot(first, second)
    else:
        numerator = (first[family] * second[family]).sum()
    return float(numerator / denominator)


def annotation_partitions(
    graph_path: Path, raw_dir: Path
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    graph = np.load(graph_path)
    node_ids = graph["node_ids"]
    annotations = _read_annotations(raw_dir / ANNOTATIONS_FILE)
    annotation_rows = {int(body): row for row, body in enumerate(annotations["bodyId"])}
    superclass = np.asarray(
        [str(annotations["superclass"][annotation_rows[int(body)]]) for body in node_ids]
    )
    visual = np.char.startswith(superclass.astype(str), "ol_") | np.char.startswith(
        superclass.astype(str), "visual_"
    )
    central = (
        np.char.startswith(superclass.astype(str), "cb_")
        | (np.char.find(superclass.astype(str), "ascending") >= 0)
        | (np.char.find(superclass.astype(str), "descending") >= 0)
    )
    vnc = np.char.startswith(superclass.astype(str), "vnc_") & (superclass != "vnc_motor")
    offsets = graph["output_pool_offsets"]
    pool_indices = graph["output_pool_indices"]
    output_motor = np.zeros(len(node_ids), dtype=bool)
    output_motor[pool_indices[offsets[0] : offsets[-1]]] = True
    visual &= ~output_motor
    central &= ~(visual | output_motor)
    vnc &= ~(visual | central | output_motor)
    remaining = ~(visual | central | vnc | output_motor)
    return {
        "visual_neurons_and_pathways": visual,
        "central_brain_descending_ascending": central,
        "vnc_interneurons": vnc,
        "output_motor_26": output_motor,
        "remaining": remaining,
    }, graph["edge_post"]


def gradient_partition_masks(
    partitions: dict[str, np.ndarray], edge_post: np.ndarray, device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "edge_magnitude": torch.from_numpy(node_mask[edge_post]).to(device=device),
            "bias": torch.from_numpy(node_mask).to(device=device),
            "raw_time_constant": torch.from_numpy(node_mask).to(device=device),
        }
        for name, node_mask in partitions.items()
    }


def _partition_stats(
    gradient: dict[str, torch.Tensor], masks: dict[str, dict[str, torch.Tensor]]
) -> dict[str, Any]:
    result = {}
    for partition, family_masks in masks.items():
        families = {}
        squared_norm = 0.0
        for family, mask in family_masks.items():
            values = gradient[family][mask]
            norm = float(torch.linalg.vector_norm(values))
            squared_norm += norm * norm
            families[family] = {
                "parameters": int(mask.sum()),
                "gradient_norm": norm,
                "gradient_rms": float(torch.sqrt(values.square().mean()))
                if values.numel()
                else None,
            }
        result[partition] = {
            "weighted_gradient_norm": math.sqrt(squared_norm),
            "by_parameter_family": families,
        }
    return result


def _partition_cosine(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> float | None:
    numerator = sum((first[name][mask] * second[name][mask]).sum() for name, mask in masks.items())
    first_norm = torch.sqrt(sum(first[name][mask].square().sum() for name, mask in masks.items()))
    second_norm = torch.sqrt(
        sum(second[name][mask].square().sum() for name, mask in masks.items())
    )
    denominator = first_norm * second_norm
    if float(denominator) < 1.0e-12:
        return None
    return float(numerator / denominator)


def _summarize(values: list[float | None]) -> dict[str, float | int | None]:
    valid = [value for value in values if value is not None and math.isfinite(value)]
    if not valid:
        return {"mean": None, "standard_deviation": None, "samples": 0}
    return {
        "mean": float(np.mean(valid)),
        "standard_deviation": float(np.std(valid)),
        "samples": len(valid),
    }


def _lesson_builders(
    controller: ConnectomeController,
    source: ConnectomeController,
    source_parameters: dict[str, torch.Tensor],
    *,
    prefix_steps: int,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Callable[[], torch.Tensor]]:
    def teacher(half_step: float, objective: str) -> torch.Tensor:
        loss, _ = bridge._teacher_pair_loss(
            controller,
            source,
            half_step=half_step,
            batch=args.batch_size,
            unroll=args.unroll,
            prefix_steps=prefix_steps,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=True,
            objective=objective,
        )
        return loss

    def paired_amplitudes(objective: str) -> torch.Tensor:
        small = teacher(bridge.PAIR_HALF_STEPS["small"], objective)
        medium = teacher(bridge.PAIR_HALF_STEPS["medium"], objective)
        return 0.5 * (small + medium)

    def cross_band() -> torch.Tensor:
        loss, _ = bridge._source_cross_band_loss(
            controller,
            source,
            batch=args.batch_size,
            unroll=args.unroll,
            prefix_steps=prefix_steps,
            physics_steps=physics_steps,
            device=device,
            config=config,
        )
        return loss

    def replay() -> torch.Tensor:
        loss, _ = bridge._dynamic_source_replay_loss(
            controller,
            source,
            batch=args.batch_size,
            unroll=args.unroll,
            prefix_steps=prefix_steps,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=True,
            axis_weights=(1.0, 1.0, 1.0, 0.0),
        )
        return loss

    return {
        "new_paired_contrast": lambda: paired_amplitudes("contrast"),
        "legacy_response_preservation": cross_band,
        "common_throttle_preservation": lambda: paired_amplitudes("common_throttle"),
        "dynamic_rpy_preservation": replay,
        "parameter_anchor": lambda: hover.source_regularization(controller, source_parameters),
    }


def truncated_pair_objective(
    student: ConnectomeController,
    source: ConnectomeController,
    *,
    half_step: float,
    batch: int,
    seed: int,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Contrast objective after a fixed source-generated recurrent/physical burn-in."""

    hover.seed_everything(seed)
    pairs = sample_marker_pairs(
        batch,
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
    state, _, _, source_neural, valid = bridge._dual_prefix(
        source,
        source,
        state,
        stick_state,
        prefix_marker,
        scene,
        steps=args.policy_hz,
        physics_steps=physics_steps,
        config=config,
    )
    loss_start = min(args.unroll - 1, 10)
    student_a, student_b = bridge._branch_outputs(
        student,
        state,
        source_neural,
        scene,
        pairs.marker_a,
        pairs.marker_b,
        response_steps=args.unroll,
        loss_start=loss_start,
    )
    with torch.no_grad():
        source_a, source_b = bridge._branch_outputs(
            source,
            state,
            source_neural,
            scene,
            pairs.marker_a,
            pairs.marker_b,
            response_steps=args.unroll,
            loss_start=loss_start,
        )
        target_a = hover.teacher_motor(state, pairs.marker_a, config)
        target_b = hover.teacher_motor(state, pairs.marker_b, config)
    desired = target_a[:, 3] - target_b[:, 3]
    predicted = student_a[:, :, 3] - student_b[:, :, 3]
    scale = max(0.01, 0.214 * (2.0 * half_step))
    valid_steps = valid[None].expand(student_a.shape[0], -1)
    contrast = hover.masked_mean(((predicted - desired[None]) / scale).square(), valid_steps)
    common = hover.masked_mean(
        (
            (
                0.5 * (student_a[:, :, 3] + student_b[:, :, 3])
                - 0.5 * (source_a[:, :, 3] + source_b[:, :, 3])
            )
            / 0.05
        ).square(),
        valid_steps,
    )
    rpy = hover.masked_mean(
        0.5
        * (
            ((student_a[:, :, :3] - source_a[:, :, :3]) / 0.05).square().mean(dim=2)
            + ((student_b[:, :, :3] - source_b[:, :, :3]) / 0.05)
            .square()
            .mean(dim=2)
        ),
        valid_steps,
    )
    return contrast, {
        "contrast_nrmse": float(torch.sqrt(contrast.detach())),
        "common_throttle_nrmse": float(torch.sqrt(common.detach())),
        "rpy_nrmse": float(torch.sqrt(rpy.detach())),
        "valid_fraction": float(valid.float().mean()),
    }


def functional_direction_test(
    source: ConnectomeController,
    source_state: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    """Apply a small family-RMS-normalized contrast descent direction to source clones."""

    batch_per_amplitude = 8
    fixed_seed = args.seed + 80_000

    def objective() -> torch.Tensor:
        small, _ = truncated_pair_objective(
            source,
            source,
            half_step=bridge.PAIR_HALF_STEPS["small"],
            batch=batch_per_amplitude,
            seed=fixed_seed,
            args=args,
            physics_steps=physics_steps,
            device=device,
            config=config,
        )
        medium, _ = truncated_pair_objective(
            source,
            source,
            half_step=bridge.PAIR_HALF_STEPS["medium"],
            batch=batch_per_amplitude,
            seed=fixed_seed + 1,
            args=args,
            physics_steps=physics_steps,
            device=device,
            config=config,
        )
        return 0.5 * (small + medium)

    source.requires_grad_(True)
    gradient, objective_value = _gradient(source, objective)
    direction = {}
    for family in PARAMETER_FAMILIES:
        rms = torch.sqrt(gradient[family].square().mean()).clamp_min(1.0e-12)
        direction[family] = -gradient[family] / rms
    source.requires_grad_(False)

    def evaluate(controller: ConnectomeController) -> dict[str, Any]:
        truncated = {}
        for index, (name, half_step) in enumerate(bridge.PAIR_HALF_STEPS.items()):
            _, truncated[name] = truncated_pair_objective(
                controller,
                source,
                half_step=half_step,
                batch=batch_per_amplitude,
                seed=fixed_seed + index,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
        hover.seed_everything(fixed_seed + 10)
        _, full_pair = bridge._teacher_pair_loss(
            controller,
            source,
            half_step=bridge.PAIR_HALF_STEPS["medium"],
            batch=16,
            unroll=args.unroll,
            prefix_steps=args.policy_hz,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=True,
        )
        hover.seed_everything(fixed_seed + 20)
        _, replay = bridge._dynamic_source_replay_loss(
            controller,
            source,
            batch=16,
            unroll=args.unroll,
            prefix_steps=args.policy_hz,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=True,
        )
        return {
            "fixed_source_burn_in": truncated,
            "full_replay_from_zero_pair": full_pair,
            "full_replay_from_zero_dynamic_source_error": replay,
        }

    baseline = evaluate(source)
    steps = {}
    for step_rms in (2.0e-5, 1.0e-4):
        controller = ConnectomeController(
            args.graph, neural_dt=1.0 / args.policy_hz
        ).to(device)
        controller.load_state_dict(source_state)
        with torch.no_grad():
            for family in PARAMETER_FAMILIES:
                getattr(controller, family).add_(step_rms * direction[family])
        controller.project_parameters()
        controller.eval()
        steps[f"{step_rms:.0e}"] = {
            "requested_parameter_rms_step_per_family": step_rms,
            "actual_parameter_rms_delta": {
                family: float(
                    torch.sqrt(
                        (getattr(controller, family) - source_state[family]).square().mean()
                    ).detach()
                )
                for family in PARAMETER_FAMILIES
            },
            "evaluation": evaluate(controller),
        }
    return {
        "direction": (
            "negative fixed-burn-in training-support contrast gradient, independently "
            "normalized to unit RMS in each native parameter family"
        ),
        "fixed_pairs": 16,
        "contrast_objective_at_source": objective_value,
        "baseline": baseline,
        "steps": steps,
    }


def audit_gradients(
    controller: ConnectomeController,
    source: ConnectomeController,
    source_parameters: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
    seed_offset: int,
    partition_masks: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    trials = []
    for trial in range(args.trials):
        prefix_steps = args.policy_hz
        gradients = {}
        losses = {}
        for lesson_index, (lesson, builder) in enumerate(
            _lesson_builders(
                controller,
                source,
                source_parameters,
                prefix_steps=prefix_steps,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            ).items()
        ):
            hover.seed_everything(args.seed + seed_offset + 100 * trial + lesson_index)
            gradients[lesson], losses[lesson] = _gradient(controller, builder)

        cosine_global = {}
        cosine_by_family = {family: {} for family in PARAMETER_FAMILIES}
        cosine_by_partition = {partition: {} for partition in partition_masks}
        for first_index, first in enumerate(LESSONS):
            for second in LESSONS[first_index + 1 :]:
                key = f"{first}__{second}"
                cosine_global[key] = _cosine(gradients[first], gradients[second])
                for family in PARAMETER_FAMILIES:
                    cosine_by_family[family][key] = _cosine(
                        gradients[first], gradients[second], family
                    )
                for partition, masks in partition_masks.items():
                    cosine_by_partition[partition][key] = _partition_cosine(
                        gradients[first], gradients[second], masks
                    )

        average = {
            family: sum(gradients[lesson][family] for lesson in LESSONS[:-1])
            / len(LESSONS[:-1])
            + gradients["parameter_anchor"][family]
            for family in PARAMETER_FAMILIES
        }
        directional_derivatives = {
            lesson: float(_dot(gradients[lesson], average)) for lesson in LESSONS
        }
        trials.append(
            {
                "trial": trial,
                "prefix_steps": prefix_steps,
                "losses": losses,
                "gradient_norms": {
                    lesson: {
                        "global": float(_norm(gradient)),
                        **{
                            family: {
                                "gradient_norm": float(_norm(gradient, family)),
                                "gradient_rms": float(
                                    torch.sqrt(gradient[family].square().mean())
                                ),
                            }
                            for family in PARAMETER_FAMILIES
                        },
                    }
                    for lesson, gradient in gradients.items()
                },
                "cosine_global": cosine_global,
                "cosine_by_parameter_family": cosine_by_family,
                "gradient_by_annotation_partition": {
                    lesson: _partition_stats(gradient, partition_masks)
                    for lesson, gradient in gradients.items()
                },
                "cosine_by_annotation_partition": cosine_by_partition,
                "accumulated_gradient_norm": float(_norm(average)),
                "lesson_dot_accumulated_gradient": directional_derivatives,
                "combined_step_would_ascend_lesson": {
                    lesson: value < 0.0 for lesson, value in directional_derivatives.items()
                },
            }
        )
        print(
            json.dumps(
                {
                    "progress": "gradient_trial_complete",
                    "trial": trial + 1,
                    "trials": args.trials,
                    "prefix_steps": prefix_steps,
                }
            ),
            flush=True,
        )

    global_keys = trials[0]["cosine_global"]
    aggregate_cosines = {
        key: _summarize([trial["cosine_global"][key] for trial in trials])
        for key in global_keys
    }
    aggregate_families = {
        family: {
            key: _summarize(
                [trial["cosine_by_parameter_family"][family][key] for trial in trials]
            )
            for key in global_keys
        }
        for family in PARAMETER_FAMILIES
    }
    aggregate_partitions = {
        partition: {
            key: _summarize(
                [trial["cosine_by_annotation_partition"][partition][key] for trial in trials]
            )
            for key in global_keys
        }
        for partition in partition_masks
    }
    return {
        "trials": trials,
        "aggregate_cosine_global": aggregate_cosines,
        "aggregate_cosine_by_parameter_family": aggregate_families,
        "aggregate_cosine_by_annotation_partition": aggregate_partitions,
        "negative_accumulated_direction_counts": {
            lesson: sum(
                trial["combined_step_would_ascend_lesson"][lesson] for trial in trials
            )
            for lesson in LESSONS
        },
    }


def directed_hop_distances(graph_path: Path, maximum_hops: int) -> dict[str, np.ndarray]:
    graph = np.load(graph_path)
    edge_pre = graph["edge_pre"]
    edge_post = graph["edge_post"]
    node_count = len(graph["node_ids"])
    visual = graph["visual_node_indices"]
    offsets = graph["output_pool_offsets"]
    pools = graph["output_pool_indices"]
    throttle = pools[offsets[6] : offsets[8]]

    def expand(starts: np.ndarray, *, reverse: bool) -> np.ndarray:
        distance = np.full(node_count, -1, dtype=np.int16)
        distance[starts] = 0
        frontier = np.zeros(node_count, dtype=bool)
        frontier[starts] = True
        source_edges, target_edges = (edge_post, edge_pre) if reverse else (edge_pre, edge_post)
        for hop in range(1, maximum_hops + 1):
            next_nodes = np.unique(target_edges[frontier[source_edges]])
            next_nodes = next_nodes[distance[next_nodes] < 0]
            if not len(next_nodes):
                break
            distance[next_nodes] = hop
            frontier.fill(False)
            frontier[next_nodes] = True
        return distance

    return {
        "from_visual": expand(visual, reverse=False),
        "to_throttle_motor": expand(throttle, reverse=True),
        "visual_nodes": visual,
        "throttle_nodes": throttle,
    }


def _signal_masks(
    distances: dict[str, np.ndarray], partitions: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    forward = distances["from_visual"]
    reverse = distances["to_throttle_motor"]
    return {
        **partitions,
        "visual_nodes": np.isin(np.arange(len(forward)), distances["visual_nodes"]),
        "forward_hop_1": forward == 1,
        "forward_hops_2_3": (forward >= 2) & (forward <= 3),
        "forward_hops_4_6": (forward >= 4) & (forward <= 6),
        "forward_hops_7_12": (forward >= 7) & (forward <= 12),
        "forward_hops_13_25": (forward >= 13) & (forward <= 25),
        "visual_to_throttle_corridor_25":
            (forward >= 0) & (reverse >= 0) & (forward + reverse <= 25),
        "throttle_motor_pool": np.isin(
            np.arange(len(forward)), distances["throttle_nodes"]
        ),
    }


@torch.no_grad()
def audit_marker_signal(
    controller: ConnectomeController,
    masks: dict[str, np.ndarray],
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
    seed: int,
    half_step: float,
) -> dict[str, Any]:
    hover.seed_everything(seed)
    batch = args.signal_pairs
    pairs = sample_marker_pairs(
        batch, device=device, held_out=True, half_step_metres=half_step
    )
    state, stick_state, scene = hover._paired_initial_state(
        pairs,
        device=device,
        config=config,
        held_out_scene=False,
        all_style_combinations=True,
    )
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    state, _, neural, valid = hover.current_policy_prefix(
        controller,
        state,
        stick_state,
        neural,
        pairs.centre_height,
        scene,
        steps=args.policy_hz,
        physics_steps=physics_steps,
        config=config,
    )
    image_a = render_visual_hover_scene(state, pairs.marker_a, scene=scene)
    image_b = render_visual_hover_scene(state, pairs.marker_b, scene=scene)
    retina_a = controller.sample_retina(image_a)
    retina_b = controller.sample_retina(image_b)
    desired = hover.teacher_motor(state, pairs.marker_a, config)[:, 3] - hover.teacher_motor(
        state, pairs.marker_b, config
    )[:, 3]
    desired_rms = torch.sqrt(desired.square().mean()).clamp_min(1.0e-12)

    nuisance_a = sample_visual_scenes(
        batch,
        device=device,
        held_out_combinations=False,
        all_style_combinations=True,
    )
    nuisance_b = sample_visual_scenes(
        batch,
        device=device,
        held_out_combinations=False,
        all_style_combinations=True,
    )
    image_nuisance_a = render_visual_hover_scene(state, pairs.centre_height, scene=nuisance_a)
    image_nuisance_b = render_visual_hover_scene(state, pairs.centre_height, scene=nuisance_b)
    retina_nuisance_a = controller.sample_retina(image_nuisance_a)
    retina_nuisance_b = controller.sample_retina(image_nuisance_b)

    images = torch.cat((image_a, image_b))
    attitudes = torch.cat((state.euler[:, :2], state.euler[:, :2]))
    recurrent = torch.cat((neural.clone(), neural.clone()))
    mask_tensors = {
        name: torch.from_numpy(mask).to(device=device) for name, mask in masks.items()
    }
    propagation = {}
    outputs = []
    for step in range(1, max(SIGNAL_STEPS) + 1):
        motor, recurrent = controller(images, attitudes, recurrent)
        outputs.append(motor)
        if step in SIGNAL_STEPS:
            delta = recurrent[:batch] - recurrent[batch:]
            propagation[str(step)] = {
                name: {
                    "nodes": int(mask.sum()),
                    "state_delta_rms": float(torch.sqrt(delta[:, mask].square().mean()))
                    if bool(mask.any())
                    else None,
                    "state_delta_rms_per_teacher_contrast": float(
                        torch.sqrt(delta[:, mask].square().mean()) / desired_rms
                    )
                    if bool(mask.any())
                    else None,
                    "state_delta_absolute_max": float(delta[:, mask].abs().max())
                    if bool(mask.any())
                    else None,
                }
                for name, mask_tensor in mask_tensors.items()
                for mask in (mask_tensor,)
            }
    final_outputs = torch.stack(outputs[-10:]).mean(dim=0)
    predicted = final_outputs[:batch, 3] - final_outputs[batch:, 3]
    final_state = recurrent
    final_activity = torch.sigmoid(final_state)
    positive_begin = int(controller.pool_offsets[6])
    positive_end = int(controller.pool_offsets[7])
    negative_end = int(controller.pool_offsets[8])
    positive_nodes = controller.pool_indices[positive_begin:positive_end]
    negative_nodes = controller.pool_indices[positive_end:negative_end]
    positive_contrast = final_activity[:batch, positive_nodes].mean(dim=1) - final_activity[
        batch:, positive_nodes
    ].mean(dim=1)
    negative_contrast = final_activity[:batch, negative_nodes].mean(dim=1) - final_activity[
        batch:, negative_nodes
    ].mean(dim=1)

    def rms(value: torch.Tensor) -> float:
        return float(torch.sqrt(value.square().mean()))

    marker_retina_rms = rms(retina_a - retina_b)
    nuisance_retina_rms = rms(retina_nuisance_a - retina_nuisance_b)
    return {
        "pairs": batch,
        "marker_difference_metres": 2.0 * half_step,
        "valid_prefix_rate": float(valid.float().mean()),
        "image_marker_difference_rms": rms(image_a - image_b),
        "image_nuisance_difference_rms": rms(image_nuisance_a - image_nuisance_b),
        "retina_marker_difference_rms": marker_retina_rms,
        "injected_retinal_current_marker_difference_rms": 5.0 * marker_retina_rms,
        "retina_nuisance_difference_rms": nuisance_retina_rms,
        "retina_marker_to_nuisance_rms_ratio": marker_retina_rms
        / max(nuisance_retina_rms, 1.0e-12),
        "retina_marker_changed_fraction_above_1e_3": float(
            ((retina_a - retina_b).abs() > 1.0e-3).float().mean()
        ),
        "predicted_throttle_contrast_rms": rms(predicted),
        "predicted_throttle_sensitivity_rms_per_metre": rms(predicted)
        / (2.0 * half_step),
        "desired_throttle_contrast_rms": rms(desired),
        "positive_throttle_pool_activity_contrast_rms": rms(positive_contrast),
        "negative_throttle_pool_activity_contrast_rms": rms(negative_contrast),
        "final_common_throttle_motor_mean": float(
            0.5 * (final_outputs[:batch, 3] + final_outputs[batch:, 3]).mean()
        ),
        "throttle_response_slope": float(
            (predicted * desired).sum() / desired.square().sum().clamp_min(1.0e-8)
        ),
        "propagation_by_recurrent_step": propagation,
    }


@torch.no_grad()
def audit_style_collective(
    controller: ConnectomeController,
    *,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    batch = 4
    position = torch.tensor(((0.0, 0.0, 1.0),), device=device).expand(batch, -1).clone()
    state = DifferentiableQuad(config).to(device).initial_state(
        batch, device=device, dtype=torch.float32, position=position
    )
    wall_style = torch.tensor((0, 0, 1, 1), device=device)
    floor_style = torch.tensor((0, 1, 0, 1), device=device)
    centres = torch.tensor(TEXTURE_PHASE_CENTRES, device=device)
    ones = torch.ones(batch, device=device)
    scene = VisualScene(
        wall_x=5.0 * ones,
        wall_phase=centres[wall_style],
        floor_phase=centres[floor_style],
        wall_texture_gain=ones,
        floor_texture_gain=ones,
        illumination=ones,
        marker_gain=ones,
        wall_style=wall_style,
        floor_style=floor_style,
    )
    marker = torch.full((batch,), 0.75, device=device)
    image = render_visual_hover_scene(state, marker, config=config, scene=scene)
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    output = None
    for _ in range(25):
        output, neural = controller(image, state.euler[:, :2], neural)
    assert output is not None
    style_outputs = {
        f"wall_{int(wall_style[index])}_floor_{int(floor_style[index])}": output[index].tolist()
        for index in range(batch)
    }
    return {
        "fixed_pose_marker_and_static_history": True,
        "marker_height_metres": 0.75,
        "style_motor_outputs": style_outputs,
        "throttle_range_across_styles": float(output[:, 3].max() - output[:, 3].min()),
        "roll_pitch_yaw_max_range_across_styles": float(
            (output[:, :3].max(dim=0).values - output[:, :3].min(dim=0).values).max()
        ),
    }


def main() -> int:
    args = parse_args()
    for path in (
        args.graph,
        args.raw_dir / ANNOTATIONS_FILE,
        args.source_checkpoint,
        args.candidate_checkpoint,
    ):
        if not path.is_file():
            raise SystemExit(f"required input is missing: {path}")
    if args.trials < 1 or args.batch_size < 1 or args.signal_pairs < 1:
        raise SystemExit("trials, batch-size and signal-pairs must be positive")
    if args.batch_size % 4 or args.signal_pairs % 4:
        raise SystemExit("balanced all-style batches must be divisible by four")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    physics_steps = round(1.0 / config.dt) // args.policy_hz
    started = perf_counter()
    source, source_state = _load_controller(
        args.source_checkpoint,
        graph=args.graph,
        device=device,
        policy_hz=args.policy_hz,
    )
    candidate, _ = _load_controller(
        args.candidate_checkpoint,
        graph=args.graph,
        device=device,
        policy_hz=args.policy_hz,
    )
    source.eval()
    source.requires_grad_(False)
    source_parameters = {
        name: source_state[name].detach().clone() for name in PARAMETER_FAMILIES
    }
    partitions, edge_post = annotation_partitions(args.graph, args.raw_dir)
    partition_masks = gradient_partition_masks(partitions, edge_post, device)
    distances = directed_hop_distances(args.graph, maximum_hops=max(SIGNAL_STEPS))
    masks = _signal_masks(distances, partitions)

    checkpoint_results = {}
    for index, (name, controller) in enumerate((("source", source), ("candidate", candidate))):
        controller.requires_grad_(True)
        controller.eval()
        checkpoint_results[name] = {
            "gradient_geometry": audit_gradients(
                controller,
                source,
                source_parameters,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
                seed_offset=index * 10_000,
                partition_masks=partition_masks,
            ),
            "marker_signal": {
                name: audit_marker_signal(
                    controller,
                    masks,
                    args=args,
                    physics_steps=physics_steps,
                    device=device,
                    config=config,
                    seed=args.seed + 50_000 + amplitude_index,
                    half_step=half_step,
                )
                for amplitude_index, (name, half_step) in enumerate(
                    bridge.PAIR_HALF_STEPS.items()
                )
            },
            "style_collective": audit_style_collective(
                controller,
                device=device,
                config=config,
            ),
        }
        print(
            json.dumps({"progress": "checkpoint_complete", "checkpoint": name}),
            flush=True,
        )
        controller.zero_grad(set_to_none=True)
        if name == "source":
            controller.requires_grad_(False)

    direction_test = functional_direction_test(
        source,
        source_state,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    print(json.dumps({"progress": "functional_direction_complete"}), flush=True)
    report = {
        "experiment": "variable-height-bridge-gradient-signal-audit-v1",
        "diagnostic_only": True,
        "actor_parameters_changed": False,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": hover.file_sha256(args.source_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_checkpoint_sha256": hover.file_sha256(args.candidate_checkpoint),
        "graph": str(args.graph),
        "graph_sha256": hover.file_sha256(args.graph),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": args.seed,
        "trials": args.trials,
        "gradient_parameter_families": list(PARAMETER_FAMILIES),
        "signal_path_node_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "checkpoints": checkpoint_results,
        "functional_direction_test": direction_test,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
