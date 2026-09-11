#!/usr/bin/env python3
"""Audit frozen T4/T5 optic-motion tuning in the full MaleCNS controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_continuous_cns_solver as continuous  # noqa: E402
import audit_variable_height_neural_integration_rate as rate  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402

from flydrone.connectome_data import ANNOTATIONS_FILE, _read_annotations  # noqa: E402
from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "frozen-t4t5-optic-motion-audit-v1"
PROTOCOL_COMMIT = "753a7cd"
EXPECTED_GRAPH_SHA256 = "8c6ba28d149e9ac4a5223c5919657a1734f2c2cac9828c51114fd0174a383665"
EXPECTED_CHECKPOINT_SHA256 = (
    "7238b0e3ca39dc1a8bfd35dcf9f8b6fc9e8c8881ff989f64cb135fa6fed4e572"
)
EXPECTED_ANNOTATIONS_SHA256 = (
    "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"
)
EXPECTED_TAU_REPORT_SHA256 = (
    "c5062fb18039d248d045eebbbfc17ccacc43522d5a34c2d239f7a14b309ba64a"
)

TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
EXPECTED_TYPE_COUNTS = {
    "T4a": 1684,
    "T4b": 1690,
    "T4c": 1778,
    "T4d": 1709,
    "T5a": 1664,
    "T5b": 1715,
    "T5c": 1720,
    "T5d": 1620,
}
EXPECTED_BODY_IDS_SHA256 = (
    "3af4c6cbccc60438067dcc4143f07e7cbae56b8132ac61ff3ed545f57a69a176"
)
EXPECTED_INDICES_SHA256 = (
    "4146cd93be4cfc9ca7344081d3f9ae87690e21e5b41fc1cb8d54db4857b86cb5"
)

WIDTH = 320
HEIGHT = 200
HFOV_DEGREES = 125.0
FOCAL_PIXELS = WIDTH / (2.0 * math.tan(math.radians(HFOV_DEGREES) / 2.0))
VFOV_DEGREES = math.degrees(2.0 * math.atan(HEIGHT / (2.0 * FOCAL_PIXELS)))
POLICY_HZ = 50
PREFIX_FRAMES = 25
MOTION_FRAMES = 16
TOTAL_FRAMES = MOTION_FRAMES + 1
STIMULUS_SEED = 480_991
CORE_CASES = 64
GENERALIZATION_CASES = 16
BLOCK_CASES = 8
NUMERICAL_CASES = 8
REVERSE_CASES = 16
REFERENCE_SUBSTEPS = 32
FINE_SUBSTEPS = 64
RK4_STEPS = (2, 4)
ACTIVITY_RMS_LIMIT = 0.005
OPPONENT_RMS_LIMIT = 0.005
MOTOR_RMS_LIMIT = 0.005
SIGN_FRACTION_MINIMUM = 0.90
MEDIAN_DSI_MINIMUM = 0.30
ACTIVE_FRACTION_MINIMUM = 0.50
STATIONARY_RATIO_MAXIMUM = 0.10
ACTIVE_ABSOLUTE_FLOOR = 1.0e-6


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
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0"
    )
    parser.add_argument(
        "--tau-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-premotor-tau-preflight-002/report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/frozen-t4t5-audit-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-cases", type=int, default=BLOCK_CASES)
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "learning": False,
        "source": "original paired-dynamic-001 controller",
        "inputs": ["320x200 linear grayscale replicated to RGB", "zero roll", "zero pitch"],
        "privileged_or_engineered_actor_inputs": [],
        "rates": {
            "camera_hz": POLICY_HZ,
            "primary_exponential_euler_substeps_per_frame": REFERENCE_SUBSTEPS,
            "primary_neural_step_hz": POLICY_HZ * REFERENCE_SUBSTEPS,
            "fine_control_substeps_per_frame": FINE_SUBSTEPS,
            "rk4_steps_per_frame": list(RK4_STEPS),
            "image_held_during_neural_substeps": True,
        },
        "anatomy": {
            "exact_types": list(TYPES),
            "excluded_type": "T5a_unclear",
            "cells": sum(EXPECTED_TYPE_COUNTS.values()),
            "body_ids_sha256": EXPECTED_BODY_IDS_SHA256,
            "graph_indices_sha256": EXPECTED_INDICES_SHA256,
        },
        "stimuli": {
            "seed": STIMULUS_SEED,
            "projection": {
                "horizontal_fov_degrees": HFOV_DEGREES,
                "vertical_fov_degrees": VFOV_DEGREES,
                "focal_length_pixels": FOCAL_PIXELS,
            },
            "core_pairs": CORE_CASES,
            "core_factorial": {
                "axis": ["vertical", "horizontal"],
                "polarity": ["bright", "dark"],
                "pixels_per_frame": [2, 4],
                "phase": list(range(8)),
            },
            "generalization_pairs": GENERALIZATION_CASES,
            "generalization_families": ["sine", "texture"],
            "prefix_frames": PREFIX_FRAMES,
            "motion_frames": MOTION_FRAMES,
            "common_terminal_frames": 1,
            "opposite_pair_terminal_pixels_exact": True,
            "labels": "angular image motion only",
            "edge_polarity": (
                "polarity-preserving advancing half-plane; bright is pixelwise "
                "nondecreasing and dark is pixelwise nonincreasing"
            ),
            "reverse_control": (
                "literal frame reversal with its own first-frame stationary baseline; "
                "ON reversals are measured in T5 and OFF reversals in T4"
            ),
            "camera_to_anatomy_vertical_convention": (
                "decreasing image row is anatomical upward under the fixed negative-hex2 "
                "camera map; the map is an engineering approximation"
            ),
            "common_terminal_interpretation": (
                "post-offset retention after uniform gray, not ongoing motion detection"
            ),
        },
        "numerical_gate": {
            "pairs": NUMERICAL_CASES,
            "K32_vs_K64_selected_activity_rms_maximum": ACTIVITY_RMS_LIMIT,
            "K32_vs_K64_opponent_rms_maximum": OPPONENT_RMS_LIMIT,
            "K32_vs_K64_terminal_motor_rms_maximum": MOTOR_RMS_LIMIT,
        },
        "qualification": {
            "vertical_sign_fraction_minimum": SIGN_FRACTION_MINIMUM,
            "active_subtype_median_dsi_minimum": MEDIAN_DSI_MINIMUM,
            "active_subtype_fraction_minimum": ACTIVE_FRACTION_MINIMUM,
            "stationary_to_moving_opponent_rms_maximum": STATIONARY_RATIO_MAXIMUM,
            "reverse_sign_inversion_fraction_minimum": SIGN_FRACTION_MINIMUM,
            "fitted_decoder": False,
        },
        "pair_mean_t4t5_intervention_is_diagnostic_only": True,
        "candidate_retained": False,
        "authorizes_hover_gate_or_promotion": False,
    }


def file_sha256(path: Path) -> str:
    return assisted.responsibility.file_sha256(path)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def validate_inputs(args: argparse.Namespace) -> dict[str, str]:
    expected = {
        args.graph: EXPECTED_GRAPH_SHA256,
        args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        args.raw_dir / ANNOTATIONS_FILE: EXPECTED_ANNOTATIONS_SHA256,
        args.tau_report: EXPECTED_TAU_REPORT_SHA256,
    }
    observed = {}
    for path, digest in expected.items():
        if not path.is_file() or file_sha256(path) != digest:
            raise SystemExit(f"locked optic-motion input is missing or changed: {path}")
        observed[assisted.responsibility.stable_path(path)] = digest
    with args.tau_report.open() as stream:
        tau_report = json.load(stream)
    if (
        tau_report.get("classification")
        != "premotor_tau_source_solver_requalification_failed"
        or tau_report.get("passed")
        or tau_report.get("development") is not None
        or not tau_report.get("source_state_restored")
    ):
        raise SystemExit("premotor-tau report is not the registered terminal stop")
    return observed


def anatomy_manifest(args: argparse.Namespace) -> dict[str, Any]:
    graph = np.load(args.graph)
    node_ids = graph["node_ids"]
    annotations = _read_annotations(args.raw_dir / ANNOTATIONS_FILE)
    rows = {int(body): row for row, body in enumerate(annotations["bodyId"])}
    type_values = np.asarray([str(value) for value in annotations["type"]])
    selected_rows = np.flatnonzero(np.isin(type_values, TYPES))
    body_ids = np.sort(np.asarray(annotations["bodyId"], dtype=np.int64)[selected_rows])
    indices = np.searchsorted(node_ids, body_ids)
    if (
        len(body_ids) != sum(EXPECTED_TYPE_COUNTS.values())
        or not np.array_equal(node_ids[indices], body_ids)
        or array_sha256(body_ids) != EXPECTED_BODY_IDS_SHA256
        or array_sha256(indices) != EXPECTED_INDICES_SHA256
    ):
        raise SystemExit("frozen T4/T5 cohort changed")
    by_type: dict[str, Any] = {}
    type_indices: dict[str, np.ndarray] = {}
    for cell_type in TYPES:
        typed = np.sort(
            np.asarray(
                [
                    body
                    for body in body_ids
                    if str(annotations["type"][rows[int(body)]]) == cell_type
                ],
                dtype=np.int64,
            )
        )
        typed_indices = np.searchsorted(node_ids, typed)
        instances = [str(annotations["instance"][rows[int(body)]]) for body in typed]
        if len(typed) != EXPECTED_TYPE_COUNTS[cell_type]:
            raise SystemExit(f"frozen {cell_type} count changed")
        type_indices[cell_type] = typed_indices
        by_type[cell_type] = {
            "count": len(typed),
            "left": sum(value.endswith("_L") for value in instances),
            "right": sum(value.endswith("_R") for value in instances),
            "body_ids_sha256": array_sha256(typed),
            "indices_sha256": array_sha256(typed_indices),
        }

    selected_mask = np.zeros(len(node_ids), dtype=bool)
    selected_mask[indices] = True
    edge_pre, edge_post = graph["edge_pre"], graph["edge_post"]
    afferent_edges = (~selected_mask[edge_pre]) & selected_mask[edge_post]
    target_edges = selected_mask[edge_pre] & (~selected_mask[edge_post])
    outgoing_unique = np.unique(edge_post[target_edges])
    visual_targets: dict[str, list[int]] = {}
    for index in outgoing_unique:
        row = rows[int(node_ids[index])]
        superclass = str(annotations["superclass"][row])
        cell_type = str(annotations["type"][row])
        if superclass in ("ol_intrinsic", "visual_projection") and cell_type not in (
            "None",
            "nan",
        ):
            visual_targets.setdefault(cell_type, []).append(int(index))
    visual_target_indices = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in sorted(visual_targets.items())
    }

    def typed_counts(graph_indices: np.ndarray) -> dict[str, int]:
        values = [
            str(annotations["type"][rows[int(node_ids[index])]])
            for index in np.unique(graph_indices)
        ]
        names, counts = np.unique(values, return_counts=True)
        return {
            str(name): int(count)
            for name, count in zip(names, counts, strict=True)
            if name not in ("None", "nan")
        }

    return {
        "selected_nodes": len(indices),
        "body_ids_sha256": array_sha256(body_ids),
        "indices_sha256": array_sha256(indices),
        "by_type": by_type,
        "selected_indices": indices,
        "type_indices": type_indices,
        "afferent_edges": int(afferent_edges.sum()),
        "afferent_nodes": int(len(np.unique(edge_pre[afferent_edges]))),
        "afferent_type_counts": typed_counts(edge_pre[afferent_edges]),
        "outgoing_edges": int(target_edges.sum()),
        "outgoing_nodes": int(len(np.unique(edge_post[target_edges]))),
        "outgoing_type_counts": typed_counts(edge_post[target_edges]),
        "typed_visual_target_counts": {
            name: len(values) for name, values in visual_target_indices.items()
        },
        "visual_target_type_indices": visual_target_indices,
    }


def report_anatomy(anatomy: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in anatomy.items()
        if name not in (
            "selected_indices",
            "type_indices",
            "visual_target_type_indices",
        )
    }


def angular_motion_metadata(
    *, axis: str, family: str, speed: int, phase: int
) -> dict[str, Any]:
    directions = (
        ("down", "up") if axis == "vertical" else ("right", "left")
    )
    local_degrees = math.degrees(math.atan(speed / FOCAL_PIXELS))
    metadata: dict[str, Any] = {
        "branch_directions": list(directions),
        "local_degrees_per_frame_at_optical_axis": local_degrees,
        "local_degrees_per_second_at_optical_axis": local_degrees * POLICY_HZ,
    }
    if family == "edge":
        size = HEIGHT if axis == "vertical" else WIDTH
        principal = (size - 1.0) / 2.0
        center = (0.30 + 0.40 * (phase + 0.5) / 8.0) * size

        def angle(pixel: float) -> float:
            return math.degrees(math.atan((pixel - principal) / FOCAL_PIXELS))

        branch_rates = []
        for sign in (1.0, -1.0):
            first = center - sign * speed * (MOTION_FRAMES - 1) / 2.0
            last = center + sign * speed * (MOTION_FRAMES - 1) / 2.0
            branch_rates.append((angle(last) - angle(first)) / (MOTION_FRAMES - 1))
        metadata["mean_degrees_per_frame_by_branch"] = branch_rates
        metadata["mean_degrees_per_second_by_branch"] = [
            value * POLICY_HZ for value in branch_rates
        ]
    return metadata


def stimulus_specs() -> list[dict[str, Any]]:
    specs = []
    case = 0
    for axis in ("vertical", "horizontal"):
        for polarity in ("bright", "dark"):
            for speed in (2, 4):
                for phase in range(8):
                    specs.append(
                        {
                            "case": case,
                            "cohort": "core",
                            "family": "edge",
                            "axis": axis,
                            "polarity": polarity,
                            "speed_pixels_per_frame": speed,
                            "phase": phase,
                            "angular_motion": angular_motion_metadata(
                                axis=axis,
                                family="edge",
                                speed=speed,
                                phase=phase,
                            ),
                        }
                    )
                    case += 1
    for axis in ("vertical", "horizontal"):
        for family in ("sine", "texture"):
            for speed in (2, 4):
                for phase in range(2):
                    specs.append(
                        {
                            "case": case,
                            "cohort": "generalization",
                            "family": family,
                            "axis": axis,
                            "polarity": "mixed",
                            "speed_pixels_per_frame": speed,
                            "phase": phase,
                            "angular_motion": angular_motion_metadata(
                                axis=axis,
                                family=family,
                                speed=speed,
                                phase=phase,
                            ),
                        }
                    )
                    case += 1
    if len(specs) != CORE_CASES + GENERALIZATION_CASES:
        raise RuntimeError("optic-motion stimulus factorial is incomplete")
    return specs


def numerical_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in specs
        if item["cohort"] == "core"
        and item["speed_pixels_per_frame"] == 4
        and item["phase"] in (0, 1)
    ]


def reverse_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in specs
        if item["cohort"] == "core"
        and item["axis"] == "vertical"
        and item["speed_pixels_per_frame"] == 4
    ]


@lru_cache(maxsize=1)
def _coordinate_grids() -> tuple[Tensor, Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(HEIGHT, dtype=torch.float32),
        torch.arange(WIDTH, dtype=torch.float32),
        indexing="ij",
    )
    return yy, xx


def motion_frame(spec: dict[str, Any], branch: int, frame: int) -> Tensor:
    if frame == MOTION_FRAMES:
        return torch.full((HEIGHT, WIDTH), 0.5, dtype=torch.float32)
    yy, xx = _coordinate_grids()
    coordinate = yy if spec["axis"] == "vertical" else xx
    transverse = xx if spec["axis"] == "vertical" else yy
    sign = 1.0 if branch == 0 else -1.0
    speed = float(spec["speed_pixels_per_frame"])
    phase = int(spec["phase"])
    if spec["family"] == "edge":
        size = HEIGHT if spec["axis"] == "vertical" else WIDTH
        center = (0.30 + 0.40 * (phase + 0.5) / 8.0) * size
        center += sign * speed * (frame - (MOTION_FRAMES - 1) / 2.0)
        # The spatial edge reverses orientation with travel direction.  Consequently,
        # every pixel crossed by a bright/ON edge increases and every pixel crossed by
        # a dark/OFF edge decreases, for either member of the direction pair.
        swept = (sign * (coordinate - center) <= 0.0).to(coordinate.dtype)
        if spec["polarity"] == "bright":
            return 0.2 + 0.6 * swept
        return 0.8 - 0.6 * swept
    shift = sign * speed * frame
    if spec["family"] == "sine":
        return 0.5 + 0.22 * torch.sin(
            2.0 * math.pi * (coordinate - shift) / 32.0 + phase * math.pi
        )
    seed_phase = (STIMULUS_SEED % 997) / 997.0 + phase * 0.37
    value = (
        0.5
        + 0.055
        * torch.sin(
            2.0
            * math.pi
            * ((coordinate - shift) / 29.0 + transverse / 61.0 + seed_phase)
        )
        + 0.035
        * torch.sin(
            2.0
            * math.pi
            * (
                (coordinate - shift) / 13.0
                - transverse / 43.0
                + 0.7 * seed_phase
            )
        )
        + 0.020
        * torch.sin(
            2.0
            * math.pi
            * (
                (coordinate - shift) / 7.0
                + transverse / 23.0
                + 1.3 * seed_phase
            )
        )
    )
    return value.clamp(0.0, 1.0)


def render_sequence(spec: dict[str, Any], *, mode: str = "normal") -> Tensor:
    motion = []
    for frame in range(MOTION_FRAMES):
        source_frame = MOTION_FRAMES - 1 - frame if mode == "reverse" else frame
        motion.append(
            torch.stack(
                [motion_frame(spec, branch, source_frame) for branch in (0, 1)]
            )
        )
    terminal = torch.stack(
        [motion_frame(spec, branch, MOTION_FRAMES) for branch in (0, 1)]
    )
    motion.append(terminal)
    moved = torch.stack(motion)
    stationary_source_frame = MOTION_FRAMES - 1 if mode == "reverse" else 0
    stationary = torch.stack(
        [
            torch.stack(
                [
                    motion_frame(spec, branch, stationary_source_frame)
                    for branch in (0, 1)
                ]
            )
            for _ in range(MOTION_FRAMES)
        ]
        + [terminal]
    )
    return torch.cat((moved, stationary), dim=1)


def stimulus_manifest(specs: list[dict[str, Any]]) -> dict[str, Any]:
    hasher = hashlib.sha256()
    endpoint_max = 0.0
    duplicate_max = 0.0
    for spec in specs:
        sequence = render_sequence(spec)
        hasher.update(np.ascontiguousarray(sequence.numpy()).view(np.uint8))
        endpoint_max = max(
            endpoint_max,
            float((sequence[-1] - sequence[-1, 0][None]).abs().max()),
        )
        duplicate_max = max(
            duplicate_max, float((sequence - render_sequence(spec)).abs().max())
        )
    return {
        "seed": STIMULUS_SEED,
        "specs": specs,
        "rendered_float32_sha256": hasher.hexdigest(),
        "opposite_terminal_maximum_difference": endpoint_max,
        "duplicate_render_maximum_difference": duplicate_max,
    }


def load_checkpoint(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        name: value.detach().cpu().clone()
        for name, value in payload["controller"].items()
    }


def state_sha256(state: dict[str, Tensor]) -> str:
    return assisted.audit.semantic_sha256(state)


def make_controller(
    graph: Path,
    state: dict[str, Tensor],
    *,
    method: str,
    steps: int,
    device: torch.device,
) -> ConnectomeController:
    neural_dt = 1.0 / (POLICY_HZ * steps) if method == "exponential_euler" else 1.0 / POLICY_HZ
    controller = ConnectomeController(graph, neural_dt=neural_dt).to(device)
    controller.load_state_dict(state, strict=True)
    controller.eval().requires_grad_(False)
    return controller


@torch.inference_mode()
def advance_frame(
    controller: ConnectomeController,
    image: Tensor,
    attitude: Tensor,
    recurrent: Tensor,
    *,
    method: str,
    steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if method == "exponential_euler":
        return rate._advance_frame(
            controller, image, attitude, recurrent, substeps=steps
        )
    if method == "rk4":
        return continuous.advance_rk4_frame(
            controller, image, attitude, recurrent, solver_steps=steps
        )
    raise ValueError(f"unknown optic-motion solver method: {method}")


def relative_type_indices(anatomy: dict[str, Any]) -> dict[str, Tensor]:
    selected = anatomy["selected_indices"]
    return {
        cell_type: torch.from_numpy(np.searchsorted(selected, indices))
        for cell_type, indices in anatomy["type_indices"].items()
    }


@torch.inference_mode()
def neutral_prefix(
    controller: ConnectomeController,
    *,
    method: str,
    steps: int,
    device: torch.device,
) -> tuple[Tensor, bool]:
    image = torch.full((1, 3, HEIGHT, WIDTH), 0.5, device=device)
    attitude = torch.zeros(1, 2, device=device)
    recurrent = controller.initial_state(1, device=device, dtype=image.dtype)
    finite = True
    for _ in range(PREFIX_FRAMES):
        _, recurrent, frame_finite = advance_frame(
            controller,
            image,
            attitude,
            recurrent,
            method=method,
            steps=steps,
        )
        finite = finite and bool(frame_finite)
    return recurrent, finite


@torch.inference_mode()
def evaluate_specs(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    anatomy: dict[str, Any],
    source_state: dict[str, Tensor],
    *,
    method: str,
    steps: int,
    mode: str = "normal",
    device: torch.device,
) -> dict[str, Any]:
    controller = make_controller(
        args.graph, source_state, method=method, steps=steps, device=device
    )
    source_sha = state_sha256(source_state)
    before = rate._source_state_sha256(controller)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    prefix, finite = neutral_prefix(
        controller, method=method, steps=steps, device=device
    )
    selected = torch.from_numpy(anatomy["selected_indices"]).to(device)
    relative = {
        name: value.to(device) for name, value in relative_type_indices(anatomy).items()
    }
    integrated_selected = []
    terminal_selected = []
    integrated_motion_means = []
    integrated_static_means = []
    terminal_motion_means = []
    terminal_static_means = []
    terminal_motors = []
    downstream_means: dict[str, list[Tensor]] = {
        name: [] for name in anatomy["visual_target_type_indices"]
    }
    endpoint_max = 0.0
    for begin in range(0, len(specs), args.block_cases):
        block = specs[begin : begin + args.block_cases]
        sequences = torch.stack([render_sequence(item, mode=mode) for item in block])
        sequences = sequences.permute(1, 0, 2, 3, 4).reshape(
            TOTAL_FRAMES, 4 * len(block), HEIGHT, WIDTH
        )
        endpoint = sequences[-1].reshape(len(block), 4, HEIGHT, WIDTH)
        endpoint_max = max(
            endpoint_max,
            float((endpoint - endpoint[:, :1]).abs().max()),
        )
        recurrent = prefix.expand(4 * len(block), -1).clone()
        attitude = torch.zeros(4 * len(block), 2, device=device)
        selected_sum = torch.zeros(
            len(block), 2, len(selected), device=device, dtype=recurrent.dtype
        )
        moving_type_sum = torch.zeros(
            len(block), 2, len(TYPES), device=device, dtype=recurrent.dtype
        )
        static_type_sum = torch.zeros_like(moving_type_sum)
        last_selected = None
        last_means = None
        output = torch.zeros(4 * len(block), 4, device=device)
        for frame in range(TOTAL_FRAMES):
            gray = sequences[frame].to(device)
            image = gray[:, None].expand(-1, 3, -1, -1)
            output, recurrent, frame_finite = advance_frame(
                controller,
                image,
                attitude,
                recurrent,
                method=method,
                steps=steps,
            )
            finite = finite and bool(frame_finite)
            activity = torch.tanh(recurrent[:, selected]).reshape(
                len(block), 4, len(selected)
            )
            means = torch.stack(
                [activity[:, :, relative[name]].mean(dim=2) for name in TYPES], dim=2
            )
            if frame < MOTION_FRAMES:
                selected_sum += activity[:, :2] - activity[:, 2:]
                moving_type_sum += means[:, :2]
                static_type_sum += means[:, 2:]
            last_selected = activity
            last_means = means
        if last_selected is None or last_means is None:
            raise RuntimeError("optic-motion evaluation produced no frames")
        integrated_selected.append((selected_sum / MOTION_FRAMES).cpu())
        terminal_selected.append(last_selected.cpu())
        integrated_motion_means.append((moving_type_sum / MOTION_FRAMES).cpu())
        integrated_static_means.append((static_type_sum / MOTION_FRAMES).cpu())
        terminal_motion_means.append(last_means[:, :2].cpu())
        terminal_static_means.append(last_means[:, 2:].cpu())
        terminal_motors.append(output.reshape(len(block), 4, 4).cpu())
        full_activity = torch.tanh(recurrent).reshape(len(block), 4, -1)
        for name, indices in anatomy["visual_target_type_indices"].items():
            typed_indices = torch.from_numpy(indices).to(device)
            downstream_means[name].append(
                full_activity[:, :, typed_indices].mean(dim=2).cpu()
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = perf_counter() - started
    after = rate._source_state_sha256(controller)
    result = {
        "method": method,
        "steps_per_camera_frame": steps,
        "neural_step_seconds": 1.0 / (POLICY_HZ * steps),
        "graph_evaluations_per_camera_frame": steps if method == "exponential_euler" else 4 * steps,
        "cases": len(specs),
        "integrated_selected_stationary_subtracted": torch.cat(integrated_selected),
        "terminal_selected_activity": torch.cat(terminal_selected),
        "integrated_motion_type_means": torch.cat(integrated_motion_means),
        "integrated_static_type_means": torch.cat(integrated_static_means),
        "terminal_motion_type_means": torch.cat(terminal_motion_means),
        "terminal_static_type_means": torch.cat(terminal_static_means),
        "terminal_motor_outputs": torch.cat(terminal_motors),
        "terminal_visual_target_type_means": {
            name: torch.cat(values) for name, values in downstream_means.items()
        },
        "opposite_terminal_pixel_difference_maximum": endpoint_max,
        "all_states_and_outputs_finite": finite,
        "source_loaded_exactly": before == source_sha,
        "source_restored": after == before,
        "wall_time_seconds": wall_time,
    }
    del controller
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def type_index(cell_type: str) -> int:
    return TYPES.index(cell_type)


def opponent_values(
    evaluation: dict[str, Any],
    specs: list[dict[str, Any]],
    *,
    reverse_polarity: bool = False,
) -> Tensor:
    means = evaluation["integrated_motion_type_means"] - evaluation[
        "integrated_static_type_means"
    ]
    values = []
    for index, spec in enumerate(specs):
        bright = spec["polarity"] == "bright"
        if reverse_polarity and spec["polarity"] != "mixed":
            bright = not bright
        prefix = "T4" if bright else "T5"
        if spec["polarity"] == "mixed":
            prefixes = ("T4", "T5")
        else:
            prefixes = (prefix,)
        positive, negative = ("c", "d") if spec["axis"] == "vertical" else ("a", "b")
        score = torch.stack(
            [
                means[index, :, type_index(name + positive)]
                - means[index, :, type_index(name + negative)]
                for name in prefixes
            ]
        ).mean(dim=0)
        values.append(score)
    return torch.stack(values)


def numerical_comparison(
    candidate: dict[str, Any], reference: dict[str, Any], specs: list[dict[str, Any]]
) -> dict[str, Any]:
    activity_rms = float(
        (
            candidate["terminal_selected_activity"]
            - reference["terminal_selected_activity"]
        )
        .square()
        .mean()
        .sqrt()
    )
    opponent_rms = float(
        (opponent_values(candidate, specs) - opponent_values(reference, specs))
        .square()
        .mean()
        .sqrt()
    )
    motor_rms = float(
        (
            candidate["terminal_motor_outputs"]
            - reference["terminal_motor_outputs"]
        )
        .square()
        .mean()
        .sqrt()
    )
    identities = all(
        item["all_states_and_outputs_finite"]
        and item["source_loaded_exactly"]
        and item["source_restored"]
        and item["opposite_terminal_pixel_difference_maximum"] == 0.0
        for item in (candidate, reference)
    )
    return {
        "pass": bool(
            identities
            and activity_rms <= ACTIVITY_RMS_LIMIT
            and opponent_rms <= OPPONENT_RMS_LIMIT
            and motor_rms <= MOTOR_RMS_LIMIT
        ),
        "identity_finiteness_and_endpoint_pass": identities,
        "selected_activity_rms_difference": activity_rms,
        "opponent_rms_difference": opponent_rms,
        "terminal_motor_rms_difference": motor_rms,
        "limits": {
            "selected_activity_rms": ACTIVITY_RMS_LIMIT,
            "opponent_rms": OPPONENT_RMS_LIMIT,
            "terminal_motor_rms": MOTOR_RMS_LIMIT,
        },
    }


def compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "integrated_selected_stationary_subtracted",
        "terminal_selected_activity",
        "integrated_motion_type_means",
        "integrated_static_type_means",
        "terminal_motion_type_means",
        "terminal_static_type_means",
        "terminal_motor_outputs",
        "terminal_visual_target_type_means",
    }
    return {name: value for name, value in evaluation.items() if name not in excluded}


def duplicate_control(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    tensor_names = (
        "integrated_selected_stationary_subtracted",
        "terminal_selected_activity",
        "integrated_motion_type_means",
        "integrated_static_type_means",
        "terminal_motion_type_means",
        "terminal_static_type_means",
        "terminal_motor_outputs",
    )
    differences = {
        name: float((first[name] - second[name]).abs().max())
        for name in tensor_names
    }
    first_targets = first["terminal_visual_target_type_means"]
    second_targets = second["terminal_visual_target_type_means"]
    if first_targets.keys() != second_targets.keys():
        raise RuntimeError("duplicate optic-motion replay target types changed")
    target_maximum = max(
        (
            float((first_targets[name] - second_targets[name]).abs().max())
            for name in first_targets
        ),
        default=0.0,
    )
    maximum = max((*differences.values(), target_maximum))
    return {
        "pass": bool(
            maximum == 0.0
            and first["source_loaded_exactly"]
            and first["source_restored"]
            and second["source_loaded_exactly"]
            and second["source_restored"]
        ),
        "tensor_maximum_absolute_differences": differences,
        "visual_target_maximum_absolute_difference": target_maximum,
        "numerical_noise_maximum_absolute": maximum,
    }


def selected_case_indices(
    specs: list[dict[str, Any]], **requirements: Any
) -> list[int]:
    return [
        index
        for index, spec in enumerate(specs)
        if all(spec[name] == value for name, value in requirements.items())
    ]


def _vertical_score(
    type_means: Tensor, *, prefix: str
) -> Tensor:
    return (
        type_means[..., type_index(prefix + "c")]
        - type_means[..., type_index(prefix + "d")]
    )


def subtype_dsi(
    evaluation: dict[str, Any],
    specs: list[dict[str, Any]],
    anatomy: dict[str, Any],
    *,
    cell_type: str,
    polarity: str,
    duplicate_noise: float,
) -> dict[str, Any]:
    cases = selected_case_indices(
        specs, cohort="core", axis="vertical", polarity=polarity
    )
    relative = relative_type_indices(anatomy)[cell_type]
    responses = evaluation["integrated_selected_stationary_subtracted"][cases][
        :, :, relative
    ]
    if cell_type.endswith("c"):
        preferred, null = responses[:, 1].mean(dim=0), responses[:, 0].mean(dim=0)
    else:
        preferred, null = responses[:, 0].mean(dim=0), responses[:, 1].mean(dim=0)
    active_floor = max(ACTIVE_ABSOLUTE_FLOOR, 10.0 * duplicate_noise)
    active = torch.maximum(preferred.abs(), null.abs()) > active_floor
    dsi = (preferred - null) / (
        preferred.abs() + null.abs() + max(active_floor, 1.0e-12)
    )
    active_fraction = float(active.float().mean())
    median = float(dsi[active].median()) if bool(active.any()) else math.nan
    passed = bool(
        math.isfinite(median)
        and active_fraction >= ACTIVE_FRACTION_MINIMUM
        and median >= MEDIAN_DSI_MINIMUM
    )
    return {
        "pass": passed,
        "cells": len(relative),
        "active_cells": int(active.sum()),
        "active_fraction": active_fraction,
        "active_floor": active_floor,
        "median_dsi_active": median,
        "mean_preferred_response": float(preferred.mean()),
        "mean_null_response": float(null.mean()),
    }


def tuning_decision(
    evaluation: dict[str, Any],
    specs: list[dict[str, Any]],
    anatomy: dict[str, Any],
    duplicate: dict[str, Any],
    reverse_evaluation: dict[str, Any],
    reverse_items: list[dict[str, Any]],
) -> dict[str, Any]:
    responses_by_window = {
        "integrated": evaluation["integrated_motion_type_means"]
        - evaluation["integrated_static_type_means"],
        "terminal": evaluation["terminal_motion_type_means"]
        - evaluation["terminal_static_type_means"],
    }
    static = evaluation["integrated_static_type_means"]
    sign_metrics = {}
    moving_contrasts = []
    static_contrasts = []
    for window, window_responses in responses_by_window.items():
        sign_metrics[window] = {}
        for prefix, polarity, label in (
            ("T4", "bright", "T4_ON"),
            ("T5", "dark", "T5_OFF"),
        ):
            cases = selected_case_indices(
                specs, cohort="core", axis="vertical", polarity=polarity
            )
            response_score = _vertical_score(window_responses[cases], prefix=prefix)
            upward_fraction = float((response_score[:, 1] > 0.0).float().mean())
            downward_fraction = float((response_score[:, 0] < 0.0).float().mean())
            sign_metrics[window][label] = {
                "upward_correct_fraction": upward_fraction,
                "downward_correct_fraction": downward_fraction,
                "upward_pass": upward_fraction >= SIGN_FRACTION_MINIMUM,
                "downward_pass": downward_fraction >= SIGN_FRACTION_MINIMUM,
                "pairs": len(cases),
            }
            if window == "integrated":
                static_score = _vertical_score(static[cases], prefix=prefix)
                moving_contrasts.append(response_score[:, 1] - response_score[:, 0])
                static_contrasts.append(static_score[:, 1] - static_score[:, 0])
    moving_vector = torch.cat(moving_contrasts)
    static_vector = torch.cat(static_contrasts)
    moving_rms = float(moving_vector.square().mean().sqrt())
    stationary_rms = float(static_vector.square().mean().sqrt())
    stationary_ratio = stationary_rms / max(moving_rms, 1.0e-30)

    dsi = {
        "T4c": subtype_dsi(
            evaluation,
            specs,
            anatomy,
            cell_type="T4c",
            polarity="bright",
            duplicate_noise=duplicate["numerical_noise_maximum_absolute"],
        ),
        "T4d": subtype_dsi(
            evaluation,
            specs,
            anatomy,
            cell_type="T4d",
            polarity="bright",
            duplicate_noise=duplicate["numerical_noise_maximum_absolute"],
        ),
        "T5c": subtype_dsi(
            evaluation,
            specs,
            anatomy,
            cell_type="T5c",
            polarity="dark",
            duplicate_noise=duplicate["numerical_noise_maximum_absolute"],
        ),
        "T5d": subtype_dsi(
            evaluation,
            specs,
            anatomy,
            cell_type="T5d",
            polarity="dark",
            duplicate_noise=duplicate["numerical_noise_maximum_absolute"],
        ),
    }

    normal_by_case = {
        spec["case"]: values
        for spec, values in zip(specs, opponent_values(evaluation, specs), strict=True)
    }
    # Literal time reversal changes an ON edge into an OFF edge and vice versa, so
    # compare the reversed history in the corresponding opposite T4/T5 pathway.
    reversed_values = opponent_values(
        reverse_evaluation, reverse_items, reverse_polarity=True
    )
    reverse_noise_floor = max(
        ACTIVE_ABSOLUTE_FLOOR,
        10.0 * duplicate["numerical_noise_maximum_absolute"],
    )
    reverse_passes = []
    reverse_strata: dict[str, list[bool]] = {}
    for spec, values in zip(reverse_items, reversed_values, strict=True):
        normal_values = normal_by_case[spec["case"]]
        for branch, original_direction in enumerate(("down", "up")):
            reversed_direction = "up" if original_direction == "down" else "down"
            normal_score = float(normal_values[branch])
            reverse_score = float(values[branch])
            correct = bool(
                math.isfinite(normal_score)
                and math.isfinite(reverse_score)
                and abs(normal_score) > reverse_noise_floor
                and abs(reverse_score) > reverse_noise_floor
                and normal_score * reverse_score < 0.0
            )
            reverse_passes.append(correct)
            key = (
                f"{spec['polarity']}_{original_direction}_to_{reversed_direction}"
            )
            reverse_strata.setdefault(key, []).append(correct)
    reverse_fraction = sum(reverse_passes) / len(reverse_passes)
    reverse_by_stratum = {
        name: {
            "branches": len(values),
            "correct": sum(values),
            "correct_fraction": sum(values) / len(values),
        }
        for name, values in sorted(reverse_strata.items())
    }

    generalization = {}
    for axis in ("vertical", "horizontal"):
        cases = selected_case_indices(specs, cohort="generalization", axis=axis)
        scores = opponent_values(evaluation, specs)[cases]
        generalization[axis] = {
            "pairs": len(cases),
            "opponent_contrast_rms": float(
                (scores[:, 1] - scores[:, 0]).square().mean().sqrt()
            ),
            "positive_pair_fraction": float(
                ((scores[:, 1] - scores[:, 0]) > 0.0).float().mean()
            ),
        }

    reasons = []
    if not duplicate["pass"]:
        reasons.append("exact duplicate neural replay differed")
    for window, groups in sign_metrics.items():
        for label, values in groups.items():
            if not values["upward_pass"]:
                reasons.append(
                    f"{window} {label} upward sign fraction was below 90%"
                )
            if not values["downward_pass"]:
                reasons.append(
                    f"{window} {label} downward sign fraction was below 90%"
                )
    for cell_type, values in dsi.items():
        if not values["pass"]:
            reasons.append(f"{cell_type} active fraction or median DSI failed")
    if stationary_ratio > STATIONARY_RATIO_MAXIMUM:
        reasons.append("stationary opponent RMS exceeded 10% of moving RMS")
    if reverse_fraction < SIGN_FRACTION_MINIMUM:
        reasons.append("reversed-history sign inversion was below 90%")
    evaluation_controls_pass = bool(
        evaluation["all_states_and_outputs_finite"]
        and evaluation["source_loaded_exactly"]
        and evaluation["source_restored"]
        and evaluation["opposite_terminal_pixel_difference_maximum"] == 0.0
        and reverse_evaluation["all_states_and_outputs_finite"]
        and reverse_evaluation["source_loaded_exactly"]
        and reverse_evaluation["source_restored"]
    )
    if not evaluation_controls_pass:
        reasons.append("source identity, endpoint or finiteness control failed")
    integrated_sign_pass = all(
        values[direction]
        for values in sign_metrics["integrated"].values()
        for direction in ("upward_pass", "downward_pass")
    )
    post_offset_retention_pass = all(
        values[direction]
        for values in sign_metrics["terminal"].values()
        for direction in ("upward_pass", "downward_pass")
    )
    motion_window_evidence_pass = bool(
        duplicate["pass"]
        and integrated_sign_pass
        and all(values["pass"] for values in dsi.values())
        and stationary_ratio <= STATIONARY_RATIO_MAXIMUM
        and reverse_fraction >= SIGN_FRACTION_MINIMUM
        and evaluation_controls_pass
    )
    return {
        "pass": not reasons,
        "reasons": reasons,
        "motion_window_evidence_pass": motion_window_evidence_pass,
        "post_offset_retention_pass": post_offset_retention_pass,
        "vertical_sign": sign_metrics,
        "subtype_dsi": dsi,
        "moving_opponent_rms": moving_rms,
        "stationary_opponent_rms": stationary_rms,
        "stationary_to_moving_ratio": stationary_ratio,
        "reverse_noise_floor": reverse_noise_floor,
        "reverse_sign_inversion_fraction": reverse_fraction,
        "reverse_sign_by_polarity_and_original_direction": reverse_by_stratum,
        "generalization": generalization,
    }


@torch.inference_mode()
def evaluate_pair_mean_intervention(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    anatomy: dict[str, Any],
    source_state: dict[str, Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    controller = make_controller(
        args.graph,
        source_state,
        method="exponential_euler",
        steps=REFERENCE_SUBSTEPS,
        device=device,
    )
    before = rate._source_state_sha256(controller)
    prefix, finite = neutral_prefix(
        controller,
        method="exponential_euler",
        steps=REFERENCE_SUBSTEPS,
        device=device,
    )
    selected = torch.from_numpy(anatomy["selected_indices"]).to(device)
    target_indices = {
        name: torch.from_numpy(values).to(device)
        for name, values in anatomy["visual_target_type_indices"].items()
    }
    outputs = []
    downstream: dict[str, list[Tensor]] = {name: [] for name in target_indices}
    pair_activity_preservation_maximum = torch.zeros((), device=device)
    pair_mean_activity_maximum_absolute = torch.zeros((), device=device)
    pair_mean_activity_domain_pass = True
    started = perf_counter()
    for begin in range(0, len(specs), args.block_cases):
        block = specs[begin : begin + args.block_cases]
        sequences = torch.stack([render_sequence(item)[:, :2] for item in block])
        sequences = sequences.permute(1, 0, 2, 3, 4).reshape(
            TOTAL_FRAMES, 2 * len(block), HEIGHT, WIDTH
        )
        recurrent = prefix.expand(2 * len(block), -1).clone()
        attitude = torch.zeros(2 * len(block), 2, device=device)
        output = torch.zeros(2 * len(block), 4, device=device)
        for frame in range(TOTAL_FRAMES):
            gray = sequences[frame].to(device)
            image = gray[:, None].expand(-1, 3, -1, -1)
            for _ in range(REFERENCE_SUBSTEPS):
                output, recurrent = controller(image, attitude, recurrent)
                shaped = recurrent.reshape(len(block), 2, -1)
                pair_mean_activity = torch.tanh(shaped[:, :, selected]).mean(dim=1)
                pair_mean_activity_maximum_absolute = torch.maximum(
                    pair_mean_activity_maximum_absolute,
                    pair_mean_activity.abs().max(),
                )
                pair_mean_activity_domain_pass = pair_mean_activity_domain_pass and bool(
                    torch.isfinite(pair_mean_activity).all()
                    & (pair_mean_activity.abs() < 1.0).all()
                )
                pair_mean_state = torch.atanh(
                    pair_mean_activity.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
                )
                shaped[:, 0, selected] = pair_mean_state
                shaped[:, 1, selected] = pair_mean_state
                intervened_activity = torch.tanh(shaped[:, :, selected])
                pair_activity_preservation_maximum = torch.maximum(
                    pair_activity_preservation_maximum,
                    (intervened_activity - pair_mean_activity[:, None]).abs().max(),
                )
                output = controller.motor_drive(recurrent)
                finite = finite and bool(
                    torch.isfinite(recurrent).all() & torch.isfinite(output).all()
                )
        shaped_activity = torch.tanh(recurrent).reshape(len(block), 2, -1)
        for name, indices in target_indices.items():
            downstream[name].append(shaped_activity[:, :, indices].mean(dim=2).cpu())
        outputs.append(output.reshape(len(block), 2, 4).cpu())
    after = rate._source_state_sha256(controller)
    return {
        "terminal_motor_outputs": torch.cat(outputs),
        "terminal_visual_target_type_means": {
            name: torch.cat(values) for name, values in downstream.items()
        },
        "all_states_and_outputs_finite": finite and pair_mean_activity_domain_pass,
        "source_loaded_exactly": before == state_sha256(source_state),
        "source_restored": after == before,
        "pair_mean_activity_domain_pass": pair_mean_activity_domain_pass,
        "pair_mean_activity_maximum_absolute": float(
            pair_mean_activity_maximum_absolute
        ),
        "pair_activity_preservation_maximum_absolute_error": float(
            pair_activity_preservation_maximum
        ),
        "wall_time_seconds": perf_counter() - started,
    }


def intervention_report(
    baseline: dict[str, Any],
    all_specs: list[dict[str, Any]],
    intervention: dict[str, Any],
    intervention_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    positions = {spec["case"]: index for index, spec in enumerate(all_specs)}
    baseline_motor = torch.stack(
        [
            baseline["terminal_motor_outputs"][positions[item["case"]], :2]
            for item in intervention_specs
        ]
    )
    candidate_motor = intervention["terminal_motor_outputs"]
    baseline_contrast = baseline_motor[:, 1] - baseline_motor[:, 0]
    candidate_contrast = candidate_motor[:, 1] - candidate_motor[:, 0]
    axes = {}
    for axis, name in enumerate(("roll", "pitch", "yaw", "throttle")):
        source_rms = float(baseline_contrast[:, axis].square().mean().sqrt())
        intervened_rms = float(candidate_contrast[:, axis].square().mean().sqrt())
        axes[name] = {
            "baseline_contrast_rms": source_rms,
            "intervened_contrast_rms": intervened_rms,
            "rms_ratio": intervened_rms / max(source_rms, 1.0e-30),
            "common_output_drift_rms": float(
                (
                    candidate_motor[:, :, axis].mean(dim=1)
                    - baseline_motor[:, :, axis].mean(dim=1)
                )
                .square()
                .mean()
                .sqrt()
            ),
        }
    downstream = {}
    baseline_targets = baseline["terminal_visual_target_type_means"]
    for name, values in intervention["terminal_visual_target_type_means"].items():
        source = torch.stack(
            [baseline_targets[name][positions[item["case"]], :2] for item in intervention_specs]
        )
        source_contrast = source[:, 1] - source[:, 0]
        candidate_target_contrast = values[:, 1] - values[:, 0]
        source_rms = float(source_contrast.square().mean().sqrt())
        candidate_rms = float(candidate_target_contrast.square().mean().sqrt())
        downstream[name] = {
            "baseline_contrast_rms": source_rms,
            "intervened_contrast_rms": candidate_rms,
            "rms_ratio": candidate_rms / max(source_rms, 1.0e-30),
            "common_activity_drift_rms": float(
                (values.mean(dim=1) - source.mean(dim=1)).square().mean().sqrt()
            ),
        }
    return {
        "diagnostic_only": True,
        "cases": len(intervention_specs),
        "motor_axes": axes,
        "visual_targets": downstream,
        "all_states_and_outputs_finite": intervention[
            "all_states_and_outputs_finite"
        ],
        "source_loaded_exactly": intervention["source_loaded_exactly"],
        "source_restored": intervention["source_restored"],
        "pair_mean_activity_domain_pass": intervention[
            "pair_mean_activity_domain_pass"
        ],
        "pair_mean_activity_maximum_absolute": intervention[
            "pair_mean_activity_maximum_absolute"
        ],
        "pair_activity_preservation_maximum_absolute_error": intervention[
            "pair_activity_preservation_maximum_absolute_error"
        ],
        "wall_time_seconds": intervention["wall_time_seconds"],
    }


def _write_or_validate_start(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open() as stream:
            if json.load(stream) != payload:
                raise SystemExit("optic-motion start marker mismatch")
        return
    assisted._atomic_json_save(payload, path)


def main() -> int:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    if report_path.is_file():
        raise SystemExit("the frozen optic-motion audit already has a terminal report")
    if args.block_cases < 1:
        raise SystemExit("--block-cases must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    input_hashes = validate_inputs(args)
    anatomy = anatomy_manifest(args)
    specs = stimulus_specs()
    stimuli = stimulus_manifest(specs)
    if (
        stimuli["opposite_terminal_maximum_difference"] != 0.0
        or stimuli["duplicate_render_maximum_difference"] != 0.0
        or len(numerical_specs(specs)) != NUMERICAL_CASES
        or len(reverse_specs(specs)) != REVERSE_CASES
    ):
        raise SystemExit("optic-motion stimulus controls failed before execution")
    source_state = load_checkpoint(args.checkpoint)
    source_sha = state_sha256(source_state)
    start = {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha,
        "anatomy": report_anatomy(anatomy),
        "stimulus_manifest": stimuli,
        "device": str(device),
        "block_cases": args.block_cases,
    }
    _write_or_validate_start(args.output_dir / "start.json", start)

    classification = "optic_motion_exception_failed_closed"
    passed = False
    exception: dict[str, str] | None = None
    numerical: dict[str, Any] | None = None
    duplicate: dict[str, Any] | None = None
    tuning: dict[str, Any] | None = None
    intervention: dict[str, Any] | None = None
    primary_summary: dict[str, Any] | None = None
    reverse_summary: dict[str, Any] | None = None
    executed_evaluations: list[dict[str, Any]] = []
    try:
        subset = numerical_specs(specs)
        print(json.dumps({"stage": "numerical_K32"}), flush=True)
        k32 = evaluate_specs(
            args,
            subset,
            anatomy,
            source_state,
            method="exponential_euler",
            steps=REFERENCE_SUBSTEPS,
            device=device,
        )
        print(json.dumps({"stage": "numerical_K64"}), flush=True)
        k64 = evaluate_specs(
            args,
            subset,
            anatomy,
            source_state,
            method="exponential_euler",
            steps=FINE_SUBSTEPS,
            device=device,
        )
        executed_evaluations.extend((k32, k64))
        reference_comparison = numerical_comparison(k32, k64, subset)
        numerical = {
            "K32": compact_evaluation(k32),
            "K64": compact_evaluation(k64),
            "K32_vs_K64": reference_comparison,
            "rk4": {},
        }
        if not reference_comparison["pass"]:
            classification = "optic_motion_numerical_gate_failed"
        else:
            for rk_steps in RK4_STEPS:
                print(
                    json.dumps({"stage": "numerical_rk4", "steps": rk_steps}),
                    flush=True,
                )
                rk = evaluate_specs(
                    args,
                    subset,
                    anatomy,
                    source_state,
                    method="rk4",
                    steps=rk_steps,
                    device=device,
                )
                executed_evaluations.append(rk)
                numerical["rk4"][str(rk_steps)] = {
                    "evaluation": compact_evaluation(rk),
                    "vs_K64": numerical_comparison(rk, k64, subset),
                }

            print(json.dumps({"stage": "duplicate_K32"}), flush=True)
            duplicate_replay = evaluate_specs(
                args,
                subset,
                anatomy,
                source_state,
                method="exponential_euler",
                steps=REFERENCE_SUBSTEPS,
                device=device,
            )
            executed_evaluations.append(duplicate_replay)
            duplicate = duplicate_control(k32, duplicate_replay)

            print(json.dumps({"stage": "primary_K32"}), flush=True)
            primary = evaluate_specs(
                args,
                specs,
                anatomy,
                source_state,
                method="exponential_euler",
                steps=REFERENCE_SUBSTEPS,
                device=device,
            )
            executed_evaluations.append(primary)
            primary_summary = compact_evaluation(primary)
            reversed_items = reverse_specs(specs)
            print(json.dumps({"stage": "reverse_K32"}), flush=True)
            reversed_evaluation = evaluate_specs(
                args,
                reversed_items,
                anatomy,
                source_state,
                method="exponential_euler",
                steps=REFERENCE_SUBSTEPS,
                mode="reverse",
                device=device,
            )
            executed_evaluations.append(reversed_evaluation)
            reverse_summary = compact_evaluation(reversed_evaluation)
            tuning = tuning_decision(
                primary,
                specs,
                anatomy,
                duplicate,
                reversed_evaluation,
                reversed_items,
            )

            vertical = [
                item
                for item in specs
                if item["cohort"] == "core" and item["axis"] == "vertical"
            ]
            print(json.dumps({"stage": "pair_mean_intervention"}), flush=True)
            intervention_evaluation = evaluate_pair_mean_intervention(
                args, vertical, anatomy, source_state, device=device
            )
            executed_evaluations.append(intervention_evaluation)
            intervention = intervention_report(
                primary, specs, intervention_evaluation, vertical
            )
            if tuning["pass"]:
                classification = "frozen_optic_motion_module_qualified"
                passed = True
            else:
                classification = "frozen_optic_motion_module_uncommissioned"
    except Exception as error:
        exception = {"type": type(error).__name__, "message": str(error)}

    source_restored = bool(
        exception is None
        and executed_evaluations
        and state_sha256(source_state) == source_sha
        and all(
            item.get("source_loaded_exactly") and item.get("source_restored")
            for item in executed_evaluations
        )
    )
    execution_controls_pass = bool(
        source_restored
        and all(item.get("all_states_and_outputs_finite") for item in executed_evaluations)
    )
    routing_authorized = bool(passed and execution_controls_pass)
    commissioning_authorized = bool(
        execution_controls_pass
        and numerical is not None
        and numerical["K32_vs_K64"]["pass"]
        and duplicate is not None
        and duplicate["pass"]
        and tuning is not None
        and not tuning["pass"]
    )
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "exception": exception,
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha,
        "anatomy": report_anatomy(anatomy),
        "stimulus_manifest": stimuli,
        "numerical": numerical,
        "duplicate_control": duplicate,
        "primary_evaluation": primary_summary,
        "reverse_evaluation": reverse_summary,
        "tuning": tuning,
        "pair_mean_intervention": intervention,
        "source_restored": source_restored,
        "execution_controls_pass": execution_controls_pass,
        "motion_output_routing_preregistration_authorized": routing_authorized,
        "local_motion_commissioning_preregistration_authorized": (
            commissioning_authorized
        ),
        "candidate_retained": False,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": perf_counter() - started,
    }
    if not execution_controls_pass:
        report["passed"] = False
        report["motion_output_routing_preregistration_authorized"] = False
        report["local_motion_commissioning_preregistration_authorized"] = False
        if exception is None:
            report["classification"] = (
                "optic_motion_source_identity_failed"
                if not source_restored
                else "optic_motion_execution_control_failed"
            )
    assisted._atomic_json_save(report, report_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "classification": report["classification"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
