"""Reusable differentiable machinery for vertical T4/T5 commissioning."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preregister_vertical_motion_commissioning as registration  # noqa: E402

from flydrone.connectome_data import _read_annotations  # noqa: E402
from flydrone.hover import ConnectomeController  # noqa: E402

WINDOWS = ("integrated", "terminal")
PATHWAYS = ("T4", "T5")
DIRECTION_MARGIN = 0.30
NORMALIZATION_FLOOR = 1.0e-3
REVERSE_WEIGHT = 0.25
ACTIVITY_WEIGHT = 0.25
REGULARIZATION_WEIGHT = 1.0e-3


def semantic_sha256(value: Any) -> str:
    """Hash nested JSON values and tensors with exact dtype/shape/bytes semantics."""

    hasher = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            hasher.update(b"tensor\0")
            hasher.update(str(tensor.dtype).encode() + b"\0")
            hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            hasher.update(b"\0")
            hasher.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, dict):
            hasher.update(b"dict\0")
            for key in sorted(item):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(b"sequence\0")
            for value in item:
                visit(value)
        else:
            hasher.update(b"json\0")
            hasher.update(
                json.dumps(item, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
            )
            hasher.update(b"\0")

    visit(value)
    return hasher.hexdigest()


def load_source_checkpoint(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {name: value.detach().cpu().clone() for name, value in payload["controller"].items()}


def load_registered_manifest(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        payload = json.load(stream)
    if payload.get("experiment") != registration.EXPERIMENT:
        raise SystemExit("vertical-motion registration experiment changed")
    if registration.semantic_sha256(payload) != registration.EXPECTED_MANIFEST_SEMANTIC_SHA256:
        raise SystemExit("vertical-motion registration semantic hash changed")
    return payload


def anatomy_arrays(graph_path: Path, annotations_path: Path) -> dict[str, np.ndarray]:
    graph = np.load(graph_path)
    node_ids = graph["node_ids"]
    edge_pre = graph["edge_pre"]
    edge_post = graph["edge_post"]
    annotations = _read_annotations(annotations_path)
    rows = {int(body): row for row, body in enumerate(annotations["bodyId"])}
    node_types = np.asarray(
        [str(annotations["type"][rows[int(body)]]) for body in node_ids], dtype=object
    )
    edge_group = np.full(len(edge_pre), -1, dtype=np.int64)
    for group_index, (target, source, _) in enumerate(registration.GROUPS):
        selected = (node_types[edge_pre] == source) & (node_types[edge_post] == target)
        edge_group[selected] = group_index
    edge_indices = np.flatnonzero(edge_group >= 0).astype(np.int64)
    edge_group_pairs = np.column_stack((edge_indices, edge_group[edge_indices])).astype(np.int64)
    target_indices = np.flatnonzero(np.isin(node_types, registration.TARGET_SUBTYPES)).astype(
        np.int64
    )
    target_subtypes = np.asarray(
        [registration.TARGET_SUBTYPES.index(str(node_types[index])) for index in target_indices],
        dtype=np.int64,
    )
    target_subtype_pairs = np.column_stack((target_indices, target_subtypes)).astype(np.int64)
    observed = {
        "edge_indices": registration.int64_array_sha256(edge_indices),
        "edge_group_pairs": registration.int64_array_sha256(edge_group_pairs),
        "target_indices": registration.int64_array_sha256(target_indices),
        "target_subtype_pairs": registration.int64_array_sha256(target_subtype_pairs),
    }
    expected = {
        "edge_indices": registration.EXPECTED_EDGE_INDICES_SHA256,
        "edge_group_pairs": registration.EXPECTED_EDGE_GROUP_PAIRS_SHA256,
        "target_indices": registration.EXPECTED_TARGET_INDICES_SHA256,
        "target_subtype_pairs": registration.EXPECTED_TARGET_SUBTYPE_PAIRS_SHA256,
    }
    if observed != expected:
        raise SystemExit("preflight anatomy arrays do not match the committed manifest")
    return {
        "edge_indices": edge_indices,
        "edge_groups": edge_group[edge_indices],
        "target_indices": target_indices,
        "target_subtypes": target_subtypes,
    }


def _texture(spec: dict[str, Any]) -> np.ndarray:
    seed = int(spec["realization_seed_uint64_hex"], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    white = rng.standard_normal((registration.HEIGHT, registration.WIDTH), dtype=np.float64)
    spectrum = np.fft.rfft2(white)
    vertical = np.fft.fftfreq(registration.HEIGHT)[:, None]
    horizontal = np.fft.rfftfreq(registration.WIDTH)[None, :]
    radius = np.sqrt(vertical * vertical + horizontal * horizontal)
    passband = (radius >= 1.0 / 96.0) & (radius <= 1.0 / 8.0)
    passband[0, 0] = False
    spatial = np.fft.irfft2(spectrum * passband, s=white.shape)
    maximum = np.max(np.abs(spatial))
    if not math.isfinite(float(maximum)) or maximum <= 0.0:
        raise RuntimeError("band-limited texture normalization is invalid")
    return np.asarray(0.5 + 0.32 * spatial / maximum, dtype=np.float32)


def motion_frame(spec: dict[str, Any], branch: int, frame: int) -> Tensor:
    if frame == registration.MOTION_FRAMES:
        return torch.full((registration.HEIGHT, registration.WIDTH), 0.5, dtype=torch.float32)
    branch_sign = 1 if branch == 0 else -1
    speed = int(spec["speed_pixels_per_frame"])
    if spec["family"] == "polarity_preserving_edge":
        numerator = int(spec["phase_row_numerator"])
        center = 60.0 + numerator / registration.PHASE_DENOMINATOR
        center += branch_sign * speed * (frame - (registration.MOTION_FRAMES - 1) / 2.0)
        rows = torch.arange(registration.HEIGHT, dtype=torch.float32)[:, None]
        swept = (branch_sign * (rows - center) <= 0.0).to(torch.float32)
        swept = swept.expand(-1, registration.WIDTH)
        if spec["polarity"] == "ON":
            return 0.2 + 0.6 * swept
        if spec["polarity"] == "OFF":
            return 0.8 - 0.6 * swept
        raise ValueError("edge stimulus has an unknown polarity")
    if spec["family"] == "band_limited_texture":
        base = _texture(spec)
        shifted = np.roll(base, shift=branch_sign * speed * frame, axis=0)
        return torch.from_numpy(shifted.copy())
    raise ValueError("unknown vertical-motion stimulus family")


def render_sequence(spec: dict[str, Any], *, reverse: bool = False) -> Tensor:
    moving = []
    for frame in range(registration.MOTION_FRAMES):
        source_frame = registration.MOTION_FRAMES - 1 - frame if reverse else frame
        moving.append(torch.stack([motion_frame(spec, branch, source_frame) for branch in (0, 1)]))
    stationary_frame = registration.MOTION_FRAMES - 1 if reverse else 0
    stationary = torch.stack([motion_frame(spec, branch, stationary_frame) for branch in (0, 1)])
    terminal = torch.stack(
        [motion_frame(spec, branch, registration.MOTION_FRAMES) for branch in (0, 1)]
    )
    moving_tensor = torch.stack(moving + [terminal])
    stationary_tensor = torch.stack(
        [stationary for _ in range(registration.MOTION_FRAMES)] + [terminal]
    )
    return torch.cat((moving_tensor, stationary_tensor), dim=1).contiguous()


def rendered_subset_sha256(specs: list[dict[str, Any]], indices: tuple[int, ...]) -> str:
    hasher = hashlib.sha256()
    for index in indices:
        for reverse in (False, True):
            values = render_sequence(specs[index], reverse=reverse).numpy()
            hasher.update(np.ascontiguousarray(values).view(np.uint8))
    return hasher.hexdigest()


class CommissionedController(nn.Module):
    """Apply 24 shared local parameters without mutating the source controller."""

    def __init__(
        self,
        source: ConnectomeController,
        anatomy: dict[str, np.ndarray],
    ) -> None:
        super().__init__()
        source.eval().requires_grad_(False)
        self.source = source
        self.register_buffer("selected_edge_indices", torch.from_numpy(anatomy["edge_indices"]))
        self.register_buffer("selected_edge_groups", torch.from_numpy(anatomy["edge_groups"]))
        self.register_buffer("selected_target_indices", torch.from_numpy(anatomy["target_indices"]))
        self.register_buffer(
            "selected_target_subtypes", torch.from_numpy(anatomy["target_subtypes"])
        )
        self.gain = nn.Parameter(torch.ones(16))
        self.bias_offset = nn.Parameter(torch.zeros(4))
        self.tau_ratio = nn.Parameter(torch.ones(4))

    @property
    def n_nodes(self) -> int:
        return self.source.n_nodes

    def initial_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.source.initial_state(batch, device=device, dtype=dtype)

    def motor_drive(self, state: Tensor) -> Tensor:
        return self.source.motor_drive(state)

    def parameter_values(self) -> dict[str, Tensor]:
        return {
            "gain": self.gain.detach().cpu().clone(),
            "bias_offset": self.bias_offset.detach().cpu().clone(),
            "tau_ratio": self.tau_ratio.detach().cpu().clone(),
        }

    def load_parameter_values(self, values: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for name, parameter in (
                ("gain", self.gain),
                ("bias_offset", self.bias_offset),
                ("tau_ratio", self.tau_ratio),
            ):
                parameter.copy_(values[name].to(parameter))

    def project_parameters(self) -> None:
        with torch.no_grad():
            self.gain.clamp_(0.25, 4.0)
            self.bias_offset.clamp_(-0.25, 0.25)
            self.tau_ratio.clamp_(0.5, 2.0)

    def identity_restored(self) -> bool:
        return bool(
            torch.equal(self.gain.detach(), torch.ones_like(self.gain))
            and torch.equal(self.bias_offset.detach(), torch.zeros_like(self.bias_offset))
            and torch.equal(self.tau_ratio.detach(), torch.ones_like(self.tau_ratio))
        )

    def forward(self, image: Tensor, roll_pitch: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        activity = torch.tanh(state)
        source_weight = self.source.edge_sign * self.source.edge_magnitude
        messages = activity[:, self.source.edge_pre] * source_weight
        recurrent = torch.zeros_like(state).index_add(1, self.source.edge_post, messages)

        selected = self.selected_edge_indices
        delta_weight = source_weight[selected] * (self.gain[self.selected_edge_groups] - 1.0)
        delta_messages = activity[:, self.source.edge_pre[selected]] * delta_weight
        recurrent = recurrent.index_add(1, self.source.edge_post[selected], delta_messages)

        drive = recurrent + self.source.bias + self.source.sensory_drive(image, roll_pitch)
        batch_bias = self.bias_offset[self.selected_target_subtypes][None].expand(
            image.shape[0], -1
        )
        drive = drive.index_add(1, self.selected_target_indices, batch_bias)
        target = 5.0 * torch.tanh(drive / 5.0)

        source_tau = self.source.time_constant
        source_alpha = 1.0 - torch.exp(-self.source.neural_dt / source_tau)
        next_state = state + source_alpha * (target - state)
        selected_target = self.selected_target_indices
        candidate_tau = (
            source_tau[selected_target] * self.tau_ratio[self.selected_target_subtypes]
        ).clamp(0.010, 0.250)
        candidate_alpha = 1.0 - torch.exp(-self.source.neural_dt / candidate_tau)
        alpha_delta = candidate_alpha - source_alpha[selected_target]
        state_delta = alpha_delta * (target[:, selected_target] - state[:, selected_target])
        next_state = next_state.index_add(1, selected_target, state_delta)
        return self.source.motor_drive(next_state), next_state


def make_source_controller(
    graph: Path, source_state: dict[str, Tensor], *, device: torch.device
) -> ConnectomeController:
    controller = ConnectomeController(
        graph,
        neural_dt=1.0 / (registration.CAMERA_HZ * registration.CNS_SUBSTEPS_PER_FRAME),
    ).to(device)
    controller.load_state_dict(source_state, strict=True)
    controller.eval().requires_grad_(False)
    return controller


def _advance_frame(
    controller: nn.Module,
    image: Tensor,
    state: Tensor,
    *,
    checkpoint_frames: bool,
) -> tuple[Tensor, Tensor]:
    attitude = torch.zeros(image.shape[0], 2, device=image.device, dtype=image.dtype)

    def integrate(frame_state: Tensor) -> tuple[Tensor, Tensor]:
        output = torch.zeros(image.shape[0], 4, device=image.device, dtype=image.dtype)
        for _ in range(registration.CNS_SUBSTEPS_PER_FRAME):
            output, frame_state = controller(image, attitude, frame_state)
        return output, frame_state

    if checkpoint_frames and torch.is_grad_enabled():
        return activation_checkpoint(
            integrate,
            state,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return integrate(state)


def evaluate_pair(
    controller: nn.Module,
    sequence: Tensor,
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
    checkpoint_frames: bool,
    prefix_state: Tensor | None = None,
) -> dict[str, Tensor]:
    if prefix_state is None:
        prefix_state = neutral_prefix(
            controller, device=device, checkpoint_frames=checkpoint_frames
        )
    state = prefix_state.expand(4, -1).clone()
    selected = torch.from_numpy(anatomy["target_indices"]).to(device)
    integrated_response = torch.zeros(2, len(selected), device=device)
    integrated_static = torch.zeros_like(integrated_response)
    terminal_response = torch.empty_like(integrated_response)
    terminal_static = torch.empty_like(integrated_response)
    for frame in range(registration.MOTION_FRAMES + registration.TERMINAL_FRAMES):
        gray = sequence[frame].to(device)
        image = gray[:, None].expand(-1, 3, -1, -1)
        output, state = _advance_frame(
            controller, image, state, checkpoint_frames=checkpoint_frames
        )
        selected_activity = torch.tanh(state[:, selected])
        moving = selected_activity[:2]
        stationary = selected_activity[2:]
        response = moving - stationary
        if frame < registration.MOTION_FRAMES:
            integrated_response = integrated_response + response
            integrated_static = integrated_static + stationary
        else:
            terminal_response = response
            terminal_static = stationary
    return {
        "integrated_response": integrated_response / registration.MOTION_FRAMES,
        "integrated_static": integrated_static / registration.MOTION_FRAMES,
        "terminal_response": terminal_response,
        "terminal_static": terminal_static,
        "terminal_motor": output.reshape(4, 4),
    }


def neutral_prefix(
    controller: nn.Module,
    *,
    device: torch.device,
    checkpoint_frames: bool,
) -> Tensor:
    neutral = torch.full(
        (1, 3, registration.HEIGHT, registration.WIDTH),
        0.5,
        device=device,
    )
    state = controller.initial_state(1, device=device, dtype=neutral.dtype)
    for _ in range(registration.PREFIX_FRAMES):
        _, state = _advance_frame(controller, neutral, state, checkpoint_frames=checkpoint_frames)
    return state


def response_tree_cpu(response: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in response.items()}


def pathway_for_spec(spec: dict[str, Any]) -> tuple[str, ...]:
    if spec["family"] == "band_limited_texture":
        return PATHWAYS
    if spec["polarity"] == "ON":
        return ("T4",)
    if spec["polarity"] == "OFF":
        return ("T5",)
    raise ValueError("edge specification has invalid polarity")


def reverse_pathway(spec: dict[str, Any], pathway: str) -> str:
    if spec["family"] == "band_limited_texture":
        return pathway
    return "T5" if pathway == "T4" else "T4"


def relative_subtype_indices(anatomy: dict[str, np.ndarray]) -> dict[str, Tensor]:
    subtype = anatomy["target_subtypes"]
    return {
        name: torch.from_numpy(np.flatnonzero(subtype == index).astype(np.int64))
        for index, name in enumerate(registration.TARGET_SUBTYPES)
    }


def _window_values(response: dict[str, Tensor], window: str) -> tuple[Tensor, Tensor]:
    return response[f"{window}_response"], response[f"{window}_static"]


def _opponent(
    response: Tensor,
    relative: dict[str, Tensor],
    pathway: str,
) -> Tensor:
    c = relative[pathway + "c"].to(response.device)
    d = relative[pathway + "d"].to(response.device)
    return response[:, c].mean(dim=1) - response[:, d].mean(dim=1)


def source_references(
    specs: list[dict[str, Any]],
    source_normal: list[dict[str, Tensor]],
    source_reverse: list[dict[str, Tensor]],
    anatomy: dict[str, np.ndarray],
) -> dict[str, Any]:
    relative = relative_subtype_indices(anatomy)
    references: dict[str, Any] = {"population": {}, "cells": {}}
    for window in WINDOWS:
        references["population"][window] = []
        for spec, normal, reverse in zip(specs, source_normal, source_reverse, strict=True):
            normal_response, _ = _window_values(normal, window)
            reverse_response, _ = _window_values(reverse, window)
            pair = {}
            for pathway in pathway_for_spec(spec):
                normal_opponent = _opponent(normal_response, relative, pathway)
                reverse_opponent = _opponent(
                    reverse_response, relative, reverse_pathway(spec, pathway)
                )
                pair[pathway] = {
                    "normal_scale": torch.clamp(
                        normal_opponent.abs().mean(), min=NORMALIZATION_FLOOR
                    ).detach(),
                    "reverse_scale": torch.clamp(
                        reverse_opponent.abs().mean(), min=NORMALIZATION_FLOOR
                    ).detach(),
                }
            references["population"][window].append(pair)

        references["cells"][window] = {}
        for pathway in PATHWAYS:
            selected_cases = [
                index for index, spec in enumerate(specs) if pathway in pathway_for_spec(spec)
            ]
            subtype_refs = {}
            for suffix, preferred_branch, null_branch in (
                ("c", 1, 0),
                ("d", 0, 1),
            ):
                positions = relative[pathway + suffix]
                source_values = torch.stack(
                    [
                        _window_values(source_normal[index], window)[0][:, positions]
                        for index in selected_cases
                    ]
                )
                preferred = source_values[:, preferred_branch].mean(dim=0)
                null = source_values[:, null_branch].mean(dim=0)
                activity = torch.maximum(preferred.abs(), null.abs())
                subtype_refs[suffix] = {
                    "case_indices": selected_cases,
                    "dsi_denominator": (preferred.abs() + null.abs()).clamp_min(
                        NORMALIZATION_FLOOR
                    ),
                    # Preserve at most the fixed absolute floor, and never demand that a
                    # source-weak cell be raised to that floor.
                    "activity_target": activity.clamp_max(NORMALIZATION_FLOOR),
                    "activity_normalization": activity.clamp_min(NORMALIZATION_FLOOR),
                }
            references["cells"][window][pathway] = subtype_refs
    references["semantic_sha256"] = semantic_sha256(references)
    return references


def _mean(values: list[Tensor]) -> Tensor:
    if not values:
        raise RuntimeError("loss component has no values")
    return torch.stack(values).mean()


def commissioning_loss(
    specs: list[dict[str, Any]],
    normal: list[dict[str, Tensor]],
    reverse: list[dict[str, Tensor]],
    references: dict[str, Any],
    anatomy: dict[str, np.ndarray],
    controller: CommissionedController,
) -> tuple[Tensor, dict[str, Tensor]]:
    relative = relative_subtype_indices(anatomy)
    window_direction = []
    window_bias = []
    window_stationary = []
    window_reverse = []
    window_dsi = []
    window_activity = []

    for window in WINDOWS:
        pair_direction = []
        pair_bias = []
        pair_stationary = []
        pair_reverse = []
        for pair_index, (spec, normal_response, reverse_response) in enumerate(
            zip(specs, normal, reverse, strict=True)
        ):
            response, stationary = _window_values(normal_response, window)
            reversed_values, _ = _window_values(reverse_response, window)
            path_direction = []
            path_bias = []
            path_stationary = []
            path_reverse = []
            for pathway in pathway_for_spec(spec):
                normal_scale = references["population"][window][pair_index][pathway][
                    "normal_scale"
                ].to(response.device)
                reverse_scale = references["population"][window][pair_index][pathway][
                    "reverse_scale"
                ].to(response.device)
                opponent = _opponent(response, relative, pathway)
                stationary_opponent = _opponent(stationary, relative, pathway)
                direction = (opponent[1] - opponent[0]) / 2.0
                bias = (opponent[1] + opponent[0]) / 2.0
                path_direction.append(
                    torch.relu(DIRECTION_MARGIN - direction / normal_scale).square()
                )
                path_bias.append((bias / normal_scale).square())
                path_stationary.append(
                    (
                        (stationary_opponent[1] - stationary_opponent[0]) / (2.0 * normal_scale)
                    ).square()
                )

                reversed_opponent = _opponent(
                    reversed_values, relative, reverse_pathway(spec, pathway)
                )
                reverse_down = torch.relu(
                    DIRECTION_MARGIN - reversed_opponent[0] / reverse_scale
                ).square()
                reverse_up = torch.relu(
                    DIRECTION_MARGIN + reversed_opponent[1] / reverse_scale
                ).square()
                path_reverse.append((reverse_down + reverse_up) / 2.0)
            pair_direction.append(_mean(path_direction))
            pair_bias.append(_mean(path_bias))
            pair_stationary.append(_mean(path_stationary))
            pair_reverse.append(_mean(path_reverse))
        window_direction.append(_mean(pair_direction))
        window_bias.append(_mean(pair_bias))
        window_stationary.append(_mean(pair_stationary))
        window_reverse.append(_mean(pair_reverse))

        pathway_dsi = []
        pathway_activity = []
        for pathway in PATHWAYS:
            subtype_dsi = []
            subtype_activity = []
            for suffix, preferred_branch, null_branch in (
                ("c", 1, 0),
                ("d", 0, 1),
            ):
                cell_reference = references["cells"][window][pathway][suffix]
                positions = relative[pathway + suffix].to(normal[0]["integrated_response"].device)
                values = torch.stack(
                    [
                        _window_values(normal[index], window)[0][:, positions]
                        for index in cell_reference["case_indices"]
                    ]
                )
                preferred = values[:, preferred_branch].mean(dim=0)
                null = values[:, null_branch].mean(dim=0)
                denominator = cell_reference["dsi_denominator"].to(values.device)
                dsi = (preferred - null) / denominator
                subtype_dsi.append(torch.relu(DIRECTION_MARGIN - dsi).square().mean())
                activity = torch.maximum(preferred.abs(), null.abs())
                target = cell_reference["activity_target"].to(values.device)
                normalization = cell_reference["activity_normalization"].to(values.device)
                subtype_activity.append(
                    (torch.relu(target - activity) / normalization).square().mean()
                )
            pathway_dsi.append(_mean(subtype_dsi))
            pathway_activity.append(_mean(subtype_activity))
        window_dsi.append(_mean(pathway_dsi))
        window_activity.append(_mean(pathway_activity))

    components = {
        "direction": _mean(window_direction),
        "bias": _mean(window_bias),
        "stationary": _mean(window_stationary),
        "dsi": _mean(window_dsi),
        "reverse": _mean(window_reverse),
        "activity": _mean(window_activity),
        "regularization": (
            (controller.gain - 1.0).square().mean()
            + controller.bias_offset.square().mean()
            + (controller.tau_ratio - 1.0).square().mean()
        ),
    }
    total = (
        components["direction"]
        + components["bias"]
        + components["stationary"]
        + components["dsi"]
        + REVERSE_WEIGHT * components["reverse"]
        + ACTIVITY_WEIGHT * components["activity"]
        + REGULARIZATION_WEIGHT * components["regularization"]
    )
    return total, components
