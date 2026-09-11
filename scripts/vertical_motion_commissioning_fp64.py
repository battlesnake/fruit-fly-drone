#!/usr/bin/env python3
"""Independent full-float64 reference path for vertical-motion commissioning."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preregister_vertical_motion_commissioning as registration  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402


def promote_tree_fp64(value: Any) -> Any:
    """Clone a nested reference tree, promoting floating tensors to float64."""

    if isinstance(value, Tensor):
        if value.is_floating_point():
            return value.detach().clone().to(dtype=torch.float64)
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: promote_tree_fp64(item) for key, item in value.items()}
    if isinstance(value, list):
        return [promote_tree_fp64(item) for item in value]
    if isinstance(value, tuple):
        return tuple(promote_tree_fp64(item) for item in value)
    return value


def promotion_roundtrip_maximum_difference(source: Any, promoted: Any) -> float:
    """Return the largest value change after promoting then restoring source dtype."""

    differences: list[float] = []

    def visit(first: Any, second: Any) -> None:
        if isinstance(first, Tensor):
            restored = second.to(dtype=first.dtype, device=first.device)
            differences.append(float((first - restored).abs().max()) if first.numel() else 0.0)
            return
        if isinstance(first, dict):
            if first.keys() != second.keys():
                raise ValueError("promoted reference dictionary keys changed")
            for key in first:
                visit(first[key], second[key])
            return
        if isinstance(first, (list, tuple)):
            if len(first) != len(second):
                raise ValueError("promoted reference sequence length changed")
            for left, right in zip(first, second, strict=True):
                visit(left, right)
            return
        if first != second:
            raise ValueError("non-tensor reference value changed during promotion")

    visit(source, promoted)
    return max(differences, default=0.0)


class FP64CommissionedController(nn.Module):
    """Apply the 24 local parameters using a direct, entirely float64 state path."""

    def __init__(self, source: nn.Module, anatomy: dict[str, np.ndarray]) -> None:
        super().__init__()
        source.eval().requires_grad_(False)
        if source.bias.dtype != torch.float64:
            raise ValueError("FP64 reference source must already be float64")
        self.source = source
        self.register_buffer("selected_edge_indices", torch.from_numpy(anatomy["edge_indices"]))
        self.register_buffer("selected_edge_groups", torch.from_numpy(anatomy["edge_groups"]))
        self.register_buffer("selected_target_indices", torch.from_numpy(anatomy["target_indices"]))
        self.register_buffer(
            "selected_target_subtypes", torch.from_numpy(anatomy["target_subtypes"])
        )
        self.gain = nn.Parameter(torch.ones(16, dtype=torch.float64))
        self.bias_offset = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.tau_ratio = nn.Parameter(torch.ones(4, dtype=torch.float64))

    @property
    def n_nodes(self) -> int:
        return self.source.n_nodes

    def initial_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if dtype != torch.float64:
            raise ValueError("FP64 reference state must be float64")
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

    def identity_restored(self) -> bool:
        return bool(
            torch.equal(self.gain.detach(), torch.ones_like(self.gain))
            and torch.equal(self.bias_offset.detach(), torch.zeros_like(self.bias_offset))
            and torch.equal(self.tau_ratio.detach(), torch.ones_like(self.tau_ratio))
        )

    def forward(self, image: Tensor, roll_pitch: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if image.dtype != torch.float64 or state.dtype != torch.float64:
            raise ValueError("FP64 reference received a non-float64 image or state")
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
        candidate_next_state = state[:, selected_target] + candidate_alpha * (
            target[:, selected_target] - state[:, selected_target]
        )
        next_state = next_state.index_copy(1, selected_target, candidate_next_state)
        return self.source.motor_drive(next_state), next_state


def _advance_frame(
    controller: nn.Module,
    image: Tensor,
    state: Tensor,
    *,
    checkpoint_frames: bool,
) -> tuple[Tensor, Tensor]:
    attitude = torch.zeros(image.shape[0], 2, device=image.device, dtype=torch.float64)

    def integrate(frame_state: Tensor) -> tuple[Tensor, Tensor]:
        output = torch.zeros(image.shape[0], 4, device=image.device, dtype=torch.float64)
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
        dtype=torch.float64,
    )
    state = controller.initial_state(1, device=device, dtype=torch.float64)
    for _ in range(registration.PREFIX_FRAMES):
        _, state = _advance_frame(controller, neutral, state, checkpoint_frames=checkpoint_frames)
    return state


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
    integrated_response = torch.zeros(2, len(selected), device=device, dtype=torch.float64)
    integrated_static = torch.zeros_like(integrated_response)
    terminal_response = torch.empty_like(integrated_response)
    terminal_static = torch.empty_like(integrated_response)
    for frame in range(registration.MOTION_FRAMES + registration.TERMINAL_FRAMES):
        gray = sequence[frame].to(device=device, dtype=torch.float64)
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


def evaluate_cases(
    controller: nn.Module,
    training_specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    *,
    indices: tuple[int, ...],
    device: torch.device,
    checkpoint_frames: bool,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    del training_specs
    prefix = neutral_prefix(controller, device=device, checkpoint_frames=checkpoint_frames)
    normal = []
    reverse = []
    for case in indices:
        normal.append(
            evaluate_pair(
                controller,
                sequences[(case, False)],
                anatomy,
                device=device,
                checkpoint_frames=checkpoint_frames,
                prefix_state=prefix,
            )
        )
        reverse.append(
            evaluate_pair(
                controller,
                sequences[(case, True)],
                anatomy,
                device=device,
                checkpoint_frames=checkpoint_frames,
                prefix_state=prefix,
            )
        )
    return normal, reverse


def batch_loss(
    controller: FP64CommissionedController,
    batch_specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    references: dict[str, Any],
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
    checkpoint_frames: bool,
) -> tuple[Tensor, dict[str, Tensor]]:
    normal, reverse = evaluate_cases(
        controller,
        batch_specs,
        sequences,
        anatomy,
        indices=(0, 24, 48, 60),
        device=device,
        checkpoint_frames=checkpoint_frames,
    )
    return commissioning.commissioning_loss(
        batch_specs, normal, reverse, references, anatomy, controller
    )


def fixed_state_tau_derivative(
    source_tau: Tensor,
    neural_dt: float,
    *,
    state_value: float = 0.125,
    target_value: float = -0.375,
) -> dict[str, float | bool]:
    """Compare autograd and closed-form derivatives for one direct tau update."""

    if source_tau.numel() != 1 or source_tau.dtype != torch.float64:
        raise ValueError("one-step source tau must be one float64 scalar")
    ratio = torch.ones((), dtype=torch.float64, device=source_tau.device, requires_grad=True)
    candidate_tau = (source_tau.detach().reshape(()) * ratio).clamp(0.010, 0.250)
    state = ratio.new_tensor(state_value)
    target = ratio.new_tensor(target_value)
    alpha = 1.0 - torch.exp(-neural_dt / candidate_tau)
    next_state = state + alpha * (target - state)
    autograd_value = torch.autograd.grad(next_state, ratio)[0]
    unclamped_tau = source_tau.detach().reshape(())
    explicit_value = (
        (target - state)
        * (-torch.exp(-neural_dt / unclamped_tau))
        * neural_dt
        / unclamped_tau.square()
        * source_tau.detach().reshape(())
    )
    observed = float(autograd_value.detach().cpu())
    expected = float(explicit_value.detach().cpu())
    relative_error = abs(observed - expected) / max(abs(observed), abs(expected), 1.0e-300)
    clamp_inactive = bool(0.010 < float(unclamped_tau.cpu()) < 0.250)
    sign_matches = bool(
        observed != 0.0 and expected != 0.0 and np.sign(observed) == np.sign(expected)
    )
    return {
        "source_tau": float(unclamped_tau.cpu()),
        "state": state_value,
        "target": target_value,
        "autograd_derivative": observed,
        "explicit_derivative": expected,
        "symmetric_relative_error": relative_error,
        "clamp_inactive": clamp_inactive,
        "sign_matches": sign_matches,
        "finite_nonzero": bool(
            np.isfinite(observed) and np.isfinite(expected) and observed != 0.0 and expected != 0.0
        ),
    }
