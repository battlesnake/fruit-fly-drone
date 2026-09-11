"""Cancellation-resistant selected-node tau update for commissioning preflight v2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import vertical_motion_commissioning as v1  # noqa: E402


class CommissionedController(v1.CommissionedController):
    """Use a direct selected-state write instead of a small alpha correction."""

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
        candidate_next_state = state[:, selected_target] + candidate_alpha * (
            target[:, selected_target] - state[:, selected_target]
        )
        # selected_target is a hash-locked unique, sorted node-index vector. A direct
        # write avoids subtracting nearly equal alphas and adding a tiny correction to
        # a much larger already-rounded source next state.
        next_state = next_state.index_copy(1, selected_target, candidate_next_state)
        return self.source.motor_drive(next_state), next_state
