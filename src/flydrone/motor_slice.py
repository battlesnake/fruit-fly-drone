"""Fast training slice for native motor neurons that are neural graph sinks.

This is a training computation, not an extra deployed decoder. Fitted parameters
are copied only into the existing incoming synapses of the full connectome.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinkMotorSlice(nn.Module):
    def __init__(self, controller, *, axis=0, train_edge_mask=None):
        super().__init__()
        begin, middle, end = [int(controller.pool_offsets[2 * axis + i]) for i in range(3)]
        motors = controller.pool_indices[begin:end]
        if len(torch.unique(motors)) != len(motors):
            raise ValueError("motor pools must be disjoint")
        if bool(torch.isin(controller.edge_pre, motors).any()):
            raise ValueError("slice requires motor neurons with no outgoing neural edges")
        for name in (
            "visual_nodes",
            "attitude_nodes",
            "acceleration_nodes",
            "proprioception_nodes",
        ):
            nodes = getattr(controller, name, motors.new_empty(0))
            if bool(torch.isin(nodes, motors).any()):
                raise ValueError("slice requires motor neurons without direct sensory injection")
        incoming = torch.nonzero(torch.isin(controller.edge_post, motors)).flatten()
        parents, pre_local = torch.unique(controller.edge_pre[incoming], return_inverse=True)
        lookup = torch.full_like(controller.bias, -1, dtype=torch.long)
        lookup[motors] = torch.arange(len(motors), device=motors.device)
        post_local = lookup[controller.edge_post[incoming]]
        slots = torch.arange(len(incoming), device=incoming.device)
        if train_edge_mask is not None:
            slots = slots[train_edge_mask[incoming]]
        if not len(slots):
            raise ValueError("slice has no trainable incoming synapses")
        self.positive_count = middle - begin
        self.motor_count = len(motors)
        self.register_buffer("motors", motors.clone())
        self.register_buffer("parents", parents)
        self.register_buffer("incoming_edges", incoming)
        self.register_buffer("train_edges", incoming[slots])
        self.register_buffer("train_slots", slots)
        self.register_buffer("matrix_indices", post_local * len(parents) + pre_local)
        self.register_buffer("signs", controller.edge_sign[incoming].detach().clone())
        magnitudes = controller.edge_magnitude[incoming].detach()
        self.register_buffer("source_magnitudes", magnitudes.clone())
        self.magnitudes = nn.Parameter(magnitudes[slots].clone())
        self.register_buffer("bias", controller.bias[motors].detach().clone())
        decay = torch.exp(-controller.neural_dt / controller.time_constant[motors].detach())
        self.register_buffer("decay", decay)
        # The fixed leaky recurrence is a causal exponential convolution. Retain
        # enough taps that the omitted bounded-state tail is below 5e-9; FP32
        # roundoff dominates this truncation. No membrane is reset between frames.
        self.filter_steps = max(1, math.ceil(math.log(1e-9) / math.log(float(decay.max()))))
        powers = torch.arange(self.filter_steps, device=decay.device, dtype=decay.dtype)
        kernel = (1 - decay[:, None]) * decay[:, None].pow(powers[None])
        self.register_buffer("kernel", kernel.flip(1)[:, None])

    def targets(self, features):
        """Synaptic drive from existing incoming edges and fixed motor biases."""
        magnitudes = self.source_magnitudes.index_copy(0, self.train_slots, self.magnitudes)
        matrix = (
            features.new_zeros(self.motor_count * len(self.parents))
            .index_add(0, self.matrix_indices, self.signs * magnitudes)
            .reshape(self.motor_count, len(self.parents))
        )
        drive = F.linear(features, matrix, self.bias)
        return 5 * torch.tanh(drive / 5)

    def forward(self, features):
        """Return native motor drive using the tiny-tail convolution approximation."""
        target = self.targets(features)
        states = F.conv1d(
            target.permute(1, 2, 0),
            self.kernel,
            padding=self.filter_steps - 1,
            groups=self.motor_count,
        )[:, :, : len(features)].permute(2, 0, 1)
        rates = torch.sigmoid(states)
        return rates[..., : self.positive_count].mean(-1) - rates[..., self.positive_count :].mean(
            -1
        )

    def forward_recurrent(self, features):
        """All six-cell time steps from zero, including supplied warmup features.

        This uses the full controller's membrane update without a truncated filter
        tail or detached boundaries. Features are activities BEFORE each brain tick.
        """
        target = self.targets(features)
        state = target.new_zeros(target.shape[1:])
        states = []
        alpha = 1 - self.decay
        for frame in target:
            state = state + alpha * (frame - state)
            states.append(state)
        rates = torch.sigmoid(torch.stack(states))
        return (rates[..., :self.positive_count].mean(-1)
                - rates[..., self.positive_count:].mean(-1))

    @torch.no_grad()
    def project_parameters(self):
        self.magnitudes.clamp_(0, 8)

    @torch.no_grad()
    def compile_into(self, controller):
        controller.edge_magnitude.index_copy_(0, self.train_edges, self.magnitudes)

    def manifest(self):
        return dict(
            motor_neurons=self.motor_count,
            incoming_edges=len(self.incoming_edges),
            trainable_incoming_edges=len(self.train_edges),
            presynaptic_neurons=len(self.parents),
            outgoing_motor_edges=0,
            causal_filter_steps=self.filter_steps,
            deployed_extra_decoder=False,
        )
