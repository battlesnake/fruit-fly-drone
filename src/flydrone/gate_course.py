"""Training/evaluation-only directed course events; never actor observations.

The geometry is a swept centre segment against zero-thickness annuli with a
radial drone-clearance margin. It is not a full swept-body collision solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from flydrone.gate import DEFAULT_GATE_CONFIG, AnnularGate, GateConfig


@dataclass(frozen=True)
class GateCourseEvents:
    next_gate_index: Tensor
    passed: Tensor
    wrong_order: Tensor
    wrong_direction: Tensor
    ring_collision: Tensor
    expected_forward_crossing: Tensor
    crossing_lateral: Tensor
    crossing_vertical: Tensor

    @property
    def illegal_traversal(self) -> Tensor:
        """A backwards, out-of-order traversal is one violation, not two."""
        return self.wrong_order | self.wrong_direction

    @property
    def failed(self) -> Tensor:
        return (self.illegal_traversal | self.ring_collision).any(dim=1)

    def penalty(self, *, traversal_cost: float = 1.0, collision_cost: float = 2.0) -> Tensor:
        """One-shot event penalty, separate from progress/time/completion rewards."""
        if traversal_cost < 0.0 or collision_cost < 0.0:
            raise ValueError("event costs must be nonnegative")
        return -(
            traversal_cost * self.illegal_traversal.sum(dim=1)
            + collision_cost * self.ring_collision.sum(dim=1)
        )


def classify_course_step(
    previous_position: Tensor,
    position: Tensor,
    gates: Sequence[AnnularGate],
    current_gate_index: Tensor,
    config: GateConfig = DEFAULT_GATE_CONFIG,
) -> GateCourseEvents:
    """Check every gate in both directions, in physical crossing-time order.

    Only a clean forward traversal of the current gate advances the index.
    Crossing a plane outside its annulus is legal (e.g. circling for a corkscrew).
    Passed/dark gates remain physical. Failures do not halt recovery or erase
    later passes: the caller must latch failures for clean-course completion.
    All event masks have shape (batch, gates); the input index is not mutated.
    """
    if not gates:
        raise ValueError("at least one gate is required")
    batch = position.shape[0]
    if position.shape != (batch, 3) or previous_position.shape != position.shape:
        raise ValueError("positions must have shape (batch, 3)")
    if current_gate_index.shape != (batch,) or current_gate_index.dtype != torch.long:
        raise ValueError("current_gate_index must be a long tensor of shape (batch,)")
    if any(g.center.shape != (batch, 3) or g.yaw.shape != (batch,) for g in gates):
        raise ValueError("each gate must contain one centre and yaw per batch item")
    centers = torch.stack([gate.center for gate in gates], dim=1)
    normals = torch.stack([gate.normal for gate in gates], dim=1)
    previous_signed = ((previous_position[:, None] - centers) * normals).sum(dim=-1)
    signed = ((position[:, None] - centers) * normals).sum(dim=-1)
    forward = (previous_signed < 0.0) & (signed >= 0.0)
    # The plane belongs to the positive half-space in BOTH directions, so a
    # negative -> zero -> negative touch-and-return cannot hide a reversal.
    backward = (previous_signed >= 0.0) & (signed < 0.0)
    crossed = forward | backward
    denominator = signed - previous_signed
    # Retain the denominator's sign: clamp_min would break reverse crossings.
    denominator = torch.where(crossed, denominator, torch.ones_like(denominator))
    fraction = (-previous_signed / denominator).clamp(0.0, 1.0)
    hit = previous_position[:, None] + fraction[..., None] * (
        position - previous_position
    )[:, None]
    offset = hit - centers
    laterals = torch.stack([gate.lateral for gate in gates], dim=1)
    lateral = (offset * laterals).sum(dim=-1)
    vertical = offset[..., 2]
    radial_squared = lateral.square() + vertical.square()
    clean = crossed & (radial_squared <= (config.inner_radius - config.drone_radius) ** 2)
    collision = crossed & ~clean & (
        radial_squared <= (config.outer_radius + config.drone_radius) ** 2
    )
    chronological_order = torch.argsort(
        torch.where(crossed, fraction, float("inf")), dim=1, stable=True
    )
    current = current_gate_index.clone()
    passed = torch.zeros_like(clean)
    wrong_order = torch.zeros_like(clean)
    expected_forward_crossing = torch.zeros_like(clean)
    row = torch.arange(batch, device=position.device)
    for slot in range(len(gates)):
        gate_index = chronological_order[:, slot]
        aperture = clean[row, gate_index]
        expected = gate_index == current
        expected_forward_crossing[row, gate_index] = expected & forward[row, gate_index]
        valid_pass = aperture & expected & forward[row, gate_index]
        passed[row, gate_index] = valid_pass
        wrong_order[row, gate_index] = aperture & ~expected
        current = current + valid_pass.long()
    return GateCourseEvents(
        next_gate_index=current,
        passed=passed,
        wrong_order=wrong_order,
        wrong_direction=clean & backward,
        ring_collision=collision,
        expected_forward_crossing=expected_forward_crossing,
        crossing_lateral=lateral,
        crossing_vertical=vertical,
    )
