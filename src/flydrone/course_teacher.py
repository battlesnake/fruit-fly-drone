"""Training-only continuous course reference and full-state tracking teacher.

This teacher is deliberately privileged. None of its path, position, velocity or
gate data belongs in a deployed fly observation or neural state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from flydrone.gate import AnnularGate, wrap_angle
from flydrone.hover import HoverConfig, QuadState, motor_target_for_rc


@dataclass(frozen=True)
class CoursePath:
    """Cubic Hermite reference for forward, monotonically increasing-X courses."""

    knots: Tensor
    slopes: Tensor

    @classmethod
    def through_gates(cls, launch: Tensor, gates: Sequence[AnnularGate]) -> CoursePath:
        if not gates:
            raise ValueError("at least one gate is required")
        points = torch.stack((launch, *(gate.center for gate in gates)), dim=1)
        delta = points[:, 1:] - points[:, :-1]
        if bool((delta[..., 0] <= 0.05).any()):
            raise ValueError("teacher currently requires increasing-X gate centres")
        last_slope = delta[:, -1] / delta[:, -1, :1]
        exit_point = points[:, -1] + 1.5 * last_slope
        knots = torch.cat((points, exit_point[:, None]), dim=1)
        interval = knots[:, 1:] - knots[:, :-1]
        secant = interval / interval[..., :1]
        central = knots[:, 2:] - knots[:, :-2]
        central = central / central[..., :1]
        slopes = torch.cat((secant[:, :1], central, secant[:, -1:]), dim=1)
        return cls(knots, slopes)

    def sample(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return reference position and its first/second derivatives w.r.t. X."""
        index = (
            (x[:, None] >= self.knots[..., 0]).sum(dim=1).sub(1).clamp(0, self.knots.shape[1] - 2)
        )
        row = torch.arange(len(x), device=x.device)
        a, b = self.knots[row, index], self.knots[row, index + 1]
        da, db = self.slopes[row, index], self.slopes[row, index + 1]
        length = b[:, :1] - a[:, :1]
        t = ((x[:, None] - a[:, :1]) / length).clamp(0.0, 1.0)
        c = 3.0 * (b - a) - length * (2.0 * da + db)
        d = 2.0 * (a - b) + length * (da + db)
        position = a + length * da * t + c * t.square() + d * t.pow(3)
        derivative = da + (2.0 * c * t + 3.0 * d * t.square()) / length
        curvature = (2.0 * c + 6.0 * d * t) / length.square()
        return position, derivative, curvature


@dataclass(frozen=True)
class CourseTeacherConfig:
    forward_speed: float = 0.55
    position_gain: float = 1.5
    velocity_gain: float = 2.5
    attitude_gain: float = 3.0
    rate_damping: float = 0.15
    yaw_gain: float = 2.0
    heading_mode: str = "tangent"

    def __post_init__(self) -> None:
        if self.heading_mode not in ("tangent", "world-x"):
            raise ValueError("heading_mode must be tangent or world-x")


DEFAULT_COURSE_TEACHER_CONFIG = CourseTeacherConfig()


def course_teacher_motor(
    state: QuadState,
    path: CoursePath,
    quad: HoverConfig,
    config: CourseTeacherConfig = DEFAULT_COURSE_TEACHER_CONFIG,
) -> Tensor:
    """Track the multi-gate reference through the same physical foreleg plant."""
    reference, tangent, curvature = path.sample(state.position[:, 0])
    forward_speed = (0.8 * (path.knots[:, -1, 0] - state.position[:, 0])).clamp(
        -0.3, config.forward_speed
    )
    position_error = reference - state.position
    position_error[:, 0] = 0.0
    desired_velocity = tangent * forward_speed[:, None] + config.position_gain * position_error
    acceleration = config.velocity_gain * (desired_velocity - state.velocity)
    acceleration += curvature * forward_speed[:, None].square()
    acceleration += quad.linear_drag * state.velocity / quad.mass
    acceleration = acceleration.clamp(-5.0, 5.0)
    force = acceleration.clone()
    force[:, 2] += 9.81
    yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
    if config.heading_mode == "world-x":
        yaw = torch.zeros_like(yaw)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    force_norm = torch.linalg.vector_norm(force, dim=1).clamp_min(1.0e-6)
    desired_roll = torch.asin(((force[:, 0] * sy - force[:, 1] * cy) / force_norm).clamp(-0.5, 0.5))
    desired_pitch = torch.atan2(force[:, 0] * cy + force[:, 1] * sy, force[:, 2])
    desired_pitch = desired_pitch.clamp(-0.5, 0.5)
    rates = (
        torch.stack(
            (
                config.attitude_gain * (desired_roll - state.euler[:, 0]),
                config.attitude_gain * (desired_pitch - state.euler[:, 1]),
                config.yaw_gain * wrap_angle(yaw - state.euler[:, 2]),
            ),
            dim=1,
        )
        - config.rate_damping * state.rates
    )
    rate_limits = rates.new_tensor(
        (quad.max_roll_pitch_rate, quad.max_roll_pitch_rate, quad.max_yaw_rate)
    )
    tilt_compensation = (torch.cos(state.euler[:, 0]) * torch.cos(state.euler[:, 1])).clamp_min(0.6)
    throttle = force[:, 2] / (9.81 * quad.thrust_to_weight * tilt_compensation)
    rc = torch.cat(
        ((rates / rate_limits).clamp(-1.0, 1.0), throttle[:, None].clamp(0.0, 1.0)), dim=1
    )
    return motor_target_for_rc(rc, quad)
