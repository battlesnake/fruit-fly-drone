"""Differentiable annular-gate world for the connectome flight milestone."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from flydrone.hover import (
    DEFAULT_HOVER_CONFIG,
    HoverConfig,
    QuadState,
    make_camera_rays,
    rotation_matrix,
)
from flydrone.visual_hover import DEFAULT_VISUAL_CAMERA, CameraSpec, make_pinhole_rays

GATE_TASK_VERSION = "annular-gate-v1"


@dataclass(frozen=True)
class GateConfig:
    inner_radius: float = 0.62
    outer_radius: float = 0.76
    drone_radius: float = 0.09
    edge_softness: float = 0.018
    minimum_obliquity_degrees: float = 10.0
    maximum_obliquity_degrees: float = 30.0
    minimum_lateral_offset: float = 0.35
    maximum_lateral_offset: float = 1.25
    minimum_distance: float = 4.2
    maximum_distance: float = 5.2
    minimum_height: float = 0.95
    maximum_height: float = 1.25
    # Keep old checkpoints' appearance unless a new lesson explicitly enables it.
    back_pattern: str = "solid"

    def __post_init__(self) -> None:
        if self.back_pattern not in ("solid", "checkerboard"):
            raise ValueError("back_pattern must be solid or checkerboard")


DEFAULT_GATE_CONFIG = GateConfig()


@dataclass(frozen=True)
class AnnularGate:
    center: Tensor
    yaw: Tensor

    @property
    def normal(self) -> Tensor:
        zeros = torch.zeros_like(self.yaw)
        return torch.stack((torch.cos(self.yaw), torch.sin(self.yaw), zeros), dim=-1)

    @property
    def lateral(self) -> Tensor:
        zeros = torch.zeros_like(self.yaw)
        return torch.stack((-torch.sin(self.yaw), torch.cos(self.yaw), zeros), dim=-1)


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def sample_annular_gates(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    config: GateConfig = DEFAULT_GATE_CONFIG,
    strict: bool = True,
) -> AnnularGate:
    """Sample visible gates; strict samples are always off-axis and oblique."""

    distance = torch.empty(batch, device=device, dtype=dtype).uniform_(
        config.minimum_distance, config.maximum_distance
    )
    if strict:
        offset_magnitude = torch.empty(batch, device=device, dtype=dtype).uniform_(
            config.minimum_lateral_offset, config.maximum_lateral_offset
        )
        offset_sign = torch.where(
            torch.rand(batch, device=device) < 0.5,
            -torch.ones(batch, device=device, dtype=dtype),
            torch.ones(batch, device=device, dtype=dtype),
        )
        lateral_offset = offset_magnitude * offset_sign
        obliquity_magnitude = torch.empty(batch, device=device, dtype=dtype).uniform_(
            math.radians(config.minimum_obliquity_degrees),
            math.radians(config.maximum_obliquity_degrees),
        )
        obliquity_sign = torch.where(
            torch.rand(batch, device=device) < 0.5,
            -torch.ones(batch, device=device, dtype=dtype),
            torch.ones(batch, device=device, dtype=dtype),
        )
        obliquity = obliquity_magnitude * obliquity_sign
    else:
        lateral_offset = torch.empty(batch, device=device, dtype=dtype).uniform_(
            -config.maximum_lateral_offset, config.maximum_lateral_offset
        )
        obliquity = torch.empty(batch, device=device, dtype=dtype).uniform_(
            -math.radians(config.maximum_obliquity_degrees),
            math.radians(config.maximum_obliquity_degrees),
        )
    height = torch.empty(batch, device=device, dtype=dtype).uniform_(
        config.minimum_height, config.maximum_height
    )
    center = torch.stack((distance, lateral_offset, height), dim=-1)
    bearing = torch.atan2(center[:, 1], center[:, 0])
    return AnnularGate(center=center, yaw=bearing + obliquity)


def gate_coordinates(position: Tensor, gate: AnnularGate) -> tuple[Tensor, Tensor, Tensor]:
    """Return signed normal, lateral and vertical coordinates relative to the gate."""

    offset = position - gate.center
    signed = (offset * gate.normal).sum(dim=-1)
    lateral = (offset * gate.lateral).sum(dim=-1)
    vertical = offset[:, 2]
    return signed, lateral, vertical


def initial_gate_geometry(gate: AnnularGate) -> tuple[Tensor, Tensor]:
    """Return launch-view centre offset and plane obliquity in radians."""

    displacement = gate.center
    bearing = torch.atan2(displacement[:, 1], displacement[:, 0])
    elevation = torch.atan2(
        displacement[:, 2], torch.linalg.vector_norm(displacement[:, :2], dim=-1)
    )
    centre_offset = torch.sqrt(bearing.square() + elevation.square())
    obliquity = wrap_angle(gate.yaw - bearing).abs()
    return centre_offset, obliquity


def render_annular_gate(
    state: QuadState,
    gate: AnnularGate,
    *,
    resolution: int = 32,
    hover_config: HoverConfig = DEFAULT_HOVER_CONFIG,
    gate_config: GateConfig = DEFAULT_GATE_CONFIG,
) -> Tensor:
    """Render a bright physical annulus over a black world and textured grey floor."""

    rays_body = make_camera_rays(
        resolution,
        hover_config.camera_fov_degrees,
        device=state.position.device,
        dtype=state.position.dtype,
    )
    # rotation_matrix uses a right-handed +X-forward/+Z-up body frame, hence +Y is
    # camera-left.  The original height-band raster is horizontally symmetric and did
    # not expose its legacy mirror convention; gate steering must use physical screen
    # right = -body-Y explicitly.
    rays_body = torch.stack((rays_body[..., 0], -rays_body[..., 1], rays_body[..., 2]), dim=-1)
    body_to_world = rotation_matrix(state.euler)
    rays_world = torch.einsum("bij,hwj->bhwi", body_to_world, rays_body)
    origin = state.position[:, None, None, :]

    downward = rays_world[..., 2] < -1.0e-4
    floor_time = -origin[..., 2] / rays_world[..., 2].clamp_max(-1.0e-4)
    floor_valid = downward & (floor_time > 0.0)
    floor_x = origin[..., 0] + floor_time * rays_world[..., 0]
    floor_y = origin[..., 1] + floor_time * rays_world[..., 1]
    floor_texture = 0.24 + 0.035 * torch.sin(3.0 * floor_x) * torch.sin(3.0 * floor_y)
    image = torch.where(floor_valid, floor_texture, torch.zeros_like(floor_texture))

    normal = gate.normal[:, None, None, :]
    plane_numerator = ((gate.center[:, None, None, :] - origin) * normal).sum(dim=-1)
    plane_denominator = (rays_world * normal).sum(dim=-1)
    safe_denominator = torch.where(
        plane_denominator.abs() > 1.0e-4,
        plane_denominator,
        torch.ones_like(plane_denominator),
    )
    plane_time = plane_numerator / safe_denominator
    plane_valid = (plane_denominator.abs() > 1.0e-4) & (plane_time > 0.0)
    hit = origin + plane_time[..., None] * rays_world
    hit_offset = hit - gate.center[:, None, None, :]
    gate_lateral = (hit_offset * gate.lateral[:, None, None, :]).sum(dim=-1)
    gate_vertical = hit_offset[..., 2]
    radial = torch.sqrt(gate_lateral.square() + gate_vertical.square() + 1.0e-8)
    outer = torch.sigmoid((gate_config.outer_radius - radial) / gate_config.edge_softness)
    inner = torch.sigmoid((radial - gate_config.inner_radius) / gate_config.edge_softness)
    # Fixed paint asymmetry makes the pose of a monocular circular gate observable
    # without encoding task state or using a gate detector.
    paint = (
        0.65
        + 0.20 * torch.sigmoid(gate_lateral / 0.06)
        + 0.15 * torch.sigmoid(gate_vertical / 0.06)
    )
    annulus = outer * inner * paint
    gate_in_front = plane_valid & (~floor_valid | (plane_time < floor_time))
    return torch.maximum(image, torch.where(gate_in_front, annulus, 0.0)).clamp(0.0, 1.0)


def render_annular_gate_rgb(
    state: QuadState,
    gate: AnnularGate,
    *,
    camera: CameraSpec = DEFAULT_VISUAL_CAMERA,
    gate_config: GateConfig = DEFAULT_GATE_CONFIG,
    gate_colour: tuple[float, float, float] = (0.08, 0.86, 0.66),
) -> Tensor:
    """Render one coloured annulus in the ordinary RGB gate world."""

    current = torch.zeros(
        state.position.shape[0],
        device=state.position.device,
        dtype=torch.long,
    )
    return render_annular_gates_rgb(
        state,
        (gate,),
        current_gate_index=current,
        camera=camera,
        gate_config=gate_config,
        role_colours=(gate_colour,),
    )


def render_annular_gates_rgb(
    state: QuadState,
    gates: Sequence[AnnularGate],
    *,
    current_gate_index: Tensor,
    camera: CameraSpec = DEFAULT_VISUAL_CAMERA,
    gate_config: GateConfig = DEFAULT_GATE_CONFIG,
    role_colours: tuple[tuple[float, float, float], ...] = (
        (0.08, 0.86, 0.66),
        (0.92, 0.22, 0.08),
        (0.22, 0.38, 0.92),
    ),
) -> Tensor:
    """Render role-coloured gates, a world-fixed grey floor, and black background.

    The result is ordinary linear RGB camera data with shape ``(B, 3, H, W)``.
    It contains no gate mask, bearing, range, optical flow, or other decoded task
    feature.  Texture is fixed in world coordinates, so motion in the image is caused
    only by motion of the camera. Passed gates are black, the current gate uses the
    first colour, and subsequent gates use the remaining fixed colour sequence.
    Ranks beyond the palette share its last colour. Optional checkerboard backs
    encode traversal direction without changing the role hue or the aperture.
    """

    if not gates:
        raise ValueError("at least one gate is required")
    if not role_colours:
        raise ValueError("at least one non-passed gate colour is required")
    batch = state.position.shape[0]
    if current_gate_index.shape != (batch,):
        raise ValueError("current_gate_index must have shape (batch,)")
    rays_body = make_pinhole_rays(
        camera,
        device=state.position.device,
        dtype=state.position.dtype,
    )
    # rotation_matrix uses a +X-forward/+Z-up frame, for which +Y is camera-left.
    # make_pinhole_rays follows image columns, so mirror its lateral component here.
    rays_body = torch.stack((rays_body[..., 0], -rays_body[..., 1], rays_body[..., 2]), dim=-1)
    body_to_world = rotation_matrix(state.euler)
    rays_world = torch.einsum("bij,hwj->bhwi", body_to_world, rays_body)
    origin = state.position[:, None, None, :]

    downward = rays_world[..., 2] < -1.0e-4
    floor_time = -origin[..., 2] / rays_world[..., 2].clamp_max(-1.0e-4)
    floor_valid = downward & (floor_time > 0.0)
    floor_x = origin[..., 0] + floor_time * rays_world[..., 0]
    floor_y = origin[..., 1] + floor_time * rays_world[..., 1]
    broad_texture = torch.sin(0.73 * floor_x + 0.41 * floor_y + 2.1) * torch.sin(
        0.37 * floor_x - 0.89 * floor_y - 1.26
    )
    fine_texture = torch.sin(2.31 * floor_x - 1.67 * floor_y + 2.94) * torch.cos(
        1.13 * floor_x + 2.03 * floor_y - 2.1
    )
    floor_texture = 0.68 * broad_texture + 0.32 * fine_texture
    floor_base = state.position.new_tensor((0.235, 0.235, 0.235))
    floor_tint = state.position.new_tensor((0.023, 0.021, 0.018))
    floor_rgb = floor_base + floor_texture[..., None] * floor_tint
    image = torch.zeros(
        batch,
        camera.height,
        camera.width,
        3,
        device=state.position.device,
        dtype=state.position.dtype,
    )
    image = torch.where(floor_valid[..., None], floor_rgb, image)
    infinity = torch.full_like(floor_time, float("inf"))
    nearest_surface = torch.where(floor_valid, floor_time, infinity)
    colour_table = state.position.new_tensor(role_colours)

    for gate_number, gate in enumerate(gates):
        if gate.center.shape != (batch, 3) or gate.yaw.shape != (batch,):
            raise ValueError("each gate must contain one centre and yaw per batch item")
        normal = gate.normal[:, None, None, :]
        plane_numerator = ((gate.center[:, None, None, :] - origin) * normal).sum(dim=-1)
        plane_denominator = (rays_world * normal).sum(dim=-1)
        safe_denominator = torch.where(
            plane_denominator.abs() > 1.0e-4,
            plane_denominator,
            torch.ones_like(plane_denominator),
        )
        plane_time = plane_numerator / safe_denominator
        plane_valid = (plane_denominator.abs() > 1.0e-4) & (plane_time > 0.0)
        hit = origin + plane_time[..., None] * rays_world
        hit_offset = hit - gate.center[:, None, None, :]
        gate_lateral = (hit_offset * gate.lateral[:, None, None, :]).sum(dim=-1)
        gate_vertical = hit_offset[..., 2]
        radial = torch.sqrt(gate_lateral.square() + gate_vertical.square() + 1.0e-8)
        outer = torch.sigmoid((gate_config.outer_radius - radial) / gate_config.edge_softness)
        inner = torch.sigmoid((radial - gate_config.inner_radius) / gate_config.edge_softness)
        annulus_alpha = outer * inner
        gate_visible = plane_valid & (plane_time < nearest_surface)
        annulus_alpha = annulus_alpha * gate_visible

        # A small top/bottom paint cue makes pose observable without breaking the
        # left/right mirror symmetry needed by the paired steering curriculum.
        paint = 0.82 + 0.18 * torch.sigmoid(gate_vertical / 0.06)
        if gate_config.back_pattern == "checkerboard":
            # Coarse cells fixed to the gate, with reflection symmetry about its
            # centreline. Legal approach is signed distance < 0 (numerator > 0).
            cells = torch.floor(gate_lateral.abs() / 0.22) + torch.floor(
                gate_vertical.abs() / 0.22
            )
            checker = 0.22 + 0.78 * torch.remainder(cells, 2.0)
            paint = paint * torch.where(plane_numerator < 0.0, checker, 1.0)
        role = gate_number - current_gate_index
        colour_index = role.clamp(0, len(role_colours) - 1)
        colour = colour_table[colour_index]
        colour = torch.where((role >= 0)[:, None], colour, torch.zeros_like(colour))
        gate_rgb = paint[..., None] * colour[:, None, None, :]
        image = image * (1.0 - annulus_alpha[..., None]) + gate_rgb * annulus_alpha[..., None]
        nearest_surface = torch.where(annulus_alpha > 1.0e-3, plane_time, nearest_surface)
    return image.clamp(0.0, 1.0).permute(0, 3, 1, 2)


def crossing_geometry(
    previous_position: Tensor,
    position: Tensor,
    gate: AnnularGate,
) -> tuple[Tensor, Tensor]:
    """Return directed-crossing mask and radial distance at the swept plane crossing."""

    directed, lateral, vertical = crossing_coordinates(previous_position, position, gate)
    radial = torch.sqrt(lateral.square() + vertical.square())
    return directed, radial


def crossing_coordinates(
    previous_position: Tensor,
    position: Tensor,
    gate: AnnularGate,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return directed-crossing mask and signed in-plane crossing coordinates."""

    previous_signed, _, _ = gate_coordinates(previous_position, gate)
    signed, _, _ = gate_coordinates(position, gate)
    directed = (previous_signed < 0.0) & (signed >= 0.0)
    fraction = (-previous_signed / (signed - previous_signed).clamp_min(1.0e-8)).clamp(0.0, 1.0)
    crossing = previous_position + fraction[:, None] * (position - previous_position)
    _, lateral, vertical = gate_coordinates(crossing, gate)
    return directed, lateral, vertical


def classify_gate_crossing(
    previous_position: Tensor,
    position: Tensor,
    gate: AnnularGate,
    config: GateConfig = DEFAULT_GATE_CONFIG,
) -> tuple[Tensor, Tensor, Tensor]:
    """Classify a directed plane crossing as clean pass, ring collision, or miss."""

    directed, radial = crossing_geometry(previous_position, position, gate)
    clean_radius = config.inner_radius - config.drone_radius
    pass_gate = directed & (radial <= clean_radius)
    ring_collision = (
        directed & (radial > clean_radius) & (radial <= config.outer_radius + config.drone_radius)
    )
    miss = directed & ~pass_gate & ~ring_collision
    return pass_gate, ring_collision, miss


def teacher_gate_rc_state_feedback(
    state: QuadState,
    gate: AnnularGate,
    hover_config: HoverConfig = DEFAULT_HOVER_CONFIG,
) -> Tensor:
    """Full-state reference controller retained for teacher quality audits."""

    travel_direction = gate.center / torch.linalg.vector_norm(
        gate.center, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    line_progress = (state.position * travel_direction).sum(dim=1, keepdim=True)
    line_error = state.position - line_progress * travel_direction
    desired_velocity = 0.82 * travel_direction - 1.45 * line_error
    desired_velocity[:, 2] = (1.1 * (gate.center[:, 2] - state.position[:, 2])).clamp(-0.6, 0.9)
    velocity_error = desired_velocity - state.velocity
    drag_compensation = hover_config.linear_drag * state.velocity / hover_config.mass
    desired_acceleration = 2.4 * velocity_error + drag_compensation

    yaw = state.euler[:, 2]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    body_forward_accel = cy * desired_acceleration[:, 0] + sy * desired_acceleration[:, 1]
    body_left_accel = -sy * desired_acceleration[:, 0] + cy * desired_acceleration[:, 1]
    desired_roll = (-body_left_accel / 9.81).clamp(-0.32, 0.32)
    desired_pitch = (body_forward_accel / 9.81).clamp(-0.32, 0.32)
    roll_rate = 4.0 * (desired_roll - state.euler[:, 0]) - 0.35 * state.rates[:, 0]
    pitch_rate = 4.0 * (desired_pitch - state.euler[:, 1]) - 0.35 * state.rates[:, 1]

    # Normal alignment is not required for a clean annular crossing.  Holding launch
    # heading makes the plane obliquity a real traversal angle rather than a yaw target.
    yaw_error = wrap_angle(-yaw)
    yaw_rate = 2.0 * yaw_error - 0.25 * state.rates[:, 2]
    hover = 1.0 / hover_config.thrust_to_weight
    tilt_compensation = 1.0 / (
        torch.cos(state.euler[:, 0]) * torch.cos(state.euler[:, 1])
    ).clamp_min(0.75)
    throttle = (
        hover * tilt_compensation + 0.08 * desired_acceleration[:, 2] - 0.05 * state.velocity[:, 2]
    ).clamp(0.0, 1.0)
    return torch.stack(
        (
            (roll_rate / hover_config.max_roll_pitch_rate).clamp(-1.0, 1.0),
            (pitch_rate / hover_config.max_roll_pitch_rate).clamp(-1.0, 1.0),
            (yaw_rate / hover_config.max_yaw_rate).clamp(-1.0, 1.0),
            throttle,
        ),
        dim=-1,
    )


def teacher_gate_rc(
    state: QuadState,
    gate: AnnularGate,
    hover_config: HoverConfig = DEFAULT_HOVER_CONFIG,
) -> Tensor:
    """Observation-compatible visual-servo teacher for connectome distillation.

    Relative gate bearing and elevation are privileged during training, but both are
    directly encoded by the rendered annulus.  Unlike the full-state reference above,
    this target uses no translational velocity, angular rate, or hidden course state.
    """

    to_gate_world = gate.center - state.position
    world_to_body = rotation_matrix(state.euler).transpose(1, 2)
    to_gate_body = torch.einsum("bij,bj->bi", world_to_body, to_gate_world)
    horizontal_range = torch.linalg.vector_norm(to_gate_body[:, :2], dim=1)
    bearing = torch.atan2(to_gate_body[:, 1], to_gate_body[:, 0])
    elevation = torch.atan2(to_gate_body[:, 2], horizontal_range.clamp_min(1.0e-6))

    approaching = to_gate_body[:, 0] > 0.0
    clearing = to_gate_body[:, 0] > -1.0
    desired_roll = torch.where(
        approaching,
        (-0.95 * bearing).clamp(-0.30, 0.30),
        torch.zeros_like(bearing),
    )
    desired_pitch = torch.where(
        clearing,
        torch.full_like(desired_roll, 0.24),
        torch.zeros_like(desired_roll),
    )
    roll_rate = 4.0 * (desired_roll - state.euler[:, 0])
    pitch_rate = 4.0 * (desired_pitch - state.euler[:, 1])
    yaw_rate = torch.zeros_like(roll_rate)

    # The actor is not given the episode's randomized mass.  A fixed 8% reserve keeps
    # the heaviest vehicle airborne; visual elevation feedback trims it before passage.
    hover = 1.08 / hover_config.thrust_to_weight
    tilt_compensation = 1.0 / (
        torch.cos(state.euler[:, 0]) * torch.cos(state.euler[:, 1])
    ).clamp_min(0.75)
    throttle = (
        hover * tilt_compensation
        + 0.30 * torch.where(approaching, elevation, torch.zeros_like(elevation))
    ).clamp(0.0, 1.0)
    return torch.stack(
        (
            (roll_rate / hover_config.max_roll_pitch_rate).clamp(-1.0, 1.0),
            (pitch_rate / hover_config.max_roll_pitch_rate).clamp(-1.0, 1.0),
            yaw_rate,
            throttle,
        ),
        dim=-1,
    )


def teacher_gate_rc_with_accelerometer(
    state: QuadState,
    gate: AnnularGate,
    hover_config: HoverConfig = DEFAULT_HOVER_CONFIG,
    feedback_gain: float = -0.16,
) -> Tensor:
    """Observation-compatible teacher augmented by centered body-Z specific force."""

    rc = teacher_gate_rc(state, gate, hover_config)
    centered_z = ((state.specific_force[:, 2] - 9.81) / (0.25 * 9.81)).clamp(-2.0, 2.0)
    throttle = (rc[:, 3] - feedback_gain * centered_z).clamp(0.0, 1.0)
    return torch.cat((rc[:, :3], throttle[:, None]), dim=1)
