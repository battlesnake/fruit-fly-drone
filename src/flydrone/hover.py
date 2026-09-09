"""Differentiable first-hover plant and connectome-constrained controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

AXES = ("roll", "pitch", "yaw", "throttle")
POOL_NAMES = (
    "roll_pos",
    "roll_neg",
    "pitch_pos",
    "pitch_neg",
    "yaw_pos",
    "yaw_neg",
    "throttle_pos",
    "throttle_neg",
)

PLANT_MODEL_VERSION = "hover-surrogate-v2"


@dataclass(frozen=True)
class HoverConfig:
    dt: float = 0.01
    mass: float = 0.035
    inertia_x: float = 3.0e-5
    inertia_y: float = 3.0e-5
    inertia_z: float = 5.0e-5
    thrust_to_weight: float = 1.5
    motor_time_constant: float = 0.045
    max_roll_pitch_rate: float = math.radians(160.0)
    max_yaw_rate: float = math.radians(180.0)
    max_roll_pitch_torque: float = 2.0e-3
    max_yaw_torque: float = 8.0e-4
    rate_gain: float = 0.018
    linear_drag: float = 0.12
    angular_drag: float = 2.0e-5
    stick_motor_strength: float = 100.0
    roll_stick_gain: float = 1.0
    stick_spring: float = 25.0
    stick_damping: float = 8.0
    foreleg_joint_limit: float = 0.65
    wall_x: float = 5.0
    target_band_width: float = 0.18
    camera_fov_degrees: float = 90.0


DEFAULT_HOVER_CONFIG = HoverConfig()


@dataclass
class QuadState:
    position: Tensor
    velocity: Tensor
    euler: Tensor
    rates: Tensor
    actuator: Tensor
    specific_force: Tensor

    def detach(self) -> QuadState:
        return QuadState(*(value.detach() for value in self.as_tuple()))

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.position,
            self.velocity,
            self.euler,
            self.rates,
            self.actuator,
            self.specific_force,
        )


@dataclass
class StickState:
    joint_position: Tensor
    joint_velocity: Tensor
    position: Tensor
    velocity: Tensor

    def detach(self) -> StickState:
        return StickState(
            self.joint_position.detach(),
            self.joint_velocity.detach(),
            self.position.detach(),
            self.velocity.detach(),
        )


def rotation_matrix(euler: Tensor) -> Tensor:
    """Body-to-world ZYX rotation for roll, pitch, yaw Euler angles."""

    roll, pitch, yaw = euler.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        dim=-1,
    ).reshape(*euler.shape[:-1], 3, 3)


def make_camera_rays(
    resolution: int, fov_degrees: float, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    half = math.tan(math.radians(fov_degrees) / 2.0)
    horizontal = torch.linspace(-half, half, resolution, device=device, dtype=dtype)
    vertical = torch.linspace(half, -half, resolution, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(vertical, horizontal, indexing="ij")
    rays = torch.stack((torch.ones_like(uu), uu, vv), dim=-1)
    return rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)


def render_target_band(
    state: QuadState,
    target_height: Tensor,
    *,
    resolution: int = 32,
    config: HoverConfig = DEFAULT_HOVER_CONFIG,
) -> Tensor:
    """Render a physical horizontal band on a wall and a grey floor.

    This is analytic soft rasterization, but the pixels arise from camera rays intersecting
    world geometry.  Target height is never passed to the controller as a scalar.
    """

    batch = state.position.shape[0]
    rays_body = make_camera_rays(
        resolution,
        config.camera_fov_degrees,
        device=state.position.device,
        dtype=state.position.dtype,
    )
    body_to_world = rotation_matrix(state.euler)
    rays_world = torch.einsum("bij,hwj->bhwi", body_to_world, rays_body)
    origin = state.position[:, None, None, :]

    wall_distance = config.wall_x - origin[..., 0]
    wall_time = wall_distance / rays_world[..., 0].clamp_min(1.0e-4)
    wall_valid = (rays_world[..., 0] > 1.0e-4) & (wall_time > 0.0)
    wall_height = origin[..., 2] + wall_time * rays_world[..., 2]
    band = torch.exp(
        -0.5 * torch.square((wall_height - target_height[:, None, None]) / config.target_band_width)
    )
    wall_pixels = torch.where(wall_valid, band, torch.zeros_like(band))

    downward = rays_world[..., 2] < -1.0e-4
    floor_time = -origin[..., 2] / rays_world[..., 2].clamp_max(-1.0e-4)
    floor_before_wall = downward & (floor_time > 0.0) & ((~wall_valid) | (floor_time < wall_time))
    floor_checker_x = origin[..., 0] + floor_time * rays_world[..., 0]
    floor_checker_y = origin[..., 1] + floor_time * rays_world[..., 1]
    # A faint continuous texture provides optic-flow information without encoding target Z.
    floor_texture = 0.24 + 0.035 * torch.sin(3.0 * floor_checker_x) * torch.sin(
        3.0 * floor_checker_y
    )
    image = torch.where(floor_before_wall, floor_texture, wall_pixels)
    return image.clamp(0.0, 1.0).reshape(batch, resolution, resolution)


class ConnectomeController(nn.Module):
    """Leaky-rate network whose only recurrent edges come from MaleCNS."""

    def __init__(
        self,
        graph_path: Path,
        neural_dt: float = 0.01,
        retinal_receptive_field: int = 1,
    ) -> None:
        super().__init__()
        if retinal_receptive_field < 1 or retinal_receptive_field % 2 != 1:
            raise ValueError("retinal_receptive_field must be a positive odd integer")
        graph = np.load(graph_path)
        node_ids = graph["node_ids"]
        edge_pre = graph["edge_pre"]
        edge_post = graph["edge_post"]
        edge_count = graph["edge_count"].astype(np.float32)
        edge_sign = graph["edge_sign"].astype(np.float32)
        visual_nodes = graph["visual_node_indices"]
        visual_hex = graph["visual_hex"].astype(np.float32)
        visual_eye = graph["visual_eye"]
        attitude_nodes = graph["attitude_node_indices"]
        attitude_channels = graph["attitude_channels"]
        acceleration_nodes = graph.get("acceleration_node_indices", np.empty(0, dtype=np.int64))
        acceleration_channels = graph.get("acceleration_channels", np.empty(0, dtype=np.int64))
        proprioception_nodes = graph.get("proprioception_node_indices", np.empty(0, dtype=np.int64))
        proprioception_channels = graph.get("proprioception_channels", np.empty(0, dtype=np.int64))
        pool_offsets = graph["output_pool_offsets"]
        pool_indices = graph["output_pool_indices"]

        if len(pool_offsets) != len(POOL_NAMES) + 1:
            raise ValueError("derived graph output pools do not match the controller contract")

        magnitude = np.log1p(edge_count)
        incoming = np.zeros(len(node_ids), dtype=np.float32)
        np.add.at(incoming, edge_post, magnitude)
        normalized = edge_sign * magnitude / np.maximum(incoming[edge_post], 1.0)

        self.n_nodes = len(node_ids)
        self.neural_dt = neural_dt
        self.retinal_receptive_field = retinal_receptive_field
        self.register_buffer("node_ids", torch.from_numpy(node_ids))
        self.register_buffer("edge_pre", torch.from_numpy(edge_pre))
        self.register_buffer("edge_post", torch.from_numpy(edge_post))
        initial_edge_weight = torch.from_numpy(normalized * 3.0)
        self.register_buffer("initial_edge_magnitude", initial_edge_weight.abs())
        self.register_buffer("edge_sign", initial_edge_weight.sign())
        self.register_buffer("visual_nodes", torch.from_numpy(visual_nodes))
        self.register_buffer("attitude_nodes", torch.from_numpy(attitude_nodes))
        self.register_buffer("attitude_channels", torch.from_numpy(attitude_channels))
        # Reconstruct these interface mappings from the graph.  Non-persistent buffers
        # keep the committed pre-accelerometer checkpoints strictly loadable.
        self.register_buffer(
            "acceleration_nodes", torch.from_numpy(acceleration_nodes), persistent=False
        )
        self.register_buffer(
            "acceleration_channels",
            torch.from_numpy(acceleration_channels),
            persistent=False,
        )
        self.register_buffer(
            "proprioception_nodes", torch.from_numpy(proprioception_nodes), persistent=False
        )
        self.register_buffer(
            "proprioception_channels",
            torch.from_numpy(proprioception_channels),
            persistent=False,
        )
        self.register_buffer("pool_offsets", torch.from_numpy(pool_offsets))
        self.register_buffer("pool_indices", torch.from_numpy(pool_indices))
        self.register_buffer("visual_grid", self._make_visual_grid(visual_hex, visual_eye))

        # MaleCNS weights are synapse counts, not physiological strengths.  The mask and
        # transmitter-sign hypothesis stay fixed while nonnegative magnitudes are learned.
        self.edge_magnitude = nn.Parameter(initial_edge_weight.abs().clone())
        self.bias = nn.Parameter(torch.zeros(self.n_nodes))
        # MaleCNS supplies no physiological time constants.  Start every selected neuron
        # at about 21 ms; recurrence still carries all history and tau remains trainable.
        initial_raw_tau = torch.full((self.n_nodes,), -3.0)
        self.register_buffer("initial_raw_time_constant", initial_raw_tau)
        self.raw_time_constant = nn.Parameter(initial_raw_tau.clone())

    @staticmethod
    def _make_visual_grid(visual_hex: np.ndarray, visual_eye: np.ndarray) -> Tensor:
        grid = np.zeros_like(visual_hex, dtype=np.float32)
        for eye in (-1, 1):
            selected = visual_eye == eye
            points = visual_hex[selected]
            low, span = points.min(axis=0), np.ptp(points, axis=0)
            normalized = 2.0 * (points - low) / np.maximum(span, 1.0) - 1.0
            # Hex coordinates are not Cartesian pixels; this affine map is a declared,
            # fixed engineering approximation.  The vertical axis is flipped for grid_sample.
            grid[selected, 0] = normalized[:, 0]
            grid[selected, 1] = -normalized[:, 1]
        return torch.from_numpy(grid).reshape(1, len(grid), 1, 2)

    @property
    def time_constant(self) -> Tensor:
        return 0.01 + 0.24 * torch.sigmoid(self.raw_time_constant)

    def project_parameters(self) -> None:
        """Enforce fixed transmitter signs after an optimizer update."""

        with torch.no_grad():
            self.edge_magnitude.clamp_(min=0.0, max=8.0)

    def initial_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch, self.n_nodes, device=device, dtype=dtype)

    def sample_retina(self, image: Tensor) -> Tensor:
        if self.retinal_receptive_field > 1:
            image = functional.avg_pool2d(
                image[:, None],
                self.retinal_receptive_field,
                stride=1,
                padding=self.retinal_receptive_field // 2,
            )[:, 0]
        grid = self.visual_grid.expand(image.shape[0], -1, -1, -1)
        sampled = functional.grid_sample(
            image[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return sampled[:, 0, :, 0]

    @property
    def uses_accelerometer(self) -> bool:
        return bool(self.acceleration_nodes.numel())

    @property
    def uses_proprioception(self) -> bool:
        return bool(self.proprioception_nodes.numel())

    def sensory_drive(
        self,
        image: Tensor,
        roll_pitch: Tensor,
        body_specific_force: Tensor | None = None,
        stick_position: Tensor | None = None,
    ) -> Tensor:
        drive = torch.zeros(image.shape[0], self.n_nodes, device=image.device, dtype=image.dtype)
        retina = self.sample_retina(image)
        drive = drive.index_add(1, self.visual_nodes, 5.0 * retina)
        normalized_attitude = (roll_pitch / math.radians(30.0)).clamp(-1.5, 1.5)
        push_pull = torch.stack(
            (
                normalized_attitude[:, 0].clamp_min(0.0),
                (-normalized_attitude[:, 0]).clamp_min(0.0),
                normalized_attitude[:, 1].clamp_min(0.0),
                (-normalized_attitude[:, 1]).clamp_min(0.0),
            ),
            dim=-1,
        )
        attitude = push_pull[:, self.attitude_channels]
        drive = drive.index_add(1, self.attitude_nodes, 4.0 * attitude)
        if self.uses_accelerometer:
            if body_specific_force is None:
                raise ValueError("this connectome graph requires body specific force")
            # The first causal altitude experiment uses only centered body-Z specific
            # force.  No world-frame rotation, mass estimate, integration, or history is
            # computed outside the connectome.
            centered_z = ((body_specific_force[:, 2] - 9.81) / (0.25 * 9.81)).clamp(-2.0, 2.0)
            acceleration_push_pull = torch.stack(
                (centered_z.clamp_min(0.0), (-centered_z).clamp_min(0.0)), dim=-1
            )
            acceleration = acceleration_push_pull[:, self.acceleration_channels]
            drive = drive.index_add(1, self.acceleration_nodes, 2.0 * acceleration)
        if self.uses_proprioception:
            if stick_position is None:
                raise ValueError("this connectome graph requires foreleg stick position")
            throttle_joint = stick_position[:, 3].clamp(-1.0, 1.0)
            position_channels = torch.stack(
                ((throttle_joint + 1.0) / 2.0, (1.0 - throttle_joint) / 2.0), dim=-1
            )
            proprioception = position_channels[:, self.proprioception_channels]
            drive = drive.index_add(1, self.proprioception_nodes, 2.0 * proprioception)
        return drive

    def forward(
        self,
        image: Tensor,
        roll_pitch: Tensor,
        state: Tensor,
        body_specific_force: Tensor | None = None,
        stick_position: Tensor | None = None,
        privileged_throttle_pool_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # Recurrent values are deviations from each neuron's baseline rate.  A centered
        # activation avoids multiplying small sensory changes by sigmoid'(0)=0.25 at
        # every hop; physical motor-pool rates below remain nonnegative sigmoids.
        activity = torch.tanh(state)
        edge_weight = self.edge_sign * self.edge_magnitude
        messages = activity[:, self.edge_pre] * edge_weight
        recurrent = torch.zeros_like(state).index_add(1, self.edge_post, messages)
        drive = (
            recurrent
            + self.bias
            + self.sensory_drive(image, roll_pitch, body_specific_force, stick_position)
        )
        if privileged_throttle_pool_bias is not None:
            if privileged_throttle_pool_bias.shape != (image.shape[0],):
                raise ValueError("privileged throttle-pool bias must have shape (batch,)")
            throttle_positive_begin = int(self.pool_offsets[6].item())
            throttle_positive_end = int(self.pool_offsets[7].item())
            throttle_negative_end = int(self.pool_offsets[8].item())
            throttle_positive = self.pool_indices[throttle_positive_begin:throttle_positive_end]
            throttle_negative = self.pool_indices[throttle_positive_end:throttle_negative_end]
            positive_values = privileged_throttle_pool_bias[:, None].expand(
                -1, len(throttle_positive)
            )
            negative_values = -privileged_throttle_pool_bias[:, None].expand(
                -1, len(throttle_negative)
            )
            drive = drive.index_add(1, throttle_positive, positive_values)
            drive = drive.index_add(1, throttle_negative, negative_values)
        # Membrane state is bounded for numerical stability but not to [-1, 1], which
        # would artificially cap an antagonist pair below the calibrated stick range.
        target = 5.0 * torch.tanh(drive / 5.0)
        alpha = 1.0 - torch.exp(-self.neural_dt / self.time_constant)
        next_state = state + alpha * (target - state)
        return self.motor_drive(next_state), next_state

    def motor_drive(self, state: Tensor) -> Tensor:
        activity = torch.sigmoid(state)
        means = []
        for pool in range(len(POOL_NAMES)):
            begin = int(self.pool_offsets[pool].item())
            end = int(self.pool_offsets[pool + 1].item())
            means.append(activity[:, self.pool_indices[begin:end]].mean(dim=1))
        pools = torch.stack(means, dim=-1)
        return torch.stack(
            (
                pools[:, 0] - pools[:, 1],
                pools[:, 2] - pools[:, 3],
                pools[:, 4] - pools[:, 5],
                pools[:, 6] - pools[:, 7],
            ),
            dim=-1,
        )


class ForelegStickPlant(nn.Module):
    """Abstract bilateral 2-DoF endpoints constrained to Mode-2 stick caps.

    This is deliberately smaller than an articulated fly-leg model: each foreleg is a
    two-axis gimbal whose joint angles determine a unit-length tarsus endpoint.  It gives
    the action path physical joint state and forward kinematics without claiming FlyGym
    morphology, identified muscles, or contact mechanics.
    """

    def __init__(self, config: HoverConfig = DEFAULT_HOVER_CONFIG) -> None:
        super().__init__()
        self.config = config
        # Joint order maps to roll, pitch, yaw, throttle.  Throttle rests at minimum.
        self.register_buffer("rest", torch.tensor((0.0, 0.0, 0.0, -1.0)))

    def initial_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> StickState:
        rest = self.rest.to(device=device, dtype=dtype).expand(batch, -1).clone()
        sine_limit = math.sin(self.config.foreleg_joint_limit)
        joint_rest = torch.asin((rest * sine_limit).clamp(-1.0, 1.0))
        zeros = torch.zeros_like(rest)
        return StickState(joint_rest, zeros.clone(), rest, zeros)

    def forward(self, motor_drive: Tensor, state: StickState) -> tuple[Tensor, StickState]:
        limit = self.config.foreleg_joint_limit
        sine_limit = math.sin(limit)
        joint_rest = torch.asin((self.rest * sine_limit).clamp(-1.0, 1.0))
        axis_gain = motor_drive.new_tensor((self.config.roll_stick_gain, 1.0, 1.0, 1.0))
        acceleration = (
            self.config.stick_motor_strength * axis_gain * sine_limit * motor_drive
            - self.config.stick_spring * (state.joint_position - joint_rest)
            - self.config.stick_damping * state.joint_velocity
        )
        joint_velocity = state.joint_velocity + self.config.dt * acceleration
        joint_position = (state.joint_position + self.config.dt * joint_velocity).clamp(
            -limit, limit
        )
        blocked = ((joint_position <= -limit) & (joint_velocity < 0.0)) | (
            (joint_position >= limit) & (joint_velocity > 0.0)
        )
        joint_velocity = torch.where(blocked, torch.zeros_like(joint_velocity), joint_velocity)
        # Forward kinematics of each two-axis gimbal puts the tarsus on the stick cap.
        # The cap's normalized displacement is therefore measured from sin(joint angle),
        # rather than calculated directly from neural activity.
        position = (torch.sin(joint_position) / sine_limit).clamp(-1.0, 1.0)
        velocity = (position - state.position) / self.config.dt
        measured_rc = torch.cat((position[:, :3], (position[:, 3:4] + 1.0) / 2.0), dim=1)
        return measured_rc, StickState(joint_position, joint_velocity, position, velocity)

    def tarsus_positions(self, joint_position: Tensor) -> Tensor:
        """Return left/right tarsus XYZ positions from bilateral two-axis joints."""

        left_angles = joint_position[:, (2, 3)]
        right_angles = joint_position[:, (0, 1)]

        def endpoint(angles: Tensor) -> Tensor:
            lateral = torch.sin(angles[:, 0])
            forward = torch.sin(angles[:, 1])
            down = -torch.sqrt((1.0 - lateral.square() - forward.square()).clamp_min(0.0))
            return torch.stack((lateral, forward, down), dim=-1)

        return torch.stack((endpoint(left_angles), endpoint(right_angles)), dim=1)


class DifferentiableQuad(nn.Module):
    """Lumped 6-DoF quad surrogate with an acro rate controller and actuator lag."""

    def __init__(self, config: HoverConfig = DEFAULT_HOVER_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "inertia", torch.tensor((config.inertia_x, config.inertia_y, config.inertia_z))
        )

    def initial_state(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        position: Tensor | None = None,
        euler: Tensor | None = None,
    ) -> QuadState:
        zeros3 = torch.zeros(batch, 3, device=device, dtype=dtype)
        if position is None:
            position = zeros3.clone()
        if euler is None:
            euler = zeros3.clone()
        world_up = zeros3.new_tensor((0.0, 0.0, 9.81)).expand(batch, -1)
        specific_force = torch.einsum(
            "bij,bj->bi", rotation_matrix(euler).transpose(1, 2), world_up
        )
        return QuadState(
            position=position,
            velocity=zeros3.clone(),
            euler=euler,
            rates=zeros3.clone(),
            actuator=torch.zeros(batch, 4, device=device, dtype=dtype),
            specific_force=specific_force,
        )

    def forward(self, rc: Tensor, state: QuadState, mass_scale: Tensor | None = None) -> QuadState:
        config = self.config
        batch = rc.shape[0]
        if mass_scale is None:
            mass_scale = torch.ones(batch, device=rc.device, dtype=rc.dtype)
        mass = config.mass * mass_scale
        max_rates = rc.new_tensor(
            (config.max_roll_pitch_rate, config.max_roll_pitch_rate, config.max_yaw_rate)
        )
        desired_rates = rc[:, :3] * max_rates
        rate_error = desired_rates - state.rates
        torque_limit = rc.new_tensor(
            (config.max_roll_pitch_torque, config.max_roll_pitch_torque, config.max_yaw_torque)
        )
        torque_command = (config.rate_gain * rate_error).clamp(-torque_limit, torque_limit)
        # A lumped wrench replaces explicit rotors and a mixer.  Differential thrust
        # cannot create new torque at zero collective; authority reaches its configured
        # limit by half throttle, leaving normal hover commands unaffected.
        torque_authority = (2.0 * rc[:, 3:4]).clamp(0.0, 1.0)
        torque_command = torque_command * torque_authority
        actuator_target = torch.cat((rc[:, 3:4], torque_command), dim=1)
        motor_alpha = 1.0 - math.exp(-config.dt / config.motor_time_constant)
        actuator = state.actuator + motor_alpha * (actuator_target - state.actuator)

        rotation = rotation_matrix(state.euler)
        body_up = rotation[:, :, 2]
        max_thrust = config.thrust_to_weight * config.mass * 9.81
        thrust = max_thrust * actuator[:, 0]
        gravity = rc.new_tensor((0.0, 0.0, -9.81)).expand(batch, -1)
        acceleration = gravity + body_up * (thrust / mass)[:, None]
        acceleration = acceleration - config.linear_drag * state.velocity / mass[:, None]
        velocity = state.velocity + config.dt * acceleration
        position = state.position + config.dt * velocity

        inertia = self.inertia.to(dtype=rc.dtype) * mass_scale[:, None]
        angular_momentum = inertia * state.rates
        gyroscopic = torch.linalg.cross(state.rates, angular_momentum, dim=1)
        angular_acceleration = (
            actuator[:, 1:] - gyroscopic - config.angular_drag * state.rates
        ) / inertia
        rates = state.rates + config.dt * angular_acceleration

        roll, pitch = state.euler[:, 0], state.euler[:, 1]
        p, q, r = rates.unbind(dim=1)
        cos_pitch = torch.cos(pitch).clamp_min(0.15)
        tan_pitch = torch.sin(pitch) / cos_pitch
        euler_rate = torch.stack(
            (
                p + torch.sin(roll) * tan_pitch * q + torch.cos(roll) * tan_pitch * r,
                torch.cos(roll) * q - torch.sin(roll) * r,
                torch.sin(roll) / cos_pitch * q + torch.cos(roll) / cos_pitch * r,
            ),
            dim=1,
        )
        euler = state.euler + config.dt * euler_rate

        on_ground = position[:, 2] < 0.0
        position = torch.cat((position[:, :2], position[:, 2:3].clamp_min(0.0)), dim=1)
        vertical_velocity = torch.where(
            on_ground & (velocity[:, 2] < 0.0), torch.zeros_like(velocity[:, 2]), velocity[:, 2]
        )
        velocity = torch.cat((velocity[:, :2], vertical_velocity[:, None]), dim=1)
        completed_world_acceleration = (velocity - state.velocity) / config.dt
        specific_force_world = completed_world_acceleration - gravity
        specific_force = torch.einsum("bij,bj->bi", rotation.transpose(1, 2), specific_force_world)
        return QuadState(position, velocity, euler, rates, actuator, specific_force)


def teacher_rc(
    state: QuadState, target_height: Tensor, config: HoverConfig = DEFAULT_HOVER_CONFIG
) -> Tensor:
    """Privileged training-only stabilizer; never used by controller evaluation."""

    roll_rate = -3.5 * state.euler[:, 0] - 0.45 * state.rates[:, 0]
    pitch_rate = -3.5 * state.euler[:, 1] - 0.45 * state.rates[:, 1]
    roll = (roll_rate / config.max_roll_pitch_rate).clamp(-1.0, 1.0)
    pitch = (pitch_rate / config.max_roll_pitch_rate).clamp(-1.0, 1.0)
    yaw = (-0.4 * state.rates[:, 2] / config.max_yaw_rate).clamp(-1.0, 1.0)
    hover = 1.0 / config.thrust_to_weight
    throttle = (
        hover + 0.42 * (target_height - state.position[:, 2]) - 0.18 * state.velocity[:, 2]
    ).clamp(0.0, 1.0)
    return torch.stack((roll, pitch, yaw, throttle), dim=1)


def motor_target_for_rc(rc: Tensor, config: HoverConfig = DEFAULT_HOVER_CONFIG) -> Tensor:
    """Steady-state antagonist difference needed to hold measured stick positions."""

    stick_target = torch.cat((rc[:, :3], 2.0 * rc[:, 3:4] - 1.0), dim=1)
    rest = rc.new_tensor((0.0, 0.0, 0.0, -1.0))
    sine_limit = math.sin(config.foreleg_joint_limit)
    target_joint = torch.asin((stick_target * sine_limit).clamp(-1.0, 1.0))
    rest_joint = torch.asin((rest * sine_limit).clamp(-1.0, 1.0))
    axis_gain = rc.new_tensor((config.roll_stick_gain, 1.0, 1.0, 1.0))
    return (
        config.stick_spring
        * (target_joint - rest_joint)
        / (config.stick_motor_strength * axis_gain * sine_limit)
    )
