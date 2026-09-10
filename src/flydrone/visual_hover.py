"""High-field-of-view RGB scene for native visual hover experiments.

The renderer is deliberately analytic and headless.  It supplies ordinary camera
pixels, not optical flow, image differences, a horizon estimate, or a decoded target
position.  All texture is fixed in world coordinates so apparent motion is caused only
by camera motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from flydrone.hover import DEFAULT_HOVER_CONFIG, HoverConfig, QuadState, rotation_matrix


@dataclass(frozen=True)
class CameraSpec:
    """Rectangular pinhole camera with square pixels."""

    width: int = 320
    height: int = 200
    horizontal_fov_degrees: float = 125.0

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("camera dimensions must both be at least two pixels")
        if not 0.0 < self.horizontal_fov_degrees < 180.0:
            raise ValueError("horizontal field of view must lie between 0 and 180 degrees")

    @property
    def vertical_fov_degrees(self) -> float:
        half_width = math.tan(math.radians(self.horizontal_fov_degrees) / 2.0)
        half_height = half_width * self.height / self.width
        return math.degrees(2.0 * math.atan(half_height))


DEFAULT_VISUAL_CAMERA = CameraSpec()
TRAIN_MARKER_HEIGHT_BANDS = ((0.60, 0.85), (1.15, 1.40))
HELD_OUT_MARKER_HEIGHT_BAND = (0.90, 1.10)
WALL_DISTANCE_RANGE = (4.25, 5.35)
TEXTURE_PHASE_CENTRES = (0.55, 3.65)
TEXTURE_PHASE_JITTER = (-0.35, 0.35)
TEXTURE_GAIN_RANGE = (0.65, 1.35)
ILLUMINATION_RANGE = (0.82, 1.18)
MARKER_GAIN_RANGE = (0.90, 1.10)


@dataclass(frozen=True)
class VisualScene:
    """Per-environment nuisance variables used only by the physical renderer."""

    wall_x: Tensor
    wall_phase: Tensor
    floor_phase: Tensor
    wall_texture_gain: Tensor
    floor_texture_gain: Tensor
    illumination: Tensor
    marker_gain: Tensor
    wall_style: Tensor
    floor_style: Tensor


def sample_visual_scenes(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    held_out_combinations: bool,
    fixed_wall_style: int | None = None,
    fixed_floor_style: int | None = None,
    all_style_combinations: bool = False,
) -> VisualScene:
    """Sample fresh scene values with a checkerboard train/evaluation style split.

    Both phase families occur in both splits.  Training pairs matching wall/floor style
    families; evaluation pairs opposite families, reserving their combination while
    keeping each individual family familiar.  Every continuous nuisance is sampled
    independently of marker height by the caller.
    """

    if batch < 1:
        raise ValueError("batch must be positive")
    if fixed_wall_style not in (None, 0, 1) or fixed_floor_style not in (None, 0, 1):
        raise ValueError("fixed styles must be 0, 1, or None")
    if all_style_combinations and (
        held_out_combinations or fixed_wall_style is not None or fixed_floor_style is not None
    ):
        raise ValueError("all_style_combinations cannot be combined with fixed or held-out styles")
    if fixed_floor_style is not None and fixed_wall_style is None:
        raise ValueError("fixed_floor_style requires fixed_wall_style")
    if all_style_combinations:
        combinations = torch.arange(batch, device=device) % 4
        combinations = combinations[torch.randperm(batch, device=device)]
        wall_style = combinations // 2
        floor_style = combinations % 2
    else:
        wall_style = (
            torch.randint(0, 2, (batch,), device=device)
            if fixed_wall_style is None
            else torch.full((batch,), fixed_wall_style, device=device, dtype=torch.long)
        )
        if fixed_floor_style is not None:
            floor_style = torch.full(
                (batch,), fixed_floor_style, device=device, dtype=torch.long
            )
        else:
            floor_style = 1 - wall_style if held_out_combinations else wall_style.clone()
    phase_centres = torch.tensor(TEXTURE_PHASE_CENTRES, device=device, dtype=dtype)

    def uniform(low: float, high: float) -> Tensor:
        return torch.empty(batch, device=device, dtype=dtype).uniform_(low, high)

    return VisualScene(
        wall_x=uniform(*WALL_DISTANCE_RANGE),
        wall_phase=phase_centres[wall_style] + uniform(*TEXTURE_PHASE_JITTER),
        floor_phase=phase_centres[floor_style] + uniform(*TEXTURE_PHASE_JITTER),
        wall_texture_gain=uniform(*TEXTURE_GAIN_RANGE),
        floor_texture_gain=uniform(*TEXTURE_GAIN_RANGE),
        illumination=uniform(*ILLUMINATION_RANGE),
        marker_gain=uniform(*MARKER_GAIN_RANGE),
        wall_style=wall_style,
        floor_style=floor_style,
    )


def visual_scene_manifest() -> dict[str, object]:
    return {
        "wall_distance_range_metres": list(WALL_DISTANCE_RANGE),
        "texture_phase_centres": list(TEXTURE_PHASE_CENTRES),
        "texture_phase_jitter": list(TEXTURE_PHASE_JITTER),
        "wall_and_floor_texture_gain_range": list(TEXTURE_GAIN_RANGE),
        "illumination_range": list(ILLUMINATION_RANGE),
        "marker_gain_range": list(MARKER_GAIN_RANGE),
        "style_combination_split": {
            "training": "wall style equals floor style",
            "held_out": "wall style differs from floor style",
            "marginals_shared": True,
        },
        "sampled_independently_of_marker_height": True,
    }


def default_visual_scene(state: QuadState, config: HoverConfig) -> VisualScene:
    """Return the deterministic scene used by earlier visual-hover checkpoints."""

    batch = state.position.shape[0]
    zeros = torch.zeros(batch, device=state.position.device, dtype=state.position.dtype)
    ones = torch.ones_like(zeros)
    styles = torch.zeros(batch, device=state.position.device, dtype=torch.long)
    return VisualScene(
        wall_x=torch.full_like(zeros, config.wall_x),
        wall_phase=zeros + 0.35,
        floor_phase=zeros + 2.1,
        wall_texture_gain=ones,
        floor_texture_gain=ones,
        illumination=ones,
        marker_gain=ones,
        wall_style=styles,
        floor_style=styles.clone(),
    )


def _validate_scene(scene: VisualScene, state: QuadState) -> None:
    batch = state.position.shape[0]
    floating_fields = (
        scene.wall_x,
        scene.wall_phase,
        scene.floor_phase,
        scene.wall_texture_gain,
        scene.floor_texture_gain,
        scene.illumination,
        scene.marker_gain,
    )
    categorical_fields = (scene.wall_style, scene.floor_style)
    if any(value.shape != (batch,) for value in (*floating_fields, *categorical_fields)):
        raise ValueError("every visual-scene field must have shape (batch,)")
    if any(
        value.device != state.position.device for value in (*floating_fields, *categorical_fields)
    ):
        raise ValueError("visual-scene fields must share the physical state's device")
    # Do not introduce a host synchronization into every rendered CUDA frame. Values
    # sampled by this module are finite by construction; direct CPU callers still get a
    # useful validation error.
    if state.position.device.type == "cpu" and any(
        not bool(torch.isfinite(value).all()) for value in floating_fields
    ):
        raise ValueError("visual-scene fields must be finite")


def sample_marker_heights(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    held_out: bool,
) -> Tensor:
    """Sample disjoint train or validation marker-height distributions."""

    if batch < 1:
        raise ValueError("batch must be positive")
    if held_out:
        return torch.empty(batch, device=device, dtype=dtype).uniform_(*HELD_OUT_MARKER_HEIGHT_BAND)
    choose_high = torch.rand(batch, device=device) >= 0.5
    low = torch.empty(batch, device=device, dtype=dtype).uniform_(*TRAIN_MARKER_HEIGHT_BANDS[0])
    high = torch.empty(batch, device=device, dtype=dtype).uniform_(*TRAIN_MARKER_HEIGHT_BANDS[1])
    return torch.where(choose_high, high, low)


def make_pinhole_rays(
    camera: CameraSpec = DEFAULT_VISUAL_CAMERA,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return body-frame rays for a square-pixel rectangular pinhole camera.

    Body X is forward, Y points to image right, and Z points to image up.  The declared
    field of view spans the sensor boundaries; returned rays pass through pixel centres.
    """

    half_sensor_width = math.tan(math.radians(camera.horizontal_fov_degrees) / 2.0)
    focal_pixels = camera.width / (2.0 * half_sensor_width)
    columns = torch.arange(camera.width, device=device, dtype=dtype)
    rows = torch.arange(camera.height, device=device, dtype=dtype)
    horizontal = (columns + 0.5 - camera.width / 2.0) / focal_pixels
    vertical = (camera.height / 2.0 - rows - 0.5) / focal_pixels
    vv, uu = torch.meshgrid(vertical, horizontal, indexing="ij")
    rays = torch.stack((torch.ones_like(uu), uu, vv), dim=-1)
    return rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)


def _smooth_world_pattern(first: Tensor, second: Tensor, *, phase: float | Tensor) -> Tensor:
    """Small deterministic multiscale pattern with no per-frame randomness."""

    broad = torch.sin(0.73 * first + 0.41 * second + phase) * torch.sin(
        0.37 * first - 0.89 * second - 0.6 * phase
    )
    fine = torch.sin(2.31 * first - 1.67 * second + 1.4 * phase) * torch.cos(
        1.13 * first + 2.03 * second - phase
    )
    return 0.68 * broad + 0.32 * fine


def render_visual_hover_scene(
    state: QuadState,
    target_height: Tensor,
    *,
    camera: CameraSpec = DEFAULT_VISUAL_CAMERA,
    config: HoverConfig = DEFAULT_HOVER_CONFIG,
    scene: VisualScene | None = None,
) -> Tensor:
    """Render a textured floor, textured wall, and physical horizontal height marker.

    The result has shape ``(batch, 3, height, width)`` and contains linear RGB values.
    Texture coordinates are world coordinates.  This makes optic flow available to the
    connectome while keeping velocity, optical flow, and target geometry out of the actor
    interface.
    """

    batch = state.position.shape[0]
    if target_height.shape != (batch,):
        raise ValueError("target_height must have shape (batch,)")
    if scene is None:
        scene = default_visual_scene(state, config)
    _validate_scene(scene, state)

    rays_body = make_pinhole_rays(camera, device=state.position.device, dtype=state.position.dtype)
    body_to_world = rotation_matrix(state.euler)
    rays_world = torch.einsum("bij,hwj->bhwi", body_to_world, rays_body)
    origin = state.position[:, None, None, :]

    wall_distance = scene.wall_x[:, None, None] - origin[..., 0]
    wall_time = wall_distance / rays_world[..., 0].clamp_min(1.0e-4)
    wall_valid = (rays_world[..., 0] > 1.0e-4) & (wall_time > 0.0)
    wall_y = origin[..., 1] + wall_time * rays_world[..., 1]
    wall_z = origin[..., 2] + wall_time * rays_world[..., 2]

    # Cool, dark wall and warmer, brighter floor make the horizon available without an
    # engineered horizon channel.  Their low-contrast patterns use different spatial
    # spectra as well as different colours.
    wall_pattern = _smooth_world_pattern(
        wall_y,
        wall_z,
        phase=scene.wall_phase[:, None, None],
    )
    wall_base = state.position.new_tensor((0.105, 0.135, 0.175))
    wall_tint = state.position.new_tensor((0.014, 0.018, 0.024))
    wall_rgb = (
        wall_base + (wall_pattern * scene.wall_texture_gain[:, None, None])[..., None] * wall_tint
    )

    downward = rays_world[..., 2] < -1.0e-4
    floor_time = -origin[..., 2] / rays_world[..., 2].clamp_max(-1.0e-4)
    floor_before_wall = downward & (floor_time > 0.0) & ((~wall_valid) | (floor_time < wall_time))
    floor_x = origin[..., 0] + floor_time * rays_world[..., 0]
    floor_y = origin[..., 1] + floor_time * rays_world[..., 1]
    floor_pattern = _smooth_world_pattern(
        1.27 * floor_x,
        0.83 * floor_y,
        phase=scene.floor_phase[:, None, None],
    )
    floor_base = state.position.new_tensor((0.255, 0.235, 0.175))
    floor_tint = state.position.new_tensor((0.025, 0.022, 0.014))
    floor_rgb = (
        floor_base
        + (floor_pattern * scene.floor_texture_gain[:, None, None])[..., None] * floor_tint
    )

    illumination = scene.illumination[:, None, None, None]
    wall_rgb = wall_rgb * illumination
    floor_rgb = floor_rgb * illumination

    black = torch.zeros(
        batch,
        camera.height,
        camera.width,
        3,
        device=state.position.device,
        dtype=state.position.dtype,
    )
    image = torch.where(wall_valid[..., None], wall_rgb, black)
    image = torch.where(floor_before_wall[..., None], floor_rgb, image)

    # This is physical wall paint, not a screen-space overlay.  A smooth edge keeps the
    # renderer differentiable and preserves subpixel motion.
    marker_alpha = torch.exp(
        -0.5 * torch.square((wall_z - target_height[:, None, None]) / config.target_band_width)
    )
    marker_alpha = marker_alpha * wall_valid * ~floor_before_wall
    marker_rgb = (
        state.position.new_tensor((0.82, 0.52, 0.075))
        * illumination
        * scene.marker_gain[:, None, None, None]
    )
    image = image * (1.0 - marker_alpha[..., None]) + marker_rgb * marker_alpha[..., None]
    return image.clamp(0.0, 1.0).permute(0, 3, 1, 2)
