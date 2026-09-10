from __future__ import annotations

import math

import pytest
import torch

from flydrone.hover import DifferentiableQuad
from flydrone.visual_hover import (
    HELD_OUT_MARKER_HEIGHT_BAND,
    TRAIN_MARKER_HEIGHT_BANDS,
    CameraSpec,
    make_pinhole_rays,
    render_visual_hover_scene,
    sample_marker_heights,
)


def _centre_of_brightness(image: torch.Tensor) -> torch.Tensor:
    luminance = image.mean(dim=0)
    background = torch.quantile(luminance.flatten(), 0.6)
    weight = (luminance - background).clamp_min(0.0)
    rows = torch.arange(image.shape[1], dtype=image.dtype)
    return (weight.sum(dim=1) * rows).sum() / weight.sum()


def test_default_camera_has_requested_geometry_and_square_pixels() -> None:
    camera = CameraSpec()
    rays = make_pinhole_rays(camera, device=torch.device("cpu"), dtype=torch.float64)
    slopes_y = rays[..., 1] / rays[..., 0]
    slopes_z = rays[..., 2] / rays[..., 0]

    expected_vertical = math.degrees(
        2.0
        * math.atan(
            math.tan(math.radians(camera.horizontal_fov_degrees) / 2.0)
            * camera.height
            / camera.width
        )
    )
    horizontal_step = slopes_y[camera.height // 2, 1] - slopes_y[camera.height // 2, 0]
    vertical_step = slopes_z[0, camera.width // 2] - slopes_z[1, camera.width // 2]

    assert rays.shape == (200, 320, 3)
    assert camera.horizontal_fov_degrees == 125.0
    assert camera.vertical_fov_degrees == pytest.approx(expected_vertical)
    assert horizontal_step == pytest.approx(vertical_step)


def test_training_and_validation_marker_heights_are_varied_and_disjoint() -> None:
    torch.manual_seed(7)
    training = sample_marker_heights(4096, device=torch.device("cpu"), held_out=False)
    held_out = sample_marker_heights(4096, device=torch.device("cpu"), held_out=True)
    low, high = TRAIN_MARKER_HEIGHT_BANDS

    assert bool(((training >= low[0]) & (training <= low[1])).any())
    assert bool(((training >= high[0]) & (training <= high[1])).any())
    outside_holdout = (training < HELD_OUT_MARKER_HEIGHT_BAND[0]) | (
        training > HELD_OUT_MARKER_HEIGHT_BAND[1]
    )
    assert bool(outside_holdout.all())
    assert float(held_out.min()) >= HELD_OUT_MARKER_HEIGHT_BAND[0]
    assert float(held_out.max()) <= HELD_OUT_MARKER_HEIGHT_BAND[1]


def test_rgb_scene_has_distinct_textured_floor_and_wall() -> None:
    quad = DifferentiableQuad()
    position = torch.tensor([[0.0, 0.0, 1.0]])
    state = quad.initial_state(
        1, device=torch.device("cpu"), dtype=torch.float32, position=position
    )
    image = render_visual_hover_scene(state, torch.tensor([8.0]))
    upper = image[0, :, 20:60].mean(dim=(1, 2))
    lower = image[0, :, 160:195].mean(dim=(1, 2))

    assert image.shape == (1, 3, 200, 320)
    assert torch.all((0.0 <= image) & (image <= 1.0))
    assert not torch.allclose(upper, lower, atol=0.03)
    assert image[0, :, 20:60].std() > 0.001
    assert image[0, :, 160:195].std() > 0.001


def test_world_fixed_texture_produces_visual_motion() -> None:
    quad = DifferentiableQuad()
    first = quad.initial_state(
        1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        position=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    second = quad.initial_state(
        1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        position=torch.tensor([[0.17, 0.11, 1.0]]),
    )
    target = torch.tensor([8.0])
    first_image = render_visual_hover_scene(first, target)
    repeated = render_visual_hover_scene(first, target)
    moved_image = render_visual_hover_scene(second, target)

    assert torch.equal(first_image, repeated)
    assert not torch.allclose(first_image, moved_image)


def test_physical_marker_moves_down_as_camera_rises() -> None:
    quad = DifferentiableQuad()
    low = quad.initial_state(
        1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        position=torch.tensor([[0.0, 0.0, 0.5]]),
    )
    high = quad.initial_state(
        1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        position=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    target = torch.tensor([1.5])

    low_row = _centre_of_brightness(render_visual_hover_scene(low, target)[0])
    high_row = _centre_of_brightness(render_visual_hover_scene(high, target)[0])

    assert high_row > low_row
