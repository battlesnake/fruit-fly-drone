from __future__ import annotations

import math

import pytest
import torch

from flydrone.hover import DifferentiableQuad
from flydrone.visual_hover import (
    HELD_OUT_MARKER_HEIGHT_BAND,
    TRAIN_MARKER_HEIGHT_BANDS,
    CameraSpec,
    VisualScene,
    make_pinhole_rays,
    render_visual_hover_scene,
    sample_marker_heights,
    sample_visual_scenes,
    visual_scene_manifest,
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


def test_scene_randomization_reserves_wall_floor_style_combinations() -> None:
    torch.manual_seed(11)
    training = sample_visual_scenes(1024, device=torch.device("cpu"), held_out_combinations=False)
    held_out = sample_visual_scenes(1024, device=torch.device("cpu"), held_out_combinations=True)

    assert torch.equal(training.wall_style, training.floor_style)
    assert torch.equal(held_out.wall_style, 1 - held_out.floor_style)
    assert set(training.wall_style.tolist()) == {0, 1}
    assert set(held_out.wall_style.tolist()) == {0, 1}
    assert float(training.wall_x.min()) >= 4.25
    assert float(training.wall_x.max()) <= 5.35
    manifest = visual_scene_manifest()
    assert manifest["sampled_independently_of_marker_height"] is True
    assert manifest["style_combination_split"]["marginals_shared"] is True


@pytest.mark.parametrize("wall_style", [0, 1])
def test_scene_sampler_can_fix_held_out_wall_style_family(wall_style: int) -> None:
    scene = sample_visual_scenes(
        128,
        device=torch.device("cpu"),
        held_out_combinations=True,
        fixed_wall_style=wall_style,
    )

    assert bool((scene.wall_style == wall_style).all())
    assert bool((scene.floor_style == 1 - wall_style).all())


def test_scene_sampler_rejects_unknown_fixed_style() -> None:
    with pytest.raises(ValueError, match="fixed styles"):
        sample_visual_scenes(
            1,
            device=torch.device("cpu"),
            held_out_combinations=False,
            fixed_wall_style=2,
        )


def test_scene_sampler_balances_all_four_style_combinations() -> None:
    scene = sample_visual_scenes(
        400,
        device=torch.device("cpu"),
        held_out_combinations=False,
        all_style_combinations=True,
    )
    combinations = 2 * scene.wall_style + scene.floor_style

    assert torch.equal(torch.bincount(combinations), torch.full((4,), 100))


def test_all_style_mode_rejects_conflicting_style_selection() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        sample_visual_scenes(
            4,
            device=torch.device("cpu"),
            held_out_combinations=True,
            all_style_combinations=True,
        )


def test_paired_marker_render_keeps_randomized_background_identical() -> None:
    quad = DifferentiableQuad()
    state = quad.initial_state(
        1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        position=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    scene = sample_visual_scenes(1, device=torch.device("cpu"), held_out_combinations=False)
    low = render_visual_hover_scene(state, torch.tensor([0.8]), scene=scene)
    high = render_visual_hover_scene(state, torch.tensor([1.2]), scene=scene)
    repeated_low = render_visual_hover_scene(state, torch.tensor([0.8]), scene=scene)

    assert torch.equal(low, repeated_low)
    assert not torch.equal(low, high)


def test_scene_validation_rejects_unpaired_batch_fields() -> None:
    quad = DifferentiableQuad()
    state = quad.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    bad = VisualScene(
        wall_x=torch.ones(2),
        wall_phase=torch.ones(1),
        floor_phase=torch.ones(1),
        wall_texture_gain=torch.ones(1),
        floor_texture_gain=torch.ones(1),
        illumination=torch.ones(1),
        marker_gain=torch.ones(1),
        wall_style=torch.zeros(1, dtype=torch.long),
        floor_style=torch.zeros(1, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="shape"):
        render_visual_hover_scene(state, torch.ones(1), scene=bad)


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
