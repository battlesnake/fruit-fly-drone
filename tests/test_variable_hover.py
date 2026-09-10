from __future__ import annotations

import pytest
import torch

from flydrone.variable_hover import (
    CAMERA_HEIGHT_BANDS,
    HELD_OUT_ABSOLUTE_MARKER_BAND,
    TRAIN_ABSOLUTE_MARKER_BANDS,
    protocol_manifest,
    sample_cross_band_marker_pairs,
    sample_height_conditions,
    sample_marker_pairs,
    sample_marker_steps,
)


def _inside_union(values: torch.Tensor, bands: tuple[tuple[float, float], ...]) -> torch.Tensor:
    inside = torch.zeros_like(values, dtype=torch.bool)
    for low, high in bands:
        inside |= (values >= low) & (values <= high)
    return inside


def test_absolute_marker_holdout_is_disjoint_from_training() -> None:
    torch.manual_seed(1)
    training = sample_height_conditions(4096, device=torch.device("cpu"), split="train")
    held_out = sample_height_conditions(4096, device=torch.device("cpu"), split="held_out_marker")

    assert bool(_inside_union(training.marker_height, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert not bool(_inside_union(training.marker_height, (HELD_OUT_ABSOLUTE_MARKER_BAND,)).any())
    assert bool(_inside_union(held_out.marker_height, (HELD_OUT_ABSOLUTE_MARKER_BAND,)).all())
    assert not bool(_inside_union(held_out.marker_height, TRAIN_ABSOLUTE_MARKER_BANDS).any())


def test_checkerboard_height_split_shares_marginals_but_holds_out_pairs() -> None:
    torch.manual_seed(2)
    training = sample_height_conditions(4096, device=torch.device("cpu"), split="train")
    held_out = sample_height_conditions(
        4096, device=torch.device("cpu"), split="held_out_combination"
    )

    assert torch.equal(training.marker_family, training.camera_family)
    assert bool((held_out.marker_family != held_out.camera_family).all())
    assert set(training.marker_family.tolist()) == {0, 1}
    assert set(training.camera_family.tolist()) == {0, 1}
    assert set(held_out.marker_family.tolist()) == {0, 1}
    assert set(held_out.camera_family.tolist()) == {0, 1}
    assert bool(_inside_union(training.camera_height, CAMERA_HEIGHT_BANDS).all())
    assert bool(_inside_union(held_out.camera_height, CAMERA_HEIGHT_BANDS).all())


def test_marker_step_endpoints_obey_actual_absolute_height_split() -> None:
    torch.manual_seed(3)
    training = sample_marker_steps(4096, device=torch.device("cpu"), held_out_final=False)
    held_out = sample_marker_steps(4096, device=torch.device("cpu"), held_out_final=True)

    assert bool(_inside_union(training.initial_marker_height, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert bool(_inside_union(training.final_marker_height, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert bool(_inside_union(held_out.initial_marker_height, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert bool(_inside_union(held_out.final_marker_height, (HELD_OUT_ABSOLUTE_MARKER_BAND,)).all())
    assert torch.allclose(
        held_out.final_marker_height - held_out.initial_marker_height,
        0.20 * held_out.direction,
    )


def test_protocol_manifest_records_the_anti_shortcut_split() -> None:
    manifest = protocol_manifest()

    assert manifest["training_step_endpoints_stay_in_training_marker_bands"] is True
    assert manifest["evaluation_step_endpoints_finish_in_held_out_marker_band"] is True
    assert manifest["initial_height_combination_split"]["marginals_shared"] is True
    assert "unconstrained" in manifest["initial_height_combination_split"]["scope"]


def test_paired_markers_never_cross_the_absolute_height_split() -> None:
    torch.manual_seed(4)
    training = sample_marker_pairs(4096, device=torch.device("cpu"), held_out=False)
    held_out = sample_marker_pairs(4096, device=torch.device("cpu"), held_out=True)

    assert bool(_inside_union(training.centre_height, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert bool(_inside_union(training.marker_a, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert bool(_inside_union(training.marker_b, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert bool(_inside_union(held_out.centre_height, (HELD_OUT_ABSOLUTE_MARKER_BAND,)).all())
    assert bool(_inside_union(held_out.marker_a, (HELD_OUT_ABSOLUTE_MARKER_BAND,)).all())
    assert bool(_inside_union(held_out.marker_b, (HELD_OUT_ABSOLUTE_MARKER_BAND,)).all())
    assert torch.allclose(training.marker_a - training.marker_b, 0.10 * training.signed_direction)


@pytest.mark.parametrize("half_step", [0.025, 0.05, 0.10])
def test_training_pair_amplitudes_stay_inside_training_bands(half_step: float) -> None:
    torch.manual_seed(5)
    pairs = sample_marker_pairs(
        4096,
        device=torch.device("cpu"),
        held_out=False,
        half_step_metres=half_step,
    )

    assert bool(_inside_union(pairs.marker_a, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert bool(_inside_union(pairs.marker_b, TRAIN_ABSOLUTE_MARKER_BANDS).all())
    assert torch.allclose(
        pairs.marker_a - pairs.marker_b,
        2.0 * half_step * pairs.signed_direction,
    )


def test_cross_band_pairs_are_legal_and_direction_balanced() -> None:
    torch.manual_seed(6)
    pairs = sample_cross_band_marker_pairs(4096, device=torch.device("cpu"))
    low, high = TRAIN_ABSOLUTE_MARKER_BANDS
    a_low = (pairs.marker_a >= low[0]) & (pairs.marker_a <= low[1])
    a_high = (pairs.marker_a >= high[0]) & (pairs.marker_a <= high[1])
    b_low = (pairs.marker_b >= low[0]) & (pairs.marker_b <= low[1])
    b_high = (pairs.marker_b >= high[0]) & (pairs.marker_b <= high[1])

    assert bool(((a_low & b_high) | (a_high & b_low)).all())
    assert int((pairs.signed_direction > 0).sum()) == 2048
    assert bool(((pairs.marker_a - pairs.marker_b) * pairs.signed_direction > 0).all())
