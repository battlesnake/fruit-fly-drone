"""Predeclared height distributions for variable-marker hover training.

The deployed controller never receives these values.  They define simulator initial
conditions, training-only teacher targets, and evaluation strata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

TRAIN_ABSOLUTE_MARKER_BANDS = ((0.55, 0.85), (1.15, 1.45))
HELD_OUT_ABSOLUTE_MARKER_BAND = (0.90, 1.10)
CAMERA_HEIGHT_BANDS = ((0.65, 0.95), (1.05, 1.35))

HeightSplit = Literal["train", "held_out_marker", "held_out_combination"]


@dataclass(frozen=True)
class HeightConditions:
    marker_height: Tensor
    camera_height: Tensor
    marker_family: Tensor
    camera_family: Tensor
    split: HeightSplit


@dataclass(frozen=True)
class MarkerStepConditions:
    initial_marker_height: Tensor
    final_marker_height: Tensor
    direction: Tensor
    final_is_held_out: bool


@dataclass(frozen=True)
class MarkerPairConditions:
    centre_height: Tensor
    marker_a: Tensor
    marker_b: Tensor
    signed_direction: Tensor
    held_out: bool


def _uniform_by_family(
    families: Tensor,
    bands: tuple[tuple[float, float], tuple[float, float]],
    *,
    dtype: torch.dtype,
) -> Tensor:
    unit = torch.rand(families.shape, device=families.device, dtype=dtype)
    bounds = torch.tensor(bands, device=families.device, dtype=dtype)
    low = bounds[families, 0]
    high = bounds[families, 1]
    return low + unit * (high - low)


def sample_height_conditions(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    split: HeightSplit,
) -> HeightConditions:
    """Sample train, unseen-marker, or checkerboard combination conditions."""

    if batch < 1:
        raise ValueError("batch must be positive")
    camera_family = torch.randint(0, 2, (batch,), device=device)
    camera_height = _uniform_by_family(camera_family, CAMERA_HEIGHT_BANDS, dtype=dtype)
    if split == "held_out_marker":
        low, high = HELD_OUT_ABSOLUTE_MARKER_BAND
        marker_height = torch.empty(batch, device=device, dtype=dtype).uniform_(low, high)
        marker_family = torch.full((batch,), -1, device=device, dtype=torch.long)
    elif split in ("train", "held_out_combination"):
        marker_family = camera_family.clone() if split == "train" else 1 - camera_family
        marker_height = _uniform_by_family(marker_family, TRAIN_ABSOLUTE_MARKER_BANDS, dtype=dtype)
    else:
        raise ValueError(f"unknown height split: {split}")
    return HeightConditions(
        marker_height=marker_height,
        camera_height=camera_height,
        marker_family=marker_family,
        camera_family=camera_family,
        split=split,
    )


def sample_marker_steps(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    held_out_final: bool,
) -> MarkerStepConditions:
    """Sample balanced 20 cm steps whose final markers are train or held-out only."""

    if batch < 1:
        raise ValueError("batch must be positive")
    direction = torch.where(
        torch.arange(batch, device=device) % 2 == 0,
        torch.ones(batch, device=device, dtype=dtype),
        -torch.ones(batch, device=device, dtype=dtype),
    )
    unit = torch.rand(batch, device=device, dtype=dtype)
    if held_out_final:
        # Up: 0.70-0.85 -> 0.90-1.05. Down: 1.15-1.30 -> 0.95-1.10.
        initial_up = 0.70 + 0.15 * unit
        initial_down = 1.15 + 0.15 * unit
    else:
        # Both endpoints remain inside the declared training marker bands.
        # Up: 0.55-0.65 -> 0.75-0.85. Down: 1.35-1.45 -> 1.15-1.25.
        initial_up = 0.55 + 0.10 * unit
        initial_down = 1.35 + 0.10 * unit
    initial = torch.where(direction > 0.0, initial_up, initial_down)
    final = initial + 0.20 * direction
    return MarkerStepConditions(initial, final, direction, held_out_final)


def sample_marker_pairs(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    held_out: bool,
    half_step_metres: float = 0.05,
) -> MarkerPairConditions:
    """Sample paired markers with every rendered target inside its declared split."""

    if batch < 1:
        raise ValueError("batch must be positive")
    sampled_band = (
        HELD_OUT_ABSOLUTE_MARKER_BAND
        if held_out
        else TRAIN_ABSOLUTE_MARKER_BANDS[0]
    )
    if not 0.0 < half_step_metres < 0.5 * (sampled_band[1] - sampled_band[0]):
        raise ValueError("half_step_metres must fit strictly inside each sampled band")
    direction = torch.where(
        torch.arange(batch, device=device) % 2 == 0,
        torch.ones(batch, device=device, dtype=dtype),
        -torch.ones(batch, device=device, dtype=dtype),
    )
    unit = torch.rand(batch, device=device, dtype=dtype)
    if held_out:
        low, high = HELD_OUT_ABSOLUTE_MARKER_BAND
        centre = low + half_step_metres + unit * (high - low - 2.0 * half_step_metres)
    else:
        family = torch.randint(0, 2, (batch,), device=device)
        low_min, low_max = TRAIN_ABSOLUTE_MARKER_BANDS[0]
        high_min, high_max = TRAIN_ABSOLUTE_MARKER_BANDS[1]
        low_centre = low_min + half_step_metres + unit * (
            low_max - low_min - 2.0 * half_step_metres
        )
        high_centre = high_min + half_step_metres + unit * (
            high_max - high_min - 2.0 * half_step_metres
        )
        centre = torch.where(family == 0, low_centre, high_centre)
    delta = half_step_metres * direction
    return MarkerPairConditions(centre, centre + delta, centre - delta, direction, held_out)


def sample_cross_band_marker_pairs(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> MarkerPairConditions:
    """Sample direction-balanced pairs spanning the two legal training bands."""

    if batch < 1:
        raise ValueError("batch must be positive")
    direction = torch.where(
        torch.arange(batch, device=device) % 2 == 0,
        torch.ones(batch, device=device, dtype=dtype),
        -torch.ones(batch, device=device, dtype=dtype),
    )
    low = torch.empty(batch, device=device, dtype=dtype).uniform_(
        *TRAIN_ABSOLUTE_MARKER_BANDS[0]
    )
    high = torch.empty(batch, device=device, dtype=dtype).uniform_(
        *TRAIN_ABSOLUTE_MARKER_BANDS[1]
    )
    marker_a = torch.where(direction > 0.0, high, low)
    marker_b = torch.where(direction > 0.0, low, high)
    return MarkerPairConditions(
        centre_height=0.5 * (low + high),
        marker_a=marker_a,
        marker_b=marker_b,
        signed_direction=direction,
        held_out=False,
    )


def protocol_manifest() -> dict[str, object]:
    return {
        "training_absolute_marker_bands_metres": [
            list(band) for band in TRAIN_ABSOLUTE_MARKER_BANDS
        ],
        "held_out_absolute_marker_band_metres": list(HELD_OUT_ABSOLUTE_MARKER_BAND),
        "camera_height_bands_metres": [list(band) for band in CAMERA_HEIGHT_BANDS],
        "initial_height_combination_split": {
            "training": "marker family equals camera family",
            "held_out": "marker family differs from camera family",
            "marginals_shared": True,
            "scope": "initial/reset heights; flown camera height remains unconstrained",
        },
        "marker_step_metres": 0.20,
        "training_step_endpoints_stay_in_training_marker_bands": True,
        "evaluation_step_endpoints_finish_in_held_out_marker_band": True,
        "paired_marker_half_step_metres": 0.05,
        "all_paired_markers_stay_inside_their_declared_split": True,
    }
