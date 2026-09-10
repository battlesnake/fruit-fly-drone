from __future__ import annotations

import numpy as np
import pytest

from flydrone.full_connectome_data import (
    RetinotopyAssignment,
    fixed_spectral_weights,
    fixed_visual_grid,
)


def test_visual_grid_uses_frozen_full_eye_bounds() -> None:
    assignments = [
        RetinotopyAssignment(1, 1, 1, 10, 10, 1),
        RetinotopyAssignment(2, 18, 20, 12, 9, 2),
        RetinotopyAssignment(3, 36, 39, 8, 8, 1),
    ]
    bounds = {"L": (1, 36, 1, 39), "R": (1, 36, 1, 39)}

    grid = fixed_visual_grid(assignments, ["L", "L", "R"], bounds)

    assert np.array_equal(grid[0], np.asarray((-1.0, 1.0), dtype=np.float32))
    assert grid[1, 0] == pytest.approx(-1.0 / 35.0)
    assert grid[1, 1] == pytest.approx(0.0)
    assert np.array_equal(grid[2], np.asarray((1.0, -1.0), dtype=np.float32))


def test_spectral_mapping_does_not_invent_an_ultraviolet_channel() -> None:
    weights = fixed_spectral_weights(["R1-R6", "R8p", "R8y"])

    assert weights.shape == (3, 3)
    assert weights[0].sum() == pytest.approx(1.0)
    assert np.array_equal(weights[1], np.asarray((0.0, 0.0, 1.0)))
    assert np.array_equal(weights[2], np.asarray((0.0, 1.0, 0.0)))
    with pytest.raises(ValueError, match="unsupported driven photoreceptor"):
        fixed_spectral_weights(["R7p"])
