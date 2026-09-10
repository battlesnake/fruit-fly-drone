from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_damping_routes import (  # noqa: E402
    cap_masked_displacement,
    masked_family_rms,
    project_masked_displacement,
    route_mask_arrays,
)


def test_route_mask_includes_direct_and_two_hop_but_not_motor_intermediate() -> None:
    # desc=0, VNC intrinsic=1, ordinary intermediate=2, other motor=3, throttle=4,
    # unrelated=5. Edge 0 is direct; 1+2 is legal two-hop; 3+4 uses a motor intermediate.
    edge_pre = np.array((0, 1, 2, 1, 3, 5))
    edge_post = np.array((4, 2, 4, 3, 4, 4))
    superclass = np.array(
        (
            "descending_neuron",
            "vnc_intrinsic",
            "ascending_neuron",
            "vnc_motor",
            "vnc_motor",
            "cb_intrinsic",
        )
    )

    edges, biases, report = route_mask_arrays(edge_pre, edge_post, superclass, np.array((4,)))

    assert edges.tolist() == [0, 1, 2]
    assert biases.tolist() == [2]
    assert report["direct_edges"] == 1
    assert report["all_motor_nodes_excluded_as_intermediates"] == 2


def test_masked_cap_preserves_family_ratio() -> None:
    displacement = {
        "edge_magnitude": torch.full((4,), 4.0e-5),
        "bias": torch.full((2,), 2.0e-5),
    }

    capped, scale = cap_masked_displacement(displacement, 2.0e-5)

    assert scale == pytest.approx(0.5)
    assert max(masked_family_rms(capped).values()) == pytest.approx(2.0e-5)
    assert capped["edge_magnitude"][0] / capped["bias"][0] == pytest.approx(2.0)


def test_masked_projection_satisfies_linear_constraints() -> None:
    displacement = {
        "edge_magnitude": torch.tensor((0.3, -0.2, 0.1)),
        "bias": torch.tensor((0.4, -0.1)),
    }
    rows = [
        {
            "edge_magnitude": torch.tensor((1.0, 0.0, 0.0)),
            "bias": torch.tensor((0.5, 0.0)),
        },
        {
            "edge_magnitude": torch.tensor((0.0, 1.0, 0.0)),
            "bias": torch.tensor((0.0, -0.25)),
        },
    ]
    residual = torch.tensor((0.2, -0.15))

    projected, report = project_masked_displacement(displacement, rows, residual)
    actual = torch.stack(
        [
            residual[index] + sum((row[name] * projected[name]).sum() for name in displacement)
            for index, row in enumerate(rows)
        ]
    )

    assert torch.allclose(actual, torch.zeros_like(actual), atol=1.0e-6)
    assert report["retained_rank"] == 2
