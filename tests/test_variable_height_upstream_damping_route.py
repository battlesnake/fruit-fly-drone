from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_upstream_damping_route import (  # noqa: E402
    expanded_route_mask_arrays,
    project_in_reference_metric,
    reference_family_rms,
    scale_to_reference_cap,
)


def test_expanded_mask_adds_all_afferents_of_route_descending_neurons() -> None:
    # 0=descending route source, 1=VNC route source, 2=intermediate,
    # 3=throttle motor, 4/5=arbitrary afferent sources, 6=unrelated descending.
    pre = np.array([0, 1, 2, 4, 5, 6], dtype=np.int64)
    post = np.array([2, 2, 3, 0, 0, 5], dtype=np.int64)
    superclass = np.array(
        [
            "descending_neuron",
            "vnc_intrinsic",
            "vnc_intrinsic",
            "vnc_motor",
            "optic_lobes",
            "descending_neuron",
            "central",
        ]
    )

    edges, biases, added_edges, descending, counts = expanded_route_mask_arrays(
        pre, post, superclass, np.array([3], dtype=np.int64)
    )

    assert edges.tolist() == [0, 1, 2, 3, 4]
    assert biases.tolist() == [0, 2]
    assert added_edges.tolist() == [3, 4]
    assert descending.tolist() == [0]
    assert counts["selected_descending_neurons"] == 1


def test_reference_metric_keeps_shallow_denominators() -> None:
    values = {
        "edge_magnitude": torch.ones(24_628),
        "bias": torch.ones(606),
    }

    rms = reference_family_rms(values)

    assert rms["edge_magnitude"] == pytest.approx(2.0**0.5)
    assert rms["bias"] == pytest.approx(2.0**0.5)


def test_reference_cap_accounts_for_expanded_parameter_count() -> None:
    values = {
        "edge_magnitude": torch.full((24_628,), 2.0e-5),
        "bias": torch.zeros(606),
    }

    scaled, factor = scale_to_reference_cap(values, 2.0e-5)

    assert factor == pytest.approx(2.0**-0.5)
    assert reference_family_rms(scaled)["edge_magnitude"] == pytest.approx(2.0e-5)


def test_reference_metric_projection_satisfies_linearized_constraint() -> None:
    displacement = {
        "edge_magnitude": torch.tensor((0.3, -0.2)),
        "bias": torch.tensor((0.1,)),
    }
    rows = [
        {
            "edge_magnitude": torch.tensor((1.0, 0.5)),
            "bias": torch.tensor((-0.25,)),
        }
    ]
    residual = torch.tensor((0.04,))

    projected, report = project_in_reference_metric(displacement, rows, residual)
    remaining = residual[0] + sum((rows[0][name] * projected[name]).sum() for name in displacement)

    assert float(remaining) == pytest.approx(0.0, abs=1e-6)
    assert report["retained_rank"] == 1
