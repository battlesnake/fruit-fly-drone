from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_pragmatic_two_gate_zero_shot import sample_two_gate_cases  # noqa: E402

from flydrone.hover import HoverConfig  # noqa: E402


def test_aligned_sequence_inserts_equally_spaced_collinear_gates() -> None:
    cases, gates = sample_two_gate_cases(
        3,
        seed=1234,
        device=torch.device("cpu"),
        hover_config=HoverConfig(),
        layout="aligned",
        spacing_range=(2.0, 2.0),
        gate_count=3,
    )

    assert len(gates) == 3
    first_displacement = cases.gate.center - cases.state.position
    slope = first_displacement[:, 1] / first_displacement[:, 0]
    for number, gate in enumerate(gates[1:], start=1):
        assert torch.allclose(
            gate.center[:, 0] - gates[0].center[:, 0],
            torch.full((6,), 2.0 * number),
        )
        assert torch.allclose(
            gate.center[:, 1] - gates[0].center[:, 1],
            2.0 * number * slope,
        )
        assert torch.allclose(gate.center[:, 2], torch.full((6,), 1.10))
        assert torch.equal(gate.yaw, gates[0].yaw)


def test_s_turn_rejects_more_than_two_gates() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        sample_two_gate_cases(
            1,
            seed=1234,
            device=torch.device("cpu"),
            hover_config=HoverConfig(),
            layout="s-turn",
            gate_count=3,
        )


def test_variable_sequence_mirrors_lateral_path_and_shares_height_path() -> None:
    cases, gates = sample_two_gate_cases(
        4,
        seed=4321,
        device=torch.device("cpu"),
        hover_config=HoverConfig(),
        layout="variable",
        spacing_range=(1.0, 1.0),
        gate_count=5,
        lateral_step_range=(0.10, 0.10),
        lateral_deviation_limit=0.25,
        height_step_range=(0.05, 0.05),
        height_range=(0.95, 1.25),
    )

    assert len(gates) == 5
    for gate in gates:
        relative_lateral = gate.center[:, 1] - cases.state.position[:, 1]
        assert torch.allclose(
            relative_lateral.reshape(-1, 2).sum(dim=1),
            torch.zeros(4),
            atol=1.0e-6,
        )
        heights = gate.center[:, 2].reshape(-1, 2)
        assert torch.equal(heights[:, 0], heights[:, 1])

    first_displacement = gates[0].center - cases.state.position
    slope = first_displacement[:, 1] / first_displacement[:, 0]
    deviations = []
    for gate in gates[1:]:
        distance = gate.center[:, 0] - gates[0].center[:, 0]
        centreline = gates[0].center[:, 1] + distance * slope
        deviations.append((gate.center[:, 1] - centreline).abs())
        assert bool((gate.center[:, 2] >= 0.95).all())
        assert bool((gate.center[:, 2] <= 1.25).all())
    assert float(torch.stack(deviations).max()) <= 0.250001
    assert bool((torch.stack(deviations) > 0.0).any())
    assert bool(
        (
            torch.stack([gate.center[:, 2] for gate in gates[1:]])
            != 1.10
        ).any()
    )


def test_variable_sequence_is_seed_deterministic() -> None:
    kwargs = dict(
        pairs=2,
        seed=777,
        device=torch.device("cpu"),
        hover_config=HoverConfig(),
        layout="variable",
        spacing_range=(0.9, 1.1),
        gate_count=5,
    )
    first_cases, first_gates = sample_two_gate_cases(**kwargs)
    second_cases, second_gates = sample_two_gate_cases(**kwargs)

    assert torch.equal(first_cases.state.position, second_cases.state.position)
    for first, second in zip(first_gates, second_gates, strict=True):
        assert torch.equal(first.center, second.center)
        assert torch.equal(first.yaw, second.yaw)
