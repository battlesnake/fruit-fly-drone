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
