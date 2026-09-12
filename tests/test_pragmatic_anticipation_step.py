from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_pragmatic_anticipation_step import restored_step  # noqa: E402


def test_scaled_steps_are_projected_independently_and_always_restored():
    source = torch.tensor([0.1, 7.9, 1.0])
    parameter = torch.nn.Parameter(source.clone())
    delta = torch.tensor([-0.2, 0.2, 0.1])
    with restored_step(parameter, source, delta, 2) as info:
        assert torch.allclose(parameter, torch.tensor([0.0, 8.0, 1.2]))
        assert info["clipped_edges"] == 2
    assert torch.equal(parameter, source)
    with pytest.raises(RuntimeError, match="deliberate"):
        with restored_step(parameter, source, delta, 0.1) as info:
            assert torch.allclose(parameter, source + 0.1 * delta)
            assert info["clipped_edges"] == 0
            raise RuntimeError("deliberate")
    assert torch.equal(parameter, source)
