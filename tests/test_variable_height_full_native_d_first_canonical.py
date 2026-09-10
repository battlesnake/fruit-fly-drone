from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402


class TinyController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = nn.Parameter(torch.tensor([0.5, 8.0], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.raw_time_constant = nn.Parameter(torch.tensor([-3.0], dtype=torch.float32))

    def project_parameters(self) -> None:
        with torch.no_grad():
            self.edge_magnitude.clamp_(0.0, 8.0)


def _parameters(controller: TinyController) -> dict[str, torch.Tensor]:
    return {
        name: getattr(controller, name).detach().clone()
        for name in ("edge_magnitude", "bias", "raw_time_constant")
    }


def test_authoritative_materialization_is_directly_idempotent() -> None:
    controller = TinyController()
    source = _parameters(controller)
    proposal = {
        "edge_magnitude": torch.tensor([-1.0, 0.25]),
        "bias": torch.tensor([1.0e-7]),
        "raw_time_constant": torch.tensor([1.0e-7]),
    }

    authoritative, effective, report = canonical.materialize_authoritative_candidate(
        controller, source, proposal
    )

    assert report["pass"] is True
    assert report["second_direct_projection_maximum_parameter_change"] == 0.0
    assert authoritative["edge_magnitude"] == pytest.approx(torch.tensor([0.0, 8.0]))
    assert effective["edge_magnitude"] == pytest.approx(torch.tensor([-0.5, 0.0]).double())
    assert report["maximum_rounding_coordinate"]["local_float32_ulp"] > 0.0


def test_installation_copies_full_candidate_and_interpolates_once() -> None:
    controller = TinyController()
    source = _parameters(controller)
    canonical._AUTHORITATIVE_BASE = source
    canonical._AUTHORITATIVE_CANDIDATE = {
        "edge_magnitude": torch.tensor([0.25, 7.5]),
        "bias": torch.tensor([1.5]),
        "raw_time_constant": torch.tensor([-2.5]),
    }
    unused = {name: torch.zeros_like(value) for name, value in source.items()}

    canonical.install_trial(controller, source, unused, scale=1.0)
    assert all(
        torch.equal(getattr(controller, name).detach(), value)
        for name, value in canonical._AUTHORITATIVE_CANDIDATE.items()
    )

    canonical.install_trial(controller, source, unused, scale=0.5)
    assert controller.edge_magnitude.detach() == pytest.approx(torch.tensor([0.375, 7.75]))
    assert controller.bias.detach() == pytest.approx(torch.tensor([1.25]))


def test_protocol_keeps_tolerance_and_forbids_iterative_canonicalization() -> None:
    protocol = canonical.protocol_manifest()
    projection = protocol["constraint_projection"]

    assert projection["second_direct_projection_parameter_change_maximum"] == pytest.approx(1e-7)
    assert projection["canonicalization_repeated_until_pass"] is False
    assert projection["effective_displacement_arithmetic"] == (
        "float64 subtraction and dot products"
    )
    assert protocol["qualification"]["total_scenes"] == 64
