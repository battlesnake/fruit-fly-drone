from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_vertical_motion_derivative_ladder as ladder  # noqa: E402
import preflight_vertical_motion_commissioning as preflight  # noqa: E402
import preflight_vertical_motion_commissioning_v2 as preflight_v2  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402


def _training_specs() -> list[dict]:
    path = (
        registration.REPO_ROOT / "artifacts/vertical-motion-commissioning-manifest-v1/manifest.json"
    )
    return commissioning.load_registered_manifest(path)["stimuli"]["splits"]["training"]["specs"]


def test_preflight_protocol_is_disposable_and_training_only() -> None:
    protocol = preflight.protocol_manifest()

    assert protocol["identity_cases"] == list(preflight.IDENTITY_CASES)
    assert protocol["fixed_effective_minibatch_cases"] == list(preflight.MINIBATCH_CASES)
    assert protocol["development_or_acceptance_specs_used"] is False
    assert protocol["development_or_acceptance_pixels_rendered"] is False
    assert protocol["candidate_retained"] is False
    assert protocol["solver"]["neural_state_updates_hz"] == 1600


def test_edge_sequences_preserve_polarity_and_share_terminal() -> None:
    specs = _training_specs()
    for case, polarity in ((0, "ON"), (24, "OFF")):
        sequence = commissioning.render_sequence(specs[case])
        moving = sequence[: registration.MOTION_FRAMES, :2]
        temporal = moving[1:] - moving[:-1]
        if polarity == "ON":
            assert float(temporal.min()) >= 0.0
        else:
            assert float(temporal.max()) <= 0.0
        assert torch.equal(sequence[-1], torch.full_like(sequence[-1], 0.5))
        assert torch.equal(
            sequence[: registration.MOTION_FRAMES, 2],
            sequence[0, 0].expand(registration.MOTION_FRAMES, -1, -1),
        )
        assert torch.equal(
            sequence[: registration.MOTION_FRAMES, 3],
            sequence[0, 1].expand(registration.MOTION_FRAMES, -1, -1),
        )


def test_literal_reverse_uses_last_motion_frame_as_stationary_baseline() -> None:
    spec = _training_specs()[0]
    normal = commissioning.render_sequence(spec)
    reverse = commissioning.render_sequence(spec, reverse=True)

    assert torch.equal(
        reverse[: registration.MOTION_FRAMES, :2],
        normal[: registration.MOTION_FRAMES, :2].flip(0),
    )
    assert torch.equal(
        reverse[: registration.MOTION_FRAMES, 2],
        normal[registration.MOTION_FRAMES - 1, 0].expand(registration.MOTION_FRAMES, -1, -1),
    )
    assert torch.equal(reverse[-1], normal[-1])


def test_texture_render_is_deterministic_bounded_and_periodic() -> None:
    spec = _training_specs()[48]
    first = commissioning.render_sequence(spec)
    second = commissioning.render_sequence(spec)

    assert torch.equal(first, second)
    assert float(first.min()) >= 0.18 - 1.0e-6
    assert float(first.max()) <= 0.82 + 1.0e-6
    speed = spec["speed_pixels_per_frame"]
    assert torch.equal(first[1, 0], torch.roll(first[0, 0], speed, dims=0))


def test_fd_probes_cover_one_gain_bias_and_tau_per_pathway() -> None:
    assert preflight.PROBES == (
        ("T4_gain_Mi1_to_T4c", "gain", 0),
        ("T5_gain_Tm1_to_T5c", "gain", 8),
        ("T4c_bias", "bias_offset", 0),
        ("T5c_bias", "bias_offset", 2),
        ("T4c_tau", "tau_ratio", 0),
        ("T5c_tau", "tau_ratio", 2),
    )


def test_registered_loss_accepts_balanced_pathway_shapes() -> None:
    specs = [
        {"family": "polarity_preserving_edge", "polarity": "ON"},
        {"family": "polarity_preserving_edge", "polarity": "OFF"},
        {"family": "band_limited_texture", "polarity": "mixed"},
        {"family": "band_limited_texture", "polarity": "mixed"},
    ]
    anatomy = {
        "target_indices": np.arange(4, dtype=np.int64),
        "target_subtypes": np.arange(4, dtype=np.int64),
    }
    correct = torch.tensor([[-0.5, 0.5, -0.5, 0.5], [0.5, -0.5, 0.5, -0.5]])
    reversed_direction = -correct

    def response(values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "integrated_response": values.clone(),
            "integrated_static": torch.zeros_like(values),
            "terminal_response": values.clone(),
            "terminal_static": torch.zeros_like(values),
            "terminal_motor": torch.zeros(4, 4),
        }

    normal = [response(correct) for _ in specs]
    reverse = [response(reversed_direction) for _ in specs]
    references = commissioning.source_references(specs, normal, reverse, anatomy)
    controller = SimpleNamespace(
        gain=torch.ones(16),
        bias_offset=torch.zeros(4),
        tau_ratio=torch.ones(4),
    )

    loss, components = commissioning.commissioning_loss(
        specs, normal, reverse, references, anatomy, controller
    )

    assert torch.isfinite(loss)
    assert set(components) == {
        "direction",
        "bias",
        "stationary",
        "dsi",
        "reverse",
        "activity",
        "regularization",
    }
    assert float(components["stationary"]) == 0.0
    assert float(components["direction"]) == 0.0
    assert float(components["reverse"]) == 0.0


def test_v2_changes_only_tau_arithmetic_protocol() -> None:
    v1_protocol = preflight.protocol_manifest()
    v2_protocol = preflight_v2.protocol_manifest()

    ignored = {
        "experiment",
        "protocol_commit",
        "locked_v1_report_sha256",
        "locked_v1_classification",
        "numerical_correction",
        "scientific_or_threshold_changes_from_v1",
        "v1_frozen_source_normalizations_sha256_required",
        "v1_rendered_training_subset_sha256_required",
    }
    assert {key: value for key, value in v2_protocol.items() if key not in ignored} == {
        key: value
        for key, value in v1_protocol.items()
        if key not in {"experiment", "protocol_commit"}
    }
    assert v2_protocol["scientific_or_threshold_changes_from_v1"] == []
    assert v2_protocol["finite_difference"]["central_step"] == 1.0e-3
    assert v2_protocol["finite_difference"]["symmetric_relative_error_maximum"] == 0.02


def test_derivative_ladder_uses_adjacent_resolved_scales() -> None:
    assert ladder.STEPS == (0.008, 0.004, 0.002, 0.001, 0.0005, 0.00025)
    rows = [{"step": step, "pass": step in (0.004, 0.002)} for step in ladder.STEPS]

    qualification = ladder.qualify_probe(rows)

    assert qualification["pass"] is True
    assert qualification["passing_adjacent_step_pairs"] == [[0.004, 0.002]]


def test_resolution_floor_never_collapses_below_eight_float32_ulps() -> None:
    result = ladder.loss_resolution_floor([1.0, 1.0, 1.0])

    assert result["repeat_range"] == 0.0
    assert result["loss_resolution_floor"] == 8.0 * float(np.spacing(np.float32(1.0)))
