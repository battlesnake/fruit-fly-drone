from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_frozen_optic_motion as audit  # noqa: E402


def test_protocol_separates_camera_and_neural_rates() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["protocol_commit"] == "753a7cd"
    assert protocol["rates"]["camera_hz"] == 50
    assert protocol["rates"]["primary_neural_step_hz"] == 1600
    assert protocol["anatomy"]["cells"] == 13_580
    assert protocol["qualification"]["fitted_decoder"] is False
    assert protocol["authorizes_hover_gate_or_promotion"] is False


def test_stimulus_factorial_and_controls_are_exact() -> None:
    specs = audit.stimulus_specs()

    assert len(specs) == 80
    assert len(audit.numerical_specs(specs)) == 8
    assert len(audit.reverse_specs(specs)) == 16
    sequence = audit.render_sequence(specs[0])
    assert sequence.shape == (17, 4, 200, 320)
    assert torch.equal(sequence[-1], sequence[-1, :1].expand_as(sequence[-1]))
    assert torch.equal(sequence, audit.render_sequence(specs[0]))
    assert not torch.equal(sequence[4, 0], sequence[4, 1])
    assert specs[0]["angular_motion"]["mean_degrees_per_frame_by_branch"][0] > 0
    assert specs[0]["angular_motion"]["mean_degrees_per_frame_by_branch"][1] < 0


def test_edge_polarity_and_motion_direction_are_declared() -> None:
    specs = audit.stimulus_specs()
    bright = next(
        item
        for item in specs
        if item["family"] == "edge"
        and item["polarity"] == "bright"
        and item["axis"] == "vertical"
    )
    dark = {**bright, "polarity": "dark"}

    for branch in (0, 1):
        bright_sequence = torch.stack(
            [audit.motion_frame(bright, branch, frame) for frame in range(16)]
        )
        dark_sequence = torch.stack(
            [audit.motion_frame(dark, branch, frame) for frame in range(16)]
        )
        assert bool(((bright_sequence[1:] - bright_sequence[:-1]) >= 0).all())
        assert bool(((dark_sequence[1:] - dark_sequence[:-1]) <= 0).all())


def test_reverse_history_uses_its_actual_first_frame_for_stationary_control() -> None:
    spec = audit.stimulus_specs()[0]
    normal = audit.render_sequence(spec)
    reversed_history = audit.render_sequence(spec, mode="reverse")

    assert torch.equal(reversed_history[:16, :2], normal[:16, :2].flip(0))
    assert torch.equal(
        reversed_history[0, :2], reversed_history[0, 2:]
    )
    assert torch.equal(reversed_history[-1], normal[-1])


def test_numerical_comparison_applies_all_three_limits() -> None:
    base = {
        "integrated_selected_stationary_subtracted": torch.zeros(1, 2, 8),
        "terminal_selected_activity": torch.zeros(1, 4, 8),
        "integrated_motion_type_means": torch.zeros(1, 2, 8),
        "integrated_static_type_means": torch.zeros(1, 2, 8),
        "terminal_motion_type_means": torch.zeros(1, 2, 8),
        "terminal_static_type_means": torch.zeros(1, 2, 8),
        "terminal_motor_outputs": torch.zeros(1, 4, 4),
        "terminal_visual_target_type_means": {
            "LPTC": torch.zeros(1, 4)
        },
        "all_states_and_outputs_finite": True,
        "source_loaded_exactly": True,
        "source_restored": True,
        "opposite_terminal_pixel_difference_maximum": 0.0,
    }
    spec = [
        {
            "polarity": "bright",
            "axis": "vertical",
        }
    ]
    assert audit.numerical_comparison(base, base, spec)["pass"] is True

    changed = {**base, "terminal_selected_activity": torch.full((1, 4, 8), 0.006)}
    assert audit.numerical_comparison(changed, base, spec)["pass"] is False


def test_literal_reversal_swaps_on_and_off_measurement_pathways() -> None:
    evaluation = {
        "integrated_motion_type_means": torch.zeros(1, 2, 8),
        "integrated_static_type_means": torch.zeros(1, 2, 8),
    }
    evaluation["integrated_motion_type_means"][:, :, audit.type_index("T4c")] = 1.0
    evaluation["integrated_motion_type_means"][:, :, audit.type_index("T5c")] = 2.0
    spec = [{"polarity": "bright", "axis": "vertical"}]

    normal = audit.opponent_values(evaluation, spec)
    reversed_measurement = audit.opponent_values(
        evaluation, spec, reverse_polarity=True
    )
    assert torch.equal(normal, torch.ones(1, 2))
    assert torch.equal(reversed_measurement, torch.full((1, 2), 2.0))


def test_duplicate_control_requires_bit_identity() -> None:
    first = {
        "integrated_selected_stationary_subtracted": torch.zeros(1, 2, 3),
        "terminal_selected_activity": torch.zeros(1, 4, 3),
        "integrated_motion_type_means": torch.zeros(1, 2, 8),
        "integrated_static_type_means": torch.zeros(1, 2, 8),
        "terminal_motion_type_means": torch.zeros(1, 2, 8),
        "terminal_static_type_means": torch.zeros(1, 2, 8),
        "terminal_motor_outputs": torch.zeros(1, 4, 4),
        "terminal_visual_target_type_means": {"LPTC": torch.zeros(1, 4)},
        "source_loaded_exactly": True,
        "source_restored": True,
    }
    assert audit.duplicate_control(first, first)["pass"] is True
    second = {**first, "terminal_motor_outputs": torch.full((1, 4, 4), 1e-7)}
    assert audit.duplicate_control(first, second)["pass"] is False


def test_maximum_thresholds_are_strictly_enforced() -> None:
    assert audit.SIGN_FRACTION_MINIMUM == pytest.approx(0.9)
    assert audit.MEDIAN_DSI_MINIMUM == pytest.approx(0.3)
    assert audit.STATIONARY_RATIO_MAXIMUM == pytest.approx(0.1)
