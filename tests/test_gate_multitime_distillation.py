from __future__ import annotations

import copy
from pathlib import Path

import torch

from scripts.audit_gate_analytic_teachers import teacher_rc_for_mode
from scripts.train_gate_full_network_oracle import load_frozen_controller, parameter_snapshot
from scripts.train_gate_multitime_distillation import (
    collect_analytic_trajectories,
    diverse_matched_cases,
    fidelity_margin_score,
    fidelity_passed,
    fidelity_score,
    joint_multitime_loss,
    make_dataset,
    multitime_loss,
    select_pairs,
    target_scales,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_multitime_loss_replays_complete_native_prefix() -> None:
    device = torch.device("cpu")
    graph = REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz"
    source, _, hover, gate, resolution = load_frozen_controller(
        graph,
        REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
        device,
    )
    cases = diverse_matched_cases(
        8,
        seed=81,
        device=device,
        hover_config=hover,
        gate_config=gate,
        extreme_fraction=0.5,
    )
    assert torch.unique(cases.gate.center[0::2], dim=0).shape[0] == 4
    assert torch.equal(cases.mass_scale[:4], torch.tensor([0.92, 1.08, 0.92, 1.08]))

    trajectories = collect_analytic_trajectories(
        source,
        source,
        cases,
        teacher_mode="visual_accelerometer_exact_mass",
        seconds=0.05,
        resolution=resolution,
        hover_config=hover,
        gate_config=gate,
    )
    alternate_labels = collect_analytic_trajectories(
        source,
        source,
        cases,
        teacher_mode="state_feedback_nominal_mass",
        seconds=0.05,
        resolution=resolution,
        hover_config=hover,
        gate_config=gate,
    )
    for field in (
        "images",
        "roll_pitch",
        "specific_force",
        "stick_position",
        "reference_motor",
        "valid",
    ):
        assert torch.equal(getattr(trajectories, field), getattr(alternate_labels, field))
    assert not torch.equal(trajectories.oracle_motor, alternate_labels.oracle_motor)
    horizons = (3, 5)
    dataset = make_dataset(
        trajectories,
        source,
        horizon_steps=horizons,
        anchor_step=1,
        device=device,
    )
    indices = torch.tensor([step - 1 for step in horizons])
    assert torch.equal(
        dataset.target_correction,
        trajectories.oracle_motor[indices] - dataset.source_motor,
    )
    dataset.valid_at_horizons[1, 2:] = False
    dataset.source_motor[1, 2:, :3] = 1_000.0
    dataset.target_correction[:, :, 3] = 0.0
    scales = target_scales(
        dataset,
        axis_action_floor=0.01,
        throttle_action_floor=0.01,
    )
    assert torch.equal(scales.contrast_squared, torch.full_like(scales.contrast_squared, 0.01**2))
    assert torch.equal(scales.mean_squared, torch.full_like(scales.mean_squared, 0.01**2))
    expected_axis_scale = (
        (dataset.source_motor[1, :2, :3] + dataset.target_correction[1, :2, :3])
        .square()
        .mean(dim=0)
        .clamp_min(0.01**2)
    )
    assert torch.equal(scales.axis_squared[1], expected_axis_scale)
    batch = select_pairs(
        dataset,
        pairs=2,
        horizon_index=1,
        generator=torch.Generator(device="cpu").manual_seed(82),
    )
    student = copy.deepcopy(source)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    loss, details = multitime_loss(
        student,
        batch,
        scales,
        parameter_snapshot(student),
        horizon_index=1,
        horizon_steps=horizons,
        anchor_step=1,
        axis_action_weight=0.25,
        anchor_weight=0.25,
        regularization_weight=1.0e-6,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert details["valid_pair_fraction"] == 1.0
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in student.parameters()
    )

    student.zero_grad(set_to_none=True)
    joint_loss, joint_details = joint_multitime_loss(
        student,
        batch,
        scales,
        parameter_snapshot(student),
        horizon_steps=horizons,
        horizon_weights=(2.0, 1.0),
        anchor_step=1,
        axis_action_weight=3.0,
        anchor_weight=0.25,
        regularization_weight=1.0e-6,
    )
    joint_loss.backward()
    assert torch.isfinite(joint_loss)
    assert joint_details["included_horizon_weight"] == 3.0
    assert all(item["included"] for item in joint_details["endpoints"])
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in student.parameters()
    )

    assert trajectories.summary["teacher_mode"] == "visual_accelerometer_exact_mass"

    reserve = teacher_rc_for_mode(
        "visual_accelerometer_reserve",
        source,
        cases.state,
        cases.gate,
        cases.mass_scale,
        hover,
    )
    reserve_with_swapped_mass = teacher_rc_for_mode(
        "visual_accelerometer_reserve",
        source,
        cases.state,
        cases.gate,
        cases.mass_scale.flip(0),
        hover,
    )
    exact_mass_with_swapped_mass = teacher_rc_for_mode(
        "visual_accelerometer_exact_mass",
        source,
        cases.state,
        cases.gate,
        cases.mass_scale.flip(0),
        hover,
    )
    assert torch.equal(reserve, reserve_with_swapped_mass)
    assert not torch.equal(reserve, exact_mass_with_swapped_mass)


def test_fidelity_gate_rejects_empty_and_nonfinite_metrics() -> None:
    valid_metrics = {
        "0.50": {
            "valid_episode_count": 2,
            "valid_matched_pair_count": 1,
            "contrast_normalized_rmse": 0.10,
            "mean_normalized_rmse": 0.10,
            "axis_teacher_action_normalized_rmse": {
                "roll": 0.10,
                "pitch": 0.10,
                "yaw": 0.10,
            },
        }
    }
    assert fidelity_passed(valid_metrics, threshold=0.25)
    assert fidelity_score(valid_metrics) == 0.10
    assert fidelity_margin_score(valid_metrics, threshold=0.25) == 0.5

    missing = copy.deepcopy(valid_metrics)
    missing["0.50"]["valid_matched_pair_count"] = 0
    missing["0.50"]["contrast_normalized_rmse"] = None
    assert not fidelity_passed(missing, threshold=0.25)
    assert fidelity_score(missing) == float("inf")
    assert fidelity_margin_score(missing, threshold=0.25) == float("inf")

    nonfinite = copy.deepcopy(valid_metrics)
    nonfinite["0.50"]["axis_teacher_action_normalized_rmse"]["yaw"] = float("nan")
    assert not fidelity_passed(nonfinite, threshold=0.25)
    assert fidelity_score(nonfinite) == float("inf")
