from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from scripts.train_gate_full_network_oracle import (
    collect_trajectories,
    load_frozen_controller,
    parameter_snapshot,
)
from scripts.train_gate_multitime_distillation import (
    diverse_matched_cases,
    fidelity_passed,
    fidelity_score,
    hybrid_teacher_takeover_audit,
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
    teacher, _, _, _, _ = load_frozen_controller(
        graph,
        REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
        device,
    )
    calibration = json.loads(
        (REPO_ROOT / "artifacts" / "gate-mass-oracle-v1" / "candidate.json").read_text()
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

    trajectories = collect_trajectories(
        source,
        teacher,
        cases,
        calibration,
        seconds=0.05,
        target_onset_seconds=0.02,
        resolution=resolution,
        hover_config=hover,
        gate_config=gate,
        teacher_drives_physics=False,
    )
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
    scales = target_scales(dataset, action_floor=0.01)
    expected_axis_scale = (
        dataset.source_motor[1, :2, :3].std(dim=0, unbiased=False).clamp_min(0.01).square()
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
        axis_preservation_weight=0.25,
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

    takeover = hybrid_teacher_takeover_audit(
        student,
        source,
        teacher,
        cases,
        calibration,
        takeover_seconds=0.0,
        seconds=0.05,
        target_onset_seconds=0.02,
        resolution=resolution,
        hover_config=hover,
        gate_config=gate,
    )
    assert takeover["target_policy"] == ("promoted_source_roll_pitch_yaw_plus_oracle_throttle")


def test_fidelity_gate_rejects_empty_and_nonfinite_metrics() -> None:
    valid_metrics = {
        "0.50": {
            "valid_episode_count": 2,
            "valid_matched_pair_count": 1,
            "contrast_normalized_rmse": 0.10,
            "mean_normalized_rmse": 0.10,
            "axis_preservation_normalized_rmse": {
                "roll": 0.10,
                "pitch": 0.10,
                "yaw": 0.10,
            },
        }
    }
    assert fidelity_passed(valid_metrics, threshold=0.25)
    assert fidelity_score(valid_metrics) == 0.10

    missing = copy.deepcopy(valid_metrics)
    missing["0.50"]["valid_matched_pair_count"] = 0
    missing["0.50"]["contrast_normalized_rmse"] = None
    assert not fidelity_passed(missing, threshold=0.25)
    assert fidelity_score(missing) == float("inf")

    nonfinite = copy.deepcopy(valid_metrics)
    nonfinite["0.50"]["axis_preservation_normalized_rmse"]["yaw"] = float("nan")
    assert not fidelity_passed(nonfinite, threshold=0.25)
    assert fidelity_score(nonfinite) == float("inf")
