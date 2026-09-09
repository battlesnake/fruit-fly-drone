from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.search_gate_acceleration_path_es import sample_matched_cases
from scripts.train_gate_full_network_oracle import (
    action_fidelity_passed,
    action_fidelity_score,
    collect_trajectories,
    label_axis_scales,
    load_frozen_controller,
    normalized_action_loss,
    reconstruct_prefix,
    replay_window,
    select_window,
    trajectory_causality_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_student_collection_is_unaffected_by_oracle_label_generation() -> None:
    device = torch.device("cpu")
    graph = REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz"
    student, _, hover, gate, resolution = load_frozen_controller(
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
    cases = sample_matched_cases(
        8,
        seed=71,
        device=device,
        hover_config=hover,
        gate_config=gate,
    )
    common = {
        "seconds": 0.05,
        "target_onset_seconds": 0.02,
        "resolution": resolution,
        "hover_config": hover,
        "gate_config": gate,
        "teacher_drives_physics": False,
    }
    labeled = collect_trajectories(student, teacher, cases, calibration, **common)
    unlabeled = collect_trajectories(
        student,
        teacher,
        cases,
        calibration,
        oracle_labels_enabled=False,
        **common,
    )
    assert trajectory_causality_audit(labeled, unlabeled)["passed"]

    generator = torch.Generator(device="cpu").manual_seed(72)
    window = select_window(
        labeled,
        labeled,
        pairs=2,
        student_fraction=0.5,
        start=0,
        steps=5,
        generator=generator,
        device=device,
    )
    hidden = reconstruct_prefix(student, window)
    prediction = replay_window(student, window, hidden)
    scales = label_axis_scales(labeled)
    loss, details = normalized_action_loss(
        prediction,
        window.oracle_motor,
        window.valid,
        scales,
        pair_loss_weight=0.25,
    )
    assert prediction.shape == (5, 4, 4)
    assert torch.isfinite(loss)
    assert details["valid_fraction"] == 1.0


def test_action_fidelity_rejects_missing_matched_pairs() -> None:
    valid_delta = {
        "normalized_rmse": 0.1,
        "signed_slope": 1.0,
    }
    missing_delta = {
        "normalized_rmse": None,
        "signed_slope": None,
    }
    metrics = {
        horizon: {
            "valid_episode_count": 2,
            "valid_matched_pair_count": 0,
            "throttle_delta": valid_delta,
            "matched_throttle_contrast": missing_delta,
        }
        for horizon in ("0.75", "1.50", "3.00")
    }
    assert not action_fidelity_passed(metrics)
    assert action_fidelity_score(metrics) == float("inf")
