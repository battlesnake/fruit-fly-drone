from __future__ import annotations

import copy
from pathlib import Path

import torch

from scripts.train_gate_dense_dagger import (
    DenseTrajectories,
    choose_dense_batch,
    collect_dense_trajectories,
    controller_replay_parity,
    dense_window_loss,
    diverse_matched_cases,
    fixed_axis_scales,
    validation_safe_and_improved,
)
from scripts.train_gate_full_network_oracle import load_frozen_controller, parameter_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dense_expert_labels_executed_actions_and_window_rebuilds_state() -> None:
    device = torch.device("cpu")
    source, _, hover, gate, resolution = load_frozen_controller(
        REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
        REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
        device,
    )
    cases = diverse_matched_cases(
        8,
        seed=91,
        device=device,
        hover_config=hover,
        gate_config=gate,
        extreme_fraction=0.5,
    )
    trajectories = collect_dense_trajectories(
        source,
        source,
        cases,
        teacher_mode="visual_accelerometer_reserve",
        teacher_driven=True,
        steps=5,
        takeover_step=2,
        resolution=resolution,
        hover_config=hover,
        gate_config=gate,
    )
    assert torch.equal(trajectories.targets, trajectories.executed_motor)
    assert trajectories.teacher_driven
    assert trajectories.behavior_parameter_sha256
    parity = controller_replay_parity(source, trajectories, trajectories.targets, steps=2)
    assert parity["exact"]
    assert parity["within_tolerance"]
    assert parity["maximum_absolute_error"] == 0.0
    assert parity["stored_image_dtype"] == "torch.float32"
    scales = fixed_axis_scales(trajectories, floor=0.01)
    assert scales.shape == (4,)
    assert bool((scales >= 0.01).all())

    student = copy.deepcopy(source)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    loss, details = dense_window_loss(
        student,
        trajectories,
        parameter_snapshot(source),
        scales,
        start=1,
        window_steps=3,
        episodes=torch.arange(8),
        regularization_weight=1.0e-6,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert details["window_start"] == 1
    assert details["window_end"] == 4
    assert details["valid_fraction"] == 1.0
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in student.parameters()
    )


def test_midpoint_gate_requires_gain_without_mass_regression() -> None:
    baseline = {"success_rate": 0.20, "light_success_rate": 0.10, "heavy_success_rate": 0.30}
    passing = {"success_rate": 0.26, "light_success_rate": 0.08, "heavy_success_rate": 0.32}
    regressed = {"success_rate": 0.30, "light_success_rate": 0.02, "heavy_success_rate": 0.58}
    assert validation_safe_and_improved(baseline, passing, improvement=0.05, mass_drop=0.05)
    assert not validation_safe_and_improved(baseline, regressed, improvement=0.05, mass_drop=0.05)


def test_dense_batch_selection_checks_selected_pairs_and_falls_back() -> None:
    steps = 110
    episodes = 8
    valid = torch.zeros(steps, episodes, dtype=torch.bool)
    valid[:20] = True
    trajectories = DenseTrajectories(
        images=torch.empty(steps, episodes, 1, 1),
        roll_pitch=torch.empty(steps, episodes, 2),
        specific_force=torch.empty(steps, episodes, 3),
        targets=torch.empty(steps, episodes, 4),
        executed_motor=torch.empty(steps, episodes, 4),
        valid=valid,
        mass_scale=torch.tensor([0.8, 1.2] * (episodes // 2)),
        codes=torch.arange(episodes),
        teacher_driven=False,
        behavior_parameter_sha256="test",
        summary={},
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    start, selected, effective_region = choose_dense_batch(
        trajectories,
        window_steps=10,
        region="crossing_post",
        pairs=2,
        generator=generator,
        attempts_per_region=2,
    )
    assert effective_region == "launch"
    assert 0 <= start <= 19
    assert selected.shape == (4,)
    assert bool(trajectories.valid[start : start + 10, selected].float().mean() >= 0.10)
    assert all(
        abs(int(selected[index]) - int(selected[index + 1])) == 1
        for index in range(0, len(selected), 2)
    )
