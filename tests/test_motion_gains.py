from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from flydrone.hover import ConnectomeController
from flydrone.motion_gains import (
    attach_motion_gains,
    compiled_controller_state,
    motion_edge_groups,
)
from scripts.train_pragmatic_course_motion_gains import (
    continuation_check,
    improvement_allowed,
    motor_residual_summary,
)


class SmallController(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4, 0.5]))
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("edge_sign", torch.tensor([1.0, -1.0, 1.0, -1.0]))


def test_motion_groups_include_both_instances_and_only_existing_type_pairs():
    types = ["Mi1", "T4a", "Mi1", "T4a", "Tm1", "T5a", "other"]
    pre, post = np.array([0, 2, 0, 4, 6]), np.array([1, 3, 6, 5, 1])
    indices, groups, counts = motion_edge_groups(types, pre, post, (("Mi1", "T4a"), ("Tm1", "T5a")))
    assert indices.tolist() == [0, 1, 3]
    assert groups.tolist() == [0, 0, 1]
    assert counts == [2, 1]
    with pytest.raises(ValueError, match="no existing edges"):
        motion_edge_groups(types, pre, post, (("Mi4", "T4b"),))


def test_only_shared_gains_train_and_export_has_ordinary_controller_schema():
    controller = SmallController()
    source = {k: v.clone() for k, v in controller.state_dict().items()}
    module = attach_motion_gains(controller, torch.tensor([0, 1, 3]), torch.tensor([0, 0, 1]))
    assert torch.equal(controller.edge_magnitude, source["edge_magnitude"])
    controller.edge_magnitude.sum().backward()
    assert torch.allclose(module.gains.grad, torch.tensor([0.5, 0.5]))
    assert [p for p in controller.parameters() if p.requires_grad] == [module.gains]
    assert controller.bias.grad is None
    assert controller.parametrizations.edge_magnitude.original.grad is None
    with torch.no_grad():
        module.gains[:] = torch.tensor([0.6, 1.7])
    state = compiled_controller_state(controller)
    assert set(state) == set(source)
    assert torch.allclose(state["edge_magnitude"], torch.tensor([0.12, 0.18, 0.4, 0.85]))
    assert torch.equal(state["edge_sign"], source["edge_sign"])
    assert torch.equal(state["bias"], source["bias"])
    plain = SmallController()
    plain.load_state_dict(state)
    assert torch.equal(plain.edge_magnitude, controller.edge_magnitude)
    with torch.no_grad():
        module.gains.fill_(1)
    assert not torch.equal(plain.edge_magnitude, controller.edge_magnitude)


def test_projection_bounds_gains_not_a_temporary_computed_weight_tensor():
    controller = SmallController()
    module = attach_motion_gains(controller, torch.tensor([0, 1]), torch.tensor([0, 1]))
    with torch.no_grad():
        module.gains[:] = torch.tensor([-1.0, 3.0])
    module.project_parameters()
    assert module.gains.tolist() == [0.5, 2.0]
    assert torch.allclose(controller.edge_magnitude[:2], torch.tensor([0.1, 0.6]))


def test_plain_export_matches_recurrent_states_and_gains_keep_gradient_after_no_grad_prefix():
    graph = Path(__file__).resolve().parents[1] / "artifacts/gate-v1/connectome.npz"
    actor = ConnectomeController(graph)
    plain = copy.deepcopy(actor)
    indices = torch.arange(min(10, len(actor.edge_pre)))
    gains = attach_motion_gains(actor, indices, indices % 2)
    with torch.no_grad():
        gains.gains[:] = torch.tensor([0.7, 1.3])
    plain.load_state_dict(compiled_controller_state(actor))
    image = torch.rand(2, 1, 16, 16, generator=torch.Generator().manual_seed(9))
    attitude = torch.tensor([[0.1, -0.1], [-0.1, 0.1]])
    neural = actor.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    other = neural.clone()
    with torch.no_grad():
        for _ in range(10):
            _, neural = actor(image, attitude, neural)
            _, other = plain(image, attitude, other)
    for _ in range(3):
        motor, neural = actor(image, attitude, neural)
        plain_motor, other = plain(image, attitude, other)
        assert torch.allclose(neural, other)
        assert torch.allclose(motor, plain_motor)
    neural.square().sum().backward()
    assert gains.gains.grad is not None and torch.isfinite(gains.gains.grad).all()
    assert bool((gains.gains.grad != 0).any())


def test_continuation_requires_improvement_on_both_sides_and_early_preservation():
    baseline = dict(late_roll_rmse_by_side=[0.02, 0.03])
    candidate = dict(late_roll_rmse_by_side=[0.019, 0.028], early_roll_rmse_by_side=[0.001, 0.002])
    assert continuation_check(baseline, candidate)
    candidate["late_roll_rmse_by_side"][0] = 0.021
    assert not continuation_check(baseline, candidate)
    candidate["late_roll_rmse_by_side"][0] = 0.019
    candidate["early_roll_rmse_by_side"][1] = 0.009
    assert not continuation_check(baseline, candidate)


def test_diagnostics_separate_actual_labels_when_early_window_crosses_gate_one():
    roles = torch.tensor([[0, 0], [0, 1], [1, 1]])
    error = torch.zeros(3, 2, 4)
    error[:, :, 0] = torch.where(roles == 0, 0.002, 0.1)
    result = motor_residual_summary([error], [roles])
    assert result["early_frames_by_side"] == [2, 1]
    assert result["late_frames_by_side"] == [1, 2]
    assert result["early_roll_rmse_by_side"] == pytest.approx([0.002, 0.002])
    assert result["late_roll_rmse_by_side"] == pytest.approx([0.1, 0.1])


def test_better_completion_cannot_bypass_first_gate_preservation_floor():
    retained = dict(
        first_gate_pass_rate=0.9,
        clean_course_success_rate=0.3,
        clean_course_negative_success_rate=0.4,
        clean_course_positive_success_rate=0.2,
        gates_before_failure_mean=3,
        course_race_fitness=3,
    )
    candidate = dict(retained, first_gate_pass_rate=0.8, clean_course_success_rate=0.4)
    assert not improvement_allowed(candidate, retained, 0.85)
    candidate["first_gate_pass_rate"] = 0.9
    assert improvement_allowed(candidate, retained, 0.85)
