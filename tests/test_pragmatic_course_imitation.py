from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_pragmatic_course_teacher as imitation  # noqa: E402


def test_path_mask_can_include_existing_attitude_without_adding_edges(monkeypatch):
    graph = dict(
        node_ids=np.arange(5),
        edge_pre=np.array([0, 1, 2, 3, 4]),
        edge_post=np.array([1, 2, 2, 2, 4]),
        visual_node_indices=np.array([0]),
        attitude_node_indices=np.array([3]),
        output_pool_indices=np.array([2]),
    )
    monkeypatch.setattr(imitation.np, "load", lambda _: nullcontext(graph))
    visual_edges, _, _ = imitation.native_sensorimotor_mask(None, 3, torch.device("cpu"))
    all_edges, nodes, manifest = imitation.native_sensorimotor_mask(
        None, 3, torch.device("cpu"), include_attitude=True
    )
    assert visual_edges.tolist() == [True, True, True, False, False]
    assert all_edges.tolist() == [True, True, True, True, False]
    assert nodes.tolist() == [True, True, True, True, False]
    assert manifest["attitude_neurons"] == 1


def test_direct_only_loss_balances_common_and_differential_motor_errors():
    target = torch.zeros(2, 4)
    active = torch.ones(2, dtype=torch.bool)
    scale = torch.ones(4)
    common = torch.ones(2, 4)
    differential = common.clone()
    differential[0] = -1.0
    for weight, expected_ratio in ((0.0, 1.0), (1.0, 5.0)):
        common_loss, _ = imitation.action_imitation_loss(common, target, active, scale, weight)
        differential_loss, _ = imitation.action_imitation_loss(
            differential, target, active, scale, weight
        )
        assert torch.allclose(differential_loss, expected_ratio * common_loss)


def test_unpaired_active_flight_still_gets_direct_imitation_loss():
    prediction = torch.ones(2, 4)
    prediction[1] = 999.0
    loss, axis = imitation.action_imitation_loss(
        prediction, torch.zeros_like(prediction), torch.tensor([True, False]),
        torch.ones(4), 1.0,
    )
    assert loss == 1.0
    assert torch.equal(axis, torch.ones(4))
