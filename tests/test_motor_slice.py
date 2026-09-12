from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flydrone.motor_slice import SinkMotorSlice
from scripts.blend_pragmatic_roll_checkpoints import blend_roll_states
from scripts.train_pragmatic_roll_motor_slice import pair_split_weights


def small_controller():
    return SimpleNamespace(
        pool_offsets=torch.tensor([0, 1, 2]),
        pool_indices=torch.tensor([2, 3]),
        edge_pre=torch.tensor([0, 1, 0, 1]),
        edge_post=torch.tensor([2, 2, 3, 3]),
        edge_sign=torch.tensor([1.0, -1.0, 1.0, -1.0]),
        edge_magnitude=torch.tensor([0.2, 0.1, 0.1, 0.2]),
        bias=torch.tensor([0.0, 0.0, 0.03, -0.02]),
        time_constant=torch.tensor([0.02, 0.02, 0.021, 0.028]),
        neural_dt=0.02,
    )


def test_causal_slice_matches_recurrent_motor_states_from_zero():
    controller = small_controller()
    model = SinkMotorSlice(controller)
    features = torch.randn(120, 3, 2, generator=torch.Generator().manual_seed(4))
    state = torch.zeros(3, 2)
    expected = []
    matrix = torch.tensor([[0.2, -0.1], [0.1, -0.2]])
    for frame in features:
        target = 5 * torch.tanh((frame @ matrix.T + controller.bias[2:]) / 5)
        state = model.decay * state + (1 - model.decay) * target
        expected.append(torch.sigmoid(state[:, 0]) - torch.sigmoid(state[:, 1]))
    actual = model(features)
    assert torch.allclose(actual, torch.stack(expected), atol=2e-7)
    actual.square().mean().backward()
    assert torch.isfinite(model.magnitudes.grad).all()


def test_compile_changes_only_selected_existing_synapses():
    controller = small_controller()
    source = controller.edge_magnitude.clone()
    model = SinkMotorSlice(controller, train_edge_mask=torch.tensor([True, False, True, False]))
    with torch.no_grad():
        model.magnitudes[:] = torch.tensor([-2.0, 10.0])
    model.project_parameters()
    model.compile_into(controller)
    assert torch.equal(controller.edge_magnitude[[1, 3]], source[[1, 3]])
    assert controller.edge_magnitude[[0, 2]].tolist() == [0.0, 8.0]
    assert controller.edge_sign.tolist() == [1.0, -1.0, 1.0, -1.0]


def test_slice_refuses_to_ignore_outgoing_motor_feedback():
    controller = small_controller()
    controller.edge_pre[0] = 2
    with pytest.raises(ValueError, match="outgoing"):
        SinkMotorSlice(controller)


def test_fitting_split_keeps_whole_mirrored_pairs_and_excludes_warmup():
    bank = dict(
        current=torch.arange(5)[:, None].expand(5, 8).repeat_interleave(2, dim=0),
        active=torch.ones(10, 8, dtype=torch.bool),
    )
    bank["active"][0] = False
    training = pair_split_weights(bank, 1, validation=False)
    validation = pair_split_weights(bank, 1, validation=True)
    assert training[:, 6:].sum() == 0
    assert validation[:, :6].sum() == 0
    assert training[0].sum() == validation[0].sum() == 0
    assert training.sum() == pytest.approx(1.0)
    assert validation.sum() == pytest.approx(1.0)
    assert training[bank["current"] == 0].sum() == pytest.approx(0.5)
    assert torch.equal(training[:, 0], training[:, 1])


def test_static_blend_cannot_change_nonroll_controller_parameters():
    source = dict(
        pool_offsets=torch.tensor([0, 1, 2]),
        pool_indices=torch.tensor([2, 3]),
        edge_post=torch.tensor([2, 3, 4]),
        edge_magnitude=torch.tensor([1.0, 2.0, 3.0]),
        bias=torch.zeros(5),
    )
    target = {k: v.clone() for k, v in source.items()}
    target["edge_magnitude"][:2] += 2
    blended = blend_roll_states(source, target, 0.25)
    assert torch.equal(blended["edge_magnitude"], torch.tensor([1.5, 2.5, 3.0]))
    target["bias"][0] = 1
    with pytest.raises(ValueError, match="bias"):
        blend_roll_states(source, target, 0.25)
