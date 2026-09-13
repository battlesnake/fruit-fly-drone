from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pragmatic_centered_premotor import (  # noqa: E402
    MeanCompensatedPremotor,
    PresynapticMoments,
    balanced_input_mean,
)
from pragmatic_premotor_bearing import source_window_bearing  # noqa: E402
from test_pragmatic_premotor_bearing import bank  # noqa: E402


class SmallNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([0.4, 0.5, 0.3]))
        self.bias = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]), requires_grad=False)
        self.register_buffer("edge_pre", torch.tensor([0, 1, 1]))
        self.register_buffer("edge_post", torch.tensor([2, 2, 1]))
        self.register_buffer("edge_sign", torch.tensor([1.0, -1.0, 1.0]))

    def initial_state(self, count, **kwargs):
        return torch.zeros(count, 3, **kwargs)

    def forward(self, image, attitude, neural):
        assert image.shape == (len(neural), 3, 1, 1) and attitude.shape == (len(neural), 2)
        target = self.bias.expand(len(neural), -1).index_add(
            1,
            self.edge_post,
            neural[:, self.edge_pre].tanh() * self.edge_sign * self.edge_magnitude,
        )
        return target[:, 2:].expand(-1, 4), target


def fixture():
    controller = SmallNative()
    wrapper = MeanCompensatedPremotor(
        controller,
        torch.tensor([True, True, False]),
        torch.tensor([0, 1]),
        torch.tensor([0.2, -0.3]),
    )
    return controller, wrapper, (torch.zeros(1, 3, 1, 1), torch.zeros(1, 2))


def test_compensation_preserves_reference_mean_drive_and_compiles_to_native_parameters():
    controller, wrapper, observations = fixture()
    bias_parameter = controller.bias
    keys = set(controller.state_dict())
    reference_bias = controller.bias.detach().clone()
    reference_neural = torch.atanh(torch.tensor([[0.2, -0.3, 0.0]]))
    old, _ = controller(*observations, reference_neural)
    with torch.no_grad():
        controller.edge_magnitude[:2].add_(torch.tensor([0.7, 0.2]))
    transformed, _ = wrapper(*observations, reference_neural)
    assert torch.allclose(old, transformed, atol=1e-6)
    assert controller.bias is bias_parameter and torch.equal(controller.bias, reference_bias)
    stats = wrapper.compile_bias()
    assert stats["changed_biases"] == 1
    assert torch.allclose(controller.bias, torch.tensor([0.1, 0.2, 0.1]), atol=1e-6)
    assert set(controller.state_dict()) == keys  # No means/wrapper added to native export.
    assert not any(k.startswith("controller.") for k in wrapper.training_state())
    native_state = reference_neural.clone()
    training_state = reference_neural.clone()
    for _ in range(6):
        motor, native_state = controller(*observations, native_state)
        functional, training_state = wrapper(*observations, training_state)
        assert torch.allclose(motor, functional, atol=1e-6)
        assert torch.allclose(native_state, training_state, atol=1e-6)


def test_gradient_includes_bias_tie_not_just_post_step_compensation():
    controller, wrapper, observations = fixture()
    neural = torch.atanh(torch.tensor([[0.6, 0.1, 0.0]]))
    output, _ = wrapper(*observations, neural)
    output.sum().backward()
    # Four equal axes: dL/dw = 4 * sign * (activity - reference mean).
    assert torch.allclose(controller.edge_magnitude.grad, torch.tensor([1.6, -1.6, 0.0]), atol=1e-6)
    assert controller.bias.grad is None
    controller.edge_magnitude.grad = None
    output, _ = controller(*observations, neural)
    output.sum().backward()
    assert torch.allclose(controller.edge_magnitude.grad, torch.tensor([2.4, -0.4, 0.0]), atol=1e-6)


def test_input_statistics_exclude_holdout_invisible_and_inactive_and_balance_strata():
    moments = PresynapticMoments(torch.tensor([0]))
    neural = torch.atanh(torch.tensor([[0.2], [0.4], [0.9], [-0.9]]))
    current = torch.zeros(4, dtype=torch.long)
    eligible = torch.ones(4, dtype=torch.bool)
    moments.observe(neural, current, eligible, 0)
    moments.observe(neural, current, torch.tensor([False, True, True, True]), 1)
    moments.observe(neural, current, eligible, 50)
    moments.observe(neural, current + 1, eligible, 60)
    assert moments.counts.tolist() == [1, 2, 1, 1, 1, 1]
    assert torch.allclose(moments.sums[:, 0], torch.tensor([0.2, 0.8, 0.2, 0.4, 0.2, 0.4]))
    native = SimpleNamespace(replay=SimpleNamespace(kind="native"), input_moments=moments.archive())
    assisted = SimpleNamespace(
        replay=SimpleNamespace(kind="roll-assisted"),
        input_moments=dict(
            nodes=torch.tensor([0]),
            sums=torch.tensor([[0.9], [0.0], [0.0], [0.0], [0.0], [0.0]]),
            counts=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    )
    mean, report = balanced_input_mean([native, assisted], torch.tensor([0]))
    assert mean.item() == pytest.approx((1.8 + 0.9) / 7)
    assert report["contributing_strata"] == 7 and report["heldout_episodes_excluded"]
    assert report["coverage"]["roll-assisted/later-gates/1"] == 0


def test_source_window_comparison_uses_original_activity_and_same_mask_not_motor_targets():
    from pragmatic_premotor_bearing import fit_bearing_head

    data = bank()
    head, _ = fit_bearing_head([data], device=torch.device("cpu"), seed=1)
    original = source_window_bearing(head, data, [4, 5], [5, 8], unroll=4)
    assert len(original["original_source_normalized_bearing_mse_by_branch"]) == 2
    data.replay.target.fill_(1e6)
    repeated = source_window_bearing(head, data, [4, 5], [5, 8], unroll=4)
    assert original == repeated
    data.visible[6, 4] = False
    masked = source_window_bearing(head, data, [4, 5], [5, 8], unroll=4)
    data.labels[6, 4] = 1e5
    assert source_window_bearing(head, data, [4, 5], [5, 8], unroll=4) == masked


def test_compensation_does_not_clip_existing_bias_or_optimize_it_independently():
    controller, wrapper, _ = fixture()
    with torch.no_grad():
        controller.edge_magnitude[0] = 100.0
    wrapper.compile_bias()
    assert controller.bias[2] < -5  # Affine tie is not silently clipped.
    controller.bias.requires_grad_(True)
    with pytest.raises(ValueError, match="independently"):
        MeanCompensatedPremotor(
            controller,
            torch.tensor([True, True, False]),
            torch.tensor([0, 1]),
            torch.tensor([0.2, -0.3]),
        )
