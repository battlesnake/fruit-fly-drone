from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pragmatic_box_natural as box  # noqa: E402
from pragmatic_policy_optimization import scaled_parameters  # noqa: E402
from pragmatic_sink_policy import replay_sink_policy_gradient  # noqa: E402
from test_pragmatic_policy_optimization import ParameterActor  # noqa: E402
from test_pragmatic_sink_policy import fixture  # noqa: E402


def enumerated_quadratic(hessian, linear, lower, upper):
    """Independent tiny reference: enumerate all faces and solve on each face."""
    candidates = []
    for status in itertools.product((-1, 0, 1), repeat=len(linear)):
        status = torch.tensor(status)
        free = status == 0
        value = torch.where(status < 0, lower, torch.where(status > 0, upper, 0.))
        if bool(free.any()):
            value[free] = torch.linalg.solve(
                hessian[free][:, free],
                -linear[free]-hessian[free][:, ~free] @ value[~free],
            )
        if bool(((value >= lower-1e-6) & (value <= upper+1e-6)).all()):
            candidates.append((float(.5*value@hessian@value + linear@value), value))
    return min(candidates, key=lambda x: x[0])


def test_box_quadratic_interior_matches_unconstrained_solve():
    hessian = torch.tensor([[2., .5], [.5, 1.5]])
    linear = torch.tensor([.1, -.2])
    step, stats = box.box_quadratic_step(hessian, linear, torch.full((2,), -8.),
                                        torch.full((2,), 8.))
    assert torch.allclose(step, -torch.linalg.solve(hessian, linear), atol=1e-7)
    assert stats["converged"] and stats["active_bounds"] == 0
    assert not step.requires_grad and step.dtype == torch.float32


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_box_quadratic_matches_independently_enumerated_faces(seed):
    torch.manual_seed(seed)
    matrix = torch.randn(3, 3)
    hessian = matrix @ matrix.T + .5*torch.eye(3)
    linear = 2*torch.randn(3)
    lower, upper = -torch.rand(3), torch.rand(3)
    expected_value, expected_step = enumerated_quadratic(hessian, linear, lower, upper)
    step, stats = box.box_quadratic_step(hessian, linear, lower, upper)
    assert bool(((step >= lower) & (step <= upper)).all())
    assert stats["model_change"] == pytest.approx(expected_value, abs=2e-6)
    assert torch.allclose(step, expected_step, atol=2e-6)


def test_box_quadratic_releases_a_bound_when_its_multiplier_changes_sign():
    hessian = torch.tensor([[2., -1.], [-1., 2.]])
    linear = torch.tensor([.2, -1.])
    step, stats = box.box_quadratic_step(hessian, linear, torch.zeros(2), torch.full((2,), 8.))
    assert torch.allclose(step, torch.tensor([.2, .6]), atol=1e-6)
    assert stats["released_bounds"] == 2 and stats["converged"]


def test_box_quadratic_finds_descent_when_clipping_unconstrained_direction_destroys_it():
    hessian = torch.tensor([[2., 3.], [3., 5.]])
    linear = torch.ones(2)
    lower, upper = torch.tensor([0., -1.]), torch.tensor([8., 7.])
    clipped = (-torch.linalg.solve(hessian, linear)).clamp(lower, upper)
    assert float(linear @ clipped) > 0
    step, stats = box.box_quadratic_step(hessian, linear, lower, upper)
    assert torch.allclose(step, torch.tensor([0., -.2]), atol=1e-6)
    assert float(linear @ step) < 0 and stats["model_change"] < 0


def test_box_quadratic_iteration_cap_returns_feasible_approximate_progress():
    hessian = torch.tensor([[2., -1.], [-1., 2.]])
    linear = torch.tensor([.2, -1.])
    step, stats = box.box_quadratic_step(hessian, linear, torch.zeros(2), torch.full((2,), 8.),
                                        max_iterations=2)
    assert stats["iterations"] == 2 and stats["termination"] == "iteration-cap"
    assert not stats["converged"] and stats["model_change"] < 0
    assert bool((step >= 0).all()) and float(linear @ step) < 0


@pytest.mark.parametrize("fault", ["nan", "outside", "singular"])
def test_box_quadratic_rejects_nonfinite_invalid_or_failed_solve(fault):
    hessian = torch.eye(2)
    lower, upper = -torch.ones(2), torch.ones(2)
    if fault == "nan":
        hessian[0, 0] = float("nan")
    elif fault == "outside":
        lower[0] = 1.
    else:
        hessian.zero_()
    with pytest.raises((ValueError, FloatingPointError, RuntimeError)):
        box.box_quadratic_step(hessian, torch.ones(2), lower, upper)


@pytest.mark.parametrize("outcome", ["accept", "retry_loss", "retry_trust", "reject", "error"])
def test_box_natural_backtracking_masks_and_restores_from_same_base(monkeypatch, outcome):
    actor = ParameterActor()
    base = actor.edge_magnitude.detach().clone()
    mask = torch.tensor([True, False, True])
    data = SimpleNamespace(stationary_std=[.01]*4)
    fisher = torch.eye(3)
    monkeypatch.setattr(box, "conditional_fisher", lambda *a: fisher)
    trials = []

    def replay(backward):
        loss = actor.edge_magnitude.square().sum()
        if backward:
            loss.backward()
        else:
            assert actor.edge_magnitude[1] == base[1]
            trials.append(actor.edge_magnitude.detach().clone()-base)
            if outcome == "error":
                raise FloatingPointError("deliberate box replay error")
        fail_loss = not backward and (outcome == "reject" or
                                      (outcome == "retry_loss" and len(trials) == 1))
        fail_trust = not backward and outcome == "retry_trust" and len(trials) == 1
        return dict(loss=float(base.square().sum())+.1 if fail_loss else float(loss.detach()),
                    joint_kl_mean=.02 if fail_trust else .001, joint_kl_p99=.002,
                    native_latent_mean_displacement_rms=[.001]*4)

    if outcome == "error":
        with pytest.raises(FloatingPointError, match="deliberate"):
            box.box_natural_actor_proposal(actor, data, mask, replay)
    else:
        stats = box.box_natural_actor_proposal(actor, data, mask, replay)
        assert stats["accepted"] is (outcome != "reject")
        assert len(trials) == (8 if outcome == "reject" else 1 if outcome == "accept" else 2)
        for trial, attempt in zip(trials, stats["attempts"], strict=True):
            assert attempt["predicted_loss_change"] == pytest.approx(float(2*base @ trial))
            assert attempt["predicted_local_joint_kl"] == pytest.approx(
                float(.5*trial@fisher@trial))
            assert attempt["predicted_local_joint_kl"] <= .002001
        for index, trial in enumerate(trials):
            proposed, _ = scaled_parameters(base, trials[0], .5**index, mask)
            assert torch.allclose(trial, proposed-base, atol=2e-7)
    assert actor.edge_magnitude.grad is None
    if outcome in ("reject", "error"):
        assert torch.equal(actor.edge_magnitude, base)


def test_box_natural_no_feasible_descent_is_an_ordinary_no_step(monkeypatch):
    actor = ParameterActor()
    with torch.no_grad():
        actor.edge_magnitude.zero_()
    monkeypatch.setattr(box, "conditional_fisher", lambda *a: torch.eye(3))

    def replay(backward):
        assert backward
        actor.edge_magnitude.sum().backward()
        return dict(loss=0.)

    stats = box.box_natural_actor_proposal(actor, SimpleNamespace(stationary_std=[.01]*4),
                                          torch.ones(3, dtype=torch.bool), replay)
    assert not stats["accepted"] and stats["stop_round"]
    assert stats["no_step_reason"] == "no useful feasible descent"
    assert actor.edge_magnitude.grad is None and bool((actor.edge_magnitude == 0).all())


def test_box_natural_integrates_with_actual_native_sink_recurrence():
    _, model, data, _, advantages, _ = fixture()
    original = model.edge_magnitude.detach().clone()
    stats = box.box_natural_actor_proposal(
        model, data, torch.ones_like(model.edge_magnitude, dtype=torch.bool),
        lambda backward: replay_sink_policy_gradient(model, data, advantages, backward=backward),
    )
    assert stats["accepted"] and stats["surrogate_descent"]
    assert stats["predicted_local_joint_kl"] <= .002001
    assert not torch.equal(model.edge_magnitude, original)
