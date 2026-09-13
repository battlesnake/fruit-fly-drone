from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pragmatic_policy_optimization import (  # noqa: E402
    directional_parameters,
    projected_actor_proposal,
)
from pragmatic_sink_fisher import (  # noqa: E402
    conditional_fisher,
    damped_direction,
    natural_sink_actor_proposal,
    roll_mean_jacobian,
)
from pragmatic_sink_policy import replay_sink_policy_gradient  # noqa: E402
from test_pragmatic_policy_optimization import ParameterActor  # noqa: E402
from test_pragmatic_sink_policy import fixture  # noqa: E402


@pytest.mark.parametrize("rho", [0., .8, .967])
def test_analytic_jacobian_and_fisher_match_independent_full_brain_autograd(rho):
    brain, model, data, observations, _, _ = fixture()
    data.rho = rho
    with torch.no_grad():
        model.edge_magnitude.add_(torch.tensor([.01, -.001, .002, .02]))
    model.compile_into(brain)
    _, full = brain.rollout(observations)
    physical_roll = full[data.warmup_steps:, :, 0]
    reference = torch.stack([
        torch.autograd.grad(value, brain.edge_magnitude, retain_graph=True)[0][:4]
        for value in physical_roll.flatten()
    ]).reshape(*data.valid.shape, 4)
    analytic = roll_mean_jacobian(model, data)
    assert not analytic.requires_grad and analytic.dtype == torch.float32
    assert torch.allclose(analytic, reference, atol=2e-7, rtol=2e-5)
    # Build the conditional reference separately; first action has no AR parent,
    # despite three prior warmup brain ticks. Invalid padding is not Fisher data.
    rows = []
    for time in range(len(physical_roll)):
        derivative = reference[time]
        sigma = data.stationary_std[0]
        if time:
            derivative = derivative - rho * reference[time-1]
            sigma *= (1-rho*rho)**.5
        rows.extend(derivative[data.valid[time]] / sigma)
    expected = sum(torch.outer(row, row) for row in rows) / len(rows)
    fisher = conditional_fisher(model, data)
    assert torch.allclose(fisher, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(fisher, fisher.T)


def test_fisher_is_recomputed_after_native_weight_change():
    _, model, data, _, _, _ = fixture()
    before = conditional_fisher(model, data)
    with torch.no_grad():
        model.edge_magnitude.add_(1.)
    after = conditional_fisher(model, data)
    assert not torch.allclose(before, after, atol=1e-4)


@pytest.mark.parametrize("fault", ["noise", "rho", "valid", "sigma"])
def test_fisher_rejects_undefined_conditionals(fault):
    _, model, data, _, _, _ = fixture()
    if fault == "noise":
        data.stationary_std = None
    elif fault == "rho":
        data.rho = 1.
    elif fault == "valid":
        data.valid.zero_()
    else:
        data.stationary_std[0] = 0.
    with pytest.raises(ValueError):
        conditional_fisher(model, data)


def test_damped_solve_and_projected_sizing_use_raw_gradient_contraction():
    fisher = torch.tensor([[100., 1.], [1., .1]])
    gradient = torch.tensor([1., .3])
    direction, stats = damped_direction(fisher, gradient)
    damping = .01 * float(fisher.diag().mean())
    assert stats["damping"] == pytest.approx(damping)
    assert torch.allclose((fisher + damping*torch.eye(2)) @ -direction, gradient, atol=1e-6)
    base = torch.tensor([.001, .2])
    proposed, sizing = directional_parameters(base, gradient, direction,
                                               torch.ones(2, dtype=torch.bool), .05)
    assert bool((proposed >= 0).all() & (proposed <= 8).all())
    raw_contraction = float(gradient @ (proposed-base))
    assert sizing["predicted_loss_change"] == pytest.approx(raw_contraction)
    assert raw_contraction == pytest.approx(-.05, rel=.1)
    assert float(direction @ (proposed-base)) != pytest.approx(raw_contraction)


def test_projected_direction_is_not_assumed_monotone_and_may_have_no_feasible_descent():
    # The only descent component immediately clips at zero. Remaining motion
    # increases the raw objective even though the unprojected direction descends.
    base = torch.tensor([0., 1.])
    gradient = torch.tensor([1., 1.])
    direction = torch.tensor([-2., 1.])
    with pytest.raises(ValueError, match="could not resolve"):
        directional_parameters(base, gradient, direction, torch.ones(2, dtype=torch.bool), .001)


@pytest.mark.parametrize("fault", ["zero", "nan", "damping"])
def test_bad_fisher_solve_fails_closed(fault):
    fisher = torch.eye(2)
    damping = .01
    if fault == "zero":
        fisher.zero_()
    elif fault == "nan":
        fisher[0, 0] = float("nan")
    else:
        damping = -1.
    with pytest.raises(ValueError):
        damped_direction(fisher, torch.ones(2), damping_fraction=damping)


@pytest.mark.parametrize("outcome", ["accept", "retry", "reject", "exception"])
def test_conditioned_update_keeps_trust_descent_and_transactional_restore(outcome):
    actor = ParameterActor()
    base = actor.edge_magnitude.detach().clone()
    mask = torch.tensor([True, False, True])
    fisher = torch.diag(torch.tensor([10., 3., .1]))
    conditions, trials = [], []

    def condition(gradient):
        conditions.append(gradient.clone())
        assert torch.equal(actor.edge_magnitude, base)
        direction, stats = damped_direction(fisher, gradient)
        return direction, stats, fisher

    def replay(backward):
        loss = actor.edge_magnitude.square().sum()
        if backward:
            loss.backward()
        else:
            trials.append(actor.edge_magnitude.detach().clone()-base)
            if outcome == "exception":
                raise RuntimeError("deliberate conditioned replay failure")
        fail = not backward and (outcome == "reject" or (outcome == "retry" and len(trials) == 1))
        return dict(loss=float(loss.detach()) + (.01 if fail else 0.),
                    joint_kl_mean=.001, joint_kl_p99=.002,
                    native_latent_mean_displacement_rms=[.001]*4)

    if outcome == "exception":
        with pytest.raises(RuntimeError, match="deliberate"):
            projected_actor_proposal(actor, mask, replay, [.01]*4, conditioner=condition)
    else:
        stats = projected_actor_proposal(actor, mask, replay, [.01]*4, conditioner=condition)
        assert stats["accepted"] is (outcome != "reject")
        assert len(trials) == (1 if outcome == "accept" else 2)
        for trial, result in zip(trials, stats["attempts"], strict=True):
            assert result["predicted_local_joint_kl"] == pytest.approx(float(.5*trial@fisher@trial))
            assert result["predicted_loss_change"] == pytest.approx(float(2*base @ trial))
    assert len(conditions) == 1 and actor.edge_magnitude.grad is None
    assert actor.edge_magnitude[1] == base[1]
    if outcome in ("reject", "exception"):
        assert torch.equal(actor.edge_magnitude, base)


def test_natural_sink_proposal_composes_with_actual_recurrent_ppo_replay():
    _, model, data, _, advantages, _ = fixture()
    before = model.edge_magnitude.detach().clone()
    stats = natural_sink_actor_proposal(
        model, data, torch.ones_like(model.edge_magnitude, dtype=torch.bool),
        lambda backward: replay_sink_policy_gradient(model, data, advantages, backward=backward),
    )
    assert stats["accepted"] and stats["surrogate_descent"]
    assert stats["conditioning"]["dtype"] == "torch.float32"
    assert stats["predicted_local_joint_kl"] >= 0
    assert not torch.equal(model.edge_magnitude, before)


@pytest.mark.parametrize("base_value,accepted", [(3e-5, True), (1e-5, False)])
def test_unattainable_projected_target_retries_half_then_stops_round(base_value, accepted):
    actor = ParameterActor()
    actor.edge_magnitude = torch.nn.Parameter(torch.tensor([base_value]))
    calls = []

    def replay(backward):
        calls.append(backward)
        loss = actor.edge_magnitude.sum()
        if backward:
            loss.backward()
        return dict(loss=float(loss.detach()), joint_kl_mean=0., joint_kl_p99=0.,
                    native_latent_mean_displacement_rms=[0.]*4)

    def condition(gradient):
        return -gradient, {}, torch.eye(1)

    stats = projected_actor_proposal(actor, torch.ones(1, dtype=torch.bool), replay, [.01]*4,
                                     conditioner=condition)
    assert stats["accepted"] is accepted
    assert stats["stop_round"] is not accepted
    assert len(stats["attempts"]) == 2
    assert "sizing_failure" in stats["attempts"][0]
    assert calls == ([True, False] if accepted else [True])
    assert actor.edge_magnitude.grad is None
    assert float(actor.edge_magnitude.detach()[0]) == pytest.approx(
        base_value - 2.5e-5 if accepted else base_value, abs=1e-10)


def test_natural_update_refuses_full_network_before_loading(monkeypatch, tmp_path):
    import train_pragmatic_course_ppo as driver
    monkeypatch.setattr(sys, "argv", ["pilot", "--checkpoint", str(tmp_path / "source.pt"),
                                     "--output-dir", str(tmp_path / "run"),
                                     "--actor-update", "natural", "--full-history"])
    with pytest.raises(SystemExit, match="require --actor-scope roll-sinks"):
        driver.main()
