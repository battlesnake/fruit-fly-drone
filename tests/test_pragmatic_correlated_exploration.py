from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pragmatic_correlated_exploration as exploration  # noqa: E402


def test_stationary_first_sample_and_conditional_history_not_independent_marginals():
    mean = torch.zeros(2, 4)
    previous = torch.ones_like(mean)
    latent = torch.full_like(mean, 3.0)
    conditional, std = exploration.ar1_conditional(
        mean, previous, latent, [2.0] * 4, 0.6, torch.tensor([True, False])
    )
    assert torch.equal(conditional[0], mean[0])
    assert torch.allclose(conditional[1], torch.full((4,), 1.2))
    assert torch.allclose(std, torch.tensor([[2.0] * 4, [1.6] * 4]))


def test_previous_mean_receives_gradient_at_noninitial_chunk_boundary():
    mean = torch.zeros(1, 4, requires_grad=True)
    previous = torch.zeros_like(mean, requires_grad=True)
    conditional, std = exploration.ar1_conditional(
        mean, previous, torch.ones_like(mean), [1.0] * 4, 0.8, torch.tensor([False])
    )
    exploration.joint_log_prob(torch.zeros_like(mean), conditional, std).sum().backward()
    assert torch.all(mean.grad != 0)
    assert torch.allclose(previous.grad, -0.8 * mean.grad)


def test_sequence_density_and_gradients_match_full_correlated_gaussian():
    torch.manual_seed(22)
    times, rho = 6, 0.7
    means = (torch.randn(times, 4) * 0.1).requires_grad_(True)
    latent = torch.randn_like(means)
    stationary = torch.tensor([0.4, 0.3, 0.2, 0.5])
    first = torch.arange(times) == 0
    conditional, std = exploration.ar1_conditional(
        means,
        torch.cat((means[:1], means[:-1])),
        torch.cat((latent[:1], latent[:-1])),
        stationary,
        rho,
        first,
    )
    sequential = exploration.joint_log_prob(latent, conditional, std, squashed=False).sum()
    indices = torch.arange(times)
    covariance = rho ** (indices[:, None] - indices[None, :]).abs()
    dense = sum(
        torch.distributions.MultivariateNormal(
            means[:, axis], covariance_matrix=covariance * stationary[axis].square()
        ).log_prob(latent[:, axis])
        for axis in range(4)
    )
    assert torch.allclose(sequential, dense, atol=3e-5)
    sequential_grad = torch.autograd.grad(sequential, means, retain_graph=True)[0]
    dense_grad = torch.autograd.grad(dense, means)[0]
    assert torch.allclose(sequential_grad, dense_grad, atol=3e-5, rtol=3e-5)


def test_squash_jacobian_cancels_in_ratio_and_large_latents_stay_finite():
    latent = torch.tensor([[0.3, -0.2, 20.0, -20.0]])
    old, new, std = torch.zeros_like(latent), torch.full_like(latent, 0.1), torch.ones_like(latent)
    ratios = []
    for squashed in (False, True):
        ratios.append(
            exploration.joint_log_prob(latent, new, std, squashed=squashed)
            - exploration.joint_log_prob(latent, old, std, squashed=squashed)
        )
    assert torch.isfinite(ratios[1]).all()
    assert torch.allclose(ratios[0], ratios[1], atol=5e-5)


def test_analytic_kl_is_joint_and_penalizes_changes_to_any_axis():
    old, new, std = torch.zeros(2, 4), torch.zeros(2, 4), torch.ones(2, 4)
    new[0, 1] = 2  # Shared pathways can change pitch even with a roll-path mask.
    assert exploration.conditional_kl(old, new, std).tolist() == [2.0, 0.0]


@pytest.mark.parametrize("rho", [1, -0.1, float("nan")])
def test_invalid_correlation_is_rejected(rho):
    means = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="rho"):
        exploration.ar1_conditional(
            means, means, means, [1.0] * 4, rho, torch.zeros(2, dtype=torch.bool)
        )
