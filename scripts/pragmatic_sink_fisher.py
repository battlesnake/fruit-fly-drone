"""Training-only conditional Fisher for existing, no-outgoing roll motor cells.

No extra inference state or decoder: the Jacobian and damped solve only determine
updates to native incoming magnitudes. Recompute both at every current proposal.
"""

from __future__ import annotations

import math

import torch
from pragmatic_policy_optimization import projected_actor_proposal


@torch.no_grad()
def roll_mean_jacobian(model, data):
    """Full-history d(atanh(native roll))/d(selected native magnitudes), FP32."""
    model.validate_recording(data)
    sink = model.sink
    target = sink.targets(data.sink_features)
    indices = sink.matrix_indices[sink.train_slots]
    post = indices // len(sink.parents)
    parent = indices % len(sink.parents)
    signs = sink.signs[sink.train_slots]
    alpha = 1 - sink.decay
    state = target.new_zeros(target.shape[1:])
    sensitivity = target.new_zeros((target.shape[1], len(indices)))
    pool = target.new_full((sink.motor_count,), -1 / (sink.motor_count-sink.positive_count))
    pool[:sink.positive_count] = 1 / sink.positive_count
    jacobians = []
    for time, frame in enumerate(target):
        state = state + alpha * (frame - state)
        local = ((1 - (frame[:, post] / 5).square()) * signs
                 * data.sink_features[time, :, parent])
        sensitivity = sensitivity + alpha[post] * (local - sensitivity)
        if time < data.warmup_steps:
            continue
        rates = state.sigmoid()
        roll = (rates[:, :sink.positive_count].mean(-1)
                - rates[:, sink.positive_count:].mean(-1))
        bounded = roll.clamp(-.999999, .999999)
        readout = pool[post] * rates[:, post] * (1-rates[:, post])
        jacobian = readout * sensitivity / (1-bounded.square())[:, None]
        jacobians.append(torch.where((bounded == roll)[:, None], jacobian, 0.))
    result = torch.stack(jacobians)
    if not bool(result.isfinite().all()):
        raise FloatingPointError("nonfinite native motor Jacobian")
    return result


@torch.no_grad()
def conditional_fisher(model, data):
    """Empirical Fisher of fixed-variance AR action conditionals on valid steps."""
    if data.stationary_std is None or not 0 <= data.rho < 1:
        raise ValueError("conditional Fisher requires stationary action noise and 0 <= rho < 1")
    sigma = data.stationary_std[0]
    if not math.isfinite(sigma) or sigma <= 0 or not bool(data.valid.any()):
        raise ValueError("conditional Fisher requires positive sigma and valid commands")
    jacobian = roll_mean_jacobian(model, data)
    # No previous physical action at t=0, even when neural warmup was nonzero.
    conditional = torch.cat((jacobian[:1] / sigma,
                             (jacobian[1:] - data.rho * jacobian[:-1])
                             / (sigma * math.sqrt(1-data.rho**2))))
    values = conditional[data.valid]
    fisher = values.T @ values / len(values)
    fisher = .5 * (fisher + fisher.T)
    if not bool(fisher.isfinite().all()):
        raise FloatingPointError("nonfinite conditional Fisher")
    return fisher


@torch.no_grad()
def damped_direction(fisher, gradient, *, damping_fraction=.01):
    """Solve in FP32; damping relative to average diagonal, not weight units."""
    if (fisher.shape != (gradient.numel(), gradient.numel())
            or not bool(fisher.isfinite().all() & gradient.isfinite().all())):
        raise ValueError("finite matching Fisher and gradient required")
    diagonal_mean = float(fisher.diag().mean())
    damping = damping_fraction * diagonal_mean
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("positive finite Fisher damping required")
    matrix = fisher + damping * torch.eye(len(gradient), device=fisher.device, dtype=fisher.dtype)
    solved = torch.linalg.solve(matrix, gradient)
    contraction = float(gradient @ solved)
    if not bool(solved.isfinite().all()) or not math.isfinite(contraction) or contraction <= 0:
        raise FloatingPointError("damped Fisher solve did not give a finite descent direction")
    return -solved, dict(damping=damping, damping_fraction=damping_fraction,
                         fisher_diagonal_mean=diagonal_mean,
                         raw_gradient_dot_preconditioned=contraction,
                         direction_norm=float(solved.norm()), dtype=str(fisher.dtype))


def natural_sink_actor_proposal(model, data, mask, replay_fn, *, predicted_decrease=5e-5):
    def condition(gradient):
        fisher = conditional_fisher(model, data)
        direction, stats = damped_direction(fisher, gradient)
        return direction, stats, fisher

    return projected_actor_proposal(model, mask, replay_fn, data.stationary_std,
                                    predicted_decrease=predicted_decrease, conditioner=condition)
