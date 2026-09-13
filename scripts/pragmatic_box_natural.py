"""Training-only box-feasible, KL-budgeted updates of existing motor synapses."""

from __future__ import annotations

import math

import torch
from pragmatic_policy_optimization import scaled_parameters, trust_decision
from pragmatic_sink_fisher import conditional_fisher, damped_direction


@torch.no_grad()
def box_quadratic_step(hessian, linear, lower, upper, *, max_iterations=128):
    """Feasible active-set Newton for a small convex quadratic, all in FP32.

    Minimize .5*d'H*d + linear'd, subject to lower <= d <= upper.
    Zero must be feasible. A capped/stalled solve returns its best feasible
    model-decreasing iterate, explicitly marked approximate, never a KKT claim.
    """
    count = linear.numel()
    if (linear.ndim != 1 or hessian.shape != (count, count)
            or lower.shape != linear.shape or upper.shape != linear.shape
            or max_iterations < 1 or not count):
        raise ValueError("matching nonempty box-QP dimensions and positive iteration cap required")
    if any(not bool(x.isfinite().all()) for x in (hessian, linear, lower, upper)):
        raise FloatingPointError("nonfinite box quadratic")
    if not bool(((lower <= 0) & (upper >= 0)).all()):
        raise ValueError("box must contain the zero displacement")
    delta = torch.zeros_like(linear)
    # -1 lower, +1 upper, +2 fixed, 0 free. Bounds can be RELEASED.
    active = torch.zeros_like(linear, dtype=torch.int8)
    active[lower == 0] = -1
    active[upper == 0] = 1
    active[lower == upper] = 2
    tolerance = 1e-5 * max(float(linear.abs().max()), 1e-6)
    best, best_value = delta.clone(), 0.
    releases = 0
    iterations = 0
    reason = "iteration-cap"
    for _ in range(max_iterations):
        iterations += 1
        gradient = hessian @ delta + linear
        if not bool(gradient.isfinite().all()):
            raise FloatingPointError("nonfinite box-QP gradient")
        free = active == 0
        stationary = not bool(free.any()) or float(gradient[free].abs().max()) <= tolerance
        if stationary:
            violation = torch.where(active == -1, -gradient,
                                    torch.where(active == 1, gradient, -torch.inf))
            if float(violation.max()) <= tolerance:
                reason = "stationary"
                break
            active[violation.argmax()] = 0
            releases += 1
            continue
        direction = torch.zeros_like(linear)
        direction[free] = torch.linalg.solve(hessian[free][:, free], -gradient[free])
        if not bool(direction.isfinite().all()):
            raise FloatingPointError("nonfinite box-QP Newton solve")
        ratios = torch.full_like(linear, torch.inf)
        positive, negative = direction > 0, direction < 0
        ratios[positive] = (upper[positive]-delta[positive]) / direction[positive]
        ratios[negative] = (lower[negative]-delta[negative]) / direction[negative]
        limiting = int(ratios.argmin())
        fraction = max(0., min(1., float(ratios[limiting])))
        candidate = (delta + fraction*direction).clamp(lower, upper)
        hits_bound = float(ratios[limiting]) <= 1.
        if hits_bound:
            at_upper = bool(direction[limiting] > 0)
            candidate[limiting] = upper[limiting] if at_upper else lower[limiting]
            active[limiting] = 1 if at_upper else -1
        value = float(.5*candidate @ hessian @ candidate + linear @ candidate)
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite box-QP objective")
        if value < best_value:
            best, best_value = candidate.clone(), value
        if torch.equal(candidate, delta) and not hits_bound:
            reason = "roundoff-stall"
            break
        delta = candidate
    # Diagnose the returned best iterate, which may differ from the last one.
    gradient = hessian @ best + linear
    residual = torch.where(lower == upper, 0., torch.where(
        best == lower, gradient.clamp_max(0),
        torch.where(best == upper, gradient.clamp_min(0), gradient)))
    kkt_residual = float(residual.abs().max())
    return best, dict(iterations=iterations, max_iterations=max_iterations,
                      termination=reason, converged=kkt_residual <= tolerance,
                      kkt_residual=kkt_residual, kkt_tolerance=tolerance,
                      model_change=best_value, released_bounds=releases,
                      active_bounds=int(((best == lower) | (best == upper)).sum()))


def box_natural_actor_proposal(model, data, mask, replay_fn, *, kl_budget=.002):
    """One box-QP direction, then fixed 1,...,1/128 replay backtracking.

    The Fisher budget is LOCAL to current weights. Existing behavior-relative
    trust and measured surrogate descent remain authoritative acceptance tests.
    Every trial starts from the same base; there is no deployed optimizer state.
    """
    if not math.isfinite(kl_budget) or kl_budget <= 0:
        raise ValueError("positive finite local KL budget required")
    parameter = model.edge_magnitude
    base = parameter.detach().clone()
    parameter.grad = None
    method = "full-history box-QP natural gradient; local KL budget; no momentum"
    try:
        before = replay_fn(True)
        if not math.isfinite(before["loss"]) or parameter.grad is None:
            raise ValueError("finite pre-update loss and actor gradient required")
        gradient = parameter.grad.detach() * mask
        if not bool(gradient.isfinite().all()):
            raise FloatingPointError("nonfinite box-natural gradient")

        def no_step(reason, **details):
            return dict(before=before, after=before, accepted=False, stop_round=True,
                        no_step_reason=reason, proposed_edge_delta_l2=0., changed_edge_count=0,
                        gradient_norm=float(gradient.norm()), attempts=[], update_method=method,
                        **details)

        if not bool((gradient != 0).any()):
            return no_step("zero raw gradient")
        fisher = conditional_fisher(model, data)
        direction, conditioning = damped_direction(fisher, gradient)
        curvature = float(.5*direction @ fisher @ direction)
        if not math.isfinite(curvature):
            raise FloatingPointError("nonfinite unconstrained Fisher curvature")
        if curvature <= 0:
            return no_step("no positive Fisher curvature", conditioning=conditioning)
        gain = math.sqrt(kl_budget / curvature)
        hessian = fisher + conditioning["damping"] * torch.eye(
            len(gradient), device=fisher.device, dtype=fisher.dtype)
        lower, upper = torch.where(mask, -base, 0.), torch.where(mask, 8-base, 0.)
        delta, qp = box_quadratic_step(hessian, gain*gradient, lower, upper)
        full, _ = scaled_parameters(base, delta, 1., mask)
        delta = full - base  # Account for actual native-parameter FP32 rounding.
        full_kl = float(.5*delta @ fisher @ delta)
        if not math.isfinite(full_kl):
            raise FloatingPointError("nonfinite materialized box-step Fisher curvature")
        if full_kl <= 0 or float(gradient @ delta) >= 0:
            return no_step("no useful feasible descent", conditioning=conditioning, box_qp=qp)
        budget_scale = min(1., math.sqrt(kl_budget/full_kl))
        attempts = []
        for backtrack in range(8):
            fraction = budget_scale * .5**backtrack
            proposed, projections = scaled_parameters(base, delta, fraction, mask)
            actual_delta = proposed - base
            predicted = float(gradient @ actual_delta)
            predicted_kl = float(.5*actual_delta @ fisher @ actual_delta)
            if not math.isfinite(predicted) or not math.isfinite(predicted_kl):
                raise FloatingPointError("nonfinite materialized box-natural prediction")
            attempt = dict(fraction=fraction, predicted_loss_change=predicted,
                           predicted_local_joint_kl=predicted_kl,
                           target_local_joint_kl=kl_budget, projected_magnitudes=projections,
                           proposed_edge_delta_l2=float(actual_delta.norm()),
                           changed_edge_count=int((proposed != base).sum()))
            if predicted >= 0 or predicted_kl < 0 or predicted_kl > kl_budget*(1+1e-4):
                attempts.append(dict(
                    **attempt, accepted=False, after=before,
                    no_replay_reason="rounded step lacks descent/budget feasibility"))
                continue
            with torch.no_grad():
                parameter.copy_(proposed)
            after = replay_fn(False)
            change = after["loss"] - before["loss"]
            if not math.isfinite(change):
                raise FloatingPointError("nonfinite box-natural replay loss")
            decision = trust_decision(after, data.stationary_std)
            required = max(5e-6, .1 * -predicted)
            accepted = decision["accepted"] and change <= -required
            attempts.append(dict(**attempt, after=after, accepted=accepted,
                                 observed_loss_change=change, required_loss_decrease=required,
                                 surrogate_descent=change <= -required,
                                 trust_accepted=decision["accepted"],
                                 native_mean_shift_in_stationary_std=
                                 decision["native_mean_shift_in_stationary_std"]))
            if accepted:
                break
        accepted = attempts[-1]["accepted"]
        if not accepted:
            with torch.no_grad():
                parameter.copy_(base)
        return dict(before=before, **attempts[-1], gradient_norm=float(gradient.norm()),
                    conditioning=conditioning, box_qp=qp, unconstrained_scale=gain,
                    full_box_local_kl=full_kl, budget_scale=budget_scale, attempts=attempts,
                    stop_round=not accepted or decision["stop_round"], update_method=method)
    except BaseException:
        with torch.no_grad():
            parameter.copy_(base)
        raise
    finally:
        parameter.grad = None
