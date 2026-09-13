"""Training-only critic and transactional, full-round recurrent PPO updates."""

from __future__ import annotations

import copy
import math
from time import perf_counter

import torch
from pragmatic_full_policy_gradient import replay_full_policy_gradient
from pragmatic_recurrent_policy_gradient import replay_joint_policy_gradient


class OutcomeCritic(torch.nn.Module):
    """Separate privileged baseline; never exported as part of the actor."""

    def __init__(self, features):
        super().__init__()
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.network = torch.nn.Sequential(
            torch.nn.Linear(features, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1),
        )

    @torch.no_grad()
    def initialize_normalization(self, values):
        self.center.copy_(values.mean(0))
        self.scale.copy_(values.std(0, unbiased=False).clamp_min(0.01))

    def forward(self, features):
        return self.network((features - self.center) / self.scale).squeeze(-1)


@torch.no_grad()
def round_advantages(data, critic, *, zero_baseline=False, failure_aware=False):
    """Freeze advantages once across ALL valid samples, before critic fitting.

    Optionally replace ALREADY failed tails with uncentered remaining returns.
    Preserve original normalization and healthy-command credit exactly, including
    the failure-causing command. Later ground/invalid costs remain in the return;
    no safety rewards, validity masks or actor observations are changed.
    """
    device = next(critic.parameters()).device
    baseline = (torch.zeros_like(data.returns) if zero_baseline else
                critic(data.critic_features.to(device)).cpu())
    raw = data.returns - baseline
    values = raw[data.valid]
    if not bool(values.isfinite().all()):
        raise FloatingPointError("nonfinite critic baseline or returns")
    mean, scale = values.mean(), values.std(unbiased=False).clamp_min(1e-6)
    result = torch.where(data.valid, (raw - mean) / scale, 0).detach()
    stats = dict(zero_baseline=zero_baseline, raw_mean=float(mean), raw_std=float(scale),
                 baseline_mse=float((baseline - data.returns)[data.valid].square().mean()),
                 failure_aware=failure_aware)
    if failure_aware:
        failed = getattr(data, "failed_before_command", None)
        if (failed is None or failed.shape != data.valid.shape or failed.dtype != torch.bool
                or failed.device != data.valid.device):
            raise ValueError("failure-aware advantages require matching boolean pre-command flags")
        if bool((failed[:-1] & ~failed[1:]).any()):
            raise ValueError("latched course failure cannot clear within a rollout")
        tail = failed & data.valid
        original_tail_squared = float(result[tail].square().sum())
        result = torch.where(tail, data.returns / scale, result)
        stats.update(
            failed_tail_valid_commands=int(tail.sum()),
            failed_tail_zero_return_commands=int((tail & (data.returns == 0)).sum()),
            failed_tail_safety_return_commands=int((tail & (data.returns < 0)).sum()),
            original_failed_tail_squared_advantage_sum=original_tail_squared,
            revised_failed_tail_squared_advantage_sum=float(result[tail].square().sum()),
            original_normalization_preserved=True,
        )
    return result, stats


def fit_critic(critic, optimizer, data, *, epochs=5, batch_size=1024, seed=1):
    device = next(critic.parameters()).device
    x, y = data.critic_features[data.valid].to(device), data.returns[data.valid].to(device)
    rng = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(y), device=device, generator=rng)
        for rows in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = (critic(x[rows]) - y[rows]).square().mean()
            if not bool(loss.isfinite()):
                raise FloatingPointError("nonfinite critic loss")
            loss.backward()
            if any(p.grad is not None and not bool(p.grad.isfinite().all())
                   for p in critic.parameters()):
                raise FloatingPointError("nonfinite critic gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    with torch.no_grad():
        mse = float((critic(x) - y).square().mean())
    if not math.isfinite(mse):
        raise FloatingPointError("nonfinite fitted critic")
    return dict(epochs=epochs, minibatches=len(losses), final_mse=mse)


def aggregate_replays(parts):
    """Count-weight means/RMS and compute p99 from ALL individual valid KLs."""
    count = sum(part["valid_commands"] for part in parts)
    if not count:
        raise ValueError("no valid replay commands")
    kl = torch.tensor([value for part in parts for value in part["joint_kl_samples"]])
    if len(kl) != count or not bool(kl.isfinite().all()):
        raise FloatingPointError("invalid per-command KL diagnostics")
    result = dict(valid_commands=count, joint_kl_mean=float(kl.mean()),
                  joint_kl_p99=float(torch.quantile(kl, 0.99)), joint_kl_max=float(kl.max()))
    for name in ("loss", "joint_ratio_mean", "clipped_command_fraction"):
        result[name] = sum(p[name] * p["valid_commands"] for p in parts) / count
    for name in ("native_latent_mean_displacement_rms", "native_clamped_fraction"):
        values = torch.tensor([p[name] for p in parts])
        weights = torch.tensor([p["valid_commands"] / count for p in parts])[:, None]
        result[name] = (((values.square() * weights).sum(0).sqrt() if name.endswith("rms")
                         else (values * weights).sum(0)).tolist())
    return result


def replay_round(controller, data, advantages, *, camera, gate_config, microbatch=4,
                 chunk_steps=20, backward=True, progress=None, full_history=False):
    device = next(controller.parameters()).device
    count = int(data.valid.sum())
    parts = []
    for start in range(0, data.valid.shape[1], microbatch):
        started = perf_counter()
        rows = range(start, min(start + microbatch, data.valid.shape[1]))
        batch = data.select(rows, device)
        n = int(batch.valid.sum())
        if not n:
            continue
        replay = replay_full_policy_gradient if full_history else replay_joint_policy_gradient
        stats = replay(
            controller, lambda t, b=batch: b.observation(t, camera, gate_config), batch.latents,
            batch.old_means, batch.old_log_prob, advantages[:, rows].to(device), batch.valid,
            stationary_std=batch.stationary_std, rho=batch.rho, chunk_steps=chunk_steps,
            warmup_steps=batch.warmup_steps, backward=backward, gradient_scale=n / count,
            include_kl_samples=True,
        )
        parts.append(stats)
        if progress:
            progress(dict(stage="backward-microbatch" if backward else "trust-microbatch",
                          first_episode=start, episodes=len(rows),
                          joint_kl_mean=stats["joint_kl_mean"],
                          wall_seconds=perf_counter()-started,
                          cuda_peak_reserved_mib=(torch.cuda.max_memory_reserved(device)/2**20
                                                  if device.type == "cuda" else None)))
    return aggregate_replays(parts)


def trust_decision(stats, stationary_std):
    shift = [value / sigma for value, sigma in
             zip(stats["native_latent_mean_displacement_rms"], stationary_std, strict=True)]
    checks = [stats["joint_kl_mean"], stats["joint_kl_p99"], *shift]
    if not all(math.isfinite(value) for value in checks):
        raise FloatingPointError("nonfinite policy trust diagnostics")
    reject = stats["joint_kl_mean"] > 0.01 or stats["joint_kl_p99"] > 0.10 or max(shift) > 0.5
    return dict(accepted=not reject, stop_round=reject or stats["joint_kl_mean"] > 0.005,
                native_mean_shift_in_stationary_std=shift)


def actor_proposal(controller, optimizer, mask, replay_fn, stationary_std):
    """One proposal; replay_fn must rebuild all current-weight observation prefixes."""
    parameter = controller.edge_magnitude
    before = parameter.detach().clone()
    adam_before = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    try:
        before_stats = replay_fn(True)
        if parameter.grad is None:
            raise ValueError("actor replay produced no gradient")
        parameter.grad.mul_(mask)
        norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
        optimizer.step()
        # Only selected existing magnitudes may change; signs/topology stay fixed.
        with torch.no_grad():
            parameter[mask] = parameter[mask].clamp(0, 8)
            parameter[~mask] = before[~mask]
        if not bool(parameter.isfinite().all()):
            raise FloatingPointError("nonfinite actor parameter")
        after_stats = replay_fn(False)
        decision = trust_decision(after_stats, stationary_std)
        result = dict(before=before_stats, after=after_stats, gradient_norm=float(norm),
                      proposed_edge_delta_l2=float((parameter.detach() - before).norm()),
                      **decision)
        if not decision["accepted"]:
            with torch.no_grad():
                parameter.copy_(before)
            optimizer.load_state_dict(adam_before)
        return result
    except Exception:
        with torch.no_grad():
            parameter.copy_(before)
        optimizer.load_state_dict(adam_before)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)


def scaled_parameters(base, delta, scale, mask):
    """Start from base and account for actual FP32 rounding and native bounds."""
    result = base.clone()
    proposed = base[mask] + scale * delta[mask]
    result[mask] = proposed.clamp(0, 8)
    return result, int((proposed != result[mask]).sum())


def steepest_parameters(base, gradient, mask, target):
    """Parameter-only sizing against actual projected contraction, not loss."""
    gradient = gradient * mask
    squared = float(gradient.square().sum())
    if not 0 < squared < float("inf") or not 0 < target < float("inf"):
        raise ValueError("need finite nonzero gradient and positive target")
    low, high = 0., 2 * target / squared
    best, best_error = None, float("inf")
    bracketed = False
    for attempt in range(24):
        eta = .5 * (low + high) if bracketed else high
        proposed, projections = scaled_parameters(base, -gradient, eta, mask)
        contraction = float((gradient * (proposed-base)).sum())
        error = abs(contraction+target)
        if error < best_error:
            best_error = error
            best = proposed, dict(eta=eta, projected_magnitudes=projections,
                                  predicted_loss_change=contraction,
                                  target_predicted_decrease=target, sizing_attempts=attempt+1)
        if error <= .1 * target:
            return best
        if -contraction < target:
            low = eta
            if not bracketed:
                high *= 2
        else:
            high = eta
            bracketed = True
    raise ValueError("projected FP32 step could not resolve target within 10 percent")


class UnresolvedProjectedStep(ValueError):
    """Finite local sizing exhausted; caller may retry a smaller target."""


def directional_parameters(base, gradient, direction, mask, target):
    """Bounded local sizing without assuming projected contraction is monotone.

    Clipping a preconditioned direction can destroy descent. Try local scale
    corrections, checking the actual rounded displacement against RAW gradient
    every time; never accept an unresolved or non-descending materialized step.
    """
    gradient, direction = gradient * mask, direction * mask
    contraction_rate = float((gradient * direction).sum())
    if (not bool(direction.isfinite().all()) or not math.isfinite(contraction_rate)
            or contraction_rate >= 0 or not 0 < target < float("inf")):
        raise ValueError("need finite descending direction and positive target")
    eta = target / -contraction_rate
    for attempt in range(24):
        proposed, projections = scaled_parameters(base, direction, eta, mask)
        contraction = float((gradient * (proposed-base)).sum())
        if not math.isfinite(contraction):
            raise FloatingPointError("nonfinite projected directional contraction")
        if abs(contraction + target) <= .1 * target:
            return proposed, dict(eta=eta, projected_magnitudes=projections,
                                  predicted_loss_change=contraction,
                                  target_predicted_decrease=target, sizing_attempts=attempt+1)
        if contraction > 0:
            eta *= .5  # Projection lost descent: retreat, not a monotone bracket.
        elif contraction == 0:
            # A rounded zero step may need enlargement; a nonzero step with no
            # contraction needs retreat because projection changed its direction.
            eta *= 2 if torch.equal(proposed, base) else .5
        else:
            eta *= min(2., max(.25, target / -contraction))
    raise UnresolvedProjectedStep(
        "projected direction could not resolve descending target within 10 percent")


def projected_actor_proposal(controller, mask, replay_fn, stationary_std, *,
                             predicted_decrease=5e-5, conditioner=None):
    """Full-history gradient, projected direction, and at most one half retry.

    The caller must use untruncated current-weight replay including warmup.
    Both trials start at the same base and use the same raw masked gradient.
    Acceptance needs actual surrogate descent as well as behavior-relative trust.
    No actor optimizer state exists; failures restore parameters and clear grads.
    """
    parameter = controller.edge_magnitude
    before = parameter.detach().clone()
    parameter.grad = None
    try:
        before_stats = replay_fn(True)
        if not math.isfinite(before_stats["loss"]):
            raise FloatingPointError("nonfinite pre-update surrogate")
        if parameter.grad is None:
            raise ValueError("actor replay produced no gradient")
        gradient = parameter.grad.detach() * mask
        if not bool(gradient.isfinite().all()):
            raise FloatingPointError("nonfinite steepest actor gradient")
        direction, conditioning, fisher = (None, {}, None)
        if conditioner is not None:
            direction, conditioning, fisher = conditioner(gradient)
        attempts = []
        for target in (predicted_decrease, .5 * predicted_decrease):
            if direction is None:
                proposed, sizing = steepest_parameters(before, gradient, mask, target)
            else:
                try:
                    proposed, sizing = directional_parameters(
                        before, gradient, direction, mask, target)
                except UnresolvedProjectedStep as error:
                    # Projection can make the full target unattainable while the
                    # half target remains feasible. This is not a broken solve.
                    with torch.no_grad():
                        parameter.copy_(before)
                    decision = trust_decision(before_stats, stationary_std)
                    accepted = False
                    attempts.append(dict(
                        after=before_stats, accepted=False, sizing_failure=str(error),
                        target_predicted_decrease=target, predicted_loss_change=None,
                        observed_loss_change=0., surrogate_descent=False, trust_accepted=None,
                        proposed_edge_delta_l2=0., changed_edge_count=0,
                        native_mean_shift_in_stationary_std=
                        decision["native_mean_shift_in_stationary_std"],
                    ))
                    continue
                delta = proposed - before
                sizing["predicted_local_joint_kl"] = float(.5 * delta @ fisher @ delta)
            with torch.no_grad():
                parameter.copy_(proposed)
            if not bool(parameter.isfinite().all()):
                raise FloatingPointError("nonfinite steepest actor parameter")
            after_stats = replay_fn(False)
            change = after_stats["loss"] - before_stats["loss"]
            if not math.isfinite(change):
                raise FloatingPointError("nonfinite post-update surrogate")
            decision = trust_decision(after_stats, stationary_std)
            required = max(5e-6, .1 * -sizing["predicted_loss_change"])
            descent = change <= -required
            accepted = decision["accepted"] and descent
            attempts.append(dict(
                **sizing, after=after_stats, observed_loss_change=change,
                required_loss_decrease=required, surrogate_descent=descent,
                trust_accepted=decision["accepted"], accepted=accepted,
                proposed_edge_delta_l2=float((proposed-before).norm()),
                changed_edge_count=int((proposed != before).sum()),
                native_mean_shift_in_stationary_std=decision["native_mean_shift_in_stationary_std"],
            ))
            if accepted:
                break
        if not accepted:
            with torch.no_grad():
                parameter.copy_(before)
        return dict(before=before_stats, **attempts[-1], gradient_norm=float(gradient.norm()),
                    stop_round=not accepted or decision["stop_round"], attempts=attempts,
                    conditioning=conditioning,
                    update_method=("full-history projected damped natural gradient; no momentum"
                                   if conditioner else
                                   "full-history projected steepest; no momentum"))
    except BaseException:
        with torch.no_grad():
            parameter.copy_(before)
        raise
    finally:
        parameter.grad = None


def steepest_actor_proposal(controller, mask, replay_fn, stationary_std, *,
                            predicted_decrease=5e-5):
    return projected_actor_proposal(controller, mask, replay_fn, stationary_std,
                                    predicted_decrease=predicted_decrease)


def native_score(metrics):
    return (metrics["clean_course_success_rate"],
            min(metrics["clean_course_negative_success_rate"],
                metrics["clean_course_positive_success_rate"]),
            metrics["gates_before_failure_mean"])


def eligible_native(metrics):
    return metrics["ground_contact_rate"] == 0 and metrics["invalid_rate"] == 0


def select_new_native(metrics, best, revision, last_evaluated_revision):
    """A repeated evaluation of unchanged weights cannot manufacture a winner."""
    return (revision > last_evaluated_revision and eligible_native(metrics)
            and native_score(metrics) > native_score(best))
