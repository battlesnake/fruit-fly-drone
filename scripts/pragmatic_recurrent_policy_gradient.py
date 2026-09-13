"""Training-only full-episode microbatch replay for joint correlated-action PPO.

This is a gradient helper, not a trainer or deployed controller. The caller owns
collection, advantages and the optimizer. Weights MUST remain fixed for one whole
call; make an optimizer step only afterwards and restart replay on the next call.
"""

from __future__ import annotations

import math

import torch
from pragmatic_correlated_exploration import ar1_conditional, conditional_kl, joint_log_prob


def reward_to_go(rewards, valid):
    """Undiscounted finite-episode returns; include the terminal-causing command.

Rows may stop only once: false entries are an absorbing tail, never chunk or gate
boundaries. A ring/order failure alone must not end a row because ground penalties
may still occur later. No value bootstrap is used beyond the complete episode.
"""
    if (rewards.shape != valid.shape or rewards.ndim != 2 or valid.dtype != torch.bool
            or min(rewards.shape) < 1):
        raise ValueError("rewards and boolean validity must have matching time/episode shapes")
    if bool((valid[1:] & ~valid[:-1]).any()):
        raise ValueError("absorbing rows cannot revive")
    if bool((rewards[~valid] != 0).any()):
        raise ValueError("absorbing tails must have zero rewards")
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for time in range(len(rewards) - 1, -1, -1):
        running = torch.where(valid[time], rewards[time] + running, 0)
        result[time] = running
    return result


def replay_joint_policy_gradient(
    controller, observe, latents, old_means, old_log_prob, advantages, valid,
    *, stationary_std, rho, chunk_steps=20, warmup_steps=10, clip=0.1, backward=True,
    gradient_scale=1.0,
):
    """Replay complete physical observation histories under CURRENT weights.

observe(t) returns (RGB, roll/pitch) reconstructed from stored physical state.
No plant gradients, replayed neural states, previous motor inputs or oracle actor
features are used. Short neural chunks truncate gradients only, never values.
The preceding frame is re-evaluated with gradient at each chunk boundary so the
AR conditional's previous native mean is not accidentally detached.

    Returns diagnostics and accumulates gradients normalized by this microbatch's
    total valid commands. Does not zero gradients, change weights or take a step.
    old_log_prob MUST be the unsquashed latent density. All padding and replayed
    observations must be finite, including absorbing rows. To accumulate multiple
    microbatches before ONE optimizer step, set gradient_scale to this batch's valid
    count divided by the combined count; otherwise each call is independently normalized.
On error the caller must discard any partially accumulated gradients.
"""
    if latents.ndim != 3 or latents.shape[-1] != 4 or old_means.shape != latents.shape:
        raise ValueError("latents and old means must have shape (time, episodes, 4)")
    times, episodes, _ = latents.shape
    if times < 1 or episodes < 1 or chunk_steps < 1 or warmup_steps < 0 or not 0 < clip < 1:
        raise ValueError("invalid replay dimensions, chunk length, warmup or clipping")
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient scale must be finite and positive")
    if (old_log_prob.shape != (times, episodes) or advantages.shape != old_log_prob.shape
            or valid.shape != old_log_prob.shape or valid.dtype != torch.bool):
        raise ValueError("log probabilities, advantages and valid mask must match time/episodes")
    if any(value.requires_grad for value in (latents, old_means, old_log_prob, advantages)):
        raise ValueError("behavior samples and advantages must be detached")
    if any(not bool(value.isfinite().all()) for value in
           (latents, old_means, old_log_prob, advantages)):
        raise ValueError("all replay values, including inactive padding, must be finite")
    sample_count = int(valid.sum())
    if not sample_count:
        raise ValueError("no valid commands to replay")
    first = torch.zeros(times, episodes, device=latents.device, dtype=torch.bool)
    first[0] = True
    previous_latents = torch.cat((latents[:1], latents[:-1]))
    old_conditional, std = ar1_conditional(
        old_means, torch.cat((old_means[:1], old_means[:-1])), previous_latents,
        stationary_std, rho, first,
    )
    expected_log_prob = joint_log_prob(latents, old_conditional, std, squashed=False)
    if not torch.allclose(old_log_prob[valid], expected_log_prob[valid], rtol=1e-5, atol=1e-4):
        raise ValueError("behavior log probabilities must match unsquashed latent densities")
    neural = controller.initial_state(episodes, device=latents.device, dtype=latents.dtype)
    with torch.no_grad():
        image, attitude = observe(0)
        for _ in range(warmup_steps):
            _, neural = controller(image, attitude, neural)
    neural = neural.detach()
    previous_input_state = None
    loss_sum = torch.zeros((), device=latents.device)
    means, kls, ratios = [], [], []
    clamped = torch.zeros(4, device=latents.device)
    with torch.set_grad_enabled(backward):
        for start in range(0, times, chunk_steps):
            finite_actor = torch.ones((), device=latents.device, dtype=torch.bool)

            def forward(time, state):
                nonlocal finite_actor
                observations = observe(time)
                for value in observations:
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        finite_actor &= value.isfinite().all()
                output, advanced = controller(*observations, state)
                finite_actor &= output.isfinite().all() & advanced.isfinite().all()
                return output, advanced

            if start:
                # Numerical re-evaluation, NOT an additional physical brain tick.
                motor, neural = forward(start - 1, previous_input_state)
                previous_mean = torch.atanh(motor.clamp(-0.999999, 0.999999))
            else:
                previous_mean = None
            losses = []
            for time in range(start, min(start + chunk_steps, times)):
                previous_input_state = neural.detach()
                motor, neural = forward(time, neural)
                bounded = motor.clamp(-0.999999, 0.999999)
                mean = torch.atanh(bounded)
                conditional = (mean if time == 0 else mean + rho * (
                    previous_latents[time] - previous_mean
                ))
                log_prob = joint_log_prob(latents[time], conditional, std[time], squashed=False)
                ratio = (log_prob - old_log_prob[time]).exp()
                unclipped = ratio * advantages[time]
                clipped = ratio.clamp(1 - clip, 1 + clip) * advantages[time]
                losses.append(-torch.minimum(unclipped, clipped)[valid[time]].sum() / sample_count)
                with torch.no_grad():
                    means.append(mean.detach())
                    kls.append(conditional_kl(old_conditional[time], conditional, std[time]))
                    ratios.append(ratio.detach())
                    clamped += ((bounded != motor) & valid[time, :, None]).sum(0)
                previous_mean = mean
            chunk_loss = torch.stack(losses).sum()
            if not bool(finite_actor & torch.isfinite(chunk_loss)):
                raise FloatingPointError("nonfinite policy loss; discard accumulated gradients")
            loss_sum += chunk_loss.detach()
            if backward:
                (gradient_scale * chunk_loss).backward()
            # No optimizer step here. The next chunk resumes from the detached
            # state BEFORE the preceding frame and recomputes that overlap.
            neural = neural.detach()
            previous_mean = None
    if backward and any(parameter.grad is not None and not bool(parameter.grad.isfinite().all())
                        for parameter in controller.parameters()):
        raise FloatingPointError("nonfinite parameter gradient; discard accumulated gradients")
    kl = torch.stack(kls)[valid]
    ratio = torch.stack(ratios)[valid]
    displacement = (torch.stack(means) - old_means)[valid]
    return dict(
        loss=float(loss_sum), valid_commands=sample_count,
        joint_kl_mean=float(kl.mean()), joint_kl_p99=float(torch.quantile(kl, 0.99)),
        joint_kl_max=float(kl.max()), joint_ratio_mean=float(ratio.mean()),
        clipped_command_fraction=float(((ratio - 1).abs() > clip).float().mean()),
        native_latent_mean_displacement_rms=displacement.square().mean(0).sqrt().tolist(),
        native_clamped_fraction=(clamped / sample_count).tolist(),
        gradient_chunk_steps=chunk_steps,
        replay_forward_calls=warmup_steps + times + (times - 1) // chunk_steps,
        recurrent_gradient_is_truncated=chunk_steps < times,
    )
