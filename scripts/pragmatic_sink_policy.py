"""Outcome-PPO replay of existing roll motor sinks, never a deployed extra head.

Only incoming roll-motor magnitudes change. Their ancestors are frozen and cannot
receive neural feedback from these sinks, so recorded parent activities are valid
on each fixed observation history. Fresh flights always run the full connectome.
"""

from __future__ import annotations

import torch
from pragmatic_correlated_exploration import conditional_kl, joint_log_prob
from pragmatic_recurrent_policy_gradient import prepare_policy_replay

from flydrone.motor_slice import SinkMotorSlice


class NativeRollSinkPolicy(torch.nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.sink = SinkMotorSlice(controller, axis=0)
        others = controller.pool_indices[int(controller.pool_offsets[2]):]
        if bool(torch.isin(self.sink.motors, others).any()):
            raise ValueError("roll and non-roll motor pools must be disjoint")

    @property
    def edge_magnitude(self):
        return self.sink.magnitudes

    def compile_into(self, controller):
        self.sink.compile_into(controller)

    def validate_recording(self, data):
        if data.sink_features is None or data.sink_parent_nodes is None:
            raise ValueError("sink replay requires freshly recorded parent activities")
        if not torch.equal(data.sink_parent_nodes, self.sink.parents):
            raise ValueError("sink parent neuron order does not match this policy")
        expected = (len(data.latents) + data.warmup_steps, data.latents.shape[1],
                    len(self.sink.parents))
        if data.sink_features.shape != expected or data.sink_features.requires_grad:
            raise ValueError("sink features must be detached complete histories including warmup")
        if not bool(data.sink_features.isfinite().all()):
            raise FloatingPointError("nonfinite recorded sink features")

    def means(self, data):
        self.validate_recording(data)
        roll = self.sink.forward_recurrent(data.sink_features)[data.warmup_steps:]
        if not bool(roll.isfinite().all()):
            raise FloatingPointError("nonfinite recurrent sink output")
        bounded = roll.clamp(-.999999, .999999)
        # These three means are unchanged on FIXED observations: only disjoint,
        # no-outgoing roll motor cells were modified. No such cache is deployed.
        means = torch.cat((torch.atanh(bounded)[..., None], data.old_means[..., 1:]), -1)
        return means, bounded != roll


def replay_sink_policy_gradient(model, data, advantages, *, backward=True, clip=.1):
    count, previous_latents, old_conditional, std = prepare_policy_replay(
        data.latents, data.old_means, data.old_log_prob, advantages, data.valid,
        stationary_std=data.stationary_std, rho=data.rho, chunk_steps=1,
        warmup_steps=data.warmup_steps, clip=clip, gradient_scale=1.,
    )
    with torch.set_grad_enabled(backward):
        means, clamped = model.means(data)
        conditional = torch.cat((means[:1], means[1:] + data.rho *
                                 (previous_latents[1:] - means[:-1])))
        log_prob = joint_log_prob(data.latents, conditional, std, squashed=False)
        ratio = (log_prob - data.old_log_prob).exp()
        loss = -torch.minimum(ratio * advantages,
                              ratio.clamp(1-clip, 1+clip) * advantages)[data.valid].mean()
        if not bool(log_prob.isfinite().all() & ratio.isfinite().all() & loss.isfinite()):
            raise FloatingPointError("nonfinite sink policy objective")
        if backward:
            loss.backward()
            if not bool(model.edge_magnitude.grad.isfinite().all()):
                raise FloatingPointError("nonfinite sink policy gradient")
    with torch.no_grad():
        kl = conditional_kl(old_conditional, conditional, std)[data.valid]
        displacement = (means - data.old_means)[data.valid]
        return dict(
            loss=float(loss.detach()), valid_commands=count, joint_kl_mean=float(kl.mean()),
            joint_kl_p99=float(torch.quantile(kl, .99)), joint_kl_max=float(kl.max()),
            joint_ratio_mean=float(ratio[data.valid].mean()),
            clipped_command_fraction=float(((ratio[data.valid]-1).abs() > clip).float().mean()),
            native_latent_mean_displacement_rms=displacement.square().mean(0).sqrt().tolist(),
            native_clamped_fraction=[float(clamped[data.valid].float().mean()), 0., 0., 0.],
            recurrent_gradient_is_truncated=False, warmup_is_differentiated=True,
            training_replay="native roll sinks; frozen parent activity on fixed observations",
        )


@torch.no_grad()
def verify_sink_recording(model, data):
    """Before optimization, compare against means produced during full-brain flight."""
    means, _ = model.means(data)
    error = (means - data.old_means)[data.valid]
    maximum = error.abs().amax(0)
    if not bool(maximum.isfinite().all()) or float(maximum.max()) > 1e-5:
        raise ValueError("sink replay does not reconstruct the collected action means")
    return dict(latent_mean_max_error=maximum.tolist(),
                latent_mean_rms_error=error.square().mean(0).sqrt().tolist(),
                nonroll_means_fixed_by_disjoint_sink_structure=True)


@torch.no_grad()
def verify_compiled_sink_policy(controller, model, data, *, camera, gate_config):
    """First accepted policy: ordinary full-brain replay of one training pair.

    This is a practical equivalence check, not byte-exactness or new flight data.
    Full-brain FP32 accumulation can differ between batch sizes, so report all
    axis errors and stop only if RMS exceeds 10% of exploration sigma.
    """
    device = model.edge_magnitude.device
    batch = data.select(range(min(2, data.valid.shape[1])), device)
    expected, _ = model.means(batch)
    neural = controller.initial_state(batch.valid.shape[1], device=device,
                                      dtype=batch.latents.dtype)
    observation = batch.observation(0, camera, gate_config)
    for _ in range(batch.warmup_steps):
        _, neural = controller(*observation, neural)
    observed = []
    for time in range(len(batch.latents)):
        native, neural = controller(*batch.observation(time, camera, gate_config), neural)
        observed.append(torch.atanh(native.clamp(-.999999, .999999)))
    error = (torch.stack(observed) - expected)[batch.valid]
    rms, maximum = error.square().mean(0).sqrt(), error.abs().amax(0)
    relative = rms / rms.new_tensor(data.stationary_std)
    if not bool(maximum.isfinite().all()) or float(relative.max()) > .1:
        raise ValueError("compiled full controller differs materially from sink replay")
    return dict(training_episodes=batch.valid.shape[1], latent_mean_rms_error=rms.tolist(),
                latent_mean_max_error=maximum.tolist(), rms_in_stationary_std=relative.tolist(),
                new_flights_collected=0)
