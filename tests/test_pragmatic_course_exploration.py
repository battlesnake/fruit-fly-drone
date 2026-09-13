from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_pragmatic_course_exploration as calibration  # noqa: E402


class StubController:
    def initial_state(self, count, **kwargs):
        return torch.zeros(count, 1, **kwargs)

    def __call__(self, image, attitude, neural):
        # Native mean changes through the actor's own recurrence every frame.
        return neural.expand(-1, 4).tanh(), neural + 0.03


def test_sampler_keeps_native_warmup_then_stationary_and_correlated_physical_commands():
    actor = calibration.ExplorationAuditController(
        StubController(), transformed=True, tau=0.6, scale=1, noise_seed=123, warmup_steps=2,
    )
    expected_generator = torch.Generator().manual_seed(123)
    neural = actor.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    for step in range(2):
        motor, neural = actor(None, None, neural)
        assert torch.allclose(motor, torch.full((2, 4), step * 0.03).tanh())
    assert actor.samples == 0 and actor.previous_latent is None
    residual = torch.zeros(2, 4)
    for step in range(3):
        innovation = torch.randn(2, 4, generator=expected_generator)
        sigma = torch.tensor(calibration.BASE_STD)
        residual = (sigma * innovation if step == 0 else actor.rho * residual
                    + math.sqrt(1 - actor.rho ** 2) * sigma * innovation)
        expected = (torch.full((2, 4), (step + 2) * 0.03) + residual).tanh()
        motor, neural = actor(None, None, neural)
        assert torch.allclose(motor, expected, atol=1e-7)
        assert not torch.equal(motor[0], motor[1])
    assert actor.summary()["command_count_per_episode"] == 3
    assert actor.calls == 5


def test_episode_reset_restarts_rng_and_actor_state_not_at_each_frame():
    actor = calibration.ExplorationAuditController(
        StubController(), transformed=True, tau=0.6, scale=0.5, noise_seed=12, warmup_steps=0,
    )
    trajectories = []
    for _ in range(2):
        neural = actor.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
        commands = []
        for _ in range(3):
            motor, neural = actor(None, None, neural)
            commands.append(motor)
        trajectories.append(torch.stack(commands))
    assert torch.equal(*trajectories)


@pytest.mark.parametrize("transformed", [False, True])
def test_raw_control_and_zero_noise_roundtrip_report_clamps(transformed):
    class BoundaryController(StubController):
        def __call__(self, image, attitude, neural):
            return torch.tensor([[1.0, -1.0, 0.01, -0.02]]), neural
    actor = calibration.ExplorationAuditController(
        BoundaryController(), transformed=transformed, tau=0, scale=0,
        noise_seed=1, warmup_steps=0,
    )
    neural = actor.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    motor, _ = actor(None, None, neural)
    assert motor[0, 0] < 1 if transformed else motor[0, 0] == 1
    summary = actor.summary()
    assert summary["native_clamped_fraction"] == [1, 1, 0, 0]
    assert summary["zero_noise_roundtrip_max_error"][0] > 0
    assert summary["latent_residual_rms"] == [0] * 4
    if not transformed:
        assert summary["motor_perturbation_rms"] == [0] * 4


def test_trace_observer_copies_tensors_and_compares_only_before_either_failure():
    observer = calibration.FlightTrace()
    position, rc, failed = torch.zeros(2, 3), torch.zeros(2, 4), torch.tensor([False, True])
    observer(SimpleNamespace(position=position), rc, failed)
    position.fill_(1)
    rc.fill_(2)
    failed.fill_(True)
    trace = observer.finish()
    assert trace[0].sum() == 0
    changed = (trace[0] + 1, trace[1] + 2, trace[2])
    metrics = calibration.trace_difference(changed, trace)
    assert metrics["common_unfailed_command_samples"] == 1
    assert metrics["position_rms_metres"] == [1] * 3
    assert metrics["rc_difference_rms"] == [2] * 4


@pytest.mark.parametrize("change,eligible", [
    ({}, True),
    ({"ground_contact_rate": 1 / 32}, False),
    ({"invalid_rate": 1 / 32}, False),
    ({"clean_course_success_rate": 5 / 32}, False),
    ({"clean_first_gate_pass_rate": 20 / 32}, False),
    ({"stick_saturation_fraction_mean": 0.051}, False),
])
def test_survival_screen_allows_some_exploratory_loss_but_not_ground(change, eligible):
    baseline = dict(episodes=32, ground_contact_rate=0, invalid_rate=0,
                    clean_course_success_rate=12 / 32, clean_first_gate_pass_rate=28 / 32,
                    stick_saturation_fraction_mean=0)
    metrics = dict(baseline, clean_course_success_rate=6 / 32, clean_first_gate_pass_rate=21 / 32)
    metrics.update(change)
    difference = dict(rc_difference_rms=[0.001] * 4, position_rms_metres=[0.01] * 3)
    assert calibration.survival_screen(metrics, baseline, difference)["eligible"] == eligible


def test_no_measurable_exploration_is_not_eligible():
    baseline = dict(episodes=32, ground_contact_rate=0, invalid_rate=0,
                    clean_course_success_rate=1, clean_first_gate_pass_rate=1,
                    stick_saturation_fraction_mean=0)
    difference = dict(rc_difference_rms=None, position_rms_metres=None)
    assert not calibration.survival_screen(baseline, baseline, difference)["eligible"]


def test_roundtrip_accepts_one_case_variation_but_flags_clipping_and_ground():
    baseline = dict(episodes=32, clean_course_success_rate=12 / 32,
                    clean_first_gate_pass_rate=28 / 32, ground_contact_rate=0, invalid_rate=0)
    metrics = dict(baseline, clean_course_success_rate=11 / 32)
    commands = dict(native_clamped_fraction=[0] * 4, zero_noise_roundtrip_max_error=[1e-8] * 4)
    assert calibration.roundtrip_screen(metrics, baseline, commands)["eligible"]
    assert not calibration.roundtrip_screen(
        dict(metrics, ground_contact_rate=1 / 32), baseline, commands
    )["eligible"]
    commands["native_clamped_fraction"][2] = 0.01
    assert not calibration.roundtrip_screen(metrics, baseline, commands)["eligible"]
