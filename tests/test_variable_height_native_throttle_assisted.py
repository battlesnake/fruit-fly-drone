from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_native_throttle_optimizer as optimizer_audit  # noqa: E402
import train_variable_height_native_throttle_assisted as train  # noqa: E402
import train_variable_height_native_throttle_assisted_beta1_zero as beta1_zero  # noqa: E402
import train_variable_height_native_throttle_assisted_beta1_zero_ladder as ladder  # noqa: E402

from flydrone.hover import HoverConfig  # noqa: E402


def test_protocol_locks_assisted_boundary_data_and_claim() -> None:
    protocol = train.protocol_manifest()

    assert protocol["protocol_commit"] == "6d2c8c2"
    assert protocol["actor"]["privileged_inputs"] == []
    assert protocol["actor"]["physical_training_ownership"] == {
        "teacher": ["roll", "pitch", "yaw"],
        "native_fly": ["throttle"],
    }
    assert protocol["actor"]["motor_merge_before_foreleg_stick_plant"] is True
    assert protocol["parameters"]["rpy_source_constraints"] is False
    assert protocol["parameters"]["adam_betas"] == [0.9, 0.999]
    assert protocol["curriculum"]["detached_burn_in_fixed_across_gradient_fd_and_trials"]
    assert protocol["optimizer_transaction"]["one_gradient_and_adam_call_per_attempt"]
    assert protocol["optimizer_transaction"]["numerical_failure_is_terminal"]
    assert protocol["optimizer_transaction"]["finite_no_scale_failure_advances_sampling_rng"]
    assert protocol["final"]["final_data_generated_only_at_update_100"]
    assert protocol["passing_authorizes"] == "native attitude reintegration experiment only"
    assert protocol["full_native_hover"] is False
    assert protocol["promotion"] is False


def test_beta1_zero_entry_point_changes_only_registered_optimizer_identity() -> None:
    original = (train.EXPERIMENT, train.PROTOCOL_COMMIT, train.OPTIMIZER_BETAS)
    expected = copy.deepcopy(train.protocol_manifest())
    expected["experiment"] = beta1_zero.EXPERIMENT
    expected["protocol_commit"] = beta1_zero.PROTOCOL_COMMIT
    expected["parameters"]["adam_betas"] = list(beta1_zero.OPTIMIZER_BETAS)
    try:
        beta1_zero.configure_protocol()
        assert train.EXPERIMENT == "variable-height-native-throttle-assisted-beta1-zero-v1"
        assert train.PROTOCOL_COMMIT == "ddd3c5c"
        assert train.protocol_manifest() == expected
        controller = _ToyController()
        optimizer = train._make_optimizer(controller)
        assert all(group["betas"] == (0.0, 0.999) for group in optimizer.param_groups)
    finally:
        train.EXPERIMENT, train.PROTOCOL_COMMIT, train.OPTIMIZER_BETAS = original


def test_ladder_entry_point_changes_only_registered_derivative_control() -> None:
    names = (
        "EXPERIMENT",
        "PROTOCOL_COMMIT",
        "OPTIMIZER_BETAS",
        "DERIVATIVE_PROBE_SCALES",
        "DERIVATIVE_BASELINE_REPEATS",
        "DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES",
        "DERIVATIVE_REPLAY_NOISE_MULTIPLIER",
        "DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE",
        "AUTHORIZING_TAYLOR_AUDIT_SHA256",
    )
    original = {name: getattr(train, name) for name in names}
    try:
        beta1_zero.configure_protocol()
        expected = copy.deepcopy(train.protocol_manifest())
        expected["experiment"] = ladder.EXPERIMENT
        expected["protocol_commit"] = ladder.PROTOCOL_COMMIT
        transaction = expected["optimizer_transaction"]
        transaction["derivative_probe_scales"] = list(ladder.DERIVATIVE_PROBE_SCALES)
        transaction["derivative_baseline_repeats"] = ladder.DERIVATIVE_BASELINE_REPEATS
        transaction["derivative_required_consecutive_passes"] = (
            ladder.DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES
        )
        transaction["derivative_replay_noise_multiplier"] = (
            ladder.DERIVATIVE_REPLAY_NOISE_MULTIPLIER
        )
        transaction["derivative_minimum_objective_change"] = (
            ladder.DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE
        )
        transaction["authorizing_taylor_audit_sha256"] = (
            ladder.EXPECTED_TAYLOR_AUDIT_SHA256
        )

        ladder.configure_protocol()

        assert train.protocol_manifest() == expected
        assert train.derivative_probe_protocol_manifest() == {
            "scales": list(ladder.DERIVATIVE_PROBE_SCALES),
            "baseline_repeats": 3,
            "required_consecutive_passes": 2,
            "replay_noise_multiplier": 10.0,
            "minimum_objective_change": 1.0e-8,
            "authorizing_taylor_audit_sha256": ladder.EXPECTED_TAYLOR_AUDIT_SHA256,
        }
    finally:
        for name, value in original.items():
            setattr(train, name, value)


def test_ladder_requires_exact_authorizing_taylor_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.json"
    with pytest.raises(SystemExit, match="missing authorizing"):
        ladder.validate_authorizing_audit(report)

    report.write_text('{"pass": true}\n')
    expected = train.responsibility.file_sha256(report)
    monkeypatch.setattr(ladder, "EXPECTED_TAYLOR_AUDIT_SHA256", expected)
    ladder.validate_authorizing_audit(report)

    report.write_text('{"pass": false}\n')
    with pytest.raises(SystemExit, match="hash mismatch"):
        ladder.validate_authorizing_audit(report)


@pytest.mark.parametrize("split", ["train", "held_out_marker", "held_out_combination"])
def test_hover_case_support_is_explicit_and_balanced(split: str) -> None:
    train.responsibility.seed_everything(17)
    cases = train.sample_hover_cases(
        32,
        split=split,
        device=torch.device("cpu"),
        config=HoverConfig(),
    )

    assert train.case_support_decision(cases) == {"pass": True, "reasons": []}
    assert [(cases["step_code"] == value).sum().item() for value in (0, 1, -1)] == [
        16,
        8,
        8,
    ]
    assert [
        (cases["impulse"] == 0).sum().item(),
        (cases["impulse"] > 0).sum().item(),
        (cases["impulse"] < 0).sum().item(),
    ] == [16, 8, 8]


def test_training_marker_steps_do_not_enter_held_out_support() -> None:
    train.responsibility.seed_everything(23)
    cases = train.sample_hover_cases(
        256,
        split="train",
        device=torch.device("cpu"),
        config=HoverConfig(),
    )
    low, high = train.HELD_OUT_ABSOLUTE_MARKER_BAND

    for values in (cases["initial_marker"], cases["final_marker"]):
        assert not bool(((values >= low) & (values <= high)).any())
    assert torch.equal(cases["camera_family"], cases["marker_family"])


def test_held_out_marker_steps_cross_into_held_out_target_only() -> None:
    train.responsibility.seed_everything(29)
    cases = train.sample_hover_cases(
        64,
        split="held_out_marker",
        device=torch.device("cpu"),
        config=HoverConfig(),
    )
    stepped = cases["step_code"] != 0
    low, high = train.HELD_OUT_ABSOLUTE_MARKER_BAND

    assert bool(
        ((cases["final_marker"][stepped] >= low) & (cases["final_marker"][stepped] <= high)).all()
    )
    assert not bool(
        (
            (cases["initial_marker"][stepped] >= low) & (cases["initial_marker"][stepped] <= high)
        ).any()
    )


def test_motor_merge_keeps_only_native_throttle() -> None:
    teacher = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    native = -teacher

    merged = train.merge_teacher_attitude_native_throttle(teacher, native)

    assert torch.equal(merged[:, :3], teacher[:, :3])
    assert torch.equal(merged[:, 3], native[:, 3])


def test_update_sampling_is_seeded_and_obeys_mixture() -> None:
    first = torch.Generator().manual_seed(train.OPTIMIZER_SAMPLING_SEED)
    second = torch.Generator().manual_seed(train.OPTIMIZER_SAMPLING_SEED)

    teacher_only = train.sample_update_spec(first, block_index=0, motion_cases=24)
    repeated = train.sample_update_spec(second, block_index=0, motion_cases=24)
    mixed = train.sample_update_spec(first, block_index=1, motion_cases=24)

    assert teacher_only == repeated
    assert len(teacher_only["teacher_indices"]) == 8
    assert teacher_only["student_indices"] == []
    assert len(mixed["teacher_indices"]) == len(mixed["student_indices"]) == 4
    assert len(mixed["motion_indices"]) == 8
    assert 0 <= mixed["window_start"] <= train.EPISODE_STEPS - train.TRAIN_WINDOW_STEPS


def test_motion_bank_balances_registered_horizons_and_styles() -> None:
    bank = train.build_motion_bank(
        seed=31,
        cases=24,
        height_amplitudes=train.TRAIN_MOTION_HEIGHT_AMPLITUDES,
        speed_amplitudes=train.TRAIN_MOTION_SPEED_AMPLITUDES,
        held_out_styles=False,
        device=torch.device("cpu"),
        config=HoverConfig(),
    )

    assert [(bank["horizon"] == value).sum().item() for value in (15, 20, 25)] == [8, 8, 8]
    assert torch.equal(bank["scene"]["wall_style"], bank["scene"]["floor_style"])
    assert [(bank["height_sign"] == value).sum().item() for value in (-1, 1)] == [12, 12]
    contrasts = train.teacher_motion_contrasts(bank, config=HoverConfig())
    assert contrasts.shape == (24,)
    assert bool(torch.isfinite(contrasts).all())
    assert bool((contrasts.abs() > 1.0e-6).all())


def test_objective_scales_use_teacher_correction_not_steady_collective() -> None:
    config = HoverConfig()
    nominal_rc = torch.zeros(1, 4)
    nominal_rc[:, 3] = 1.0 / config.thrust_to_weight
    nominal = float(train.motor_target_for_rc(nominal_rc, config)[0, 3])
    teacher_bank = {
        "teacher_motor": torch.tensor([[[0.0, 0.0, 0.0, nominal + 0.02]]]),
        "eligible": torch.ones(1, 1, dtype=torch.bool),
    }
    motion_bank = train.build_motion_bank(
        seed=37,
        cases=6,
        height_amplitudes=train.TRAIN_MOTION_HEIGHT_AMPLITUDES,
        speed_amplitudes=train.TRAIN_MOTION_SPEED_AMPLITUDES,
        held_out_styles=False,
        device=torch.device("cpu"),
        config=config,
    )

    scales = train.frozen_objective_scales(teacher_bank, motion_bank, config=config)

    assert scales["dense"] == pytest.approx(0.02)
    assert scales["dense"] != pytest.approx(nominal + 0.02)


def _hover_report(success: float) -> dict[str, object]:
    return {"success_fraction": success, "all_metrics_finite": True}


def _motion_report(sign: float, gain: float = 1.0) -> dict[str, object]:
    return {
        "correct_sign_fraction": sign,
        "teacher_aligned_gain": gain,
        "endpoint_image_difference_max": 0.0,
        "all_metrics_finite": True,
        "all_recurrent_states_and_outputs_finite": True,
    }


def test_midpoint_and_final_decisions_keep_assisted_claim_strict() -> None:
    midpoint = train.midpoint_decision(_hover_report(0.50), _motion_report(0.50))
    assert midpoint["pass"] is True
    assert train.midpoint_decision(_hover_report(0.49), _motion_report(1.0))["pass"] is False

    frozen = {
        "success_fraction": 0.2,
        "all_metrics_finite": True,
        "all_frozen_frames_captured": True,
    }
    passing = train.final_decision(
        _hover_report(0.90), frozen, _motion_report(0.90, 0.5), (0.01, 0.5)
    )
    assert passing["pass"] is True
    failing = train.final_decision(
        _hover_report(0.90), frozen, _motion_report(0.90, 0.49), (0.0, 0.5)
    )
    assert failing["pass"] is False
    assert len(failing["reasons"]) == 2


def test_paired_bootstrap_is_deterministic() -> None:
    differences = torch.tensor([1.0] * 24 + [0.0] * 8)

    first = train.paired_bootstrap_mean_interval(differences, seed=41, samples=1024)
    second = train.paired_bootstrap_mean_interval(differences, seed=41, samples=1024)

    assert first == second
    assert first[0] > 0.0


def test_motion_contrast_scores_one_real_pair_without_height_averaging() -> None:
    assert train.velocity_pair_contrast(torch.tensor([-0.3, 0.2])) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="exactly one opposite-velocity pair"):
        train.velocity_pair_contrast(torch.tensor([-0.3, 0.2, 0.7, -0.4]))


def test_raw_gradient_is_preserved_before_adam_only_clipping() -> None:
    controller = SimpleNamespace(
        edge_magnitude=torch.nn.Parameter(torch.zeros(4)),
        bias=torch.nn.Parameter(torch.zeros(3)),
        raw_time_constant=torch.nn.Parameter(torch.zeros(2)),
    )
    originals = {}
    for index, name in enumerate(train.PARAMETER_FAMILIES, start=1):
        parameter = getattr(controller, name)
        parameter.grad = torch.full_like(parameter, 10.0 * index)
        originals[name] = parameter.grad.clone()

    raw, norm_before = train.clone_raw_gradients_and_clip(controller, maximum_norm=1.0)

    assert float(norm_before) > 10.0
    for name in train.PARAMETER_FAMILIES:
        assert torch.equal(raw[name], originals[name])
    clipped_norm = torch.linalg.vector_norm(
        torch.cat([getattr(controller, name).grad.flatten() for name in train.PARAMETER_FAMILIES])
    )
    assert float(clipped_norm) == pytest.approx(1.0, abs=1.0e-6)


def _attempt_state(*, accepted_updates: int, consecutive_rejections: int = 0) -> dict[str, object]:
    return {
        "accepted_updates": accepted_updates,
        "attempted_updates": accepted_updates,
        "consecutive_rejections": consecutive_rejections,
        "history": [],
        "attempt_stage": "sampled",
        "pending_sample": {"sample": True},
        "pending_terminal": None,
        "midpoint": None,
        "final": None,
        "development_started": False,
        "final_started": False,
    }


def test_attempt_outcome_persists_midpoint_and_terminal_transitions() -> None:
    midpoint_state = _attempt_state(accepted_updates=49)
    train.record_attempt_outcome(
        midpoint_state,
        {"accepted": True, "fatal_numerical_failure": False},
        attempted_number=50,
        target_update=50,
    )
    assert midpoint_state["development_started"] is True
    assert midpoint_state["pending_terminal"] is None

    fatal_state = _attempt_state(accepted_updates=12, consecutive_rejections=3)
    train.record_attempt_outcome(
        fatal_state,
        {"accepted": False, "fatal_numerical_failure": True},
        attempted_number=18,
        target_update=13,
    )
    assert fatal_state["consecutive_rejections"] == 3
    assert fatal_state["pending_terminal"]["classification"] == ("fatal_numerical_control_failure")

    rejection_state = _attempt_state(accepted_updates=12, consecutive_rejections=4)
    train.record_attempt_outcome(
        rejection_state,
        {"accepted": False, "fatal_numerical_failure": False},
        attempted_number=18,
        target_update=13,
    )
    assert rejection_state["pending_terminal"]["classification"] == (
        "consecutive_ordinary_rejections"
    )


def test_disposable_seed_guard_rejects_registered_seed() -> None:
    train.require_nonformal_seeds(990_983, 990_984, 990_985)
    with pytest.raises(ValueError, match="consume formal seeds"):
        train.require_nonformal_seeds(train.MIDPOINT_HOVER_SEED)


def test_nonfinite_or_invalid_recurrent_trial_is_fatal_not_backtrackable() -> None:
    valid = {
        "objective": 1.0,
        "all_recurrent_states_and_outputs_finite": True,
    }
    assert train.trial_reports_are_finite(valid, valid, 1.0)
    assert not train.trial_reports_are_finite(
        {**valid, "all_recurrent_states_and_outputs_finite": False}, valid, 1.0
    )
    assert not train.trial_reports_are_finite(valid, {**valid, "objective": float("nan")}, 1.0)
    assert not train.trial_reports_are_finite(valid, valid, float("inf"))


def test_resume_validation_rejects_skippable_mandatory_phase() -> None:
    with pytest.raises(SystemExit, match="skip the mandatory midpoint"):
        train._validate_resume_scientific_state(
            {
                "accepted_updates": train.MIDPOINT_UPDATE,
                "attempted_updates": train.MIDPOINT_UPDATE,
                "run_state": "active",
                "pending_terminal": None,
                "midpoint": None,
                "development_started": False,
            }
        )


class _ToyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.full((2,), 0.2))
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.raw_time_constant = torch.nn.Parameter(torch.zeros(2))

    def project_parameters(self) -> None:
        return None


def _probe_record(
    scale: float, *, local: bool, above_noise: bool, numerical: bool = False
) -> dict[str, object]:
    return {
        "scale": scale,
        "local_derivative_agreement": local,
        "objective_change_exceeds_noise_threshold": above_noise,
        "numerical_failure": numerical,
    }


@pytest.mark.parametrize(
    ("records", "classification", "passed"),
    [
        (
            [
                _probe_record(1 / 16, local=False, above_noise=True),
                _probe_record(1 / 32, local=True, above_noise=True),
                _probe_record(1 / 64, local=True, above_noise=True),
            ],
            "adjacent_local_derivative_agreement",
            True,
        ),
        (
            [_probe_record(1 / 16, local=False, above_noise=False, numerical=True)],
            "derivative_probe_numerical_failure",
            False,
        ),
        (
            [
                _probe_record(1 / 16, local=False, above_noise=False),
                _probe_record(1 / 32, local=False, above_noise=False),
            ],
            "derivative_probe_noise_limited_inconclusive",
            False,
        ),
        (
            [
                _probe_record(1 / 16, local=False, above_noise=True),
                _probe_record(1 / 32, local=False, above_noise=True),
            ],
            "derivative_probe_above_noise_nonconvergence",
            False,
        ),
    ],
)
def test_derivative_probe_decision_is_conservative(
    records: list[dict[str, object]], classification: str, passed: bool
) -> None:
    decision = train.derivative_probe_decision(records, required_consecutive_passes=2)
    assert decision["classification"] == classification
    assert decision["pass"] is passed


@pytest.mark.parametrize(
    "classification",
    [
        "derivative_probe_numerical_failure",
        "derivative_probe_noise_limited_inconclusive",
        "derivative_probe_above_noise_nonconvergence",
    ],
)
def test_run_update_attempt_probe_failures_skip_trials_restore_and_propagate_terminal(
    monkeypatch: pytest.MonkeyPatch, classification: str
) -> None:
    controller = _ToyController()
    optimizer = train._make_optimizer(controller)
    current = train._copy_parameters(controller)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    controller_hash = train.audit.semantic_sha256(current)
    optimizer_hash = train.audit.semantic_sha256(optimizer_before)
    calls = {"fixed": 0, "full": 0}

    monkeypatch.setattr(train, "DERIVATIVE_BASELINE_REPEATS", 3)
    monkeypatch.setattr(train, "materialize_dense_sample", lambda *_: {})
    monkeypatch.setattr(
        train,
        "_build_fixed_burns",
        lambda *args, **kwargs: (torch.zeros(1), []),
    )

    def fixed_report(*args, **kwargs):
        calls["fixed"] += 1
        return {
            "objective": 1.0,
            "all_recurrent_states_and_outputs_finite": True,
        }

    def full_report(*args, **kwargs):
        calls["full"] += 1
        return 1.0, {
            "objective": 1.0,
            "all_recurrent_states_and_outputs_finite": True,
        }

    def gradient_report(*args, **kwargs):
        raw_gradients = {}
        for name in train.PARAMETER_FAMILIES:
            parameter = getattr(controller, name)
            parameter.grad = torch.ones_like(parameter)
            raw_gradients[name] = parameter.grad.clone()
        return {
            "objective": 1.0,
            "gradients_finite": True,
            "all_recurrent_states_and_outputs_finite": True,
        }, raw_gradients

    monkeypatch.setattr(train, "_sample_objective_report", fixed_report)
    monkeypatch.setattr(train, "combined_full_prefix_objective", full_report)
    monkeypatch.setattr(train, "accumulate_sample_gradient", gradient_report)
    monkeypatch.setattr(
        train,
        "run_derivative_probe_ladder",
        lambda *args, **kwargs: {"pass": False, "classification": classification},
    )

    result = train.run_update_attempt(
        controller,
        optimizer,
        {},
        None,
        {},
        {"motion_indices": []},
        {"dense": 1.0, "motion": 1.0},
        device=torch.device("cpu"),
    )

    assert result["fatal_numerical_failure"] is True
    assert result["accepted"] is False
    assert result["trials"] == []
    assert result["finite_difference"]["classification"] == classification
    assert result["optimizer_transaction"]["counters_before"] == []
    assert result["optimizer_transaction"]["counters_after"] == [1.0, 1.0, 1.0]
    assert calls == {"fixed": 3, "full": 1}
    assert train.audit.semantic_sha256(train._copy_parameters(controller)) == controller_hash
    assert train.audit.semantic_sha256(optimizer.state_dict()) == optimizer_hash

    state = _attempt_state(accepted_updates=0)
    train.record_attempt_outcome(state, result, attempted_number=1, target_update=1)
    assert state["pending_terminal"]["classification"] == classification


def test_missing_derivative_resume_identity_is_allowed_only_for_legacy_single_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train, "DERIVATIVE_PROBE_SCALES", (train.FINITE_DIFFERENCE_SCALE,))
    monkeypatch.setattr(train, "DERIVATIVE_BASELINE_REPEATS", 1)
    monkeypatch.setattr(train, "DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES", 1)
    monkeypatch.setattr(train, "DERIVATIVE_REPLAY_NOISE_MULTIPLIER", 0.0)
    monkeypatch.setattr(train, "DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE", 0.0)
    monkeypatch.setattr(train, "AUTHORIZING_TAYLOR_AUDIT_SHA256", None)
    train.validate_derivative_probe_resume_identity({})
    with pytest.raises(SystemExit, match="does not match"):
        train.validate_derivative_probe_resume_identity(
            {"derivative_probe_protocol": {"scales": [1 / 32]}}
        )

    monkeypatch.setattr(train, "DERIVATIVE_PROBE_SCALES", ladder.DERIVATIVE_PROBE_SCALES)
    monkeypatch.setattr(train, "DERIVATIVE_BASELINE_REPEATS", 3)
    monkeypatch.setattr(train, "DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES", 2)
    monkeypatch.setattr(train, "DERIVATIVE_REPLAY_NOISE_MULTIPLIER", 10.0)
    monkeypatch.setattr(train, "DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE", 1.0e-8)
    monkeypatch.setattr(
        train, "AUTHORIZING_TAYLOR_AUDIT_SHA256", ladder.EXPECTED_TAYLOR_AUDIT_SHA256
    )
    with pytest.raises(SystemExit, match="missing the registered"):
        train.validate_derivative_probe_resume_identity({})


def test_beta1_zero_removes_opposed_first_moment_direction() -> None:
    controller = _ToyController()
    optimizer = train._make_optimizer(controller)
    for _ in range(18):
        for name in train.PARAMETER_FAMILIES:
            getattr(controller, name).grad = torch.full_like(getattr(controller, name), -1.0)
        optimizer.step()
    current = train._copy_parameters(controller)
    raw = {
        name: torch.full_like(getattr(controller, name), 0.01) for name in train.PARAMETER_FAMILIES
    }

    original = optimizer_audit.materialize_optimizer_direction(
        controller,
        optimizer.state_dict(),
        current,
        raw,
        raw,
        beta1=0.9,
    )
    momentum_free = optimizer_audit.materialize_optimizer_direction(
        controller,
        optimizer.state_dict(),
        current,
        raw,
        raw,
        beta1=0.0,
    )

    assert original["materialized"]["total"] > 0.0
    assert momentum_free["materialized"]["total"] < 0.0
    assert momentum_free["optimizer_counters_before"] == [18.0, 18.0, 18.0]
    assert momentum_free["optimizer_counters_after"] == [19.0, 19.0, 19.0]
    assert momentum_free["moment_mechanism"]["first_moment_replaced"] is True
    assert momentum_free["moment_mechanism"]["second_moment_retained_and_updated"] is True
    for name in train.PARAMETER_FAMILIES:
        assert torch.equal(getattr(controller, name), current[name])
    with pytest.raises(SystemExit, match="missing its terminal transition"):
        train._validate_resume_scientific_state(
            {
                "accepted_updates": 0,
                "attempted_updates": 0,
                "run_state": "stopped",
                "pending_terminal": None,
            }
        )
