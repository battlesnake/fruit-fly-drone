from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_vertical_motion_derivative_ladder as ladder  # noqa: E402
import audit_vertical_motion_tau_fp64 as tau_fp64  # noqa: E402
import preflight_vertical_motion_commissioning as preflight  # noqa: E402
import preflight_vertical_motion_commissioning_v2 as preflight_v2  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402
import vertical_motion_commissioning_fp64 as commissioning_fp64  # noqa: E402


class _TinyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor([1.0, 0.5]))
        self.bias_offset = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
        self.tau_ratio = torch.nn.Parameter(torch.tensor([1.0, 1.1]))

    def parameter_values(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).detach().cpu().clone()
            for name in ("gain", "bias_offset", "tau_ratio")
        }

    def load_parameter_values(self, values: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, value in values.items():
                getattr(self, name).copy_(value)

    def project_parameters(self) -> None:
        return None


def _seed_optimizer(controller: _TinyController) -> torch.optim.Adam:
    optimizer = training.make_optimizer(controller)
    for parameter in controller.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for group, learning_rate in zip(optimizer.param_groups, (0.003, 0.004, 0.005), strict=True):
        group["lr"] = learning_rate
    return optimizer


def _assert_nested_equal(first: object, second: object) -> None:
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert isinstance(second, dict)
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert isinstance(second, type(first))
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def _training_specs() -> list[dict]:
    path = (
        registration.REPO_ROOT / "artifacts/vertical-motion-commissioning-manifest-v1/manifest.json"
    )
    return commissioning.load_registered_manifest(path)["stimuli"]["splits"]["training"]["specs"]


def test_preflight_protocol_is_disposable_and_training_only() -> None:
    protocol = preflight.protocol_manifest()

    assert protocol["identity_cases"] == list(preflight.IDENTITY_CASES)
    assert protocol["fixed_effective_minibatch_cases"] == list(preflight.MINIBATCH_CASES)
    assert protocol["development_or_acceptance_specs_used"] is False
    assert protocol["development_or_acceptance_pixels_rendered"] is False
    assert protocol["candidate_retained"] is False
    assert protocol["solver"]["neural_state_updates_hz"] == 1600


def test_edge_sequences_preserve_polarity_and_share_terminal() -> None:
    specs = _training_specs()
    for case, polarity in ((0, "ON"), (24, "OFF")):
        sequence = commissioning.render_sequence(specs[case])
        moving = sequence[: registration.MOTION_FRAMES, :2]
        temporal = moving[1:] - moving[:-1]
        if polarity == "ON":
            assert float(temporal.min()) >= 0.0
        else:
            assert float(temporal.max()) <= 0.0
        assert torch.equal(sequence[-1], torch.full_like(sequence[-1], 0.5))
        assert torch.equal(
            sequence[: registration.MOTION_FRAMES, 2],
            sequence[0, 0].expand(registration.MOTION_FRAMES, -1, -1),
        )
        assert torch.equal(
            sequence[: registration.MOTION_FRAMES, 3],
            sequence[0, 1].expand(registration.MOTION_FRAMES, -1, -1),
        )


def test_literal_reverse_uses_last_motion_frame_as_stationary_baseline() -> None:
    spec = _training_specs()[0]
    normal = commissioning.render_sequence(spec)
    reverse = commissioning.render_sequence(spec, reverse=True)

    assert torch.equal(
        reverse[: registration.MOTION_FRAMES, :2],
        normal[: registration.MOTION_FRAMES, :2].flip(0),
    )
    assert torch.equal(
        reverse[: registration.MOTION_FRAMES, 2],
        normal[registration.MOTION_FRAMES - 1, 0].expand(registration.MOTION_FRAMES, -1, -1),
    )
    assert torch.equal(reverse[-1], normal[-1])


def test_texture_render_is_deterministic_bounded_and_periodic() -> None:
    spec = _training_specs()[48]
    first = commissioning.render_sequence(spec)
    second = commissioning.render_sequence(spec)

    assert torch.equal(first, second)
    assert float(first.min()) >= 0.18 - 1.0e-6
    assert float(first.max()) <= 0.82 + 1.0e-6
    speed = spec["speed_pixels_per_frame"]
    assert torch.equal(first[1, 0], torch.roll(first[0, 0], speed, dims=0))


def test_fd_probes_cover_one_gain_bias_and_tau_per_pathway() -> None:
    assert preflight.PROBES == (
        ("T4_gain_Mi1_to_T4c", "gain", 0),
        ("T5_gain_Tm1_to_T5c", "gain", 8),
        ("T4c_bias", "bias_offset", 0),
        ("T5c_bias", "bias_offset", 2),
        ("T4c_tau", "tau_ratio", 0),
        ("T5c_tau", "tau_ratio", 2),
    )


def test_registered_loss_accepts_balanced_pathway_shapes() -> None:
    specs = [
        {"family": "polarity_preserving_edge", "polarity": "ON"},
        {"family": "polarity_preserving_edge", "polarity": "OFF"},
        {"family": "band_limited_texture", "polarity": "mixed"},
        {"family": "band_limited_texture", "polarity": "mixed"},
    ]
    anatomy = {
        "target_indices": np.arange(4, dtype=np.int64),
        "target_subtypes": np.arange(4, dtype=np.int64),
    }
    correct = torch.tensor([[-0.5, 0.5, -0.5, 0.5], [0.5, -0.5, 0.5, -0.5]])
    reversed_direction = -correct

    def response(values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "integrated_response": values.clone(),
            "integrated_static": torch.zeros_like(values),
            "terminal_response": values.clone(),
            "terminal_static": torch.zeros_like(values),
            "terminal_motor": torch.zeros(4, 4),
        }

    normal = [response(correct) for _ in specs]
    reverse = [response(reversed_direction) for _ in specs]
    references = commissioning.source_references(specs, normal, reverse, anatomy)
    controller = SimpleNamespace(
        gain=torch.ones(16),
        bias_offset=torch.zeros(4),
        tau_ratio=torch.ones(4),
    )

    loss, components = commissioning.commissioning_loss(
        specs, normal, reverse, references, anatomy, controller
    )

    assert torch.isfinite(loss)
    assert set(components) == {
        "direction",
        "bias",
        "stationary",
        "dsi",
        "reverse",
        "activity",
        "regularization",
    }
    assert float(components["stationary"]) == 0.0
    assert float(components["direction"]) == 0.0
    assert float(components["reverse"]) == 0.0


def test_v2_changes_only_tau_arithmetic_protocol() -> None:
    v1_protocol = preflight.protocol_manifest()
    v2_protocol = preflight_v2.protocol_manifest()

    ignored = {
        "experiment",
        "protocol_commit",
        "locked_v1_report_sha256",
        "locked_v1_classification",
        "numerical_correction",
        "scientific_or_threshold_changes_from_v1",
        "v1_frozen_source_normalizations_sha256_required",
        "v1_rendered_training_subset_sha256_required",
    }
    assert {key: value for key, value in v2_protocol.items() if key not in ignored} == {
        key: value
        for key, value in v1_protocol.items()
        if key not in {"experiment", "protocol_commit"}
    }
    assert v2_protocol["scientific_or_threshold_changes_from_v1"] == []
    assert v2_protocol["finite_difference"]["central_step"] == 1.0e-3
    assert v2_protocol["finite_difference"]["symmetric_relative_error_maximum"] == 0.02


def test_derivative_ladder_uses_adjacent_resolved_scales() -> None:
    assert ladder.STEPS == (0.008, 0.004, 0.002, 0.001, 0.0005, 0.00025)
    rows = [{"step": step, "pass": step in (0.004, 0.002)} for step in ladder.STEPS]

    qualification = ladder.qualify_probe(rows)

    assert qualification["pass"] is True
    assert qualification["passing_adjacent_step_pairs"] == [[0.004, 0.002]]


def test_resolution_floor_never_collapses_below_eight_float32_ulps() -> None:
    result = ladder.loss_resolution_floor([1.0, 1.0, 1.0])

    assert result["repeat_range"] == 0.0
    assert result["loss_resolution_floor"] == 8.0 * float(np.spacing(np.float32(1.0)))


def test_fp64_tau_protocol_is_narrow_disposable_and_training_only() -> None:
    protocol = tau_fp64.protocol_manifest()

    assert tau_fp64.PROBES == (("T4c_tau", 0), ("T5c_tau", 2))
    assert tau_fp64.STEPS == (0.008, 0.004, 0.002, 0.001)
    assert (
        protocol["precision_reference"]["sensory_recurrence_state_alpha_response_and_loss_dtype"]
        == "float64"
    )
    assert protocol["development_or_acceptance_specs_used"] is False
    assert protocol["development_or_acceptance_pixels_rendered"] is False
    assert protocol["candidate_retained"] is False
    assert protocol["hover_gate_or_promotion_authorized"] is False


def test_fp64_resolution_floor_uses_float64_ulps() -> None:
    result = tau_fp64.loss_resolution_floor([1.0, 1.0, 1.0])

    assert result["repeat_range"] == 0.0
    assert result["loss_resolution_floor"] == 8.0 * float(np.spacing(np.float64(1.0)))


def test_fp64_tau_qualification_requires_adjacent_steps() -> None:
    rows = [{"step": step, "pass": step in (0.004, 0.002)} for step in tau_fp64.STEPS]

    qualification = tau_fp64.qualify_probe(rows)

    assert qualification["pass"] is True
    assert qualification["passing_adjacent_step_pairs"] == [[0.004, 0.002]]


def test_fp64_reference_promotion_roundtrips_exactly() -> None:
    source = {
        "floating": torch.tensor([0.1, -0.25], dtype=torch.float32),
        "indices": torch.tensor([1, 3], dtype=torch.int64),
        "nested": [torch.tensor(0.001, dtype=torch.float32), "fixed"],
    }

    promoted = commissioning_fp64.promote_tree_fp64(source)

    assert promoted["floating"].dtype == torch.float64
    assert promoted["indices"].dtype == torch.int64
    assert commissioning_fp64.promotion_roundtrip_maximum_difference(source, promoted) == 0.0


def test_fp64_fixed_state_tau_derivative_matches_closed_form() -> None:
    result = commissioning_fp64.fixed_state_tau_derivative(
        torch.tensor([0.021], dtype=torch.float64),
        1.0 / 1600.0,
    )

    assert result["finite_nonzero"] is True
    assert result["clamp_inactive"] is True
    assert result["sign_matches"] is True
    assert result["symmetric_relative_error"] <= tau_fp64.ONE_STEP_RELATIVE_ERROR_LIMIT


def test_training_batches_are_balanced_deterministic_partitions() -> None:
    manifest = commissioning.load_registered_manifest(
        registration.REPO_ROOT / "artifacts/vertical-motion-commissioning-manifest-v1/manifest.json"
    )
    expected = {
        "training": (24, (0, 24, 48, 49), (23, 47, 94, 95)),
        "development": (12, (0, 12, 24, 25), (11, 23, 46, 47)),
        "acceptance": (32, (0, 32, 64, 65), (31, 63, 126, 127)),
    }
    for split, (count, first, last) in expected.items():
        specs = manifest["stimuli"]["splits"][split]["specs"]
        batches = training.balanced_batches(specs, split=split)

        assert len(batches) == count
        assert batches[0] == first
        assert batches[-1] == last
        assert sorted(case for batch in batches for case in batch) == list(range(len(specs)))


def test_mandatory_training_gate_requires_loss_and_all_strata() -> None:
    baseline = {"loss": 1.0}
    passing = {"loss": 0.75}
    baseline_strata = {name: 1.0 for name in ("ON_down", "ON_up", "OFF_down", "OFF_up")}
    improved_strata = {name: 0.9 for name in baseline_strata}

    assert training.mandatory_training_decision(
        baseline, passing, baseline_strata, improved_strata
    )["pass"]

    unimproved = dict(improved_strata)
    unimproved["OFF_up"] = 1.0
    assert not training.mandatory_training_decision(baseline, passing, baseline_strata, unimproved)[
        "pass"
    ]
    assert not training.mandatory_training_decision(
        baseline, {"loss": 0.750001}, baseline_strata, improved_strata
    )["pass"]


def test_qualification_edges_cannot_be_masked_by_textures() -> None:
    specs = [
        {"family": "polarity_preserving_edge", "polarity": "ON"},
        {"family": "polarity_preserving_edge", "polarity": "OFF"},
        {"family": "band_limited_texture", "polarity": "mixed"},
        {"family": "band_limited_texture", "polarity": "mixed"},
    ]
    anatomy = {"target_subtypes": np.arange(4, dtype=np.int64)}
    correct = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    wrong = -correct

    def response(values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "integrated_response": values.clone(),
            "integrated_static": torch.zeros_like(values),
            "terminal_response": values.clone(),
            "terminal_static": torch.zeros_like(values),
            "terminal_motor": torch.zeros(4, 4),
        }

    normal = {case: response(wrong if case < 2 else correct) for case in range(4)}
    reverse = {case: response(-normal[case]["integrated_response"]) for case in range(4)}

    result = training.qualification_metrics(specs, {"normal": normal, "reverse": reverse}, anatomy)

    for window in commissioning.WINDOWS:
        assert result["vertical_sign"][window]["T4"]["cases"] == 1
        assert result["vertical_sign"][window]["T5"]["cases"] == 1
        assert result["vertical_sign"][window]["T4"]["down_correct_fraction"] == 0.0
        assert result["vertical_sign"][window]["T5"]["up_correct_fraction"] == 0.0


def test_speed_three_texture_pass_cannot_mask_other_acceptance_textures() -> None:
    specs = [
        {
            "family": "band_limited_texture",
            "polarity": "mixed",
            "speed_pixels_per_frame": speed,
        }
        for speed in (1, 3)
    ]
    anatomy = {"target_subtypes": np.arange(4, dtype=np.int64)}
    correct = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    wrong = -correct

    def response(values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "integrated_response": values.clone(),
            "integrated_static": torch.zeros_like(values),
            "terminal_response": values.clone(),
            "terminal_static": torch.zeros_like(values),
            "terminal_motor": torch.zeros(4, 4),
        }

    bank = {
        "normal": {0: response(wrong), 1: response(correct)},
        "reverse": {0: response(-wrong), 1: response(-correct)},
    }

    assert training.novel_texture_speed_metrics(specs, bank, anatomy)["pass"] is True
    assert training.texture_direction_metrics(specs, bank, anatomy)["pass"] is False


def test_optimizer_replay_exception_restores_parameters_and_adam(monkeypatch) -> None:
    controller = _TinyController()
    optimizer = _seed_optimizer(controller)
    parameters_before = controller.parameter_values()
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    calls = 0

    def failing_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("injected replay failure")
        loss = sum(parameter.square().sum() for parameter in controller.parameters())
        return loss, {"synthetic": loss * 0.0}

    monkeypatch.setattr(training, "batch_loss", failing_loss)
    with pytest.raises(RuntimeError, match="injected replay failure"):
        training.optimizer_proposal(
            controller,
            optimizer,
            [],
            {},
            {},
            (0, 1, 2, 3),
            {},
            device=torch.device("cpu"),
        )

    _assert_nested_equal(controller.parameter_values(), parameters_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)


def test_rejected_proposal_restores_scaled_learning_rates_and_adam(monkeypatch) -> None:
    controller = _TinyController()
    optimizer = _seed_optimizer(controller)
    parameters_before = controller.parameter_values()
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    calls = 0

    def worsening_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        differentiable = sum(parameter.square().sum() for parameter in controller.parameters())
        if calls == 1:
            loss = differentiable
        else:
            loss = differentiable * 0.0 + 1.0e6
        return loss, {"synthetic": loss * 0.0}

    monkeypatch.setattr(training, "batch_loss", worsening_loss)
    report = training.optimizer_proposal(
        controller,
        optimizer,
        [],
        {},
        {},
        (0, 1, 2, 3),
        {},
        device=torch.device("cpu"),
    )

    assert report["accepted"] is False
    assert report["optimizer_steps_before"] == report["optimizer_steps_after"]
    _assert_nested_equal(controller.parameter_values(), parameters_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)


def test_accepted_proposal_advances_each_adam_counter_once(monkeypatch) -> None:
    controller = _TinyController()
    optimizer = _seed_optimizer(controller)
    calls = 0

    def improving_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        differentiable = sum(parameter.square().sum() for parameter in controller.parameters())
        loss = differentiable if calls == 1 else differentiable * 0.0
        return loss, {"synthetic": loss * 0.0}

    monkeypatch.setattr(training, "batch_loss", improving_loss)
    report = training.optimizer_proposal(
        controller,
        optimizer,
        [],
        {},
        {},
        (0, 1, 2, 3),
        {},
        device=torch.device("cpu"),
    )

    assert report["accepted"] is True
    assert all(
        report["optimizer_steps_after"][name] == report["optimizer_steps_before"][name] + 1
        for name in report["optimizer_steps_before"]
    )


def test_resume_interruptions_fail_closed_without_retry() -> None:
    state = {
        "proposal_in_flight": 7,
        "scheduled_evaluation_started": [],
        "development_started": [],
        "development_history": [],
    }
    assert training.resume_interruption(state)[1] == "vertical_motion_training_proposal_interrupted"

    state["proposal_in_flight"] = None
    state["scheduled_evaluation_started"] = [25]
    assert (
        training.resume_interruption(state)[1]
        == "vertical_motion_training_scheduled_evaluation_interrupted"
    )

    state["development_started"] = [25]
    assert (
        training.resume_interruption(state)[1] == "vertical_motion_training_development_interrupted"
    )


def test_output_directory_lock_rejects_concurrent_owner(tmp_path) -> None:
    first = training.acquire_run_lock(tmp_path)
    try:
        with pytest.raises(SystemExit, match="another vertical-motion training process"):
            training.acquire_run_lock(tmp_path)
    finally:
        first.close()


def test_acceptance_requires_all_exact_scheduled_events() -> None:
    history = [
        {"proposal": proposal, "finite_gradients": True}
        for proposal in range(1, training.MAX_PROPOSALS + 1)
    ]
    scheduled = list(training.SCHEDULED_EVALUATIONS)
    evaluations = [
        {
            "proposal": proposal,
            "loss": 1.0,
            "numerical": {"pass": True, "rms": 0.0},
            **({"mandatory_decision": {"pass": True}} if proposal == 25 else {}),
        }
        for proposal in scheduled
    ]
    development = [
        {"proposal": proposal, "loss": {"loss": 1.0}, "qualification": {"pass": False}}
        for proposal in scheduled
    ]
    state = {
        "proposal_completed": training.MAX_PROPOSALS,
        "proposal_in_flight": None,
        "accepted_proposals": 80,
        "rejected_proposals": 20,
        "proposal_history": history,
        "scheduled_evaluation_started": scheduled,
        "training_evaluations": evaluations,
        "development_started": scheduled,
        "development_history": development,
    }

    assert training.acceptance_prerequisite_errors(state) == []

    state["training_evaluations"] = [item for item in evaluations if item["proposal"] != 50]
    errors = training.acceptance_prerequisite_errors(state)
    assert any("training evaluations" in error for error in errors)
    assert any("numerical gates" in error for error in errors)
