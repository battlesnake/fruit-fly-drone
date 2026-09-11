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

import audit_vertical_motion_gradient_attribution as attribution  # noqa: E402
import check_vertical_motion_grouped_execution as checker  # noqa: E402
import preflight_vertical_motion_grouped_execution as preflight  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402
import vertical_motion_grouped_execution as grouped  # noqa: E402


class _TinyRecurrentController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def initial_state(
        self, batch: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(batch, 3, device=device, dtype=dtype)

    def forward(
        self, image: torch.Tensor, attitude: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drive = image.mean(dim=(1, 2, 3)) + attitude.sum(dim=1)
        coefficients = torch.tensor([1.0, -0.5, 0.25], device=image.device)
        next_state = 0.7 * state + self.scale * drive[:, None] * coefficients
        output = next_state.mean(dim=1, keepdim=True).expand(-1, 4)
        return output, next_state


@pytest.fixture
def tiny_execution(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    monkeypatch.setattr(registration, "MOTION_FRAMES", 2)
    monkeypatch.setattr(registration, "TERMINAL_FRAMES", 1)
    monkeypatch.setattr(registration, "PREFIX_FRAMES", 2)
    monkeypatch.setattr(registration, "CNS_SUBSTEPS_PER_FRAME", 2)
    monkeypatch.setattr(registration, "HEIGHT", 2)
    monkeypatch.setattr(registration, "WIDTH", 3)
    generator = torch.Generator().manual_seed(7)
    sequences = {
        (case, reverse): torch.rand(3, 4, 2, 3, generator=generator)
        for case in range(3)
        for reverse in (False, True)
    }
    anatomy = {"target_indices": np.array([0, 2], dtype=np.int64)}
    return sequences, anatomy


def _sum_responses(values: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    return sum(
        (tensor.sum() for value in values for tensor in value.values()),
        start=torch.tensor(0.0),
    )


def test_grouped_case_execution_matches_independent_recurrence_and_gradient(
    tiny_execution: tuple[dict, dict],
) -> None:
    sequences, anatomy = tiny_execution
    sequential_controller = _TinyRecurrentController()
    grouped_controller = _TinyRecurrentController()
    cases = (0, 1, 2)
    sequential = training.evaluate_cases(
        sequential_controller,
        sequences,
        anatomy,
        cases,
        device=torch.device("cpu"),
        checkpoint_frames=False,
    )
    batched = grouped.evaluate_cases_batched(
        grouped_controller,
        sequences,
        anatomy,
        cases,
        device=torch.device("cpu"),
        checkpoint_frames=False,
    )

    for sequential_mode, batched_mode in zip(sequential, batched, strict=True):
        for sequential_value, batched_value in zip(sequential_mode, batched_mode, strict=True):
            assert set(sequential_value) == set(batched_value)
            for name in sequential_value:
                assert torch.allclose(
                    sequential_value[name], batched_value[name], atol=1.0e-7, rtol=0.0
                )
    sequential_loss = _sum_responses([*sequential[0], *sequential[1]])
    batched_loss = _sum_responses([*batched[0], *batched[1]])
    sequential_gradient = torch.autograd.grad(sequential_loss, sequential_controller.scale)[0]
    batched_gradient = torch.autograd.grad(batched_loss, grouped_controller.scale)[0]
    assert torch.allclose(sequential_gradient, batched_gradient, atol=1.0e-6, rtol=0.0)


def test_response_bank_blocks_preserve_case_identity(
    tiny_execution: tuple[dict, dict],
) -> None:
    sequences, anatomy = tiny_execution
    controller = _TinyRecurrentController()
    cases = (0, 1, 2)
    with torch.inference_mode():
        normal, reverse = training.evaluate_cases(
            controller,
            sequences,
            anatomy,
            cases,
            device=torch.device("cpu"),
            checkpoint_frames=False,
        )
    sequential = training.response_bank_cpu(normal, reverse, cases)
    blocked = grouped.response_bank_batched(
        controller,
        sequences,
        anatomy,
        cases,
        case_block_size=2,
        device=torch.device("cpu"),
    )

    assert grouped.maximum_response_difference(sequential, blocked) <= 1.0e-7


def test_response_difference_rejects_nonfinite_tensors() -> None:
    finite = {"response": torch.tensor([0.0, 1.0])}
    nonfinite = {"response": torch.tensor([0.0, float("nan")])}

    assert grouped.maximum_response_difference(finite, nonfinite) == float("inf")


def test_restoration_transaction_runs_finally_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = SimpleNamespace(source=torch.nn.Linear(1, 1))
    calls = []

    def restore(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(preflight, "_restore_and_check", restore)
    source_fingerprint = preflight.commissioning.semantic_sha256(controller.source.state_dict())

    with pytest.raises(RuntimeError, match="injected"):
        preflight._run_in_restoration_transaction(
            lambda: (_ for _ in ()).throw(RuntimeError("injected")),
            controller,
            object(),
            {},
            {},
            "archive",
            source_fingerprint,
        )

    assert len(calls) == 2


def test_restoration_oom_is_global_control_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = SimpleNamespace(source=torch.nn.Linear(1, 1))
    calls = 0

    def restore(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.cuda.OutOfMemoryError("restoration")

    monkeypatch.setattr(preflight, "_restore_and_check", restore)
    source_fingerprint = preflight.commissioning.semantic_sha256(controller.source.state_dict())

    with pytest.raises(preflight.ControlFailure, match="restoration or verification"):
        preflight._run_in_restoration_transaction(
            lambda: None,
            controller,
            object(),
            {},
            {},
            "archive",
            source_fingerprint,
        )


def test_direct_restoration_oom_is_wrapped_as_control_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_restore_and_check_unwrapped",
        lambda *args: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("direct restoration")),
    )

    with pytest.raises(preflight.ControlFailure, match="restoration or verification"):
        preflight._restore_and_check(object(), object(), {}, {}, "archive")


def test_witness_restoration_oom_is_global_control_failure() -> None:
    controller = SimpleNamespace(
        load_parameter_values=lambda values: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("witness restoration")
        )
    )

    with pytest.raises(preflight.ControlFailure, match="witness parameter restoration"):
        preflight._restore_witness_parameters(controller, {})


def test_state_fingerprint_oom_is_global_control_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = SimpleNamespace(
        parameter_values=lambda: {},
        source=SimpleNamespace(state_dict=lambda: {}),
    )
    optimizer = SimpleNamespace(state_dict=lambda: {})
    monkeypatch.setattr(
        preflight.commissioning,
        "semantic_sha256",
        lambda value: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("fingerprint")),
    )

    with pytest.raises(preflight.ControlFailure, match="fingerprint verification"):
        preflight._state_control(
            controller,
            optimizer,
            "archive",
            "optimizer",
            "source",
            require_archived_parameters=True,
        )


def _gradient_comparison() -> dict[str, float]:
    return {"maximum_absolute_difference": 0.0, "relative_l2_error": 0.0}


def _state_control(*, archived: bool) -> dict:
    return {
        "archived_parameters_required": archived,
        "archived_state_fingerprint": "a" * 64 if archived else "d" * 64,
        "optimizer_state_fingerprint": "b" * 64,
        "source_state_fingerprint": "c" * 64,
        "adam_steps": {
            name: attribution.ARCHIVED_PROPOSALS for name in attribution.PARAMETER_NAMES
        },
        "pass": True,
    }


def _passing_size_result() -> dict:
    baseline_strata = {
        name: value
        for name, value in zip(attribution.STRATUM_NAMES, (0.2, 0.1, 0.1, 0.1), strict=True)
    }
    candidate_strata = dict(baseline_strata)
    candidate_strata["ON_down"] = 0.197
    decision = attribution.trial_decision(
        1.0, baseline_strata, 0.998, candidate_strata, finite=True
    )
    return {
        "finite": True,
        "grouped_metrics": {"loss": 1.0, "strata": baseline_strata},
        "grouped_metric_comparison": {"loss": 0.0, "components": 0.0, "strata": 0.0},
        "gradient_comparison": {
            "full": _gradient_comparison(),
            "strata": {name: _gradient_comparison() for name in attribution.STRATUM_NAMES},
        },
        "proposal_response_maximum_absolute_difference": 0.0,
        "proposal_bank_metric_comparison": {
            "loss": 0.0,
            "components": 0.0,
            "strata": 0.0,
        },
        "sequential_bank_metric_comparison": {
            "loss": 0.0,
            "components": 0.0,
            "strata": 0.0,
        },
        "witness": {
            "baseline_loss": 1.0,
            "baseline_strata": baseline_strata,
            "loss": {"loss": 0.998},
            "strata": candidate_strata,
            "decision": decision,
            "locked_trial_comparison": {
                "loss": 0.0,
                "components": 0.0,
                "strata": 0.0,
            },
            "signed_directional_derivatives": {
                "full": -0.01,
                "strata": {name: -0.01 for name in attribution.STRATUM_NAMES},
            },
            "all_signed_directional_derivatives_negative": True,
            "pass": True,
        },
        "archive_evaluation_control": _state_control(archived=True),
        "witness_optimizer_source_control": _state_control(archived=True),
        "restoration_control": _state_control(archived=True),
        "cuda_peak_reserved_bytes": 3 * 2**30,
        "pass": True,
    }


def test_checker_recomputes_size_and_repeat_decisions() -> None:
    size = _passing_size_result()
    repeat = {
        "comparisons": {
            "metrics": {"loss": 0.0, "components": 0.0, "strata": 0.0},
            "full_gradient": _gradient_comparison(),
            "stratum_gradients": {
                name: _gradient_comparison() for name in attribution.STRATUM_NAMES
            },
            "proposal_response": 0.0,
            "witness_response": 0.0,
            "witness_loss": 0.0,
            "witness_strata": 0.0,
        },
        "witness": copy.deepcopy(size["witness"]),
        "archive_evaluation_control": _state_control(archived=True),
        "witness_optimizer_source_control": _state_control(archived=True),
        "restoration_control": _state_control(archived=True),
        "cuda_peak_reserved_bytes": 3 * 2**30,
        "finite": True,
        "pass": True,
    }

    assert checker._size_pass(size) is True
    assert checker._repeat_pass(repeat) is True
    size["gradient_comparison"]["full"]["relative_l2_error"] = (
        preflight.GRADIENT_RELATIVE_L2_TOLERANCE * 2.0
    )
    assert checker._size_pass(size) is False


def test_protocol_never_pools_nonlinear_objectives_or_authorizes_hover() -> None:
    protocol = preflight.protocol_manifest()

    assert protocol["execution"]["group_sizes"] == [1, 2, 4]
    assert protocol["execution"]["per_batch_references_and_nonlinear_objectives_preserved"]
    assert protocol["execution"]["selected_repeat_failure_forbids_fallback"]
    assert protocol["execution"]["size_specific_cuda_oom_is_size_failure"]
    assert protocol["execution"]["fixed_ladder_continues_only_after_verified_oom_recovery"]
    assert protocol["controls"]["shared_control_failure_invalidates_entire_preflight"]
    assert protocol["controls"]["outer_evaluation_transaction_restores_on_every_exit"]
    assert protocol["authority"]["development_or_acceptance_may_open"] is False
    assert protocol["authority"]["motion_routing_hover_gate_or_promotion_authorized"] is False


def _terminal_report() -> dict:
    sizes = []
    for group_size in preflight.GROUP_SIZES:
        item = _passing_size_result()
        item["group_size"] = group_size
        item["case_block_size"] = 4 * group_size
        sizes.append(item)
    selected = sizes[-1]
    repeat = {
        "group_size": preflight.GROUP_SIZES[-1],
        "comparisons": {
            "metrics": {"loss": 0.0, "components": 0.0, "strata": 0.0},
            "full_gradient": _gradient_comparison(),
            "stratum_gradients": {
                name: _gradient_comparison() for name in attribution.STRATUM_NAMES
            },
            "proposal_response": 0.0,
            "witness_response": 0.0,
            "witness_loss": 0.0,
            "witness_strata": 0.0,
        },
        "witness": copy.deepcopy(selected["witness"]),
        "archive_evaluation_control": _state_control(archived=True),
        "witness_optimizer_source_control": _state_control(archived=True),
        "restoration_control": _state_control(archived=True),
        "cuda_peak_reserved_bytes": 3 * 2**30,
        "finite": True,
        "pass": True,
    }
    result = {
        "sequential_reference": {
            "state_control": _state_control(archived=True),
        },
        "size_results": sizes,
        "all_sizes_evaluated": True,
        "passing_sizes_before_repeat": list(preflight.GROUP_SIZES),
        "selected_size_before_repeat": preflight.GROUP_SIZES[-1],
        "selected_repeat": repeat,
        "selected_group_size": preflight.GROUP_SIZES[-1],
        "grouped_execution_qualified": True,
        "archived_state_fingerprint_before": "a" * 64,
        "archived_state_fingerprint_after": "a" * 64,
        "archived_state_restored": True,
        "optimizer_state_fingerprint_before": "b" * 64,
        "optimizer_state_fingerprint_after": "b" * 64,
        "optimizer_state_restored": True,
        "source_state_sha256_before": "c" * 64,
        "source_state_sha256_after": "c" * 64,
        "source_restored": True,
    }
    return {
        "experiment": preflight.EXPERIMENT,
        "protocol_commit": preflight.PROTOCOL_COMMIT,
        "protocol": preflight.protocol_manifest(),
        "input_file_sha256": {
            "runs/optic-motion/vertical-motion-frozen-witness-001/report.json": (
                preflight.WITNESS_REPORT_SHA256
            ),
            "runs/optic-motion/vertical-motion-frozen-witness-001/started.json": (
                preflight.WITNESS_STARTED_SHA256
            ),
            "artifacts/vertical-motion-frozen-witness-v1/report.json": (
                preflight.WITNESS_COMPACT_REPORT_SHA256
            ),
        },
        "implementation_file_sha256": preflight.implementation_file_hashes(),
        "preflight_completed": True,
        "classification": "grouped_cuda_execution_equivalent",
        "result": result,
        "exception": None,
        "grouped_execution_qualified": True,
        "common_descent_trainer_preregistration_authorized": True,
        "development_or_acceptance_opened": False,
        "candidate_or_optimizer_retained": False,
        "motion_routing_hover_gate_or_promotion_authorized": False,
    }


def test_terminal_checker_recomputes_decisions_and_state_controls() -> None:
    report = _terminal_report()

    checker.validate_report(report)
    report["result"]["size_results"][1]["archive_evaluation_control"][
        "source_state_fingerprint"
    ] = "e" * 64
    with pytest.raises(SystemExit, match="size decision changed"):
        checker.validate_report(report)


def test_terminal_checker_accepts_recovered_size_specific_oom() -> None:
    report = _terminal_report()
    result = report["result"]
    result["size_results"][-1] = {
        "group_size": 4,
        "case_block_size": 16,
        "failure_kind": "cuda_out_of_memory",
        "exception": "OutOfMemoryError: injected",
        "oom_recovery_control": _state_control(archived=True),
        "finite": False,
        "cuda_peak_reserved_bytes": 15 * 2**30,
        "cuda_peak_reserved_gibibytes": 15.0,
        "pass": False,
    }
    result["passing_sizes_before_repeat"] = [1, 2]
    result["selected_size_before_repeat"] = 2
    result["selected_repeat"]["group_size"] = 2
    result["selected_group_size"] = 2

    assert preflight._group_size_progress(result["size_results"][-1]) == {
        "stage": "group_size_complete",
        "group_size": 4,
        "pass": False,
        "failure_kind": "cuda_out_of_memory",
        "gradient_seconds": None,
        "bank_seconds": None,
        "peak_gibibytes": 15.0,
    }
    checker.validate_report(report)
    result["size_results"][-1]["oom_recovery_control"]["adam_steps"]["gain"] = 26
    with pytest.raises(SystemExit, match="failed terminal validation"):
        checker.validate_report(report)
