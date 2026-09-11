#!/usr/bin/env python3
"""Qualify grouped CUDA execution against the frozen proposal-25 reference."""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_vertical_motion_frozen_witness as witness_audit  # noqa: E402
import audit_vertical_motion_gradient_attribution as attribution  # noqa: E402
import frozen_optic_motion_deterministic as deterministic  # noqa: E402
import preflight_vertical_motion_commissioning as preflight  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402
import vertical_motion_grouped_execution as grouped  # noqa: E402

EXPERIMENT = "vertical-t4t5-grouped-cuda-equivalence-v1"
PROTOCOL_COMMIT = "941df8a"
WITNESS_DIR = REPO_ROOT / "runs/optic-motion/vertical-motion-frozen-witness-001"
WITNESS_REPORT_SHA256 = "556bb5cf3f3f42427492b20d7daa47682ab399efa64541dbc79b03db8d13e879"
WITNESS_STARTED_SHA256 = "666c5aca899ec8849555df43ff3898747bc55e63fe33b7a7df1ca2848ca44141"
WITNESS_IMPLEMENTATION_SHA256 = "62b07aeff4b4adb639a74141088e0899ada3b4135f64e7edf3235179a2f89dd3"
WITNESS_COMPACT_REPORT = REPO_ROOT / "artifacts/vertical-motion-frozen-witness-v1/report.json"
WITNESS_COMPACT_REPORT_SHA256 = "482a03071937994dc92872c345de0ae71eafd8120ca0d7ac686a9a49ba8382af"
GROUP_SIZES = (1, 2, 4)
ABSOLUTE_TOLERANCE = 1.0e-6
GRADIENT_RELATIVE_L2_TOLERANCE = 1.0e-4
CUDA_PEAK_RESERVED_LIMIT_BYTES = 14 * 2**30


class ControlFailure(RuntimeError):
    """A shared input, numerical, or restoration control failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0")
    parser.add_argument(
        "--registration",
        type=Path,
        default=REPO_ROOT / "artifacts/vertical-motion-commissioning-manifest-v1/manifest.json",
    )
    parser.add_argument(
        "--frozen-audit-report",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/frozen-t4t5-audit-002/report.json",
    )
    parser.add_argument("--archive-dir", type=Path, default=attribution.ARCHIVE_DIR)
    parser.add_argument(
        "--attribution-report",
        type=Path,
        default=witness_audit.ATTRIBUTION_DIR / "report.json",
    )
    parser.add_argument(
        "--attribution-started",
        type=Path,
        default=witness_audit.ATTRIBUTION_DIR / "started.json",
    )
    parser.add_argument("--compact-report", type=Path, default=witness_audit.COMPACT_REPORT)
    parser.add_argument("--witness-report", type=Path, default=WITNESS_DIR / "report.json")
    parser.add_argument("--witness-started", type=Path, default=WITNESS_DIR / "started.json")
    parser.add_argument("--witness-compact-report", type=Path, default=WITNESS_COMPACT_REPORT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-grouped-preflight-001",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "scope": {
            "training_pairs": 96,
            "balanced_batches": 24,
            "development_specs_or_pixels_accessed": False,
            "acceptance_specs_or_pixels_accessed": False,
            "parameters_or_optimizer_updated": False,
            "candidate_retained": False,
            "scientific_objective_changed": False,
        },
        "execution": {
            "group_sizes": list(GROUP_SIZES),
            "case_block_sizes": [4 * size for size in GROUP_SIZES],
            "all_sizes_evaluated": True,
            "contiguous_original_balanced_batch_groups": True,
            "per_batch_references_and_nonlinear_objectives_preserved": True,
            "group_gradients_weighted_by_original_batch_count": True,
            "select_largest_passing_size": True,
            "repeat_selected_size_once": True,
            "selected_repeat_failure_forbids_fallback": True,
            "size_specific_cuda_oom_is_size_failure": True,
            "fixed_ladder_continues_only_after_verified_oom_recovery": True,
        },
        "equivalence": {
            "loss_component_stratum_absolute_tolerance": ABSOLUTE_TOLERANCE,
            "response_maximum_absolute_tolerance": ABSOLUTE_TOLERANCE,
            "gradient_coordinate_maximum_absolute_tolerance": ABSOLUTE_TOLERANCE,
            "gradient_relative_l2_tolerance": GRADIENT_RELATIVE_L2_TOLERANCE,
            "frozen_witness_multiplier": 1.0,
            "frozen_witness_same_decision_required": True,
            "repeat_uses_same_tolerances": True,
            "cuda_peak_reserved_limit_bytes": CUDA_PEAK_RESERVED_LIMIT_BYTES,
        },
        "controls": {
            "all_values_finite": True,
            "all_upstream_hashes_exact": True,
            "shared_control_failure_invalidates_entire_preflight": True,
            "proposal25_parameters_optimizer_and_source_restored_exactly": True,
            "outer_evaluation_transaction_restores_on_every_exit": True,
            "adam_counters_unchanged_at": attribution.ARCHIVED_PROPOSALS,
            "exclusive_started_marker_before_neural_evaluation": True,
            "interrupted_start_fails_closed": True,
        },
        "authority": {
            "pass_authorizes_only_grouped_common_descent_trainer_preregistration": True,
            "development_or_acceptance_may_open": False,
            "module_may_be_retained": False,
            "motion_routing_hover_gate_or_promotion_authorized": False,
        },
    }


def implementation_file_hashes() -> dict[str, str]:
    paths = (Path(__file__).resolve(), Path(grouped.__file__).resolve())
    return {registration.stable_path(path): registration.file_sha256(path) for path in paths}


def acquire_run_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = (output_dir / "run.lock").open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise SystemExit("another grouped-execution preflight owns this output directory") from None
    return stream


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ControlFailure(f"expected JSON object: {path}")
    return value


def load_frozen_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    locked = {
        args.witness_report: WITNESS_REPORT_SHA256,
        args.witness_started: WITNESS_STARTED_SHA256,
        args.witness_compact_report: WITNESS_COMPACT_REPORT_SHA256,
        Path(witness_audit.__file__).resolve(): WITNESS_IMPLEMENTATION_SHA256,
    }
    observed: dict[str, str] = {}
    for path, digest in locked.items():
        if not path.is_file() or registration.file_sha256(path) != digest:
            raise ControlFailure(f"locked grouped-preflight input is missing or changed: {path}")
        observed[registration.stable_path(path)] = digest
    witness_report = _load_json(args.witness_report)
    witness_compact = _load_json(args.witness_compact_report)
    if not (
        witness_report.get("experiment") == witness_audit.EXPERIMENT
        and witness_report.get("protocol_commit") == witness_audit.PROTOCOL_COMMIT
        and witness_report.get("implementation_file_sha256") == WITNESS_IMPLEMENTATION_SHA256
        and witness_report.get("audit_completed") is True
        and witness_report.get("classification") == "frozen_witness_finite_step_supported"
        and witness_report.get("finite_step_supported") is True
        and witness_report.get("constrained_training_preregistration_authorized") is True
        and witness_report.get("development_or_acceptance_opened") is False
        and witness_report.get("candidate_or_optimizer_retained") is False
        and witness_report.get("exception") is None
        and witness_report.get("result", {}).get("first_passing_multiplier") == 1.0
    ):
        raise ControlFailure("locked frozen-witness report is not the required passing result")
    if not (
        witness_compact.get("full_report_file_sha256") == WITNESS_REPORT_SHA256
        and witness_compact.get("full_started_file_sha256") == WITNESS_STARTED_SHA256
        and witness_compact.get("finite_step_supported") is True
        and witness_compact.get("first_passing_multiplier") == 1.0
    ):
        raise ControlFailure("tracked frozen-witness result identity changed")
    upstream_observed, archived, attribution_report, _ = witness_audit.load_frozen_inputs(args)
    observed.update(upstream_observed)
    return observed, archived, attribution_report, witness_report


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(function: Any, *, device: torch.device) -> tuple[Any, float]:
    _synchronize(device)
    start = time.perf_counter()
    value = function()
    _synchronize(device)
    return value, time.perf_counter() - start


def _vector_comparison(observed: Tensor, expected: object) -> dict[str, float]:
    expected_tensor = torch.as_tensor(expected, dtype=torch.float64)
    observed = observed.detach().cpu().to(torch.float64)
    if observed.shape != expected_tensor.shape:
        return {"maximum_absolute_difference": math.inf, "relative_l2_error": math.inf}
    delta = observed - expected_tensor
    denominator = max(float(torch.linalg.vector_norm(expected_tensor)), 1.0e-30)
    return {
        "maximum_absolute_difference": float(torch.max(torch.abs(delta))),
        "relative_l2_error": float(torch.linalg.vector_norm(delta)) / denominator,
    }


def _metrics_comparison(observed: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": attribution.maximum_absolute_difference(observed["loss"], expected["loss"]),
        "components": attribution.maximum_absolute_difference(
            observed["components"], expected["components"]
        ),
    }


def _comparison_pass(comparison: dict[str, Any]) -> bool:
    return bool(
        training.finite_numeric_tree(comparison)
        and all(
            value <= ABSOLUTE_TOLERANCE
            for key, value in comparison.items()
            if key != "relative_l2_error"
        )
        and comparison.get("relative_l2_error", 0.0) <= GRADIENT_RELATIVE_L2_TOLERANCE
    )


def _observe_state_control(
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_fingerprint: str,
    archived_optimizer_fingerprint: str,
    source_fingerprint: str,
    *,
    require_archived_parameters: bool,
) -> dict[str, Any]:
    observed_fingerprint = commissioning.semantic_sha256(
        {"parameters": controller.parameter_values(), "optimizer": optimizer.state_dict()}
    )
    observed_optimizer = commissioning.semantic_sha256(optimizer.state_dict())
    observed_source = commissioning.semantic_sha256(controller.source.state_dict())
    counters = training.optimizer_step_counters(optimizer)
    passed = bool(
        (not require_archived_parameters or observed_fingerprint == archived_fingerprint)
        and observed_optimizer == archived_optimizer_fingerprint
        and observed_source == source_fingerprint
        and set(counters) == set(attribution.PARAMETER_NAMES)
        and set(counters.values()) == {attribution.ARCHIVED_PROPOSALS}
        and training.parameters_are_finite(controller)
        and training.optimizer_state_is_finite(optimizer)
    )
    result = {
        "archived_parameters_required": require_archived_parameters,
        "archived_state_fingerprint": observed_fingerprint,
        "optimizer_state_fingerprint": observed_optimizer,
        "source_state_fingerprint": observed_source,
        "adam_steps": counters,
        "pass": passed,
    }
    if not passed:
        raise ControlFailure("proposal-25 parameters, optimizer, or source changed unexpectedly")
    return result


def _state_control(
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_fingerprint: str,
    archived_optimizer_fingerprint: str,
    source_fingerprint: str,
    *,
    require_archived_parameters: bool,
) -> dict[str, Any]:
    try:
        return _observe_state_control(
            controller,
            optimizer,
            archived_fingerprint,
            archived_optimizer_fingerprint,
            source_fingerprint,
            require_archived_parameters=require_archived_parameters,
        )
    except ControlFailure:
        raise
    except Exception as error:
        raise ControlFailure(
            f"state fingerprint verification failed: {type(error).__name__}: {error}"
        ) from error


def _restore_and_check_unwrapped(
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_parameters: dict[str, Tensor],
    archived_optimizer: dict[str, Any],
    archived_fingerprint: str,
) -> None:
    controller.load_parameter_values(archived_parameters)
    optimizer.load_state_dict(copy.deepcopy(archived_optimizer))
    optimizer.zero_grad(set_to_none=True)
    restored = commissioning.semantic_sha256(
        {"parameters": controller.parameter_values(), "optimizer": optimizer.state_dict()}
    )
    if restored != archived_fingerprint:
        raise ControlFailure("proposal-25 parameters or optimizer did not restore exactly")


def _restore_and_check(
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_parameters: dict[str, Tensor],
    archived_optimizer: dict[str, Any],
    archived_fingerprint: str,
) -> None:
    try:
        _restore_and_check_unwrapped(
            controller,
            optimizer,
            archived_parameters,
            archived_optimizer,
            archived_fingerprint,
        )
    except ControlFailure:
        raise
    except Exception as error:
        raise ControlFailure(
            f"proposal-25 restoration or verification failed: {type(error).__name__}: {error}"
        ) from error


def _restore_transaction_state(
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_parameters: dict[str, Tensor],
    archived_optimizer: dict[str, Any],
    archived_fingerprint: str,
    source_fingerprint: str,
    *,
    stage: str,
) -> None:
    try:
        _restore_and_check(
            controller,
            optimizer,
            archived_parameters,
            archived_optimizer,
            archived_fingerprint,
        )
        if commissioning.semantic_sha256(controller.source.state_dict()) != source_fingerprint:
            raise ControlFailure(f"frozen source changed {stage} evaluation transaction")
    except ControlFailure:
        raise
    except Exception as error:
        raise ControlFailure(
            f"state restoration or verification failed {stage} evaluation transaction: "
            f"{type(error).__name__}: {error}"
        ) from error


def _run_in_restoration_transaction(
    action: Any,
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_parameters: dict[str, Tensor],
    archived_optimizer: dict[str, Any],
    archived_fingerprint: str,
    source_fingerprint: str,
) -> Any:
    _restore_transaction_state(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        archived_fingerprint,
        source_fingerprint,
        stage="before",
    )
    try:
        return action()
    finally:
        _restore_transaction_state(
            controller,
            optimizer,
            archived_parameters,
            archived_optimizer,
            archived_fingerprint,
            source_fingerprint,
            stage="after",
        )


def _recover_cuda_after_oom(
    size: int,
    message: str,
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_fingerprint: str,
    archived_optimizer_fingerprint: str,
    source_fingerprint: str,
    *,
    device: torch.device,
) -> dict[str, Any]:
    peak = torch.cuda.max_memory_reserved(device)
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize(device)
    recovery = _state_control(
        controller,
        optimizer,
        archived_fingerprint,
        archived_optimizer_fingerprint,
        source_fingerprint,
        require_archived_parameters=True,
    )
    return {
        "group_size": size,
        "case_block_size": 4 * size,
        "failure_kind": "cuda_out_of_memory",
        "exception": message,
        "oom_recovery_control": recovery,
        "finite": False,
        "cuda_peak_reserved_bytes": peak,
        "cuda_peak_reserved_gibibytes": peak / 2**30,
        "pass": False,
    }


def _group_size_progress(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "group_size_complete",
        "group_size": result["group_size"],
        "pass": result["pass"],
        "failure_kind": result.get("failure_kind"),
        "gradient_seconds": result.get("gradient_elapsed_seconds"),
        "bank_seconds": result.get("proposal_bank_elapsed_seconds"),
        "peak_gibibytes": result["cuda_peak_reserved_gibibytes"],
    }


def _materialize_witness(
    controller: commissioning.CommissionedController,
    archived_parameters: dict[str, Tensor],
    direction: Tensor,
) -> tuple[dict[str, Tensor], Tensor]:
    intended = direction * witness_audit.REFERENCE_DISPLACEMENT_NORM
    groups = attribution.unflatten_gradient(intended, archived_parameters)
    requested = {
        name: archived_parameters[name].to(torch.float64) + groups[name]
        for name in attribution.PARAMETER_NAMES
    }
    controller.load_parameter_values(requested)
    controller.project_parameters()
    materialized = controller.parameter_values()
    actual = attribution.flatten_named(materialized) - attribution.flatten_named(
        archived_parameters
    )
    return materialized, actual


def _restore_witness_parameters(
    controller: commissioning.CommissionedController,
    archived_parameters: dict[str, Tensor],
) -> None:
    try:
        controller.load_parameter_values(archived_parameters)
    except Exception as error:
        raise ControlFailure(
            f"witness parameter restoration failed: {type(error).__name__}: {error}"
        ) from error


def _witness_evaluation(
    controller: commissioning.CommissionedController,
    archived_parameters: dict[str, Tensor],
    direction: Tensor,
    full_gradient: Tensor,
    stratum_gradients: dict[str, Tensor],
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    sequences: dict[tuple[int, bool], Tensor],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_strata: dict[str, float],
    expected_trial: dict[str, Any],
    *,
    case_block_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    materialized, actual = _materialize_witness(controller, archived_parameters, direction)
    bank, elapsed = _timed(
        lambda: grouped.response_bank_batched(
            controller,
            sequences,
            anatomy,
            tuple(range(len(specs))),
            case_block_size=case_block_size,
            device=device,
        ),
        device=device,
    )
    metrics = training.bank_loss(specs, batches, source_bank, bank, anatomy, materialized)
    strata = training.edge_direction_strata(specs, batches, source_bank, bank, anatomy)
    derivatives = {
        "full": float(torch.dot(full_gradient, actual)),
        "strata": {
            name: float(torch.dot(stratum_gradients[name], actual))
            for name in attribution.STRATUM_NAMES
        },
    }
    decision = attribution.trial_decision(
        baseline_metrics["loss"],
        baseline_strata,
        metrics["loss"],
        strata,
        finite=bool(
            training.finite_numeric_tree(bank)
            and training.finite_numeric_tree(metrics)
            and training.finite_numeric_tree(strata)
            and training.finite_numeric_tree(derivatives)
            and training.parameters_are_finite(controller)
        ),
    )
    comparison = {
        **_metrics_comparison(metrics, expected_trial["loss"]),
        "strata": attribution.maximum_absolute_difference(strata, expected_trial["strata"]),
    }
    result = {
        "case_block_size": case_block_size,
        "elapsed_seconds": elapsed,
        "case_pairs_per_second": len(specs) / elapsed,
        "actual_displacement_norm": float(torch.linalg.vector_norm(actual)),
        "signed_directional_derivatives": derivatives,
        "all_signed_directional_derivatives_negative": bool(
            derivatives["full"] < 0.0
            and all(value < 0.0 for value in derivatives["strata"].values())
        ),
        "baseline_loss": baseline_metrics["loss"],
        "baseline_strata": baseline_strata,
        "loss": metrics,
        "strata": strata,
        "decision": decision,
        "locked_trial_comparison": comparison,
        "pass": bool(
            decision["pass"] == expected_trial["decision"]["pass"]
            and decision["pass"] is True
            and all(value <= ABSOLUTE_TOLERANCE for value in comparison.values())
            and derivatives["full"] < 0.0
            and all(value < 0.0 for value in derivatives["strata"].values())
        ),
    }
    return result, bank


def _size_evaluation(
    size: int,
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_parameters: dict[str, Tensor],
    archived_optimizer: dict[str, Any],
    archived_fingerprint: str,
    archived_optimizer_fingerprint: str,
    source_fingerprint: str,
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    sequences: dict[tuple[int, bool], Tensor],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, Any],
    sequential_bank: dict[str, Any],
    sequential_metrics: dict[str, Any],
    sequential_strata: dict[str, float],
    locked_gradient: dict[str, Any],
    locked_baseline: dict[str, Any],
    locked_baseline_strata: dict[str, float],
    locked_trial: dict[str, Any],
    direction: Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _restore_and_check(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        archived_fingerprint,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    (gradient_result, elapsed_gradient) = _timed(
        lambda: grouped.grouped_objective_gradients(
            controller,
            specs,
            batches,
            sequences,
            source_bank,
            anatomy,
            group_size=size,
            device=device,
        ),
        device=device,
    )
    summary, full_gradient, stratum_gradients = gradient_result
    metric_comparison = {
        **_metrics_comparison(summary, locked_baseline),
        "strata": attribution.maximum_absolute_difference(
            summary["strata"], locked_baseline_strata
        ),
    }
    gradient_comparison = {
        "full": _vector_comparison(full_gradient, locked_gradient["full_gradient"]),
        "strata": {
            name: _vector_comparison(
                stratum_gradients[name], locked_gradient["stratum_gradients"][name]
            )
            for name in attribution.STRATUM_NAMES
        },
    }
    case_block_size = 4 * size
    proposal_bank, elapsed_bank = _timed(
        lambda: grouped.response_bank_batched(
            controller,
            sequences,
            anatomy,
            tuple(range(len(specs))),
            case_block_size=case_block_size,
            device=device,
        ),
        device=device,
    )
    response_difference = grouped.maximum_response_difference(proposal_bank, sequential_bank)
    proposal_metrics = training.bank_loss(
        specs, batches, source_bank, proposal_bank, anatomy, archived_parameters
    )
    proposal_strata = training.edge_direction_strata(
        specs, batches, source_bank, proposal_bank, anatomy
    )
    bank_metric_comparison = {
        **_metrics_comparison(proposal_metrics, locked_baseline),
        "strata": attribution.maximum_absolute_difference(proposal_strata, locked_baseline_strata),
    }
    sequential_bank_metric_comparison = {
        **_metrics_comparison(proposal_metrics, sequential_metrics),
        "strata": attribution.maximum_absolute_difference(proposal_strata, sequential_strata),
    }
    archive_evaluation_control = _state_control(
        controller,
        optimizer,
        archived_fingerprint,
        archived_optimizer_fingerprint,
        source_fingerprint,
        require_archived_parameters=True,
    )
    _restore_and_check(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        archived_fingerprint,
    )
    try:
        witness_result, witness_bank = _witness_evaluation(
            controller,
            archived_parameters,
            direction,
            full_gradient,
            stratum_gradients,
            specs,
            batches,
            sequences,
            source_bank,
            anatomy,
            locked_baseline,
            locked_baseline_strata,
            locked_trial,
            case_block_size=case_block_size,
            device=device,
        )
    finally:
        _restore_witness_parameters(controller, archived_parameters)
    witness_optimizer_source_control = _state_control(
        controller,
        optimizer,
        archived_fingerprint,
        archived_optimizer_fingerprint,
        source_fingerprint,
        require_archived_parameters=True,
    )
    peak = torch.cuda.max_memory_reserved(device)
    finite = bool(
        training.finite_numeric_tree(summary)
        and training.finite_numeric_tree(proposal_bank)
        and training.finite_numeric_tree(proposal_metrics)
        and training.finite_numeric_tree(proposal_strata)
        and training.finite_numeric_tree(witness_result)
        and training.finite_numeric_tree(witness_bank)
    )
    gradients_pass = bool(
        _comparison_pass(gradient_comparison["full"])
        and all(_comparison_pass(value) for value in gradient_comparison["strata"].values())
    )
    passed = bool(
        finite
        and all(value <= ABSOLUTE_TOLERANCE for value in metric_comparison.values())
        and gradients_pass
        and response_difference <= ABSOLUTE_TOLERANCE
        and all(value <= ABSOLUTE_TOLERANCE for value in bank_metric_comparison.values())
        and all(value <= ABSOLUTE_TOLERANCE for value in sequential_bank_metric_comparison.values())
        and witness_result["pass"]
        and peak <= CUDA_PEAK_RESERVED_LIMIT_BYTES
    )
    result = {
        "group_size": size,
        "case_block_size": case_block_size,
        "gradient_elapsed_seconds": elapsed_gradient,
        "gradient_original_batches_per_second": len(batches) / elapsed_gradient,
        "proposal_bank_elapsed_seconds": elapsed_bank,
        "proposal_bank_case_pairs_per_second": len(specs) / elapsed_bank,
        "grouped_metrics": summary,
        "grouped_metric_comparison": metric_comparison,
        "gradient_comparison": gradient_comparison,
        "proposal_response_maximum_absolute_difference": response_difference,
        "proposal_bank_metrics": proposal_metrics,
        "proposal_bank_strata": proposal_strata,
        "proposal_bank_metric_comparison": bank_metric_comparison,
        "sequential_bank_metric_comparison": sequential_bank_metric_comparison,
        "witness": witness_result,
        "archive_evaluation_control": archive_evaluation_control,
        "witness_optimizer_source_control": witness_optimizer_source_control,
        "finite": finite,
        "cuda_peak_reserved_bytes": peak,
        "cuda_peak_reserved_gibibytes": peak / 2**30,
        "pass": passed,
    }
    cached = {
        "summary": summary,
        "full_gradient": full_gradient,
        "stratum_gradients": stratum_gradients,
        "proposal_bank": proposal_bank,
        "witness_bank": witness_bank,
        "witness": witness_result,
    }
    _restore_and_check(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        archived_fingerprint,
    )
    result["restoration_control"] = _state_control(
        controller,
        optimizer,
        archived_fingerprint,
        archived_optimizer_fingerprint,
        source_fingerprint,
        require_archived_parameters=True,
    )
    return result, cached


def _repeat_selected(
    size: int,
    cached: dict[str, Any],
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_parameters: dict[str, Tensor],
    archived_optimizer: dict[str, Any],
    archived_fingerprint: str,
    archived_optimizer_fingerprint: str,
    source_fingerprint: str,
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    sequences: dict[tuple[int, bool], Tensor],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, Any],
    locked_baseline: dict[str, Any],
    locked_baseline_strata: dict[str, float],
    locked_trial: dict[str, Any],
    direction: Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _restore_and_check(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        archived_fingerprint,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    (gradient_result, gradient_elapsed) = _timed(
        lambda: grouped.grouped_objective_gradients(
            controller,
            specs,
            batches,
            sequences,
            source_bank,
            anatomy,
            group_size=size,
            device=device,
        ),
        device=device,
    )
    summary, full_gradient, stratum_gradients = gradient_result
    case_block_size = 4 * size
    proposal_bank, proposal_elapsed = _timed(
        lambda: grouped.response_bank_batched(
            controller,
            sequences,
            anatomy,
            tuple(range(len(specs))),
            case_block_size=case_block_size,
            device=device,
        ),
        device=device,
    )
    archive_evaluation_control = _state_control(
        controller,
        optimizer,
        archived_fingerprint,
        archived_optimizer_fingerprint,
        source_fingerprint,
        require_archived_parameters=True,
    )
    _restore_and_check(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        archived_fingerprint,
    )
    try:
        witness_result, witness_bank = _witness_evaluation(
            controller,
            archived_parameters,
            direction,
            full_gradient,
            stratum_gradients,
            specs,
            batches,
            sequences,
            source_bank,
            anatomy,
            locked_baseline,
            locked_baseline_strata,
            locked_trial,
            case_block_size=case_block_size,
            device=device,
        )
    finally:
        _restore_witness_parameters(controller, archived_parameters)
    witness_optimizer_source_control = _state_control(
        controller,
        optimizer,
        archived_fingerprint,
        archived_optimizer_fingerprint,
        source_fingerprint,
        require_archived_parameters=True,
    )
    comparisons = {
        "metrics": {
            **_metrics_comparison(summary, cached["summary"]),
            "strata": attribution.maximum_absolute_difference(
                summary["strata"], cached["summary"]["strata"]
            ),
        },
        "full_gradient": _vector_comparison(full_gradient, cached["full_gradient"]),
        "stratum_gradients": {
            name: _vector_comparison(stratum_gradients[name], cached["stratum_gradients"][name])
            for name in attribution.STRATUM_NAMES
        },
        "proposal_response": grouped.maximum_response_difference(
            proposal_bank, cached["proposal_bank"]
        ),
        "witness_response": grouped.maximum_response_difference(
            witness_bank, cached["witness_bank"]
        ),
        "witness_loss": attribution.maximum_absolute_difference(
            witness_result["loss"], cached["witness"]["loss"]
        ),
        "witness_strata": attribution.maximum_absolute_difference(
            witness_result["strata"], cached["witness"]["strata"]
        ),
    }
    peak = torch.cuda.max_memory_reserved(device)
    finite = bool(
        training.finite_numeric_tree(summary)
        and training.finite_numeric_tree(full_gradient)
        and training.finite_numeric_tree(stratum_gradients)
        and training.finite_numeric_tree(proposal_bank)
        and training.finite_numeric_tree(witness_result)
        and training.finite_numeric_tree(witness_bank)
        and training.finite_numeric_tree(comparisons)
    )
    passed = bool(
        finite
        and all(value <= ABSOLUTE_TOLERANCE for value in comparisons["metrics"].values())
        and _comparison_pass(comparisons["full_gradient"])
        and all(_comparison_pass(value) for value in comparisons["stratum_gradients"].values())
        and comparisons["proposal_response"] <= ABSOLUTE_TOLERANCE
        and comparisons["witness_response"] <= ABSOLUTE_TOLERANCE
        and comparisons["witness_loss"] <= ABSOLUTE_TOLERANCE
        and comparisons["witness_strata"] <= ABSOLUTE_TOLERANCE
        and witness_result["pass"]
        and peak <= CUDA_PEAK_RESERVED_LIMIT_BYTES
    )
    _restore_and_check(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        archived_fingerprint,
    )
    restoration_control = _state_control(
        controller,
        optimizer,
        archived_fingerprint,
        archived_optimizer_fingerprint,
        source_fingerprint,
        require_archived_parameters=True,
    )
    return {
        "group_size": size,
        "gradient_elapsed_seconds": gradient_elapsed,
        "proposal_bank_elapsed_seconds": proposal_elapsed,
        "comparisons": comparisons,
        "witness": witness_result,
        "finite": finite,
        "archive_evaluation_control": archive_evaluation_control,
        "witness_optimizer_source_control": witness_optimizer_source_control,
        "restoration_control": restoration_control,
        "cuda_peak_reserved_bytes": peak,
        "cuda_peak_reserved_gibibytes": peak / 2**30,
        "pass": passed,
    }


def run_preflight(
    args: argparse.Namespace,
    archived: dict[str, Any],
    attribution_report: dict[str, Any],
    witness_report: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    registered = commissioning.load_registered_manifest(args.registration)
    specs = registered["stimuli"]["splits"]["training"]["specs"]
    batches = training.balanced_batches(specs, split="training")
    sequences = training.render_bank(specs)
    if (
        commissioning.semantic_sha256(sequences)
        != attribution.EXPECTED_TRAINING_PIXELS_SEMANTIC_SHA256
    ):
        raise ControlFailure("rendered training pixel identity changed")
    source_state = commissioning.load_source_checkpoint(args.checkpoint)
    if commissioning.semantic_sha256(source_state) != attribution.EXPECTED_SOURCE_STATE_SHA256:
        raise ControlFailure("source checkpoint semantic identity changed")
    anatomy = commissioning.anatomy_arrays(args.graph, args.raw_dir / registration.ANNOTATIONS_FILE)
    source = commissioning.make_source_controller(args.graph, source_state, device=device)
    source_before = commissioning.semantic_sha256(source.state_dict())
    controller = commissioning.CommissionedController(source, anatomy).to(device)
    archived_parameters = {
        name: value.detach().cpu().clone()
        for name, value in archived["state"]["parameter_values"].items()
    }
    archived_optimizer = copy.deepcopy(archived["state"]["optimizer_state"])
    controller.load_parameter_values(archived_parameters)
    optimizer = training.make_optimizer(controller)
    optimizer.load_state_dict(copy.deepcopy(archived_optimizer))
    optimizer.zero_grad(set_to_none=True)
    archived_fingerprint = commissioning.semantic_sha256(
        {"parameters": archived_parameters, "optimizer": archived_optimizer}
    )
    archived_optimizer_fingerprint = commissioning.semantic_sha256(archived_optimizer)
    if (
        set(training.optimizer_step_counters(optimizer).values())
        != {attribution.ARCHIVED_PROPOSALS}
        or not training.parameters_are_finite(controller)
        or not training.optimizer_state_is_finite(optimizer)
    ):
        raise ControlFailure("archived proposal-25 controller or optimizer is invalid")
    source_bank = archived["source_bank"]
    locked_gradient = attribution_report["result"]["gradient_attribution"]
    locked_baseline = attribution_report["result"]["proposal25_loss"]
    locked_baseline_strata = attribution_report["result"]["proposal25_strata"]
    locked_trial = witness_report["result"]["trials"][0]
    direction = torch.tensor(
        witness_report["result"]["witness_control"]["direction"], dtype=torch.float64
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    sequential_bank, sequential_elapsed = _timed(
        lambda: training._candidate_bank(
            controller,
            sequences,
            anatomy,
            tuple(range(len(specs))),
            device=device,
        ),
        device=device,
    )
    sequential_metrics = training.bank_loss(
        specs, batches, source_bank, sequential_bank, anatomy, archived_parameters
    )
    sequential_strata = training.edge_direction_strata(
        specs, batches, source_bank, sequential_bank, anatomy
    )
    sequential_peak = torch.cuda.max_memory_reserved(device)
    sequential_comparison = {
        **_metrics_comparison(sequential_metrics, locked_baseline),
        "strata": attribution.maximum_absolute_difference(
            sequential_strata, locked_baseline_strata
        ),
    }
    if not (
        training.finite_numeric_tree(sequential_bank)
        and all(value <= ABSOLUTE_TOLERANCE for value in sequential_comparison.values())
    ):
        raise ControlFailure("sequential proposal-25 reference did not reproduce")
    sequential_state_control = _state_control(
        controller,
        optimizer,
        archived_fingerprint,
        archived_optimizer_fingerprint,
        source_before,
        require_archived_parameters=True,
    )

    size_results = []
    caches = {}
    for size in GROUP_SIZES:
        oom_message = None
        try:
            result, cache = _run_in_restoration_transaction(
                lambda size=size: _size_evaluation(
                    size,
                    controller,
                    optimizer,
                    archived_parameters,
                    archived_optimizer,
                    archived_fingerprint,
                    archived_optimizer_fingerprint,
                    source_before,
                    specs,
                    batches,
                    sequences,
                    source_bank,
                    anatomy,
                    sequential_bank,
                    sequential_metrics,
                    sequential_strata,
                    locked_gradient,
                    locked_baseline,
                    locked_baseline_strata,
                    locked_trial,
                    direction,
                    device=device,
                ),
                controller,
                optimizer,
                archived_parameters,
                archived_optimizer,
                archived_fingerprint,
                source_before,
            )
        except torch.cuda.OutOfMemoryError as error:
            oom_message = f"{type(error).__name__}: {error}"
        if oom_message is not None:
            result = _recover_cuda_after_oom(
                size,
                oom_message,
                controller,
                optimizer,
                archived_fingerprint,
                archived_optimizer_fingerprint,
                source_before,
                device=device,
            )
            cache = None
        size_results.append(result)
        if cache is not None:
            caches[size] = cache
        print(
            json.dumps(_group_size_progress(result)),
            flush=True,
        )
    passing_sizes = [item["group_size"] for item in size_results if item["pass"]]
    selected_before_repeat = max(passing_sizes) if passing_sizes else None
    repeat = None
    selected = None
    if selected_before_repeat is not None:
        oom_message = None
        try:
            repeat = _run_in_restoration_transaction(
                lambda: _repeat_selected(
                    selected_before_repeat,
                    caches[selected_before_repeat],
                    controller,
                    optimizer,
                    archived_parameters,
                    archived_optimizer,
                    archived_fingerprint,
                    archived_optimizer_fingerprint,
                    source_before,
                    specs,
                    batches,
                    sequences,
                    source_bank,
                    anatomy,
                    locked_baseline,
                    locked_baseline_strata,
                    locked_trial,
                    direction,
                    device=device,
                ),
                controller,
                optimizer,
                archived_parameters,
                archived_optimizer,
                archived_fingerprint,
                source_before,
            )
        except torch.cuda.OutOfMemoryError as error:
            oom_message = f"{type(error).__name__}: {error}"
        if oom_message is not None:
            repeat = _recover_cuda_after_oom(
                selected_before_repeat,
                oom_message,
                controller,
                optimizer,
                archived_fingerprint,
                archived_optimizer_fingerprint,
                source_before,
                device=device,
            )
        if repeat["pass"]:
            selected = selected_before_repeat

    restored = commissioning.semantic_sha256(
        {"parameters": controller.parameter_values(), "optimizer": optimizer.state_dict()}
    )
    optimizer_after = commissioning.semantic_sha256(optimizer.state_dict())
    source_after = commissioning.semantic_sha256(source.state_dict())
    if (
        restored != archived_fingerprint
        or optimizer_after != archived_optimizer_fingerprint
        or source_after != source_before
    ):
        raise ControlFailure("final proposal-25, optimizer or source restoration failed")
    return {
        "sequential_reference": {
            "elapsed_seconds": sequential_elapsed,
            "case_pairs_per_second": len(specs) / sequential_elapsed,
            "comparison": sequential_comparison,
            "state_control": sequential_state_control,
            "cuda_peak_reserved_bytes": sequential_peak,
            "cuda_peak_reserved_gibibytes": sequential_peak / 2**30,
        },
        "size_results": size_results,
        "all_sizes_evaluated": [item["group_size"] for item in size_results] == list(GROUP_SIZES),
        "passing_sizes_before_repeat": passing_sizes,
        "selected_size_before_repeat": selected_before_repeat,
        "selected_repeat": repeat,
        "selected_group_size": selected,
        "grouped_execution_qualified": selected is not None,
        "archived_state_fingerprint_before": archived_fingerprint,
        "archived_state_fingerprint_after": restored,
        "archived_state_restored": restored == archived_fingerprint,
        "optimizer_state_fingerprint_before": archived_optimizer_fingerprint,
        "optimizer_state_fingerprint_after": optimizer_after,
        "optimizer_state_restored": optimizer_after == archived_optimizer_fingerprint,
        "source_state_sha256_before": source_before,
        "source_state_sha256_after": source_after,
        "source_restored": source_after == source_before,
    }


def main() -> int:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    started_path = args.output_dir / "started.json"
    lock = acquire_run_lock(args.output_dir)
    try:
        if report_path.exists():
            raise SystemExit("grouped-execution preflight already terminated")
        if started_path.exists():
            registration.write_exclusive(
                report_path,
                {
                    "experiment": EXPERIMENT,
                    "protocol_commit": PROTOCOL_COMMIT,
                    "protocol": protocol_manifest(),
                    "preflight_completed": False,
                    "classification": "grouped_cuda_equivalence_interrupted_failed_closed",
                    "grouped_execution_qualified": False,
                    "common_descent_trainer_preregistration_authorized": False,
                    "development_or_acceptance_opened": False,
                    "candidate_or_optimizer_retained": False,
                    "motion_routing_hover_gate_or_promotion_authorized": False,
                },
            )
            return 0
        observed, archived, attribution_report, witness_report = load_frozen_inputs(args)
        deterministic.configure_determinism()
        device = torch.device(args.device)
        start = {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol": protocol_manifest(),
            "input_file_sha256": observed,
            "implementation_commit": preflight._git_head(),
            "implementation_file_sha256": implementation_file_hashes(),
            "runtime": deterministic.runtime_manifest(device),
        }
        registration.write_exclusive(started_path, start)
        classification = "grouped_cuda_equivalence_exception_failed_closed"
        result = None
        exception = None
        try:
            result = run_preflight(
                args,
                archived,
                attribution_report,
                witness_report,
                device=device,
            )
            classification = (
                "grouped_cuda_execution_equivalent"
                if result["grouped_execution_qualified"]
                else "grouped_cuda_execution_not_qualified"
            )
        except ControlFailure as error:
            classification = "grouped_cuda_equivalence_control_failed"
            exception = f"{type(error).__name__}: {error}"
        except Exception as error:  # pragma: no cover - terminal fail-closed path
            exception = f"{type(error).__name__}: {error}"
        scientific_pass = bool(result and result["grouped_execution_qualified"])
        report = {
            **start,
            "preflight_completed": result is not None,
            "classification": classification,
            "result": result,
            "exception": exception,
            "grouped_execution_qualified": scientific_pass,
            "common_descent_trainer_preregistration_authorized": scientific_pass,
            "development_or_acceptance_opened": False,
            "candidate_or_optimizer_retained": False,
            "motion_routing_hover_gate_or_promotion_authorized": False,
        }
        registration.write_exclusive(report_path, attribution._json_safe(report))
        print(
            json.dumps(
                {
                    "classification": classification,
                    "preflight_completed": report["preflight_completed"],
                    "grouped_execution_qualified": scientific_pass,
                    "selected_group_size": (result["selected_group_size"] if result else None),
                    "report": str(report_path),
                }
            ),
            flush=True,
        )
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
