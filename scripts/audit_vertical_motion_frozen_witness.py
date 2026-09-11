#!/usr/bin/env python3
"""Causally test the frozen proposal-25 common-descent witness at four step lengths."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_vertical_motion_gradient_attribution as attribution  # noqa: E402
import frozen_optic_motion_deterministic as deterministic  # noqa: E402
import preflight_vertical_motion_commissioning as preflight  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402

EXPERIMENT = "vertical-t4t5-frozen-witness-finite-step-v1"
PROTOCOL_COMMIT = "66185dc"
ATTRIBUTION_DIR = REPO_ROOT / "runs/optic-motion/vertical-motion-gradient-attribution-001"
ATTRIBUTION_REPORT_SHA256 = "451d71b7752b4b6f25b210b8ae6b2c790cb507c3b50ac56fbf436327cb6ebe3a"
ATTRIBUTION_STARTED_SHA256 = "fd2a173c5050167152efa2787287d0d6c9c3c03b5df8e6a04e565fde7de3764a"
ATTRIBUTION_IMPLEMENTATION_SHA256 = (
    "215a0327d8f41edc95d2b4ebc9051f5c94bdd2b7783c0d5acb9e6992c292eac0"
)
COMPACT_REPORT = REPO_ROOT / "artifacts/vertical-motion-gradient-attribution-v1/report.json"
COMPACT_REPORT_SHA256 = "bb8fd60da7006b7e8b5b11c231e04f44337bc958b72e9c9b9b4a769322ebc609"
REFERENCE_DISPLACEMENT_NORM = 0.03152330128027931
MULTIPLIERS = (1.0, 0.5, 0.25, 0.125)
UNIT_NORM_TOLERANCE = 1.0e-8
GRADIENT_REPRODUCTION_TOLERANCE = 1.0e-6
NORMALIZED_DERIVATIVE_MAXIMUM = -0.09


class ControlFailure(RuntimeError):
    """A frozen-input, numerical, or restoration control failed."""


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
        default=ATTRIBUTION_DIR / "report.json",
    )
    parser.add_argument(
        "--attribution-started",
        type=Path,
        default=ATTRIBUTION_DIR / "started.json",
    )
    parser.add_argument("--compact-report", type=Path, default=COMPACT_REPORT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-frozen-witness-001",
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
            "candidate_or_optimizer_retained": False,
            "objective_optimizer_and_anatomy_unchanged": True,
        },
        "frozen_witness": {
            "source": "locked proposal-25 gradient-attribution report",
            "parameter_order": list(attribution.PARAMETER_NAMES),
            "parameter_count": 24,
            "use_verbatim_without_renormalization": True,
            "unit_norm_tolerance": UNIT_NORM_TOLERANCE,
            "normalized_derivative_definition": "gradient dot direction / L2 gradient norm",
            "normalized_derivative_maximum": NORMALIZED_DERIVATIVE_MAXIMUM,
            "rerun_solver_or_optimize_direction": False,
        },
        "finite_step_ladder": {
            "metric": "unweighted Euclidean raw 24-parameter coordinates",
            "reference_displacement_norm": REFERENCE_DISPLACEMENT_NORM,
            "multipliers": list(MULTIPLIERS),
            "all_multipliers_evaluated": True,
            "original_parameter_bounds_applied": True,
            "same_archived_proposal25_start_each_trial": True,
            "optimizer_stepped_or_modified": False,
            "comparison_baseline": "archived proposal-25 controller",
            "total_loss_improvement_fraction_minimum": (attribution.TOTAL_IMPROVEMENT_MINIMUM),
            "maximum_stratum_loss_improvement_fraction_minimum": (
                attribution.WORST_STRATUM_IMPROVEMENT_MINIMUM
            ),
            "each_stratum_absolute_increase_maximum": (
                attribution.STRATUM_ABSOLUTE_INCREASE_MAXIMUM
            ),
            "selection": "first passing multiplier in fixed order, never retrospective best",
        },
        "controls": {
            "proposal25_reproduction_absolute_tolerance": (
                attribution.REPRODUCTION_ABSOLUTE_TOLERANCE
            ),
            "gradient_reproduction_absolute_tolerance": (GRADIENT_REPRODUCTION_TOLERANCE),
            "all_frozen_hashes_exact": True,
            "all_values_finite_or_entire_audit_invalid": True,
            "adam_counters_unchanged_at": attribution.ARCHIVED_PROPOSALS,
            "per_trial_and_final_restoration_exact": True,
            "exclusive_started_marker_before_neural_evaluation": True,
            "interrupted_start_fails_closed": True,
        },
        "authority": {
            "passing_trial_authorizes_only_separate_constrained_training_preregistration": True,
            "closed_training_run_may_resume": False,
            "development_or_acceptance_may_open": False,
            "module_may_be_retained": False,
            "motion_routing_hover_gate_or_promotion_authorized": False,
        },
    }


def acquire_run_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = (output_dir / "run.lock").open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise SystemExit("another frozen-witness audit owns this output directory") from None
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
    locked_paths = {
        args.attribution_report: ATTRIBUTION_REPORT_SHA256,
        args.attribution_started: ATTRIBUTION_STARTED_SHA256,
        args.compact_report: COMPACT_REPORT_SHA256,
        Path(attribution.__file__).resolve(): ATTRIBUTION_IMPLEMENTATION_SHA256,
    }
    observed: dict[str, str] = {}
    for path, digest in locked_paths.items():
        if not path.is_file() or registration.file_sha256(path) != digest:
            raise ControlFailure(f"locked frozen-witness input is missing or changed: {path}")
        observed[registration.stable_path(path)] = digest

    full_report = _load_json(args.attribution_report)
    compact_report = _load_json(args.compact_report)
    if not (
        full_report.get("experiment") == attribution.EXPERIMENT
        and full_report.get("protocol_commit") == attribution.PROTOCOL_COMMIT
        and full_report.get("implementation_file_sha256") == ATTRIBUTION_IMPLEMENTATION_SHA256
        and full_report.get("audit_completed") is True
        and full_report.get("classification")
        == "vertical_motion_gradient_attribution_requires_objective_review"
        and full_report.get("whole_bank_trial_pass") is False
        and full_report.get("larger_effective_batch_preregistration_authorized") is False
        and full_report.get("development_or_acceptance_opened") is False
        and full_report.get("candidate_or_optimizer_retained") is False
        and full_report.get("motion_routing_hover_gate_or_promotion_authorized") is False
        and full_report.get("exception") is None
    ):
        raise ControlFailure("locked attribution report is not the required terminal result")
    if not (
        compact_report.get("full_report_file_sha256") == ATTRIBUTION_REPORT_SHA256
        and compact_report.get("full_started_file_sha256") == ATTRIBUTION_STARTED_SHA256
        and compact_report.get("audit_completed") is True
        and compact_report.get("whole_bank_trial_pass") is False
    ):
        raise ControlFailure("tracked compact attribution result identity changed")
    full_direction = full_report["result"]["gradient_attribution"]["common_descent"][
        "unit_direction"
    ]
    compact_direction = compact_report["gradient_attribution"]["common_descent"]["unit_direction"]
    if compact_direction != full_direction:
        raise ControlFailure("tracked and complete frozen witnesses differ")

    archive_observed, archived = attribution.archive_inputs(args)
    observed.update(archive_observed)
    return observed, archived, full_report, compact_report


def _maximum_vector_difference(observed: Tensor, expected: object) -> float:
    expected_tensor = torch.as_tensor(expected, dtype=torch.float64)
    if observed.shape != expected_tensor.shape:
        return math.inf
    return float(torch.max(torch.abs(observed.detach().cpu().to(torch.float64) - expected_tensor)))


def witness_control(
    gradient: dict[str, Any],
    stratum_gradients: dict[str, Tensor],
    archived_parameters: dict[str, Tensor],
    full_report: dict[str, Any],
    compact_report: dict[str, Any],
) -> tuple[dict[str, Any], Tensor, Tensor]:
    locked = full_report["result"]["gradient_attribution"]
    direction = torch.tensor(
        compact_report["gradient_attribution"]["common_descent"]["unit_direction"],
        dtype=torch.float64,
    )
    full_gradient = torch.tensor(gradient["full_gradient"], dtype=torch.float64)
    gradient_differences = {
        "full": _maximum_vector_difference(full_gradient, locked["full_gradient"]),
        "strata": {
            name: _maximum_vector_difference(
                stratum_gradients[name], locked["stratum_gradients"][name]
            )
            for name in attribution.STRATUM_NAMES
        },
    }
    norm = float(torch.linalg.vector_norm(direction))
    active_bounds = attribution.parameter_bound_activity(archived_parameters)
    tangent_pass = all(
        (status != "lower" or value >= -UNIT_NORM_TOLERANCE)
        and (status != "upper" or value <= UNIT_NORM_TOLERANCE)
        for status, value in zip(active_bounds["flattened_status"], direction.tolist(), strict=True)
    )

    gradient_norms = {
        "full": float(torch.linalg.vector_norm(full_gradient)),
        **{
            name: float(torch.linalg.vector_norm(stratum_gradients[name]))
            for name in attribution.STRATUM_NAMES
        },
    }
    if not all(math.isfinite(value) and value > 0.0 for value in gradient_norms.values()):
        raise ControlFailure("frozen-witness gradient norm is nonfinite or zero")
    raw_full = float(torch.dot(full_gradient, direction))
    full_norm = gradient_norms["full"]
    normalized = {"full": raw_full / full_norm}
    raw = {"full": raw_full}
    for name in attribution.STRATUM_NAMES:
        value = stratum_gradients[name]
        raw[name] = float(torch.dot(value, direction))
        normalized[name] = raw[name] / gradient_norms[name]

    finite = bool(
        training.finite_numeric_tree(gradient_differences)
        and training.finite_numeric_tree(raw)
        and training.finite_numeric_tree(normalized)
        and bool(torch.isfinite(direction).all())
        and math.isfinite(norm)
    )
    result = {
        "source_report_gradient_semantic_sha256": locked["gradient_semantic_sha256"],
        "gradient_reproduction_maximum_absolute_differences": gradient_differences,
        "direction": direction.tolist(),
        "direction_norm": norm,
        "direction_norm_error": abs(norm - 1.0),
        "unit_norm_tolerance": UNIT_NORM_TOLERANCE,
        "active_bounds_at_archive": active_bounds,
        "active_bound_tangent_pass": tangent_pass,
        "gradient_norms": gradient_norms,
        "raw_directional_derivatives": raw,
        "normalized_directional_derivatives": normalized,
        "normalized_derivative_definition": "gradient dot direction / L2 gradient norm",
        "normalized_derivative_maximum": NORMALIZED_DERIVATIVE_MAXIMUM,
        "finite": finite,
    }
    result["pass"] = bool(
        finite
        and direction.numel() == 24
        and abs(norm - 1.0) <= UNIT_NORM_TOLERANCE
        and tangent_pass
        and gradient_differences["full"] <= GRADIENT_REPRODUCTION_TOLERANCE
        and all(
            value <= GRADIENT_REPRODUCTION_TOLERANCE
            for value in gradient_differences["strata"].values()
        )
        and all(value <= NORMALIZED_DERIVATIVE_MAXIMUM for value in normalized.values())
    )
    if not result["pass"]:
        raise ControlFailure(f"frozen witness qualification failed: {result}")
    return result, direction, full_gradient


def _optimizer_fingerprint(parameters: dict[str, Tensor], optimizer_state: dict[str, Any]) -> str:
    return commissioning.semantic_sha256({"parameters": parameters, "optimizer": optimizer_state})


def run_trials(
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_parameters: dict[str, Tensor],
    archived_optimizer: dict[str, Any],
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
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float | None]:
    archived_fingerprint = _optimizer_fingerprint(archived_parameters, archived_optimizer)
    archived_optimizer_sha256 = commissioning.semantic_sha256(archived_optimizer)
    trials: list[dict[str, Any]] = []
    try:
        for multiplier in MULTIPLIERS:
            controller.load_parameter_values(archived_parameters)
            optimizer.load_state_dict(copy.deepcopy(archived_optimizer))
            optimizer.zero_grad(set_to_none=True)
            optimizer_before = commissioning.semantic_sha256(optimizer.state_dict())
            counters_before = training.optimizer_step_counters(optimizer)

            intended = direction * (REFERENCE_DISPLACEMENT_NORM * multiplier)
            intended_groups = attribution.unflatten_gradient(intended, archived_parameters)
            materialized_request = {
                name: archived_parameters[name].to(torch.float64) + intended_groups[name]
                for name in attribution.PARAMETER_NAMES
            }
            controller.load_parameter_values(materialized_request)
            controller.project_parameters()
            materialized = controller.parameter_values()
            actual = attribution.flatten_named(materialized) - attribution.flatten_named(
                archived_parameters
            )
            projection_difference = actual - intended
            derivatives = {
                "full": float(torch.dot(full_gradient, actual)),
                "strata": {
                    name: float(torch.dot(stratum_gradients[name], actual))
                    for name in attribution.STRATUM_NAMES
                },
            }
            optimizer_after_materialization = commissioning.semantic_sha256(optimizer.state_dict())
            counters_after_materialization = training.optimizer_step_counters(optimizer)
            preevaluation_finite = bool(
                training.parameters_are_finite(controller)
                and training.optimizer_state_is_finite(optimizer)
                and bool(torch.isfinite(intended).all())
                and bool(torch.isfinite(actual).all())
                and bool(torch.isfinite(projection_difference).all())
                and training.finite_numeric_tree(derivatives)
            )
            derivative_pass = bool(
                derivatives["full"] < 0.0
                and all(value < 0.0 for value in derivatives["strata"].values())
            )
            optimizer_unchanged = bool(
                optimizer_before == archived_optimizer_sha256
                and optimizer_before == optimizer_after_materialization
                and set(counters_before.values()) == {attribution.ARCHIVED_PROPOSALS}
                and counters_after_materialization == counters_before
            )
            if not (preevaluation_finite and derivative_pass and optimizer_unchanged):
                raise ControlFailure(f"multiplier {multiplier} failed pre-evaluation controls")

            with torch.inference_mode():
                candidate_bank = training._candidate_bank(
                    controller,
                    sequences,
                    anatomy,
                    tuple(range(len(specs))),
                    device=device,
                )
            candidate_bank_finite = training.finite_numeric_tree(candidate_bank)
            metrics = training.bank_loss(
                specs, batches, source_bank, candidate_bank, anatomy, materialized
            )
            strata = training.edge_direction_strata(
                specs, batches, source_bank, candidate_bank, anatomy
            )
            del candidate_bank
            optimizer_after_evaluation = commissioning.semantic_sha256(optimizer.state_dict())
            counters_after_evaluation = training.optimizer_step_counters(optimizer)
            finite = bool(
                preevaluation_finite
                and candidate_bank_finite
                and training.finite_numeric_tree(metrics)
                and training.finite_numeric_tree(strata)
                and optimizer_after_evaluation == optimizer_before
                and counters_after_evaluation == counters_before
            )
            if not finite:
                raise ControlFailure(f"multiplier {multiplier} failed numerical controls")
            decision = attribution.trial_decision(
                baseline_metrics["loss"],
                baseline_strata,
                metrics["loss"],
                strata,
                finite=True,
            )
            trial = {
                "multiplier": multiplier,
                "reference_displacement_norm": REFERENCE_DISPLACEMENT_NORM,
                "intended_displacement": intended.tolist(),
                "intended_displacement_norm": float(torch.linalg.vector_norm(intended)),
                "actual_displacement": actual.tolist(),
                "actual_displacement_norm": float(torch.linalg.vector_norm(actual)),
                "projection_difference": projection_difference.tolist(),
                "projection_difference_norm": float(
                    torch.linalg.vector_norm(projection_difference)
                ),
                "active_bounds_after_projection": attribution.parameter_bound_activity(
                    materialized
                ),
                "signed_directional_derivatives": derivatives,
                "all_signed_directional_derivatives_negative": derivative_pass,
                "optimizer_semantic_sha256_before": optimizer_before,
                "optimizer_semantic_sha256_after": optimizer_after_evaluation,
                "optimizer_unchanged": bool(
                    optimizer_before == archived_optimizer_sha256
                    and optimizer_after_evaluation == optimizer_before
                ),
                "adam_steps_before": counters_before,
                "adam_steps_after": counters_after_evaluation,
                "loss": metrics,
                "strata": strata,
                "decision": decision,
            }
            trials.append(trial)
            print(
                json.dumps(
                    {
                        "stage": "frozen_witness_trial",
                        "multiplier": multiplier,
                        "loss": metrics["loss"],
                        "maximum_stratum": max(strata.values()),
                        "pass": decision["pass"],
                    }
                ),
                flush=True,
            )

            controller.load_parameter_values(archived_parameters)
            optimizer.load_state_dict(copy.deepcopy(archived_optimizer))
            optimizer.zero_grad(set_to_none=True)
            restored = _optimizer_fingerprint(controller.parameter_values(), optimizer.state_dict())
            if restored != archived_fingerprint:
                raise ControlFailure(
                    f"multiplier {multiplier} did not restore parameters and optimizer"
                )
    finally:
        controller.load_parameter_values(archived_parameters)
        optimizer.load_state_dict(copy.deepcopy(archived_optimizer))
        optimizer.zero_grad(set_to_none=True)

    first_passing = next(
        (trial["multiplier"] for trial in trials if trial["decision"]["pass"]), None
    )
    return trials, first_passing


def run_audit(
    args: argparse.Namespace,
    archived: dict[str, Any],
    full_report: dict[str, Any],
    compact_report: dict[str, Any],
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
    archived_fingerprint = _optimizer_fingerprint(archived_parameters, archived_optimizer)
    if (
        set(training.optimizer_step_counters(optimizer).values())
        != {attribution.ARCHIVED_PROPOSALS}
        or not training.parameters_are_finite(controller)
        or not training.optimizer_state_is_finite(optimizer)
    ):
        raise ControlFailure("archived controller or optimizer control failed")

    source_bank = archived["source_bank"]
    source_metrics = training.bank_loss(specs, batches, source_bank, source_bank, anatomy)
    with torch.inference_mode():
        candidate_bank = training._candidate_bank(
            controller,
            sequences,
            anatomy,
            tuple(range(len(specs))),
            device=device,
        )
    if not training.finite_numeric_tree(candidate_bank):
        raise ControlFailure("proposal-25 response bank is nonfinite")
    baseline_metrics = training.bank_loss(
        specs, batches, source_bank, candidate_bank, anatomy, archived_parameters
    )
    baseline_strata = training.edge_direction_strata(
        specs, batches, source_bank, candidate_bank, anatomy
    )
    del candidate_bank
    reproduction = attribution.reproduction_control(
        source_metrics, baseline_metrics, baseline_strata, archived
    )
    if not reproduction["pass"]:
        raise ControlFailure(f"proposal-25 reproduction control failed: {reproduction}")

    gradient, _, stratum_gradients, _ = attribution.gradient_attribution(
        controller,
        specs,
        batches,
        sequences,
        source_bank,
        anatomy,
        device=device,
    )
    witness, direction, full_gradient = witness_control(
        gradient,
        stratum_gradients,
        archived_parameters,
        full_report,
        compact_report,
    )
    trials, first_passing = run_trials(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        direction,
        full_gradient,
        stratum_gradients,
        specs,
        batches,
        sequences,
        source_bank,
        anatomy,
        baseline_metrics,
        baseline_strata,
        device=device,
    )

    restored_fingerprint = _optimizer_fingerprint(
        controller.parameter_values(), optimizer.state_dict()
    )
    source_after = commissioning.semantic_sha256(source.state_dict())
    if restored_fingerprint != archived_fingerprint or source_after != source_before:
        raise ControlFailure("final parameter, optimizer or source restoration failed")
    if [trial["multiplier"] for trial in trials] != list(MULTIPLIERS):
        raise ControlFailure("finite-step ladder did not complete in registered order")
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "reproduction_control": reproduction,
        "source_loss": source_metrics,
        "proposal25_loss": baseline_metrics,
        "proposal25_strata": baseline_strata,
        "witness_control": witness,
        "trials": trials,
        "all_four_multipliers_evaluated": True,
        "first_passing_multiplier": first_passing,
        "finite_step_supported": first_passing is not None,
        "archived_state_fingerprint_before": archived_fingerprint,
        "archived_state_fingerprint_after": restored_fingerprint,
        "archived_state_restored": restored_fingerprint == archived_fingerprint,
        "source_state_sha256_before": source_before,
        "source_state_sha256_after": source_after,
        "source_restored": source_after == source_before,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_reserved_gibibytes": peak_reserved / 2**30,
    }


def main() -> int:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    started_path = args.output_dir / "started.json"
    lock = acquire_run_lock(args.output_dir)
    try:
        if report_path.exists():
            raise SystemExit("frozen-witness audit already terminated")
        if started_path.exists():
            registration.write_exclusive(
                report_path,
                {
                    "experiment": EXPERIMENT,
                    "protocol_commit": PROTOCOL_COMMIT,
                    "protocol": protocol_manifest(),
                    "audit_completed": False,
                    "classification": "frozen_witness_finite_step_interrupted_failed_closed",
                    "finite_step_supported": False,
                    "constrained_training_preregistration_authorized": False,
                    "closed_training_run_may_resume": False,
                    "development_or_acceptance_opened": False,
                    "candidate_or_optimizer_retained": False,
                    "motion_routing_hover_gate_or_promotion_authorized": False,
                },
            )
            return 0

        observed, archived, full_report, compact_report = load_frozen_inputs(args)
        deterministic.configure_determinism()
        device = torch.device(args.device)
        runtime = deterministic.runtime_manifest(device)
        start = {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol": protocol_manifest(),
            "input_file_sha256": observed,
            "implementation_commit": preflight._git_head(),
            "implementation_file_sha256": registration.file_sha256(Path(__file__)),
            "runtime": runtime,
        }
        registration.write_exclusive(started_path, start)
        torch.cuda.reset_peak_memory_stats(device)
        classification = "frozen_witness_finite_step_exception_failed_closed"
        result = None
        exception = None
        try:
            result = run_audit(
                args,
                archived,
                full_report,
                compact_report,
                device=device,
            )
            classification = (
                "frozen_witness_finite_step_supported"
                if result["finite_step_supported"]
                else "frozen_witness_finite_step_no_useful_leverage"
            )
        except ControlFailure as error:
            classification = "frozen_witness_finite_step_control_failed"
            exception = f"{type(error).__name__}: {error}"
        except Exception as error:  # pragma: no cover - terminal fail-closed path
            exception = f"{type(error).__name__}: {error}"

        scientific_pass = bool(result and result["finite_step_supported"])
        report = {
            **start,
            "audit_completed": result is not None,
            "classification": classification,
            "result": result,
            "exception": exception,
            "finite_step_supported": scientific_pass,
            "constrained_training_preregistration_authorized": scientific_pass,
            "closed_training_run_may_resume": False,
            "development_or_acceptance_opened": False,
            "candidate_or_optimizer_retained": False,
            "motion_routing_hover_gate_or_promotion_authorized": False,
        }
        registration.write_exclusive(report_path, attribution._json_safe(report))
        print(
            json.dumps(
                {
                    "classification": classification,
                    "audit_completed": report["audit_completed"],
                    "finite_step_supported": scientific_pass,
                    "first_passing_multiplier": (
                        result["first_passing_multiplier"] if result else None
                    ),
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
