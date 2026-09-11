#!/usr/bin/env python3
"""Audit proposal-25 vertical-motion gradients without retaining or training a candidate."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import frozen_optic_motion_deterministic as deterministic  # noqa: E402
import preflight_vertical_motion_commissioning as preflight  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402

EXPERIMENT = "vertical-t4t5-proposal25-gradient-attribution-v1"
PROTOCOL_COMMIT = "b026247"
ARCHIVE_DIR = REPO_ROOT / "runs/optic-motion/vertical-motion-commissioning-training-001"
ARCHIVE_FILE_SHA256 = {
    "report.json": "53925b64c412bc0b50d3483bfd0120076205e362dc3bb214fecfce7fce370ca9",
    "start.json": "01f0fb8a8b72503ace45425a57a30c198eb06d1e9365c4f4c2970c6114198eb7",
    "state.pt": "943b1c850511fc5bcf1efc8e7038b443391932bb508957f22a977f9822ae044c",
    "training-source-responses.pt": (
        "407ab59dcf1597a838f49eaf2b98ac9893e3edd837fe2764ec05b99559368509"
    ),
    "training-source-cache.json": (
        "cc83094035c70c74145764d81e201d1f31c6422459634e4a5a8f41988482eb34"
    ),
}
EXPECTED_TRAINER_SHA256 = "9ba1ad42625cb52c5c24be538ff328e3f35746645a6d9a70db5e0959e2eff727"
EXPECTED_REGISTRATION_SHA256 = (
    "1403c552b371378a59b9cd9830d06ce144f336c7adf05d5061fd385dfc5dbb88"
)
EXPECTED_SOURCE_CACHE_SEMANTIC_SHA256 = (
    "0920384f89e62adc1d62f125648402796008ec26f6b9bd36ea915d885085bab9"
)
EXPECTED_TRAINING_PIXELS_SEMANTIC_SHA256 = (
    "d7b668926ac9419890d90e0615e84a1aca6eef2e29a9a55634a0158e425678a9"
)
EXPECTED_SOURCE_STATE_SHA256 = (
    "8bd3417704779172b34ef79579464485ff94c6377441e65eb630838b2c7e70c3"
)
ARCHIVED_PROPOSALS = 25
REPRODUCTION_ABSOLUTE_TOLERANCE = 1.0e-6
TOTAL_IMPROVEMENT_MINIMUM = 0.001
WORST_STRATUM_IMPROVEMENT_MINIMUM = 0.01
STRATUM_ABSOLUTE_INCREASE_MAXIMUM = 1.0e-4
MULTIPLIERS = (1.0, 0.5, 0.25, 0.125)
PARAMETER_NAMES = ("gain", "bias_offset", "tau_ratio")
STRATUM_NAMES = ("ON_down", "ON_up", "OFF_down", "OFF_up")
BASE_LEARNING_RATES = (0.02, 0.01, 0.01)
PARAMETER_BOUNDS = {
    "gain": (0.25, 4.0),
    "bias_offset": (-0.25, 0.25),
    "tau_ratio": (0.5, 2.0),
}
ACTIVE_BOUND_TOLERANCE = 1.0e-7
COMMON_DESCENT_MARGIN_TOLERANCE = 1.0e-8


class ControlFailure(RuntimeError):
    """An archive or deterministic-reproduction control failed."""


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
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-gradient-attribution-001",
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
            "source_normalizations": "archived batch-specific frozen training references",
            "development_specs_or_pixels_accessed": False,
            "acceptance_specs_or_pixels_accessed": False,
            "candidate_or_optimizer_retained": False,
        },
        "archive": {
            "proposal": ARCHIVED_PROPOSALS,
            "accepted": 25,
            "rejected": 0,
            "adam_step_before": 25,
            "locked_file_sha256": ARCHIVE_FILE_SHA256,
            "trainer_sha256": EXPECTED_TRAINER_SHA256,
            "registration_sha256": EXPECTED_REGISTRATION_SHA256,
            "source_cache_semantic_sha256": EXPECTED_SOURCE_CACHE_SEMANTIC_SHA256,
            "training_pixels_semantic_sha256": EXPECTED_TRAINING_PIXELS_SEMANTIC_SHA256,
            "source_state_semantic_sha256": EXPECTED_SOURCE_STATE_SHA256,
        },
        "controls": {
            "proposal25_loss_components_and_strata_absolute_tolerance": (
                REPRODUCTION_ABSOLUTE_TOLERANCE
            ),
            "all_archive_hashes_and_semantic_hashes_exact": True,
            "all_values_and_adam_moments_finite": True,
            "exclusive_started_marker_before_neural_evaluation": True,
            "interrupted_start_fails_closed": True,
            "control_failure_distinct_from_negative_scientific_result": True,
        },
        "gradients": {
            "dtype": "production FP32 graph; FP64 host-side attribution arithmetic",
            "parameter_order": list(PARAMETER_NAMES),
            "parameter_count": 24,
            "total": "equal mean of 24 registered balanced-batch loss gradients",
            "strata": list(STRATUM_NAMES),
            "stratum_weighting": "own 24 edge cases x two windows, no texture dilution",
            "gradient_clip": preflight.GRADIENT_NORM_CLIP,
            "clip_accumulated_total_once": True,
            "report_minibatch_full_alignment": True,
            "report_stratum_gram_and_cosine": True,
            "report_trial_post_bound_signed_derivatives": True,
            "common_descent_metric": "unweighted Euclidean raw 24-parameter coordinates",
            "common_descent_respects_active_bound_tangent_cone": True,
            "common_descent_is_local_diagnostic_only": True,
        },
        "response_attribution": {
            "normalized_direction": "D=(O_up-O_down)/(2*source_normal_scale)",
            "normalized_bias": "B=(O_up+O_down)/(2*source_normal_scale)",
            "groups": "pathway x window x training speed",
            "also_report": [
                "normalized_down_branch",
                "normalized_up_branch",
                "normalized_opponent_activity",
                "all_registered_loss_components",
            ],
        },
        "full_bank_adam_trial": {
            "multipliers": list(MULTIPLIERS),
            "all_multipliers_evaluated": True,
            "same_archived_parameters_moments_and_clipped_gradient_each_trial": True,
            "adam_counter_transition": [25, 26],
            "project_original_parameter_bounds": True,
            "comparison_baseline": "archived proposal-25 controller",
            "total_loss_improvement_fraction_minimum": TOTAL_IMPROVEMENT_MINIMUM,
            "maximum_stratum_loss_improvement_fraction_minimum": (
                WORST_STRATUM_IMPROVEMENT_MINIMUM
            ),
            "each_stratum_absolute_increase_maximum": (
                STRATUM_ABSOLUTE_INCREASE_MAXIMUM
            ),
            "selection": "first passing multiplier in fixed order, never retrospective best",
        },
        "authority": {
            "passing_trial_authorizes_only_separate_larger_effective_batch_preregistration": True,
            "closed_training_run_may_resume": False,
            "development_or_acceptance_may_open": False,
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
        raise SystemExit("another gradient-attribution audit owns this output directory") from None
    return stream


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ControlFailure(f"expected JSON object: {path}")
    return value


def _load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _finite_tree(value: Any) -> bool:
    return training.finite_numeric_tree(value)


def _tree_semantic_sha256(value: Any) -> str:
    return commissioning.semantic_sha256(value)


def archive_inputs(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    expected = {
        args.archive_dir / name: digest for name, digest in ARCHIVE_FILE_SHA256.items()
    }
    expected[Path(training.__file__).resolve()] = EXPECTED_TRAINER_SHA256
    expected[args.registration] = EXPECTED_REGISTRATION_SHA256
    observed = {}
    for path, digest in expected.items():
        if not path.is_file() or registration.file_sha256(path) != digest:
            raise ControlFailure(f"locked gradient-attribution input is missing or changed: {path}")
        observed[registration.stable_path(path)] = digest

    report = _load_json(args.archive_dir / "report.json")
    start = _load_json(args.archive_dir / "start.json")
    state = _load_torch(args.archive_dir / "state.pt")
    cache_metadata = _load_json(args.archive_dir / "training-source-cache.json")
    source_bank = _load_torch(args.archive_dir / "training-source-responses.pt")
    if (
        report.get("classification") != "vertical_motion_training_mandatory_gate_failed"
        or report.get("passed") is not False
        or report.get("proposals_completed") != ARCHIVED_PROPOSALS
        or report.get("accepted_proposals") != ARCHIVED_PROPOSALS
        or report.get("rejected_proposals") != 0
        or report.get("scheduled_evaluation_started") != [ARCHIVED_PROPOSALS]
        or report.get("development_started") != []
        or report.get("development_history") != []
        or report.get("acceptance") is not None
        or report.get("vertical_motion_module_retained") is not False
        or report.get("source_restored") is not True
        or report.get("exception") is not None
    ):
        raise ControlFailure("archived report is not the exact valid proposal-25 stop")
    if (
        start.get("implementation_commit") != "cc6082411ddc5b15b00750f43c283dccfcd05953"
        or start.get("implementation_file_sha256", {}).get(
            "scripts/train_vertical_motion_commissioning.py"
        )
        != EXPECTED_TRAINER_SHA256
        or start.get("source_state_sha256") != EXPECTED_SOURCE_STATE_SHA256
    ):
        raise ControlFailure("archived start marker identity changed")
    for stable_name, digest in start.get("input_file_sha256", {}).items():
        path = REPO_ROOT / stable_name
        if not path.is_file() or registration.file_sha256(path) != digest:
            raise ControlFailure(f"original training input is missing or changed: {path}")
        observed[stable_name] = digest
    if (
        state.get("protocol_commit") != training.PROTOCOL_COMMIT
        or state.get("implementation_sha256") != EXPECTED_TRAINER_SHA256
        or state.get("proposal_completed") != ARCHIVED_PROPOSALS
        or state.get("accepted_proposals") != ARCHIVED_PROPOSALS
        or state.get("rejected_proposals") != 0
        or state.get("proposal_in_flight") is not None
        or state.get("stop_reason") != "mandatory proposal-25 training gate failed"
        or state.get("scheduled_evaluation_started") != [ARCHIVED_PROPOSALS]
        or state.get("development_started") != []
        or state.get("development_history") != []
        or state.get("training_pixels_sha256") != EXPECTED_TRAINING_PIXELS_SEMANTIC_SHA256
        or state.get("training_source_semantic_sha256")
        != EXPECTED_SOURCE_CACHE_SEMANTIC_SHA256
    ):
        raise ControlFailure("archived training state identity or terminal status changed")
    if (
        cache_metadata.get("cases") != 96
        or cache_metadata.get("semantic_sha256") != EXPECTED_SOURCE_CACHE_SEMANTIC_SHA256
        or cache_metadata.get("file_sha256")
        != ARCHIVE_FILE_SHA256["training-source-responses.pt"]
        or _tree_semantic_sha256(source_bank) != EXPECTED_SOURCE_CACHE_SEMANTIC_SHA256
    ):
        raise ControlFailure("archived source-response cache identity changed")
    if not _finite_tree(state):
        raise ControlFailure("archived parameters or Adam moments are nonfinite")
    return observed, {
        "report": report,
        "start": start,
        "state": state,
        "cache_metadata": cache_metadata,
        "source_bank": source_bank,
    }


def flatten_named(values: dict[str, Tensor], *, dtype: torch.dtype = torch.float64) -> Tensor:
    return torch.cat(
        [
            values[name].detach().reshape(-1).to(device="cpu", dtype=dtype)
            for name in PARAMETER_NAMES
        ]
    )


def unflatten_gradient(vector: Tensor, template: dict[str, Tensor]) -> dict[str, Tensor]:
    result = {}
    offset = 0
    for name in PARAMETER_NAMES:
        count = template[name].numel()
        result[name] = vector[offset : offset + count].reshape(template[name].shape).clone()
        offset += count
    if offset != vector.numel():
        raise ValueError("gradient vector does not match parameter template")
    return result


def vector_cosine(first: Tensor, second: Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second))
    if denominator == 0.0:
        return math.nan
    return float(torch.dot(first, second) / denominator)


def matrix_geometry(vectors: dict[str, Tensor]) -> dict[str, Any]:
    matrix = torch.stack([vectors[name] for name in STRATUM_NAMES])
    gram = matrix @ matrix.T
    norms = torch.linalg.vector_norm(matrix, dim=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = torch.where(denominator > 0, gram / denominator, torch.full_like(gram, torch.nan))
    return {
        "order": list(STRATUM_NAMES),
        "norms": norms.tolist(),
        "gram": gram.tolist(),
        "cosine": cosine.tolist(),
    }


def parameter_bound_activity(values: dict[str, Tensor]) -> dict[str, Any]:
    groups = {}
    flattened_status = []
    for name in PARAMETER_NAMES:
        vector = values[name].detach().cpu().to(torch.float64).reshape(-1)
        lower, upper = PARAMETER_BOUNDS[name]
        lower_indices = [
            index
            for index, value in enumerate(vector.tolist())
            if abs(value - lower) <= ACTIVE_BOUND_TOLERANCE
        ]
        upper_indices = [
            index
            for index, value in enumerate(vector.tolist())
            if abs(value - upper) <= ACTIVE_BOUND_TOLERANCE
        ]
        groups[name] = {
            "bounds": [lower, upper],
            "lower_indices": lower_indices,
            "upper_indices": upper_indices,
        }
        for index in range(vector.numel()):
            if index in lower_indices:
                flattened_status.append("lower")
            elif index in upper_indices:
                flattened_status.append("upper")
            else:
                flattened_status.append("free")
    return {"groups": groups, "flattened_status": flattened_status}


def common_descent_geometry(
    stratum_gradients: dict[str, Tensor], bound_activity: dict[str, Any]
) -> dict[str, Any]:
    matrix = np.stack(
        [stratum_gradients[name].detach().cpu().numpy() for name in STRATUM_NAMES]
    ).astype(np.float64, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)) or np.any(norms == 0.0):
        return {
            "metric": "unweighted Euclidean raw 24-parameter coordinates",
            "solver_success": False,
            "strict_common_descent": False,
            "reason": "nonfinite or zero stratum gradient",
        }
    unit = matrix / norms[:, None]
    coordinate_bounds = []
    for status in bound_activity["flattened_status"]:
        if status == "lower":
            coordinate_bounds.append((0.0, 1.0))
        elif status == "upper":
            coordinate_bounds.append((-1.0, 0.0))
        else:
            coordinate_bounds.append((-1.0, 1.0))
    variable_bounds = coordinate_bounds + [(-1.0, 1.0)]

    def objective(value: np.ndarray) -> float:
        return -float(value[-1])

    def objective_jacobian(value: np.ndarray) -> np.ndarray:
        result = np.zeros_like(value)
        result[-1] = -1.0
        return result

    def norm_constraint_jacobian(value: np.ndarray) -> np.ndarray:
        result = np.zeros_like(value)
        result[:-1] = -2.0 * value[:-1]
        return result

    constraints = [
        {
            "type": "ineq",
            "fun": lambda value: 1.0 - float(np.dot(value[:-1], value[:-1])),
            "jac": norm_constraint_jacobian,
        }
    ]
    for row in unit:
        constraint_jacobian = np.concatenate((-row, [-1.0]))
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda value, gradient=row: (
                    -float(np.dot(gradient, value[:-1])) - float(value[-1])
                ),
                "jac": lambda value, jacobian=constraint_jacobian: jacobian,
            }
        )
    seed = -unit.mean(axis=0)
    for index, status in enumerate(bound_activity["flattened_status"]):
        if status == "lower" and seed[index] < 0.0:
            seed[index] = 0.0
        elif status == "upper" and seed[index] > 0.0:
            seed[index] = 0.0
    seed_norm = float(np.linalg.norm(seed))
    if seed_norm > 1.0:
        seed /= seed_norm
    initial_margin = float(np.min(-unit @ seed))
    initial = np.concatenate((seed, [min(initial_margin, 0.0)]))
    result = minimize(
        objective,
        initial,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=variable_bounds,
        constraints=constraints,
        options={"ftol": 1.0e-12, "maxiter": 2000, "disp": False},
    )
    direction = np.asarray(result.x[:-1], dtype=np.float64)
    claimed_margin = float(result.x[-1])
    signed_cosines = unit @ direction
    margin = float(np.min(-signed_cosines))
    finite_result = bool(
        np.all(np.isfinite(direction))
        and np.all(np.isfinite(signed_cosines))
        and math.isfinite(claimed_margin)
        and math.isfinite(margin)
    )
    norm_feasible = bool(finite_result and np.linalg.norm(direction) <= 1.0 + 1.0e-8)
    tangent_bounds_feasible = bool(
        finite_result
        and all(
            lower - 1.0e-8 <= value <= upper + 1.0e-8
            for value, (lower, upper) in zip(direction, coordinate_bounds, strict=True)
        )
    )
    claimed_constraints_feasible = bool(
        finite_result and np.all(-signed_cosines - claimed_margin >= -1.0e-8)
    )
    certified = bool(
        result.success
        and finite_result
        and norm_feasible
        and tangent_bounds_feasible
        and claimed_constraints_feasible
    )
    strict_common_descent = bool(
        certified
        and claimed_margin > COMMON_DESCENT_MARGIN_TOLERANCE
        and margin > COMMON_DESCENT_MARGIN_TOLERANCE
    )
    if not certified:
        assessment = "inconclusive_solver_or_feasibility_failure"
    elif strict_common_descent:
        assessment = "certified_local_strict_common_descent_direction"
    else:
        assessment = "no_strict_common_descent_found_in_registered_local_metric"
    return {
        "metric": "unweighted Euclidean raw 24-parameter coordinates",
        "solver": "SciPy SLSQP max-min unit-direction margin",
        "solver_success": bool(result.success),
        "certified_feasible_solution": certified,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "unit_direction": direction.tolist(),
        "direction_norm": float(np.linalg.norm(direction)),
        "normalized_stratum_directional_derivatives": {
            name: float(value) for name, value in zip(STRATUM_NAMES, signed_cosines, strict=True)
        },
        "claimed_minimum_descent_margin": claimed_margin,
        "observed_minimum_descent_margin": margin,
        "finite_result": finite_result,
        "unit_norm_constraint_pass": norm_feasible,
        "active_bound_tangent_constraints_pass": tangent_bounds_feasible,
        "claimed_margin_constraints_pass": claimed_constraints_feasible,
        "strict_common_descent": strict_common_descent,
        "assessment": assessment,
        "margin_tolerance": COMMON_DESCENT_MARGIN_TOLERANCE,
        "local_diagnostic_only": True,
        "active_bounds_respected": tangent_bounds_feasible,
    }


def differentiable_edge_strata(
    normal: list[dict[str, Tensor]],
    references: dict[str, Any],
    anatomy: dict[str, np.ndarray],
) -> dict[str, Tensor]:
    relative = commissioning.relative_subtype_indices(anatomy)
    values: dict[str, list[Tensor]] = {name: [] for name in STRATUM_NAMES}
    for pair_index, (polarity, pathway) in enumerate((("ON", "T4"), ("OFF", "T5"))):
        for window in commissioning.WINDOWS:
            response = normal[pair_index][f"{window}_response"]
            opponent = commissioning._opponent(response, relative, pathway)
            scale = references["population"][window][pair_index][pathway]["normal_scale"].to(
                response.device
            )
            values[f"{polarity}_down"].append(
                torch.relu(commissioning.DIRECTION_MARGIN + opponent[0] / scale).square()
            )
            values[f"{polarity}_up"].append(
                torch.relu(commissioning.DIRECTION_MARGIN - opponent[1] / scale).square()
            )
    return {name: torch.stack(items).mean() for name, items in values.items()}


def batched_objective_gradients(
    objectives: list[Tensor], parameters: tuple[Tensor, ...]
) -> Tensor:
    rows = []
    for index, objective in enumerate(objectives):
        gradients = torch.autograd.grad(
            objective,
            parameters,
            retain_graph=index + 1 < len(objectives),
            allow_unused=False,
        )
        rows.append(
            torch.cat(
                [
                    gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
                    for gradient in gradients
                ]
            )
        )
    return torch.stack(rows)


def gradient_attribution(
    controller: commissioning.CommissionedController,
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    sequences: dict[tuple[int, bool], Tensor],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor, dict[str, Tensor], list[Tensor]]:
    parameters = tuple(getattr(controller, name) for name in PARAMETER_NAMES)
    batch_gradients = []
    stratum_batch_gradients: dict[str, list[Tensor]] = {name: [] for name in STRATUM_NAMES}
    rows = []
    for batch_index, cases in enumerate(batches):
        normal, reverse = training.evaluate_cases(
            controller,
            sequences,
            anatomy,
            cases,
            device=device,
            checkpoint_frames=True,
        )
        references = training.batch_references(specs, source_bank, cases, anatomy)
        total, components = commissioning.commissioning_loss(
            [specs[case] for case in cases],
            normal,
            reverse,
            references,
            anatomy,
            controller,
        )
        strata = differentiable_edge_strata(normal, references, anatomy)
        objectives = [total, *(strata[name] for name in STRATUM_NAMES)]
        gradient_matrix = batched_objective_gradients(objectives, parameters)
        batch_gradients.append(gradient_matrix[0])
        for row_index, name in enumerate(STRATUM_NAMES, start=1):
            stratum_batch_gradients[name].append(gradient_matrix[row_index])
        row = {
            "batch": batch_index,
            "cases": list(cases),
            "loss": float(total.detach().cpu()),
            "components": {
                name: float(value.detach().cpu()) for name, value in components.items()
            },
            "strata": {name: float(value.detach().cpu()) for name, value in strata.items()},
            "gradient_norm": float(torch.linalg.vector_norm(gradient_matrix[0])),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "gradient_batch",
                    "batch": batch_index + 1,
                    "batches": len(batches),
                    "loss": row["loss"],
                }
            ),
            flush=True,
        )
        del normal, reverse, total, components, strata, objectives, gradient_matrix

    stacked_batches = torch.stack(batch_gradients)
    full_gradient = stacked_batches.mean(dim=0)
    stratum_gradients = {
        name: torch.stack(stratum_batch_gradients[name]).mean(dim=0) for name in STRATUM_NAMES
    }
    full_norm = float(torch.linalg.vector_norm(full_gradient))
    clip_scale = min(1.0, preflight.GRADIENT_NORM_CLIP / max(full_norm, 1.0e-30))
    clipped = full_gradient * clip_scale
    for row, gradient in zip(rows, batch_gradients, strict=True):
        row["cosine_with_full_gradient"] = vector_cosine(gradient, full_gradient)
        row["dot_with_full_gradient"] = float(torch.dot(gradient, full_gradient))
    components = rows[0]["components"].keys()
    summary = {
        "loss": float(np.mean([row["loss"] for row in rows])),
        "components": {
            name: float(np.mean([row["components"][name] for row in rows]))
            for name in components
        },
        "strata": {
            name: float(np.mean([row["strata"][name] for row in rows]))
            for name in STRATUM_NAMES
        },
        "parameter_order": list(PARAMETER_NAMES),
        "parameter_count": int(full_gradient.numel()),
        "full_gradient": full_gradient.tolist(),
        "full_gradient_unclipped_norm": full_norm,
        "gradient_clip_limit": preflight.GRADIENT_NORM_CLIP,
        "gradient_clip_scale": clip_scale,
        "clipped_full_gradient": clipped.tolist(),
        "stratum_gradients": {name: value.tolist() for name, value in stratum_gradients.items()},
        "stratum_geometry": matrix_geometry(stratum_gradients),
        "batch_rows": rows,
        "gradient_semantic_sha256": _tree_semantic_sha256(
            {
                "full": full_gradient,
                "clipped": clipped,
                "strata": stratum_gradients,
                "batches": stacked_batches,
            }
        ),
    }
    return summary, clipped, stratum_gradients, batch_gradients


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "mean_absolute": float(np.mean(np.abs(array))),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def response_attribution(
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    candidate_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
) -> dict[str, Any]:
    relative = commissioning.relative_subtype_indices(anatomy)
    collected: dict[tuple[str, str, int], dict[str, dict[str, list[float]]]] = {}
    metrics = (
        "normalized_D",
        "normalized_B",
        "normalized_down",
        "normalized_up",
        "normalized_activity",
    )
    for cases in batches:
        references = training.batch_references(specs, source_bank, cases, anatomy)
        for pair_index, case in enumerate(cases[:2]):
            spec = specs[case]
            pathway = "T4" if spec["polarity"] == "ON" else "T5"
            speed = int(spec["speed_pixels_per_frame"])
            for window in commissioning.WINDOWS:
                key = (pathway, window, speed)
                group = collected.setdefault(
                    key,
                    {
                        bank_name: {metric: [] for metric in metrics}
                        for bank_name in ("source", "candidate")
                    },
                )
                scale = float(
                    references["population"][window][pair_index][pathway]["normal_scale"]
                )
                for bank_name, bank in (("source", source_bank), ("candidate", candidate_bank)):
                    response = bank["normal"][case][f"{window}_response"]
                    opponent = commissioning._opponent(response, relative, pathway).detach().cpu()
                    down = float(opponent[0]) / scale
                    up = float(opponent[1]) / scale
                    group[bank_name]["normalized_down"].append(down)
                    group[bank_name]["normalized_up"].append(up)
                    group[bank_name]["normalized_D"].append((up - down) / 2.0)
                    group[bank_name]["normalized_B"].append((up + down) / 2.0)
                    group[bank_name]["normalized_activity"].append((abs(up) + abs(down)) / 2.0)
    rows = []
    for (pathway, window, speed), values in sorted(collected.items()):
        rows.append(
            {
                "pathway": pathway,
                "window": window,
                "speed_pixels_per_frame": speed,
                "edge_cases": len(values["source"]["normalized_D"]),
                "source": {
                    metric: _distribution(items) for metric, items in values["source"].items()
                },
                "candidate": {
                    metric: _distribution(items)
                    for metric, items in values["candidate"].items()
                },
            }
        )
    return {
        "definitions": {
            "D": "(O_up-O_down)/(2*source_normal_scale)",
            "B": "(O_up+O_down)/(2*source_normal_scale)",
            "down": "O_down/source_normal_scale",
            "up": "O_up/source_normal_scale",
            "activity": "(|O_down|+|O_up|)/(2*source_normal_scale)",
        },
        "rows": rows,
    }


def maximum_absolute_difference(observed: Any, expected: Any) -> float:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            return math.inf
        return max(
            (maximum_absolute_difference(observed[key], expected[key]) for key in expected),
            default=0.0,
        )
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not isinstance(observed, (float, int)) or isinstance(observed, bool):
            return math.inf
        return abs(float(observed) - float(expected))
    return 0.0 if observed == expected else math.inf


def reproduction_control(
    source_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    candidate_strata: dict[str, float],
    archived: dict[str, Any],
) -> dict[str, Any]:
    report = archived["report"]
    state = archived["state"]
    expected_candidate = report["training_evaluations"][0]["loss"]
    expected_strata = report["training_evaluations"][0]["edge_direction_strata"]
    comparisons = {
        "source_loss": maximum_absolute_difference(
            source_metrics["loss"], report["training_baseline"]["loss"]
        ),
        "source_components": maximum_absolute_difference(
            source_metrics["components"], report["training_baseline"]["components"]
        ),
        "candidate_loss": maximum_absolute_difference(
            candidate_metrics["loss"], expected_candidate["loss"]
        ),
        "candidate_components": maximum_absolute_difference(
            candidate_metrics["components"], expected_candidate["components"]
        ),
        "candidate_strata": maximum_absolute_difference(candidate_strata, expected_strata),
        "state_training_baseline": maximum_absolute_difference(
            state["training_baseline"], report["training_baseline"]
        ),
        "state_baseline_strata": maximum_absolute_difference(
            state["baseline_strata"], report["baseline_edge_direction_strata"]
        ),
    }
    passed = bool(
        _finite_tree(source_metrics)
        and _finite_tree(candidate_metrics)
        and _finite_tree(candidate_strata)
        and all(value <= REPRODUCTION_ABSOLUTE_TOLERANCE for value in comparisons.values())
    )
    return {
        "pass": passed,
        "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "maximum_absolute_differences": comparisons,
    }


def trial_decision(
    baseline_loss: float,
    baseline_strata: dict[str, float],
    candidate_loss: float,
    candidate_strata: dict[str, float],
    *,
    finite: bool,
) -> dict[str, Any]:
    total_improvement = (baseline_loss - candidate_loss) / baseline_loss
    baseline_worst = max(baseline_strata.values())
    candidate_worst = max(candidate_strata.values())
    worst_improvement = (baseline_worst - candidate_worst) / baseline_worst
    increases = {
        name: candidate_strata[name] - baseline_strata[name] for name in STRATUM_NAMES
    }
    return {
        "pass": bool(
            finite
            and total_improvement >= TOTAL_IMPROVEMENT_MINIMUM
            and worst_improvement >= WORST_STRATUM_IMPROVEMENT_MINIMUM
            and all(value <= STRATUM_ABSOLUTE_INCREASE_MAXIMUM for value in increases.values())
        ),
        "finite": finite,
        "total_loss_improvement_fraction": total_improvement,
        "total_loss_improvement_fraction_minimum": TOTAL_IMPROVEMENT_MINIMUM,
        "maximum_stratum_before": baseline_worst,
        "maximum_stratum_after": candidate_worst,
        "maximum_stratum_improvement_fraction": worst_improvement,
        "maximum_stratum_improvement_fraction_minimum": WORST_STRATUM_IMPROVEMENT_MINIMUM,
        "stratum_absolute_increases": increases,
        "stratum_absolute_increase_maximum": STRATUM_ABSOLUTE_INCREASE_MAXIMUM,
    }


def run_trials(
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    archived_parameters: dict[str, Tensor],
    archived_optimizer: dict[str, Any],
    clipped_gradient: Tensor,
    full_gradient: Tensor,
    stratum_gradients: dict[str, Tensor],
    batch_gradients: list[Tensor],
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    sequences: dict[tuple[int, bool], Tensor],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
    baseline_metrics: dict[str, Any],
    baseline_strata: dict[str, float],
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float | None]:
    archived_fingerprint = _tree_semantic_sha256(
        {"parameters": archived_parameters, "optimizer": archived_optimizer}
    )
    gradient_groups = unflatten_gradient(clipped_gradient, archived_parameters)
    trials = []
    try:
        for multiplier in MULTIPLIERS:
            controller.load_parameter_values(archived_parameters)
            optimizer.load_state_dict(copy.deepcopy(archived_optimizer))
            before_steps = training.optimizer_step_counters(optimizer)
            for group, rate in zip(optimizer.param_groups, BASE_LEARNING_RATES, strict=True):
                group["lr"] = rate * multiplier
            for name in PARAMETER_NAMES:
                parameter = getattr(controller, name)
                parameter.grad = gradient_groups[name].to(parameter).clone()
            optimizer.step()
            controller.project_parameters()
            after_steps = training.optimizer_step_counters(optimizer)
            materialized = controller.parameter_values()
            displacement = flatten_named(materialized) - flatten_named(archived_parameters)
            bound_activity = parameter_bound_activity(materialized)
            with torch.inference_mode():
                candidate_bank = training._candidate_bank(
                    controller,
                    sequences,
                    anatomy,
                    tuple(range(len(specs))),
                    device=device,
                )
            metrics = training.bank_loss(
                specs,
                batches,
                source_bank,
                candidate_bank,
                anatomy,
                materialized,
            )
            strata = training.edge_direction_strata(
                specs, batches, source_bank, candidate_bank, anatomy
            )
            del candidate_bank
            counter_pass = bool(
                set(before_steps.values()) == {ARCHIVED_PROPOSALS}
                and set(after_steps.values()) == {ARCHIVED_PROPOSALS + 1}
            )
            finite = bool(
                _finite_tree(metrics)
                and _finite_tree(strata)
                and _finite_tree(after_steps)
                and bool(torch.isfinite(displacement).all())
                and training.parameters_are_finite(controller)
                and training.optimizer_state_is_finite(optimizer)
                and counter_pass
            )
            if not finite:
                raise ControlFailure(
                    f"full-bank multiplier {multiplier} failed finite or Adam controls"
                )
            decision = trial_decision(
                baseline_metrics["loss"],
                baseline_strata,
                metrics["loss"],
                strata,
                finite=True,
            )
            trial = {
                "multiplier": multiplier,
                "loss": metrics,
                "strata": strata,
                "decision": decision,
                "adam_steps_before": before_steps,
                "adam_steps_after": after_steps,
                "adam_counter_transition_pass": counter_pass,
                "parameter_displacement": displacement.tolist(),
                "parameter_displacement_norm": float(torch.linalg.vector_norm(displacement)),
                "active_bounds_after_projection": bound_activity,
                "signed_directional_derivatives": {
                    "full_loss": float(torch.dot(full_gradient, displacement)),
                    "strata": {
                        name: float(torch.dot(stratum_gradients[name], displacement))
                        for name in STRATUM_NAMES
                    },
                    "minibatches": [
                        float(torch.dot(gradient, displacement)) for gradient in batch_gradients
                    ],
                },
            }
            trials.append(trial)
            print(
                json.dumps(
                    {
                        "stage": "full_bank_trial",
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
            restored = _tree_semantic_sha256(
                {
                    "parameters": controller.parameter_values(),
                    "optimizer": optimizer.state_dict(),
                }
            )
            if restored != archived_fingerprint:
                raise ControlFailure("trial did not restore archived parameters and Adam state")
    finally:
        controller.load_parameter_values(archived_parameters)
        optimizer.load_state_dict(copy.deepcopy(archived_optimizer))
        optimizer.zero_grad(set_to_none=True)
    first_passing = next(
        (trial["multiplier"] for trial in trials if trial["decision"]["pass"]), None
    )
    return trials, first_passing


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_audit(
    args: argparse.Namespace,
    archived: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    registered = commissioning.load_registered_manifest(args.registration)
    specs = registered["stimuli"]["splits"]["training"]["specs"]
    batches = training.balanced_batches(specs, split="training")
    sequences = training.render_bank(specs)
    pixels_sha256 = _tree_semantic_sha256(sequences)
    if pixels_sha256 != EXPECTED_TRAINING_PIXELS_SEMANTIC_SHA256:
        raise ControlFailure("rendered training pixel identity changed")
    source_state = commissioning.load_source_checkpoint(args.checkpoint)
    if _tree_semantic_sha256(source_state) != EXPECTED_SOURCE_STATE_SHA256:
        raise ControlFailure("source checkpoint semantic identity changed")
    anatomy = commissioning.anatomy_arrays(
        args.graph, args.raw_dir / registration.ANNOTATIONS_FILE
    )
    source = commissioning.make_source_controller(args.graph, source_state, device=device)
    source_before = _tree_semantic_sha256(source.state_dict())
    controller = commissioning.CommissionedController(source, anatomy).to(device)
    archived_parameters = {
        name: value.detach().cpu().clone()
        for name, value in archived["state"]["parameter_values"].items()
    }
    archived_optimizer = copy.deepcopy(archived["state"]["optimizer_state"])
    controller.load_parameter_values(archived_parameters)
    optimizer = training.make_optimizer(controller)
    optimizer.load_state_dict(copy.deepcopy(archived_optimizer))
    archived_fingerprint = _tree_semantic_sha256(
        {"parameters": archived_parameters, "optimizer": archived_optimizer}
    )
    if set(training.optimizer_step_counters(optimizer).values()) != {ARCHIVED_PROPOSALS}:
        raise ControlFailure("archived Adam counters are not exactly 25")
    if not training.parameters_are_finite(controller) or not training.optimizer_state_is_finite(
        optimizer
    ):
        raise ControlFailure("archived controller or optimizer is nonfinite")
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
    candidate_metrics = training.bank_loss(
        specs,
        batches,
        source_bank,
        candidate_bank,
        anatomy,
        archived_parameters,
    )
    candidate_strata = training.edge_direction_strata(
        specs, batches, source_bank, candidate_bank, anatomy
    )
    attribution = response_attribution(specs, batches, source_bank, candidate_bank, anatomy)
    control = reproduction_control(
        source_metrics, candidate_metrics, candidate_strata, archived
    )
    del candidate_bank
    if not control["pass"]:
        raise ControlFailure(f"proposal-25 reproduction control failed: {control}")

    gradient, clipped, stratum_gradients, batch_gradients = gradient_attribution(
        controller,
        specs,
        batches,
        sequences,
        source_bank,
        anatomy,
        device=device,
    )
    gradient_reproduction = {
        "loss": maximum_absolute_difference(gradient["loss"], candidate_metrics["loss"]),
        "components": maximum_absolute_difference(
            gradient["components"], candidate_metrics["components"]
        ),
        "strata": maximum_absolute_difference(gradient["strata"], candidate_strata),
    }
    gradient_reproduction["pass"] = bool(
        all(
            value <= REPRODUCTION_ABSOLUTE_TOLERANCE
            for name, value in gradient_reproduction.items()
            if name != "pass"
        )
    )
    if not gradient_reproduction["pass"] or not _finite_tree(gradient):
        raise ControlFailure("differentiable gradient replay did not reproduce proposal 25")

    full_gradient = torch.tensor(gradient["full_gradient"], dtype=torch.float64)
    archived_bounds = parameter_bound_activity(archived_parameters)
    gradient["common_descent"] = common_descent_geometry(
        stratum_gradients, archived_bounds
    )
    gradient["active_bounds_at_archive"] = archived_bounds
    if not _finite_tree(gradient):
        raise ControlFailure("common-descent geometry produced nonfinite values")
    trials, first_passing = run_trials(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        clipped,
        full_gradient,
        stratum_gradients,
        batch_gradients,
        specs,
        batches,
        sequences,
        source_bank,
        anatomy,
        candidate_metrics,
        candidate_strata,
        device=device,
    )
    restored_fingerprint = _tree_semantic_sha256(
        {"parameters": controller.parameter_values(), "optimizer": optimizer.state_dict()}
    )
    source_after = _tree_semantic_sha256(source.state_dict())
    if restored_fingerprint != archived_fingerprint or source_after != source_before:
        raise ControlFailure("final parameter, optimizer or source restoration failed")
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "reproduction_control": control,
        "gradient_reproduction_control": gradient_reproduction,
        "source_loss": source_metrics,
        "proposal25_loss": candidate_metrics,
        "proposal25_strata": candidate_strata,
        "response_attribution": attribution,
        "gradient_attribution": gradient,
        "trials": trials,
        "all_four_multipliers_evaluated": [trial["multiplier"] for trial in trials]
        == list(MULTIPLIERS),
        "first_passing_multiplier": first_passing,
        "whole_bank_trial_pass": first_passing is not None,
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
    start_path = args.output_dir / "started.json"
    lock = acquire_run_lock(args.output_dir)
    try:
        if report_path.exists():
            raise SystemExit("gradient-attribution audit already terminated")
        if start_path.exists():
            registration.write_exclusive(
                report_path,
                {
                    "experiment": EXPERIMENT,
                    "protocol_commit": PROTOCOL_COMMIT,
                    "protocol": protocol_manifest(),
                    "audit_completed": False,
                    "classification": (
                        "vertical_motion_gradient_attribution_interrupted_failed_closed"
                    ),
                    "whole_bank_trial_pass": False,
                    "larger_effective_batch_preregistration_authorized": False,
                    "closed_training_run_may_resume": False,
                    "development_or_acceptance_opened": False,
                    "candidate_or_optimizer_retained": False,
                    "motion_routing_hover_gate_or_promotion_authorized": False,
                },
            )
            return 0

        observed_inputs, archived = archive_inputs(args)
        deterministic.configure_determinism()
        device = torch.device(args.device)
        runtime = deterministic.runtime_manifest(device)
        implementation_file_sha256 = registration.file_sha256(Path(__file__))
        implementation_commit = preflight._git_head()
        start = {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol": protocol_manifest(),
            "input_file_sha256": observed_inputs,
            "implementation_commit": implementation_commit,
            "implementation_file_sha256": implementation_file_sha256,
            "runtime": runtime,
        }
        registration.write_exclusive(start_path, start)
        torch.cuda.reset_peak_memory_stats(device)
        classification = "vertical_motion_gradient_attribution_exception_failed_closed"
        result = None
        exception = None
        try:
            result = run_audit(args, archived, device=device)
            if result["whole_bank_trial_pass"]:
                classification = (
                    "vertical_motion_gradient_attribution_supports_larger_effective_batch"
                )
            else:
                classification = "vertical_motion_gradient_attribution_requires_objective_review"
        except ControlFailure as error:
            classification = "vertical_motion_gradient_attribution_control_failed"
            exception = f"{type(error).__name__}: {error}"
        except Exception as error:  # pragma: no cover - terminal fail-closed path
            exception = f"{type(error).__name__}: {error}"

        report = {
            **start,
            "audit_completed": result is not None,
            "classification": classification,
            "result": result,
            "exception": exception,
            "whole_bank_trial_pass": bool(result and result["whole_bank_trial_pass"]),
            "larger_effective_batch_preregistration_authorized": bool(
                result and result["whole_bank_trial_pass"]
            ),
            "closed_training_run_may_resume": False,
            "development_or_acceptance_opened": False,
            "candidate_or_optimizer_retained": False,
            "motion_routing_hover_gate_or_promotion_authorized": False,
        }
        registration.write_exclusive(report_path, _json_safe(report))
        print(
            json.dumps(
                {
                    "classification": classification,
                    "audit_completed": report["audit_completed"],
                    "whole_bank_trial_pass": report["whole_bank_trial_pass"],
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
