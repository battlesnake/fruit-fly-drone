#!/usr/bin/env python3
"""Run the disposable one-update vertical T4/T5 commissioning preflight."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import frozen_optic_motion_deterministic as deterministic  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402

EXPERIMENT = "vertical-t4t5-local-commissioning-preflight-v1"
PROTOCOL_COMMIT = "7d114e0"
EXPECTED_REGISTRATION_FILE_SHA256 = (
    "1403c552b371378a59b9cd9830d06ce144f336c7adf05d5061fd385dfc5dbb88"
)
EXPECTED_REGISTRATION_GENERATOR_SHA256 = (
    "b8a39a9e324a7aad71e8e4e21aaf494c393019d2af13ad2fbfb4a280fdd57ea4"
)

IDENTITY_CASES = (0, 1, 24, 25, 48, 49, 60, 61)
MINIBATCH_CASES = (0, 24, 48, 60)
FINITE_DIFFERENCE_STEP = 1.0e-3
FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT = 0.02
MINIMUM_STEP_IMPROVEMENT_FRACTION = 0.001
GRADIENT_NORM_CLIP = 1.0
BACKTRACK_MULTIPLIERS = (1.0, 0.5, 0.25, 0.125)
PEAK_RESERVED_LIMIT_BYTES = 14 * 1024**3
PROBES = (
    ("T4_gain_Mi1_to_T4c", "gain", 0),
    ("T5_gain_Tm1_to_T5c", "gain", 8),
    ("T4c_bias", "bias_offset", 0),
    ("T5c_bias", "bias_offset", 2),
    ("T4c_tau", "tau_ratio", 0),
    ("T5c_tau", "tau_ratio", 2),
)


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-preflight-001",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source": "original paired-dynamic-001 controller",
        "registration_semantic_sha256": registration.EXPECTED_MANIFEST_SEMANTIC_SHA256,
        "training_specs_semantic_sha256": registration.EXPECTED_SPLIT_SPEC_SHA256["training"],
        "identity_cases": list(IDENTITY_CASES),
        "fixed_effective_minibatch_cases": list(MINIBATCH_CASES),
        "case_interpretation": (
            "ON edge, OFF edge, and two mixed textures; every pair contains down/up "
            "moving branches and their matched stationary controls"
        ),
        "solver": {
            "method": "deterministic exponential Euler",
            "camera_hz": registration.CAMERA_HZ,
            "substeps_per_frame": registration.CNS_SUBSTEPS_PER_FRAME,
            "neural_state_updates_hz": (
                registration.CAMERA_HZ * registration.CNS_SUBSTEPS_PER_FRAME
            ),
            "frame_activation_checkpointing": True,
            "current_candidate_prefix_recomputed": True,
            "complete_graph_nodes": 165_122,
        },
        "loss": {
            "population_scale": ("frozen max(mean absolute source up/down opponent),1e-3)"),
            "direction_margin": commissioning.DIRECTION_MARGIN,
            "stationary": "squared stationary up-minus-down opponent over twice scale",
            "cell_dsi": (
                "preferred/null averaged across represented cases before each cell; "
                "denominator frozen from source"
            ),
            "activity": (
                "target min(source amplitude,1e-3), normalized by max(source amplitude,1e-3)"
            ),
            "reverse": (
                "opposite-sign margin using the reverse history's matched stationary "
                "baseline and source-frozen scale; edge pathways swap T4/T5"
            ),
            "averaging": (
                "cells within subtype, subtypes within pathway, pathways within pair, "
                "pairs within window, then integrated/terminal windows"
            ),
            "reverse_weight": commissioning.REVERSE_WEIGHT,
            "activity_weight": commissioning.ACTIVITY_WEIGHT,
            "identity_regularization_weight": commissioning.REGULARIZATION_WEIGHT,
            "labels_are_loss_only": True,
            "aircraft_or_motor_objective": False,
        },
        "finite_difference": {
            "central_step": FINITE_DIFFERENCE_STEP,
            "probes": [
                {"name": name, "parameter": parameter, "index": index}
                for name, parameter, index in PROBES
            ],
            "finite_nonzero_signal_required": True,
            "symmetric_relative_error_maximum": (FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT),
            "frozen_source_normalizations_reused": True,
        },
        "optimizer": {
            "name": "Adam",
            "betas": [0.0, 0.99],
            "eps": 1.0e-8,
            "learning_rates": {
                "gain": 0.02,
                "bias_offset": 0.01,
                "tau_ratio": 0.01,
            },
            "gradient_norm_clip": GRADIENT_NORM_CLIP,
            "backtrack_multipliers": list(BACKTRACK_MULTIPLIERS),
            "first_finite_nonincreasing_step": True,
        },
        "gates": {
            "exact_identity": True,
            "finite_difference_relative_error_maximum": (FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT),
            "accepted_step_minimum_improvement_fraction": (MINIMUM_STEP_IMPROVEMENT_FRACTION),
            "source_and_local_identity_restored": True,
            "peak_cuda_reserved_bytes_maximum": PEAK_RESERVED_LIMIT_BYTES,
        },
        "development_or_acceptance_specs_used": False,
        "development_or_acceptance_pixels_rendered": False,
        "candidate_retained": False,
        "hover_gate_or_promotion_authorized": False,
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_inputs(args: argparse.Namespace) -> dict[str, str]:
    expected = {
        args.graph: registration.EXPECTED_GRAPH_SHA256,
        args.checkpoint: registration.EXPECTED_CHECKPOINT_SHA256,
        args.raw_dir / registration.ANNOTATIONS_FILE: (registration.EXPECTED_ANNOTATIONS_SHA256),
        args.frozen_audit_report: registration.EXPECTED_FROZEN_AUDIT_REPORT_SHA256,
        args.registration: EXPECTED_REGISTRATION_FILE_SHA256,
        Path(registration.__file__).resolve(): EXPECTED_REGISTRATION_GENERATOR_SHA256,
    }
    observed = {}
    for path, digest in expected.items():
        if not path.is_file() or registration.file_sha256(path) != digest:
            raise SystemExit(f"locked vertical-motion preflight input changed: {path}")
        observed[registration.stable_path(path)] = digest
    return observed


def _max_response_difference(
    first: dict[str, Tensor], second: dict[str, Tensor]
) -> tuple[float, dict[str, float]]:
    differences = {name: float((first[name] - second[name]).abs().max()) for name in first}
    return max(differences.values()), differences


def _evaluate_cases(
    controller: torch.nn.Module,
    specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    *,
    indices: tuple[int, ...],
    device: torch.device,
    checkpoint_frames: bool,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    prefix = commissioning.neutral_prefix(
        controller, device=device, checkpoint_frames=checkpoint_frames
    )
    normal = []
    reverse = []
    for case in indices:
        normal.append(
            commissioning.evaluate_pair(
                controller,
                sequences[(case, False)],
                anatomy,
                device=device,
                checkpoint_frames=checkpoint_frames,
                prefix_state=prefix,
            )
        )
        reverse.append(
            commissioning.evaluate_pair(
                controller,
                sequences[(case, True)],
                anatomy,
                device=device,
                checkpoint_frames=checkpoint_frames,
                prefix_state=prefix,
            )
        )
    return normal, reverse


def _batch_loss(
    controller: commissioning.CommissionedController,
    batch_specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    references: dict[str, Any],
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
    checkpoint_frames: bool,
) -> tuple[Tensor, dict[str, Tensor]]:
    normal, reverse = _evaluate_cases(
        controller,
        batch_specs,
        sequences,
        anatomy,
        indices=MINIBATCH_CASES,
        device=device,
        checkpoint_frames=checkpoint_frames,
    )
    return commissioning.commissioning_loss(
        batch_specs, normal, reverse, references, anatomy, controller
    )


def _float_components(components: dict[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in components.items()}


def _parameter(controller: commissioning.CommissionedController, name: str) -> Tensor:
    return getattr(controller, name)


def finite_difference_report(
    controller: commissioning.CommissionedController,
    batch_specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    references: dict[str, Any],
    anatomy: dict[str, np.ndarray],
    analytic_gradients: dict[str, Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    identity = controller.parameter_values()
    rows = []
    for probe_name, parameter_name, index in PROBES:
        losses = {}
        for direction, sign in (("plus", 1.0), ("minus", -1.0)):
            controller.load_parameter_values(identity)
            with torch.no_grad():
                _parameter(controller, parameter_name)[index].add_(sign * FINITE_DIFFERENCE_STEP)
                loss, _ = _batch_loss(
                    controller,
                    batch_specs,
                    sequences,
                    references,
                    anatomy,
                    device=device,
                    checkpoint_frames=False,
                )
            losses[direction] = float(loss.detach().cpu())
        analytic = float(analytic_gradients[parameter_name][index])
        central = (losses["plus"] - losses["minus"]) / (2.0 * FINITE_DIFFERENCE_STEP)
        relative_error = abs(analytic - central) / max(abs(analytic), abs(central), 1.0e-12)
        finite_nonzero = bool(
            math.isfinite(analytic)
            and math.isfinite(central)
            and analytic != 0.0
            and central != 0.0
            and losses["plus"] != losses["minus"]
        )
        rows.append(
            {
                "name": probe_name,
                "parameter": parameter_name,
                "index": index,
                "plus_loss": losses["plus"],
                "minus_loss": losses["minus"],
                "analytic_gradient": analytic,
                "central_finite_difference": central,
                "symmetric_relative_error": relative_error,
                "finite_nonzero_signal": finite_nonzero,
                "pass": bool(
                    finite_nonzero and relative_error <= FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
                ),
            }
        )
    controller.load_parameter_values(identity)
    return {
        "pass": all(row["pass"] for row in rows),
        "step": FINITE_DIFFERENCE_STEP,
        "relative_error_limit": FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
        "frozen_source_normalizations_semantic_sha256": references["semantic_sha256"],
        "probes": rows,
    }


def optimizer_step_report(
    controller: commissioning.CommissionedController,
    batch_specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    references: dict[str, Any],
    anatomy: dict[str, np.ndarray],
    gradients: dict[str, Tensor],
    baseline_loss: float,
    *,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(
        [
            {"params": [controller.gain], "lr": 0.02, "name": "gain"},
            {
                "params": [controller.bias_offset],
                "lr": 0.01,
                "name": "bias_offset",
            },
            {"params": [controller.tau_ratio], "lr": 0.01, "name": "tau_ratio"},
        ],
        betas=(0.0, 0.99),
        eps=1.0e-8,
    )
    identity = controller.parameter_values()
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    base_learning_rates = (0.02, 0.01, 0.01)
    trials = []
    accepted = None
    for multiplier in BACKTRACK_MULTIPLIERS:
        controller.load_parameter_values(identity)
        optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
        for group, base_lr in zip(optimizer.param_groups, base_learning_rates, strict=True):
            group["lr"] = base_lr * multiplier
        for name in gradients:
            _parameter(controller, name).grad = gradients[name].to(device).clone()
        unclipped_norm = torch.nn.utils.clip_grad_norm_(
            (controller.gain, controller.bias_offset, controller.tau_ratio),
            GRADIENT_NORM_CLIP,
        )
        optimizer.step()
        controller.project_parameters()
        with torch.no_grad():
            candidate_loss, candidate_components = _batch_loss(
                controller,
                batch_specs,
                sequences,
                references,
                anatomy,
                device=device,
                checkpoint_frames=False,
            )
        loss_value = float(candidate_loss.detach().cpu())
        finite = bool(math.isfinite(loss_value))
        nonincreasing = finite and loss_value <= baseline_loss
        trial = {
            "multiplier": multiplier,
            "unclipped_gradient_norm": float(unclipped_norm.detach().cpu()),
            "loss": loss_value,
            "components": _float_components(candidate_components),
            "finite": finite,
            "nonincreasing": nonincreasing,
            "accepted": nonincreasing,
            "parameter_values": {
                name: values.tolist() for name, values in controller.parameter_values().items()
            },
        }
        trials.append(trial)
        if nonincreasing:
            accepted = trial
            break
    if accepted is None:
        improvement = None
        useful = False
    else:
        improvement = (baseline_loss - accepted["loss"]) / max(abs(baseline_loss), 1.0e-30)
        useful = bool(improvement >= MINIMUM_STEP_IMPROVEMENT_FRACTION)
    controller.load_parameter_values(identity)
    optimizer.zero_grad(set_to_none=True)
    return {
        "pass": bool(accepted is not None and useful),
        "baseline_loss": baseline_loss,
        "trials": trials,
        "accepted_multiplier": (accepted["multiplier"] if accepted is not None else None),
        "improvement_fraction": improvement,
        "minimum_improvement_fraction": MINIMUM_STEP_IMPROVEMENT_FRACTION,
        "local_identity_restored": controller.identity_restored(),
    }


def main() -> int:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    start_path = args.output_dir / "start.json"
    if report_path.exists() or start_path.exists():
        raise SystemExit("vertical-motion preflight is one-shot and already started")
    deterministic.configure_determinism()
    device = torch.device(args.device)
    runtime = deterministic.runtime_manifest(device)
    input_hashes = validate_inputs(args)
    registered = commissioning.load_registered_manifest(args.registration)
    # The registration is one JSON object, but only its training split is selected or used.
    training_specs = registered["stimuli"]["splits"]["training"]["specs"]
    if (
        registration.semantic_sha256(training_specs)
        != registration.EXPECTED_SPLIT_SPEC_SHA256["training"]
    ):
        raise SystemExit("training specifications changed")
    if max((*IDENTITY_CASES, *MINIBATCH_CASES)) >= len(training_specs):
        raise SystemExit("preflight case index lies outside the training split")
    batch_specs = [training_specs[index] for index in MINIBATCH_CASES]
    anatomy = commissioning.anatomy_arrays(args.graph, args.raw_dir / registration.ANNOTATIONS_FILE)
    source_state = commissioning.load_source_checkpoint(args.checkpoint)
    source_state_sha256 = commissioning.semantic_sha256(source_state)
    start = {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_state_sha256,
        "implementation_commit": _git_head(),
        "implementation_file_sha256": {
            registration.stable_path(Path(__file__)): registration.file_sha256(Path(__file__)),
            registration.stable_path(Path(commissioning.__file__)): registration.file_sha256(
                Path(commissioning.__file__)
            ),
        },
        "runtime": runtime,
    }
    registration.write_exclusive(start_path, start)

    started = perf_counter()
    source = commissioning.make_source_controller(args.graph, source_state, device=device)
    source_before = commissioning.semantic_sha256(source.state_dict())
    controller = commissioning.CommissionedController(source, anatomy).to(device)
    if not controller.identity_restored():
        raise SystemExit("commissioning controller did not start at source identity")
    torch.cuda.reset_peak_memory_stats(device)

    all_cases = tuple(sorted(set((*IDENTITY_CASES, *MINIBATCH_CASES))))
    sequences = {
        (case, reverse): commissioning.render_sequence(training_specs[case], reverse=reverse)
        for case in all_cases
        for reverse in (False, True)
    }
    rendered_sha256 = commissioning.rendered_subset_sha256(training_specs, IDENTITY_CASES)

    classification = "vertical_motion_preflight_exception_failed_closed"
    passed = False
    exception = None
    identity_report = None
    derivative_report = None
    step_report = None
    references = None
    baseline_components = None
    gradient_summary = None
    try:
        print(json.dumps({"stage": "exact_identity"}), flush=True)
        with torch.inference_mode():
            source_normal, source_reverse = _evaluate_cases(
                source,
                training_specs,
                sequences,
                anatomy,
                indices=IDENTITY_CASES,
                device=device,
                checkpoint_frames=False,
            )
            candidate_normal, candidate_reverse = _evaluate_cases(
                controller,
                training_specs,
                sequences,
                anatomy,
                indices=IDENTITY_CASES,
                device=device,
                checkpoint_frames=False,
            )
        source_tree = {
            "normal": [commissioning.response_tree_cpu(item) for item in source_normal],
            "reverse": [commissioning.response_tree_cpu(item) for item in source_reverse],
        }
        candidate_tree = {
            "normal": [commissioning.response_tree_cpu(item) for item in candidate_normal],
            "reverse": [commissioning.response_tree_cpu(item) for item in candidate_reverse],
        }
        differences = []
        named_differences = []
        for mode in ("normal", "reverse"):
            for source_item, candidate_item in zip(
                source_tree[mode], candidate_tree[mode], strict=True
            ):
                maximum, by_tensor = _max_response_difference(source_item, candidate_item)
                differences.append(maximum)
                named_differences.append(by_tensor)
        identity_report = {
            "pass": bool(
                max(differences) == 0.0
                and commissioning.semantic_sha256(source_tree)
                == commissioning.semantic_sha256(candidate_tree)
            ),
            "cases": list(IDENTITY_CASES),
            "normal_and_literal_reverse_checked": True,
            "maximum_absolute_difference": max(differences),
            "per_evaluation_tensor_maximum_absolute_differences": named_differences,
            "source_response_semantic_sha256": commissioning.semantic_sha256(source_tree),
            "candidate_response_semantic_sha256": commissioning.semantic_sha256(candidate_tree),
        }
        if not identity_report["pass"]:
            classification = "vertical_motion_preflight_identity_failed"
        else:
            # Ordinary CPU clones avoid passing inference-mode tensors into autograd's
            # saved source-normalization constants.
            source_by_case_normal = dict(zip(IDENTITY_CASES, source_tree["normal"], strict=True))
            source_by_case_reverse = dict(zip(IDENTITY_CASES, source_tree["reverse"], strict=True))
            batch_source_normal = [source_by_case_normal[index] for index in MINIBATCH_CASES]
            batch_source_reverse = [source_by_case_reverse[index] for index in MINIBATCH_CASES]
            references = commissioning.source_references(
                batch_specs,
                batch_source_normal,
                batch_source_reverse,
                anatomy,
            )

            print(json.dumps({"stage": "analytic_gradient"}), flush=True)
            controller.zero_grad(set_to_none=True)
            baseline_loss, components = _batch_loss(
                controller,
                batch_specs,
                sequences,
                references,
                anatomy,
                device=device,
                checkpoint_frames=True,
            )
            baseline_components = _float_components(components)
            baseline_loss.backward()
            gradients = {
                name: _parameter(controller, name).grad.detach().cpu().clone()
                for name in ("gain", "bias_offset", "tau_ratio")
            }
            gradient_summary = {
                name: {
                    "all_finite": bool(torch.isfinite(values).all()),
                    "norm": float(torch.linalg.vector_norm(values)),
                    "maximum_absolute": float(values.abs().max()),
                    "nonzero": int(torch.count_nonzero(values)),
                }
                for name, values in gradients.items()
            }
            gradients_valid = all(
                item["all_finite"] and item["nonzero"] > 0 for item in gradient_summary.values()
            )
            print(json.dumps({"stage": "central_finite_differences"}), flush=True)
            derivative_report = finite_difference_report(
                controller,
                batch_specs,
                sequences,
                references,
                anatomy,
                gradients,
                device=device,
            )
            if not gradients_valid or not derivative_report["pass"]:
                classification = "vertical_motion_preflight_derivative_failed"
            else:
                print(json.dumps({"stage": "one_update"}), flush=True)
                step_report = optimizer_step_report(
                    controller,
                    batch_specs,
                    sequences,
                    references,
                    anatomy,
                    gradients,
                    float(baseline_loss.detach().cpu()),
                    device=device,
                )
                if step_report["pass"]:
                    classification = "vertical_motion_preflight_passed"
                    passed = True
                else:
                    classification = "vertical_motion_preflight_step_failed"
    except Exception as error:
        exception = {"type": type(error).__name__, "message": str(error)}
    finally:
        identity = {
            "gain": torch.ones(16),
            "bias_offset": torch.zeros(4),
            "tau_ratio": torch.ones(4),
        }
        controller.load_parameter_values(identity)
        source_after = commissioning.semantic_sha256(source.state_dict())
        source_restored = bool(source_after == source_before and controller.identity_restored())
        peak_reserved = torch.cuda.max_memory_reserved(device)

    memory_pass = peak_reserved <= PEAK_RESERVED_LIMIT_BYTES
    if not source_restored:
        passed = False
        classification = "vertical_motion_preflight_source_restoration_failed"
    elif not memory_pass:
        passed = False
        classification = "vertical_motion_preflight_memory_gate_failed"
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "input_file_sha256": input_hashes,
        "implementation_commit": start["implementation_commit"],
        "implementation_file_sha256": start["implementation_file_sha256"],
        "runtime": runtime,
        "source_state_sha256": source_state_sha256,
        "source_controller_state_sha256_before": source_before,
        "source_controller_state_sha256_after": source_after,
        "source_and_local_identity_restored": source_restored,
        "training_specs_semantic_sha256": registration.semantic_sha256(training_specs),
        "rendered_training_identity_subset_float32_sha256": rendered_sha256,
        "development_specs_used": False,
        "acceptance_specs_used": False,
        "development_or_acceptance_pixels_rendered": False,
        "identity": identity_report,
        "frozen_source_normalizations_semantic_sha256": (
            references["semantic_sha256"] if references is not None else None
        ),
        "baseline_loss_components": baseline_components,
        "gradient_summary": gradient_summary,
        "finite_difference": derivative_report,
        "one_update": step_report,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_reserved_gibibytes": peak_reserved / 1024**3,
        "cuda_peak_reserved_limit_bytes": PEAK_RESERVED_LIMIT_BYTES,
        "cuda_peak_reserved_memory_pass": memory_pass,
        "exception": exception,
        "training_execution_authorized": passed,
        "candidate_retained": False,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": perf_counter() - started,
    }
    registration.write_exclusive(report_path, report)
    print(
        json.dumps(
            {
                "report": registration.stable_path(report_path),
                "classification": classification,
                "passed": passed,
                "peak_reserved_gibibytes": report["cuda_peak_reserved_gibibytes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
