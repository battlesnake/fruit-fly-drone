#!/usr/bin/env python3
"""Run the preregistered independent FP64 vertical-motion tau audit."""

from __future__ import annotations

import argparse
import json
import math
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
import preflight_vertical_motion_commissioning as preflight  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402
import vertical_motion_commissioning_fp64 as fp64  # noqa: E402

EXPERIMENT = "vertical-t4t5-tau-fp64-audit-v1"
PROTOCOL_COMMIT = "9b7cd72"
V1_REPORT = REPO_ROOT / "runs/optic-motion/vertical-motion-preflight-001/report.json"
V2_REPORT = REPO_ROOT / "runs/optic-motion/vertical-motion-preflight-002/report.json"
LADDER_REPORT = REPO_ROOT / "runs/optic-motion/vertical-motion-derivative-ladder-001/report.json"
EXPECTED_REPORT_SHA256 = {
    V1_REPORT: "661868629adae98e1ee664a4ec4c4e50607137097a4e4dcdb07164510917f8e4",
    V2_REPORT: "df8396dd18e83a66e206dbfba3d9276bccdc409d795745e0cb4c4c2dc6a05248",
    LADDER_REPORT: "fb1038ee688874ebbc369a3d7f36bd80b63a7371b9bd4b3a0aabec933db1c167",
}
EXPECTED_NORMALIZATIONS_SHA256 = "796cf8f0488edff30dd34bf556268b4faac6ad5564244a1ce9b13cfcb131e475"
EXPECTED_RENDERED_SUBSET_SHA256 = "c04bb10318cd3b0620c9a71abf7a7420afe166a0eff2cf1d1d154d9a41440576"
EXPECTED_FP32_ANALYTIC = {
    "T4c_tau": 0.020206790417432785,
    "T5c_tau": -0.12225273251533508,
}
PROBES = (
    ("T4c_tau", 0),
    ("T5c_tau", 2),
)
STEPS = (0.008, 0.004, 0.002, 0.001)
BASELINE_REPEATS = 3
LOSS_RESOLUTION_ULPS = 8
DERIVATIVE_RELATIVE_ERROR_LIMIT = 0.02
CROSS_PRECISION_RELATIVE_ERROR_LIMIT = 0.02
FP32_ANALYTIC_ABSOLUTE_TOLERANCE = 1.0e-6
ONE_STEP_RELATIVE_ERROR_LIMIT = 1.0e-10


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
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-tau-fp64-001",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "locked_reports": {
            registration.stable_path(path): digest
            for path, digest in EXPECTED_REPORT_SHA256.items()
        },
        "source_protocol": preflight.protocol_manifest(),
        "training_case_indices": list(preflight.MINIBATCH_CASES),
        "rendered_subset_identity_hash_reproduced_only": list(preflight.IDENTITY_CASES),
        "precision_reference": {
            "source_coefficients": "effective checkpoint float32 values promoted to float64",
            "pixels": "exact registered float32 rendering promoted to float64",
            "source_normalizations": "exact v1 float32 tensors promoted to float64",
            "sensory_recurrence_state_alpha_response_and_loss_dtype": "float64",
            "selected_state_arithmetic": "direct index_copy",
            "complete_graph_nodes": 165_122,
            "camera_hz": registration.CAMERA_HZ,
            "substeps_per_frame": registration.CNS_SUBSTEPS_PER_FRAME,
            "neural_state_updates_hz": (
                registration.CAMERA_HZ * registration.CNS_SUBSTEPS_PER_FRAME
            ),
            "objective_or_biological_changes": [],
        },
        "fp64_tau_ladder": {
            "probes": [{"name": name, "tau_ratio_index": index} for name, index in PROBES],
            "steps": list(STEPS),
            "baseline_repeats": BASELINE_REPEATS,
            "resolution_floor": {
                "definition": "max(repeat range, eight float64 ULPs at mean baseline)",
                "float64_ulps": LOSS_RESOLUTION_ULPS,
            },
            "relative_error_maximum": DERIVATIVE_RELATIVE_ERROR_LIMIT,
            "analytic_sign_required": True,
            "resolved_required": True,
            "two_adjacent_steps_required": True,
            "record_actual_offsets_total_and_all_component_losses": True,
        },
        "cross_precision": {
            "same_sign_required": True,
            "symmetric_relative_error_maximum": CROSS_PRECISION_RELATIVE_ERROR_LIMIT,
            "fp32_analytic_absolute_reproduction_tolerance": (FP32_ANALYTIC_ABSOLUTE_TOLERANCE),
            "expected_fp32_analytic": EXPECTED_FP32_ANALYTIC,
            "expected_source_normalizations_sha256": EXPECTED_NORMALIZATIONS_SHA256,
            "expected_rendered_subset_sha256": EXPECTED_RENDERED_SUBSET_SHA256,
        },
        "fixed_state_one_step": {
            "target_node": "first hash-locked target index in each probed subtype",
            "state": 0.125,
            "target": -0.375,
            "tau_ratio": 1.0,
            "clamp_inactive_required": True,
            "finite_nonzero_and_same_sign_required": True,
            "symmetric_relative_error_maximum": ONE_STEP_RELATIVE_ERROR_LIMIT,
        },
        "conditional_one_update": {
            "allowed_only_after_all_numerical_gates": True,
            "implementation": "original production float32 disposable Adam/backtracking step",
            "optimizer": preflight.protocol_manifest()["optimizer"],
            "minimum_improvement_fraction": preflight.MINIMUM_STEP_IMPROVEMENT_FRACTION,
        },
        "peak_cuda_reserved_bytes_maximum": preflight.PEAK_RESERVED_LIMIT_BYTES,
        "source_and_local_identity_restored": True,
        "development_or_acceptance_specs_used": False,
        "development_or_acceptance_pixels_rendered": False,
        "candidate_retained": False,
        "hover_gate_or_promotion_authorized": False,
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, str]:
    observed = preflight.validate_inputs(args)
    for path, digest in EXPECTED_REPORT_SHA256.items():
        if not path.is_file() or registration.file_sha256(path) != digest:
            raise SystemExit(f"locked FP64 tau-audit input changed: {path}")
        observed[registration.stable_path(path)] = digest
    return observed


def loss_resolution_floor(values: list[float]) -> dict[str, float]:
    if len(values) != BASELINE_REPEATS or not all(math.isfinite(value) for value in values):
        raise ValueError("FP64 resolution floor requires three finite losses")
    repeat_range = max(values) - min(values)
    mean = float(np.mean(np.asarray(values, dtype=np.float64)))
    ulp = float(np.spacing(np.float64(abs(mean))))
    ulp_floor = LOSS_RESOLUTION_ULPS * ulp
    return {
        "baseline_mean": mean,
        "repeat_range": repeat_range,
        "float64_ulp_at_baseline": ulp,
        "float64_ulp_floor": ulp_floor,
        "loss_resolution_floor": max(repeat_range, ulp_floor),
    }


def qualify_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_step = {row["step"]: row for row in rows}
    adjacent = []
    for coarse, fine in zip(STEPS[:-1], STEPS[1:], strict=True):
        pair_pass = bool(by_step[coarse]["pass"] and by_step[fine]["pass"])
        adjacent.append({"steps": [coarse, fine], "pass": pair_pass})
    passing = [item["steps"] for item in adjacent if item["pass"]]
    return {
        "pass": bool(passing),
        "passing_adjacent_step_pairs": passing,
        "adjacent_step_pairs": adjacent,
    }


def _float_components(components: dict[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in components.items()}


def _response_tree(responses: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
    return [commissioning.response_tree_cpu(item) for item in responses]


def _identity_report(
    source_normal: list[dict[str, Tensor]],
    source_reverse: list[dict[str, Tensor]],
    candidate_normal: list[dict[str, Tensor]],
    candidate_reverse: list[dict[str, Tensor]],
) -> dict[str, Any]:
    source = {"normal": _response_tree(source_normal), "reverse": _response_tree(source_reverse)}
    candidate = {
        "normal": _response_tree(candidate_normal),
        "reverse": _response_tree(candidate_reverse),
    }
    differences = []
    for mode in ("normal", "reverse"):
        for left, right in zip(source[mode], candidate[mode], strict=True):
            maximum, _ = preflight._max_response_difference(left, right)
            differences.append(maximum)
    source_hash = commissioning.semantic_sha256(source)
    candidate_hash = commissioning.semantic_sha256(candidate)
    return {
        "pass": bool(max(differences) == 0.0 and source_hash == candidate_hash),
        "maximum_absolute_difference": max(differences),
        "source_response_semantic_sha256": source_hash,
        "candidate_response_semantic_sha256": candidate_hash,
    }


def _cross_precision(fp32_values: dict[str, float], fp64_values: dict[str, float]) -> dict:
    rows = {}
    for name, _ in PROBES:
        first = fp32_values[name]
        second = fp64_values[name]
        relative = abs(first - second) / max(abs(first), abs(second), 1.0e-300)
        same_sign = bool(
            first != 0.0
            and second != 0.0
            and math.copysign(1.0, first) == math.copysign(1.0, second)
        )
        rows[name] = {
            "fp32_analytic_gradient": first,
            "fp64_analytic_gradient": second,
            "symmetric_relative_error": relative,
            "same_sign": same_sign,
            "pass": bool(same_sign and relative <= CROSS_PRECISION_RELATIVE_ERROR_LIMIT),
        }
    return {"pass": all(row["pass"] for row in rows.values()), "probes": rows}


def main() -> int:
    args = parse_args()
    start_path = args.output_dir / "start.json"
    report_path = args.output_dir / "report.json"
    if start_path.exists() or report_path.exists():
        raise SystemExit("vertical-motion FP64 tau audit is one-shot and already started")
    deterministic.configure_determinism()
    device = torch.device(args.device)
    runtime = deterministic.runtime_manifest(device)
    input_hashes = validate_inputs(args)
    registered = commissioning.load_registered_manifest(args.registration)
    training_specs = registered["stimuli"]["splits"]["training"]["specs"]
    batch_specs = [training_specs[index] for index in preflight.MINIBATCH_CASES]
    anatomy = commissioning.anatomy_arrays(args.graph, args.raw_dir / registration.ANNOTATIONS_FILE)
    source_state = commissioning.load_source_checkpoint(args.checkpoint)
    source_state_sha256 = commissioning.semantic_sha256(source_state)
    implementation_paths = (Path(__file__), Path(fp64.__file__))
    start = {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_state_sha256,
        "implementation_commit": preflight._git_head(),
        "implementation_file_sha256": {
            registration.stable_path(path): registration.file_sha256(path)
            for path in implementation_paths
        },
        "runtime": runtime,
    }
    registration.write_exclusive(start_path, start)

    sequences = {
        (case, reverse): commissioning.render_sequence(training_specs[case], reverse=reverse)
        for case in preflight.MINIBATCH_CASES
        for reverse in (False, True)
    }
    rendered_sha256 = commissioning.rendered_subset_sha256(training_specs, preflight.IDENTITY_CASES)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    classification = "vertical_motion_tau_fp64_exception_failed_closed"
    passed = False
    exception = None
    references32 = None
    references64 = None
    reference_promotion = None
    fp32_reproduction = None
    fp32_analytic_values = None
    fp32_gradients = None
    fp32_baseline_loss = None
    fp32_baseline_components = None
    fp64_identity = None
    fp64_analytic = None
    fp64_baseline_components = None
    fp64_baseline_repeats = None
    resolution = None
    one_step = None
    cross_precision = None
    ladder = None
    qualification = None
    step_report = None
    source32 = None
    controller32 = None
    source64 = None
    controller64 = None
    source32_before = None
    source64_before = None
    try:
        print(json.dumps({"stage": "fp32_reproduction_and_references"}), flush=True)
        source32 = commissioning.make_source_controller(args.graph, source_state, device=device)
        source32_before = commissioning.semantic_sha256(source32.state_dict())
        controller32 = commissioning.CommissionedController(source32, anatomy).to(device)
        with torch.inference_mode():
            source32_normal, source32_reverse = preflight._evaluate_cases(
                source32,
                training_specs,
                sequences,
                anatomy,
                indices=preflight.MINIBATCH_CASES,
                device=device,
                checkpoint_frames=False,
            )
        references32 = commissioning.source_references(
            batch_specs,
            _response_tree(source32_normal),
            _response_tree(source32_reverse),
            anatomy,
        )
        controller32.zero_grad(set_to_none=True)
        baseline32, components32 = preflight._batch_loss(
            controller32,
            batch_specs,
            sequences,
            references32,
            anatomy,
            device=device,
            checkpoint_frames=True,
        )
        baseline32.backward()
        fp32_gradients = {
            name: getattr(controller32, name).grad.detach().cpu().clone()
            for name in ("gain", "bias_offset", "tau_ratio")
        }
        fp32_analytic_values = {
            name: float(fp32_gradients["tau_ratio"][index]) for name, index in PROBES
        }
        fp32_differences = {
            name: abs(fp32_analytic_values[name] - expected)
            for name, expected in EXPECTED_FP32_ANALYTIC.items()
        }
        fp32_reproduction = {
            "pass": bool(
                references32["semantic_sha256"] == EXPECTED_NORMALIZATIONS_SHA256
                and rendered_sha256 == EXPECTED_RENDERED_SUBSET_SHA256
                and max(fp32_differences.values()) <= FP32_ANALYTIC_ABSOLUTE_TOLERANCE
                and all(torch.isfinite(values).all() for values in fp32_gradients.values())
            ),
            "analytic_gradients": fp32_analytic_values,
            "analytic_absolute_differences": fp32_differences,
            "analytic_absolute_tolerance": FP32_ANALYTIC_ABSOLUTE_TOLERANCE,
            "all_parameter_gradients_finite": all(
                bool(torch.isfinite(values).all()) for values in fp32_gradients.values()
            ),
        }
        fp32_baseline_loss = float(baseline32.detach().cpu())
        fp32_baseline_components = _float_components(components32)
        references64 = fp64.promote_tree_fp64(references32)
        reference_promotion = {
            "roundtrip_maximum_absolute_difference": (
                fp64.promotion_roundtrip_maximum_difference(references32, references64)
            ),
            "fp32_semantic_sha256": references32["semantic_sha256"],
            "fp64_promoted_semantic_sha256": commissioning.semantic_sha256(
                {key: value for key, value in references64.items() if key != "semantic_sha256"}
            ),
        }
        del baseline32, components32, source32_normal, source32_reverse
        controller32.to("cpu")
        torch.cuda.empty_cache()

        if not fp32_reproduction["pass"]:
            classification = "vertical_motion_tau_fp64_fp32_reproduction_failed"
        elif reference_promotion["roundtrip_maximum_absolute_difference"] != 0.0:
            classification = "vertical_motion_tau_fp64_reference_promotion_failed"
        else:
            print(json.dumps({"stage": "fp64_identity"}), flush=True)
            source64 = commissioning.make_source_controller(
                args.graph, source_state, device=device
            ).to(dtype=torch.float64)
            source64_before = commissioning.semantic_sha256(source64.state_dict())
            controller64 = fp64.FP64CommissionedController(source64, anatomy).to(device)
            with torch.inference_mode():
                source64_normal, source64_reverse = fp64.evaluate_cases(
                    source64,
                    training_specs,
                    sequences,
                    anatomy,
                    indices=preflight.MINIBATCH_CASES,
                    device=device,
                    checkpoint_frames=False,
                )
                candidate64_normal, candidate64_reverse = fp64.evaluate_cases(
                    controller64,
                    training_specs,
                    sequences,
                    anatomy,
                    indices=preflight.MINIBATCH_CASES,
                    device=device,
                    checkpoint_frames=False,
                )
            fp64_identity = _identity_report(
                source64_normal,
                source64_reverse,
                candidate64_normal,
                candidate64_reverse,
            )
            del source64_normal, source64_reverse, candidate64_normal, candidate64_reverse
            if not fp64_identity["pass"]:
                classification = "vertical_motion_tau_fp64_identity_failed"
            else:
                print(json.dumps({"stage": "fp64_analytic"}), flush=True)
                controller64.zero_grad(set_to_none=True)
                baseline64, components64 = fp64.batch_loss(
                    controller64,
                    batch_specs,
                    sequences,
                    references64,
                    anatomy,
                    device=device,
                    checkpoint_frames=True,
                )
                component_gradients = {}
                for component_name, component in components64.items():
                    gradient = torch.autograd.grad(
                        component,
                        controller64.tau_ratio,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    component_gradients[component_name] = (
                        (torch.zeros_like(controller64.tau_ratio) if gradient is None else gradient)
                        .detach()
                        .cpu()
                    )
                baseline64.backward()
                all_fp64_gradients = {
                    name: getattr(controller64, name).grad.detach().cpu().clone()
                    for name in ("gain", "bias_offset", "tau_ratio")
                }
                fp64_analytic_values = {
                    name: float(all_fp64_gradients["tau_ratio"][index]) for name, index in PROBES
                }
                fp64_analytic = {
                    "probe_gradients": fp64_analytic_values,
                    "tau_component_gradients": {
                        name: {
                            component_name: float(values[index])
                            for component_name, values in component_gradients.items()
                        }
                        for name, index in PROBES
                    },
                    "all_parameter_gradients_finite": all(
                        bool(torch.isfinite(values).all()) for values in all_fp64_gradients.values()
                    ),
                }
                fp64_baseline_components = _float_components(components64)
                cross_precision = _cross_precision(fp32_analytic_values, fp64_analytic_values)
                del baseline64, components64

                print(json.dumps({"stage": "fixed_state_one_step"}), flush=True)
                one_step_probes = {}
                selected_indices = controller64.selected_target_indices
                selected_subtypes = controller64.selected_target_subtypes
                for name, subtype in PROBES:
                    node = selected_indices[selected_subtypes == subtype][0]
                    result = fp64.fixed_state_tau_derivative(
                        source64.time_constant[node], source64.neural_dt
                    )
                    result["node_index"] = int(node.detach().cpu())
                    result["pass"] = bool(
                        result["finite_nonzero"]
                        and result["clamp_inactive"]
                        and result["sign_matches"]
                        and result["symmetric_relative_error"] <= ONE_STEP_RELATIVE_ERROR_LIMIT
                    )
                    one_step_probes[name] = result
                one_step = {
                    "pass": all(item["pass"] for item in one_step_probes.values()),
                    "probes": one_step_probes,
                }

                print(json.dumps({"stage": "fp64_baseline_repeats"}), flush=True)
                repeated = []
                for _ in range(BASELINE_REPEATS):
                    with torch.no_grad():
                        value, repeated_components = fp64.batch_loss(
                            controller64,
                            batch_specs,
                            sequences,
                            references64,
                            anatomy,
                            device=device,
                            checkpoint_frames=False,
                        )
                    repeated.append(
                        {
                            "loss": float(value.detach().cpu()),
                            "components": _float_components(repeated_components),
                        }
                    )
                fp64_baseline_repeats = repeated
                resolution = loss_resolution_floor([item["loss"] for item in repeated])

                print(json.dumps({"stage": "fp64_tau_ladder"}), flush=True)
                identity_parameters = controller64.parameter_values()
                ladder = {}
                for probe_name, index in PROBES:
                    rows = []
                    for step in STEPS:
                        directions = {}
                        for direction, sign in (("plus", 1.0), ("minus", -1.0)):
                            controller64.load_parameter_values(identity_parameters)
                            with torch.no_grad():
                                identity_value = float(controller64.tau_ratio[index].cpu())
                                controller64.tau_ratio[index].add_(sign * step)
                                materialized_value = float(controller64.tau_ratio[index].cpu())
                                value, component_values = fp64.batch_loss(
                                    controller64,
                                    batch_specs,
                                    sequences,
                                    references64,
                                    anatomy,
                                    device=device,
                                    checkpoint_frames=False,
                                )
                            directions[direction] = {
                                "requested_offset": sign * step,
                                "materialized_value": materialized_value,
                                "materialized_offset": materialized_value - identity_value,
                                "loss": float(value.detach().cpu()),
                                "components": _float_components(component_values),
                            }
                        actual_span = (
                            directions["plus"]["materialized_value"]
                            - directions["minus"]["materialized_value"]
                        )
                        central = (
                            directions["plus"]["loss"] - directions["minus"]["loss"]
                        ) / actual_span
                        analytic_value = fp64_analytic_values[probe_name]
                        relative_error = abs(central - analytic_value) / max(
                            abs(central), abs(analytic_value), 1.0e-300
                        )
                        loss_difference = abs(
                            directions["plus"]["loss"] - directions["minus"]["loss"]
                        )
                        resolved = loss_difference >= resolution["loss_resolution_floor"]
                        sign_matches = bool(
                            central != 0.0
                            and analytic_value != 0.0
                            and math.copysign(1.0, central) == math.copysign(1.0, analytic_value)
                        )
                        rows.append(
                            {
                                "step": step,
                                "plus": directions["plus"],
                                "minus": directions["minus"],
                                "actual_parameter_span": actual_span,
                                "central_finite_difference": central,
                                "analytic_gradient": analytic_value,
                                "symmetric_relative_error": relative_error,
                                "absolute_loss_difference": loss_difference,
                                "resolved": resolved,
                                "analytic_sign_matches": sign_matches,
                                "pass": bool(
                                    resolved
                                    and sign_matches
                                    and relative_error <= DERIVATIVE_RELATIVE_ERROR_LIMIT
                                ),
                            }
                        )
                    ladder[probe_name] = {
                        "tau_ratio_index": index,
                        "rows": rows,
                        "qualification": qualify_probe(rows),
                    }
                controller64.load_parameter_values(identity_parameters)
                qualification = {
                    "pass": bool(
                        fp32_reproduction["pass"]
                        and reference_promotion["roundtrip_maximum_absolute_difference"] == 0.0
                        and fp64_identity["pass"]
                        and fp64_analytic["all_parameter_gradients_finite"]
                        and one_step["pass"]
                        and cross_precision["pass"]
                        and all(item["qualification"]["pass"] for item in ladder.values())
                    ),
                    "probes": {name: item["qualification"] for name, item in ladder.items()},
                }
                if not fp64_analytic["all_parameter_gradients_finite"]:
                    classification = "vertical_motion_tau_fp64_analytic_failed"
                elif not one_step["pass"]:
                    classification = "vertical_motion_tau_fp64_one_step_failed"
                elif not cross_precision["pass"]:
                    classification = "vertical_motion_tau_fp64_cross_precision_failed"
                elif not qualification["pass"]:
                    classification = "vertical_motion_tau_fp64_ladder_failed"
                else:
                    print(json.dumps({"stage": "conditional_fp32_one_update"}), flush=True)
                    controller64.to("cpu")
                    source64.to("cpu")
                    torch.cuda.empty_cache()
                    controller32.to(device)
                    step_report = preflight.optimizer_step_report(
                        controller32,
                        batch_specs,
                        sequences,
                        references32,
                        anatomy,
                        fp32_gradients,
                        fp32_baseline_loss,
                        device=device,
                    )
                    if step_report["pass"]:
                        classification = "vertical_motion_tau_fp64_audit_passed"
                        passed = True
                    else:
                        classification = "vertical_motion_tau_fp64_step_failed"
    except Exception as error:
        exception = {"type": type(error).__name__, "message": str(error)}
    finally:
        if controller32 is not None:
            controller32.load_parameter_values(
                {
                    "gain": torch.ones(16),
                    "bias_offset": torch.zeros(4),
                    "tau_ratio": torch.ones(4),
                }
            )
        if controller64 is not None:
            controller64.load_parameter_values(
                {
                    "gain": torch.ones(16, dtype=torch.float64),
                    "bias_offset": torch.zeros(4, dtype=torch.float64),
                    "tau_ratio": torch.ones(4, dtype=torch.float64),
                }
            )
        source32_after = (
            commissioning.semantic_sha256(source32.state_dict()) if source32 is not None else None
        )
        source64_after = (
            commissioning.semantic_sha256(source64.state_dict()) if source64 is not None else None
        )
        source_restored = bool(
            source32_before is not None
            and source32_after == source32_before
            and controller32 is not None
            and controller32.identity_restored()
            and (
                source64 is None
                or (
                    source64_after == source64_before
                    and controller64 is not None
                    and controller64.identity_restored()
                )
            )
        )
        peak_reserved = torch.cuda.max_memory_reserved(device)

    memory_pass = peak_reserved <= preflight.PEAK_RESERVED_LIMIT_BYTES
    if not source_restored:
        passed = False
        classification = "vertical_motion_tau_fp64_restoration_failed"
    elif not memory_pass:
        passed = False
        classification = "vertical_motion_tau_fp64_memory_failed"
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
        "source_fp32_state_sha256_before": source32_before,
        "source_fp32_state_sha256_after": source32_after,
        "source_fp64_state_sha256_before": source64_before,
        "source_fp64_state_sha256_after": source64_after,
        "source_and_local_identity_restored": source_restored,
        "rendered_training_identity_subset_float32_sha256": rendered_sha256,
        "frozen_source_normalizations_semantic_sha256": (
            references32["semantic_sha256"] if references32 is not None else None
        ),
        "reference_promotion": reference_promotion,
        "fp32_reproduction": fp32_reproduction,
        "fp32_baseline_loss": fp32_baseline_loss,
        "fp32_baseline_components": fp32_baseline_components,
        "fp64_identity": fp64_identity,
        "fp64_analytic": fp64_analytic,
        "fp64_baseline_components": fp64_baseline_components,
        "fp64_baseline_repeats": fp64_baseline_repeats,
        "fp64_loss_resolution": resolution,
        "fixed_state_one_step": one_step,
        "cross_precision": cross_precision,
        "fp64_tau_ladder": ladder,
        "qualification": qualification,
        "one_update": step_report,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_reserved_gibibytes": peak_reserved / 1024**3,
        "cuda_peak_reserved_memory_pass": memory_pass,
        "development_specs_used": False,
        "acceptance_specs_used": False,
        "development_or_acceptance_pixels_rendered": False,
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
                "qualification": qualification,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
