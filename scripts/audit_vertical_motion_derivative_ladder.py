#!/usr/bin/env python3
"""Run the preregistered multi-scale vertical-motion derivative diagnostic."""

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

EXPERIMENT = "vertical-t4t5-derivative-conditioning-ladder-v1"
PROTOCOL_COMMIT = "356cd4a"
V1_REPORT = REPO_ROOT / "runs/optic-motion/vertical-motion-preflight-001/report.json"
V2_REPORT = REPO_ROOT / "runs/optic-motion/vertical-motion-preflight-002/report.json"
EXPECTED_V1_REPORT_SHA256 = "661868629adae98e1ee664a4ec4c4e50607137097a4e4dcdb07164510917f8e4"
EXPECTED_V2_REPORT_SHA256 = "df8396dd18e83a66e206dbfba3d9276bccdc409d795745e0cb4c4c2dc6a05248"
EXPECTED_NORMALIZATIONS_SHA256 = "796cf8f0488edff30dd34bf556268b4faac6ad5564244a1ce9b13cfcb131e475"
EXPECTED_RENDERED_SUBSET_SHA256 = "c04bb10318cd3b0620c9a71abf7a7420afe166a0eff2cf1d1d154d9a41440576"
STEPS = (0.008, 0.004, 0.002, 0.001, 0.0005, 0.00025)
BASELINE_REPEATS = 3
LOSS_RESOLUTION_ULPS = 8
DERIVATIVE_RELATIVE_ERROR_LIMIT = 0.02
ANALYTIC_REPRODUCTION_ABSOLUTE_TOLERANCE = 1.0e-6
LOSS_REPRODUCTION_ABSOLUTE_TOLERANCE = 2.0e-6

EXPECTED_V1_ANALYTIC = {
    "T4_gain_Mi1_to_T4c": 1.57722008228302,
    "T5_gain_Tm1_to_T5c": -0.15678033232688904,
    "T4c_bias": 0.6460350751876831,
    "T5c_bias": 0.2906070053577423,
    "T4c_tau": 0.020206790417432785,
    "T5c_tau": -0.12225273251533508,
}
EXPECTED_V1_H001_LOSSES = {
    "T4_gain_Mi1_to_T4c": (0.9979069828987122, 0.9947404861450195),
    "T5_gain_Tm1_to_T5c": (0.9961665272712708, 0.9964777827262878),
    "T4c_bias": (0.9969621300697327, 0.9956749677658081),
    "T5c_bias": (0.9966110587120056, 0.9960295557975769),
    "T4c_tau": (0.996341347694397, 0.9962962865829468),
    "T5c_tau": (0.9961917400360107, 0.9964433908462524),
}


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
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-derivative-ladder-001",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    source = preflight.protocol_manifest()
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "locked_preflight_reports": {
            "v1": EXPECTED_V1_REPORT_SHA256,
            "v2": EXPECTED_V2_REPORT_SHA256,
        },
        "source_protocol": source,
        "state_arithmetic": "original v1 source-next-state plus alpha-delta correction",
        "changes_to_source_science_loss_mask_or_thresholds": [],
        "steps": list(STEPS),
        "baseline_repeats": BASELINE_REPEATS,
        "resolution_floor": {
            "repeat_range": "maximum minus minimum repeated unperturbed total loss",
            "float32_ulps": LOSS_RESOLUTION_ULPS,
            "definition": "max(repeat range, eight float32 ULPs at mean baseline)",
            "resolved": "absolute plus/minus total-loss difference >= resolution floor",
        },
        "record_actual_materialized_float32_offsets": True,
        "record_total_and_every_component_loss": True,
        "tau_componentwise_analytic_gradients": ["T4c_tau", "T5c_tau"],
        "v1_reproduction": {
            "analytic_absolute_tolerance": (ANALYTIC_REPRODUCTION_ABSOLUTE_TOLERANCE),
            "h_0.001_plus_minus_loss_absolute_tolerance": (LOSS_REPRODUCTION_ABSOLUTE_TOLERANCE),
            "normalizations_sha256": EXPECTED_NORMALIZATIONS_SHA256,
            "rendered_subset_sha256": EXPECTED_RENDERED_SUBSET_SHA256,
        },
        "probe_qualification": {
            "relative_error_maximum": DERIVATIVE_RELATIVE_ERROR_LIMIT,
            "analytic_sign_required": True,
            "resolved_required": True,
            "two_adjacent_steps_required": True,
            "all_six_probes_required": True,
        },
        "one_update": source["optimizer"],
        "minimum_step_improvement_fraction": (preflight.MINIMUM_STEP_IMPROVEMENT_FRACTION),
        "source_and_local_identity_restored": True,
        "peak_cuda_reserved_bytes_maximum": preflight.PEAK_RESERVED_LIMIT_BYTES,
        "development_or_acceptance_specs_used": False,
        "development_or_acceptance_pixels_rendered": False,
        "candidate_retained": False,
        "hover_gate_or_promotion_authorized": False,
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, str]:
    observed = preflight.validate_inputs(args)
    for path, digest in (
        (V1_REPORT, EXPECTED_V1_REPORT_SHA256),
        (V2_REPORT, EXPECTED_V2_REPORT_SHA256),
    ):
        if not path.is_file() or registration.file_sha256(path) != digest:
            raise SystemExit(f"locked derivative-ladder input changed: {path}")
        observed[registration.stable_path(path)] = digest
    return observed


def loss_resolution_floor(values: list[float]) -> dict[str, float]:
    if len(values) != BASELINE_REPEATS or not all(math.isfinite(item) for item in values):
        raise ValueError("resolution floor requires three finite baseline losses")
    repeat_range = max(values) - min(values)
    mean = float(np.mean(values))
    ulp = float(np.spacing(np.float32(abs(mean))))
    ulp_floor = LOSS_RESOLUTION_ULPS * ulp
    return {
        "baseline_mean": mean,
        "repeat_range": repeat_range,
        "float32_ulp_at_baseline": ulp,
        "float32_ulp_floor": ulp_floor,
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


def _evaluate(
    controller: commissioning.CommissionedController,
    batch_specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    references: dict[str, Any],
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
    checkpoint_frames: bool,
) -> tuple[Tensor, dict[str, Tensor]]:
    return preflight._batch_loss(
        controller,
        batch_specs,
        sequences,
        references,
        anatomy,
        device=device,
        checkpoint_frames=checkpoint_frames,
    )


def _identity_and_references(
    source,
    controller,
    training_specs,
    sequences,
    anatomy,
    *,
    device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with torch.inference_mode():
        source_normal, source_reverse = preflight._evaluate_cases(
            source,
            training_specs,
            sequences,
            anatomy,
            indices=preflight.IDENTITY_CASES,
            device=device,
            checkpoint_frames=False,
        )
        candidate_normal, candidate_reverse = preflight._evaluate_cases(
            controller,
            training_specs,
            sequences,
            anatomy,
            indices=preflight.IDENTITY_CASES,
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
    for mode in ("normal", "reverse"):
        for source_item, candidate_item in zip(
            source_tree[mode], candidate_tree[mode], strict=True
        ):
            maximum, _ = preflight._max_response_difference(source_item, candidate_item)
            differences.append(maximum)
    source_hash = commissioning.semantic_sha256(source_tree)
    candidate_hash = commissioning.semantic_sha256(candidate_tree)
    identity = {
        "pass": bool(max(differences) == 0.0 and source_hash == candidate_hash),
        "maximum_absolute_difference": max(differences),
        "source_response_semantic_sha256": source_hash,
        "candidate_response_semantic_sha256": candidate_hash,
    }
    source_by_case_normal = dict(zip(preflight.IDENTITY_CASES, source_tree["normal"], strict=True))
    source_by_case_reverse = dict(
        zip(preflight.IDENTITY_CASES, source_tree["reverse"], strict=True)
    )
    references = commissioning.source_references(
        [training_specs[index] for index in preflight.MINIBATCH_CASES],
        [source_by_case_normal[index] for index in preflight.MINIBATCH_CASES],
        [source_by_case_reverse[index] for index in preflight.MINIBATCH_CASES],
        anatomy,
    )
    return identity, references


def main() -> int:
    args = parse_args()
    start_path = args.output_dir / "start.json"
    report_path = args.output_dir / "report.json"
    if start_path.exists() or report_path.exists():
        raise SystemExit("vertical-motion derivative ladder is one-shot and already started")
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
    start = {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_state_sha256,
        "implementation_commit": preflight._git_head(),
        "implementation_file_sha256": {
            registration.stable_path(Path(__file__)): registration.file_sha256(Path(__file__))
        },
        "runtime": runtime,
    }
    registration.write_exclusive(start_path, start)

    source = commissioning.make_source_controller(args.graph, source_state, device=device)
    source_before = commissioning.semantic_sha256(source.state_dict())
    controller = commissioning.CommissionedController(source, anatomy).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    sequences = {
        (case, reverse): commissioning.render_sequence(training_specs[case], reverse=reverse)
        for case in preflight.IDENTITY_CASES
        for reverse in (False, True)
    }
    rendered_sha256 = commissioning.rendered_subset_sha256(training_specs, preflight.IDENTITY_CASES)
    started = perf_counter()
    classification = "vertical_motion_derivative_ladder_exception_failed_closed"
    passed = False
    exception = None
    identity = None
    references = None
    analytic = None
    baseline_components = None
    baseline_repeats = None
    resolution = None
    ladder = None
    reproduction = None
    qualification = None
    step_report = None
    try:
        print(json.dumps({"stage": "identity_and_source_references"}), flush=True)
        identity, references = _identity_and_references(
            source,
            controller,
            training_specs,
            sequences,
            anatomy,
            device=device,
        )
        if not identity["pass"]:
            classification = "vertical_motion_derivative_ladder_identity_failed"
        elif references["semantic_sha256"] != EXPECTED_NORMALIZATIONS_SHA256:
            classification = "vertical_motion_derivative_ladder_normalization_failed"
        elif rendered_sha256 != EXPECTED_RENDERED_SUBSET_SHA256:
            classification = "vertical_motion_derivative_ladder_stimulus_failed"
        else:
            print(json.dumps({"stage": "analytic_gradient"}), flush=True)
            controller.zero_grad(set_to_none=True)
            baseline_loss, components = _evaluate(
                controller,
                batch_specs,
                sequences,
                references,
                anatomy,
                device=device,
                checkpoint_frames=True,
            )
            baseline_components = preflight._float_components(components)
            tau_component_gradients = {}
            for name, component in components.items():
                gradient = torch.autograd.grad(
                    component,
                    controller.tau_ratio,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                tau_component_gradients[name] = (
                    (torch.zeros_like(controller.tau_ratio) if gradient is None else gradient)
                    .detach()
                    .cpu()
                )
            baseline_loss.backward()
            gradients = {
                name: preflight._parameter(controller, name).grad.detach().cpu().clone()
                for name in ("gain", "bias_offset", "tau_ratio")
            }
            probe_gradients = {
                probe_name: float(gradients[parameter_name][index])
                for probe_name, parameter_name, index in preflight.PROBES
            }
            analytic = {
                "probe_gradients": probe_gradients,
                "tau_component_gradients": {
                    probe_name: {
                        component: float(values[index])
                        for component, values in tau_component_gradients.items()
                    }
                    for probe_name, parameter_name, index in preflight.PROBES
                    if parameter_name == "tau_ratio"
                },
                "all_parameter_gradients_finite": all(
                    bool(torch.isfinite(values).all()) for values in gradients.values()
                ),
            }

            print(json.dumps({"stage": "baseline_repeats"}), flush=True)
            repeated = []
            for _ in range(BASELINE_REPEATS):
                with torch.no_grad():
                    value, repeated_components = _evaluate(
                        controller,
                        batch_specs,
                        sequences,
                        references,
                        anatomy,
                        device=device,
                        checkpoint_frames=False,
                    )
                repeated.append(
                    {
                        "loss": float(value.detach().cpu()),
                        "components": preflight._float_components(repeated_components),
                    }
                )
            baseline_repeats = repeated
            resolution = loss_resolution_floor([item["loss"] for item in repeated])

            print(json.dumps({"stage": "finite_difference_ladder"}), flush=True)
            identity_parameters = controller.parameter_values()
            ladder = {}
            for probe_name, parameter_name, index in preflight.PROBES:
                rows = []
                for step in STEPS:
                    directions = {}
                    for direction, sign in (("plus", 1.0), ("minus", -1.0)):
                        controller.load_parameter_values(identity_parameters)
                        parameter = preflight._parameter(controller, parameter_name)
                        with torch.no_grad():
                            identity_value = float(parameter[index].detach().cpu())
                            parameter[index].add_(sign * step)
                            materialized_value = float(parameter[index].detach().cpu())
                            value, component_values = _evaluate(
                                controller,
                                batch_specs,
                                sequences,
                                references,
                                anatomy,
                                device=device,
                                checkpoint_frames=False,
                            )
                        directions[direction] = {
                            "requested_offset": sign * step,
                            "materialized_value": materialized_value,
                            "materialized_offset": materialized_value - identity_value,
                            "loss": float(value.detach().cpu()),
                            "components": preflight._float_components(component_values),
                        }
                    actual_span = (
                        directions["plus"]["materialized_value"]
                        - directions["minus"]["materialized_value"]
                    )
                    central = (
                        directions["plus"]["loss"] - directions["minus"]["loss"]
                    ) / actual_span
                    analytic_value = probe_gradients[probe_name]
                    relative_error = abs(central - analytic_value) / max(
                        abs(central), abs(analytic_value), 1.0e-12
                    )
                    loss_difference = abs(directions["plus"]["loss"] - directions["minus"]["loss"])
                    resolved = loss_difference >= resolution["loss_resolution_floor"]
                    sign_matches = bool(
                        central != 0.0
                        and analytic_value != 0.0
                        and math.copysign(1.0, central) == math.copysign(1.0, analytic_value)
                    )
                    row = {
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
                    rows.append(row)
                ladder[probe_name] = {
                    "parameter": parameter_name,
                    "index": index,
                    "rows": rows,
                    "qualification": qualify_probe(rows),
                }
            controller.load_parameter_values(identity_parameters)

            h001 = {
                name: next(row for row in values["rows"] if row["step"] == 0.001)
                for name, values in ladder.items()
            }
            analytic_differences = {
                name: abs(probe_gradients[name] - expected)
                for name, expected in EXPECTED_V1_ANALYTIC.items()
            }
            loss_differences = {
                name: {
                    "plus": abs(row["plus"]["loss"] - EXPECTED_V1_H001_LOSSES[name][0]),
                    "minus": abs(row["minus"]["loss"] - EXPECTED_V1_H001_LOSSES[name][1]),
                }
                for name, row in h001.items()
            }
            reproduction = {
                "pass": bool(
                    max(analytic_differences.values()) <= ANALYTIC_REPRODUCTION_ABSOLUTE_TOLERANCE
                    and max(
                        value
                        for differences in loss_differences.values()
                        for value in differences.values()
                    )
                    <= LOSS_REPRODUCTION_ABSOLUTE_TOLERANCE
                ),
                "analytic_absolute_differences": analytic_differences,
                "h_0.001_loss_absolute_differences": loss_differences,
                "analytic_absolute_tolerance": (ANALYTIC_REPRODUCTION_ABSOLUTE_TOLERANCE),
                "loss_absolute_tolerance": LOSS_REPRODUCTION_ABSOLUTE_TOLERANCE,
            }
            qualification = {
                "pass": bool(
                    reproduction["pass"]
                    and analytic["all_parameter_gradients_finite"]
                    and all(item["qualification"]["pass"] for item in ladder.values())
                ),
                "probes": {name: item["qualification"] for name, item in ladder.items()},
            }
            if not reproduction["pass"]:
                classification = "vertical_motion_derivative_ladder_reproduction_failed"
            elif not qualification["pass"]:
                classification = "vertical_motion_derivative_ladder_qualification_failed"
            else:
                print(json.dumps({"stage": "one_update"}), flush=True)
                step_report = preflight.optimizer_step_report(
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
                    classification = "vertical_motion_derivative_ladder_passed"
                    passed = True
                else:
                    classification = "vertical_motion_derivative_ladder_step_failed"
    except Exception as error:
        exception = {"type": type(error).__name__, "message": str(error)}
    finally:
        controller.load_parameter_values(
            {
                "gain": torch.ones(16),
                "bias_offset": torch.zeros(4),
                "tau_ratio": torch.ones(4),
            }
        )
        source_after = commissioning.semantic_sha256(source.state_dict())
        source_restored = bool(source_after == source_before and controller.identity_restored())
        peak_reserved = torch.cuda.max_memory_reserved(device)

    memory_pass = peak_reserved <= preflight.PEAK_RESERVED_LIMIT_BYTES
    if not source_restored:
        passed = False
        classification = "vertical_motion_derivative_ladder_restoration_failed"
    elif not memory_pass:
        passed = False
        classification = "vertical_motion_derivative_ladder_memory_failed"
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
        "identity": identity,
        "rendered_training_identity_subset_float32_sha256": rendered_sha256,
        "frozen_source_normalizations_semantic_sha256": (
            references["semantic_sha256"] if references is not None else None
        ),
        "baseline_loss_components": baseline_components,
        "analytic": analytic,
        "baseline_repeats": baseline_repeats,
        "loss_resolution": resolution,
        "finite_difference_ladder": ladder,
        "v1_reproduction": reproduction,
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
