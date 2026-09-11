#!/usr/bin/env python3
"""Replay the exact readout direction on a preregistered upward trust-region ladder."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_neural_integration_rate as rate  # noqa: E402
import audit_variable_height_rk4_throttle_readout_step as readout  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-rk4-readout-trust-region-v1"
PROTOCOL_COMMIT = "06c904f"
EXPECTED_READOUT_REPORT_SHA256 = "508c54ebb00aa8419e66e540bbc7ca45bd0d254858e76eab590cc4f39dc878f7"
EXPECTED_PENDING_PARAMETER_SHA256 = (
    "73d633cecf3fd0b42616142eb16eac931d9e41dac6158d640ab07f124353ec0d"
)
EXPECTED_OPTIMIZER_AFTER_SHA256 = "0692f59a15c27ef9c3cff3a09dc6611246e0d1dcea8b051e9c32a97cd371976e"
REPRODUCTION_ABSOLUTE_TOLERANCE = 2.0e-5
TRUST_REGION_SCALES = (16.0, 8.0, 4.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = readout.parse_args()
    parser.description = __doc__
    parser.add_argument(
        "--readout-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-throttle-readout-step-001/report.json",
    )
    parser.set_defaults(
        output_dir=REPO_ROOT / "runs/variable-height-hover/native-rk4-readout-trust-region-001"
    )
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "locked_readout_report_sha256": EXPECTED_READOUT_REPORT_SHA256,
        "required_pending_parameter_sha256": EXPECTED_PENDING_PARAMETER_SHA256,
        "required_optimizer_after_sha256": EXPECTED_OPTIMIZER_AFTER_SHA256,
        "reproduction_absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "trust_region_scales_descending": list(TRUST_REGION_SCALES),
        "scales_at_or_below_one_replayed": False,
        "selection": "first actual nonlinear replay passing every original readout gate",
        "actor_parameter_mask_solver_and_data_unchanged": True,
        "candidate_retained": False,
        "training_hover_gate_or_promotion_authorized": False,
    }


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    input_hashes, locked = readout.validate_inputs(args)
    if (
        not args.readout_report.is_file()
        or readout.assisted.responsibility.file_sha256(args.readout_report)
        != EXPECTED_READOUT_REPORT_SHA256
    ):
        raise SystemExit("locked readout-step report is missing or changed")
    with args.readout_report.open() as stream:
        prior = json.load(stream)
    if (
        prior.get("classification") != "no_safe_local_readout_step"
        or prior.get("selected_scale") is not None
        or not prior.get("numerical_controls_pass")
        or prior.get("pending_parameter_sha256") != EXPECTED_PENDING_PARAMETER_SHA256
        or prior.get("optimizer", {}).get("state_after_sha256") != EXPECTED_OPTIMIZER_AFTER_SHA256
    ):
        raise SystemExit("readout-step report is not the registered failed result")
    input_hashes[readout.assisted.responsibility.stable_path(args.readout_report)] = (
        EXPECTED_READOUT_REPORT_SHA256
    )
    return input_hashes, locked, prior


def reproduction_decision(
    prior: dict[str, Any],
    *,
    baseline_values: list[float],
    gradient_objective: float,
    directional_derivative: float,
    pending_parameter_sha256: str,
    optimizer_after_sha256: str,
    optimizer_counters_before: list[float],
    optimizer_counters_after: list[float],
) -> dict[str, Any]:
    expected_scalars = {
        "gradient_objective": float(prior["gradient_objective"]),
        "directional_derivative": float(prior["full_proposal_directional_derivative"]),
    }
    actual_scalars = {
        "gradient_objective": gradient_objective,
        "directional_derivative": directional_derivative,
    }
    scalar_differences = {
        name: abs(actual_scalars[name] - expected) for name, expected in expected_scalars.items()
    }
    baseline_differences = [
        abs(actual - expected)
        for actual, expected in zip(
            baseline_values, prior["baseline_fixed_objectives"], strict=True
        )
    ]
    reasons = []
    if max([*scalar_differences.values(), *baseline_differences]) > (
        REPRODUCTION_ABSOLUTE_TOLERANCE
    ):
        reasons.append("one or more scalar controls exceeded reproduction tolerance")
    if pending_parameter_sha256 != EXPECTED_PENDING_PARAMETER_SHA256:
        reasons.append("pending parameter semantic hash differed")
    if optimizer_after_sha256 != EXPECTED_OPTIMIZER_AFTER_SHA256:
        reasons.append("post-step optimizer semantic hash differed")
    if optimizer_counters_before != [] or optimizer_counters_after != [1.0, 1.0, 1.0]:
        reasons.append("optimizer counters did not reproduce empty-to-one transition")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "baseline_absolute_differences": baseline_differences,
        "scalar_absolute_differences": scalar_differences,
        "pending_parameter_sha256": pending_parameter_sha256,
        "optimizer_after_sha256": optimizer_after_sha256,
        "optimizer_counters_before": optimizer_counters_before,
        "optimizer_counters_after": optimizer_counters_after,
    }


def masked_displacement_report(
    source: dict[str, Tensor], candidate: dict[str, Tensor], mask: dict[str, Any]
) -> dict[str, Any]:
    edge_indices = torch.from_numpy(mask["edge_indices"])
    node_indices = torch.from_numpy(mask["node_indices"])
    selections = {
        "edge_magnitude": edge_indices,
        "bias": node_indices,
        "raw_time_constant": node_indices,
    }
    rms = {}
    for name, indices in selections.items():
        displacement = candidate[name][indices] - source[name][indices]
        rms[name] = float(displacement.square().mean().sqrt())
    edge_values = candidate["edge_magnitude"][edge_indices]
    return {
        "masked_family_rms": rms,
        "edge_values_at_lower_bound": int((edge_values == 0.0).sum()),
        "edge_values_at_upper_bound": int((edge_values == 8.0).sum()),
    }


def _write_start_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise SystemExit("trust-region start marker already exists; replay is forbidden") from exc


def main() -> int:
    args = parse_args()
    if (args.output_dir / "report.json").exists():
        raise SystemExit("the readout trust-region extension already has a terminal report")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    input_hashes, locked, prior = validate_inputs(args)
    cache = locked["cache"]
    source_full = locked["solver_report"]["conditions"]["rk4_m1"]
    mask = readout.build_readout_mask(args.graph)
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source_state = readout._clone_state_dict(loaded["controller"])
    source_sha256 = readout.assisted.audit.semantic_sha256(source_state)
    _write_start_marker(
        args.output_dir / "start.json",
        {
            "experiment": EXPERIMENT,
            "protocol": protocol_manifest(),
            "input_file_sha256": input_hashes,
            "source_state_sha256": source_sha256,
            "readout_mask": mask["manifest"],
            "device": str(device),
        },
    )
    started = perf_counter()
    controller = ConnectomeController(args.graph, neural_dt=1.0 / rate.POLICY_HZ).to(device)
    controller.load_state_dict(source_state, strict=True)
    prefix_states = readout.source_prefix_states(controller, cache, device=device)
    baseline_values = []
    source_fixed = None
    print(json.dumps({"stage": "reconstructing_direction"}), flush=True)
    with torch.no_grad():
        for _ in range(3):
            objective, summary = readout.fixed_prefix_objective(
                controller, cache, prefix_states, device=device, backward=False
            )
            baseline_values.append(float(objective))
            source_fixed = summary
    assert source_fixed is not None
    optimizer = readout.make_optimizer(controller)
    for parameter in controller.parameters():
        parameter.grad = None
    gradient_objective_tensor, _ = readout.fixed_prefix_objective(
        controller, cache, prefix_states, device=device, backward=True
    )
    clipped_gradient, gradient_controls = readout._masked_gradients(controller, mask)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    controller.project_parameters()
    pending_state = readout._clone_state_dict(controller.state_dict())
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    displacement = readout.parameter_displacement(source_state, pending_state)
    directional_derivative = sum(
        float((clipped_gradient[name].detach().cpu().double() * displacement[name].double()).sum())
        for name in readout.PARAMETER_FAMILIES
    )
    pending_sha256 = readout.assisted.audit.semantic_sha256(pending_state)
    optimizer_after_sha256 = readout.assisted.audit.semantic_sha256(optimizer_after)
    reproduction = reproduction_decision(
        prior,
        baseline_values=baseline_values,
        gradient_objective=float(gradient_objective_tensor),
        directional_derivative=directional_derivative,
        pending_parameter_sha256=pending_sha256,
        optimizer_after_sha256=optimizer_after_sha256,
        optimizer_counters_before=readout.assisted.optimizer_step_counters(optimizer_before),
        optimizer_counters_after=readout.assisted.optimizer_step_counters(optimizer_after),
    )
    controller.load_state_dict(source_state, strict=True)

    candidates = []
    selected = None
    if reproduction["pass"] and gradient_controls["outside_mask_gradients_exactly_zero"]:
        for scale in TRUST_REGION_SCALES:
            print(
                json.dumps({"stage": "trust_region_candidate", "scale": scale}),
                flush=True,
            )
            candidate_state = readout.materialize_candidate(
                source_state, pending_state, scale=scale
            )
            record = readout.evaluate_candidate(
                graph=args.graph,
                source_state=source_state,
                candidate_state=candidate_state,
                cache=cache,
                prefix_states=prefix_states,
                source_fixed=source_fixed,
                source_full=source_full,
                mask=mask,
                device=device,
                scale=scale,
                families=readout.PARAMETER_FAMILIES,
            )
            record["masked_displacement"] = masked_displacement_report(
                source_state, candidate_state, mask
            )
            candidates.append(record)
            if record["decision"]["pass"]:
                selected = record
                break

    controller.load_state_dict(source_state, strict=True)
    optimizer.load_state_dict(optimizer_before)
    terminal_source_restored = rate._source_state_sha256(controller) == source_sha256
    terminal_optimizer_restored = readout.assisted.audit.semantic_sha256(
        optimizer.state_dict()
    ) == readout.assisted.audit.semantic_sha256(optimizer_before)
    passed = bool(
        reproduction["pass"]
        and gradient_controls["outside_mask_gradients_exactly_zero"]
        and selected is not None
        and terminal_source_restored
        and terminal_optimizer_restored
    )
    classification = (
        "direct_readout_trust_region_exists"
        if passed
        else (
            "direction_reproduction_failed"
            if not reproduction["pass"]
            else "no_direct_readout_trust_region"
        )
    )
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha256,
        "readout_mask": mask["manifest"],
        "reproduction": reproduction,
        "gradient_controls": gradient_controls,
        "trust_region_candidates": candidates,
        "selected_scale": None if selected is None else selected["scale"],
        "selected_candidate_retained": False,
        "terminal_source_restored": terminal_source_restored,
        "terminal_optimizer_restored": terminal_optimizer_restored,
        "bounded_last_hop_fitting_preregistration_authorized": passed,
        "training_execution_authorized": False,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": perf_counter() - started,
    }
    readout.assisted._atomic_json_save(report, args.output_dir / "report.json")
    print(
        json.dumps(
            {
                "output": readout.assisted.responsibility.stable_path(
                    args.output_dir / "report.json"
                ),
                "classification": classification,
                "pass": passed,
                "reproduction_pass": reproduction["pass"],
                "selected_scale": report["selected_scale"],
                "bounded_last_hop_fitting_preregistration_authorized": passed,
                "training_execution_authorized": False,
                "hover_or_gate_flight_authorized": False,
                "promotion_authorized": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
