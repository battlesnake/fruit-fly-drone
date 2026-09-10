#!/usr/bin/env python3
"""Test fixed-bank constrained contrast descent inside the v3 trust region."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_bridge as bridge  # noqa: E402
import train_variable_height_hover as hover  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402

from flydrone.hover import PLANT_MODEL_VERSION, ConnectomeController, HoverConfig  # noqa: E402
from flydrone.variable_hover import protocol_manifest  # noqa: E402
from flydrone.visual_hover import DEFAULT_VISUAL_CAMERA  # noqa: E402

MINIMUM_CONTRAST_IMPROVEMENT = 1.0e-5
RADIAL_CONSTRAINT_ACTIVATION = 0.90


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
    parser.add_argument(
        "--boundary-checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/trust-region-001/nonpromotional-endpoint.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/constrained-descent-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--constraint-batch-size", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=25)
    parser.add_argument("--unroll", type=int, default=25)
    parser.add_argument("--constraint-prefix-steps", type=int, default=50)
    parser.add_argument("--interim-evaluation-episodes", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--prefix-seconds", type=float, default=2.0)
    parser.add_argument("--response-seconds", type=float, default=6.0)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def project_to_source_ball(
    base_parameters: dict[str, torch.Tensor],
    source_parameters: dict[str, torch.Tensor],
    displacement: dict[str, torch.Tensor],
    radius: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float | bool]]:
    """Project the absolute candidate into the family-metric source ball."""

    source_delta = trust.parameter_delta(base_parameters, source_parameters)
    candidate_delta = {
        name: source_delta[name] + displacement[name]
        for name in trust.PARAMETER_FAMILIES
    }
    norm_before = trust.family_metric_norm(candidate_delta)
    ball_scale = min(1.0, radius / max(norm_before, 1.0e-30))
    constrained = {
        name: ball_scale * candidate_delta[name] - source_delta[name]
        for name in trust.PARAMETER_FAMILIES
    }
    constrained, step_scale = trust.cap_displacement(
        constrained, trust.STEP_FAMILY_RMS_CAP
    )
    final_delta = {
        name: source_delta[name] + constrained[name]
        for name in trust.PARAMETER_FAMILIES
    }
    return constrained, {
        "source_metric_before_ball_projection": norm_before,
        "source_ball_projection_active": ball_scale < 1.0,
        "source_ball_projection_scale": ball_scale,
        "step_cap_scale_after_ball_projection": step_scale,
        "source_metric_after_constraints": trust.family_metric_norm(final_delta),
    }


def _riesz_descent(
    gradient: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    # M_f=I/n_f for sum-of-family-mean-squares geometry, hence M^-1 g=n_f g.
    return {
        name: -gradient[name] * gradient[name].numel()
        for name in trust.PARAMETER_FAMILIES
    }


def _radial_row(
    source_delta: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    # Gradient of 0.5 * sum_f mean(source_delta_f^2).
    return {
        name: source_delta[name] / source_delta[name].numel()
        for name in trust.PARAMETER_FAMILIES
    }


def constrained_direction(
    student: ConnectomeController,
    source: ConnectomeController,
    source_parameters: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    base = trust.clone_parameters(student)
    rows, residual, contrast_gradient, jacobian = trust.common_output_jacobian(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    raw = _riesz_descent(contrast_gradient)
    source_delta = trust.parameter_delta(base, source_parameters)
    source_norm = trust.family_metric_norm(source_delta)
    radial_active = source_norm >= RADIAL_CONSTRAINT_ACTIVATION * trust.SOURCE_METRIC_RADIUS
    if radial_active:
        rows.append(_radial_row(source_delta))
        residual = torch.cat((residual, residual.new_zeros(1)))
        jacobian["labels"].append("source_radius_tangent")

    raw, raw_cap_scale = trust.cap_displacement(raw, trust.STEP_FAMILY_RMS_CAP)
    projected, projection = trust.project_displacement(raw, rows, residual)
    constrained, geometry = project_to_source_ball(
        base,
        source_parameters,
        projected,
        trust.SOURCE_METRIC_RADIUS,
    )
    raw_derivative = float(
        sum(
            (contrast_gradient[name] * raw[name]).sum()
            for name in trust.PARAMETER_FAMILIES
        )
    )
    projected_derivative = float(
        sum(
            (contrast_gradient[name] * constrained[name]).sum()
            for name in trust.PARAMETER_FAMILIES
        )
    )
    report = {
        "objective": "fixed style/amplitude-balanced contrast loss only",
        "common_jacobian": jacobian,
        "projection": projection,
        "radial_constraint_active": radial_active,
        "source_metric_at_base": source_norm,
        "raw_single_scalar_step_cap_scale": raw_cap_scale,
        "raw_family_metric_norm": trust.family_metric_norm(raw),
        "constrained_family_metric_norm": trust.family_metric_norm(constrained),
        "constrained_family_rms": trust.family_rms(constrained),
        "raw_first_order_contrast_derivative": raw_derivative,
        "constrained_first_order_contrast_derivative": projected_derivative,
        "retained_descent_fraction": (
            projected_derivative / raw_derivative
            if raw_derivative < 0.0 and projected_derivative < 0.0
            else None
        ),
        "geometry": geometry,
    }
    del rows
    return constrained, report


def load_controller(
    checkpoint_path: Path,
    *,
    graph: Path,
    graph_sha256: str,
    policy_hz: int,
    device: torch.device,
) -> ConnectomeController:
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if loaded["graph_sha256"] != graph_sha256:
        raise ValueError(f"checkpoint graph hash mismatch: {checkpoint_path}")
    controller = ConnectomeController(graph, neural_dt=1.0 / policy_hz).to(device)
    controller.load_state_dict(loaded["controller"])
    return controller


def direction_preflight(
    controller: ConnectomeController,
    source: ConnectomeController,
    source_parameters: dict[str, torch.Tensor],
    *,
    label: str,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    previous = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    previous.load_state_dict(controller.state_dict())
    previous.eval()
    previous.requires_grad_(False)
    base = trust.clone_parameters(controller)
    baseline_fixed = trust.evaluate_fixed_pairs(
        controller,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    direction, direction_report = constrained_direction(
        controller,
        source,
        source_parameters,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    accepted_scale, trials = trust.find_safe_trial(
        controller,
        source,
        previous,
        base_parameters=base,
        source_parameters=source_parameters,
        displacement=direction,
        baseline_fixed=baseline_fixed,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
        minimum_contrast_improvement=MINIMUM_CONTRAST_IMPROVEMENT,
    )
    trust.load_parameters(controller, base)
    accepted_improvement = None
    if accepted_scale is not None:
        accepted_improvement = (
            baseline_fixed["mean_contrast_nrmse"]
            - trials[-1]["fixed_pair"]["mean_contrast_nrmse"]
        )
    return {
        "label": label,
        "pass": accepted_scale is not None,
        "minimum_contrast_improvement": MINIMUM_CONTRAST_IMPROVEMENT,
        "accepted_scale": accepted_scale,
        "accepted_contrast_improvement": accepted_improvement,
        "baseline_fixed_pair": baseline_fixed,
        "direction": direction_report,
        "trials": trials,
    }


def checkpoint_payload(
    controller: ConnectomeController,
    args: argparse.Namespace,
    config: HoverConfig,
    *,
    source_sha256: str,
    attempts: int,
    accepted_updates: int,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 1,
        "experiment": "variable-height-constraint-aware-descent-v1",
        "non_promotional": True,
        "controller": controller.state_dict(),
        "graph_sha256": hover.file_sha256(args.graph),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "attempts": attempts,
        "accepted_updates": accepted_updates,
        "policy_hz": args.policy_hz,
        "plant_model_version": PLANT_MODEL_VERSION,
        "hover_config": asdict(config),
        "camera": asdict(DEFAULT_VISUAL_CAMERA),
        "actor_inputs": {
            "rgb_camera": True,
            "estimated_roll_pitch": True,
            "accelerometer": False,
            "mass": False,
            "hover_thrust": False,
            "external_history": False,
        },
        "protocol": protocol_manifest(),
        "scene_protocol": bridge.bridge_scene_manifest(args),
    }


def main() -> int:
    args = parse_args()
    for path in (args.graph, args.checkpoint, args.boundary_checkpoint):
        if not path.is_file():
            raise SystemExit(f"required input is missing: {path}")
    if not args.smoke_test and args.attempts != 25:
        raise SystemExit("v1 is preregistered for exactly 25 attempted updates")
    if args.batch_size % 4 or args.constraint_batch_size % 4:
        raise SystemExit("balanced batches must be divisible by four")
    if args.unroll < 11 or args.constraint_prefix_steps < 0:
        raise SystemExit("unroll must be at least 11 and prefix steps nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy-hz must evenly divide the physics rate")
    physics_steps = physics_hz // args.policy_hz
    args.scene_coverage = "all"
    hover.seed_everything(args.seed)
    graph_sha256 = hover.file_sha256(args.graph)
    student = load_controller(
        args.checkpoint,
        graph=args.graph,
        graph_sha256=graph_sha256,
        policy_hz=args.policy_hz,
        device=device,
    )
    source = load_controller(
        args.checkpoint,
        graph=args.graph,
        graph_sha256=graph_sha256,
        policy_hz=args.policy_hz,
        device=device,
    )
    boundary = load_controller(
        args.boundary_checkpoint,
        graph=args.graph,
        graph_sha256=graph_sha256,
        policy_hz=args.policy_hz,
        device=device,
    )
    source.eval()
    source.requires_grad_(False)
    source_parameters = trust.clone_parameters(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    preflight = {
        "source": direction_preflight(
            student,
            source,
            source_parameters,
            label="preserved source",
            args=args,
            physics_steps=physics_steps,
            device=device,
            config=config,
        ),
        "v3_boundary": direction_preflight(
            boundary,
            source,
            source_parameters,
            label="retained v3 active-boundary endpoint",
            args=args,
            physics_steps=physics_steps,
            device=device,
            config=config,
        ),
    }
    preflight["pass"] = bool(
        preflight["source"]["pass"] and preflight["v3_boundary"]["pass"]
    )
    print(
        json.dumps(
            {
                "progress": "preflight",
                "pass": preflight["pass"],
                "source_scale": preflight["source"]["accepted_scale"],
                "source_improvement": preflight["source"][
                    "accepted_contrast_improvement"
                ],
                "boundary_scale": preflight["v3_boundary"]["accepted_scale"],
                "boundary_improvement": preflight["v3_boundary"][
                    "accepted_contrast_improvement"
                ],
            }
        ),
        flush=True,
    )

    baseline = None
    candidate = None
    decision = None
    history = []
    attempted = 0
    accepted = 0
    consecutive_rejections = 0
    stop_reason = None
    if preflight["pass"] and not args.smoke_test:
        baseline = bridge.evaluate_bridge(
            student,
            episodes=args.interim_evaluation_episodes,
            args=args,
            device=device,
            config=config,
            physics_steps=physics_steps,
            seed=args.seed + 40_000,
        )
        previous = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
        previous.eval()
        previous.requires_grad_(False)
        for attempt in range(1, args.attempts + 1):
            attempted = attempt
            base = trust.clone_parameters(student)
            previous.load_state_dict(student.state_dict())
            baseline_fixed = trust.evaluate_fixed_pairs(
                student,
                source,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
            direction, direction_report = constrained_direction(
                student,
                source,
                source_parameters,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
            accepted_scale, trials = trust.find_safe_trial(
                student,
                source,
                previous,
                base_parameters=base,
                source_parameters=source_parameters,
                displacement=direction,
                baseline_fixed=baseline_fixed,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
                minimum_contrast_improvement=MINIMUM_CONTRAST_IMPROVEMENT,
            )
            if accepted_scale is None:
                trust.load_parameters(student, base)
                consecutive_rejections += 1
            else:
                accepted += 1
                consecutive_rejections = 0
            history.append(
                {
                    "attempt": attempt,
                    "accepted": accepted_scale is not None,
                    "accepted_scale": accepted_scale,
                    "consecutive_rejections": consecutive_rejections,
                    "baseline_fixed_pair": baseline_fixed,
                    "direction": direction_report,
                    "trials": trials,
                }
            )
            print(
                json.dumps(
                    {
                        "progress": "attempt",
                        "attempt": attempt,
                        "accepted": accepted_scale is not None,
                        "accepted_scale": accepted_scale,
                        "consecutive_rejections": consecutive_rejections,
                        "contrast_before": baseline_fixed["mean_contrast_nrmse"],
                        "contrast_after": trials[-1]["fixed_pair"][
                            "mean_contrast_nrmse"
                        ],
                    }
                ),
                flush=True,
            )
            if consecutive_rejections >= 5:
                stop_reason = "five consecutive infeasible or sub-noise directions"
                break
    elif not preflight["pass"]:
        stop_reason = "source/boundary feasible-direction preflight failed"
    else:
        stop_reason = "smoke test ended after preflight"

    endpoint_path = None
    if baseline is not None:
        endpoint_path = args.output_dir / "nonpromotional-endpoint.pt"
        torch.save(
            checkpoint_payload(
                student,
                args,
                config,
                source_sha256=hover.file_sha256(args.checkpoint),
                attempts=attempted,
                accepted_updates=accepted,
            ),
            endpoint_path,
        )
    reached_limit = attempted == 25
    if baseline is not None and reached_limit:
        candidate = bridge.evaluate_bridge(
            student,
            episodes=args.interim_evaluation_episodes,
            args=args,
            device=device,
            config=config,
            physics_steps=physics_steps,
            seed=args.seed + 40_000,
        )
        decision = trust.final_decision(candidate, baseline, reached_limit=True)
        if not decision["pass"] and stop_reason is None:
            stop_reason = "; ".join(decision["reasons"])

    report = {
        "experiment": "variable-height-constraint-aware-descent-v1",
        "passed": bool(decision and decision["pass"]),
        "promoted": False,
        "claim": "local constrained-descent feasibility only; no hover promotion",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": hover.file_sha256(args.checkpoint),
        "boundary_checkpoint": str(args.boundary_checkpoint),
        "boundary_checkpoint_sha256": hover.file_sha256(args.boundary_checkpoint),
        "graph": str(args.graph),
        "graph_sha256": graph_sha256,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": args.seed,
        "policy_hz": args.policy_hz,
        "physics_hz": physics_hz,
        "attempts_requested": args.attempts,
        "attempts_completed": attempted,
        "accepted_updates": accepted,
        "stop_reason": stop_reason,
        "minimum_contrast_improvement_per_step": MINIMUM_CONTRAST_IMPROVEMENT,
        "radial_constraint_activation_fraction": RADIAL_CONSTRAINT_ACTIVATION,
        "thresholds": {
            "step_family_rms_cap": trust.STEP_FAMILY_RMS_CAP,
            "source_family_metric_radius": trust.SOURCE_METRIC_RADIUS,
            "per_update_common_nrmse_limit": trust.PER_UPDATE_COMMON_NRMSE_LIMIT,
            "source_common_nrmse_limit": trust.SOURCE_COMMON_NRMSE_LIMIT,
            "source_common_max_absolute_motor_limit": (
                trust.SOURCE_COMMON_MAX_ABSOLUTE_LIMIT
            ),
            "source_function_nrmse_limit": trust.SOURCE_FUNCTION_NRMSE_LIMIT,
            "backtrack_scales": list(trust.BACKTRACK_SCALES),
        },
        "preflight": preflight,
        "baseline": baseline,
        "history": history,
        "candidate": candidate,
        "decision": decision,
        "nonpromotional_endpoint": str(endpoint_path) if endpoint_path else None,
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.smoke_test:
        return 0 if preflight["pass"] else 1
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
