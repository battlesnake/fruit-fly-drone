#!/usr/bin/env python3
"""Run bound-aware D-first fitting with authoritative materialized parameters."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_bound_aware as bounded  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-canonical-fitting-v1"
PROTOCOL_COMMIT = "99b95f5"
PARAMETER_IDEMPOTENCE_TOLERANCE = 1.0e-7
_ORIGINAL_PROTOCOL_MANIFEST = bounded.protocol_manifest
FAILED_UNBOUNDED_PREFLIGHT_EXPERIMENT = base.EXPERIMENT
_AUTHORITATIVE_BASE: dict[str, Tensor] | None = None
_AUTHORITATIVE_CANDIDATE: dict[str, Tensor] | None = None


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
        "--source-audit-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/full-native-endpoint-damping-step-audit-001"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/full-native-d-first-canonical-fitting-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(_ORIGINAL_PROTOCOL_MANIFEST())
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "preserves_prior_preflight_failures": [
                FAILED_UNBOUNDED_PREFLIGHT_EXPERIMENT,
                bounded.EXPERIMENT,
            ],
        }
    )
    manifest["constraint_projection"].update(
        {
            "authoritative_full_step": "once-materialized float32 parameter tensors",
            "second_direct_projection_parameter_change_maximum": (
                PARAMETER_IDEMPOTENCE_TOLERANCE
            ),
            "effective_displacement_arithmetic": "float64 subtraction and dot products",
            "full_scale_installation": "direct authoritative parameter copy",
            "backtrack_materialization": (
                "one float64 interpolation from current to authoritative parameters, "
                "one dtype conversion, then one controller projection"
            ),
            "canonicalization_repeated_until_pass": False,
        }
    )
    return manifest


def _maximum_rounding_coordinate(
    base_parameters: dict[str, Tensor],
    proposed_displacement: dict[str, Tensor],
    authoritative_parameters: dict[str, Tensor],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for family in joint.PARAMETER_FAMILIES:
        effective = authoritative_parameters[family] - base_parameters[family]
        difference = (effective - proposed_displacement[family]).abs()
        maximum, index = difference.reshape(-1).max(dim=0)
        flat_index = int(index)
        candidate_value = authoritative_parameters[family].reshape(-1)[flat_index]
        next_value = torch.nextafter(
            candidate_value,
            torch.tensor(math.inf, device=candidate_value.device),
        )
        ulp = float((next_value - candidate_value).abs())
        report = {
            "family": family,
            "flat_index": flat_index,
            "source_parameter": float(base_parameters[family].reshape(-1)[flat_index]),
            "proposed_displacement": float(proposed_displacement[family].reshape(-1)[flat_index]),
            "materialized_parameter": float(candidate_value),
            "effective_float32_displacement": float(effective.reshape(-1)[flat_index]),
            "absolute_displacement_difference": float(maximum),
            "local_float32_ulp": ulp,
            "difference_in_local_ulps": float(maximum) / ulp if ulp > 0.0 else None,
        }
        if best is None or report["absolute_displacement_difference"] > best[
            "absolute_displacement_difference"
        ]:
            best = report
    if best is None:
        raise RuntimeError("no native parameter families were available")
    return best


@torch.no_grad()
def materialize_authoritative_candidate(
    controller: ConnectomeController,
    base_parameters: dict[str, Tensor],
    proposed_displacement: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    for name in joint.PARAMETER_FAMILIES:
        target = base_parameters[name].double() + proposed_displacement[name].double()
        getattr(controller, name).copy_(target.to(base_parameters[name].dtype))
    controller.project_parameters()
    authoritative = joint._copy_parameters(controller)
    before_second_projection = joint._copy_parameters(controller)
    controller.project_parameters()
    second_projection_difference = max(
        float((getattr(controller, name) - before_second_projection[name]).abs().max())
        for name in joint.PARAMETER_FAMILIES
    )
    authoritative = joint._copy_parameters(controller)
    effective_float64 = {
        name: authoritative[name].double() - base_parameters[name].double()
        for name in joint.PARAMETER_FAMILIES
    }
    return authoritative, effective_float64, {
        "pass": second_projection_difference <= PARAMETER_IDEMPOTENCE_TOLERANCE,
        "second_direct_projection_maximum_parameter_change": second_projection_difference,
        "second_direct_projection_maximum_parameter_change_limit": (
            PARAMETER_IDEMPOTENCE_TOLERANCE
        ),
        "maximum_rounding_coordinate": _maximum_rounding_coordinate(
            base_parameters, proposed_displacement, authoritative
        ),
    }


def _dot_float64(first: dict[str, Tensor], second: dict[str, Tensor]) -> float:
    return float(
        sum(
            (first[name].double() * second[name].double()).sum()
            for name in joint.PARAMETER_FAMILIES
        )
    )


def make_projected_proposal(
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    endpoint_scale: float,
    *,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    global _AUTHORITATIVE_BASE, _AUTHORITATIVE_CANDIDATE

    base_parameters = joint._copy_parameters(student)
    specs = base.constraint_specs(source_metrics, current_metrics)
    print(json.dumps({"stage": "canonical_constraint_jacobian", "rows": len(specs)}), flush=True)
    rows = base.constraint_gradient_rows(
        student, factorial_cache, attitude_cache, scales, specs, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    endpoint.accumulated_endpoint_damping_gradient(
        student,
        factorial_cache,
        scale=endpoint_scale,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    raw_gradients = {
        name: getattr(student, name).grad.detach().clone() for name in joint.PARAMETER_FAMILIES
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
    optimizer.step()
    student.project_parameters()
    raw_parameters = joint._copy_parameters(student)
    raw_displacement = {
        name: raw_parameters[name] - base_parameters[name] for name in joint.PARAMETER_FAMILIES
    }
    proposed, projection = bounded.bound_aware_project_inequality_displacement(
        raw_displacement, rows, specs, base_parameters["edge_magnitude"]
    )
    authoritative, effective64, canonicalization = materialize_authoritative_candidate(
        student, base_parameters, proposed
    )
    effective32 = {
        name: value.to(base_parameters[name].dtype) for name, value in effective64.items()
    }
    linearized = [
        spec["current_mse"]
        - spec["limit_mse"]
        + _dot_float64(row, effective64)
        for spec, row in zip(specs, rows, strict=True)
    ]
    maximum_violation = max(linearized)
    raw64 = {name: value.double() for name, value in raw_displacement.items()}
    raw_derivative = _dot_float64(raw_gradients, raw64)
    effective_derivative = _dot_float64(raw_gradients, effective64)
    projection.update(
        {
            "pass_after_parameter_bounds": (
                projection["pass"]
                and canonicalization["pass"]
                and maximum_violation <= base.LINEAR_CONSTRAINT_TOLERANCE
                and effective_derivative < 0.0
            ),
            "authoritative_parameter_canonicalization": canonicalization,
            "bounded_linearized_violation_after": linearized,
            "maximum_bounded_linearized_violation_after": maximum_violation,
            "raw_damping_directional_derivative": raw_derivative,
            "bounded_projected_damping_directional_derivative": effective_derivative,
            "damping_descent_fraction_retained": (
                effective_derivative / raw_derivative if raw_derivative < 0.0 else None
            ),
            "raw_displacement_family_rms": {
                name: float(value.square().mean().sqrt())
                for name, value in raw_displacement.items()
            },
            "bounded_projected_displacement_family_rms": {
                name: float(value.square().mean().sqrt())
                for name, value in effective64.items()
            },
            "constraint_specs": specs,
            "gradient_norm_before_clipping": float(gradient_norm),
            "effective_displacement_computed_in_float64": True,
        }
    )
    _AUTHORITATIVE_BASE = base_parameters
    _AUTHORITATIVE_CANDIDATE = authoritative
    joint._load_parameters(student, base_parameters)
    return effective32, raw_gradients, projection


@torch.no_grad()
def install_trial(
    controller: ConnectomeController,
    source: dict[str, Tensor],
    displacement: dict[str, Tensor],
    *,
    scale: float,
) -> dict[str, float]:
    del displacement
    if _AUTHORITATIVE_BASE is None or _AUTHORITATIVE_CANDIDATE is None:
        raise RuntimeError("authoritative proposal parameters are unavailable")
    if any(
        not torch.equal(source[name], _AUTHORITATIVE_BASE[name])
        for name in joint.PARAMETER_FAMILIES
    ):
        raise RuntimeError("trial source differs from the authoritative proposal base")
    for name in joint.PARAMETER_FAMILIES:
        if scale == 1.0:
            target = _AUTHORITATIVE_CANDIDATE[name]
        else:
            target = source[name].double() + scale * (
                _AUTHORITATIVE_CANDIDATE[name].double() - source[name].double()
            )
        getattr(controller, name).copy_(target.to(source[name].dtype))
    controller.project_parameters()
    return {
        name: float((getattr(controller, name) - source[name]).square().mean().sqrt())
        for name in joint.PARAMETER_FAMILIES
    }


def main() -> int:
    base.EXPERIMENT = EXPERIMENT
    base.PROTOCOL_COMMIT = PROTOCOL_COMMIT
    base.parse_args = parse_args
    base.protocol_manifest = protocol_manifest
    base.make_projected_proposal = make_projected_proposal
    base.install_trial = install_trial
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
