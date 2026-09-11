#!/usr/bin/env python3
"""Disposable real-connectome gradient check for the RK4 readout-step audit."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_continuous_cns_solver as solver  # noqa: E402
import audit_variable_height_neural_integration_rate as rate  # noqa: E402
import audit_variable_height_rk4_throttle_readout_step as readout  # noqa: E402
import check_variable_height_neural_integration_rate_audit as check_rate  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402


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
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source_state = readout._clone_state_dict(loaded["controller"])
    cache = check_rate.synthetic_cache()
    mask = readout.build_readout_mask(args.graph)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / rate.POLICY_HZ).to(device)
    controller.load_state_dict(source_state, strict=True)
    prefix = readout.source_prefix_states(controller, cache, device=device)
    with torch.no_grad():
        baseline, source_fixed = readout.fixed_prefix_objective(
            controller, cache, prefix, device=device, backward=False
        )
    source_full = solver.evaluate_condition(
        graph=args.graph,
        checkpoint_state=source_state,
        cache=cache,
        method="rk4",
        solver_steps=1,
        device=device,
    )

    optimizer = readout.make_optimizer(controller)
    for parameter in controller.parameters():
        parameter.grad = None
    gradient_objective, _ = readout.fixed_prefix_objective(
        controller, cache, prefix, device=device, backward=True
    )
    clipped, controls = readout._masked_gradients(controller, mask)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    controller.project_parameters()
    pending = readout._clone_state_dict(controller.state_dict())
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    displacement = readout.parameter_displacement(source_state, pending)
    direction = sum(
        float((clipped[name].detach().cpu().double() * displacement[name].double()).sum())
        for name in readout.PARAMETER_FAMILIES
    )
    candidate_state = readout.materialize_candidate(
        source_state, pending, scale=readout.PROBE_SCALE
    )
    candidate = readout.evaluate_candidate(
        graph=args.graph,
        source_state=source_state,
        candidate_state=candidate_state,
        cache=cache,
        prefix_states=prefix,
        source_fixed=source_fixed,
        source_full=source_full,
        mask=mask,
        device=device,
        scale=readout.PROBE_SCALE,
        families=readout.PARAMETER_FAMILIES,
    )
    controller.load_state_dict(source_state, strict=True)
    passed = bool(
        torch.isfinite(baseline)
        and torch.isfinite(gradient_objective)
        and controls["outside_mask_gradients_exactly_zero"]
        and controls["all_masked_gradients_nonzero"]
        and direction < 0.0
        and readout.outside_mask_equal(source_state, pending, mask)
        and candidate["fixed"]["all_recurrent_states_and_outputs_finite"]
        and candidate["full"]["all_recurrent_states_and_outputs_finite"]
        and candidate["candidate_state_loaded_and_restored"]
        and rate._source_state_sha256(controller)
        == readout.assisted.audit.semantic_sha256(source_state)
        and readout.assisted.optimizer_step_counters(optimizer_before) == []
        and readout.assisted.optimizer_step_counters(optimizer_after) == [1.0, 1.0, 1.0]
    )
    print(
        json.dumps(
            {
                "pass": passed,
                "device": str(device),
                "baseline_objective": float(baseline),
                "gradient_objective": float(gradient_objective),
                "full_proposal_directional_derivative": direction,
                "probe_fixed_nrmse_improvement": candidate["fixed_nrmse_improvement"],
                "probe_full_nrmse_improvement": candidate["full_nrmse_improvement"],
                "gradient_controls": controls,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
