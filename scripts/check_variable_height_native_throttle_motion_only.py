#!/usr/bin/env python3
"""Disposable CUDA integration check for the motion-only learnability trainer."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_native_throttle_motion_only as motion  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

CHECK_TRAINING_SEED = 990_991


def main() -> int:
    motion.require_nonformal_seeds(CHECK_TRAINING_SEED)
    motion.validate_authorizing_report()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the registered disposable integration")
    graph = REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    checkpoint = REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt"
    loaded = torch.load(checkpoint, map_location=device, weights_only=True)
    source = ConnectomeController(graph, neural_dt=1.0 / motion.POLICY_HZ).to(device)
    controller = ConnectomeController(graph, neural_dt=1.0 / motion.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    controller.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    controller.eval()
    config = HoverConfig()
    bank = motion.build_exact_factorial_motion_bank(
        seed=CHECK_TRAINING_SEED,
        held_out_styles=False,
        device=device,
        config=config,
    )
    manifest = motion.motion_bank_manifest(bank)
    scale = motion.frozen_motion_scale(bank, config=config)
    source_report = motion.motion_full_prefix_report(
        source, bank, motion_scale=scale["motion"], device=device
    )
    trajectory = motion.sampled_endpoint_velocity_report(bank)
    teacher_stick = motion.assisted.teacher_motion_stick_positive_control(
        bank, device=device, config=config
    )
    optimizer = motion._make_optimizer(controller)
    result = motion.run_update_attempt(
        controller,
        optimizer,
        bank,
        motion_scale=scale["motion"],
        device=device,
    )
    selected = next(
        (trial for trial in result["trials"] if trial["pass"]),
        None,
    )
    if not (
        manifest["exact_full_factorial"]
        and trajectory["pass"]
        and teacher_stick["pass"]
        and source_report["endpoint_image_difference_max"] == 0.0
        and result["accepted"]
        and not result["fatal_numerical_failure"]
        and result["finite_difference"]["pass"]
        and result["finite_difference"]["ordinary_step_selected_by_probes"] is False
        and selected is not None
        and selected["fixed_burn_in_improvement"] >= motion.MINIMUM_OBJECTIVE_IMPROVEMENT
        and selected["full_prefix_improvement"] >= motion.MINIMUM_OBJECTIVE_IMPROVEMENT
        and result["optimizer_transaction"]["counters_after"] == [1.0, 1.0, 1.0]
        and motion.assisted.optimizer_step_counters(optimizer.state_dict()) == [1.0, 1.0, 1.0]
    ):
        raise SystemExit("motion-only disposable update failed its integration controls")

    state = {
        "run_state": "active",
        "accepted_updates": 1,
        "attempted_updates": 1,
        "history": [result],
        "training_bank": bank,
        "training_bank_manifest": manifest,
        "objective_scale": scale,
        "objective_scale_sha256": motion.assisted.audit.semantic_sha256(scale),
        "preflight": {"disposable": True},
        "source_training_report": source_report,
        "milestone_25": None,
        "milestone_25_started": False,
        "training_final": None,
        "training_final_started": False,
        "development": None,
        "development_started": False,
        "development_completed": False,
        "attempt_stage": "idle",
        "pending_terminal": None,
        "optimizer_step_counters": [1.0, 1.0, 1.0],
    }
    task_tmp = Path("/home/mark/tmp/fly-variable-height-native-throttle-motion-only-check")
    task_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=task_tmp) as temporary:
        path = Path(temporary) / "resume.pt"
        payload_before = motion._resume_payload(controller, optimizer, state)
        motion.assisted._atomic_torch_save(payload_before, path)
        payload_after = torch.load(path, map_location="cpu", weights_only=True)
        excluded = {"controller", "optimizer"}
        before_science = {
            key: value for key, value in payload_before.items() if key not in excluded
        }
        after_science = {
            key: value for key, value in payload_after.items() if key not in excluded
        }
        round_trip = bool(
            motion.assisted.audit.semantic_sha256(before_science)
            == motion.assisted.audit.semantic_sha256(after_science)
            and payload_before["controller_parameter_sha256"]
            == payload_after["controller_parameter_sha256"]
            and payload_before["optimizer_sha256"] == payload_after["optimizer_sha256"]
            and motion.assisted.audit.semantic_sha256(copy.deepcopy(optimizer.state_dict()))
            == payload_after["optimizer_sha256"]
        )
    if not round_trip:
        raise SystemExit("motion-only disposable resume round trip changed scientific state")

    controller.load_state_dict(loaded["controller"])
    rollback_optimizer = motion._make_optimizer(controller)
    controller_before = motion.assisted.audit.semantic_sha256(
        motion.assisted._copy_parameters(controller)
    )
    optimizer_before = motion.assisted.audit.semantic_sha256(rollback_optimizer.state_dict())
    ordinary_scales = motion.BACKTRACK_SCALES
    try:
        motion.BACKTRACK_SCALES = ()
        rejected = motion.run_update_attempt(
            controller,
            rollback_optimizer,
            bank,
            motion_scale=scale["motion"],
            device=device,
        )
    finally:
        motion.BACKTRACK_SCALES = ordinary_scales
    rejected_classification, _ = motion._attempt_terminal(rejected)
    rollback_pass = bool(
        not rejected["accepted"]
        and not rejected["fatal_numerical_failure"]
        and rejected["unaccepted_restoration_pass"]
        and rejected_classification == "first_finite_no_scale_rejection"
        and motion.assisted.audit.semantic_sha256(
            motion.assisted._copy_parameters(controller)
        )
        == controller_before
        and motion.assisted.audit.semantic_sha256(rollback_optimizer.state_dict())
        == optimizer_before
        and motion.assisted.optimizer_step_counters(rollback_optimizer.state_dict()) == []
    )
    if not rollback_pass:
        raise SystemExit("motion-only finite rejection did not restore exact state")

    print(
        json.dumps(
            {
                "pass": True,
                "formal_seed_overlap": sorted(
                    motion.FORMAL_SEEDS.intersection((CHECK_TRAINING_SEED,))
                ),
                "bank_sha256": bank["sha256"],
                "source_motion_nrmse": source_report["nrmse"],
                "accepted_scale": result["accepted_scale"],
                "fixed_objective_improvement": selected["fixed_burn_in_improvement"],
                "full_prefix_objective_improvement": selected["full_prefix_improvement"],
                "derivative_probe_count": len(result["finite_difference"]["probes"]),
                "derivative_relative_error": result["finite_difference"]["relative_error"],
                "resume_round_trip_pass": round_trip,
                "finite_rejection_rollback_pass": rollback_pass,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
