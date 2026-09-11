#!/usr/bin/env python3
"""Disposable training-support integration check for the assisted-throttle runner."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_native_throttle_assisted as train  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

CHECK_CASE_SEED = 990_983
CHECK_MOTION_SEED = 990_984
CHECK_SAMPLING_SEED = 990_985


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
    train.require_nonformal_seeds(CHECK_CASE_SEED, CHECK_MOTION_SEED, CHECK_SAMPLING_SEED)
    device = torch.device(args.device)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / train.POLICY_HZ).to(device)
    controller.load_state_dict(loaded["controller"])
    controller.eval()
    config = HoverConfig()

    cases = train.build_hover_case_bank(
        seed=CHECK_CASE_SEED,
        counts={"train": train.DENSE_BANK_CASES},
        device=device,
        config=config,
    )
    hover, _ = train.evaluate_hover_cases(
        None,
        cases,
        teacher_all_axes=True,
        frozen_vision=False,
        device=device,
        config=config,
    )
    teacher_bank = train.collect_trajectory_bank(
        controller,
        cases,
        history_type="teacher",
        device=device,
        config=config,
    )
    motion_bank = train.build_motion_bank(
        seed=CHECK_MOTION_SEED,
        cases=train.MOTION_BANK_CASES,
        height_amplitudes=train.TRAIN_MOTION_HEIGHT_AMPLITUDES,
        speed_amplitudes=train.TRAIN_MOTION_SPEED_AMPLITUDES,
        held_out_styles=False,
        device=device,
        config=config,
    )
    scales = train.frozen_objective_scales(teacher_bank, motion_bank, config=config)
    optimizer = train._make_optimizer(controller)
    generator = torch.Generator(device="cpu").manual_seed(CHECK_SAMPLING_SEED)
    spec = train.sample_update_spec(
        generator,
        block_index=0,
        motion_cases=motion_bank["cases"],
    )
    attempt = train.run_update_attempt(
        controller,
        optimizer,
        teacher_bank,
        None,
        motion_bank,
        spec,
        scales,
        device=device,
    )
    block = {
        "index": 0,
        "teacher": teacher_bank,
        "student": None,
        "motion": motion_bank,
        "reports": {
            "teacher": train.trajectory_bank_report(teacher_bank),
            "student": None,
            "motion": train.motion_bank_report(motion_bank),
        },
    }
    block["identity"] = train.training_block_identity(block)
    state = {
        "run_state": "active",
        "accepted_updates": 0,
        "attempted_updates": 0,
        "consecutive_rejections": 0,
        "history": [],
        "block": block,
        "training_block_reports": [block["reports"]],
        "objective_scales": scales,
        "objective_scales_sha256": train.audit.semantic_sha256(scales),
        "preflight": {"disposable": True},
        "midpoint": None,
        "final": None,
        "development_started": False,
        "development_completed": False,
        "final_started": False,
        "final_completed": False,
        "attempt_stage": "sampled",
        "pending_sample": spec,
        "pending_terminal": None,
    }
    train.record_attempt_outcome(state, attempt, attempted_number=1, target_update=1)
    expected_generator_state = generator.get_state().clone()
    expected_controller_hash = train.audit.semantic_sha256(train._copy_parameters(controller))
    task_tmp = Path("/home/mark/tmp/fly-variable-height-native-throttle-assisted-check")
    task_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=task_tmp) as temporary:
        resume_path = Path(temporary) / "resume.pt"
        resume_sha256 = train._save_resume(resume_path, controller, optimizer, generator, state)
        resumed_controller = ConnectomeController(args.graph, neural_dt=1.0 / train.POLICY_HZ).to(
            device
        )
        resumed_controller.load_state_dict(loaded["controller"])
        resumed_optimizer = train._make_optimizer(resumed_controller)
        resumed_generator = torch.Generator(device="cpu").manual_seed(0)
        resumed_state = train._load_resume(
            resume_path,
            resumed_controller,
            resumed_optimizer,
            resumed_generator,
            teacher_history_seeds=(CHECK_CASE_SEED,) * 4,
            student_history_seeds=(None,) * 4,
            motion_history_seeds=(CHECK_MOTION_SEED,) * 4,
        )
        resume_round_trip = {
            "pass": bool(
                resumed_state["accepted_updates"] == state["accepted_updates"]
                and train.audit.semantic_sha256(train._copy_parameters(resumed_controller))
                == expected_controller_hash
                and train.optimizer_step_counters(resumed_optimizer.state_dict())
                == ([] if not attempt["accepted"] else [1.0, 1.0, 1.0])
                and torch.equal(resumed_generator.get_state(), expected_generator_state)
                and resumed_state["block"]["identity"] == block["identity"]
            ),
            "resume_sha256": resume_sha256,
            "loaded_from_cpu_for_identity_validation": True,
        }
    report = {
        "status": "disposable_training_support_integration_check",
        "uses_registered_training_or_evaluation_seed": False,
        "formal_seed_overlap": sorted(
            train.FORMAL_SEEDS.intersection(
                {CHECK_CASE_SEED, CHECK_MOTION_SEED, CHECK_SAMPLING_SEED}
            )
        ),
        "seeds": [CHECK_CASE_SEED, CHECK_MOTION_SEED, CHECK_SAMPLING_SEED],
        "case_support": train.case_support_decision(cases),
        "teacher_hover": hover,
        "trajectory_bank": train.trajectory_bank_report(teacher_bank),
        "motion_bank": train.motion_bank_report(motion_bank),
        "objective_scales": scales,
        "attempt": attempt,
        "resume_round_trip": resume_round_trip,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if not attempt["fatal_numerical_failure"] and resume_round_trip["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
