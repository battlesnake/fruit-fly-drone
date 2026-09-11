#!/usr/bin/env python3
"""Evaluate one fresh-process deterministic K32 optic-motion replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_frozen_optic_motion as base  # noqa: E402
import frozen_optic_motion_deterministic as deterministic  # noqa: E402


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
        "--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0"
    )
    parser.add_argument(
        "--tau-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-premotor-tau-preflight-002/report.json",
    )
    parser.add_argument(
        "--v1-report",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/frozen-t4t5-audit-001/report.json",
    )
    parser.add_argument(
        "--v1-implementation",
        type=Path,
        default=REPO_ROOT / "scripts/audit_frozen_optic_motion.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--block-cases",
        type=int,
        choices=(deterministic.BLOCK_CASES,),
        default=deterministic.BLOCK_CASES,
    )
    parser.add_argument(
        "--replay-index",
        type=int,
        choices=tuple(range(1, deterministic.REPLAY_PROCESSES + 1)),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"deterministic replay output already exists: {args.output}")
    process_identity = deterministic.process_identity()
    phase = f"replay-{args.replay_index}"
    deterministic.claim_phase(args.output, phase=phase, identity=process_identity)
    try:
        deterministic.configure_determinism()
        device = torch.device(args.device)
        runtime = deterministic.runtime_manifest(device)
        started = perf_counter()

        input_hashes = base.validate_inputs(args)
        input_hashes.update(
            deterministic.validate_locked_v1(args.v1_report, args.v1_implementation)
        )
        anatomy = base.anatomy_manifest(args)
        specs = base.stimulus_specs()
        stimuli = base.stimulus_manifest(specs)
        deterministic.validate_stimulus_manifest(stimuli)
        subset = base.numerical_specs(specs)
        if len(subset) != base.NUMERICAL_CASES:
            raise SystemExit("deterministic replay subset changed")
        source_state = base.load_checkpoint(args.checkpoint)
        source_sha = base.state_sha256(source_state)
        evaluation = base.evaluate_specs(
            args,
            subset,
            anatomy,
            source_state,
            method="exponential_euler",
            steps=base.REFERENCE_SUBSTEPS,
            device=device,
        )
        passed = bool(
            evaluation["all_states_and_outputs_finite"]
            and evaluation["source_loaded_exactly"]
            and evaluation["source_restored"]
            and evaluation["opposite_terminal_pixel_difference_maximum"] == 0.0
            and base.state_sha256(source_state) == source_sha
        )
        report = {
            "experiment": deterministic.EXPERIMENT,
            "protocol_commit": deterministic.PROTOCOL_COMMIT,
            "protocol": deterministic.protocol_manifest(),
            "passed": passed,
            "input_file_sha256": input_hashes,
            "runtime": runtime,
            "process_identity": process_identity,
            "source_state_sha256": source_sha,
            "source_restored": passed,
            "selected_nodes": anatomy["selected_nodes"],
            "stimulus_rendered_float32_sha256": stimuli[
                "rendered_float32_sha256"
            ],
            "response_tensor_sha256": deterministic.response_tensor_sha256(
                evaluation
            ),
            "evaluation_metadata": deterministic.stable_evaluation_metadata(
                evaluation
            ),
            "wall_time_seconds": perf_counter() - started,
            "candidate_retained": False,
            "training_execution_authorized": False,
            "hover_gate_or_promotion_authorized": False,
        }
        base.assisted._atomic_json_save(report, args.output)
        if not passed:
            raise RuntimeError("deterministic replay evaluation controls failed")
        deterministic.finalize_phase(
            args.output,
            phase=phase,
            identity=process_identity,
            status="completed",
        )
        print(
            json.dumps(
                {
                    "report": base.assisted.responsibility.stable_path(args.output),
                    "passed": passed,
                    "response_tensor_sha256": report["response_tensor_sha256"],
                }
            )
        )
        return 0
    except BaseException as error:
        _, terminal = deterministic.phase_paths(args.output)
        if not terminal.exists():
            deterministic.finalize_phase(
                args.output,
                phase=phase,
                identity=process_identity,
                status="failed",
                error=error,
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
