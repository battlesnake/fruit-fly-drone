#!/usr/bin/env python3
"""Confirm frozen T4/T5 motion tuning after deterministic replay qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

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
    parser.add_argument(
        "--replay-gate",
        type=Path,
        default=REPO_ROOT
        / "runs/optic-motion/frozen-t4t5-audit-002/replay-gate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/frozen-t4t5-audit-002",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--block-cases",
        type=int,
        choices=(deterministic.BLOCK_CASES,),
        default=deterministic.BLOCK_CASES,
    )
    return parser.parse_args()


def deterministic_duplicate_control(
    main_response_sha256: str,
    main_metadata_sha256: str,
    runtime_sha256: str,
    process_identity: dict[str, Any],
    replay_gate: dict[str, Any],
) -> dict[str, Any]:
    replay_nonces = {
        item["process_identity"]["nonce"] for item in replay_gate["replay_reports"]
    }
    response_matched = main_response_sha256 == replay_gate["response_tensor_sha256"]
    metadata_matched = (
        main_metadata_sha256 == replay_gate["evaluation_metadata_semantic_sha256"]
    )
    runtime_matched = runtime_sha256 == replay_gate["runtime_semantic_sha256"]
    process_distinct = process_identity["nonce"] not in replay_nonces
    passed = response_matched and metadata_matched and runtime_matched and process_distinct
    return {
        "pass": passed,
        "fresh_processes": deterministic.REPLAY_PROCESSES + 1,
        "response_tensor_sha256": main_response_sha256,
        "evaluation_metadata_semantic_sha256": main_metadata_sha256,
        "runtime_semantic_sha256": runtime_sha256,
        "main_process_identity": process_identity,
        "main_process_identity_distinct": process_distinct,
        "all_response_tensor_hashes_exact": response_matched,
        "all_evaluation_metadata_hashes_exact": metadata_matched,
        "all_runtime_hashes_exact": runtime_matched,
        "numerical_noise_maximum_absolute": 0.0 if passed else None,
    }


def _run_claimed(
    args: argparse.Namespace, process_identity: dict[str, Any]
) -> int:
    report_path = args.output_dir / "report.json"
    deterministic.configure_determinism()
    device = torch.device(args.device)
    runtime = deterministic.runtime_manifest(device)
    started = perf_counter()

    input_hashes = base.validate_inputs(args)
    input_hashes.update(
        deterministic.validate_locked_v1(args.v1_report, args.v1_implementation)
    )
    replay_gate = deterministic.load_and_validate_replay_gate(args.replay_gate)
    input_hashes[base.assisted.responsibility.stable_path(args.replay_gate)] = (
        base.file_sha256(args.replay_gate)
    )
    anatomy = base.anatomy_manifest(args)
    specs = base.stimulus_specs()
    stimuli = base.stimulus_manifest(specs)
    deterministic.validate_stimulus_manifest(stimuli)
    source_state = base.load_checkpoint(args.checkpoint)
    source_sha = base.state_sha256(source_state)
    start = {
        "experiment": deterministic.EXPERIMENT,
        "protocol": deterministic.protocol_manifest(),
        "input_file_sha256": input_hashes,
        "runtime": runtime,
        "process_identity": process_identity,
        "source_state_sha256": source_sha,
        "anatomy": base.report_anatomy(anatomy),
        "stimulus_manifest": stimuli,
        "replay_gate": replay_gate,
    }
    base._write_or_validate_start(args.output_dir / "start.json", start)

    classification = "deterministic_optic_motion_exception_failed_closed"
    passed = False
    exception: dict[str, str] | None = None
    numerical: dict[str, Any] | None = None
    duplicate: dict[str, Any] | None = None
    tuning: dict[str, Any] | None = None
    intervention: dict[str, Any] | None = None
    primary_summary: dict[str, Any] | None = None
    reverse_summary: dict[str, Any] | None = None
    executed_evaluations: list[dict[str, Any]] = []
    try:
        subset = base.numerical_specs(specs)
        print(json.dumps({"stage": "main_process_K32_replay"}), flush=True)
        k32 = base.evaluate_specs(
            args,
            subset,
            anatomy,
            source_state,
            method="exponential_euler",
            steps=base.REFERENCE_SUBSTEPS,
            device=device,
        )
        executed_evaluations.append(k32)
        main_response_sha256 = deterministic.response_tensor_sha256(k32)
        main_metadata_sha256 = base.assisted.audit.semantic_sha256(
            deterministic.stable_evaluation_metadata(k32)
        )
        runtime_sha256 = base.assisted.audit.semantic_sha256(runtime)
        duplicate = deterministic_duplicate_control(
            main_response_sha256,
            main_metadata_sha256,
            runtime_sha256,
            process_identity,
            replay_gate,
        )
        if not duplicate["pass"]:
            classification = "deterministic_optic_motion_replay_gate_failed"
        else:
            print(json.dumps({"stage": "numerical_K64"}), flush=True)
            k64 = base.evaluate_specs(
                args,
                subset,
                anatomy,
                source_state,
                method="exponential_euler",
                steps=base.FINE_SUBSTEPS,
                device=device,
            )
            executed_evaluations.append(k64)
            comparison = base.numerical_comparison(k32, k64, subset)
            numerical = {
                "K32": base.compact_evaluation(k32),
                "K64": base.compact_evaluation(k64),
                "K32_vs_K64": comparison,
                "rk4": {},
            }
            if not comparison["pass"]:
                classification = "deterministic_optic_motion_numerical_gate_failed"
            else:
                for rk_steps in base.RK4_STEPS:
                    print(
                        json.dumps(
                            {"stage": "numerical_rk4", "steps": rk_steps}
                        ),
                        flush=True,
                    )
                    rk = base.evaluate_specs(
                        args,
                        subset,
                        anatomy,
                        source_state,
                        method="rk4",
                        steps=rk_steps,
                        device=device,
                    )
                    executed_evaluations.append(rk)
                    numerical["rk4"][str(rk_steps)] = {
                        "evaluation": base.compact_evaluation(rk),
                        "vs_K64": base.numerical_comparison(rk, k64, subset),
                    }

                print(json.dumps({"stage": "primary_K32"}), flush=True)
                primary = base.evaluate_specs(
                    args,
                    specs,
                    anatomy,
                    source_state,
                    method="exponential_euler",
                    steps=base.REFERENCE_SUBSTEPS,
                    device=device,
                )
                executed_evaluations.append(primary)
                primary_summary = base.compact_evaluation(primary)
                reversed_items = base.reverse_specs(specs)
                print(json.dumps({"stage": "reverse_K32"}), flush=True)
                reversed_evaluation = base.evaluate_specs(
                    args,
                    reversed_items,
                    anatomy,
                    source_state,
                    method="exponential_euler",
                    steps=base.REFERENCE_SUBSTEPS,
                    mode="reverse",
                    device=device,
                )
                executed_evaluations.append(reversed_evaluation)
                reverse_summary = base.compact_evaluation(reversed_evaluation)
                tuning = base.tuning_decision(
                    primary,
                    specs,
                    anatomy,
                    duplicate,
                    reversed_evaluation,
                    reversed_items,
                )

                vertical = [
                    item
                    for item in specs
                    if item["cohort"] == "core" and item["axis"] == "vertical"
                ]
                print(json.dumps({"stage": "pair_mean_intervention"}), flush=True)
                intervention_evaluation = base.evaluate_pair_mean_intervention(
                    args, vertical, anatomy, source_state, device=device
                )
                executed_evaluations.append(intervention_evaluation)
                intervention = base.intervention_report(
                    primary, specs, intervention_evaluation, vertical
                )
                if tuning["pass"]:
                    classification = "frozen_optic_motion_module_qualified"
                    passed = True
                else:
                    classification = "frozen_optic_motion_module_validly_uncommissioned"
    except Exception as error:
        exception = {"type": type(error).__name__, "message": str(error)}

    source_restored = bool(
        exception is None
        and executed_evaluations
        and base.state_sha256(source_state) == source_sha
        and all(
            item.get("source_loaded_exactly") and item.get("source_restored")
            for item in executed_evaluations
        )
    )
    execution_controls_pass = bool(
        source_restored
        and all(item.get("all_states_and_outputs_finite") for item in executed_evaluations)
        and all(
            item.get("opposite_terminal_pixel_difference_maximum", 0.0) == 0.0
            for item in executed_evaluations
        )
    )
    scientific_gate_valid = bool(
        execution_controls_pass
        and duplicate is not None
        and duplicate["pass"]
        and source_sha == replay_gate["source_state_sha256"]
        and numerical is not None
        and numerical["K32_vs_K64"]["pass"]
        and tuning is not None
    )
    routing_authorized = bool(passed and scientific_gate_valid)
    commissioning_preregistration_authorized = bool(
        scientific_gate_valid and tuning is not None and not tuning["pass"]
    )
    report = {
        "experiment": deterministic.EXPERIMENT,
        "protocol_commit": deterministic.PROTOCOL_COMMIT,
        "protocol": deterministic.protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "fresh_blind_validation": False,
        "exception": exception,
        "input_file_sha256": input_hashes,
        "runtime": runtime,
        "process_identity": process_identity,
        "source_state_sha256": source_sha,
        "anatomy": base.report_anatomy(anatomy),
        "stimulus_manifest": stimuli,
        "deterministic_replay_gate": replay_gate,
        "duplicate_control": duplicate,
        "numerical": numerical,
        "primary_evaluation": primary_summary,
        "reverse_evaluation": reverse_summary,
        "tuning": tuning,
        "pair_mean_intervention": intervention,
        "source_restored": source_restored,
        "execution_controls_pass": execution_controls_pass,
        "scientific_gate_valid": scientific_gate_valid,
        "motion_output_routing_preregistration_authorized": routing_authorized,
        "local_motion_commissioning_preregistration_authorized": (
            commissioning_preregistration_authorized
        ),
        "training_execution_authorized": False,
        "candidate_retained": False,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": perf_counter() - started,
    }
    if not scientific_gate_valid:
        report["passed"] = False
        report["motion_output_routing_preregistration_authorized"] = False
        report["local_motion_commissioning_preregistration_authorized"] = False
        if exception is None and classification.endswith("uncommissioned"):
            report["classification"] = "deterministic_optic_motion_control_failed"
    base.assisted._atomic_json_save(report, report_path)
    print(
        json.dumps(
            {
                "report": base.assisted.responsibility.stable_path(report_path),
                "classification": report["classification"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    if exception is not None:
        raise RuntimeError("deterministic optic-motion confirmation recorded an exception")
    return 0


def main() -> int:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise SystemExit("deterministic frozen optic-motion report already exists")
    process_identity = deterministic.process_identity()
    phase = "main-confirmation"
    deterministic.claim_phase(report_path, phase=phase, identity=process_identity)
    try:
        result = _run_claimed(args, process_identity)
        deterministic.finalize_phase(
            report_path,
            phase=phase,
            identity=process_identity,
            status="completed",
        )
        return result
    except BaseException as error:
        _, terminal = deterministic.phase_paths(report_path)
        if not terminal.exists():
            deterministic.finalize_phase(
                report_path,
                phase=phase,
                identity=process_identity,
                status="failed",
                error=error,
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
