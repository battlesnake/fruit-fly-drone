#!/usr/bin/env python3
"""Produce or verify an exact accepted update-21 transaction snapshot."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_fp64_corrected_step as step_audit  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_frozen_input as frozen  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402
import train_variable_height_full_native_d_first_fp64_corrected as continuation  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-update21-snapshot-audit-v1"
PROTOCOL_COMMIT = "44f39c5"
EXPECTED_GRAPH_SHA256 = "8c6ba28d149e9ac4a5223c5919657a1734f2c2cac9828c51114fd0174a383665"
EXPECTED_CHECKPOINT_SHA256 = "7238b0e3ca39dc1a8bfd35dcf9f8b6fc9e8c8881ff989f64cb135fa6fed4e572"
EXPECTED_SOURCE_REPORT_SHA256 = "9b236fc6a6744b1b06984958ebc2c3ff3681cc45fc1fa6da73d1f96ae56b458b"
EXPECTED_SOURCE_RESUME_SHA256 = "61d32fab3995599286f6eee3b30f24b3bea48042d2528392fdd1ca68e9bf60c4"
EXPECTED_QUALIFICATION_REPORT_SHA256 = (
    "e1ead1779c568cacfcebda5df435175f505c9e7218592c37b1187faeaf929f25"
)
EXPECTED_FROZEN_ARCHIVE_SHA256 = "e228a3920c47f56a7eac6d1452f996d9709721f82536240d96dbe6e0c6e1830f"
EXPECTED_STEP_AUDIT_REPORT_SHA256 = (
    "8f76e9b5bd9cb1cecc782b207c3694b332926fb9f34b5be1fea8fd31b2deb6dd"
)
EXPECTED_STOPPED_CONTINUATION_REPORT_SHA256 = (
    "7790d61241280dc315323563058858e1c16f9e625f459ad2b84f5d8cbc74ee40"
)
EXPECTED_PROJECTED_DISPLACEMENT_SHA256 = step_audit.EXPECTED_PROJECTED_DISPLACEMENT_SHA256
EXPECTED_COMPLETE_PRIMAL_OBJECTIVE = step_audit.EXPECTED_COMPLETE_PRIMAL_OBJECTIVE
EXPECTED_PENDING_OPTIMIZER_SHA256 = step_audit.EXPECTED_PENDING_OPTIMIZER_SHA256
EXPECTED_ACCEPTED_UPDATES = 20
DEVELOPMENT_DAMPING_IMPROVEMENT = 0.001
ARCHIVE_NAME = "accepted-update-21-transaction.pt"
PRODUCER_REPORT_NAME = "producer-report.json"
FINAL_REPORT_NAME = "report.json"
PRODUCER_STARTED_NAME = "producer-started.json"
VERIFIER_STARTED_NAME = "verifier-started.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("produce", "verify", "finalize-interrupted"), required=True
    )
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
            REPO_ROOT / "runs/variable-height-hover/full-native-endpoint-damping-step-audit-001"
        ),
    )
    parser.add_argument(
        "--source-fit-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-corrected-fitting-001"
        ),
    )
    parser.add_argument(
        "--qualification-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-fp64-frozen-input-audit-001"
        ),
    )
    parser.add_argument(
        "--step-audit-report",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-corrected-step-audit-001/report.json"
        ),
    )
    parser.add_argument(
        "--stopped-continuation-report",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-corrected-fitting-001/report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-update21-snapshot-audit-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def locked_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "graph": args.graph,
        "checkpoint": args.checkpoint,
        "cache_manifest": args.source_audit_dir / "immutable-cache-v1/manifest.json",
        "source_report": args.source_fit_dir / "report.json",
        "source_resume": args.source_fit_dir / "resume.pt",
        "qualification_report": args.qualification_dir / "report.json",
        "frozen_input_archive": args.qualification_dir / "frozen-inputs.pt",
        "step_audit_report": args.step_audit_report,
        "stopped_continuation_report": args.stopped_continuation_report,
    }


def expected_locked_hashes() -> dict[str, str]:
    return {
        "graph": EXPECTED_GRAPH_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "cache_manifest": base.EXPECTED_SOURCE_CACHE_MANIFEST_SHA256,
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume": EXPECTED_SOURCE_RESUME_SHA256,
        "qualification_report": EXPECTED_QUALIFICATION_REPORT_SHA256,
        "frozen_input_archive": EXPECTED_FROZEN_ARCHIVE_SHA256,
        "step_audit_report": EXPECTED_STEP_AUDIT_REPORT_SHA256,
        "stopped_continuation_report": EXPECTED_STOPPED_CONTINUATION_REPORT_SHA256,
    }


def validate_args(args: argparse.Namespace) -> None:
    missing = [path for path in locked_input_paths(args).values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input: {missing[0]}")
    if args.smoke_test:
        raise SystemExit("this preregistered snapshot audit has no smoke variant")
    producer_report = args.output_dir / PRODUCER_REPORT_NAME
    archive = args.output_dir / ARCHIVE_NAME
    final_report = args.output_dir / FINAL_REPORT_NAME
    producer_started = args.output_dir / PRODUCER_STARTED_NAME
    verifier_started = args.output_dir / VERIFIER_STARTED_NAME
    if args.phase == "produce" and any(
        path.exists() for path in (producer_started, producer_report, archive, final_report)
    ):
        raise SystemExit("producer outputs already exist; this audit cannot be retried")
    if args.phase == "verify" and not producer_report.is_file():
        raise SystemExit("the independent verifier requires a completed producer report")
    if args.phase == "verify" and any(path.exists() for path in (verifier_started, final_report)):
        raise SystemExit("the verifier report already exists; this audit cannot be retried")
    if args.phase == "finalize-interrupted":
        interrupted_producer = producer_started.is_file() and not producer_report.is_file()
        interrupted_verifier = verifier_started.is_file() and not final_report.is_file()
        if final_report.exists() or not (interrupted_producer or interrupted_verifier):
            raise SystemExit("no unfinished started phase is available to finalize")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "locked_input_sha256": expected_locked_hashes(),
        "producer": {
            "source_total_accepted_updates": EXPECTED_ACCEPTED_UPDATES,
            "frozen_primary_reconstructed_exactly": True,
            "ordinary_then_repair_selection_runs": 1,
            "repair_arithmetic_unchanged": True,
            "prior_repaired_candidate_hash_required": False,
            "all_existing_repair_and_training_gates_required": True,
            "training_endpoint_damping_improvement_minimum": corrected.ENDPOINT_DAMPING_IMPROVEMENT,
            "fixed_development_endpoint_damping_improvement_minimum": (
                DEVELOPMENT_DAMPING_IMPROVEMENT
            ),
            "fixed_development_original_source_preservation_required": True,
            "alternative_candidates_or_retries": False,
        },
        "archive": {
            "candidate_parameter_tensors": True,
            "pending_adam_state": True,
            "repair_constraint_rows": True,
            "authoritative_endpoint_damping_row": True,
            "full_correction_direction": True,
            "metrics_specs_and_provenance": True,
            "semantic_hash_each_tensor_tree": True,
            "complete_file_sha256": True,
            "ignored_run_artifact_not_git_or_lfs": True,
        },
        "verifier": {
            "separate_process": True,
            "repair_or_primary_recomputation": False,
            "exact_candidate_and_optimizer_identity": True,
            "archive_file_hash_unchanged_after_load": True,
            "training_and_fixed_development_replay_use_existing_tolerances": True,
        },
        "restore_update_20_state_exactly_each_process": True,
        "fresh_data": False,
        "closed_loop_hover": False,
        "promotion": False,
        "pass_authorizes": (
            "a separately preregistered continuation that loads accepted update 21 "
            "exactly and begins fresh proposal generation at update 22"
        ),
    }


def to_cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def snapshot_semantic_hashes(snapshot: dict[str, Any]) -> dict[str, str]:
    repair = snapshot["repair_tensor_capture"]
    return {
        "candidate_parameters": audit.semantic_sha256(snapshot["candidate_parameters"]),
        "pending_optimizer_state": audit.semantic_sha256(snapshot["pending_optimizer_state"]),
        "solver_constraint_rows": audit.semantic_sha256(repair["solver_constraint_rows"]),
        "authoritative_endpoint_damping_row": audit.semantic_sha256(
            repair["authoritative_endpoint_damping_row"]
        ),
        "full_correction_direction": audit.semantic_sha256(repair["full_correction_direction"]),
        "starting_candidate_parameters": audit.semantic_sha256(
            repair["starting_candidate_parameters"]
        ),
        "full_correction_parameters": audit.semantic_sha256(repair["full_correction_parameters"]),
        "repair_tensor_capture": audit.semantic_sha256(repair),
    }


def repaired_selection_controls_pass(
    selected_scale: float | None,
    trials: list[dict[str, Any]],
    repair_controls: dict[str, Any],
    capture: dict[str, Any] | None,
    pending_optimizer_sha256: str | None,
) -> bool:
    accepted_trial = next(
        (trial for trial in trials if trial.get("decision", {}).get("pass")), None
    )
    return bool(
        repair_controls["pass"]
        and repair_controls["repair_path_present"]
        and accepted_trial is not None
        and accepted_trial.get("acceptance_kind") == "repaired"
        and selected_scale == corrected.REPAIR_PROPOSAL_SCALE
        and capture
        and pending_optimizer_sha256 == EXPECTED_PENDING_OPTIMIZER_SHA256
    )


def verifier_replay_preconditions_pass(
    *,
    producer_identity_pass: bool,
    snapshot_identity_pass: bool,
    semantic_pass: bool,
    baseline_reproductions: dict[str, dict[str, Any]],
) -> bool:
    return bool(
        producer_identity_pass
        and snapshot_identity_pass
        and semantic_pass
        and all(item["pass"] for item in baseline_reproductions.values())
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    with staging.open("w", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(value, indent=2, sort_keys=True)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)


def write_phase_marker(args: argparse.Namespace) -> Path:
    name = PRODUCER_STARTED_NAME if args.phase == "produce" else VERIFIER_STARTED_NAME
    marker = args.output_dir / name
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "phase": args.phase,
        "no_retry": True,
    }
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return marker


def finalize_interrupted(args: argparse.Namespace) -> int:
    producer_started = args.output_dir / PRODUCER_STARTED_NAME
    verifier_started = args.output_dir / VERIFIER_STARTED_NAME
    producer_report_path = args.output_dir / PRODUCER_REPORT_NAME
    archive_path = args.output_dir / ARCHIVE_NAME
    if verifier_started.is_file():
        classification = "snapshot_verifier_interrupted_after_phase_start"
        interrupted_phase = "verifier"
    else:
        classification = "snapshot_producer_interrupted_after_phase_start"
        interrupted_phase = "producer"
    report = {
        "experiment": EXPERIMENT,
        "status": "nonpromotional_snapshot_qualification",
        "pass": False,
        "classification": classification,
        "protocol": protocol_manifest(),
        "interrupted_phase": interrupted_phase,
        "phase_markers": {
            "producer": (
                responsibility.file_sha256(producer_started) if producer_started.is_file() else None
            ),
            "verifier": (
                responsibility.file_sha256(verifier_started) if verifier_started.is_file() else None
            ),
        },
        "producer_report_sha256": (
            responsibility.file_sha256(producer_report_path)
            if producer_report_path.is_file()
            else None
        ),
        "diagnostic_archive_sha256": (
            responsibility.file_sha256(archive_path) if archive_path.is_file() else None
        ),
        "phase_retried": False,
        "accepted_update_21_snapshot_authorized": False,
        "closed_loop_hover_run": False,
        "promoted": False,
    }
    output = args.output_dir / FINAL_REPORT_NAME
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": False,
                "classification": classification,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def validate_report_identities(args: argparse.Namespace) -> dict[str, Any]:
    source_report = json.loads((args.source_fit_dir / "report.json").read_text(encoding="utf-8"))
    qualification = json.loads((args.qualification_dir / "report.json").read_text(encoding="utf-8"))
    step_report = json.loads(args.step_audit_report.read_text(encoding="utf-8"))
    stopped = json.loads(args.stopped_continuation_report.read_text(encoding="utf-8"))
    if (
        source_report.get("experiment"),
        source_report.get("protocol", {}).get("protocol_commit"),
        source_report.get("accepted_updates"),
    ) != (corrected.EXPERIMENT, corrected.PROTOCOL_COMMIT, EXPECTED_ACCEPTED_UPDATES):
        raise SystemExit("source update-20 report identity does not match the protocol")
    if (
        qualification.get("experiment"),
        qualification.get("classification"),
        qualification.get("pass"),
        qualification.get("fp64_projection_implementation_authorized"),
    ) != (frozen.EXPERIMENT, "identical_input_fp64_projection_qualified", True, True):
        raise SystemExit("FP64 qualification identity does not authorize this audit")
    if (
        step_report.get("experiment"),
        step_report.get("classification"),
        step_report.get("pass"),
        step_report.get("fp64_corrected_continuation_authorized"),
    ) != (
        step_audit.EXPERIMENT,
        "fp64_repaired_update_21_transfers_to_development",
        True,
        True,
    ):
        raise SystemExit("restored update-21 audit identity does not authorize this audit")
    last = stopped.get("history", [{}])[-1]
    reproduction = last.get("trials", [{}])[-1].get("registered_update_21_reproduction", {})
    if (
        stopped.get("experiment"),
        stopped.get("protocol", {}).get("protocol_commit"),
        stopped.get("pass"),
        stopped.get("accepted_updates"),
        last.get("update"),
        last.get("accepted"),
        reproduction.get("pass"),
    ) != (continuation.EXPERIMENT, continuation.PROTOCOL_COMMIT, False, 20, 21, False, False):
        raise SystemExit("stopped continuation identity does not match the control failure")
    return {
        "source_report": source_report,
        "qualification": qualification,
        "step_report": step_report,
        "stopped_continuation": stopped,
    }


def load_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    hashes = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    if hashes != expected_locked_hashes():
        raise SystemExit("one or more hash-locked inputs do not match the protocol")
    reports = validate_report_identities(args)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded.get("graph_sha256") != EXPECTED_GRAPH_SHA256:
        raise SystemExit("source checkpoint graph identity does not match the protocol")
    source = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    student = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    resume = torch.load(args.source_fit_dir / "resume.pt", map_location=device, weights_only=True)
    if (
        resume.get("experiment"),
        resume.get("protocol_commit"),
        resume.get("accepted_updates"),
        resume.get("run_state"),
    ) != (corrected.EXPERIMENT, corrected.PROTOCOL_COMMIT, 20, "stopped"):
        raise SystemExit("source update-20 resume identity does not match the protocol")
    student.load_state_dict(resume["controller"])
    optimizer = continuation.fp64.make_optimizer(student)
    optimizer.load_state_dict(resume["optimizer"])
    source_parameters = joint._copy_parameters(student)
    source_optimizer = copy.deepcopy(optimizer.state_dict())
    config = HoverConfig()
    (
        train_factorial,
        train_attitude,
        development_factorial,
        development_attitude,
        cache_integrity,
    ) = endpoint.load_or_create_caches(
        args.source_audit_dir / "immutable-cache-v1",
        source=source,
        device=device,
        config=config,
        graph_sha256=EXPECTED_GRAPH_SHA256,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    if cache_integrity["created_this_run"]:
        raise SystemExit("authorized immutable cache was regenerated")
    if not cache_integrity["all_persisted_file_and_tensor_hashes_match"]:
        raise SystemExit("authorized immutable cache failed its integrity check")
    endpoint_scale = endpoint.endpoint_damping_scale(train_factorial)
    scales = joint.training_teacher_scales(train_factorial)
    scales["damping"] = endpoint_scale
    source_training, _ = endpoint.evaluate(
        source,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    update20_training, _ = endpoint.evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    source_development, _ = endpoint.evaluate(
        source,
        development_factorial,
        development_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    update20_development, _ = endpoint.evaluate(
        student,
        development_factorial,
        development_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    registered_update20_development = next(
        entry["metrics"]
        for entry in reports["source_report"]["development_history"]
        if int(entry["update"]) == 20
    )
    reproductions = {
        "source_training": audit.numeric_tree_comparison(
            source_training, reports["source_report"]["source_training"]
        ),
        "update_20_training": audit.numeric_tree_comparison(
            update20_training, reports["source_report"]["final_training"]
        ),
        "source_development": audit.numeric_tree_comparison(
            source_development, reports["source_report"]["source_development"]
        ),
        "update_20_development": audit.numeric_tree_comparison(
            update20_development, registered_update20_development
        ),
    }
    return {
        "hashes": hashes,
        "reports": reports,
        "source": source,
        "student": student,
        "optimizer": optimizer,
        "source_parameters": source_parameters,
        "source_optimizer": source_optimizer,
        "train_factorial": train_factorial,
        "train_attitude": train_attitude,
        "development_factorial": development_factorial,
        "development_attitude": development_attitude,
        "cache_integrity": cache_integrity,
        "endpoint_scale": endpoint_scale,
        "scales": scales,
        "source_training": source_training,
        "update20_training": update20_training,
        "source_development": source_development,
        "update20_development": update20_development,
        "reproductions": reproductions,
    }


@contextmanager
def producer_runtime(frozen_archive: dict[str, Any]) -> Iterator[dict[str, Any]]:
    previous = {
        "continuation_archive": continuation._FROZEN_ARCHIVE_CPU,
        "continuation_update": continuation._UPDATE_NUMBER,
        "corrected_capture": corrected._REPAIR_TENSOR_CAPTURE,
        "corrected_guard": corrected._GUARD_REPORT,
        "corrected_update": corrected._UPDATE_NUMBER,
        "corrected_optimizer": corrected._PENDING_OPTIMIZER,
        "corrected_optimizer_before": corrected._OPTIMIZER_BEFORE_PROPOSAL,
        "corrected_optimizer_after": corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256,
        "canonical_base": canonical._AUTHORITATIVE_BASE,
        "canonical_candidate": canonical._AUTHORITATIVE_CANDIDATE,
    }
    capture: dict[str, Any] = {}
    continuation._FROZEN_ARCHIVE_CPU = frozen_archive
    continuation._UPDATE_NUMBER = EXPECTED_ACCEPTED_UPDATES
    corrected._REPAIR_TENSOR_CAPTURE = capture
    corrected._GUARD_REPORT = None
    try:
        yield capture
    finally:
        continuation._FROZEN_ARCHIVE_CPU = previous["continuation_archive"]
        continuation._UPDATE_NUMBER = previous["continuation_update"]
        corrected._REPAIR_TENSOR_CAPTURE = previous["corrected_capture"]
        corrected._GUARD_REPORT = previous["corrected_guard"]
        corrected._UPDATE_NUMBER = previous["corrected_update"]
        corrected._PENDING_OPTIMIZER = previous["corrected_optimizer"]
        corrected._OPTIMIZER_BEFORE_PROPOSAL = previous["corrected_optimizer_before"]
        corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = previous["corrected_optimizer_after"]
        canonical._AUTHORITATIVE_BASE = previous["canonical_base"]
        canonical._AUTHORITATIVE_CANDIDATE = previous["canonical_candidate"]


def restored_exactly(context: dict[str, Any]) -> dict[str, Any]:
    student = context["student"]
    optimizer = context["optimizer"]
    parameter_pass = all(
        torch.equal(getattr(student, name).detach(), context["source_parameters"][name])
        for name in joint.PARAMETER_FAMILIES
    )
    optimizer_pass = audit.trees_equal(optimizer.state_dict(), context["source_optimizer"])
    return {
        "pass": bool(parameter_pass and optimizer_pass),
        "parameters_exact": parameter_pass,
        "optimizer_exact": optimizer_pass,
        "parameter_sha256": audit.semantic_sha256(joint._copy_parameters(student)),
        "optimizer_sha256": audit.semantic_sha256(optimizer.state_dict()),
    }


def produce(args: argparse.Namespace, device: torch.device) -> int:
    started = perf_counter()
    context = load_context(args, device)
    student = context["student"]
    optimizer = context["optimizer"]
    frozen_cpu = torch.load(
        args.qualification_dir / "frozen-inputs.pt", map_location="cpu", weights_only=True
    )
    frozen_hashes = frozen.archive_tensor_hashes(frozen_cpu)
    frozen_valid = bool(
        frozen_hashes == frozen_cpu.get("tensor_semantic_sha256")
        and frozen_hashes
        == context["reports"]["qualification"]
        .get("frozen_input_archive", {})
        .get("tensor_semantic_sha256")
    )
    frozen_parameters = frozen_cpu["current_parameters"]
    current_parameter_match = step_audit.parameter_tensor_comparison(
        context["source_parameters"], frozen_parameters
    )
    input_controls_pass = bool(
        frozen_valid
        and current_parameter_match["pass"]
        and all(item["pass"] for item in context["reproductions"].values())
    )
    selected_scale: float | None = None
    selected_training: dict[str, Any] | None = None
    selected_development: dict[str, Any] | None = None
    development_decision: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    projection: dict[str, Any] = {"pass_after_parameter_bounds": False}
    selected_parameters: dict[str, Tensor] | None = None
    pending_optimizer: dict[str, Any] | None = None
    capture_copy: dict[str, Any] | None = None
    snapshot_hashes: dict[str, str] | None = None
    archive_sha256: str | None = None
    repair_controls: dict[str, Any] = {
        "pass": False,
        "repair_path_present": False,
        "checks": {},
    }
    selection_controls_pass = False
    pending_optimizer_sha256: str | None = None
    archive_path = args.output_dir / ARCHIVE_NAME
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    print(json.dumps({"stage": "producer_reconstructing_update_21"}), flush=True)
    with audit.transactional_restoration(student, optimizer):
        with producer_runtime(frozen_cpu) as capture:
            displacement, raw_gradients, projection = continuation.make_projected_proposal(
                student,
                optimizer,
                context["source_training"],
                context["update20_training"],
                context["train_factorial"],
                context["train_attitude"],
                context["scales"],
                context["endpoint_scale"],
                device=device,
            )
            if input_controls_pass and projection["pass_after_parameter_bounds"]:
                print(
                    json.dumps({"stage": "producer_single_ordinary_then_repair_selection"}),
                    flush=True,
                )
                optimizer_after_sha256 = audit.semantic_sha256(optimizer.state_dict())
                with step_audit.corrected_trial_runtime(
                    optimizer, optimizer_before, optimizer_after_sha256
                ):
                    selected_scale, selected_training, trials = corrected.find_safe_trial(
                        student,
                        context["source_parameters"],
                        displacement,
                        raw_gradients,
                        context["source_training"],
                        context["update20_training"],
                        context["train_factorial"],
                        context["train_attitude"],
                        context["scales"],
                        context["endpoint_scale"],
                        device=device,
                    )
            if selected_training is not None:
                repair_controls = step_audit.repair_numerical_control_report(trials)
                pending_optimizer = copy.deepcopy(optimizer.state_dict())
                pending_optimizer_sha256 = audit.semantic_sha256(pending_optimizer)
                capture_copy = to_cpu_tree(capture)
                selection_controls_pass = repaired_selection_controls_pass(
                    selected_scale,
                    trials,
                    repair_controls,
                    capture_copy,
                    pending_optimizer_sha256,
                )
            if selection_controls_pass:
                selected_parameters = joint._copy_parameters(student)
                print(json.dumps({"stage": "producer_fixed_development_replay"}), flush=True)
                selected_development, _ = endpoint.evaluate(
                    student,
                    context["development_factorial"],
                    context["development_attitude"],
                    context["scales"],
                    endpoint_scale=context["endpoint_scale"],
                    device=device,
                )
                development_decision = step_audit.development_transfer_decision(
                    context["source_development"],
                    context["update20_development"],
                    selected_development,
                )
                prearchive_pass = bool(
                    projection["pass_after_parameter_bounds"]
                    and selection_controls_pass
                    and development_decision["pass"]
                )
                if prearchive_pass:
                    snapshot = {
                        "experiment": EXPERIMENT,
                        "protocol_commit": PROTOCOL_COMMIT,
                        "accepted_update": 21,
                        "locked_input_sha256": context["hashes"],
                        "candidate_parameters": to_cpu_tree(selected_parameters),
                        "pending_optimizer_state": to_cpu_tree(pending_optimizer),
                        "repair_tensor_capture": capture_copy,
                        "selected_training_metrics": copy.deepcopy(selected_training),
                        "selected_development_metrics": copy.deepcopy(selected_development),
                        "development_transfer_decision": copy.deepcopy(development_decision),
                        "selection_trials": copy.deepcopy(trials),
                        "primary_projection": copy.deepcopy(projection),
                        "source_training_metrics": copy.deepcopy(context["source_training"]),
                        "update20_training_metrics": copy.deepcopy(context["update20_training"]),
                        "source_development_metrics": copy.deepcopy(context["source_development"]),
                        "update20_development_metrics": copy.deepcopy(
                            context["update20_development"]
                        ),
                    }
                    snapshot_hashes = snapshot_semantic_hashes(snapshot)
                    snapshot["tensor_semantic_sha256"] = snapshot_hashes
                    base.atomic_torch_save(snapshot, archive_path)
                    archive_sha256 = responsibility.file_sha256(archive_path)
    restoration = restored_exactly(context)
    hashes_after = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    sources_unchanged = hashes_after == context["hashes"]
    producer_pass = bool(
        input_controls_pass
        and projection["pass_after_parameter_bounds"]
        and selected_training is not None
        and selection_controls_pass
        and development_decision is not None
        and development_decision["pass"]
        and repair_controls["pass"]
        and repair_controls["repair_path_present"]
        and capture_copy
        and snapshot_hashes
        and archive_sha256
        and restoration["pass"]
        and sources_unchanged
    )
    if not input_controls_pass or not projection["pass_after_parameter_bounds"]:
        classification = "producer_input_or_primary_control_failure"
    elif repair_controls["repair_path_present"] and not repair_controls["pass"]:
        classification = "producer_repair_control_failure"
    elif selected_training is None:
        classification = "producer_no_training_candidate_passed"
    elif not selection_controls_pass:
        classification = "producer_repaired_selection_control_failure"
    elif development_decision is None or not development_decision["pass"]:
        classification = "producer_candidate_failed_fixed_development"
    elif not snapshot_hashes or not archive_sha256:
        classification = "producer_snapshot_write_control_failure"
    elif not restoration["pass"] or not sources_unchanged:
        classification = "producer_restoration_control_failure"
    else:
        classification = "accepted_update_21_transaction_snapshot_produced"
    report = {
        "experiment": EXPERIMENT,
        "phase": "producer",
        "status": "nonpromotional_snapshot_producer",
        "pass": producer_pass,
        "classification": classification,
        "protocol": protocol_manifest(),
        "source": {
            "hashes_before": context["hashes"],
            "hashes_after": hashes_after,
            "all_locked_inputs_unchanged": sources_unchanged,
        },
        "cache_integrity": context["cache_integrity"],
        "input_reproductions": context["reproductions"],
        "frozen_archive_semantic_hashes": frozen_hashes,
        "frozen_archive_valid": frozen_valid,
        "current_parameters_match_frozen_update_20": current_parameter_match,
        "input_controls_pass": input_controls_pass,
        "primary_projection": projection,
        "selection_trials": trials,
        "selected_scale": selected_scale,
        "selected_parameter_sha256": (
            audit.semantic_sha256(selected_parameters) if selected_parameters is not None else None
        ),
        "selected_training_metrics": selected_training,
        "selected_development_metrics": selected_development,
        "development_transfer_decision": development_decision,
        "repair_numerical_controls": repair_controls,
        "repaired_selection_controls_pass": selection_controls_pass,
        "pending_optimizer_sha256": pending_optimizer_sha256,
        "snapshot": {
            "path": responsibility.stable_path(archive_path) if archive_sha256 else None,
            "file_sha256": archive_sha256,
            "tensor_semantic_sha256": snapshot_hashes,
            "written": archive_sha256 is not None,
        },
        "restoration": restoration,
        "candidate_retained_in_controller": False,
        "fresh_data_used": False,
        "closed_loop_hover_run": False,
        "promoted": False,
        "elapsed_seconds": perf_counter() - started,
    }
    producer_report = args.output_dir / PRODUCER_REPORT_NAME
    write_json(producer_report, report)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(producer_report),
                "pass": producer_pass,
                "classification": classification,
                "snapshot_sha256": archive_sha256,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def write_verifier_failure(
    args: argparse.Namespace,
    producer_report: dict[str, Any],
    producer_report_sha256: str,
    *,
    classification: str,
    producer_identity_pass: bool,
    detail: str,
    started: float,
) -> int:
    final = {
        "experiment": EXPERIMENT,
        "status": "nonpromotional_snapshot_qualification",
        "pass": False,
        "classification": classification,
        "protocol": protocol_manifest(),
        "detail": detail,
        "producer_report_sha256": producer_report_sha256,
        "producer_identity_pass": producer_identity_pass,
        "producer": producer_report,
        "accepted_update_21_snapshot_authorized": False,
        "closed_loop_hover_run": False,
        "promoted": False,
        "elapsed_seconds": perf_counter() - started,
    }
    output = args.output_dir / FINAL_REPORT_NAME
    write_json(output, final)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": False,
                "classification": classification,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def verify(args: argparse.Namespace, device: torch.device) -> int:
    started = perf_counter()
    producer_path = args.output_dir / PRODUCER_REPORT_NAME
    producer_hash_before = responsibility.file_sha256(producer_path)
    producer_report = json.loads(producer_path.read_text(encoding="utf-8"))
    producer_identity_pass = (
        producer_report.get("experiment") == EXPERIMENT
        and producer_report.get("phase") == "producer"
        and producer_report.get("protocol", {}).get("protocol_commit") == PROTOCOL_COMMIT
        and producer_report.get("classification")
        == "accepted_update_21_transaction_snapshot_produced"
    )
    if not producer_report.get("pass", False):
        return write_verifier_failure(
            args,
            producer_report,
            producer_hash_before,
            classification="snapshot_producer_failed",
            producer_identity_pass=producer_identity_pass,
            detail="producer did not qualify and write an accepted transaction",
            started=started,
        )
    archive_path = args.output_dir / ARCHIVE_NAME
    if not archive_path.is_file():
        return write_verifier_failure(
            args,
            producer_report,
            producer_hash_before,
            classification="snapshot_verifier_missing_archive",
            producer_identity_pass=producer_identity_pass,
            detail="passing producer report has no transaction snapshot",
            started=started,
        )
    archive_hash_before = responsibility.file_sha256(archive_path)
    if archive_hash_before != producer_report.get("snapshot", {}).get("file_sha256"):
        return write_verifier_failure(
            args,
            producer_report,
            producer_hash_before,
            classification="snapshot_verifier_archive_file_hash_failure",
            producer_identity_pass=producer_identity_pass,
            detail="transaction snapshot file hash does not match producer report",
            started=started,
        )
    print(json.dumps({"stage": "verifier_loading_snapshot_without_recomputation"}), flush=True)
    snapshot = torch.load(archive_path, map_location="cpu", weights_only=True)
    snapshot_identity_pass = (
        snapshot.get("experiment") == EXPERIMENT
        and snapshot.get("protocol_commit") == PROTOCOL_COMMIT
        and snapshot.get("accepted_update") == 21
        and snapshot.get("locked_input_sha256") == expected_locked_hashes()
    )
    semantic_actual = snapshot_semantic_hashes(snapshot)
    semantic_expected = snapshot.get("tensor_semantic_sha256")
    semantic_pass = isinstance(semantic_expected, dict) and semantic_actual == semantic_expected
    expected_optimizer_sha256 = (
        semantic_expected.get("pending_optimizer_state")
        if isinstance(semantic_expected, dict)
        else None
    )
    context = load_context(args, device)
    baseline_reproductions_pass = all(item["pass"] for item in context["reproductions"].values())
    replay_preconditions_pass = verifier_replay_preconditions_pass(
        producer_identity_pass=producer_identity_pass,
        snapshot_identity_pass=snapshot_identity_pass,
        semantic_pass=semantic_pass,
        baseline_reproductions=context["reproductions"],
    )
    student = context["student"]
    optimizer = context["optimizer"]
    training_replay: dict[str, Any] | None = None
    development_replay: dict[str, Any] | None = None
    parameter_load_match: dict[str, Any] | None = None
    optimizer_load_sha256: str | None = None
    parameter_stable_after_replay = False
    optimizer_stable_after_replay = False
    if replay_preconditions_pass:
        with audit.transactional_restoration(student, optimizer):
            candidate_device = {
                name: value.to(device) for name, value in snapshot["candidate_parameters"].items()
            }
            joint._load_parameters(student, candidate_device)
            optimizer.load_state_dict(snapshot["pending_optimizer_state"])
            parameter_load_match = step_audit.parameter_tensor_comparison(
                joint._copy_parameters(student), candidate_device
            )
            optimizer_load_sha256 = audit.semantic_sha256(optimizer.state_dict())
            training_metrics, _ = endpoint.evaluate(
                student,
                context["train_factorial"],
                context["train_attitude"],
                context["scales"],
                endpoint_scale=context["endpoint_scale"],
                device=device,
            )
            development_metrics, _ = endpoint.evaluate(
                student,
                context["development_factorial"],
                context["development_attitude"],
                context["scales"],
                endpoint_scale=context["endpoint_scale"],
                device=device,
            )
            training_replay = audit.numeric_tree_comparison(
                training_metrics, snapshot["selected_training_metrics"]
            )
            development_replay = audit.numeric_tree_comparison(
                development_metrics, snapshot["selected_development_metrics"]
            )
            parameter_stable_after_replay = all(
                torch.equal(getattr(student, name).detach(), candidate_device[name])
                for name in joint.PARAMETER_FAMILIES
            )
            optimizer_stable_after_replay = (
                audit.semantic_sha256(optimizer.state_dict()) == expected_optimizer_sha256
            )
    restoration = restored_exactly(context)
    archive_hash_after = responsibility.file_sha256(archive_path)
    producer_hash_after = responsibility.file_sha256(producer_path)
    locked_after = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    inputs_unchanged = locked_after == context["hashes"]
    passed = bool(
        producer_identity_pass
        and snapshot_identity_pass
        and semantic_pass
        and baseline_reproductions_pass
        and replay_preconditions_pass
        and parameter_load_match
        and parameter_load_match["pass"]
        and optimizer_load_sha256 == expected_optimizer_sha256
        and training_replay
        and training_replay["pass"]
        and development_replay
        and development_replay["pass"]
        and parameter_stable_after_replay
        and optimizer_stable_after_replay
        and archive_hash_after == archive_hash_before
        and producer_hash_after == producer_hash_before
        and inputs_unchanged
        and restoration["pass"]
    )
    classification = (
        "accepted_update_21_transaction_snapshot_qualified"
        if passed
        else "snapshot_verifier_control_failure"
    )
    final = {
        "experiment": EXPERIMENT,
        "status": "nonpromotional_snapshot_qualification",
        "pass": passed,
        "classification": classification,
        "protocol": protocol_manifest(),
        "producer_report_sha256_before": producer_hash_before,
        "producer_report_sha256_after": producer_hash_after,
        "producer_report_unchanged": producer_hash_after == producer_hash_before,
        "producer_identity_pass": producer_identity_pass,
        "baseline_reproductions": context["reproductions"],
        "baseline_reproductions_pass": baseline_reproductions_pass,
        "replay_preconditions_pass": replay_preconditions_pass,
        "producer": producer_report,
        "snapshot": {
            "path": responsibility.stable_path(archive_path),
            "file_sha256_before": archive_hash_before,
            "file_sha256_after": archive_hash_after,
            "file_unchanged_after_load": archive_hash_after == archive_hash_before,
            "identity_pass": snapshot_identity_pass,
            "semantic_hashes_expected": semantic_expected,
            "semantic_hashes_actual": semantic_actual,
            "semantic_hashes_pass": semantic_pass,
        },
        "loaded_candidate_parameters": parameter_load_match,
        "loaded_optimizer_sha256": optimizer_load_sha256,
        "loaded_optimizer_exact": optimizer_load_sha256 == expected_optimizer_sha256,
        "training_replay": training_replay,
        "fixed_development_replay": development_replay,
        "candidate_parameters_stable_after_replay": parameter_stable_after_replay,
        "pending_optimizer_stable_after_replay": optimizer_stable_after_replay,
        "locked_inputs_before": context["hashes"],
        "locked_inputs_after": locked_after,
        "locked_inputs_unchanged": inputs_unchanged,
        "source_restoration": restoration,
        "primary_or_repair_recomputed_by_verifier": False,
        "fresh_data_used": False,
        "candidate_retained_in_controller": False,
        "closed_loop_hover_run": False,
        "promoted": False,
        "accepted_update_21_snapshot_authorized": passed,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass authorizes only a separately registered continuation loading this exact "
            "accepted update-21 transaction and starting fresh proposals at update 22."
        ),
    }
    output = args.output_dir / FINAL_REPORT_NAME
    write_json(output, final)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": passed,
                "classification": classification,
                "accepted_update_21_snapshot_authorized": passed,
                "closed_loop_hover_run": False,
                "promoted": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.phase == "finalize-interrupted":
        return finalize_interrupted(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    write_phase_marker(args)
    if args.phase == "produce":
        return produce(args, device)
    return verify(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
