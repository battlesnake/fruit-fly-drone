#!/usr/bin/env python3
"""Continue FP64 corrected fitting from the qualified accepted update-21 snapshot."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_fp64_frozen_input as frozen  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_update21_snapshot as snapshot_audit  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402
import train_variable_height_full_native_d_first_fp64_corrected as continuation  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-snapshot-corrected-fitting-v1"
PROTOCOL_COMMIT = "36e573e"
EXPECTED_SOURCE_REPORT_SHA256 = snapshot_audit.EXPECTED_SOURCE_REPORT_SHA256
EXPECTED_SOURCE_RESUME_SHA256 = snapshot_audit.EXPECTED_SOURCE_RESUME_SHA256
EXPECTED_QUALIFICATION_REPORT_SHA256 = snapshot_audit.EXPECTED_QUALIFICATION_REPORT_SHA256
EXPECTED_FROZEN_ARCHIVE_SHA256 = snapshot_audit.EXPECTED_FROZEN_ARCHIVE_SHA256
EXPECTED_SNAPSHOT_REPORT_SHA256 = "c8f665732e2ebe84b0ee09fea8dee431e4e6ca021a8035bca1e9276d831413e5"
EXPECTED_SNAPSHOT_PRODUCER_REPORT_SHA256 = (
    "429e592696d114d0d8d5df396a32df205aae12707085d36e043928d87cd1fa3a"
)
EXPECTED_SNAPSHOT_ARCHIVE_SHA256 = (
    "fb4c3118348eee63e9d0365c64123a459aa8327cd39a056ae5496318cd6f9f42"
)
EXPECTED_SNAPSHOT_PARAMETER_SHA256 = (
    "223d3eda98a5c7c3280b28782d4f46ac5845c290e14ca11898749712cbba02e3"
)
EXPECTED_PENDING_OPTIMIZER_SHA256 = snapshot_audit.EXPECTED_PENDING_OPTIMIZER_SHA256
EXPECTED_INITIAL_ACCEPTED_UPDATES = 21
MAXIMUM_ACCEPTED_UPDATES = 50

_SNAPSHOT_CPU: dict[str, Any] | None = None
_SNAPSHOT_REFERENCE: dict[str, Any] | None = None
_SNAPSHOT_LOAD_CONTROL: dict[str, Any] | None = None

_ORIGINAL_BASE_VALUES = {
    "EXPERIMENT": base.EXPERIMENT,
    "PROTOCOL_COMMIT": base.PROTOCOL_COMMIT,
    "MAXIMUM_ACCEPTED_UPDATES": base.MAXIMUM_ACCEPTED_UPDATES,
    "DEVELOPMENT_PRESERVATION_REQUIRED": base.DEVELOPMENT_PRESERVATION_REQUIRED,
    "DEVELOPMENT_FAILURE_ROLLBACK_REQUIRED": (base.DEVELOPMENT_FAILURE_ROLLBACK_REQUIRED),
    "PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED": base.PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED,
    "parse_args": base.parse_args,
    "validate_args": base.validate_args,
    "protocol_manifest": base.protocol_manifest,
    "make_projected_proposal": base.make_projected_proposal,
    "install_trial": base.install_trial,
    "find_safe_trial": base.find_safe_trial,
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
        "--snapshot-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-update21-snapshot-audit-001"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-snapshot-corrected-fitting-001"
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
        "snapshot_report": args.snapshot_dir / snapshot_audit.FINAL_REPORT_NAME,
        "snapshot_producer_report": args.snapshot_dir / snapshot_audit.PRODUCER_REPORT_NAME,
        "snapshot_archive": args.snapshot_dir / snapshot_audit.ARCHIVE_NAME,
    }


def expected_locked_hashes() -> dict[str, str]:
    return {
        "graph": snapshot_audit.EXPECTED_GRAPH_SHA256,
        "checkpoint": snapshot_audit.EXPECTED_CHECKPOINT_SHA256,
        "cache_manifest": base.EXPECTED_SOURCE_CACHE_MANIFEST_SHA256,
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume": EXPECTED_SOURCE_RESUME_SHA256,
        "qualification_report": EXPECTED_QUALIFICATION_REPORT_SHA256,
        "frozen_input_archive": EXPECTED_FROZEN_ARCHIVE_SHA256,
        "snapshot_report": EXPECTED_SNAPSHOT_REPORT_SHA256,
        "snapshot_producer_report": EXPECTED_SNAPSHOT_PRODUCER_REPORT_SHA256,
        "snapshot_archive": EXPECTED_SNAPSHOT_ARCHIVE_SHA256,
    }


def validate_args(args: argparse.Namespace) -> None:
    paths = locked_input_paths(args)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input: {missing[0]}")
    if args.smoke_test:
        raise SystemExit("this preregistered snapshot-seeded continuation has no smoke variant")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("the snapshot-seeded continuation already has a final report")
    actual = {name: responsibility.file_sha256(path) for name, path in paths.items()}
    if actual != expected_locked_hashes():
        raise SystemExit("one or more hash-locked continuation inputs do not match")


def protocol_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(continuation.protocol_manifest())
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "starting_resume": {
                "source_experiment": corrected.EXPERIMENT,
                "source_protocol_commit": corrected.PROTOCOL_COMMIT,
                "source_sha256": EXPECTED_SOURCE_RESUME_SHA256,
                "source_accepted_updates": 20,
                "accepted_snapshot_update": EXPECTED_INITIAL_ACCEPTED_UPDATES,
                "candidate_parameter_sha256": EXPECTED_SNAPSHOT_PARAMETER_SHA256,
                "pending_optimizer_sha256": EXPECTED_PENDING_OPTIMIZER_SHA256,
                "copied_to_new_run": True,
                "source_overwritten": False,
            },
            "authorizing_snapshot_qualification": {
                "report_sha256": EXPECTED_SNAPSHOT_REPORT_SHA256,
                "producer_report_sha256": EXPECTED_SNAPSHOT_PRODUCER_REPORT_SHA256,
                "archive_sha256": EXPECTED_SNAPSHOT_ARCHIVE_SHA256,
                "required_pass": True,
            },
            "maximum_accepted_updates": MAXIMUM_ACCEPTED_UPDATES,
            "accepted_update_budget_is_total_not_additional": True,
            "development_totals": [30, 40, 50],
            "development_preservation_failure_is_terminal": True,
            "development_preservation_failure_rolls_back_transaction": True,
            "scheduled_candidate_persisted_before_development": True,
            "numerical_exceptions_restore_and_persist_stopped_resume": True,
            "update_21": {
                "source": "qualified accepted-transaction snapshot",
                "parameters_and_pending_adam_loaded_exactly": True,
                "primary_or_repair_recomputed": False,
                "history_entry_is_provenance_only": True,
                "fixed_development_result_is_not_a_selection_channel": True,
            },
            "before_update_22": {
                "candidate_parameter_hash_exact": True,
                "pending_optimizer_hash_exact": True,
                "source_and_candidate_training_metric_replay_required": True,
                "established_numerical_tree_tolerances": True,
                "failure_persists_stopped_resume_without_update_22": True,
            },
            "updates_22_through_50": {
                "proposal_inputs_generated_once_per_attempt": True,
                "primary_projection": "qualified FP64 active-set projection",
                "primary_solver_success_required": True,
                "original_unit_primal_violation_maximum": base.LINEAR_CONSTRAINT_TOLERANCE,
                "normalized_kkt_residual_maximum": continuation.fp64.NORMALIZED_KKT_RESIDUAL_LIMIT,
                "independent_exhaustive_support_check_required": True,
                "maximum_active_set_rounds": continuation.fp64.MAXIMUM_ACTIVE_SET_ROUNDS,
                "old_projector_fallback": False,
            },
            "repair_arithmetic_unchanged": True,
            "large_snapshot_remains_ignored_not_git_or_lfs": True,
        }
    )
    manifest.pop("authorizing_step_audit", None)
    return manifest


def validate_snapshot_history(payload: dict[str, Any]) -> None:
    matches = [
        entry
        for entry in payload.get("history", [])
        if int(entry.get("update", -1)) == EXPECTED_INITIAL_ACCEPTED_UPDATES
    ]
    if len(matches) != 1:
        raise SystemExit("continuation resume lacks its unique accepted snapshot update")
    entry = matches[0]
    provenance = entry.get("snapshot_provenance", {})
    if (
        entry.get("accepted"),
        entry.get("acceptance_kind"),
        provenance.get("snapshot_report_sha256"),
        provenance.get("snapshot_archive_sha256"),
        provenance.get("candidate_parameter_sha256"),
        provenance.get("pending_optimizer_sha256"),
    ) != (
        True,
        "qualified_snapshot_repaired",
        EXPECTED_SNAPSHOT_REPORT_SHA256,
        EXPECTED_SNAPSHOT_ARCHIVE_SHA256,
        EXPECTED_SNAPSHOT_PARAMETER_SHA256,
        EXPECTED_PENDING_OPTIMIZER_SHA256,
    ):
        raise SystemExit("accepted snapshot history provenance does not match the protocol")


def seed_resume(args: argparse.Namespace) -> int:
    global _SNAPSHOT_LOAD_CONTROL

    output = args.output_dir / "resume.pt"
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=True)
        identity = (
            payload.get("experiment"),
            payload.get("protocol_commit"),
            int(payload.get("accepted_updates", -1)),
        )
        if identity[0:2] != (EXPERIMENT, PROTOCOL_COMMIT):
            raise SystemExit("existing snapshot-seeded resume does not match the protocol")
        validate_snapshot_history(payload)
        found_control = False
        for entry in payload.get("history", []):
            control = entry.get("projection", {}).get("snapshot_seed_pre_update_22_control")
            if int(entry.get("update", -1)) == 22 and isinstance(control, dict):
                _SNAPSHOT_LOAD_CONTROL = copy.deepcopy(control)
                found_control = True
                break
        if identity[2] >= 22 and (
            not found_control or not _SNAPSHOT_LOAD_CONTROL or not _SNAPSHOT_LOAD_CONTROL["pass"]
        ):
            raise SystemExit("advanced resume lacks a passing pre-update-22 snapshot control")
        return identity[2]
    if _SNAPSHOT_CPU is None:
        raise RuntimeError("qualified accepted update-21 snapshot is unavailable")
    source_path = args.source_fit_dir / "resume.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if (
        source.get("experiment"),
        source.get("protocol_commit"),
        int(source.get("accepted_updates", -1)),
        source.get("run_state"),
    ) != (corrected.EXPERIMENT, corrected.PROTOCOL_COMMIT, 20, "stopped"):
        raise SystemExit("source resume is not the registered stopped update-20 state")
    seeded = copy.deepcopy(source)
    for name in continuation.joint.PARAMETER_FAMILIES:
        seeded["controller"][name] = _SNAPSHOT_CPU["candidate_parameters"][name].clone()
    seeded["optimizer"] = copy.deepcopy(_SNAPSHOT_CPU["pending_optimizer_state"])
    history = [
        entry
        for entry in source["history"]
        if bool(entry.get("accepted")) and int(entry.get("update", -1)) <= 20
    ]
    history.append(
        {
            "update": 21,
            "accepted": True,
            "accepted_scale": corrected.REPAIR_PROPOSAL_SCALE,
            "accepted_proposal_scale": corrected.REPAIR_PROPOSAL_SCALE,
            "accepted_correction_scale": 1.0,
            "acceptance_kind": "qualified_snapshot_repaired",
            "snapshot_training_metrics": copy.deepcopy(_SNAPSHOT_CPU["selected_training_metrics"]),
            "snapshot_provenance": {
                "snapshot_report_sha256": EXPECTED_SNAPSHOT_REPORT_SHA256,
                "snapshot_producer_report_sha256": (EXPECTED_SNAPSHOT_PRODUCER_REPORT_SHA256),
                "snapshot_archive_sha256": EXPECTED_SNAPSHOT_ARCHIVE_SHA256,
                "candidate_parameter_sha256": EXPECTED_SNAPSHOT_PARAMETER_SHA256,
                "pending_optimizer_sha256": EXPECTED_PENDING_OPTIMIZER_SHA256,
                "primary_or_repair_recomputed": False,
            },
            "trials": [],
        }
    )
    seeded.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "accepted_updates": EXPECTED_INITIAL_ACCEPTED_UPDATES,
            "history": history,
            "development_history": [
                entry
                for entry in source["development_history"]
                if int(entry.get("update", -1)) <= 20
            ],
            "run_state": "active",
            "qualification": None,
            "development_rollback": None,
        }
    )
    preflight = seeded.get("preflight")
    if not isinstance(preflight, dict) or not preflight.get("pass", False):
        raise SystemExit("source update-20 resume lacks its passing preflight")
    preflight["continued_from_passing_snapshot_seed_preflight"] = True
    base.atomic_torch_save(seeded, output)
    if responsibility.file_sha256(source_path) != EXPECTED_SOURCE_RESUME_SHA256:
        raise SystemExit("source resume changed while snapshot continuation was seeded")
    if (
        responsibility.file_sha256(args.snapshot_dir / snapshot_audit.ARCHIVE_NAME)
        != EXPECTED_SNAPSHOT_ARCHIVE_SHA256
    ):
        raise SystemExit("accepted transaction snapshot changed while resume was seeded")
    return EXPECTED_INITIAL_ACCEPTED_UPDATES


def snapshot_load_control(
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
) -> dict[str, Any]:
    if _SNAPSHOT_REFERENCE is None:
        raise RuntimeError("snapshot replay reference is unavailable")
    parameter_sha256 = audit.semantic_sha256(continuation.joint._copy_parameters(student))
    optimizer_sha256 = audit.semantic_sha256(optimizer.state_dict())
    source_replay = audit.numeric_tree_comparison(
        source_metrics, _SNAPSHOT_REFERENCE["source_training_metrics"]
    )
    candidate_replay = audit.numeric_tree_comparison(
        current_metrics, _SNAPSHOT_REFERENCE["selected_training_metrics"]
    )
    report = {
        "pass": bool(
            parameter_sha256 == EXPECTED_SNAPSHOT_PARAMETER_SHA256
            and optimizer_sha256 == EXPECTED_PENDING_OPTIMIZER_SHA256
            and source_replay["pass"]
            and candidate_replay["pass"]
        ),
        "candidate_parameter_sha256": parameter_sha256,
        "expected_candidate_parameter_sha256": EXPECTED_SNAPSHOT_PARAMETER_SHA256,
        "pending_optimizer_sha256": optimizer_sha256,
        "expected_pending_optimizer_sha256": EXPECTED_PENDING_OPTIMIZER_SHA256,
        "source_training_replay": source_replay,
        "candidate_training_replay": candidate_replay,
        "checked_before_update_22_proposal_generation": True,
    }
    return report


def make_projected_proposal(
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    factorial_cache: Any,
    attitude_cache: Any,
    scales: dict[str, float],
    endpoint_scale: float,
    *,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    global _SNAPSHOT_LOAD_CONTROL

    first_fresh_update = continuation._UPDATE_NUMBER == EXPECTED_INITIAL_ACCEPTED_UPDATES
    if first_fresh_update:
        _SNAPSHOT_LOAD_CONTROL = snapshot_load_control(
            student, optimizer, source_metrics, current_metrics
        )
        if not _SNAPSHOT_LOAD_CONTROL["pass"]:
            zero = {
                name: torch.zeros_like(getattr(student, name))
                for name in continuation.joint.PARAMETER_FAMILIES
            }
            return (
                zero,
                zero,
                {
                    "pass": False,
                    "pass_after_parameter_bounds": False,
                    "reason": "qualified update-21 snapshot failed pre-update-22 controls",
                    "snapshot_seed_pre_update_22_control": copy.deepcopy(_SNAPSHOT_LOAD_CONTROL),
                    "optimizer_step_calls": 0,
                    "fresh_proposal_inputs_generated": False,
                },
            )
    displacement, raw_gradients, projection = continuation.make_projected_proposal(
        student,
        optimizer,
        source_metrics,
        current_metrics,
        factorial_cache,
        attitude_cache,
        scales,
        endpoint_scale,
        device=device,
    )
    if first_fresh_update:
        projection["snapshot_seed_pre_update_22_control"] = copy.deepcopy(_SNAPSHOT_LOAD_CONTROL)
    return displacement, raw_gradients, projection


def validate_authorizations(args: argparse.Namespace) -> None:
    global _SNAPSHOT_CPU, _SNAPSHOT_REFERENCE

    qualification = json.loads((args.qualification_dir / "report.json").read_text(encoding="utf-8"))
    if (
        qualification.get("experiment"),
        qualification.get("classification"),
        qualification.get("pass"),
        qualification.get("fp64_projection_implementation_authorized"),
    ) != (frozen.EXPERIMENT, "identical_input_fp64_projection_qualified", True, True):
        raise SystemExit("FP64 qualification does not authorize continuation")
    report = json.loads(
        (args.snapshot_dir / snapshot_audit.FINAL_REPORT_NAME).read_text(encoding="utf-8")
    )
    producer = json.loads(
        (args.snapshot_dir / snapshot_audit.PRODUCER_REPORT_NAME).read_text(encoding="utf-8")
    )
    if (
        report.get("experiment"),
        report.get("classification"),
        report.get("pass"),
        report.get("accepted_update_21_snapshot_authorized"),
        report.get("producer_report_sha256_before"),
    ) != (
        snapshot_audit.EXPERIMENT,
        "accepted_update_21_transaction_snapshot_qualified",
        True,
        True,
        EXPECTED_SNAPSHOT_PRODUCER_REPORT_SHA256,
    ):
        raise SystemExit("snapshot qualification report does not authorize continuation")
    if (
        producer.get("experiment"),
        producer.get("classification"),
        producer.get("pass"),
        producer.get("selected_parameter_sha256"),
        producer.get("snapshot", {}).get("file_sha256"),
    ) != (
        snapshot_audit.EXPERIMENT,
        "accepted_update_21_transaction_snapshot_produced",
        True,
        EXPECTED_SNAPSHOT_PARAMETER_SHA256,
        EXPECTED_SNAPSHOT_ARCHIVE_SHA256,
    ):
        raise SystemExit("snapshot producer report identity does not match the protocol")
    archive_path = args.snapshot_dir / snapshot_audit.ARCHIVE_NAME
    snapshot = torch.load(archive_path, map_location="cpu", weights_only=True)
    semantic_actual = snapshot_audit.snapshot_semantic_hashes(snapshot)
    if (
        snapshot.get("experiment"),
        snapshot.get("protocol_commit"),
        snapshot.get("accepted_update"),
        semantic_actual,
        audit.semantic_sha256(snapshot["candidate_parameters"]),
        audit.semantic_sha256(snapshot["pending_optimizer_state"]),
        snapshot.get("development_transfer_decision", {}).get("pass"),
    ) != (
        snapshot_audit.EXPERIMENT,
        snapshot_audit.PROTOCOL_COMMIT,
        21,
        snapshot.get("tensor_semantic_sha256"),
        EXPECTED_SNAPSHOT_PARAMETER_SHA256,
        EXPECTED_PENDING_OPTIMIZER_SHA256,
        True,
    ):
        raise SystemExit("accepted update-21 transaction snapshot failed authorization")
    _SNAPSHOT_CPU = snapshot
    _SNAPSHOT_REFERENCE = {
        "source_training_metrics": copy.deepcopy(snapshot["source_training_metrics"]),
        "selected_training_metrics": copy.deepcopy(snapshot["selected_training_metrics"]),
    }


def main() -> int:
    global _SNAPSHOT_CPU, _SNAPSHOT_REFERENCE, _SNAPSHOT_LOAD_CONTROL

    args = parse_args()
    validate_args(args)
    locked_before = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    globals_before = {
        "snapshot_cpu": _SNAPSHOT_CPU,
        "snapshot_reference": _SNAPSHOT_REFERENCE,
        "snapshot_load_control": _SNAPSHOT_LOAD_CONTROL,
        "continuation_update": continuation._UPDATE_NUMBER,
        "continuation_archive": continuation._FROZEN_ARCHIVE_CPU,
        "continuation_step_report": continuation._STEP_AUDIT_REPORT,
        "corrected_guard": corrected._GUARD_REPORT,
        "corrected_update": corrected._UPDATE_NUMBER,
        "corrected_optimizer": corrected._PENDING_OPTIMIZER,
        "corrected_optimizer_before": corrected._OPTIMIZER_BEFORE_PROPOSAL,
        "corrected_optimizer_after": corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256,
        "corrected_capture": corrected._REPAIR_TENSOR_CAPTURE,
        "canonical_base": canonical._AUTHORITATIVE_BASE,
        "canonical_candidate": canonical._AUTHORITATIVE_CANDIDATE,
    }
    run_snapshot_load_control: dict[str, Any] | None = None
    try:
        validate_authorizations(args)
        accepted_updates = seed_resume(args)
        _SNAPSHOT_CPU = None
        continuation._UPDATE_NUMBER = accepted_updates
        continuation._FROZEN_ARCHIVE_CPU = None
        continuation._STEP_AUDIT_REPORT = None
        corrected._UPDATE_NUMBER = accepted_updates
        corrected._REPAIR_TENSOR_CAPTURE = None
        overrides = {
            "EXPERIMENT": EXPERIMENT,
            "PROTOCOL_COMMIT": PROTOCOL_COMMIT,
            "MAXIMUM_ACCEPTED_UPDATES": MAXIMUM_ACCEPTED_UPDATES,
            "DEVELOPMENT_PRESERVATION_REQUIRED": True,
            "DEVELOPMENT_FAILURE_ROLLBACK_REQUIRED": True,
            "PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED": True,
            "parse_args": lambda: args,
            "validate_args": validate_args,
            "protocol_manifest": protocol_manifest,
            "make_projected_proposal": make_projected_proposal,
            "install_trial": canonical.install_trial,
            "find_safe_trial": continuation.find_safe_trial,
        }
        for name, value in overrides.items():
            setattr(base, name, value)
        result = base.main()
    finally:
        run_snapshot_load_control = _SNAPSHOT_LOAD_CONTROL
        for name, value in _ORIGINAL_BASE_VALUES.items():
            setattr(base, name, value)
        _SNAPSHOT_CPU = globals_before["snapshot_cpu"]
        _SNAPSHOT_REFERENCE = globals_before["snapshot_reference"]
        _SNAPSHOT_LOAD_CONTROL = globals_before["snapshot_load_control"]
        continuation._UPDATE_NUMBER = globals_before["continuation_update"]
        continuation._FROZEN_ARCHIVE_CPU = globals_before["continuation_archive"]
        continuation._STEP_AUDIT_REPORT = globals_before["continuation_step_report"]
        corrected._GUARD_REPORT = globals_before["corrected_guard"]
        corrected._UPDATE_NUMBER = globals_before["corrected_update"]
        corrected._PENDING_OPTIMIZER = globals_before["corrected_optimizer"]
        corrected._OPTIMIZER_BEFORE_PROPOSAL = globals_before["corrected_optimizer_before"]
        corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = globals_before["corrected_optimizer_after"]
        corrected._REPAIR_TENSOR_CAPTURE = globals_before["corrected_capture"]
        canonical._AUTHORITATIVE_BASE = globals_before["canonical_base"]
        canonical._AUTHORITATIVE_CANDIDATE = globals_before["canonical_candidate"]
    locked_after = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    if locked_after != locked_before:
        raise RuntimeError("one or more locked source files changed during fitting")
    report_path = args.output_dir / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["snapshot_seed_provenance"] = {
            "locked_hashes_before": locked_before,
            "locked_hashes_after": locked_after,
            "all_locked_inputs_unchanged": True,
            "seeded_from_accepted_update": EXPECTED_INITIAL_ACCEPTED_UPDATES,
            "candidate_parameter_sha256": EXPECTED_SNAPSHOT_PARAMETER_SHA256,
            "pending_optimizer_sha256": EXPECTED_PENDING_OPTIMIZER_SHA256,
            "snapshot_load_control": run_snapshot_load_control,
            "phase_maximum_total_update": MAXIMUM_ACCEPTED_UPDATES,
        }
        snapshot_audit.write_json(report_path, report)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
