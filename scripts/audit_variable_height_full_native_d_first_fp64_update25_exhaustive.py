#!/usr/bin/env python3
"""Produce and verify a frozen update-25 exhaustive-primary projection audit."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize, nnls
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_fp64_frozen_input as frozen  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_projection as fp64  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402
import train_variable_height_full_native_d_first_fp64_corrected as continuation  # noqa: E402
import train_variable_height_full_native_d_first_fp64_snapshot_corrected as source_fit  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-update25-exhaustive-audit-v1"
PROTOCOL_COMMIT = "18a237b"
EXPECTED_GRAPH_SHA256 = "8c6ba28d149e9ac4a5223c5919657a1734f2c2cac9828c51114fd0174a383665"
EXPECTED_CHECKPOINT_SHA256 = "7238b0e3ca39dc1a8bfd35dcf9f8b6fc9e8c8881ff989f64cb135fa6fed4e572"
EXPECTED_SOURCE_REPORT_SHA256 = "55aa8e0848d2b93b5493c2c6478b93f30485bd810520f0814b126a918aec1691"
EXPECTED_SOURCE_RESUME_SHA256 = "5a500afb19910b0fb19186d575b0f75455d4b0f72b8f4e592663693697dc34b0"
EXPECTED_SOURCE_PARAMETER_SHA256 = (
    "299dae786c06b5a64b9d9d29a9e365697ea5a36333caac5eb7c1b3ea2ffbaeed"
)
EXPECTED_SOURCE_OPTIMIZER_SHA256 = (
    "5213d7ee914c97e402bc1108b6c4fcc7be385b952388981865d388ba3fc8574f"
)
EXPECTED_ACCEPTED_UPDATES = 24
RECONSTRUCTED_UPDATE = 25
REPEATS = 3
ARCHIVE_NAME = "frozen-update25-inputs.pt"
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
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-snapshot-corrected-fitting-001"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-update25-exhaustive-audit-001"
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
    }


def expected_locked_hashes() -> dict[str, str]:
    return {
        "graph": EXPECTED_GRAPH_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "cache_manifest": base.EXPECTED_SOURCE_CACHE_MANIFEST_SHA256,
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume": EXPECTED_SOURCE_RESUME_SHA256,
    }


def validate_args(args: argparse.Namespace) -> None:
    missing = [path for path in locked_input_paths(args).values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input: {missing[0]}")
    if args.smoke_test:
        raise SystemExit("this preregistered exhaustive-primary audit has no smoke variant")
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
        raise SystemExit("the verifier requires a completed producer report")
    if args.phase == "verify" and any(path.exists() for path in (verifier_started, final_report)):
        raise SystemExit("the verifier already started; this audit cannot be retried")
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
        "source_total_accepted_updates": EXPECTED_ACCEPTED_UPDATES,
        "reconstructed_update": RECONSTRUCTED_UPDATE,
        "actor_contract_unchanged": True,
        "producer": {
            "source_controller_and_adam_exact": True,
            "source_and_update_24_training_replay_required": True,
            "constraint_specs_rows_d_gradient_and_raw_adam_step_generated_once": True,
            "archive_persisted_before_any_projection": True,
            "source_run_untouched": True,
        },
        "primary_solver": {
            "name": "exhaustive active-support Lawson-Hanson nonnegative least squares",
            "all_supports_enumerated": True,
            "maximum_supports_per_round": 2**10,
            "minimum_objective_full_kkt_solution": True,
            "support_mask_tie_break": "ascending integer mask",
            "used_for_candidate": True,
            "normalized_kkt_residual_maximum": fp64.NORMALIZED_KKT_RESIDUAL_LIMIT,
            "original_unit_primal_violation_maximum": base.LINEAR_CONSTRAINT_TOLERANCE,
        },
        "diagnostic_solvers": {
            "slsqp_independent_agreement_required": True,
            "slsqp_gram_primal_relative_distance_maximum": (
                fp64.INDEPENDENT_PRIMAL_RELATIVE_DISTANCE_LIMIT
            ),
            "slsqp_objective_relative_difference_maximum": (
                fp64.INDEPENDENT_OBJECTIVE_RELATIVE_DIFFERENCE_LIMIT
            ),
            "lbfgsb_status_is_diagnostic_not_required": True,
            "historical_lbfgsb_status_reproduction_required": False,
        },
        "bound_aware_projection": {
            "monotonic_actual_boundary_active_set_unchanged": True,
            "maximum_rounds": fp64.MAXIMUM_ACTIVE_SET_ROUNDS,
            "native_bounds_required": True,
            "canonical_idempotence_required": True,
            "post_materialization_linear_violation_maximum": (base.LINEAR_CONSTRAINT_TOLERANCE),
            "negative_endpoint_damping_derivative_required": True,
            "finite_difference_direction_check_unchanged": True,
        },
        "repeats_from_cpu_archive": REPEATS,
        "repeat_primal_relative_distance_maximum": (frozen.PRIMAL_REPEAT_RELATIVE_DISTANCE_LIMIT),
        "repeat_objective_relative_difference_maximum": (
            frozen.OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT
        ),
        "round_active_and_fixed_set_hashes_exact": True,
        "ordinary_training_diagnostic": {
            "scales_descending": list(base.BACKTRACK_SCALES),
            "unchanged_candidate_decision": True,
            "repair_invoked": False,
            "development_or_fresh_data": False,
            "part_of_numerical_pass": False,
        },
        "candidate_or_optimizer_retained": False,
        "closed_loop_hover": False,
        "promotion": False,
        "pass_authorizes": (
            "only a separately preregistered continuation from exact accepted update 24"
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


def to_device_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device_tree(item, device) for item in value)
    return copy.deepcopy(value)


def archive_semantic_hashes(archive: dict[str, Any]) -> dict[str, str]:
    return {
        key: audit.semantic_sha256(archive[key])
        for key in (
            "current_parameters",
            "optimizer_before",
            "optimizer_after",
            "raw_displacement",
            "constraint_rows",
            "constraint_specs",
            "raw_damping_gradient",
        )
    }


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
    producer_report = args.output_dir / PRODUCER_REPORT_NAME
    archive = args.output_dir / ARCHIVE_NAME
    interrupted_phase = "verifier" if verifier_started.is_file() else "producer"
    report = {
        "experiment": EXPERIMENT,
        "status": "restored_numerical_audit_no_retained_candidate",
        "pass": False,
        "classification": f"{interrupted_phase}_interrupted_after_phase_start",
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
            responsibility.file_sha256(producer_report) if producer_report.is_file() else None
        ),
        "diagnostic_archive_sha256": (
            responsibility.file_sha256(archive) if archive.is_file() else None
        ),
        "phase_retried": False,
        "continuation_authorized": False,
        "candidate_retained": False,
        "development_or_fresh_data_used": False,
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
                "classification": report["classification"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def make_optimizer(controller: ConnectomeController) -> torch.optim.Optimizer:
    return continuation.fp64.make_optimizer(controller)


def load_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    hashes = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    if hashes != expected_locked_hashes():
        raise SystemExit("one or more hash-locked inputs do not match the protocol")
    report = json.loads((args.source_fit_dir / "report.json").read_text(encoding="utf-8"))
    resume = torch.load(args.source_fit_dir / "resume.pt", map_location=device, weights_only=True)
    last = report.get("history", [{}])[-1]
    identity = (
        report.get("experiment"),
        report.get("protocol", {}).get("protocol_commit"),
        report.get("accepted_updates"),
        report.get("stop_reason"),
        last.get("update"),
        last.get("accepted"),
        last.get("projection", {}).get("reason"),
        resume.get("experiment"),
        resume.get("protocol_commit"),
        int(resume.get("accepted_updates", -1)),
        resume.get("run_state"),
    )
    expected = (
        source_fit.EXPERIMENT,
        source_fit.PROTOCOL_COMMIT,
        EXPECTED_ACCEPTED_UPDATES,
        "constraint projection failed",
        RECONSTRUCTED_UPDATE,
        False,
        "free-coordinate inequality solve failed",
        source_fit.EXPERIMENT,
        source_fit.PROTOCOL_COMMIT,
        EXPECTED_ACCEPTED_UPDATES,
        "stopped",
    )
    if identity != expected:
        raise SystemExit("stopped update-24 source identity does not match the protocol")
    audit_history = copy.deepcopy(resume.get("history", []))
    if (
        not audit_history
        or int(audit_history[-1].get("update", -1)) != RECONSTRUCTED_UPDATE
        or audit_history[-1].get("accepted") is not False
    ):
        raise SystemExit("source resume lacks the unique rejected update-25 history entry")
    audit_history.pop()
    if (
        not audit_history
        or int(audit_history[-1].get("update", -1)) != EXPECTED_ACCEPTED_UPDATES
        or audit_history[-1].get("accepted") is not True
    ):
        raise SystemExit("audit history copy does not end at accepted update 24")
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded.get("graph_sha256") != EXPECTED_GRAPH_SHA256:
        raise SystemExit("source checkpoint graph identity does not match the protocol")
    source = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    student = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    student.load_state_dict(resume["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    optimizer = make_optimizer(student)
    optimizer.load_state_dict(resume["optimizer"])
    current_parameters = joint._copy_parameters(student)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    parameter_sha256 = audit.semantic_sha256(current_parameters)
    optimizer_sha256 = audit.semantic_sha256(optimizer_before)
    if (
        parameter_sha256,
        optimizer_sha256,
    ) != (EXPECTED_SOURCE_PARAMETER_SHA256, EXPECTED_SOURCE_OPTIMIZER_SHA256):
        raise SystemExit("accepted update-24 controller or Adam state does not match")

    config = HoverConfig()
    train_factorial, train_attitude, _, _, cache_integrity = endpoint.load_or_create_caches(
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
        raise SystemExit("authorized immutable cache failed integrity")
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
    current_training, _ = endpoint.evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    reproductions = {
        "source_training": audit.numeric_tree_comparison(
            source_training, report["source_training"]
        ),
        "update_24_training": audit.numeric_tree_comparison(
            current_training, report["final_training"]
        ),
    }
    return {
        "hashes": hashes,
        "report": report,
        "resume": resume,
        "audit_history": audit_history,
        "source": source,
        "student": student,
        "optimizer": optimizer,
        "current_parameters": current_parameters,
        "optimizer_before": optimizer_before,
        "parameter_sha256": parameter_sha256,
        "optimizer_sha256": optimizer_sha256,
        "train_factorial": train_factorial,
        "train_attitude": train_attitude,
        "cache_integrity": cache_integrity,
        "endpoint_scale": endpoint_scale,
        "scales": scales,
        "source_training": source_training,
        "current_training": current_training,
        "reproductions": reproductions,
    }


def restoration_report(context: dict[str, Any]) -> dict[str, Any]:
    student = context["student"]
    optimizer = context["optimizer"]
    parameter_pass = all(
        torch.equal(getattr(student, name).detach(), context["current_parameters"][name])
        for name in joint.PARAMETER_FAMILIES
    )
    optimizer_pass = audit.trees_equal(optimizer.state_dict(), context["optimizer_before"])
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
    input_controls_pass = bool(
        all(item["pass"] for item in context["reproductions"].values())
        and context["cache_integrity"]["all_persisted_file_and_tensor_hashes_match"]
    )
    archive_path = args.output_dir / ARCHIVE_NAME
    archive_sha256 = None
    tensor_hashes = None
    optimizer_transaction: dict[str, Any] | None = None
    archive_reload: dict[str, Any] = {"pass": False}
    if input_controls_pass:
        print(json.dumps({"stage": "generating_update_25_inputs_once"}), flush=True)
        student = context["student"]
        optimizer = context["optimizer"]
        with audit.transactional_restoration(student, optimizer):
            specs = base.constraint_specs(context["source_training"], context["current_training"])
            rows = base.constraint_gradient_rows(
                student,
                context["train_factorial"],
                context["train_attitude"],
                context["scales"],
                specs,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            endpoint.accumulated_endpoint_damping_gradient(
                student,
                context["train_factorial"],
                scale=context["endpoint_scale"],
                prefix_steps=joint.PREFIX_STEPS,
                device=device,
            )
            raw_gradient = {
                name: getattr(student, name).grad.detach().clone()
                for name in joint.PARAMETER_FAMILIES
            }
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
            )
            optimizer.step()
            student.project_parameters()
            raw_parameters = joint._copy_parameters(student)
            raw_displacement = {
                name: raw_parameters[name] - context["current_parameters"][name]
                for name in joint.PARAMETER_FAMILIES
            }
            optimizer_after = copy.deepcopy(optimizer.state_dict())
            optimizer_transaction = corrected.optimizer_step_transaction(
                context["optimizer_before"], optimizer_after
            )
            archive = to_cpu_tree(
                {
                    "experiment": EXPERIMENT,
                    "protocol_commit": PROTOCOL_COMMIT,
                    "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
                    "source_resume_sha256": EXPECTED_SOURCE_RESUME_SHA256,
                    "accepted_updates_before": EXPECTED_ACCEPTED_UPDATES,
                    "reconstructed_update": RECONSTRUCTED_UPDATE,
                    "current_parameters": context["current_parameters"],
                    "optimizer_before": context["optimizer_before"],
                    "optimizer_after": optimizer_after,
                    "raw_displacement": raw_displacement,
                    "constraint_rows": rows,
                    "constraint_specs": specs,
                    "raw_damping_gradient": raw_gradient,
                    "gradient_norm_before_clipping": gradient_norm,
                    "source_training_metrics": context["source_training"],
                    "current_training_metrics": context["current_training"],
                }
            )
            tensor_hashes = archive_semantic_hashes(archive)
            archive["tensor_semantic_sha256"] = tensor_hashes
            base.atomic_torch_save(archive, archive_path)
            archive_sha256 = responsibility.file_sha256(archive_path)
        reloaded = torch.load(archive_path, map_location="cpu", weights_only=True)
        archive_reload = {
            "pass": bool(
                reloaded.get("experiment") == EXPERIMENT
                and reloaded.get("protocol_commit") == PROTOCOL_COMMIT
                and archive_semantic_hashes(reloaded) == tensor_hashes
            ),
            "file_sha256": archive_sha256,
            "tensor_semantic_sha256": archive_semantic_hashes(reloaded),
            "persisted_before_any_projection": True,
        }
    restoration = restoration_report(context)
    hashes_after = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    sources_unchanged = hashes_after == context["hashes"]
    passed = bool(
        input_controls_pass
        and optimizer_transaction
        and optimizer_transaction["pass"]
        and archive_reload["pass"]
        and restoration["pass"]
        and sources_unchanged
    )
    report = {
        "experiment": EXPERIMENT,
        "status": "frozen_update_25_input_producer",
        "pass": passed,
        "classification": (
            "frozen_update_25_inputs_produced" if passed else "producer_control_failure"
        ),
        "protocol": protocol_manifest(),
        "source_hashes_before": context["hashes"],
        "source_hashes_after": hashes_after,
        "all_source_files_unchanged": sources_unchanged,
        "source_parameter_sha256": context["parameter_sha256"],
        "source_optimizer_sha256": context["optimizer_sha256"],
        "audit_copy": {
            "history_ends_at_accepted_update": context["audit_history"][-1]["update"],
            "rejected_update_25_entry_discarded": True,
            "source_resume_modified": False,
        },
        "training_reproductions": context["reproductions"],
        "input_controls_pass": input_controls_pass,
        "optimizer_one_step_transaction": optimizer_transaction,
        "archive": {
            "path": responsibility.stable_path(archive_path),
            **archive_reload,
        },
        "restoration": restoration,
        "projection_run": False,
        "candidate_retained": False,
        "development_or_fresh_data_used": False,
        "closed_loop_hover_run": False,
        "promoted": False,
        "elapsed_seconds": perf_counter() - started,
    }
    output = args.output_dir / PRODUCER_REPORT_NAME
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": passed,
                "classification": report["classification"],
                "archive_sha256": archive_sha256,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _dual_objective(gram: np.ndarray, violation: np.ndarray, dual: np.ndarray) -> float:
    return float(0.5 * dual @ gram @ dual - violation @ dual)


def exhaustive_primary_dual(
    normalized_gram: np.ndarray,
    normalized_violation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    symmetric = 0.5 * (normalized_gram + normalized_gram.T)
    dimension = len(normalized_violation)
    candidates: list[dict[str, Any]] = []
    failed_supports = 0
    for support_mask in range(1 << dimension):
        support = np.array(
            [index for index in range(dimension) if support_mask & (1 << index)],
            dtype=np.int64,
        )
        dual = np.zeros(dimension, dtype=np.float64)
        try:
            if support.size:
                support_dual, stationarity = nnls(
                    symmetric[np.ix_(support, support)],
                    normalized_violation[support],
                    maxiter=fp64.MAXIMUM_SOLVER_ITERATIONS,
                )
                dual[support] = support_dual
            else:
                stationarity = float(np.linalg.norm(normalized_violation))
        except (RuntimeError, ValueError):
            failed_supports += 1
            continue
        kkt = fp64._projected_gradient_report(symmetric, normalized_violation, dual)
        if np.all(np.isfinite(dual)) and kkt["pass"]:
            candidates.append(
                {
                    "dual": dual,
                    "objective": _dual_objective(symmetric, normalized_violation, dual),
                    "kkt": kkt,
                    "support_mask": support_mask,
                    "support_size": int(support.size),
                    "stationarity_residual_norm": float(stationarity),
                }
            )
    selected = min(
        candidates,
        key=lambda candidate: (candidate["objective"], candidate["support_mask"]),
        default=None,
    )
    if selected is None:
        dual = np.zeros(dimension, dtype=np.float64)
        return dual, {
            "pass": False,
            "reason": "no finite full-KKT-feasible support",
            "supports_examined": 1 << dimension,
            "failed_supports": failed_supports,
            "kkt_feasible_supports": 0,
            "used_for_candidate": True,
        }
    return selected["dual"], {
        "pass": True,
        "solver": "exhaustive active-support Lawson-Hanson nonnegative least squares",
        "supports_examined": 1 << dimension,
        "failed_supports": failed_supports,
        "kkt_feasible_supports": len(candidates),
        "selected_support_mask": selected["support_mask"],
        "selected_support_size": selected["support_size"],
        "selected_stationarity_residual_norm": selected["stationarity_residual_norm"],
        "dual_objective": selected["objective"],
        "normalized_kkt": selected["kkt"],
        "dual_coefficients": selected["dual"].tolist(),
        "used_for_candidate": True,
    }


def _diagnostic_solver(
    method: str,
    normalized_gram: np.ndarray,
    normalized_violation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    initial = np.zeros(len(normalized_violation), dtype=np.float64)

    def objective(value: np.ndarray) -> float:
        return _dual_objective(normalized_gram, normalized_violation, value)

    def gradient(value: np.ndarray) -> np.ndarray:
        return normalized_gram @ value - normalized_violation

    options = {"maxiter": fp64.MAXIMUM_SOLVER_ITERATIONS}
    if method == "SLSQP":
        options["ftol"] = fp64.INDEPENDENT_SLSQP_FTOL
    else:
        options.update({"ftol": 1.0e-15, "gtol": base.DUAL_GRADIENT_TOLERANCE})
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method=method,
        bounds=[(0.0, None)] * len(initial),
        options=options,
    )
    kkt = fp64._projected_gradient_report(normalized_gram, normalized_violation, result.x)
    return result.x, {
        "solver": method,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "dual_objective": float(result.fun),
        "normalized_kkt": kkt,
        "used_for_candidate": False,
    }


def exhaustive_free_projection(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    rows64 = [{name: value.detach().double() for name, value in row.items()} for row in rows]
    projected = {name: value.detach().double().clone() for name, value in displacement.items()}
    count = len(rows64)
    gram = torch.empty(
        count,
        count,
        dtype=torch.float64,
        device=next(iter(rows64[0].values())).device,
    )
    j_displacement = torch.empty(count, dtype=torch.float64, device=gram.device)
    for row_index, row in enumerate(rows64):
        j_displacement[row_index] = sum(
            (row[name] * projected[name]).sum() for name in joint.PARAMETER_FAMILIES
        )
        for column_index in range(row_index + 1):
            other = rows64[column_index]
            value = sum(
                (base.PARAMETER_LEARNING_RATES[name] ** 2) * (row[name] * other[name]).sum()
                for name in joint.PARAMETER_FAMILIES
            )
            gram[row_index, column_index] = value
            gram[column_index, row_index] = value
    residual = torch.tensor(
        [spec["current_mse"] - spec["limit_mse"] for spec in specs],
        dtype=torch.float64,
        device=gram.device,
    )
    violation_before = residual + j_displacement
    row_norm = gram.diagonal().clamp_min(0.0).sqrt()
    active = row_norm > 1.0e-15
    infeasible_zero = (~active) & (violation_before > base.LINEAR_CONSTRAINT_TOLERANCE)
    if bool(infeasible_zero.any()):
        return projected, {
            "pass": False,
            "reason": "a violated constraint has a numerically zero Jacobian row",
            "active_rows": int(active.sum()),
        }
    active_indices = active.nonzero(as_tuple=False)[:, 0]
    normalized_gram = (
        (gram[active][:, active] / (row_norm[active, None] * row_norm[None, active])).cpu().numpy()
    )
    normalized_violation = (violation_before[active] / row_norm[active]).cpu().numpy()
    primary_dual, primary = exhaustive_primary_dual(normalized_gram, normalized_violation)
    slsqp_dual, slsqp = _diagnostic_solver("SLSQP", normalized_gram, normalized_violation)
    _, lbfgsb = _diagnostic_solver("L-BFGS-B", normalized_gram, normalized_violation)
    coefficients = torch.zeros_like(violation_before)
    coefficients[active_indices] = torch.from_numpy(primary_dual).to(gram.device) / row_norm[active]
    for coefficient, row in zip(coefficients, rows64, strict=True):
        for name in joint.PARAMETER_FAMILIES:
            projected[name].add_(
                row[name],
                alpha=-(base.PARAMETER_LEARNING_RATES[name] ** 2) * float(coefficient),
            )
    linearized_after = residual.clone()
    for row_index, row in enumerate(rows64):
        linearized_after[row_index] += sum(
            (row[name] * projected[name]).sum() for name in joint.PARAMETER_FAMILIES
        )
    maximum = float(linearized_after.max())
    primal_pass = math.isfinite(maximum) and maximum <= base.LINEAR_CONSTRAINT_TOLERANCE
    delta = slsqp_dual - primary_dual
    primary_norm_squared = max(0.0, float(primary_dual @ normalized_gram @ primary_dual))
    distance_squared = max(0.0, float(delta @ normalized_gram @ delta))
    relative_distance = math.sqrt(distance_squared) / max(math.sqrt(primary_norm_squared), 1.0e-30)
    primary_objective = _dual_objective(normalized_gram, normalized_violation, primary_dual)
    slsqp_objective = _dual_objective(normalized_gram, normalized_violation, slsqp_dual)
    objective_difference = abs(slsqp_objective - primary_objective) / max(
        1.0, abs(primary_objective)
    )
    independent = {
        **slsqp,
        "gram_induced_primal_relative_distance_from_primary": relative_distance,
        "gram_induced_primal_relative_distance_limit": (
            fp64.INDEPENDENT_PRIMAL_RELATIVE_DISTANCE_LIMIT
        ),
        "dual_objective_relative_difference": objective_difference,
        "dual_objective_relative_difference_limit": (
            fp64.INDEPENDENT_OBJECTIVE_RELATIVE_DIFFERENCE_LIMIT
        ),
        "pass": bool(
            slsqp["solver_success"]
            and slsqp["normalized_kkt"]["pass"]
            and relative_distance <= fp64.INDEPENDENT_PRIMAL_RELATIVE_DISTANCE_LIMIT
            and objective_difference <= fp64.INDEPENDENT_OBJECTIVE_RELATIVE_DIFFERENCE_LIMIT
        ),
    }
    report = {
        **primary,
        "pass": bool(primary["pass"] and primal_pass and independent["pass"]),
        "constraint_names": [spec["name"] for spec in specs],
        "linearized_violation_before": violation_before.tolist(),
        "linearized_violation_after": linearized_after.tolist(),
        "maximum_linearized_violation_after": maximum,
        "original_unit_primal_violation_pass": primal_pass,
        "original_unit_primal_violation_limit": base.LINEAR_CONSTRAINT_TOLERANCE,
        "slsqp_independent": independent,
        "lbfgsb_diagnostic": lbfgsb,
        "arithmetic_dtype": "float64",
    }
    return projected, report


def exhaustive_bound_aware_projection(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    specs: list[dict[str, Any]],
    current_edges: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    displacement64 = {name: value.detach().double().clone() for name, value in displacement.items()}
    rows64 = [
        {name: value.detach().double().clone() for name, value in row.items()} for row in rows
    ]
    current_edges64 = current_edges.detach().double()
    fixed = torch.zeros_like(current_edges, dtype=torch.bool)
    fixed_displacement = torch.zeros_like(current_edges64)
    final = {name: value.clone() for name, value in displacement64.items()}
    rounds = []
    converged = False
    for round_index in range(1, fp64.MAXIMUM_ACTIVE_SET_ROUNDS + 1):
        working = {name: value.clone() for name, value in displacement64.items()}
        working["edge_magnitude"][fixed] = fixed_displacement[fixed]
        free_rows = []
        adjusted_specs = []
        for row, spec in zip(rows64, specs, strict=True):
            free_row = {name: value.clone() for name, value in row.items()}
            fixed_contribution = float(
                (free_row["edge_magnitude"][fixed] * fixed_displacement[fixed]).sum()
            )
            free_row["edge_magnitude"][fixed] = 0.0
            adjusted = dict(spec)
            adjusted["current_mse"] = spec["current_mse"] + fixed_contribution
            adjusted["remaining_mse_allowance"] = adjusted["limit_mse"] - adjusted["current_mse"]
            free_rows.append(free_row)
            adjusted_specs.append(adjusted)
        final, projection = exhaustive_free_projection(working, free_rows, adjusted_specs)
        candidate_edges = current_edges64 + final["edge_magnitude"]
        below = candidate_edges < 0.0
        above = candidate_edges > 8.0
        violations = below | above
        newly_fixed = violations & ~fixed
        rounds.append(
            {
                "round": round_index,
                "fixed_edges_before": int(fixed.sum()),
                "fixed_set_sha256_before": audit.semantic_sha256(fixed),
                "newly_fixed_edges": int(newly_fixed.sum()),
                "newly_fixed_set_sha256": audit.semantic_sha256(newly_fixed),
                "free_projection": projection,
            }
        )
        if not projection["pass"]:
            return final, {
                "pass": False,
                "reason": "free-coordinate exhaustive-primary solve failed",
                "rounds": rounds,
                "fixed_edges": int(fixed.sum()),
                "converged_without_bound_violation": False,
                "arithmetic_dtype": "float64",
            }
        if not bool(violations.any()):
            converged = True
            break
        if not bool(newly_fixed.any()):
            return final, {
                "pass": False,
                "reason": "active set made no progress",
                "rounds": rounds,
                "fixed_edges": int(fixed.sum()),
                "converged_without_bound_violation": False,
                "arithmetic_dtype": "float64",
            }
        fixed |= newly_fixed
        fixed_displacement[below] = -current_edges64[below]
        fixed_displacement[above] = 8.0 - current_edges64[above]
    final_edges = current_edges64 + final["edge_magnitude"]
    maximum_box_violation = max(
        float((-final_edges).clamp_min(0.0).max()),
        float((final_edges - 8.0).clamp_min(0.0).max()),
    )
    return final, {
        "pass": bool(converged and maximum_box_violation == 0.0),
        "reason": None if converged else "active-set round limit reached",
        "rounds": rounds,
        "fixed_edges": int(fixed.sum()),
        "final_fixed_set_sha256": audit.semantic_sha256(fixed),
        "converged_without_bound_violation": converged,
        "maximum_box_violation": maximum_box_violation,
        "arithmetic_dtype": "float64",
    }


def round_path_signature(projection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "round": item["round"],
            "fixed_edges_before": item["fixed_edges_before"],
            "fixed_set_sha256_before": item["fixed_set_sha256_before"],
            "newly_fixed_edges": item["newly_fixed_edges"],
            "newly_fixed_set_sha256": item["newly_fixed_set_sha256"],
            "selected_support_mask": item["free_projection"].get("selected_support_mask"),
        }
        for item in projection["rounds"]
    ]


def post_projection_controls(
    context: dict[str, Any],
    frozen_input: dict[str, Any],
    proposed: dict[str, Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    student = context["student"]
    with audit.transactional_restoration(student, context["optimizer"]):
        authoritative, effective, idempotence = canonical.materialize_authoritative_candidate(
            student, frozen_input["current_parameters"], proposed
        )
        bounds = audit.parameter_bounds_report(authoritative)
        linearized = audit.linearized_constraint_violations(
            frozen_input["constraint_specs"],
            frozen_input["constraint_rows"],
            effective,
        )
        maximum_linearized, linearized_finite = audit.maximum_linearized_violation(linearized)
        derivative = canonical._dot_float64(frozen_input["raw_damping_gradient"], effective)
        _, actual_fd, fd_idempotence = fp64.install_fp64_trial(
            student,
            frozen_input["current_parameters"],
            authoritative,
            scale=joint.FINITE_DIFFERENCE_SCALE,
        )
        finite_difference_metrics, _ = endpoint.evaluate(
            student,
            context["train_factorial"],
            context["train_attitude"],
            context["scales"],
            endpoint_scale=context["endpoint_scale"],
            device=device,
        )
        fd_derivative = (
            canonical._dot_float64(frozen_input["raw_damping_gradient"], actual_fd)
            / joint.FINITE_DIFFERENCE_SCALE
        )
        fd_actual = (
            finite_difference_metrics["endpoint_damping_nrmse"] ** 2
            - context["current_training"]["endpoint_damping_nrmse"] ** 2
        ) / joint.FINITE_DIFFERENCE_SCALE
        fd_relative = abs(fd_actual - fd_derivative) / max(
            abs(fd_actual), abs(fd_derivative), 1.0e-12
        )
        finite_difference = {
            "pass": bool(
                math.isfinite(fd_derivative)
                and math.isfinite(fd_actual)
                and abs(fd_actual) >= audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
                and fd_derivative < 0.0
                and fd_actual < 0.0
                and fd_relative <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
                and fd_idempotence["pass"]
            ),
            "scale": joint.FINITE_DIFFERENCE_SCALE,
            "autograd_directional_derivative": fd_derivative,
            "complete_replay_finite_difference": fd_actual,
            "relative_error": fd_relative,
            "relative_error_limit": joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
            "canonical_parameter_idempotence": fd_idempotence,
        }
        optimizer_transaction = corrected.optimizer_step_transaction(
            frozen_input["optimizer_before"], frozen_input["optimizer_after"]
        )
        report = {
            "pass": bool(
                idempotence["pass"]
                and bounds["pass"]
                and linearized_finite
                and maximum_linearized <= base.LINEAR_CONSTRAINT_TOLERANCE
                and derivative < 0.0
                and finite_difference["pass"]
                and optimizer_transaction["pass"]
            ),
            "canonical_parameter_idempotence": idempotence,
            "parameter_bounds": bounds,
            "linearized_constraint_violations": linearized,
            "maximum_linearized_constraint_violation": maximum_linearized,
            "damping_directional_derivative": derivative,
            "finite_difference": finite_difference,
            "optimizer_one_step_transaction": optimizer_transaction,
        }
    return report, authoritative


def ordinary_scale_diagnostics(
    context: dict[str, Any],
    frozen_input: dict[str, Any],
    authoritative: dict[str, Tensor],
    *,
    device: torch.device,
) -> tuple[float | None, list[dict[str, Any]]]:
    student = context["student"]
    trials = []
    selected_scale = None
    with audit.transactional_restoration(student, context["optimizer"]):
        for scale in base.BACKTRACK_SCALES:
            parameters, displacement, idempotence = audit.install_authoritative_trial(
                student,
                frozen_input["current_parameters"],
                authoritative,
                scale=scale,
            )
            bounds = audit.parameter_bounds_report(parameters)
            derivative = base.damping_directional_derivative(
                frozen_input["raw_damping_gradient"], displacement
            )
            metrics, _ = endpoint.evaluate(
                student,
                context["train_factorial"],
                context["train_attitude"],
                context["scales"],
                endpoint_scale=context["endpoint_scale"],
                device=device,
            )
            decision = base.candidate_decision(
                context["source_training"],
                context["current_training"],
                metrics,
                damping_directional_derivative=derivative,
            )
            repair_eligibility = None
            if scale == corrected.REPAIR_PROPOSAL_SCALE:
                repair_eligibility = corrected.repair_eligibility(
                    context["source_training"],
                    context["current_training"],
                    metrics,
                    damping_directional_derivative=derivative,
                    projection_controls_pass=True,
                    parameter_bounds_pass=bounds["pass"],
                    canonicalization_pass=idempotence["pass"],
                    registered_reproduction_pass=True,
                )
            trials.append(
                {
                    "scale": scale,
                    "metrics": metrics,
                    "decision": decision,
                    "canonical_parameter_idempotence": idempotence,
                    "parameter_bounds": bounds,
                    "repair_eligibility_diagnostic": repair_eligibility,
                }
            )
            if decision["pass"] and selected_scale is None:
                selected_scale = scale
    return selected_scale, trials


def audit_classification(
    *,
    producer_identity_pass: bool,
    archive_identity_pass: bool,
    baseline_pass: bool,
    all_runs_pass: bool,
    repeat_agreement_pass: bool,
    round_paths_exact: bool,
    post_projection_pass: bool,
    restoration_pass: bool,
    archive_unchanged: bool,
    sources_unchanged: bool,
    ordinary_scale: float | None,
) -> str:
    if not producer_identity_pass:
        return "producer_control_failure"
    if not archive_identity_pass or not baseline_pass:
        return "verifier_input_control_failure"
    if not all_runs_pass:
        return "one_or_more_exhaustive_primary_runs_failed"
    if not repeat_agreement_pass or not round_paths_exact:
        return "exhaustive_primary_repeats_disagreed"
    if not post_projection_pass:
        return "exhaustive_primary_failed_post_projection_gates"
    if not restoration_pass or not archive_unchanged or not sources_unchanged:
        return "verifier_restoration_or_artifact_control_failure"
    if ordinary_scale is None:
        return "projector_qualified_but_no_ordinary_scale_passed"
    return "projector_qualified_with_ordinary_scale_passed"


def all_post_projection_runs_pass(runs: list[dict[str, Any]]) -> bool:
    return bool(len(runs) == REPEATS and all(item.get("pass") is True for item in runs))


def verify(args: argparse.Namespace, device: torch.device) -> int:
    started = perf_counter()
    producer_path = args.output_dir / PRODUCER_REPORT_NAME
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    archive_path = args.output_dir / ARCHIVE_NAME
    producer_identity_pass = bool(
        producer.get("experiment") == EXPERIMENT
        and producer.get("classification") == "frozen_update_25_inputs_produced"
        and producer.get("pass") is True
        and archive_path.is_file()
        and producer.get("archive", {}).get("file_sha256")
        == responsibility.file_sha256(archive_path)
    )
    context = load_context(args, device)
    archive_sha_before = (
        responsibility.file_sha256(archive_path) if archive_path.is_file() else None
    )
    frozen_cpu = (
        torch.load(archive_path, map_location="cpu", weights_only=True)
        if archive_path.is_file()
        else {}
    )
    semantic_actual = archive_semantic_hashes(frozen_cpu) if frozen_cpu else {}
    archive_identity_pass = bool(
        producer_identity_pass
        and frozen_cpu.get("experiment") == EXPERIMENT
        and frozen_cpu.get("protocol_commit") == PROTOCOL_COMMIT
        and frozen_cpu.get("source_report_sha256") == EXPECTED_SOURCE_REPORT_SHA256
        and frozen_cpu.get("source_resume_sha256") == EXPECTED_SOURCE_RESUME_SHA256
        and frozen_cpu.get("accepted_updates_before") == EXPECTED_ACCEPTED_UPDATES
        and frozen_cpu.get("reconstructed_update") == RECONSTRUCTED_UPDATE
        and semantic_actual == frozen_cpu.get("tensor_semantic_sha256")
        and audit.trees_equal(
            frozen_cpu.get("current_parameters"),
            to_cpu_tree(context["current_parameters"]),
        )
        and audit.trees_equal(
            frozen_cpu.get("optimizer_before"),
            to_cpu_tree(context["optimizer_before"]),
        )
    )
    baseline_pass = all(item["pass"] for item in context["reproductions"].values())
    controls_before_projection = bool(archive_identity_pass and baseline_pass)

    runs = []
    displacements: list[dict[str, Tensor]] = []
    objectives = []
    if controls_before_projection:
        print(json.dumps({"stage": "three_exhaustive_primary_reloads"}), flush=True)
        for repeat in range(1, REPEATS + 1):
            loaded = torch.load(archive_path, map_location="cpu", weights_only=True)
            frozen_input = to_device_tree(loaded, device)
            displacement, projection = exhaustive_bound_aware_projection(
                frozen_input["raw_displacement"],
                frozen_input["constraint_rows"],
                frozen_input["constraint_specs"],
                frozen_input["current_parameters"]["edge_magnitude"],
            )
            objective = frozen.complete_projection_primal_objective(
                displacement, frozen_input["raw_displacement"]
            )
            runs.append(
                {
                    "repeat": repeat,
                    "pass": projection["pass"],
                    "projection": projection,
                    "round_path_signature": round_path_signature(projection),
                    "final_displacement_sha256": audit.semantic_sha256(displacement),
                    "learning_rate_scaled_displacement_norm": (
                        frozen.learning_rate_scaled_norm(displacement)
                    ),
                    "complete_projection_primal_objective": objective,
                }
            )
            displacements.append(displacement)
            objectives.append(objective)
    all_runs_pass = bool(runs and all(run["pass"] for run in runs))
    repeat_agreement = (
        frozen.primary_repeat_agreement(displacements, objectives)
        if len(displacements) == REPEATS
        else {"pass": False, "reason": "projection repeats did not run"}
    )
    round_paths_exact = bool(
        runs
        and all(run["round_path_signature"] == runs[0]["round_path_signature"] for run in runs[1:])
    )
    post_projection_runs: list[dict[str, Any]] = []
    post_projection: dict[str, Any] = {"pass": False, "runs": post_projection_runs}
    ordinary_scale = None
    ordinary_trials: list[dict[str, Any]] = []
    if all_runs_pass and repeat_agreement["pass"] and round_paths_exact:
        print(json.dumps({"stage": "post_projection_and_training_diagnostics"}), flush=True)
        authoritative = None
        frozen_input = None
        for repeat, displacement in enumerate(displacements, start=1):
            repeat_input = to_device_tree(
                torch.load(archive_path, map_location="cpu", weights_only=True), device
            )
            controls, repeat_authoritative = post_projection_controls(
                context, repeat_input, displacement, device=device
            )
            post_projection_runs.append({"repeat": repeat, **controls})
            if repeat == 1:
                authoritative = repeat_authoritative
                frozen_input = repeat_input
        post_projection = {
            "pass": all_post_projection_runs_pass(post_projection_runs),
            "runs": post_projection_runs,
            "required_on_every_repeat": True,
        }
        if post_projection["pass"]:
            if authoritative is None or frozen_input is None:
                raise RuntimeError("passing repeats did not retain their first diagnostic")
            ordinary_scale, ordinary_trials = ordinary_scale_diagnostics(
                context, frozen_input, authoritative, device=device
            )

    restoration = restoration_report(context)
    archive_sha_after = responsibility.file_sha256(archive_path) if archive_path.is_file() else None
    hashes_after = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    sources_unchanged = hashes_after == context["hashes"]
    numerical_pass = bool(
        controls_before_projection
        and all_runs_pass
        and repeat_agreement["pass"]
        and round_paths_exact
        and post_projection["pass"]
        and restoration["pass"]
        and archive_sha_after == archive_sha_before
        and sources_unchanged
    )
    classification = audit_classification(
        producer_identity_pass=producer_identity_pass,
        archive_identity_pass=archive_identity_pass,
        baseline_pass=baseline_pass,
        all_runs_pass=all_runs_pass,
        repeat_agreement_pass=repeat_agreement["pass"],
        round_paths_exact=round_paths_exact,
        post_projection_pass=post_projection["pass"],
        restoration_pass=restoration["pass"],
        archive_unchanged=archive_sha_after == archive_sha_before,
        sources_unchanged=sources_unchanged,
        ordinary_scale=ordinary_scale,
    )
    report = {
        "experiment": EXPERIMENT,
        "status": "restored_numerical_audit_no_retained_candidate",
        "pass": numerical_pass,
        "classification": classification,
        "protocol": protocol_manifest(),
        "producer_report_sha256": responsibility.file_sha256(producer_path),
        "producer_identity_pass": producer_identity_pass,
        "source_hashes_before": context["hashes"],
        "source_hashes_after": hashes_after,
        "all_source_files_unchanged": sources_unchanged,
        "training_reproductions": context["reproductions"],
        "archive": {
            "path": responsibility.stable_path(archive_path),
            "file_sha256_before": archive_sha_before,
            "file_sha256_after": archive_sha_after,
            "unchanged": archive_sha_before == archive_sha_after,
            "identity_and_semantic_hashes_pass": archive_identity_pass,
            "tensor_semantic_sha256": semantic_actual,
        },
        "exhaustive_primary_runs": runs,
        "all_exhaustive_primary_runs_pass": all_runs_pass,
        "primary_repeat_agreement": repeat_agreement,
        "round_paths_exact": round_paths_exact,
        "post_projection": post_projection,
        "ordinary_training_diagnostic": {
            "selected_scale": ordinary_scale,
            "one_or_more_scales_passed": ordinary_scale is not None,
            "trials": ordinary_trials,
            "repair_invoked": False,
            "part_of_numerical_pass": False,
        },
        "restoration": restoration,
        "candidate_retained": False,
        "optimizer_step_retained": False,
        "development_or_fresh_data_used": False,
        "closed_loop_hover_run": False,
        "promoted": False,
        "exhaustive_primary_projection_implementation_authorized": numerical_pass,
        "continuation_authorized": numerical_pass,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass qualifies only the exhaustive-primary update-25 projection numerics "
            "and authorizes a separately registered continuation from accepted update 24."
        ),
    }
    output = args.output_dir / FINAL_REPORT_NAME
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": numerical_pass,
                "classification": classification,
                "all_runs_pass": all_runs_pass,
                "repeat_agreement_pass": repeat_agreement["pass"],
                "round_paths_exact": round_paths_exact,
                "post_projection_pass": post_projection["pass"],
                "ordinary_scale": ordinary_scale,
                "candidate_retained": False,
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
