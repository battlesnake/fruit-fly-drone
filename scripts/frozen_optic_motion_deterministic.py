"""Shared locks and helpers for deterministic frozen optic-motion confirmation."""

from __future__ import annotations

import copy
import json
import os
import platform
import secrets
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_frozen_optic_motion as base  # noqa: E402

EXPERIMENT = "frozen-t4t5-optic-motion-deterministic-confirmation-v2"
PROTOCOL_COMMIT = "800c762"
EXPECTED_V1_REPORT_SHA256 = (
    "fc4d503b0e91dae11f35f4602a4fc7a47eb1b000c4424694340ff740a180b6d3"
)
EXPECTED_V1_IMPLEMENTATION_SHA256 = (
    "b835020bcff7b3cfc20603a745c081ff805208f60c0b5a032a7c34194e1d299d"
)
EXPECTED_STIMULUS_SHA256 = (
    "b48c639212fb4664eb12bc4db52e9535d3288a1aafbc657d081008510f2d9325"
)
EXPECTED_TORCH_VERSION = "2.12.0.dev20260408+cu128"
EXPECTED_CUDA_VERSION = "12.8"
EXPECTED_CUDNN_VERSION = 92_000
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5080"
EXPECTED_GPU_CAPABILITY = [12, 0]
EXPECTED_DRIVER_VERSION = "616.56"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
REPLAY_PROCESSES = 3
BLOCK_CASES = 8

RESPONSE_TENSOR_NAMES = (
    "integrated_selected_stationary_subtracted",
    "terminal_selected_activity",
    "integrated_motion_type_means",
    "integrated_static_type_means",
    "terminal_motion_type_means",
    "terminal_static_type_means",
    "terminal_motor_outputs",
)


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "fresh_blind_validation": False,
        "purpose": "deterministic confirmation after the v1 CUDA duplicate-control stop",
        "locked_v1_protocol": copy.deepcopy(base.protocol_manifest()),
        "locked_v1_report_sha256": EXPECTED_V1_REPORT_SHA256,
        "locked_v1_implementation_sha256": EXPECTED_V1_IMPLEMENTATION_SHA256,
        "locked_rendered_stimulus_sha256": EXPECTED_STIMULUS_SHA256,
        "scientific_changes_from_v1": [],
        "runtime": {
            "torch": EXPECTED_TORCH_VERSION,
            "cuda": EXPECTED_CUDA_VERSION,
            "cudnn": EXPECTED_CUDNN_VERSION,
            "gpu": EXPECTED_GPU_NAME,
            "gpu_capability": EXPECTED_GPU_CAPABILITY,
            "driver": EXPECTED_DRIVER_VERSION,
            "wsl2": True,
            "block_cases": BLOCK_CASES,
        },
        "determinism": {
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
            "PYTHONWARNINGS": "error",
            "warnings_as_errors": True,
            "torch_use_deterministic_algorithms": True,
            "warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "fresh_process_replays": REPLAY_PROCESSES,
        "main_process_must_match_replays": True,
        "comparison": "complete nested CPU response-tensor semantic SHA-256",
        "one_shot_phase_claims": "exclusive and fsynced before CUDA configuration",
        "interrupted_or_failed_phase_reuse": False,
        "unsupported_operation_or_mismatch_stops_before_full_bank": True,
        "training_execution_authorized": False,
        "candidate_retained": False,
        "hover_gate_or_promotion_authorized": False,
    }


def configure_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise SystemExit(
            f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE_CONFIG!r} before startup"
        )
    if os.environ.get("PYTHONWARNINGS") != "error":
        raise SystemExit("PYTHONWARNINGS must be 'error' before startup")
    warnings.simplefilter("error")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise SystemExit("deterministic confirmation requires one driver version")
    return versions.pop()


def runtime_manifest(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("deterministic optic-motion confirmation requires CUDA")
    observed = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "driver": _driver_version(),
        "wsl2": "microsoft" in platform.release().lower(),
        "block_cases": BLOCK_CASES,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONWARNINGS": os.environ.get("PYTHONWARNINGS"),
        "warnings_as_errors": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    expected = {
        **protocol_manifest()["runtime"],
        **protocol_manifest()["determinism"],
    }
    key_map = {
        "torch_use_deterministic_algorithms": "deterministic_algorithms",
        "warn_only": "deterministic_warn_only",
    }
    for expected_name, value in expected.items():
        observed_name = key_map.get(expected_name, expected_name)
        if observed.get(observed_name) != value:
            raise SystemExit(
                f"deterministic runtime mismatch for {expected_name}: "
                f"{observed.get(observed_name)!r} != {value!r}"
            )
    return observed


def process_identity() -> dict[str, Any]:
    return {"pid": os.getpid(), "nonce": secrets.token_hex(16)}


def validate_process_identity(identity: Any) -> None:
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("pid"), int)
        or identity["pid"] <= 0
        or not isinstance(identity.get("nonce"), str)
        or len(identity["nonce"]) != 32
    ):
        raise SystemExit("deterministic replay process identity is invalid")


def validate_locked_v1(v1_report: Path, v1_implementation: Path) -> dict[str, str]:
    imported_implementation = Path(base.__file__).resolve()
    if imported_implementation != v1_implementation.resolve():
        raise SystemExit("the imported v1 audit is not the declared locked implementation")
    expected = {
        v1_report: EXPECTED_V1_REPORT_SHA256,
        imported_implementation: EXPECTED_V1_IMPLEMENTATION_SHA256,
    }
    observed = {}
    for path, digest in expected.items():
        if not path.is_file() or base.file_sha256(path) != digest:
            raise SystemExit(f"locked v1 deterministic-confirmation input changed: {path}")
        observed[base.assisted.responsibility.stable_path(path)] = digest
    with v1_report.open() as stream:
        report = json.load(stream)
    if (
        report.get("duplicate_control", {}).get("pass") is not False
        or report.get("duplicate_control", {}).get("numerical_noise_maximum_absolute")
        != 1.2516975402832031e-6
        or report.get("motion_output_routing_preregistration_authorized")
        or report.get("local_motion_commissioning_preregistration_authorized")
        or not report.get("source_restored")
    ):
        raise SystemExit("v1 report is not the registered duplicate-control stop")
    return observed


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def phase_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".claim.json"),
        output.with_name(output.stem + ".terminal.json"),
    )


def claim_phase(
    output: Path, *, phase: str, identity: dict[str, Any]
) -> tuple[Path, Path]:
    validate_process_identity(identity)
    claim, terminal = phase_paths(output)
    if output.exists() or terminal.exists():
        raise SystemExit(f"one-shot deterministic phase already terminated: {phase}")
    _exclusive_json(
        claim,
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "phase": phase,
            "output": base.assisted.responsibility.stable_path(output),
            "process_identity": identity,
        },
    )
    return claim, terminal


def finalize_phase(
    output: Path,
    *,
    phase: str,
    identity: dict[str, Any],
    status: str,
    error: BaseException | None = None,
) -> None:
    claim, terminal = phase_paths(output)
    if not claim.is_file():
        raise RuntimeError(f"deterministic phase lacks its exclusive claim: {phase}")
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "phase": phase,
        "process_identity": identity,
        "status": status,
        "claim_sha256": base.file_sha256(claim),
        "output": base.assisted.responsibility.stable_path(output),
        "output_sha256": base.file_sha256(output) if output.is_file() else None,
        "error": None,
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    _exclusive_json(terminal, payload)


def validate_completed_phase(output: Path, *, phase: str) -> dict[str, Any]:
    claim, terminal = phase_paths(output)
    if not output.is_file() or not claim.is_file() or not terminal.is_file():
        raise SystemExit(f"deterministic phase is incomplete: {phase}")
    with claim.open() as stream:
        claim_payload = json.load(stream)
    with terminal.open() as stream:
        terminal_payload = json.load(stream)
    if (
        claim_payload.get("experiment") != EXPERIMENT
        or claim_payload.get("protocol_commit") != PROTOCOL_COMMIT
        or claim_payload.get("phase") != phase
        or terminal_payload.get("experiment") != EXPERIMENT
        or terminal_payload.get("protocol_commit") != PROTOCOL_COMMIT
        or terminal_payload.get("phase") != phase
        or terminal_payload.get("status") != "completed"
        or terminal_payload.get("claim_sha256") != base.file_sha256(claim)
        or terminal_payload.get("output_sha256") != base.file_sha256(output)
        or terminal_payload.get("process_identity")
        != claim_payload.get("process_identity")
    ):
        raise SystemExit(f"deterministic phase terminal marker is invalid: {phase}")
    return terminal_payload


def response_tensor_tree(evaluation: dict[str, Any]) -> dict[str, Any]:
    tree = {name: evaluation[name].detach().cpu() for name in RESPONSE_TENSOR_NAMES}
    tree["terminal_visual_target_type_means"] = {
        name: value.detach().cpu()
        for name, value in evaluation["terminal_visual_target_type_means"].items()
    }
    return tree


def response_tensor_sha256(evaluation: dict[str, Any]) -> str:
    return base.assisted.audit.semantic_sha256(response_tensor_tree(evaluation))


def stable_evaluation_metadata(evaluation: dict[str, Any]) -> dict[str, Any]:
    compact = base.compact_evaluation(evaluation)
    return {name: value for name, value in compact.items() if name != "wall_time_seconds"}


def validate_stimulus_manifest(stimuli: dict[str, Any]) -> None:
    if (
        stimuli.get("rendered_float32_sha256") != EXPECTED_STIMULUS_SHA256
        or stimuli.get("opposite_terminal_maximum_difference") != 0.0
        or stimuli.get("duplicate_render_maximum_difference") != 0.0
    ):
        raise SystemExit("deterministic confirmation stimulus bank changed")


def validate_replay_report(report: dict[str, Any]) -> None:
    if report.get("experiment") != EXPERIMENT or report.get(
        "protocol_commit"
    ) != PROTOCOL_COMMIT:
        raise SystemExit("deterministic replay protocol identity mismatch")
    if report.get("protocol") != protocol_manifest():
        raise SystemExit("deterministic replay protocol manifest mismatch")
    if not report.get("passed") or not report.get("source_restored"):
        raise SystemExit("deterministic replay did not pass")
    validate_process_identity(report.get("process_identity"))
    if report.get("stimulus_rendered_float32_sha256") != EXPECTED_STIMULUS_SHA256:
        raise SystemExit("deterministic replay stimulus hash mismatch")
    if not isinstance(report.get("response_tensor_sha256"), str):
        raise SystemExit("deterministic replay lacks a response tensor hash")


def load_and_validate_replay_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit("deterministic replay gate is missing")
    with path.open() as stream:
        gate = json.load(stream)
    if (
        gate.get("experiment") != EXPERIMENT
        or gate.get("protocol_commit") != PROTOCOL_COMMIT
        or gate.get("protocol") != protocol_manifest()
        or not gate.get("passed")
        or gate.get("processes") != REPLAY_PROCESSES
        or len(gate.get("replay_reports", [])) != REPLAY_PROCESSES
    ):
        raise SystemExit("deterministic replay gate is invalid")
    resolved_paths = []
    identities = []
    for replay_index, item in enumerate(gate["replay_reports"], start=1):
        report_path = REPO_ROOT / item["path"]
        resolved_paths.append(report_path.resolve())
        terminal = validate_completed_phase(
            report_path, phase=f"replay-{replay_index}"
        )
        if not report_path.is_file() or base.file_sha256(report_path) != item["sha256"]:
            raise SystemExit("deterministic replay report changed after its gate")
        with report_path.open() as stream:
            report = json.load(stream)
        validate_replay_report(report)
        if terminal["process_identity"] != report["process_identity"]:
            raise SystemExit("deterministic replay terminal identity changed")
        if report["process_identity"] != item.get("process_identity"):
            raise SystemExit("deterministic replay process identity changed")
        identities.append(report["process_identity"])
        if report["response_tensor_sha256"] != gate["response_tensor_sha256"]:
            raise SystemExit("deterministic replay response hash changed")
    if len(set(resolved_paths)) != REPLAY_PROCESSES:
        raise SystemExit("deterministic replay gate reused a report path")
    if len({item["nonce"] for item in identities}) != REPLAY_PROCESSES:
        raise SystemExit("deterministic replay gate reused a process identity")
    return gate
