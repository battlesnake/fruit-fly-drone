from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from flydrone.gate import GATE_TASK_VERSION
from flydrone.hover import PLANT_MODEL_VERSION, ConnectomeController

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "gate-v1"
ACCEL_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "gate-accel-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_gate_artifact_is_bound_and_visually_dependent() -> None:
    graph_path = ARTIFACT_DIR / "connectome.npz"
    checkpoint_path = ARTIFACT_DIR / "controller.pt"
    report = json.loads((ARTIFACT_DIR / "report.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert report["task_version"] == GATE_TASK_VERSION
    assert report["plant_model_version"] == PLANT_MODEL_VERSION
    assert report["graph_sha256"] == sha256(graph_path)
    assert report["checkpoint_sha256"] == sha256(checkpoint_path)
    assert checkpoint["graph_sha256"] == sha256(graph_path)
    assert checkpoint["task_version"] == GATE_TASK_VERSION

    controller = ConnectomeController(
        graph_path,
        neural_dt=checkpoint["hover_config"]["dt"],
        retinal_receptive_field=checkpoint["retinal_receptive_field"],
    )
    controller.load_state_dict(checkpoint["controller"])
    assert controller.n_nodes == 1122
    assert controller.edge_pre.numel() == 4324

    evaluation = report["evaluation"]
    frozen = report["frozen_visual_ablation"]
    assert evaluation["episodes"] == 1024
    assert evaluation["lift_off_rate"] == 1.0
    assert evaluation["success_rate"] >= 0.40
    assert evaluation["success_by_stratum"]["negative_lateral_offset"] >= 0.35
    assert evaluation["success_by_stratum"]["positive_lateral_offset"] >= 0.40
    assert frozen["success_rate"] == 0.0
    assert report["commands_from_measured_sticks_only"] is True
    assert report["external_actor_state_machine"] is False
    assert report["privileged_gate_geometry_given_to_actor"] is False


def test_gate_showcase_records_a_real_success() -> None:
    showcase = json.loads((ARTIFACT_DIR / "showcase.json").read_text())
    assert showcase["pass_time_seconds"] < 11.0
    assert showcase["max_tilt_degrees"] < 40.0
    assert math.isclose(abs(showcase["gate_center_m"][1]), 0.8, abs_tol=1.0e-6)
    assert (ARTIFACT_DIR / "showcase.mp4").stat().st_size > 100_000
    assert (ARTIFACT_DIR / "showcase-trajectory.npz").stat().st_size > 50_000
    assert showcase["external_actor_state_machine"] is False
    assert showcase["privileged_gate_geometry_given_to_actor"] is False


def test_accelerometer_checkpoint_is_bound_and_sensor_dependent() -> None:
    graph_path = ACCEL_ARTIFACT_DIR / "connectome.npz"
    checkpoint_path = ACCEL_ARTIFACT_DIR / "controller.pt"
    report = json.loads((ACCEL_ARTIFACT_DIR / "report.json").read_text())
    manifest = json.loads(
        (ACCEL_ARTIFACT_DIR / "connectome-manifest.json").read_text()
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert report["graph_sha256"] == sha256(graph_path)
    assert report["checkpoint_sha256"] == sha256(checkpoint_path)
    assert checkpoint["graph_sha256"] == sha256(graph_path)
    assert manifest["source_license"] == "CC BY 4.0"
    assert manifest["selection"]["acceleration_nodes"] == 8

    controller = ConnectomeController(
        graph_path,
        neural_dt=checkpoint["hover_config"]["dt"],
        retinal_receptive_field=checkpoint["retinal_receptive_field"],
    )
    controller.load_state_dict(checkpoint["controller"])
    assert controller.uses_accelerometer
    assert controller.n_nodes == 1138
    assert controller.edge_pre.numel() == 4469

    live = report["evaluation"]
    constant = report["constant_1g_accelerometer_ablation"]
    assert report["passed"] is False
    assert live["success_rate"] > constant["success_rate"]
    assert (
        live["crossing_error_components_m"]["vertical_absolute_mean"]
        < constant["crossing_error_components_m"]["vertical_absolute_mean"]
    )
    assert report["frozen_visual_ablation"]["success_rate"] == 0.0
