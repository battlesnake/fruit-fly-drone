from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from flydrone.hover import ConnectomeController

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "hover-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_hover_artifact_is_self_consistent() -> None:
    graph_path = ARTIFACT_DIR / "connectome.npz"
    checkpoint_path = ARTIFACT_DIR / "controller.pt"
    report = json.loads((ARTIFACT_DIR / "report.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert report["passed"] is True
    assert report["commands_from_measured_sticks_only"] is True
    assert report["external_actor_state_machine"] is False
    assert report["checkpoint_sha256"] == sha256(checkpoint_path)
    assert report["evaluation_protocol"]["image_resolution"] == [32, 32]
    assert report["plant_model_version"] == "hover-surrogate-v2"
    assert report["evaluation"]["goal_success_rate"] >= 0.90
    assert report["visual_target_step"]["pass"] is True
    assert report["visual_target_step"]["paired_no_step_counterfactual"] is True
    assert report["frozen_visual_ablation"]["goal_success_rate"] == 0.0
    assert report["graph_sha256"] == sha256(graph_path)
    assert checkpoint["graph_sha256"] == sha256(graph_path)
    assert checkpoint["training_plant_model_version"] == "hover-surrogate-v2"
    assert checkpoint["hover_config"] == report["config"]

    controller = ConnectomeController(graph_path)
    controller.load_state_dict(checkpoint["controller"])
    assert controller.n_nodes == report["connectome_nodes"]
    assert controller.edge_pre.numel() == report["connectome_edges"]
