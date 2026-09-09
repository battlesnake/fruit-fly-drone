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
ACCEL_V2_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "gate-accel-v2"
PROPRIO_DIAGNOSTIC_DIR = REPO_ROOT / "artifacts" / "gate-proprio-diagnostic-v1"
MASS_ORACLE_DIR = REPO_ROOT / "artifacts" / "gate-mass-oracle-v1"
MOTOR_INTERFACE_ES_DIR = REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1"
ACCELERATION_PATH_DIAGNOSTIC_DIR = REPO_ROOT / "artifacts" / "gate-acceleration-path-diagnostic-v1"
RECURRENT_PPO_DIAGNOSTIC_DIR = REPO_ROOT / "artifacts" / "gate-recurrent-ppo-diagnostic-v1"
FULL_NETWORK_ORACLE_DIAGNOSTIC_DIR = (
    REPO_ROOT / "artifacts" / "gate-full-network-oracle-diagnostic-v1"
)


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
    manifest = json.loads((ACCEL_ARTIFACT_DIR / "connectome-manifest.json").read_text())
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


def test_edge_searched_accelerometer_checkpoint_is_bound_and_improves_gate_success() -> None:
    graph_path = ACCEL_V2_ARTIFACT_DIR / "connectome.npz"
    checkpoint_path = ACCEL_V2_ARTIFACT_DIR / "controller.pt"
    search_checkpoint_path = ACCEL_V2_ARTIFACT_DIR / "search-best.pt"
    report = json.loads((ACCEL_V2_ARTIFACT_DIR / "report.json").read_text())
    search = json.loads((ACCEL_V2_ARTIFACT_DIR / "search-report.json").read_text())
    bias_search = json.loads((ACCEL_V2_ARTIFACT_DIR / "bias-calibration-report.json").read_text())
    previous = json.loads((ACCEL_ARTIFACT_DIR / "report.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert report["graph_sha256"] == sha256(graph_path)
    assert report["checkpoint_sha256"] == sha256(checkpoint_path)
    assert report["source_checkpoint_sha256"] == sha256(search_checkpoint_path)
    assert checkpoint["graph_sha256"] == sha256(graph_path)
    assert search["candidate_checkpoint_sha256"] == sha256(search_checkpoint_path)
    assert bias_search["checkpoint_sha256"] == sha256(
        ACCEL_V2_ARTIFACT_DIR / "bias-calibration-candidate.pt"
    )
    assert bias_search["goal_passed"] is False
    assert search["search"]["acceleration_path_edges"] == 109
    assert search["biases_changed"] is False
    assert search["time_constants_changed"] is False
    assert search["old_anatomy_changed"] is False

    assert report["evaluation"]["success_rate"] > previous["evaluation"]["success_rate"]
    assert report["evaluation"]["success_rate"] == 467 / 1024
    assert report["constant_1g_accelerometer_ablation"]["success_rate"] == 415 / 1024
    assert report["mass_rank_swapped_accelerometer_ablation"]["success_rate"] == 467 / 1024
    assert report["frozen_visual_ablation"]["success_rate"] == 0.0
    assert search["signed_sensor_step_response"]["candidate"]["intended_signs"] == {
        "above_1g_decreases_throttle": True,
        "below_1g_increases_throttle": True,
    }
    assert (
        search["final"]["candidate_above_1g_channel_disabled"]["success_rate"]
        < search["final"]["candidate"]["success_rate"]
    )
    assert (
        search["final"]["candidate_below_1g_channel_disabled"]["success_rate"]
        < search["final"]["candidate"]["success_rate"]
    )


def test_throttle_proprioception_diagnostic_is_bound_but_not_sensor_dependent() -> None:
    graph_path = PROPRIO_DIAGNOSTIC_DIR / "connectome.npz"
    checkpoint_path = PROPRIO_DIAGNOSTIC_DIR / "candidate.pt"
    report = json.loads((PROPRIO_DIAGNOSTIC_DIR / "report.json").read_text())
    manifest = json.loads((PROPRIO_DIAGNOSTIC_DIR / "connectome-manifest.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert report["graph_sha256"] == sha256(graph_path)
    assert report["candidate_checkpoint_sha256"] == sha256(checkpoint_path)
    assert checkpoint["graph_sha256"] == sha256(graph_path)
    assert manifest["source_license"] == "CC BY 4.0"
    assert manifest["selection"]["proprioception_nodes"] == 4
    assert manifest["proprioception_interface"]["engineering_mapping_not_claimed_physiology"]
    assert manifest["proprioception_interface"]["directional_tuning_unresolved_in_malecns_release"]

    controller = ConnectomeController(
        graph_path,
        neural_dt=checkpoint["hover_config"]["dt"],
        retinal_receptive_field=checkpoint["retinal_receptive_field"],
    )
    controller.load_state_dict(checkpoint["controller"])
    assert controller.uses_proprioception
    assert controller.n_nodes == 1147
    assert controller.edge_pre.numel() == 4542

    final = report["final"]
    assert final["candidate"]["success_rate"] == 506 / 1024
    assert final["candidate_constant_proprioception"]["success_rate"] == 499 / 1024
    assert (
        final["candidate_mass_rank_swapped_proprioception"]["success_rate"]
        == final["candidate"]["success_rate"]
    )
    assert (
        abs(
            report["position_acceleration_probe"]["candidate"][
                "position_x_acceleration_interaction"
            ]
        )
        < 1e-6
    )
    assert report["sensor_dependence_checks"] == {
        "live_success_differs_from_episode_swapped_position": False,
        "live_success_exceeds_constant_position": True,
        "position_acceleration_interaction_exceeds_1e_6": False,
    }
    assert report["sensor_dependence_demonstrated"] is False
    assert report["promoted_as_controller"] is False
    assert report["goal_passed"] is False


def test_privileged_mass_oracle_is_effective_but_excluded_from_sensor_goal() -> None:
    report = json.loads((MASS_ORACLE_DIR / "report.json").read_text())
    candidate = json.loads((MASS_ORACLE_DIR / "candidate.json").read_text())

    assert report["graph_sha256"] == sha256(ACCEL_V2_ARTIFACT_DIR / "connectome.npz")
    assert report["checkpoint_sha256"] == sha256(ACCEL_V2_ARTIFACT_DIR / "controller.pt")
    assert report["controller_parameters_changed"] is False
    assert report["per_episode_neuronal_drive_bias_changed"] is True
    assert report["synaptic_weights_changed"] is False
    assert report["connectome_edges_changed"] is False
    assert report["counts_toward_direct_sensor_goal"] is False
    assert candidate == report["selected_mass_conditioned_trim"]
    assert candidate["kind"] == "privileged_non_biological_mass_oracle"

    final = report["final"]
    assert final["mass_conditioned_trim"]["success_rate"] == 1.0
    assert final["unchanged_controller"]["success_rate"] == 438 / 1024
    assert final["validation_selected_constant_trim"]["success_rate"] == 416 / 1024
    assert final["mass_labels_shuffled_within_geometry"]["success_rate"] == 103 / 1024
    assert all(report["causal_checks"].values())


def test_motor_interface_es_checkpoint_is_native_and_materially_improved() -> None:
    report = json.loads((MOTOR_INTERFACE_ES_DIR / "report.json").read_text())
    checkpoint_path = MOTOR_INTERFACE_ES_DIR / "controller.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert report["candidate"]["checkpoint_sha256"] == sha256(checkpoint_path)
    assert checkpoint["graph_sha256"] == sha256(ACCEL_V2_ARTIFACT_DIR / "connectome.npz")
    assert checkpoint["source_checkpoint_sha256"] == sha256(ACCEL_V2_ARTIFACT_DIR / "controller.pt")
    assert checkpoint["motor_interface_es"]["compiled_into_native_parameters"] is True
    assert report["candidate"]["parameter_count"] == 24
    assert report["candidate"]["compiled_edge_magnitudes_changed"] == 196
    assert report["candidate"]["compiled_motor_biases_changed"] == 26
    assert report["candidate"]["time_constants_changed"] == 0

    baseline = report["final"]["baseline"]
    candidate = report["final"]["candidate"]
    paired = report["final"]["paired_success_difference"]
    assert candidate["success_rate"] >= baseline["success_rate"] + 0.05
    assert candidate["negative_lateral_success_rate"] > baseline["negative_lateral_success_rate"]
    assert candidate["positive_lateral_success_rate"] > baseline["positive_lateral_success_rate"]
    assert candidate["crossing_radial_mean_m"] < baseline["crossing_radial_mean_m"]
    assert paired["confidence_95"][0] > 0.0
    assert report["final"]["frozen_first_frame_success_rate"] <= 0.05
    assert report["outcome"]["promotion_passed"] is True
    assert report["outcome"]["acceleration_dependence_demonstrated"] is False
    assert report["outcome"]["goal_passed"] is False


def test_recurrent_acceleration_path_search_is_recorded_as_rejected() -> None:
    report = json.loads((ACCELERATION_PATH_DIAGNOSTIC_DIR / "report.json").read_text())
    vector = json.loads((ACCELERATION_PATH_DIAGNOSTIC_DIR / "candidate-vector.json").read_text())

    assert report["source"]["checkpoint_sha256"] == sha256(MOTOR_INTERFACE_ES_DIR / "controller.pt")
    assert report["source"]["graph_sha256"] == sha256(ACCEL_V2_ARTIFACT_DIR / "connectome.npz")
    assert report["runtime_contract"]["external_history_features"] == 0
    assert report["parameterization"]["selected_edges"] == 282
    assert report["parameterization"]["biases_changed"] == 0
    assert report["parameterization"]["time_constants_changed"] == 0
    assert report["parameterization"]["initially_zero_selected_edges"] == 13
    assert report["search"]["checkpoint_rule_passed"] is True
    assert report["search"]["selected_candidate_vector_file_sha256"] == sha256(
        ACCELERATION_PATH_DIAGNOSTIC_DIR / "candidate-vector.json"
    )
    assert len(vector["selected_edge_indices"]) == 282
    assert len(vector["normalized_parameter_vector"]) == 282
    vector_bytes = (
        torch.tensor(vector["normalized_parameter_vector"], dtype=torch.float32)
        .numpy()
        .astype("<f4", copy=False)
        .tobytes()
    )
    assert hashlib.sha256(vector_bytes).hexdigest() == report["search"]["selected_candidate_sha256"]

    matched = report["fresh_matched_1024"]
    assert matched["candidate"]["light_success_rate"] > matched["reference"]["light_success_rate"]
    assert matched["paired_light_success_difference"]["confidence_95"][0] > 0.0
    assert (
        matched["candidate"]["heavy_success_rate"]
        >= matched["reference"]["heavy_success_rate"] - 0.02
    )
    assert report["causal_acceleration_controls"]["acceleration_dependence_demonstrated"] is False
    assert report["promotion"]["light_improvement_at_least_threshold"] is False
    assert report["promotion"]["passed"] is False
    assert report["promotion"]["checkpoint_promoted"] is False
    assert not (ACCELERATION_PATH_DIAGNOSTIC_DIR / "controller.pt").exists()


def test_recurrent_ppo_is_audited_and_recorded_as_rejected() -> None:
    report = json.loads((RECURRENT_PPO_DIAGNOSTIC_DIR / "report.json").read_text())
    vector = json.loads((RECURRENT_PPO_DIAGNOSTIC_DIR / "candidate-vector.json").read_text())
    archive = torch.load(
        RECURRENT_PPO_DIAGNOSTIC_DIR / "archive.pt", map_location="cpu", weights_only=True
    )

    assert report["source_checkpoint_sha256"] == sha256(MOTOR_INTERFACE_ES_DIR / "controller.pt")
    assert report["graph_sha256"] == sha256(ACCEL_V2_ARTIFACT_DIR / "connectome.npz")
    assert report["parameterization"]["independent_motor_input_edge_magnitudes"] == 198
    assert report["parameterization"]["independent_motor_neuron_biases"] == 26
    assert report["parameterization"]["count"] == 224
    assert report["protocol"]["selected_exploration_sigma"] == 0.00375
    assert report["protocol"]["critic_training_only"] is True
    assert report["protocol"]["physics_renderer_outside_autograd"] is True
    assert report["protocol"]["proprioception_input_active"] is False
    assert report["native_forward_parity"]["passed"] is True
    assert report["batched_evaluator_parity"]["passed"] is True
    assert report["likelihood_gradient_audit"]["passed"] is True
    assert report["unchanged_policy_replay_audit"]["passed"] is True
    assert all(item["burn_in_approximation_audit"]["passed"] for item in report["iterations"])

    assert report["selected_candidate_vector_file_sha256"] == sha256(
        RECURRENT_PPO_DIAGNOSTIC_DIR / "candidate-vector.json"
    )
    assert vector["parameter_vector_sha256"] == report["selected_parameter_vector_sha256"]
    vector_values = torch.tensor(vector["edge_magnitudes"] + vector["biases"], dtype=torch.float32)
    assert (
        hashlib.sha256(vector_values.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        == report["selected_parameter_vector_sha256"]
    )
    selected = archive[report["selected_candidate"]]
    archived_values = torch.cat((selected["edge_magnitudes"], selected["biases"]))
    assert (
        hashlib.sha256(archived_values.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        == report["selected_parameter_vector_sha256"]
    )

    assert report["iterations_completed"] == 10
    assert report["early_stopped_at_checkpoint"] is True
    final = report["final"]
    assert final["reference"]["success_rate"] == 488 / 1024
    assert final["candidate"]["success_rate"] == 487 / 1024
    assert final["candidate"]["light_success_rate"] > final["reference"]["light_success_rate"]
    assert final["candidate"]["heavy_success_rate"] < final["reference"]["heavy_success_rate"]
    assert report["paired_final"]["light_success_difference"]["confidence_95"][0] > 0.0
    assert report["acceleration_dependence_demonstrated"] is False
    assert report["promotion"]["passed"] is False
    assert report["goal_passed"] is False
    assert not (RECURRENT_PPO_DIAGNOSTIC_DIR / "controller.pt").exists()


def test_full_network_oracle_distillation_is_audited_and_rejected() -> None:
    report = json.loads((FULL_NETWORK_ORACLE_DIAGNOSTIC_DIR / "report.json").read_text())
    vector = json.loads((FULL_NETWORK_ORACLE_DIAGNOSTIC_DIR / "candidate-vector.json").read_text())
    archive = torch.load(
        FULL_NETWORK_ORACLE_DIAGNOSTIC_DIR / "archive.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert report["student_checkpoint_sha256"] == sha256(MOTOR_INTERFACE_ES_DIR / "controller.pt")
    assert report["teacher_checkpoint_sha256"] == sha256(ACCEL_V2_ARTIFACT_DIR / "controller.pt")
    assert report["graph_sha256"] == sha256(ACCEL_V2_ARTIFACT_DIR / "connectome.npz")
    assert report["protocol"]["all_native_edge_magnitudes_trainable"] is True
    assert report["protocol"]["all_native_biases_trainable"] is True
    assert report["protocol"]["all_native_time_constants_trainable"] is True
    assert report["protocol"]["engineered_history_features"] is False
    assert report["protocol"]["proprioception_input_active"] is False
    assert report["student_collection_causality_audit"]["passed"] is True
    assert report["teacher_takeover_audit"]["passed"] is True
    assert all(
        family["passed"]
        for window in report["gradient_audits"].values()
        for length in window.values()
        for family in length.values()
    )

    assert vector["parameter_vector_sha256"] == report["selected_parameter_vector_sha256"]
    vector_values = torch.tensor(
        vector["edge_magnitude"] + vector["bias"] + vector["raw_time_constant"],
        dtype=torch.float32,
    )
    assert (
        hashlib.sha256(vector_values.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        == report["selected_parameter_vector_sha256"]
    )
    selected = archive[report["selected_candidate"]]["parameters"]
    archived_values = torch.cat(
        (selected["edge_magnitude"], selected["bias"], selected["raw_time_constant"])
    )
    assert (
        hashlib.sha256(archived_values.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        == report["selected_parameter_vector_sha256"]
    )

    assert report["updates_completed"] == 100
    assert report["early_stopped_at_checkpoint"] is True
    final = report["final"]
    assert final["reference"]["success_rate"] == 486 / 1024
    assert final["candidate"]["success_rate"] == 441 / 1024
    assert final["candidate"]["light_success_rate"] > final["reference"]["light_success_rate"]
    assert final["candidate"]["heavy_success_rate"] < final["reference"]["heavy_success_rate"]
    assert report["fresh_action_fidelity_passed"] is False
    assert report["acceleration_dependence_demonstrated"] is False
    assert report["promotion"]["passed"] is False
    assert report["candidate_checkpoint"] is None
    assert report["goal_passed"] is False
    assert not (FULL_NETWORK_ORACLE_DIAGNOSTIC_DIR / "candidate.pt").exists()
