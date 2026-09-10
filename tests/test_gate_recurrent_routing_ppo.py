from __future__ import annotations

from pathlib import Path

import torch

from flydrone.hover import ForelegStickPlant
from scripts.audit_gate_throttle_routing import make_readout_spec
from scripts.gate_diverse_cases import diverse_matched_cases
from scripts.search_gate_acceleration_path_es import make_path_spec
from scripts.search_gate_motor_interface_es import load_controller
from scripts.train_gate_recurrent_routing import make_recurrent_routing_spec
from scripts.train_gate_recurrent_routing_ppo import (
    PrivilegedCritic,
    RoutingActor,
    assisted_steering_motor,
    collect_rollout,
    critic_features,
    initialize_outcomes,
    mass_lateral_floor,
    outcome_potential,
    progress_gate,
    replay_consistency_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def metrics(*, light: float, heavy: float, negative: float, positive: float) -> dict[str, float]:
    return {
        "light_success_rate": light,
        "heavy_success_rate": heavy,
        "negative_lateral_success_rate": negative,
        "positive_lateral_success_rate": positive,
    }


def test_progress_gate_requires_light_and_worst_stratum_gains() -> None:
    source = metrics(light=0.20, heavy=0.75, negative=0.0, positive=0.40)
    candidate = metrics(light=0.31, heavy=0.76, negative=0.11, positive=0.50)
    assert mass_lateral_floor(source) == 0.0
    assert progress_gate(
        source,
        candidate,
        light_improvement=0.10,
        floor_improvement=0.10,
        maximum_drop=0.05,
    )
    candidate["negative_lateral_success_rate"] = 0.09
    assert not progress_gate(
        source,
        candidate,
        light_improvement=0.10,
        floor_improvement=0.10,
        maximum_drop=0.05,
    )


def test_routing_actor_projects_only_nonnegative_bounded_magnitudes() -> None:
    actor = RoutingActor(torch.tensor([-1.0, 2.0, 9.0]))
    actor.project()
    assert actor.edge_magnitudes.tolist() == [0.0, 2.0, 8.0]
    assert list(actor.named_parameters())[0][0] == "edge_magnitudes"


def test_height_potential_uses_current_not_historical_best_alignment() -> None:
    progress = torch.tensor([0.4])
    aligned = outcome_potential(progress, torch.tensor([0.9]))
    drifted = outcome_potential(progress, torch.tensor([0.1]))
    assert float(drifted) < float(aligned)


def test_launch_steering_uses_frozen_source_motor_directly() -> None:
    source_motor = torch.tensor([[0.1, -0.2, 0.3, 0.7]])
    steering = assisted_steering_motor(
        None,
        source_motor,
        None,
        None,
        torch.ones(1),
        use_reserve=False,
        hover_config=None,
    )
    assert torch.equal(steering, source_motor[:, :3])


def test_short_scalar_throttle_rollout_replays_exactly() -> None:
    device = torch.device("cpu")
    graph_path = REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz"
    checkpoint_path = REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt"
    controller, _, hover_config, gate_config, resolution = load_controller(
        graph_path, checkpoint_path, device
    )
    path_spec = make_path_spec(
        controller,
        graph_path,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=8.0,
    )
    routing_spec = make_recurrent_routing_spec(controller, path_spec, make_readout_spec(controller))
    actor = RoutingActor(controller.edge_magnitude[routing_spec.selected_edges])
    cases = diverse_matched_cases(
        8,
        seed=92_001,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    sticks = ForelegStickPlant(hover_config).initial_state(8, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    feature_count = (
        critic_features(
            cases.state,
            cases.gate,
            cases.mass_scale,
            sticks.position,
            sticks.velocity,
            outcomes["passed"],
            outcomes["maximum_progress"],
            outcomes["maximum_approach"],
            outcomes["saturation_steps"],
            0.0,
            hover_config,
        ).shape[1]
        + controller.n_nodes
    )
    critic = PrivilegedCritic(feature_count)
    rollout = collect_rollout(
        controller,
        actor,
        critic,
        routing_spec.selected_edges,
        cases,
        seconds=0.08,
        takeover_seconds=0.04,
        sigma=0.01,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=92_002,
    )
    assert rollout.latent_actions.shape == (8, 8, 1)
    audit = replay_consistency_audit(
        controller,
        actor,
        routing_spec.selected_edges,
        rollout,
        chunk_steps=4,
        burn_in_steps=2,
        sigma=0.01,
        sequence_minibatch=8,
    )
    assert audit["action_dimensions"] == 1
    assert audit["passed"]
