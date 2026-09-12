from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_pragmatic_two_gate_zero_shot import (  # noqa: E402
    compact_policy_metrics,
    sample_two_gate_cases,
)
from pragmatic_course_batch import ParameterBatchController, repeat_course_bank  # noqa: E402
from search_gate_motor_interface_es import motor_interface_spec  # noqa: E402
from search_pragmatic_full_native_gate_es import apply_vector  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


def test_candidate_scores_do_not_mix_flights_or_count_failed_completion():
    summaries = compact_policy_metrics(
        torch.ones(4, dtype=torch.bool),
        torch.tensor([True, True, False, False]),
        torch.tensor([False, False, True, True]),
        torch.tensor([5.0, 5.0, 1.0, 1.0]),
        torch.tensor([5.0, 5.0, 0.0, 0.0]),
        torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        5, 2,
    )
    assert summaries[0]["clean_course_success_rate"] == 1.0
    assert abs(summaries[0]["course_race_fitness"] - 10.2) < 1e-5
    assert summaries[1]["clean_course_success_rate"] == 0.0
    assert summaries[1]["course_race_fitness"] == -1.0


def test_repeated_course_bank_preserves_geometry_and_separates_pairs():
    bank = sample_two_gate_cases(
        2, seed=54, device=torch.device("cpu"), hover_config=HoverConfig(),
        layout="variable", gate_count=5, yaw_jitter_degrees=15.0,
    )
    cases, gates = repeat_course_bank(bank, 3)
    assert cases.pair.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    for gate, original in zip(gates, bank[1], strict=True):
        assert torch.equal(gate.center, original.center.repeat(3, 1))
        assert torch.equal(gate.yaw, original.yaw.repeat(3))
    cases.state.position[0, 0] = 100.0
    assert cases.state.position[4, 0] != 100.0
    assert bank[0].state.position[0, 0] != 100.0


@torch.no_grad()
def test_batched_native_parameters_match_compiled_checkpoint_recurrence():
    torch.manual_seed(17)
    graph = Path("artifacts/gate-v1/connectome.npz")
    base = ConnectomeController(graph, neural_dt=0.02)
    compiled = ConnectomeController(graph, neural_dt=0.02)
    compiled.load_state_dict(base.state_dict())
    spec = motor_interface_spec(
        base, bias_scale=0.0025, log_gain_scale=0.05,
        maximum_bias_delta=0.02, maximum_gain_ratio=2.0,
    )
    vectors = torch.randn(2, len(spec.labels)) * spec.scales
    actor = ParameterBatchController(base, vectors, spec, episodes=2)
    image = torch.rand(4, 3, 20, 32)
    attitude = torch.rand(4, 2) * 0.1
    initial = torch.rand(4, base.n_nodes) * 0.1
    neural = initial.clone()
    batched = []
    for _ in range(8):
        motor, neural = actor(image, attitude, neural)
        batched.append(motor)
    for candidate in range(2):
        apply_vector(compiled, base.bias, base.edge_magnitude, vectors[candidate], spec)
        take = slice(2 * candidate, 2 * candidate + 2)
        state = initial[take].clone()
        for expected in batched:
            motor, state = compiled(image[take], attitude[take], state)
            assert torch.allclose(motor, expected[take], atol=2e-6, rtol=2e-5)
        assert torch.allclose(state, neural[take], atol=2e-6, rtol=2e-5)
