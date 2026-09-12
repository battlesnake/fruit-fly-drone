from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_pragmatic_replay_transfer as audit  # noqa: E402


def bank():
    positions = torch.arange(1, 61).float()[:, None, None].expand(60, 4, 3).clone()
    return audit.replay.ReplayBank(
        states=tuple(positions.clone() for _ in range(6)),
        gates=tuple(audit.replay.AnnularGate(torch.zeros(4, 3), torch.zeros(4)) for _ in range(5)),
        current=torch.arange(5).repeat_interleave(12)[:, None].expand(60, 4).clone(),
        active=torch.ones(60, 4, dtype=torch.bool),
        target=torch.zeros(60, 4, 4),
        kind="native",
        seed=1,
        roll_teacher="current-gate",
    )


def test_old_transfer_audit_rejects_unified_label_experiment(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"roll_labels": "unified"}))
    with pytest.raises(ValueError, match="not unified labels"):
        audit.nominated_checkpoint(tmp_path, 5, 25)


def test_group_eligibility_needs_frames_and_distinct_episodes():
    data = bank()
    residual = torch.ones(60, 4, 4)
    stats = audit.phase_errors(residual, data)
    assert all(record["eligible"] for record in stats)
    assert stats[0]["frames"] == 24 and stats[0]["episodes"] == 2
    data.active[:, 2] = False
    stats = audit.phase_errors(residual, data)
    assert all(not record["eligible"] for record in stats if record["side"] == 0)
    data.active[:3, 1] = False
    data.active[:3, 3] = False
    stats = audit.phase_errors(residual, data)
    assert stats[1]["frames"] == 18 and stats[1]["episodes"] == 2
    assert not stats[1]["eligible"]


def test_inactive_nan_is_not_scored_but_active_nan_fails():
    data = bank()
    residual = torch.ones(60, 4, 4)
    data.active[0, 0] = False
    residual[0, 0, 0] = float("nan")
    audit.phase_errors(residual, data)
    data.active[0, 0] = True
    with pytest.raises(ValueError, match="nonfinite"):
        audit.phase_errors(residual, data)


def test_full_replay_retains_unscored_history_and_independent_recurrence(monkeypatch):
    class Actor(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.scale = scale
            self.calls = 0

        def initial_state(self, count, **kwargs):
            return torch.zeros(count, 1, **kwargs)

        def forward(self, image, attitude, state):
            self.calls += 1
            state = 0.5 * state + image[:, 0, 0, 0, None]
            return torch.cat((state * self.scale, torch.zeros(len(state), 3)), 1), state

    monkeypatch.setattr(
        audit.replay,
        "render_annular_gates_rgb",
        lambda state, *args, **kwargs: state.position[:, 0, None, None, None].expand(-1, 3, 1, 1),
    )
    data = bank()
    data.active[:24] = False
    source, candidate = Actor(1.0), Actor(0.5)
    result = audit.compare_bank(source, candidate, data, None, None)
    assert source.calls == candidate.calls == 70  # ten warmup + every recorded frame
    expected_state = 0.0
    for _ in range(10):
        expected_state = 0.5 * expected_state + 1.0
    outputs = []
    for value in range(1, 61):
        expected_state = 0.5 * expected_state + value
        outputs.append(expected_state)
    expected = torch.tensor(outputs[24:36]).square().mean().sqrt().item()
    assert result["source"][4]["motor_rmse"][0] == pytest.approx(expected)
    assert result["candidate"][4]["motor_rmse"][0] == pytest.approx(0.5 * expected)
    assert result["source"][0]["motor_rmse"] is None


def records():
    result = []
    for kind in ("native", "roll-assisted"):
        stats = [
            dict(
                phase=phase,
                side=side,
                frames=24,
                episodes=2,
                eligible=True,
                motor_rmse=[0.04, 0.0, 0.0, 0.0],
            )
            for phase in range(1, 6)
            for side in (0, 1)
        ]
        candidate = copy.deepcopy(stats)
        for entry in candidate:
            entry["motor_rmse"][0] = 0.02
        result.append(dict(kind=kind, source=stats, candidate=candidate))
    return result


def remove_group(record, phase, side, *, frames=0):
    for key in ("source", "candidate"):
        group = next(g for g in record[key] if g["phase"] == phase and g["side"] == side)
        group.update(
            frames=frames,
            episodes=2 if frames else 0,
            eligible=False,
            motor_rmse=[0.04] * 4 if frames else None,
        )


def test_missing_native_gate_five_is_explicit_and_assisted_coverage_can_complete_it():
    data = records()
    remove_group(data[0], 5, 1)
    result = audit.summarize_transfer(data)
    assert result["coverage_complete"]
    assert result["combined"][1]["groups"] == 7
    assert result["native_only"][1]["groups"] == 3
    assert result["permits_expanded_native_validation"]
    assert not result["autonomous_success_established"]
    remove_group(data[1], 5, 1, frames=19)
    result = audit.summarize_transfer(data)
    assert not result["coverage_complete"]
    assert not result["permits_expanded_native_validation"]


def test_assisted_only_gain_cannot_hide_native_regression():
    data = records()
    for source, candidate in zip(data[0]["source"], data[0]["candidate"], strict=True):
        source["motor_rmse"][0] = 0.01
        candidate["motor_rmse"][0] = 0.02
    result = audit.summarize_transfer(data)
    assert result["aggregate_twenty_percent_screen"]
    assert result["assisted_state_only_improvement"]
    assert not result["permits_expanded_native_validation"]


def test_mismatched_scoring_masks_fail():
    data = records()
    data[0]["candidate"][0]["frames"] -= 1
    with pytest.raises(ValueError, match="identical scoring masks"):
        audit.summarize_transfer(data)


@pytest.mark.parametrize("reductions", [[0.6, 0.7], [0.6, 0.1], [0.6, float("nan")]])
def test_transfer_collection_requires_a_two_sided_fitted_candidate(tmp_path, reductions):
    record = dict(
        arms=[
            dict(
                hops=5,
                history=[
                    dict(
                        update=25,
                        fit={},
                        late_roll_reduction_by_side=reductions,
                        checkpoint="candidate.pt",
                    )
                ],
            )
        ]
    )
    (tmp_path / "report.json").write_text(json.dumps(record))
    if reductions == [0.6, 0.7]:
        _, path = audit.nominated_checkpoint(tmp_path, 5, 25)
        assert path == tmp_path / "candidate.pt"
    else:
        with pytest.raises(ValueError, match="fifty-percent"):
            audit.nominated_checkpoint(tmp_path, 5, 25)
