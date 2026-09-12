from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_pragmatic_fixed_replay_fit as audit  # noqa: E402


def fixtures():
    banks = []
    for seed, kind in ((11, "native"), (12, "roll-assisted")):
        positions = torch.arange(20)[:, None, None].expand(20, 6, 3).float() + seed
        banks.append(
            dict(
                states=tuple(positions.clone() for _ in range(6)),
                gates=[(torch.full((6, 3), float(seed)), torch.zeros(6)) for _ in range(5)],
                current=torch.arange(5).repeat_interleave(4)[:, None].expand(20, 6).clone(),
                active=torch.ones(20, 6, dtype=torch.bool),
                target=torch.full((20, 6, 4), float(seed)),
                reference_outputs=torch.zeros(20, 6, 4),
                failure_steps=None,
                kind=kind,
                seed=seed,
                roll_teacher="current-gate",
            )
        )
    records = []
    for phase in range(1, 6):
        for kind in ("native",) if phase == 1 else ("native", "roll-assisted"):
            seed = 11 if kind == "native" else 12
            record = dict(
                phase=phase,
                requested_source=kind,
                source_by_side=[kind, kind],
                seed_by_side=[seed, seed],
                rows=[0, 1],
                starts=[(phase - 1) * 4] * 2,
            )
            if phase == 5 and kind == "native":
                record.update(
                    source_by_side=["native", "roll-assisted"],
                    seed_by_side=[11, 12],
                    rows=[0, 5],
                    starts=[16, 18],
                )
            record.update(
                axis_order=["roll", "pitch", "yaw", "throttle"],
                side_order=["negative", "positive"],
                motor_rmse_by_side=[[0.0] * 4, [0.0] * 4],
            )
            records.append(record)
    return {"banks": banks}, {"arguments": {"unroll": 2}, "source_replay_fit": {"windows": records}}


def test_restored_fixed_lessons_keep_mixed_sources_and_original_history():
    cache, report = fixtures()
    lessons, unroll = audit.fixed_lessons(cache, report, torch.device("cpu"))
    assert len(lessons) == 9 and unroll == 2
    mixed, record = lessons[7]
    assert record["source_by_side"] == ["native", "roll-assisted"]
    assert mixed[4].tolist() == [16, 18]
    assert torch.equal(mixed[0][0][:18, 0], cache["banks"][0]["states"][0][:18, 0])
    assert torch.equal(mixed[0][0][:, 1], cache["banks"][1]["states"][0][:, 5])
    assert mixed[3][16, 0, 0] == 11 and mixed[3][18, 1, 0] == 12
    assert mixed[1][0].center[:, 0].tolist() == [11, 12]
    assert torch.equal(cache["banks"][0]["states"][0], fixtures()[0]["banks"][0]["states"][0])


def test_reported_lesson_metadata_can_be_measured_again_without_duplicate_fields(monkeypatch):
    lessons, unroll = audit.fixed_lessons(*fixtures(), torch.device("cpu"))
    monkeypatch.setattr(
        audit.replay,
        "replay_window_loss",
        lambda *args, **kwargs: (torch.tensor(0.0), torch.zeros(4), torch.zeros(unroll, 2, 4)),
    )
    summary = audit.replay.replay_fit_summary(None, lessons, unroll, None, None, 1.0)
    assert len(summary["windows"]) == 9
    assert summary["windows"][0]["axis_order"] == ["roll", "pitch", "yaw", "throttle"]
    assert summary["windows"][0]["side_order"] == ["negative", "positive"]
    assert summary["windows"][0]["motor_rmse_by_side"] == [[0.0] * 4, [0.0] * 4]


@pytest.mark.parametrize("fault", ["inactive", "early-crossing", "wrong-side", "missing-role"])
def test_fixed_lesson_validation_rejects_invalid_training_intervals(fault):
    cache, report = fixtures()
    if fault == "inactive":
        cache["banks"][0]["active"][0, 0] = False
    elif fault == "early-crossing":
        cache["banks"][0]["current"][1, 0] = 1
    elif fault == "wrong-side":
        report["source_replay_fit"]["windows"][0]["rows"][0] = 1
    else:
        report["source_replay_fit"]["windows"].pop()
    with pytest.raises(ValueError):
        audit.fixed_lessons(cache, report, torch.device("cpu"))


def test_objective_weights_and_two_sided_fit_screen():
    lessons, _ = audit.fixed_lessons(*fixtures(), torch.device("cpu"))
    weights = audit.lesson_weights(lessons)
    assert sum(weights) == 1 and weights[0] == 0.5
    for source in ("native", "roll-assisted"):
        assert (
            sum(
                w
                for w, (_, r) in zip(weights, lessons, strict=True)
                if r["phase"] > 1 and r["requested_source"] == source
            )
            == 0.25
        )
    source = {"late_roll_rmse_by_side": [0.04, 0.08]}
    current = {"late_roll_rmse_by_side": [0.01, 0.06]}
    assert audit.fitting_reductions(source, current) == pytest.approx([0.75, 0.25])
    assert min(audit.fitting_reductions(source, current)) < 0.5


def test_fit_summary_excludes_early_roll_from_late_and_roll_from_nonroll():
    windows = [dict(phase=1, motor_rmse_by_side=[[9, 0.1, 0.2, 0.3], [8, 0.1, 0.2, 0.3]])]
    windows += [dict(phase=2, motor_rmse_by_side=[[3, 0.1, 0.2, 0.3], [4, 0.1, 0.2, 0.5]])]
    result = audit.fit_metrics({"windows": windows})
    assert result["late_roll_rmse_by_side"] == [3, 4]
    assert result["early_roll_rmse_by_side"] == [9, 8]
    assert result["maximum_nonroll_rmse"] == 0.5


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1])
def test_nonfinite_or_negative_errors_cannot_pass_two_sided_fit_screen(invalid):
    source = {"late_roll_rmse_by_side": [0.04, 0.08]}
    current = {"late_roll_rmse_by_side": [0.01, invalid]}
    with pytest.raises(ValueError):
        audit.fitting_reductions(source, current)
    with pytest.raises(ValueError):
        audit.fitting_reductions(current, source)


def test_fit_summary_checks_nonroll_finiteness_too():
    with pytest.raises(ValueError):
        audit.fit_metrics(
            {
                "windows": [
                    dict(
                        phase=1,
                        motor_rmse_by_side=[[0.1, float("nan"), 0.2, 0.3], [0.1, 0.1, 0.2, 0.3]],
                    )
                ]
            }
        )


def test_anchor_cost_and_gradient_on_shared_edges_do_not_weaken_with_broader_mask():
    parameter = torch.tensor([0.02, 0.0, 0.0], requires_grad=True)
    initial = torch.zeros_like(parameter)
    narrow = torch.tensor([True, False, False])
    broad = torch.ones(3, dtype=torch.bool)
    narrow_loss = audit.edge_anchor_loss(parameter, initial, narrow, 1)
    broad_loss = audit.edge_anchor_loss(parameter, initial, broad, 1)
    assert torch.equal(narrow_loss, broad_loss)
    assert torch.equal(
        torch.autograd.grad(narrow_loss, parameter)[0],
        torch.autograd.grad(broad_loss, parameter)[0],
    )
