from __future__ import annotations

import copy
import sys
from argparse import Namespace
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_full_native_d_first_fp64_snapshot_corrected as fit  # noqa: E402


def _source_resume() -> dict[str, object]:
    controller = {
        "edge_magnitude": torch.tensor([1.0]),
        "bias": torch.tensor([2.0]),
        "raw_time_constant": torch.tensor([3.0]),
        "buffer": torch.tensor([4.0]),
    }
    return {
        "experiment": fit.corrected.EXPERIMENT,
        "protocol_commit": fit.corrected.PROTOCOL_COMMIT,
        "accepted_updates": 20,
        "controller": controller,
        "optimizer": {"state": {}, "param_groups": []},
        "history": [
            {"update": 19, "accepted": True},
            {"update": 20, "accepted": True},
            {"update": 21, "accepted": False},
        ],
        "development_history": [{"update": 10}, {"update": 20}],
        "preflight": {"pass": True},
        "run_state": "stopped",
        "qualification": None,
    }


def _snapshot() -> dict[str, object]:
    return {
        "candidate_parameters": {
            "edge_magnitude": torch.tensor([1.1]),
            "bias": torch.tensor([2.1]),
            "raw_time_constant": torch.tensor([3.1]),
        },
        "pending_optimizer_state": {"state": {0: {"step": torch.tensor(21.0)}}},
        "selected_training_metrics": {"metric": 1.0},
    }


def test_seed_resume_installs_exact_snapshot_as_update_21(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    snapshot_dir = tmp_path / "snapshot"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    snapshot_dir.mkdir()
    source_path = source_dir / "resume.pt"
    archive_path = snapshot_dir / fit.snapshot_audit.ARCHIVE_NAME
    torch.save(_source_resume(), source_path)
    torch.save(_snapshot(), archive_path)
    monkeypatch.setattr(fit, "_SNAPSHOT_CPU", _snapshot())
    monkeypatch.setattr(
        fit.responsibility,
        "file_sha256",
        lambda path: (
            fit.EXPECTED_SOURCE_RESUME_SHA256
            if Path(path) == source_path
            else fit.EXPECTED_SNAPSHOT_ARCHIVE_SHA256
        ),
    )
    args = Namespace(
        source_fit_dir=source_dir,
        snapshot_dir=snapshot_dir,
        output_dir=output_dir,
    )

    assert fit.seed_resume(args) == 21
    seeded = torch.load(output_dir / "resume.pt", map_location="cpu", weights_only=True)

    assert seeded["accepted_updates"] == 21
    assert seeded["experiment"] == fit.EXPERIMENT
    assert seeded["controller"]["buffer"].item() == 4.0
    for name in fit.continuation.joint.PARAMETER_FAMILIES:
        assert torch.equal(seeded["controller"][name], _snapshot()["candidate_parameters"][name])
    assert fit.audit.trees_equal(seeded["optimizer"], _snapshot()["pending_optimizer_state"])
    assert [entry["update"] for entry in seeded["history"]] == [19, 20, 21]
    fit.validate_snapshot_history(seeded)


def test_existing_resume_requires_snapshot_history(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    torch.save(
        {
            "experiment": fit.EXPERIMENT,
            "protocol_commit": fit.PROTOCOL_COMMIT,
            "accepted_updates": 22,
            "history": [],
        },
        output_dir / "resume.pt",
    )

    try:
        fit.seed_resume(Namespace(output_dir=output_dir))
    except SystemExit as error:
        assert "unique accepted snapshot update" in str(error)
    else:
        raise AssertionError("resume without snapshot provenance was accepted")


def test_advanced_resume_requires_persisted_snapshot_load_control(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    payload = _source_resume()
    payload.update(
        {
            "experiment": fit.EXPERIMENT,
            "protocol_commit": fit.PROTOCOL_COMMIT,
            "accepted_updates": 22,
        }
    )
    payload["history"] = [
        {
            "update": 21,
            "accepted": True,
            "acceptance_kind": "qualified_snapshot_repaired",
            "snapshot_provenance": {
                "snapshot_report_sha256": fit.EXPECTED_SNAPSHOT_REPORT_SHA256,
                "snapshot_archive_sha256": fit.EXPECTED_SNAPSHOT_ARCHIVE_SHA256,
                "candidate_parameter_sha256": fit.EXPECTED_SNAPSHOT_PARAMETER_SHA256,
                "pending_optimizer_sha256": fit.EXPECTED_PENDING_OPTIMIZER_SHA256,
            },
        },
        {"update": 22, "accepted": True, "projection": {}},
    ]
    torch.save(payload, output_dir / "resume.pt")

    try:
        fit.seed_resume(Namespace(output_dir=output_dir))
    except SystemExit as error:
        assert "passing pre-update-22 snapshot control" in str(error)
    else:
        raise AssertionError("advanced resume without persisted snapshot control was accepted")


class TinyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([1.0]))
        self.bias = torch.nn.Parameter(torch.tensor([2.0]))
        self.raw_time_constant = torch.nn.Parameter(torch.tensor([3.0]))


def test_snapshot_load_control_checks_parameters_optimizer_and_metrics(monkeypatch) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    for parameter in controller.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    parameter_hash = fit.audit.semantic_sha256(fit.continuation.joint._copy_parameters(controller))
    optimizer_hash = fit.audit.semantic_sha256(optimizer.state_dict())
    monkeypatch.setattr(fit, "EXPECTED_SNAPSHOT_PARAMETER_SHA256", parameter_hash)
    monkeypatch.setattr(fit, "EXPECTED_PENDING_OPTIMIZER_SHA256", optimizer_hash)
    monkeypatch.setattr(
        fit,
        "_SNAPSHOT_REFERENCE",
        {
            "source_training_metrics": {"metric": 1.0},
            "selected_training_metrics": {"metric": 2.0},
        },
    )

    control = fit.snapshot_load_control(
        controller,
        optimizer,
        {"metric": 1.0},
        {"metric": 2.0},
    )
    failed = fit.snapshot_load_control(
        controller,
        optimizer,
        {"metric": 1.0},
        {"metric": 3.0},
    )

    assert control["pass"] is True
    assert control["checked_before_update_22_proposal_generation"] is True
    assert failed["pass"] is False


def test_failed_snapshot_load_control_generates_no_proposal(monkeypatch) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    monkeypatch.setattr(fit.continuation, "_UPDATE_NUMBER", 21)
    monkeypatch.setattr(fit, "snapshot_load_control", lambda *args: {"pass": False})
    called = []
    monkeypatch.setattr(
        fit.continuation,
        "make_projected_proposal",
        lambda *args, **kwargs: called.append(True),
    )

    displacement, gradients, projection = fit.make_projected_proposal(
        controller,
        optimizer,
        {},
        {},
        object(),
        object(),
        {},
        1.0,
        device=torch.device("cpu"),
    )

    assert called == []
    assert projection["pass_after_parameter_bounds"] is False
    assert projection["optimizer_step_calls"] == 0
    assert projection["fresh_proposal_inputs_generated"] is False
    assert all(torch.count_nonzero(value) == 0 for value in displacement.values())
    assert all(torch.count_nonzero(value) == 0 for value in gradients.values())


def test_protocol_starts_fresh_generation_at_22_and_stops_at_total_50() -> None:
    protocol = fit.protocol_manifest()

    assert protocol["protocol_commit"] == "36e573e"
    assert protocol["starting_resume"]["accepted_snapshot_update"] == 21
    assert protocol["update_21"]["primary_or_repair_recomputed"] is False
    assert protocol["before_update_22"]["candidate_parameter_hash_exact"] is True
    assert protocol["updates_22_through_50"]["old_projector_fallback"] is False
    assert protocol["maximum_accepted_updates"] == 50
    assert protocol["development_totals"] == [30, 40, 50]
    assert protocol["development_preservation_failure_rolls_back_transaction"] is True
    assert "authorizing_step_audit" not in protocol


def test_development_failure_rollback_restores_controller_adam_and_history() -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    for parameter in controller.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    parameters_before = fit.continuation.joint._copy_parameters(controller)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    metrics_before = {"endpoint_damping_nrmse": 1.25}

    optimizer.zero_grad(set_to_none=True)
    for parameter in controller.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    optimizer.step()
    history = [{"update": 30, "accepted": True, "accepted_scale": 0.5}]
    rollback = {
        "attempted_update": 30,
        "accepted_updates_before": 29,
        "parameters": parameters_before,
        "optimizer": optimizer_before,
        "training_metrics": metrics_before,
        "history_length_before": 0,
    }

    accepted, metrics = fit.base.restore_development_transaction(
        controller,
        optimizer,
        history,
        rollback,
        attempted_update=30,
    )

    assert accepted == 29
    assert metrics == metrics_before
    assert all(
        torch.equal(getattr(controller, name).detach(), parameters_before[name])
        for name in fit.continuation.joint.PARAMETER_FAMILIES
    )
    assert fit.audit.trees_equal(optimizer.state_dict(), optimizer_before)
    assert history[-1]["accepted"] is False
    assert history[-1]["accepted_before_development"] is True
    assert history[-1]["rolled_back_after_development_preservation_failure"] is True
    assert history[-1]["stop_reason"] == "scheduled development preservation failed"


def test_development_failure_rollback_requires_complete_transaction() -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)

    try:
        fit.base.restore_development_transaction(
            controller,
            optimizer,
            [{"update": 30, "accepted": True}],
            None,
            attempted_update=30,
        )
    except RuntimeError as error:
        assert "lacks its rollback transaction" in str(error)
    else:
        raise AssertionError("missing development rollback transaction was accepted")


def test_pending_resume_round_trips_development_rollback(tmp_path: Path) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    rollback = {
        "attempted_update": 30,
        "accepted_updates_before": 29,
        "parameters": fit.continuation.joint._copy_parameters(controller),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "training_metrics": {"endpoint_damping_nrmse": 1.25},
        "history_length_before": 0,
    }
    path = tmp_path / "resume.pt"
    fit.base.save_resume(
        path,
        controller,
        optimizer,
        source_checkpoint_sha256="checkpoint",
        accepted_updates=30,
        history=[{"update": 30, "accepted": True}],
        development_history=[],
        preflight={"pass": True},
        run_state="development_pending",
        development_rollback=rollback,
    )
    loaded_controller = TinyController()
    loaded_optimizer = torch.optim.Adam(loaded_controller.parameters(), lr=0.01)

    accepted, history, _, _, metadata = fit.base.load_resume(
        path,
        loaded_controller,
        loaded_optimizer,
        source_checkpoint_sha256="checkpoint",
    )

    assert accepted == 30
    assert history == [{"update": 30, "accepted": True}]
    assert metadata["run_state"] == "development_pending"
    assert fit.audit.trees_equal(metadata["development_rollback"], rollback)


def test_authorization_failure_restores_all_wrapper_globals(monkeypatch) -> None:
    sentinel = {"original": True}
    original_update = fit.continuation._UPDATE_NUMBER
    monkeypatch.setattr(fit, "_SNAPSHOT_CPU", sentinel)
    monkeypatch.setattr(fit, "parse_args", lambda: Namespace())
    monkeypatch.setattr(fit, "validate_args", lambda args: None)
    monkeypatch.setattr(fit, "locked_input_paths", lambda args: {})

    def reject(args: Namespace) -> None:
        fit._SNAPSHOT_CPU = {"mutated": True}
        fit.continuation._UPDATE_NUMBER = 999
        raise SystemExit("injected authorization failure")

    monkeypatch.setattr(fit, "validate_authorizations", reject)

    try:
        fit.main()
    except SystemExit as error:
        assert "injected authorization failure" in str(error)
    else:
        raise AssertionError("injected authorization failure did not propagate")

    assert fit._SNAPSHOT_CPU is sentinel
    assert fit.continuation._UPDATE_NUMBER == original_update
