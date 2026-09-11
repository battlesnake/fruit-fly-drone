from __future__ import annotations

import copy
import sys
from argparse import Namespace
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_full_native_d_first_fp64_corrected as fit  # noqa: E402


class TinyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([1.0]))
        self.bias = torch.nn.Parameter(torch.tensor([2.0]))
        self.raw_time_constant = torch.nn.Parameter(torch.tensor([3.0]))


def _source_resume() -> dict[str, object]:
    return {
        "experiment": fit.corrected.EXPERIMENT,
        "protocol_commit": fit.corrected.PROTOCOL_COMMIT,
        "accepted_updates": 20,
        "history": [
            {"update": 19, "accepted": True},
            {"update": 20, "accepted": True},
            {"update": 21, "accepted": False, "stop_reason": "projection"},
        ],
        "development_history": [
            {"update": 10},
            {"update": 20},
        ],
        "preflight": {"pass": True},
        "run_state": "stopped",
        "qualification": None,
    }


def test_seed_resume_removes_only_terminal_rejected_update(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    source_path = source_dir / "resume.pt"
    torch.save(_source_resume(), source_path)
    monkeypatch.setattr(fit, "EXPECTED_SOURCE_RESUME_SHA256", "expected")
    monkeypatch.setattr(
        fit.responsibility,
        "file_sha256",
        lambda path: "expected" if Path(path) == source_path else "output",
    )
    args = Namespace(source_fit_dir=source_dir, output_dir=output_dir)

    accepted = fit.seed_resume(args)
    seeded = torch.load(output_dir / "resume.pt", map_location="cpu", weights_only=True)

    assert accepted == 20
    assert seeded["experiment"] == fit.EXPERIMENT
    assert seeded["protocol_commit"] == fit.PROTOCOL_COMMIT
    assert [entry["update"] for entry in seeded["history"]] == [19, 20]
    assert seeded["run_state"] == "active"
    assert seeded["qualification"] is None


def test_existing_resume_must_match_new_protocol(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    torch.save(
        {
            "experiment": fit.EXPERIMENT,
            "protocol_commit": fit.PROTOCOL_COMMIT,
            "accepted_updates": 31,
        },
        output_dir / "resume.pt",
    )

    assert fit.seed_resume(Namespace(output_dir=output_dir)) == 31


def test_update21_candidate_reproduction_failure_restores_transaction(monkeypatch) -> None:
    student = object()
    current = {"edge_magnitude": torch.tensor([1.0])}
    trials = [{"decision": {"pass": True, "reasons": []}}]
    restored = []
    monkeypatch.setattr(fit, "_UPDATE_NUMBER", 21)
    monkeypatch.setattr(
        fit.corrected,
        "find_safe_trial",
        lambda *args, **kwargs: (0.0625, {"metric": 1.0}, trials),
    )
    monkeypatch.setattr(
        fit.joint,
        "_copy_parameters",
        lambda controller: {"edge_magnitude": torch.tensor([2.0])},
    )
    monkeypatch.setattr(fit.audit, "semantic_sha256", lambda value: "wrong")
    monkeypatch.setattr(
        fit.corrected,
        "_restore_rejected_transaction",
        lambda controller, parameters: restored.append((controller, parameters)),
    )
    monkeypatch.setattr(
        fit,
        "_STEP_AUDIT_REPORT",
        {"selected_training_metrics": {"metric": 1.0}},
    )

    selected, metrics, returned_trials = fit.find_safe_trial(
        student,
        current,
        {},
        {},
        {},
        {},
        object(),
        object(),
        {},
        1.0,
        device=torch.device("cpu"),
    )

    assert selected is None
    assert metrics is None
    assert returned_trials[-1]["registered_update_21_reproduction"]["pass"] is False
    assert restored == [(student, current)]


def test_numerical_exception_restores_and_persists_stopped_resume(monkeypatch) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    parameters = fit.joint._copy_parameters(controller)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    with torch.no_grad():
        controller.bias.add_(4.0)
    history = []
    saved = []
    monkeypatch.setattr(fit.base, "PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED", True)
    monkeypatch.setattr(
        fit.base,
        "save_resume",
        lambda *args, **kwargs: saved.append(kwargs),
    )

    fit.base.restore_and_persist_numerical_exception(
        controller,
        optimizer,
        parameters,
        optimizer_state,
        RuntimeError("injected"),
        stage="proposal",
        update_number=22,
        resume_path=Path("resume.pt"),
        source_checkpoint_sha256="checkpoint",
        accepted_updates=21,
        history=history,
        development_history=[],
        preflight={"pass": True},
    )

    assert torch.equal(controller.bias.detach(), parameters["bias"])
    assert history[-1]["accepted"] is False
    assert history[-1]["stop_reason"] == "numerical exception during proposal"
    assert saved[-1]["run_state"] == "stopped"
    assert saved[-1]["accepted_updates"] == 21


def test_development_preservation_failure_is_terminal_and_resume_safe(monkeypatch) -> None:
    monkeypatch.setattr(fit.base, "DEVELOPMENT_PRESERVATION_REQUIRED", True)
    preservation = {"pass": False}
    terminal = {"pass": False}
    history = [
        {
            "update": 30,
            "preservation": preservation,
            "mandatory_update_50_gate": None,
            "terminal": terminal,
        }
    ]

    assert (
        fit.base.development_outcome_stop_reason(preservation, None, terminal)
        == "scheduled development preservation failed"
    )
    assert fit.base.resumed_stop_state(30, [], history) == (
        "scheduled development preservation failed",
        False,
    )


def test_protocol_stops_at_existing_total_update_50_gate() -> None:
    protocol = fit.protocol_manifest()

    assert protocol["protocol_commit"] == "ba6fcc3"
    assert protocol["maximum_accepted_updates"] == 50
    assert protocol["accepted_update_budget_is_total_not_additional"] is True
    assert protocol["development_totals"] == [30, 40, 50]
    assert protocol["development_preservation_failure_is_terminal"] is True
    assert protocol["scheduled_candidate_persisted_before_development"] is True
    assert protocol["numerical_exceptions_restore_and_persist_stopped_resume"] is True
    assert protocol["mandatory_gate_total"] == 50
    assert protocol["update_21"]["jacobians_regenerated"] is False
    assert protocol["updates_22_through_50"]["old_projector_fallback"] is False
    assert protocol["repair_arithmetic_unchanged"] is True
