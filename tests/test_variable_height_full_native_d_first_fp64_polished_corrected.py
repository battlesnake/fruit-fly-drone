from __future__ import annotations

import copy
import sys
from argparse import Namespace
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_full_native_d_first_fp64_polished_corrected as fit  # noqa: E402


def _source_resume() -> dict[str, object]:
    return {
        "experiment": fit.source_fit.EXPERIMENT,
        "protocol_commit": fit.source_fit.PROTOCOL_COMMIT,
        "accepted_updates": 24,
        "controller": {
            "edge_magnitude": torch.tensor([1.0]),
            "bias": torch.tensor([2.0]),
            "raw_time_constant": torch.tensor([3.0]),
        },
        "optimizer": {"state": {}, "param_groups": []},
        "history": [
            {"update": 23, "accepted": True},
            {"update": 24, "accepted": True},
            {"update": 25, "accepted": False},
        ],
        "development_history": [{"update": 20}],
        "preflight": {"pass": True},
        "run_state": "stopped",
        "qualification": None,
    }


def test_seed_resume_removes_only_rejected_update25(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    source_path = source_dir / "resume.pt"
    torch.save(_source_resume(), source_path)
    monkeypatch.setattr(
        fit.responsibility,
        "file_sha256",
        lambda path: fit.EXPECTED_SOURCE_RESUME_SHA256,
    )

    accepted = fit.seed_resume(Namespace(source_fit_dir=source_dir, output_dir=output_dir))
    seeded = torch.load(output_dir / "resume.pt", map_location="cpu", weights_only=True)

    assert accepted == 24
    assert seeded["experiment"] == fit.EXPERIMENT
    assert seeded["protocol_commit"] == fit.PROTOCOL_COMMIT
    assert [entry["update"] for entry in seeded["history"]] == [23, 24]
    assert seeded["run_state"] == "active"
    assert seeded["development_rollback"] is None
    fit.validate_seed_history(seeded)


def test_advanced_resume_requires_passing_frozen_update25_control() -> None:
    payload = _source_resume()
    payload.update(
        {
            "experiment": fit.EXPERIMENT,
            "protocol_commit": fit.PROTOCOL_COMMIT,
            "accepted_updates": 25,
            "seed_provenance": {
                "source_report_sha256": fit.EXPECTED_SOURCE_REPORT_SHA256,
                "source_resume_sha256": fit.EXPECTED_SOURCE_RESUME_SHA256,
                "frozen_update25_archive_sha256": fit.EXPECTED_FROZEN_ARCHIVE_SHA256,
                "polished_audit_report_sha256": fit.EXPECTED_POLISHED_REPORT_SHA256,
            },
            "history": [{"update": 25, "accepted": True, "projection": {}}],
        }
    )

    try:
        fit.validate_seed_history(payload)
    except SystemExit as error:
        assert "passing frozen update-25 preflight" in str(error)
    else:
        raise AssertionError("advanced resume without frozen update-25 control was accepted")


class TinyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([1.0]))
        self.bias = torch.nn.Parameter(torch.tensor([2.0]))
        self.raw_time_constant = torch.nn.Parameter(torch.tensor([3.0]))


def test_frozen_preflight_requires_parameters_optimizer_and_metrics(monkeypatch) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    parameters = fit.joint._copy_parameters(controller)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    monkeypatch.setattr(
        fit, "EXPECTED_UPDATE24_PARAMETER_SHA256", fit.audit.semantic_sha256(parameters)
    )
    monkeypatch.setattr(
        fit, "EXPECTED_UPDATE24_OPTIMIZER_SHA256", fit.audit.semantic_sha256(optimizer_state)
    )
    monkeypatch.setattr(
        fit,
        "_FROZEN_UPDATE25_CPU",
        {"current_parameters": parameters, "optimizer_before": optimizer_state},
    )
    monkeypatch.setattr(
        fit,
        "_FROZEN_REFERENCE",
        {
            "source_training_metrics": {"metric": 1.0},
            "current_training_metrics": {"metric": 2.0},
        },
    )

    passing = fit.frozen_update25_preflight(controller, optimizer, {"metric": 1.0}, {"metric": 2.0})
    failing = fit.frozen_update25_preflight(controller, optimizer, {"metric": 1.0}, {"metric": 3.0})

    assert passing["pass"] is True
    assert passing["adam_step_called"] is False
    assert failing["pass"] is False


def test_frozen_preflight_normalizes_live_cuda_style_trees_to_cpu(monkeypatch) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    parameters = fit.joint._copy_parameters(controller)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    monkeypatch.setattr(
        fit, "EXPECTED_UPDATE24_PARAMETER_SHA256", fit.audit.semantic_sha256(parameters)
    )
    monkeypatch.setattr(
        fit,
        "EXPECTED_UPDATE24_OPTIMIZER_SHA256",
        fit.audit.semantic_sha256(optimizer_state),
    )
    monkeypatch.setattr(
        fit,
        "_FROZEN_UPDATE25_CPU",
        {"current_parameters": parameters, "optimizer_before": optimizer_state},
    )
    monkeypatch.setattr(
        fit,
        "_FROZEN_REFERENCE",
        {
            "source_training_metrics": {"metric": 1.0},
            "current_training_metrics": {"metric": 2.0},
        },
    )
    original_to_cpu = fit.audit25.to_cpu_tree
    normalized_devices: list[set[torch.device]] = []

    def recording_to_cpu(value):
        normalized = original_to_cpu(value)
        devices: set[torch.device] = set()

        def visit(item):
            if isinstance(item, torch.Tensor):
                devices.add(item.device)
            elif isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(normalized)
        normalized_devices.append(devices)
        return normalized

    monkeypatch.setattr(fit.audit25, "to_cpu_tree", recording_to_cpu)

    report = fit.frozen_update25_preflight(controller, optimizer, {"metric": 1.0}, {"metric": 2.0})

    assert report["pass"] is True
    assert len(normalized_devices) >= 2
    assert all(all(device.type == "cpu" for device in devices) for devices in normalized_devices)


def test_seed_provenance_survives_base_resume_save(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    torch.save(_source_resume(), source_dir / "resume.pt")
    monkeypatch.setattr(
        fit.responsibility,
        "file_sha256",
        lambda path: fit.EXPECTED_SOURCE_RESUME_SHA256,
    )
    fit.seed_resume(Namespace(source_fit_dir=source_dir, output_dir=output_dir))
    seeded = torch.load(output_dir / "resume.pt", map_location="cpu", weights_only=True)
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    monkeypatch.setattr(fit.base, "EXPERIMENT", fit.EXPERIMENT)
    monkeypatch.setattr(fit.base, "PROTOCOL_COMMIT", fit.PROTOCOL_COMMIT)

    fit.base.save_resume(
        output_dir / "resume.pt",
        controller,
        optimizer,
        source_checkpoint_sha256="checkpoint",
        accepted_updates=25,
        history=[
            {
                "update": 25,
                "accepted": True,
                "projection": {"frozen_update25_preflight": {"pass": True}},
            }
        ],
        development_history=[],
        preflight=seeded["preflight"],
    )

    assert fit.seed_resume(Namespace(source_fit_dir=source_dir, output_dir=output_dir)) == 25


def test_frozen_update25_uses_archive_without_fresh_gradient_or_adam(monkeypatch) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    current = fit.joint._copy_parameters(controller)
    displacement = {name: -torch.ones_like(value) for name, value in current.items()}
    gradients = {name: torch.ones_like(value) for name, value in current.items()}
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    archive = {
        "constraint_specs": [],
        "constraint_rows": {},
        "raw_damping_gradient": gradients,
        "raw_displacement": displacement,
        "gradient_norm_before_clipping": 1.0,
        "optimizer_after": optimizer_after,
    }
    monkeypatch.setattr(fit, "_UPDATE_NUMBER", 24)
    monkeypatch.setattr(fit, "_FROZEN_UPDATE25_CPU", archive)
    monkeypatch.setattr(
        fit,
        "frozen_update25_preflight",
        lambda *args, **kwargs: {
            "pass": True,
            "fresh_proposal_inputs_generated": False,
            "adam_step_called": False,
        },
    )

    def unexpected_fresh_proposal(*args, **kwargs):
        raise AssertionError("frozen update 25 regenerated proposal inputs")

    def unexpected_adam_step(*args, **kwargs):
        raise AssertionError("frozen update 25 called Adam.step")

    monkeypatch.setattr(fit.continuation, "_fresh_proposal_inputs", unexpected_fresh_proposal)
    monkeypatch.setattr(optimizer, "step", unexpected_adam_step)
    monkeypatch.setattr(
        fit.corrected,
        "optimizer_step_transaction",
        lambda before, after: {"pass": True},
    )
    monkeypatch.setattr(
        fit.polished,
        "polished_bound_aware_projection",
        lambda *args, **kwargs: (
            displacement,
            {
                "pass": True,
                "rounds": [
                    {
                        "free_projection": {
                            "pass": True,
                            "slsqp_support_polished_reference": {"pass": True},
                        }
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(
        fit.canonical,
        "materialize_authoritative_candidate",
        lambda student, base, proposal: (
            {name: base[name] + proposal[name] for name in base},
            {name: proposal[name].double() for name in proposal},
            {"pass": True},
        ),
    )
    monkeypatch.setattr(fit.audit, "parameter_bounds_report", lambda value: {"pass": True})
    monkeypatch.setattr(fit.audit, "linearized_constraint_violations", lambda *args, **kwargs: {})
    monkeypatch.setattr(fit.audit, "maximum_linearized_violation", lambda value: (0.0, True))
    monkeypatch.setattr(
        fit.frozen,
        "complete_projection_primal_objective",
        lambda *args, **kwargs: fit.EXPECTED_UPDATE25_PROJECTION_OBJECTIVE,
    )
    monkeypatch.setattr(
        fit,
        "EXPECTED_UPDATE25_PROJECTED_DISPLACEMENT_SHA256",
        fit.audit.semantic_sha256(displacement),
    )
    monkeypatch.setattr(
        fit,
        "EXPECTED_UPDATE25_PENDING_OPTIMIZER_SHA256",
        fit.audit.semantic_sha256(optimizer_after),
    )
    monkeypatch.setattr(
        fit,
        "_update25_directional_finite_difference",
        lambda *args, **kwargs: {"pass": True},
    )

    _, returned_gradients, projection = fit.make_projected_proposal(
        controller,
        optimizer,
        {},
        {},
        None,
        None,
        {},
        1.0,
        device=torch.device("cpu"),
    )

    assert projection["pass_after_parameter_bounds"] is True
    assert projection["frozen_update25_preflight"]["pass"] is True
    assert projection["update25_projection_reproduction"]["pass"] is True
    assert fit.audit.trees_equal(returned_gradients, gradients)


def test_protocol_loads_frozen_update25_and_keeps_total50_gate() -> None:
    protocol = fit.protocol_manifest()

    assert protocol["protocol_commit"] == "d1e5c82"
    assert protocol["update_25"]["adam_step_called_again"] is False
    assert protocol["update_25"]["ordinary_then_existing_eligible_repair"] is True
    assert protocol["updates_26_through_50"]["old_projector_fallback"] is False
    assert protocol["maximum_accepted_updates"] == 50
    assert protocol["development_totals"] == [30, 40, 50]
    assert protocol["development_preservation_failure_rolls_back_transaction"] is True
    assert protocol["starting_accepted_update"] == 24
    assert "L-BFGS-B" not in protocol["constraint_projection"]["solver"]
    assert protocol["constraint_projection"]["old_projector_fallback"] is False
    assert protocol["optimizer"]["adam_step_per_fresh_attempt_updates_26_plus"] == 1
    assert protocol["optimizer"]["adam_step_executed_in_continuation_for_frozen_update_25"] == 0
    assert "adam_step_per_attempt" not in protocol["optimizer"]
    assert "update_21" not in protocol
    assert "before_update_22" not in protocol
    assert "updates_22_through_50" not in protocol
