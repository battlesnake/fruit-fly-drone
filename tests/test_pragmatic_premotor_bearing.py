from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pragmatic_premotor_bearing as bearing  # noqa: E402

from flydrone.gate import AnnularGate, GateConfig, render_annular_gates_rgb  # noqa: E402
from flydrone.hover import HoverConfig, QuadState  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def state_at(positions):
    position = torch.tensor(positions, dtype=torch.float32)
    return QuadState(
        position,
        torch.zeros_like(position),
        torch.zeros_like(position),
        torch.zeros_like(position),
        torch.zeros(len(position), 4),
        torch.zeros_like(position),
    )


def bank(kind="native", steps=100, episodes=6):
    rng = torch.Generator().manual_seed(12)
    fields = [torch.zeros(steps, episodes, n) for n in (3, 3, 3, 3, 4, 3)]
    fields[0][..., 2] = 1.0
    gates = (
        AnnularGate(torch.tensor([[3.0, 0.0, 1.0]]).expand(episodes, -1), torch.zeros(episodes)),
    )
    roles = torch.zeros(steps, episodes, dtype=torch.long)
    roles[75:] = 1
    physical = bearing.replay.ReplayBank(
        tuple(fields),
        gates,
        roles,
        torch.ones(steps, episodes, dtype=torch.bool),
        torch.zeros(steps, episodes, 4),
        kind,
        1,
    )
    features = torch.randn(steps, episodes, 3, generator=rng) * 0.1
    angle = features[..., 0] * 3
    return bearing.BearingBank(
        physical,
        features,
        torch.stack((angle.sin(), angle.cos()), -1),
        torch.ones(steps, episodes, dtype=torch.bool),
        {},
    )


def test_mask_uses_annotations_and_freezes_internal_and_all_outgoing_edges(tmp_path):
    graph, annotations = tmp_path / "graph.npz", tmp_path / "annotations.feather"
    np.savez(
        graph,
        node_ids=np.arange(7) + 100,
        edge_pre=[0, 1, 2, 3, 4, 5, 2],
        edge_post=[2, 3, 3, 2, 6, 6, 6],
        output_pool_indices=[6],
        output_pool_offsets=[0, 1, 1],
    )
    # Direct motor parents: 2 eligible, 4 ascending, 5 sensory. 3 is only two-hop.
    feather.write_feather(
        pa.table(
            dict(
                bodyId=np.arange(7) + 100,
                superclass=[
                    "vnc_intrinsic",
                    "vnc_intrinsic",
                    "descending_neuron",
                    "vnc_intrinsic",
                    "ascending_neuron",
                    "vnc_sensory",
                    "vnc_motor",
                ],
            )
        ),
        annotations,
    )
    mask, nodes, manifest = bearing.premotor_mask(graph, annotations, torch.device("cpu"))
    assert nodes.tolist() == [2]
    assert mask.tolist() == [True, False, False, True, False, False, False]
    assert manifest["direct_roll_parents"] == 3
    assert manifest["trainable_edges"] == 2
    # Make 3 a direct parent too: 2->3 and 3->2 are both frozen internal edges.
    with np.load(graph) as loaded:
        values = dict(loaded)
    values.update(edge_pre=[0, 1, 2, 3, 4, 5, 2, 3], edge_post=[2, 3, 3, 2, 6, 6, 6, 6])
    np.savez(graph, **values)
    mask, nodes, manifest = bearing.premotor_mask(graph, annotations, torch.device("cpu"))
    assert nodes.tolist() == [2, 3]
    assert mask.tolist() == [True, True, False, False, False, False, False, False]
    assert manifest["frozen_internal_edges"] == 2


def test_bearing_labels_rotate_to_body_frame_and_pixel_mask_uses_actual_current_colour():
    state = state_at([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    gates = (AnnularGate(torch.tensor([[3.0, 1.0, 1.0], [3.0, 1.0, 1.0]]), torch.zeros(2)),)
    roles = torch.zeros(2, dtype=torch.long)
    labels = bearing.bearing_labels(state, gates, roles)
    assert torch.allclose(labels[0], torch.tensor([1 / math.sqrt(10), 3 / math.sqrt(10)]))
    state.euler[1, 2] = math.pi / 2
    labels = bearing.bearing_labels(state, gates, roles)
    assert torch.allclose(labels[1], torch.tensor([-3 / math.sqrt(10), 1 / math.sqrt(10)]))
    state.euler.zero_()
    image = render_annular_gates_rgb(
        state, gates, current_gate_index=roles, camera=CameraSpec(80, 50, 125)
    )
    assert bearing.visible_current_gate(image).tolist() == [True, True]
    dark = render_annular_gates_rgb(
        state, gates, current_gate_index=roles + 1, camera=CameraSpec(80, 50, 125)
    )
    assert not bool(bearing.visible_current_gate(dark).any())
    other = image[:, [1, 0, 2]]  # Red-green swap cannot become current colour.
    assert not bool(bearing.visible_current_gate(other).any())


def test_strata_holdout_is_whole_pair_and_windows_exclude_failure_and_invisible_labels():
    data = bank()
    data.replay.active[10:, 0] = False
    data.visible[:, 1] = False
    train = bearing.strata(data, unroll=20)
    heldout = bearing.strata(data, unroll=20, heldout=True)
    assert all(not len(v) or set(v[:, 1].tolist()) <= {0, 1, 2, 3} for v in train.values())
    assert all(not len(v) or set(v[:, 1].tolist()) <= {4, 5} for v in heldout.values())
    assert train["native/launch/0"][:, 1].unique().tolist() == [2]
    assert train["native/launch/1"][:, 1].unique().tolist() == [3]
    groups = []
    for i in range(6):
        _, rows, starts, info = bearing.choose_windows(
            [data, bank("roll-assisted")], torch.Generator().manual_seed(i), i
        )
        assert rows[0] % 2 == 0 and rows[1] % 2 == 1
        assert max(rows) < 4 and len(starts) == 2
        groups.append((info["kind"], info["phase"]))
        assert info["available_paired_groups"] == 6
    assert len(set(groups)) == 6

    data.replay.active[75:, -1] = False
    _, _, _, info = bearing.choose_windows(
        [data, bank("roll-assisted")],
        torch.Generator().manual_seed(1),
        0,
        heldout=True,
    )
    assert info["available_paired_groups"] == 5
    assert info["coverage"]["native/later-gates/1"] == 0


def test_head_and_normalizers_frozen_with_scalar_combined_source_mse_and_no_holdout_leak():
    data = bank()
    head, report = bearing.fit_bearing_head([data], device=torch.device("cpu"), seed=5)
    assert list(head.parameters()) == []
    assert head.source_mse.ndim == 0 and float(head.source_mse) >= 1e-4 - 1e-9
    data.labels[:, -2:] *= -10
    second, changed = bearing.fit_bearing_head([data], device=torch.device("cpu"), seed=5)
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in head.state_dict().items())
    assert changed["source_heldout_mse"] != report["source_heldout_mse"]
    activity = data.source_features[:2].clone().requires_grad_(True)
    head(activity).sum().backward()
    assert activity.grad is not None and activity.grad.norm() > 0


def test_reuse_frozen_head_requires_same_source_and_neuron_order(tmp_path):
    import json

    import train_pragmatic_premotor_bearing as driver

    head, fit = bearing.fit_bearing_head([bank()], device=torch.device("cpu"), seed=3)
    torch.save(head.state_dict(), tmp_path / "training-only-bearing-head.pt")
    source = tmp_path / "source.pt"
    manifest = dict(selected_body_ids=[10, 20, 30])
    report = dict(
        status="complete",
        arguments=dict(checkpoint=str(source)),
        native_path_manifest=dict(representation_stage=manifest),
        bearing_head_fit=fit,
    )
    (tmp_path / "report.json").write_text(json.dumps(report))
    loaded, metrics = driver.reuse_bearing_head(
        tmp_path, checkpoint=source, representation_manifest=manifest, device=torch.device("cpu")
    )
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in head.state_dict().items())
    assert not metrics["refitted"] and metrics["fit_metrics_are_from_original_run"]
    with pytest.raises(ValueError, match="identity/order"):
        driver.reuse_bearing_head(
            tmp_path,
            checkpoint=source,
            representation_manifest=dict(selected_body_ids=[20, 10, 30]),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="source checkpoint"):
        driver.reuse_bearing_head(
            tmp_path,
            checkpoint=tmp_path / "different.pt",
            representation_manifest=manifest,
            device=torch.device("cpu"),
        )


class TinyActor(torch.nn.Module):
    def __init__(self, output=None):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
        self.output = None if output is None else torch.tensor(output)
        self.prefix_calls = self.gradient_calls = 0

    def initial_state(self, count, **kwargs):
        return torch.zeros(count, 1, **kwargs)

    def forward(self, image, attitude, neural):
        assert image.ndim == 4 and image.shape[1] == 3 and attitude.shape == (len(image), 2)
        if torch.is_grad_enabled():
            self.gradient_calls += 1
        else:
            self.prefix_calls += 1
        state = 0.8 * neural + self.edge_magnitude[0] * 0.1
        output = (
            state.tanh().expand(-1, 4)
            if self.output is None
            else self.output.expand(len(state), -1)
        )
        return output, state


def test_window_replays_current_full_prefix_then_only_window_gradients_and_label_mask(monkeypatch):
    data, actor = bank(), TinyActor()
    head = bearing.FrozenBearingHead(
        torch.zeros(1), torch.ones(1), torch.tensor([[1.0, 2.0]]), torch.zeros(2), torch.tensor(0.2)
    )
    monkeypatch.setattr(
        bearing.replay,
        "window_observations",
        lambda w, t, c, g: (torch.zeros(len(t), 3, 2, 2), torch.zeros(len(t), 2)),
    )
    options = dict(camera=CameraSpec(), gate_config=GateConfig(), unroll=4)
    data.visible[6, 0] = False
    loss, stats = bearing.bearing_window_loss(
        actor, head, torch.tensor([0]), data, [0, 1], [5, 9], **options
    )
    assert actor.prefix_calls == 19 and actor.gradient_calls == 4
    assert stats["visible_frames"] == 7 and stats["gradient_is_window_truncated"]
    assert len(stats["normalized_bearing_mse_by_branch"]) == 2
    assert sum(stats["normalized_bearing_mse_by_branch"]) / 2 == pytest.approx(
        stats["normalized_bearing_mse"]
    )
    loss.backward()
    assert actor.edge_magnitude.grad[0] != 0
    original = float(loss.detach())
    data.labels[6, 0] = 1e5  # Invisible frame must not enter the bearing loss.
    unchanged, _ = bearing.bearing_window_loss(
        actor, head, torch.tensor([0]), data, [0, 1], [5, 9], **options
    )
    assert float(unchanged.detach()) == original
    with torch.no_grad():
        actor.edge_magnitude[0] += 0.1
    changed, _ = bearing.bearing_window_loss(
        actor, head, torch.tensor([0]), data, [0, 1], [5, 9], **options
    )
    assert float(changed.detach()) != original


def test_representation_update_changes_only_existing_masked_magnitudes():
    actor = TinyActor()
    source = actor.edge_magnitude.detach().clone()
    optimizer = torch.optim.Adam([actor.edge_magnitude], lr=1e-3)
    stats = bearing.representation_update(
        actor,
        optimizer,
        torch.tensor([True, False]),
        lambda: (actor.edge_magnitude.square().sum(), {}),
    )
    assert actor.edge_magnitude[0] < source[0] and actor.edge_magnitude[1] == source[1]
    assert stats["changed_edges"] == 1 and stats["edge_delta_l2"] > 0
    assert actor.bias.grad is None


@pytest.mark.parametrize("kind", ["native", "roll-assisted"])
def test_collector_never_imitation_targets_teacher_and_ground_stops_supervision(monkeypatch, kind):
    motors = []

    class Legs:
        def __init__(self, config):
            self.step = 0

        def to(self, device):
            return self

        def __call__(self, motor, sticks):
            motors.append(motor.clone())
            return motor, sticks

    class Quad(Legs):
        def __call__(self, rc, state, mass):
            self.step += 1
            position = state.position.clone()
            if self.step == 2:
                position[0, 2] = 0.01
            return replace(state, position=position)

    monkeypatch.setattr(bearing.replay, "ForelegStickPlant", Legs)
    monkeypatch.setattr(bearing.replay, "DifferentiableQuad", Quad)
    monkeypatch.setattr(
        bearing.replay, "current_gate_roll_motor", lambda *a, **k: torch.full((4,), 0.9)
    )
    learner = TinyActor([0.1, 0.2, 0.3, 0.4])
    source = TinyActor([-0.1, -0.2, -0.3, -0.4])
    with torch.no_grad():
        data = bearing.collect_bearing_bank(
            learner,
            source,
            torch.tensor([0]),
            pairs=2,
            seed=1,
            kind=kind,
            camera=CameraSpec(32, 20, 125),
            config=HoverConfig(),
            gate_config=GateConfig(),
            seconds=0.06,
        )
    assert data.replay.active[:, 0].tolist() == [True, False, False]
    assert not bool(data.visible[1:, 0].any())
    assert data.metrics["ground_contacts"] == 1
    assert torch.equal(data.replay.target[0, 0], torch.tensor([0.1, -0.2, -0.3, -0.4]))
    assert torch.allclose(
        data.source_features[0, :, 0], torch.full((4,), 0.05 * (1 - 0.8**11)).tanh()
    )
    expected = [0.1, 0.2, 0.3, 0.4] if kind == "native" else [0.9, -0.2, -0.3, -0.4]
    assert torch.equal(motors[0][0], torch.tensor(expected))


@pytest.mark.parametrize("centered", [False, True])
def test_alternating_driver_refreshes_sink_after_representation_and_exports_only_native(
    monkeypatch,
    tmp_path,
    centered,
):
    import json
    from types import SimpleNamespace

    import train_pragmatic_premotor_bearing as driver

    args = SimpleNamespace(
        checkpoint=tmp_path / "source.pt",
        output_dir=tmp_path / "run",
        graph=tmp_path / "graph.npz",
        annotations=tmp_path / "annotations.feather",
        device="cpu",
        rounds=2,
        representation_pairs=2,
        representation_updates=1,
        learning_rate=0.001,
        reuse_head_run=None,
        center_premotor_inputs=centered,
        training_pairs=2,
        development_pairs=2,
        proposals=1,
        seed=20,
        noise_seed=40,
        development_seed=100,
    )
    actors = [TinyActor(), TinyActor()]
    for actor in actors:
        actor.register_buffer("edge_pre", torch.tensor([0, 0]))
        actor.register_buffer("edge_post", torch.tensor([0, 0]))
        actor.register_buffer("edge_sign", torch.ones(2))
    initial = [actor.edge_magnitude.detach().clone() for actor in actors]
    source = dict(
        hover_config=vars(HoverConfig()),
        gate_config=vars(GateConfig()),
        image_resolution=[32, 20],
        camera_hfov_degrees=125,
        controller=actors[1].state_dict(),
    )
    loaders = iter(actors)
    monkeypatch.setattr(driver, "parse_args", lambda: args)
    monkeypatch.setattr(driver.replay, "load_controller", lambda *a: (next(loaders), source))
    monkeypatch.setattr(
        driver.representation,
        "premotor_mask",
        lambda *a: (torch.tensor([True, False]), torch.tensor([0]), {}),
    )
    monkeypatch.setattr(driver.replay, "sample_two_gate_cases", lambda *a, **k: (None, None))
    collected, sink_entries, evaluated = [], [], []

    def collect(controller, original, nodes, **kwargs):
        assert controller is actors[0] and original is actors[1]
        assert not any(p.requires_grad for p in controller.parameters())
        assert torch.equal(original.edge_magnitude, initial[1])
        collected.append(kwargs["seed"])
        data = bank(kwargs["kind"])
        if kwargs["presynaptic_nodes"] is not None:
            data.input_moments = dict(
                nodes=torch.tensor([0]), sums=torch.full((6, 1), 0.2), counts=torch.ones(6)
            )
        return data

    monkeypatch.setattr(driver.representation, "collect_bearing_bank", collect)
    head_fits = []

    def fit(*a, **k):
        head_fits.append(True)
        return torch.nn.Linear(1, 2).requires_grad_(False), {}

    monkeypatch.setattr(driver.representation, "fit_bearing_head", fit)
    monkeypatch.setattr(driver.representation, "source_window_bearing", lambda *a, **k: {})
    monkeypatch.setattr(
        driver.representation,
        "choose_windows",
        lambda banks, *a, **k: (banks[0], [0, 1], [0, 0], {"available_paired_groups": 6}),
    )
    monkeypatch.setattr(
        driver.representation,
        "bearing_window_loss",
        lambda actor, *a, **k: (
            getattr(actor, "controller", actor).edge_magnitude.square().sum(),
            {},
        ),
    )

    class Sink:
        def __init__(self, controller):
            assert not any(p.requires_grad for p in controller.parameters())
            sink_entries.append(controller.edge_magnitude.detach().clone())
            self.edge_magnitude = torch.nn.Parameter(controller.edge_magnitude[1:].detach().clone())
            self.sink = SimpleNamespace(parents=torch.tensor([0]), manifest=lambda: {"edges": 1})

        @torch.no_grad()
        def compile_into(self, controller):
            controller.edge_magnitude[1:] = self.edge_magnitude

    data = SimpleNamespace(
        valid=torch.ones(2, 4, dtype=torch.bool), critic_features=torch.zeros(2, 4, 3), metrics={}
    )
    data.select = lambda *a: data
    monkeypatch.setattr(driver.ppo, "NativeRollSinkPolicy", Sink)
    monkeypatch.setattr(driver.ppo, "collect_policy_rollout", lambda *a, **k: data)
    monkeypatch.setattr(driver.ppo, "round_advantages", lambda *a, **k: (torch.ones(2, 4), {}))
    monkeypatch.setattr(driver.ppo, "fit_critic", lambda *a, **k: {})
    monkeypatch.setattr(driver.ppo, "verify_sink_recording", lambda *a: {})
    monkeypatch.setattr(driver.ppo, "verify_compiled_sink_policy", lambda *a, **k: {})

    def proposal(sink, *a):
        with torch.no_grad():
            sink.edge_magnitude.add_(0.04)
        return dict(accepted=True, proposed_edge_delta_l2=0.04, stop_round=False)

    monkeypatch.setattr(driver.ppo, "box_natural_actor_proposal", proposal)

    def evaluate(controller, *a, **k):
        evaluated.append(controller.edge_magnitude.detach().clone())
        rate = 0.25 * len(evaluated)
        return dict(
            clean_course_success_rate=rate,
            clean_course_negative_success_rate=rate,
            clean_course_positive_success_rate=rate,
            gates_before_failure_mean=2.0,
            ground_contact_rate=0.0,
            invalid_rate=0.0,
        )

    monkeypatch.setattr(driver.replay, "evaluate", evaluate)
    assert driver.main() == 0
    assert collected == [20, 21, 23, 24] and len(head_fits) == 1
    assert len(sink_entries) == 2 and sink_entries[0][0] < initial[0][0]
    assert sink_entries[1][1] == evaluated[1][1]  # Previous sink change survives upstream SGD.
    assert torch.equal(actors[1].edge_magnitude, initial[1])
    exported = torch.load(args.output_dir / "best-controller.pt", weights_only=True)
    assert torch.equal(exported["controller"]["edge_magnitude"], evaluated[-1])
    assert set(exported["controller"]) == set(actors[0].state_dict())
    assert not exported["auxiliary_head_is_deployed"]
    assert exported["native_path_manifest"]["outcome_stage"]["edges"] == 1
    assert "representation_stage" in exported["native_path_manifest"]
    assert exported["training_round"] == 2
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "complete" and report["selected_round"] == 2
    assert report["policy_revision"] == 4 and not report["goal_verified"]
    assert report["rounds"][1]["outcome_seed"] == 25
    if centered:
        assert exported["controller"]["bias"].item() == pytest.approx(
            0.2 * (initial[0][0] - evaluated[-1][0]).item()
        )
        assert report["source_input_mean"]["heldout_episodes_excluded"]
        assert not exported["native_path_manifest"]["representation_stage"][
            "biases_independently_optimized"
        ]


def test_completed_representation_row_cannot_acquire_ground_failure_from_padding(monkeypatch):
    original_sampler = bearing.replay.sample_two_gate_cases

    def sample(*a, **kwargs):
        cases, _ = original_sampler(*a, **kwargs)
        cases.state.position[:] = torch.tensor([0.0, 0.0, 1.0])
        gate = AnnularGate(torch.tensor([[1.0, 0.0, 1.0]]).expand(4, -1), torch.zeros(4))
        return cases, (gate,)

    class Quad:
        def __init__(self, config):
            self.step = 0

        def to(self, device):
            return self

        def __call__(self, rc, state, mass):
            self.step += 1
            position = state.position.clone()
            position[0, 0] = 1.1  # Row zero completes the only gate in tick one.
            if self.step > 1:
                position[0, 2] = 0.01  # Synthetic inactive-row proposal, not a real flight tail.
            return replace(state, position=position)

    monkeypatch.setattr(bearing.replay, "sample_two_gate_cases", sample)
    monkeypatch.setattr(bearing.replay, "DifferentiableQuad", Quad)
    data = bearing.collect_bearing_bank(
        TinyActor(),
        TinyActor(),
        torch.tensor([0]),
        pairs=2,
        seed=2,
        kind="native",
        camera=CameraSpec(32, 20, 125),
        config=HoverConfig(),
        gate_config=GateConfig(),
        seconds=0.06,
        presynaptic_nodes=torch.tensor([0]),
    )
    assert data.replay.active[:, 0].tolist() == [True, False, False]
    assert data.metrics["ground_contacts"] == 0 and data.metrics["failures"] == 0
    assert data.metrics["clean_prefix_completions"] == 1
    assert data.metrics["full_flight_success_is_not_assessed"]
    assert data.input_moments["counts"].tolist() == [1, 3, 0, 0, 0, 0]
    # Statistics use pre-image warmup tick ten, not post-image tick eleven.
    assert data.input_moments["sums"][0, 0].item() == pytest.approx(
        torch.tensor(0.05 * (1 - 0.8**10)).tanh().item(), abs=1e-7
    )
