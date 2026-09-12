#!/usr/bin/env python3
"""Fit only existing roll motor synapses using the full brain's recorded activity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_hover as hover_train  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import sample_two_gate_cases  # noqa: E402
from train_pragmatic_course_replay import (  # noqa: E402
    GEOMETRY,
    late_roll_preservation_targets,
    roll_preservation_mask,
)
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.course_teacher import (  # noqa: E402
    CoursePath,
    CourseTeacherConfig,
    course_teacher_motor,
)
from flydrone.gate import AnnularGate, GateConfig, render_annular_gates_rgb  # noqa: E402
from flydrone.gate_course import classify_course_step  # noqa: E402
from flydrone.hover import (  # noqa: E402
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
)
from flydrone.motor_slice import SinkMotorSlice  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


@torch.no_grad()
def collect(controller, model, *, pairs, seed, assisted, seconds, camera, config, gate_config):
    device = controller.bias.device
    cases, gates = sample_two_gate_cases(
        pairs, seed=seed, device=device, hover_config=config, **GEOMETRY
    )
    state, sticks = cases.state, cases.sticks
    path = CoursePath.through_gates(state.position, gates)
    quad, legs = DifferentiableQuad(config).to(device), ForelegStickPlant(config).to(device)
    neural = controller.initial_state(2 * pairs, device=device, dtype=torch.float32)
    current = torch.zeros(2 * pairs, dtype=torch.long, device=device)
    failed = torch.zeros_like(current, dtype=torch.bool)
    features, targets, sources, roles, active_frames = [], [], [], [], []
    histories = [[] for _ in state.as_tuple()]
    teacher = CourseTeacherConfig(heading_mode="rate-damped")
    for step in range(-10, round(seconds * 50)):
        active = ~failed & (current < 5)
        if step >= 0 and not bool(active.any()):
            break
        image = render_annular_gates_rgb(
            state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
        )
        feature = torch.tanh(neural[:, model.parents])
        source_motor, neural = controller(image, state.euler[:, :2], neural)
        target = late_roll_preservation_targets(
            course_teacher_motor(state, path, config, teacher), source_motor, current
        )
        if not all(bool(torch.isfinite(v[active]).all()) for v in (feature, target, source_motor)):
            raise RuntimeError("nonfinite active features or labels; collection rejected")
        features.append(torch.nan_to_num(feature).clone())
        targets.append(torch.nan_to_num(target[:, 0]).clone())
        sources.append(torch.nan_to_num(source_motor).clone())
        roles.append(current.clone())
        active_frames.append(active & (step >= 0))
        for record, value in zip(histories, state.as_tuple(), strict=True):
            record.append(value.clone())
        if step < 0:
            continue
        motor = target if assisted else source_motor
        for _ in range(2):
            rc, sticks = legs(motor, sticks)
            previous = state.position
            state = quad(rc, state, cases.mass_scale)
            events = classify_course_step(previous, state.position, gates, current, gate_config)
            current = events.next_gate_index
            missed = (events.expected_forward_crossing & ~events.passed).any(dim=1)
            failed |= events.failed | missed | ~hover_train.state_is_valid(state)
    result = dict(
        features=torch.stack(features),
        target=torch.stack(targets),
        source=torch.stack(sources),
        current=torch.stack(roles),
        active=torch.stack(active_frames),
        states=tuple(torch.stack(record) for record in histories),
        gates=gates,
        seed=seed,
        assisted=assisted,
    )
    print(
        json.dumps(
            dict(
                stage="collection",
                seed=seed,
                assisted=assisted,
                frames=len(features),
                active_by_gate=[
                    int(((result["current"] == g) & result["active"]).sum()) for g in range(5)
                ],
            )
        ),
        flush=True,
    )
    return result


def pair_split_weights(bank, heldout_pairs, *, validation):
    roles, active = bank["current"], bank["active"]
    rows = torch.arange(roles.shape[1], device=roles.device)
    split = roles.shape[1] // 2 - heldout_pairs
    selected = rows // 2 >= split if validation else rows // 2 < split
    weights = torch.zeros_like(roles, dtype=torch.float32)
    for phase in range(5):
        for side in range(2):
            mask = active & (roles == phase) & selected[None] & ((rows % 2 == side)[None])
            count = int(mask.sum())
            if count:
                weights[mask] = (0.5 if phase == 0 else 0.125) / (2 * count)
    return weights / weights.sum().clamp_min(1e-12)


def coverage_report(bank, heldout_pairs):
    result = dict(seed=bank["seed"], assisted=bank["assisted"])
    rows = torch.arange(bank["current"].shape[1], device=bank["current"].device)
    heldout = rows // 2 >= len(rows) // 2 - heldout_pairs
    for validation, name in ((False, "training"), (True, "validation")):
        selected = heldout if validation else ~heldout
        result[name] = dict(
            phase_side_frame_counts=[
                [
                    int(
                        (
                            bank["active"]
                            & (bank["current"] == g)
                            & selected[None]
                            & ((rows % 2 == side)[None])
                        ).sum()
                    )
                    for side in range(2)
                ]
                for g in range(5)
            ],
            early_weight=float(
                pair_split_weights(bank, heldout_pairs, validation=validation)[
                    bank["current"] == 0
                ].sum()
            ),
        )
    return result


@torch.no_grad()
def prediction_metrics(predictions, banks, heldout_pairs):
    errors = [((p - b["target"]) / 0.02).square() for p, b in zip(predictions, banks, strict=True)]
    result = {}
    for validation, name in ((False, "training"), (True, "validation")):
        result[name] = float(
            torch.stack(
                [
                    (e * pair_split_weights(b, heldout_pairs, validation=validation)).sum()
                    for b, e in zip(banks, errors, strict=True)
                ]
            ).mean()
        )
    for side in range(2):
        for late in (False, True):
            squared, count = 0.0, 0
            for bank, error in zip(banks, errors, strict=True):
                rows = torch.arange(bank["current"].shape[1], device=error.device)
                heldout = rows // 2 >= len(rows) // 2 - heldout_pairs
                phase = bank["current"] >= 1 if late else bank["current"] == 0
                mask = bank["active"] & phase & heldout[None] & ((rows % 2 == side)[None])
                squared += float(error[mask].sum())
                count += int(mask.sum())
            period = "late" if late else "early"
            side_name = "negative" if side == 0 else "positive"
            result[f"heldout_{period}_{side_name}_roll_rmse"] = (
                (squared / count) ** 0.5 * 0.02 if count else None
            )
    return result


@torch.no_grad()
def metrics(model, banks, heldout_pairs):
    return prediction_metrics([model(b["features"]) for b in banks], banks, heldout_pairs)


@torch.no_grad()
def linear_probe(banks, heldout_pairs):
    """Diagnostic only: an unconstrained linear readout, never compiled or deployed."""
    x = torch.cat([b["features"].flatten(0, 1) for b in banks])
    y = torch.cat([b["target"].flatten() / 0.02 for b in banks])
    weights = torch.cat(
        [pair_split_weights(b, heldout_pairs, validation=False).flatten() for b in banks]
    ) / len(banks)
    mean = (weights[:, None] * x).sum(0)
    scale = ((weights[:, None] * (x - mean).square()).sum(0)).sqrt().clamp_min(1e-3)
    design = (x - mean) / scale
    target_mean = (weights * y).sum()
    covariance = design.T @ (weights[:, None] * design)
    coefficient = torch.linalg.solve(
        covariance + 1e-3 * torch.eye(len(scale), device=x.device),
        design.T @ (weights * (y - target_mean)),
    )
    predictions = [
        (((b["features"] - mean) / scale) @ coefficient + target_mean) * 0.02 for b in banks
    ]
    return dict(
        kind="unconstrained-instantaneous-parent-probe-not-deployed",
        metrics=prediction_metrics(predictions, banks, heldout_pairs),
    )


def transfer_bank(bank, device, *, serialize):
    result = {}
    for key, value in bank.items():
        if key == "states":
            result[key] = tuple(v.to(device) for v in value)
        elif key == "gates":
            result[key] = (
                [(g.center.cpu(), g.yaw.cpu()) for g in value]
                if serialize
                else tuple(AnnularGate(c.to(device), y.to(device)) for c, y in value)
            )
        else:
            result[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return result


@torch.no_grad()
def verify_full_replay(controller, model, bank, camera, gate_config):
    # One whole held-out mirrored pair, with all original warmup/history frames.
    rows = slice(-2, None)
    gates = tuple(type(g)(g.center[rows], g.yaw[rows]) for g in bank["gates"])
    neural = controller.initial_state(2, device=controller.bias.device, dtype=torch.float32)
    expected = model(bank["features"][:, rows])
    roll_squared, other_squared, count, maximum = 0.0, 0.0, 0, 0.0
    for step in range(len(expected)):
        state = QuadState(*(v[step, rows] for v in bank["states"]))
        image = render_annular_gates_rgb(
            state,
            gates,
            current_gate_index=bank["current"][step, rows],
            camera=camera,
            gate_config=gate_config,
        )
        motor, neural = controller(image, state.euler[:, :2], neural)
        active = bank["active"][step, rows]
        if bool(active.any()):
            difference = motor[active, 0] - expected[step, active]
            roll_squared += float(difference.square().sum())
            maximum = max(maximum, float(difference.abs().max()))
            other_squared += float(
                (motor[active, 1:] - bank["source"][step, rows][active, 1:]).square().sum()
            )
            count += int(active.sum())
    return dict(
        seed=bank["seed"],
        active_frames=count,
        full_vs_slice_roll_rmse=(roll_squared / count) ** 0.5 if count else None,
        full_vs_slice_roll_max=maximum,
        nonroll_source_rmse=(other_squared / (3 * count)) ** 0.5 if count else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--heldout-pairs", type=int, default=2)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, default=1190983)
    parser.add_argument("--all-incoming", action="store_true")
    parser.add_argument("--reuse-features", type=Path)
    args = parser.parse_args()
    if (
        not 0 < args.heldout_pairs < args.pairs
        or min(args.updates, args.learning_rate, args.seconds) <= 0
    ):
        raise SystemExit(
            "positive sizes/rates and a whole-pair training/validation split are required"
        )
    started = perf_counter()
    device = torch.device(args.device)
    controller, source = load_controller(args, device)
    controller.requires_grad_(False)
    mask, _ = roll_preservation_mask(args.graph, device)
    model = SinkMotorSlice(controller, train_edge_mask=None if args.all_incoming else mask)
    config = HoverConfig(**source["hover_config"])
    gate_config = replace(GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.reuse_features:
        cache = torch.load(args.reuse_features, map_location="cpu", weights_only=True)
        if cache["source_sha256"] != source_hash or not torch.equal(
            cache["parents"], model.parents.cpu()
        ):
            raise ValueError("feature cache source or parent layout does not match")
        if (
            cache["pairs"] != args.pairs
            or cache["seconds"] != args.seconds
            or cache["seed"] != args.seed
        ):
            raise ValueError("feature cache collection recipe does not match")
        banks = [transfer_bank(bank, device, serialize=False) for bank in cache["banks"]]
    else:
        banks = [
            collect(
                controller,
                model,
                pairs=args.pairs,
                seed=args.seed + 10000 * i,
                assisted=bool(i),
                seconds=args.seconds,
                camera=camera,
                config=config,
                gate_config=gate_config,
            )
            for i in range(2)
        ]
        torch.save(
            dict(
                source_sha256=source_hash,
                parents=model.parents.cpu(),
                pairs=args.pairs,
                seconds=args.seconds,
                seed=args.seed,
                banks=[transfer_bank(bank, "cpu", serialize=True) for bank in banks],
            ),
            args.output_dir / "feature-cache.pt",
        )
    weights = [pair_split_weights(bank, args.heldout_pairs, validation=False) for bank in banks]
    baseline = metrics(model, banks, args.heldout_pairs)
    probe = linear_probe(banks, args.heldout_pairs)
    print(json.dumps(dict(stage="linear_probe", **probe)), flush=True)
    with torch.no_grad():
        reconstruction = [
            float(
                (model(b["features"])[b["active"]] - b["source"][..., 0][b["active"]]).abs().max()
            )
            for b in banks
        ]
    if max(reconstruction) > 2e-5:
        raise RuntimeError(
            f"source motor slice does not reconstruct the full source: {reconstruction}"
        )
    print(
        json.dumps(
            dict(
                stage="baseline",
                manifest=model.manifest(),
                reconstruction_max=reconstruction,
                metrics=baseline,
            )
        ),
        flush=True,
    )
    initial = model.magnitudes.detach().clone()
    best_weights, best_metrics, best_update = initial.clone(), baseline, 0
    optimizer = torch.optim.Adam([model.magnitudes], lr=args.learning_rate)
    history = []

    def report():
        result = dict(
            experiment="native-sink-roll-motor-fit-v1",
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            manifest=model.manifest(),
            baseline=baseline,
            unconstrained_probe=probe,
            coverage=[coverage_report(b, args.heldout_pairs) for b in banks],
            selected=best_metrics,
            selected_update=best_update,
            source_reconstruction_max=reconstruction,
            history=history,
            elapsed_seconds=perf_counter() - started,
            teacher_or_extra_decoder_deployed=False,
            goal_success_not_established=True,
        )
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    report()
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [
                (((model(bank["features"]) - bank["target"]) / 0.02).square() * weight).sum()
                for bank, weight in zip(banks, weights, strict=True)
            ]
        ).mean()
        loss = loss + 1e-4 * ((model.magnitudes - initial) / 0.02).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        model.project_parameters()
        if update == 1 or update % 25 == 0 or update == args.updates:
            measured = metrics(model, banks, args.heldout_pairs)
            if measured["validation"] < best_metrics["validation"]:
                best_weights, best_metrics, best_update = (
                    model.magnitudes.detach().clone(),
                    measured,
                    update,
                )
            history.append(dict(update=update, **measured))
            print(
                json.dumps(
                    dict(
                        stage="fit",
                        update=update,
                        **measured,
                        elapsed_seconds=perf_counter() - started,
                    )
                ),
                flush=True,
            )
            report()

    def export(name, update):
        model.compile_into(controller)
        payload = dict(source)
        payload.update(
            controller={k: v.detach().cpu() for k, v in controller.state_dict().items()},
            experiment="native-sink-roll-motor-fit-v1",
            selection_metrics=None,
            motor_slice_manifest=model.manifest(),
            source_checkpoint=str(args.checkpoint),
            teacher_inputs_are_actor_inputs=False,
            training_update=update,
        )
        torch.save(payload, args.output_dir / name)

    export("latest-controller.pt", args.updates)
    verification = [
        verify_full_replay(controller, model, bank, camera, gate_config) for bank in banks
    ]
    for record in verification:
        values = [
            record[k]
            for k in ("full_vs_slice_roll_rmse", "full_vs_slice_roll_max", "nonroll_source_rmse")
        ]
        if record["active_frames"] <= 0 or not bool(torch.isfinite(torch.tensor(values)).all()):
            raise RuntimeError(f"empty or nonfinite full-network replay: {record}")
    with torch.no_grad():
        model.magnitudes.copy_(best_weights)
    export("candidate-controller.pt", best_update)
    result = report()
    result["full_network_replay"] = verification
    result["full_network_replay_checkpoint"] = str(args.output_dir / "latest-controller.pt")
    result["candidate_checkpoint"] = str(args.output_dir / "candidate-controller.pt")
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                stage="complete",
                selected_update=best_update,
                selected=best_metrics,
                full_network_replay=verification,
                elapsed_seconds=perf_counter() - started,
            )
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
