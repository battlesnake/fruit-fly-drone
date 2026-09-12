#!/usr/bin/env python3
"""Train native course control with phase-balanced, current-weight sensory replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_hover as hover_train  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import evaluate, sample_two_gate_cases  # noqa: E402
from search_pragmatic_gate_course_es import selection_score  # noqa: E402
from train_pragmatic_course_teacher import (  # noqa: E402
    action_imitation_loss,
    native_sensorimotor_mask,
)
from train_pragmatic_gate_visual_roll_path import (  # noqa: E402
    load_controller,
    visual_roll_path_mask,
)

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
from flydrone.visual_hover import CameraSpec  # noqa: E402

GEOMETRY = dict(
    layout="variable",
    gate_count=5,
    spacing_range=(0.9, 1.5),
    lateral_step_range=(0.0, 0.2),
    lateral_deviation_limit=0.5,
    height_step_range=(0.0, 0.08),
    height_range=(0.9, 1.3),
    yaw_jitter_degrees=15.0,
)


@dataclass
class ReplayBank:
    """Training-only CPU physical histories; never reusable neural states."""

    states: tuple[torch.Tensor, ...]  # time, episode, physical component
    gates: tuple[AnnularGate, ...]
    current: torch.Tensor
    active: torch.Tensor
    target: torch.Tensor
    kind: str
    seed: int
    failure_steps: torch.Tensor | None = None
    reference_outputs: torch.Tensor | None = None

    def starts(self, phase, unroll):
        if len(self.current) < unroll:
            return [[] for _ in range(self.current.shape[1])]
        valid_window = self.active.unfold(0, unroll, 1).all(dim=-1)
        eligible = valid_window & (self.current[: len(valid_window)] == phase)
        return [torch.nonzero(column).flatten().tolist() for column in eligible.T]

    def manifest(self):
        return dict(
            kind=self.kind,
            seed=self.seed,
            episodes=self.current.shape[1],
            recorded_steps=len(self.current),
            active_frames_by_gate=[
                int(((self.current == g) & self.active).sum()) for g in range(5)
            ],
            failed_lessons=(
                int((self.failure_steps >= 0).sum()) if self.failure_steps is not None else None
            ),
            source_preservation_outputs_stored=self.reference_outputs is not None,
        )


def late_roll_preservation_targets(teacher, reference, current):
    """Training labels only: preserve gate one and all non-roll source outputs."""
    target = reference.clone()
    target[:, 0] = torch.where(current >= 1, teacher[:, 0], reference[:, 0])
    return target


def roll_preservation_mask(graph_path, device):
    mask, manifest = visual_roll_path_mask(graph_path, hop_budget=5, device=device)
    with np.load(graph_path) as graph:
        other_motors = graph["output_pool_indices"][graph["output_pool_offsets"][2] :]
        enters_other_motor = torch.tensor(np.isin(graph["edge_post"], other_motors), device=device)
    excluded = int((mask & enters_other_motor).sum())
    mask &= ~enters_other_motor
    if not bool(mask.any()):
        raise ValueError("no roll paths remain after excluding other motor inputs")
    manifest.update(
        selected_edges=int(mask.sum()),
        excluded_edges_entering_other_motor_pools=excluded,
        supervision="late roll teacher plus frozen-source motor preservation",
    )
    return mask, manifest


@torch.no_grad()
def collect_bank(
    controller,
    pairs,
    seed,
    kind,
    seconds,
    camera,
    config,
    gate_config,
    heading_mode="world-x",
    reference_controller=None,
):
    if kind not in ("native", "teacher", "roll-assisted"):
        raise ValueError("collection kind must be native, teacher or roll-assisted")
    if kind == "roll-assisted" and reference_controller is None:
        raise ValueError("roll-assisted collection requires a frozen source controller")
    device = controller.bias.device
    cases, gates = sample_two_gate_cases(
        pairs, seed=seed, device=device, hover_config=config, **GEOMETRY
    )
    state, sticks = cases.state, cases.sticks
    path = CoursePath.through_gates(state.position, gates)
    teacher = CourseTeacherConfig(heading_mode=heading_mode)
    current = torch.zeros(2 * pairs, device=device, dtype=torch.long)
    failed = torch.zeros_like(current, dtype=torch.bool)
    failure_steps = torch.full_like(current, -1)
    neural = controller.initial_state(len(current), device=device, dtype=torch.float32)
    reference_neural = None
    if reference_controller is not None:
        reference_neural = reference_controller.initial_state(
            len(current), device=device, dtype=torch.float32
        )
    if kind == "native" or reference_controller is not None:
        image = render_annular_gates_rgb(
            state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
        )
        for _ in range(10):
            if kind == "native":
                _, neural = controller(image, state.euler[:, :2], neural)
            if reference_controller is not None:
                _, reference_neural = reference_controller(
                    image, state.euler[:, :2], reference_neural
                )
    quad, legs = DifferentiableQuad(config).to(device), ForelegStickPlant(config).to(device)
    histories = [[] for _ in state.as_tuple()]
    roles, active_frames, targets = [], [], []
    references = []
    for step in range(round(seconds * 50)):
        active = ~failed & (current < len(gates))
        if not bool(active.any()):
            break
        teacher_target = course_teacher_motor(state, path, config, teacher)
        target = teacher_target
        if kind == "native" or reference_controller is not None:
            image = render_annular_gates_rgb(
                state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
            )
        if reference_controller is not None:
            reference_motor, reference_neural = reference_controller(
                image, state.euler[:, :2], reference_neural
            )
            references.append(reference_motor.cpu().clone())
            target = late_roll_preservation_targets(teacher_target, reference_motor, current)
        for values, record in zip(state.as_tuple(), histories, strict=True):
            record.append(values.cpu().clone())
        roles.append(current.cpu().clone())
        active_frames.append(active.cpu().clone())
        targets.append(target.cpu().clone())
        if kind == "native":
            motor, neural = controller(image, state.euler[:, :2], neural)
        elif kind == "teacher":
            motor = teacher_target
        else:
            motor = target  # frozen-source flight, with teacher roll only after gate one
        for _ in range(2):
            active_before = ~failed & (current < len(gates))
            rc, sticks = legs(motor, sticks)
            previous = state.position
            state = quad(rc, state, cases.mass_scale)
            events = classify_course_step(previous, state.position, gates, current, gate_config)
            current = events.next_gate_index
            missed = (events.expected_forward_crossing & ~events.passed).any(dim=1)
            failure_now = events.failed | missed | ~hover_train.state_is_valid(state)
            failure_steps[active_before & failure_now] = step + 1
            failed |= failure_now
    if not roles:
        raise RuntimeError("collection produced no valid frames")
    bank = ReplayBank(
        tuple(torch.stack(record) for record in histories),
        tuple(AnnularGate(g.center.cpu(), g.yaw.cpu()) for g in gates),
        torch.stack(roles),
        torch.stack(active_frames),
        torch.stack(targets),
        kind,
        seed,
        failure_steps.cpu(),
        torch.stack(references) if references else None,
    )
    print(json.dumps(dict(stage="collection", **bank.manifest())), flush=True)
    return bank


def select_pair_window(native, teacher, phase, unroll, rng, window_kind):
    """Choose both mirrored branches at the same phase, not necessarily the same time."""
    for bank in (native, teacher):
        starts = bank.starts(phase, unroll)
        pairs = [p for p in range(len(starts) // 2) if starts[2 * p] and starts[2 * p + 1]]
        if not pairs:
            continue
        pair = int(rng.choice(pairs))
        rows = (2 * pair, 2 * pair + 1)
        selected = []
        actual_kinds = []
        for row in rows:
            options = starts[row]
            preferred = []
            if window_kind == "transition":
                preferred = [t for t in options if bank.current[t + unroll - 1, row] > phase]
            elif (
                window_kind == "pre-failure"
                and bank.failure_steps is not None
                and bank.failure_steps[row] >= 0
            ):
                last = int(torch.nonzero(bank.active[:, row]).flatten()[-1])
                preferred = [t for t in options if t + unroll - 1 >= last - unroll]
            selected.append(int(rng.choice(preferred or options)))
            actual_kinds.append(window_kind if preferred else "approach")
        return (
            bank,
            rows,
            selected,
            dict(
                source=bank.kind,
                seed=bank.seed,
                pair=pair,
                phase=phase + 1,
                starts=selected,
                kinds=actual_kinds,
            ),
        )
    raise RuntimeError(f"no paired replay windows for gate {phase + 1}")


def prepare_window(bank, rows, starts, unroll, device):
    stop = max(starts) + unroll
    return (
        tuple(value[:stop, list(rows)].to(device) for value in bank.states),
        tuple(
            AnnularGate(g.center[list(rows)].to(device), g.yaw[list(rows)].to(device))
            for g in bank.gates
        ),
        bank.current[:stop, list(rows)].to(device),
        bank.target[:stop, list(rows)].to(device),
        torch.tensor(starts, device=device),
    )


def window_observations(window, times, camera, gate_config):
    states, gates, roles, _, starts = window
    device = starts.device
    rows = torch.arange(len(starts), device=device)
    state = QuadState(*(value[times, rows] for value in states))
    image = render_annular_gates_rgb(
        state,
        gates,
        current_gate_index=roles[times, rows],
        camera=camera,
        gate_config=gate_config,
    )
    return image, state.euler[:, :2]


@torch.no_grad()
def replay_prefix_state(controller, window, camera, gate_config):
    """Recompute the full current-weight prefix up to each supervised window."""
    starts = window[4]
    device = starts.device
    neural = controller.initial_state(len(starts), device=device, dtype=torch.float32)
    image, attitude = window_observations(window, torch.zeros_like(starts), camera, gate_config)
    for _ in range(10):
        _, neural = controller(image, attitude, neural)
    for time in range(int(starts.max())):
        times = torch.minimum(torch.full_like(starts, time), starts)
        image, attitude = window_observations(window, times, camera, gate_config)
        _, advanced = controller(image, attitude, neural)
        # A shorter branch waits without receiving extra neural updates.
        neural = torch.where((time < starts)[:, None], advanced, neural)
    return neural


def replay_window_loss(
    controller,
    window,
    unroll,
    camera,
    gate_config,
    contrast_weight,
    *,
    diagnostics=False,
    roll_contrast_weight=None,
    diagnostic_fixed_prefix=None,
):
    """Replay with fresh prefixes; a deliberately stale prefix is an audit-only option."""
    _, _, _, targets, starts = window
    device = starts.device
    rows = torch.arange(len(starts), device=device)
    neural = (
        replay_prefix_state(controller, window, camera, gate_config)
        if diagnostic_fixed_prefix is None
        else diagnostic_fixed_prefix
    )
    neural = neural.detach()
    scale = neural.new_tensor((0.02, 0.02, 0.01, 0.025))
    active = torch.ones(len(starts), device=device, dtype=torch.bool)
    losses, axes, residuals = [], [], []
    for frame in range(unroll):
        times = starts + frame
        image, attitude = window_observations(window, times, camera, gate_config)
        prediction, neural = controller(image, attitude, neural)
        loss, axis = action_imitation_loss(
            prediction,
            targets[times, rows],
            active,
            scale,
            contrast_weight,
            roll_contrast_weight=roll_contrast_weight,
        )
        losses.append(loss)
        axes.append(axis.detach())
        if diagnostics:
            residuals.append((prediction - targets[times, rows]).detach())
    result = (torch.stack(losses).mean(), torch.stack(axes).mean(dim=0))
    return (*result, torch.stack(residuals)) if diagnostics else result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--unroll", type=int, default=20)
    parser.add_argument("--native-pairs", type=int, default=8)
    parser.add_argument("--teacher-pairs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument(
        "--supervision",
        choices=("all", "late-roll-preserve", "late-roll-anticipation"),
        default="all",
    )
    parser.add_argument("--anticipation-cache", type=Path)
    parser.add_argument("--anticipation-contrast-weight", type=float, default=1.0)
    parser.add_argument(
        "--anticipation-mean-target", choices=("teacher", "source"), default="teacher"
    )
    parser.add_argument(
        "--teacher-heading-mode", choices=("tangent", "world-x", "rate-damped"), default="world-x"
    )
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--development-pairs", type=int, default=16)
    parser.add_argument("--development-seed", type=int, default=1110983)
    parser.add_argument("--seed", type=int, default=1150983)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        min(
            args.updates,
            args.unroll,
            args.native_pairs,
            args.teacher_pairs,
            args.learning_rate,
            args.interval,
            args.seconds,
            args.development_pairs,
        )
        <= 0
    ):
        raise SystemExit("sizes, intervals and rates must be positive")
    if not np.isfinite(args.contrast_weight) or args.contrast_weight < 0:
        raise SystemExit("contrast weight must be finite and nonnegative")
    anticipation = args.supervision == "late-roll-anticipation"
    if anticipation and (
        args.anticipation_cache is None
        or args.teacher_heading_mode != "rate-damped"
        or not np.isfinite(args.anticipation_contrast_weight)
        or args.anticipation_contrast_weight < 0
    ):
        raise SystemExit(
            "anticipation needs a source cache, rate-damped teacher and valid contrast weight"
        )
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite an existing replay experiment directory")
    device = torch.device(args.device)
    controller, source = load_controller(args, device)
    reference_controller = None
    if args.supervision != "all":
        reference_controller = copy.deepcopy(controller).requires_grad_(False)
        edge_mask, manifest = roll_preservation_mask(args.graph, device)
    else:
        edge_mask, _, manifest = native_sensorimotor_mask(args.graph, 5, device)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_mask)
    anchor_edges = controller.edge_magnitude.detach()[edge_mask].clone()
    optimizer = torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)
    config = HoverConfig(**source["hover_config"])
    gate_config = replace(GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(
        width=source["image_resolution"][0],
        height=source["image_resolution"][1],
        horizontal_fov_degrees=source["camera_hfov_degrees"],
    )
    development = sample_two_gate_cases(
        args.development_pairs,
        seed=args.development_seed,
        device=device,
        hover_config=config,
        **GEOMETRY,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rng = np.random.default_rng(args.seed)
    history, collections = [], []
    anticipation_manifest = None
    validation_lessons, validation_records = [], []

    def assess():
        return evaluate(
            controller,
            *development,
            seconds=args.seconds,
            warmup_steps=10,
            camera=camera,
            hover_config=config,
            gate_config=gate_config,
        )

    baseline = assess()
    best, best_update = baseline, 0
    best_edges = controller.edge_magnitude.detach().clone()
    first_floor = max(0.0, baseline["first_gate_pass_rate"] - 0.05)

    def save(name, update, metrics):
        payload = dict(source)
        payload.update(
            controller={
                key: value.detach().cpu() for key, value in controller.state_dict().items()
            },
            experiment="phase-balanced-native-course-replay-v1",
            training_update=update,
            native_path_manifest=manifest,
            gate_config=vars(gate_config),
            course_geometry=GEOMETRY,
            selection_metrics=metrics,
            teacher_inputs_are_actor_inputs=False,
            teacher_config=vars(CourseTeacherConfig(heading_mode=args.teacher_heading_mode)),
            replay_prefix="current-weight-from-zero-with-original-10-frame-warmup",
            supervision=args.supervision,
            preservation_source_checkpoint=str(args.checkpoint)
            if reference_controller is not None
            else None,
            anticipation_manifest=anticipation_manifest,
        )
        torch.save(payload, args.output_dir / name)

    def report():
        result = dict(
            experiment="phase-balanced-native-course-replay-v1",
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            source_development=baseline,
            selected_development=best,
            selected_update=best_update,
            history=history,
            collections=collections,
            geometry=GEOMETRY,
            native_path_manifest=manifest,
            elapsed_seconds=perf_counter() - started,
            actor_inputs=["320x200 RGB", "roll", "pitch"],
            actor_outputs="native foreleg pools -> physical forelegs -> sticks",
            replay_or_teacher_state_deployed=False,
            anticipation_manifest=anticipation_manifest,
        )
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    def collect(kind, pairs, seed):
        bank = collect_bank(
            controller,
            pairs,
            seed,
            kind,
            args.seconds,
            camera,
            config,
            gate_config,
            args.teacher_heading_mode,
            reference_controller,
        )
        collections.append(bank.manifest())
        return bank

    save("best-controller.pt", 0, baseline)
    print(json.dumps(dict(stage="baseline", metrics=baseline)), flush=True)
    report()
    if anticipation:
        import pragmatic_anticipation_lessons as lessons

        cache = torch.load(args.anticipation_cache, map_location="cpu", weights_only=True)
        with args.checkpoint.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != cache["source_sha256"]:
                raise ValueError("anticipation cache belongs to a different source checkpoint")
        if len(cache["banks"]) != 2 or {b["seed"] for b in cache["banks"]} & {
            args.development_seed
        }:
            raise ValueError("need separate native/assisted cache banks and development courses")
        banks = [lessons.bank_from_motor_cache(bank) for bank in cache["banks"]]
        native_bank, teacher_bank = lessons.early_training_banks(banks)
        if native_bank.kind != "native" or teacher_bank.kind != "roll-assisted":
            raise ValueError("anticipation cache bank ordering is not native/roll-assisted")
        collections.extend(bank.manifest() for bank in banks)
        validation_rng = np.random.default_rng(args.seed + 10000)
        for phase in (1, 2, 3):
            for side in (0, 1):
                bank, row, start = lessons.select_anticipation_window(
                    banks, phase, side, args.unroll, validation_rng, validation=True
                )
                window, record = lessons.make_anticipation_lesson(
                    bank,
                    row,
                    start,
                    args.unroll,
                    reference_controller,
                    camera,
                    gate_config,
                    config,
                    mean_target=args.anticipation_mean_target,
                )
                validation_lessons.append(window)
                validation_records.append(record)
        anticipation_manifest = dict(
            cache=str(args.anticipation_cache),
            source_sha256=cache["source_sha256"],
            collection_seeds=[bank.seed for bank in banks],
            heldout_pairs_per_bank=2,
            training_pairs_per_bank=[b.current.shape[1] // 2 for b in (native_bank, teacher_bank)],
            fixed_physical_cache_current_weight_neural_replay=True,
            next_gate_placement_separation_metres=0.30,
            early_windows_are_entirely_before_first_gate=True,
            counterfactual_columns="minus/plus next-gate placement on one physical history",
            roll_only_contrast_weight=args.anticipation_contrast_weight,
            roll_pair_mean_target=args.anticipation_mean_target,
            source_validation=lessons.source_contrast_summary(validation_records),
            validation_lessons=validation_records,
            first_gate_floor_is_hard_selection_requirement=True,
        )
        print(json.dumps(dict(stage="anticipation_lessons", **anticipation_manifest)), flush=True)
    else:
        teacher_bank = collect(
            "roll-assisted" if reference_controller is not None else "teacher",
            args.teacher_pairs,
            args.seed + 100000,
        )
        native_bank = collect("native", args.native_pairs, args.seed)
    report()
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        windows, loss_values, axis_values = [], [], []
        kind = ("approach", "transition", "pre-failure")[(update - 1) % 3]
        late_phase = 1 + ((update - 1) // 2) % 3 if anticipation else 1 + (update - 1) % 4
        for phase in (0, late_phase):
            roll_contrast_weight = None
            if anticipation and phase > 0:
                side = (update - 1) % 2
                bank, row, start = lessons.select_anticipation_window(
                    banks, phase, side, args.unroll, rng
                )
                window, record = lessons.make_anticipation_lesson(
                    bank,
                    row,
                    start,
                    args.unroll,
                    reference_controller,
                    camera,
                    gate_config,
                    config,
                    mean_target=args.anticipation_mean_target,
                )
                roll_contrast_weight = args.anticipation_contrast_weight
            else:
                bank, rows, starts, record = select_pair_window(
                    native_bank, teacher_bank, phase, args.unroll, rng, kind
                )
                window = prepare_window(bank, rows, starts, args.unroll, device)
            loss, axes = replay_window_loss(
                controller,
                window,
                args.unroll,
                camera,
                gate_config,
                args.contrast_weight,
                roll_contrast_weight=roll_contrast_weight,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite replay loss; no update applied")
            (0.5 * loss).backward()
            windows.append(record)
            loss_values.append(float(loss.detach()))
            axis_values.append(axes.tolist())
            del loss, window
        anchor = (
            1e-3 * ((controller.edge_magnitude[edge_mask] - anchor_edges) / 0.02).square().mean()
        )
        anchor.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            controller.parameters(), 1.0, error_if_nonfinite=True
        )
        before = controller.edge_magnitude.detach()[edge_mask].clone()
        masked_gradient = controller.edge_magnitude.grad[edge_mask].clone()
        optimizer.step()
        controller.project_parameters()
        delta = controller.edge_magnitude.detach()[edge_mask] - before
        entry = dict(
            update=update,
            windows=windows,
            losses=loss_values,
            axis_losses=axis_values,
            gradient_norm=float(gradient),
            gradient_update_dot=float((masked_gradient * delta).sum()),
            elapsed_seconds=perf_counter() - started,
        )
        if update % args.interval == 0 or update == args.updates:
            if anticipation:
                with torch.no_grad():
                    residuals = [
                        replay_window_loss(
                            controller,
                            window,
                            args.unroll,
                            camera,
                            gate_config,
                            args.contrast_weight,
                            diagnostics=True,
                            roll_contrast_weight=args.anticipation_contrast_weight,
                        )[2]
                        for window in validation_lessons
                    ]
                    entry["anticipation_validation"] = lessons.contrast_error_summary(
                        residuals,
                        validation_records,
                        [
                            lessons.window_target_contrast(window, args.unroll)
                            for window in validation_lessons
                        ],
                    )
            metrics = assess()
            entry["development"] = metrics
            save("latest-controller.pt", update, metrics)
            eligible = not anticipation or metrics["first_gate_pass_rate"] >= first_floor
            if eligible and selection_score(metrics, first_floor) > selection_score(
                best, first_floor
            ):
                best, best_update = metrics, update
                best_edges = controller.edge_magnitude.detach().clone()
                save("best-controller.pt", update, metrics)
            print(
                json.dumps(
                    dict(
                        stage="development",
                        update=update,
                        clean=metrics["clean_course_success_rate"],
                        first=metrics["first_gate_pass_rate"],
                        prefix=metrics["gates_before_failure_mean"],
                    )
                ),
                flush=True,
            )
            # Refresh collection under retained weights and discard stale optimizer
            # momentum only when returning to an earlier, better checkpoint.
            if update < args.updates and not anticipation:
                if best_update != update:
                    with torch.no_grad():
                        controller.edge_magnitude.copy_(best_edges)
                    optimizer.state.clear()
                    entry["restored_best_update"] = best_update
                native_bank = collect("native", args.native_pairs, args.seed + update)
        history.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "development"}), flush=True)
        report()
    print(
        json.dumps(dict(stage="complete", selected_update=best_update, selected=best)), flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
