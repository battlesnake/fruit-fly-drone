"""Training-only matched-state, changed-next-gate anticipation lessons.

The two branches replay identical physical histories, but a different, fixed upcoming
gate placement is rendered throughout each complete prefix. Neither branch receives
a gate coordinate or a decoded feature as actor input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_pragmatic_course_replay import GEOMETRY, ReplayBank, prepare_window  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.course_teacher import (  # noqa: E402
    CoursePath,
    CourseTeacherConfig,
    course_teacher_motor,
)
from flydrone.gate import AnnularGate, GateConfig, render_annular_gates_rgb  # noqa: E402
from flydrone.hover import HoverConfig, QuadState  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def bank_from_motor_cache(bank):
    """Drop the stored ten warmup frames; replay regenerates its own neural warmup."""
    target = bank["source"][10:].clone()
    target[:, :, 0] = bank["target"][10:]
    return ReplayBank(
        tuple(field[10:] for field in bank["states"]),
        tuple(AnnularGate(*gate) for gate in bank["gates"]),
        bank["current"][10:],
        bank["active"][10:],
        target,
        "roll-assisted" if bank["assisted"] else "native",
        bank["seed"],
        reference_outputs=bank["source"][10:],
    )


def early_training_banks(banks, heldout_pairs=2):
    """Exclude held-out whole courses and transition frames from source preservation."""
    result = []
    for bank in banks:
        count = bank.current.shape[1] - 2 * heldout_pairs
        if count < 2 or bank.reference_outputs is None:
            raise ValueError("early preservation needs training pairs and stored source outputs")
        result.append(
            replace(
                bank,
                states=tuple(field[:, :count] for field in bank.states),
                gates=tuple(AnnularGate(g.center[:count], g.yaw[:count]) for g in bank.gates),
                current=bank.current[:, :count],
                active=bank.active[:, :count] & (bank.current[:, :count] == 0),
                target=bank.reference_outputs[:, :count],
                reference_outputs=bank.reference_outputs[:, :count],
                failure_steps=None,
            )
        )
    return result


def window_target_contrast(window, unroll):
    targets, starts = window[3], window[4]
    times = starts[None] + torch.arange(unroll, device=starts.device)[:, None]
    sampled = targets[times, torch.arange(len(starts), device=starts.device)[None]]
    return sampled[:, 1, 0] - sampled[:, 0, 0]


def source_contrast_summary(records):
    result = {}
    for side in ("negative", "positive"):
        group = [r for r in records if r["base_side"] == side]
        teacher_sq = np.mean([r["teacher_roll_contrast_rms"] ** 2 for r in group])
        source_sq = np.mean([r["source_roll_contrast_rms"] ** 2 for r in group])
        error_sq = np.mean([r["source_contrast_error_rmse"] ** 2 for r in group])
        dot = (teacher_sq + source_sq - error_sq) / 2
        pair_mean_errors = [r.get("source_pair_mean_error_rmse") for r in group]
        result[side] = dict(
            contrast_error_rmse=float(np.sqrt(error_sq)),
            zero_contrast_error_rmse=float(np.sqrt(teacher_sq)),
            teacher_alignment_cosine=float(dot / max(np.sqrt(teacher_sq * source_sq), 1e-12)),
            beats_zero_contrast=bool(error_sq < teacher_sq),
            pair_mean_roll_error_rmse=(
                float(np.sqrt(np.mean(np.square(pair_mean_errors))))
                if all(value is not None for value in pair_mean_errors)
                else None
            ),
        )
    return result


def contrast_error_summary(residuals, records, target_contrasts):
    """Report by physical base-course side, not by counterfactual column number."""
    result = {}
    for side in ("negative", "positive"):
        selected = [
            (error, target)
            for error, record, target in zip(residuals, records, target_contrasts, strict=True)
            if record["base_side"] == side
        ]
        if not selected:
            raise ValueError(f"no held-out anticipation lessons for base side {side}")
        errors = torch.cat([item[0] for item in selected], dim=0)
        desired = torch.cat([item[1] for item in selected], dim=0)
        if not bool(torch.isfinite(errors).all()):
            raise ValueError("nonfinite anticipation diagnostic residuals")
        contrast = errors[:, 1, 0] - errors[:, 0, 0]
        prediction = desired + contrast
        zero_error = desired.square().mean().sqrt()
        error = contrast.square().mean().sqrt()
        denominator = (desired.square().mean() * prediction.square().mean()).sqrt().clamp_min(1e-12)
        result[side] = dict(
            contrast_error_rmse=float(error),
            zero_contrast_error_rmse=float(zero_error),
            teacher_alignment_cosine=float((desired * prediction).mean() / denominator),
            beats_zero_contrast=bool(error < zero_error),
            pair_mean_roll_error_rmse=float(errors[:, :, 0].mean(dim=1).square().mean().sqrt()),
            roll_error_rmse=float(errors[:, :, 0].square().mean().sqrt()),
            nonroll_source_rmse=float(errors[:, :, 1:].square().mean().sqrt()),
            paired_frames=len(contrast),
        )
    return result


def changed_next_gate_pair(gates, row, phase, launch_position, half_separation=0.15):
    """Two legal next-gate positions; all other geometry remains identical."""
    if phase not in (1, 2, 3) or not 0 < half_separation <= 0.2:
        raise ValueError("anticipation lessons use current gates 2–4 and positive small shifts")
    changed = phase + 1
    first = gates[0].center[row]
    direction = first - launch_position
    if float(direction[0]) <= 0:
        raise ValueError("course centreline requires a first gate ahead of launch")
    line_y = [
        float(first[1] + (g.center[row, 0] - first[0]) * direction[1] / direction[0]) for g in gates
    ]
    residual = [float(g.center[row, 1]) - nominal for g, nominal in zip(gates, line_y, strict=True)]
    current_y = residual[phase]
    step = GEOMETRY["lateral_step_range"][1]
    deviation = GEOMETRY["lateral_deviation_limit"]
    lower, upper = (
        max(current_y - step, -deviation),
        min(current_y + step, deviation),
    )
    if changed + 1 < len(gates):
        following_y = residual[changed + 1]
        lower, upper = max(lower, following_y - step), min(upper, following_y + step)
    if upper - lower < 2 * half_separation:
        return None
    center = np.clip(residual[changed], lower + half_separation, upper - half_separation)
    result = []
    for index, gate in enumerate(gates):
        positions = gate.center[row : row + 1].repeat(2, 1)
        yaw = gate.yaw[row : row + 1].repeat(2)
        if index == changed:
            positions[:, 1] = (
                positions.new_tensor([center - half_separation, center + half_separation])
                + line_y[changed]
            )
        result.append(AnnularGate(positions, yaw))
    return tuple(result)


def select_anticipation_window(
    banks, phase, side, unroll, rng, *, heldout_pairs=2, validation=False
):
    """Select whole base courses, never frames shared across the fit/check split."""
    choices = []
    for bank in banks:
        split = bank.current.shape[1] // 2 - heldout_pairs
        for row, options in enumerate(bank.starts(phase, unroll)):
            if row % 2 != side or (row // 2 >= split) != validation:
                continue
            if changed_next_gate_pair(bank.gates, row, phase, bank.states[0][0, row]) is None:
                continue
            # Do not reuse an old role after a modified future gate has been reached.
            options = [
                t for t in options if bool((bank.current[t : t + unroll, row] == phase).all())
            ]
            if options:
                choices.append((bank, row, options))
    if not choices:
        raise ValueError(
            f"no legal anticipation windows for phase={phase}, side={side}, heldout={validation}"
        )
    bank, row, options = choices[int(rng.integers(len(choices)))]
    return bank, row, int(rng.choice(options))


@torch.no_grad()
def make_anticipation_lesson(bank, row, start, unroll, reference, camera, gate_config, config):
    """Freeze source non-roll targets on each newly rendered counterfactual history."""
    device = reference.bias.device
    phase = int(bank.current[start, row])
    if not bool(bank.active[start : start + unroll, row].all()) or not bool(
        (bank.current[start : start + unroll, row] == phase).all()
    ):
        raise ValueError("lesson must remain active and in one current-gate phase")
    gates = changed_next_gate_pair(bank.gates, row, phase, bank.states[0][0, row])
    if gates is None:
        raise ValueError("insufficient course-legal room for the next-gate pair")
    fields, _, roles, target, starts = prepare_window(
        bank, (row, row), (start, start), unroll, device
    )
    gates = tuple(AnnularGate(g.center.to(device), g.yaw.to(device)) for g in gates)
    future = gates[phase + 1]
    signed = ((fields[0] - future.center[None]) * future.normal[None]).sum(dim=-1)
    if not bool((signed < 0).all()):
        raise ValueError("modified upcoming gate must remain ahead throughout the replay")
    target = target.clone()
    neural = reference.initial_state(2, device=device, dtype=torch.float32)
    path = CoursePath.through_gates(fields[0][0], gates)
    teacher = CourseTeacherConfig(heading_mode="rate-damped")
    visible_change, target_contrast, source_contrast, pair_mean_error = [], [], [], []
    for frame in range(-10, start + unroll):
        time = max(frame, 0)
        state = QuadState(*(value[time] for value in fields))
        image = render_annular_gates_rgb(
            state, gates, current_gate_index=roles[time], camera=camera, gate_config=gate_config
        )
        source_motor, neural = reference(image, state.euler[:, :2], neural)
        if frame >= start:
            taught = course_teacher_motor(state, path, config, teacher)
            target[time] = source_motor
            target[time, :, 0] = taught[:, 0]
            visible_change.append((image[1] - image[0]).abs().mean())
            target_contrast.append(taught[1, 0] - taught[0, 0])
            source_contrast.append(source_motor[1, 0] - source_motor[0, 0])
            pair_mean_error.append((source_motor[:, 0] - taught[:, 0]).mean())
    visible_change = torch.stack(visible_change)
    desired, actual = torch.stack(target_contrast), torch.stack(source_contrast)
    if not bool(torch.isfinite(target[start:]).all()) or not bool(visible_change.max() > 1e-6):
        raise ValueError(
            "counterfactual target is nonfinite or upcoming geometry is not visibly different"
        )
    record = dict(
        bank_seed=bank.seed,
        source_kind=bank.kind,
        row=row,
        base_side="negative" if row % 2 == 0 else "positive",
        phase=phase + 1,
        changed_gate=phase + 2,
        start=start,
        next_gate_y=gates[phase + 1].center[:, 1].tolist(),
        mean_image_change=float(visible_change.mean()),
        teacher_roll_contrast_rms=float(desired.square().mean().sqrt()),
        source_roll_contrast_rms=float(actual.square().mean().sqrt()),
        source_contrast_error_rmse=float((actual - desired).square().mean().sqrt()),
        source_pair_mean_error_rmse=float(torch.stack(pair_mean_error).square().mean().sqrt()),
        matched_physical_history=True,
        changed_geometry_fixed_for_entire_prefix=True,
    )
    return (fields, gates, roles, target, starts), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--seed", type=int, default=1240983)
    parser.add_argument("--unroll", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing lesson audit")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    with args.checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != cache["source_sha256"]:
            raise ValueError("cache and source checkpoint differ")
    controller, source = load_controller(args, torch.device("cuda"))
    controller.requires_grad_(False)
    banks = [bank_from_motor_cache(bank) for bank in cache["banks"]]
    camera = CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    config = HoverConfig(**source["hover_config"])
    gate_config = replace(GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    rng = np.random.default_rng(args.seed)
    records, rejected = [], []
    for validation in (False, True):
        for phase in (1, 2, 3):
            for side in (0, 1):
                try:
                    bank, row, start = select_anticipation_window(
                        banks, phase, side, args.unroll, rng, validation=validation
                    )
                    window, record = make_anticipation_lesson(
                        bank, row, start, args.unroll, controller, camera, gate_config, config
                    )
                    record["validation"] = validation
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    del window
                except ValueError as error:
                    item = dict(
                        validation=validation, phase=phase + 1, side=side, reason=str(error)
                    )
                    rejected.append(item)
                    print(json.dumps(item), flush=True)
    result = dict(
        experiment="matched-state-next-gate-anticipation-audit-v1",
        source_checkpoint=str(args.checkpoint),
        cache=str(args.cache),
        seed=args.seed,
        unroll=args.unroll,
        records=records,
        rejected=rejected,
        actor_training_performed=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(dict(stage="complete", lessons=len(records), rejected=len(rejected))), flush=True
    )


if __name__ == "__main__":
    main()
