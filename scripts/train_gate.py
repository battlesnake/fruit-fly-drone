#!/usr/bin/env python3
"""Train and evaluate takeoff-to-annular-gate flight through the MaleCNS actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.gate import (  # noqa: E402
    GATE_TASK_VERSION,
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    crossing_coordinates,
    gate_coordinates,
    initial_gate_geometry,
    render_annular_gate,
    sample_annular_gates,
    teacher_gate_rc,
    teacher_gate_rc_with_accelerometer,
    wrap_angle,
)
from flydrone.hover import (  # noqa: E402
    PLANT_MODEL_VERSION,
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    motor_target_for_rc,
)

RETINAL_RECEPTIVE_FIELD = 3
RETINAL_FLIP_X = True


@dataclass
class DaggerTrajectories:
    """Training-only observations and expert labels from complete frozen rollouts."""

    images: torch.Tensor
    roll_pitch: torch.Tensor
    specific_force: torch.Tensor
    targets: torch.Tensor
    valid: torch.Tensor
    lateral_offset: torch.Tensor
    teacher_driven: bool
    mirrored_pairs: bool


@dataclass
class RecoveryTrajectories:
    """Student-prefix trajectories completed by the teacher during training only."""

    images: torch.Tensor
    roll_pitch: torch.Tensor
    specific_force: torch.Tensor
    targets: torch.Tensor
    baseline_motor: torch.Tensor
    valid: torch.Tensor
    lateral_offset: torch.Tensor
    takeover_steps: torch.Tensor
    mirrored_pairs: bool
    teacher_recovery_clean_crossing_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "hover-v1" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "hover-v1" / "controller.pt",
        help="Warm-start or frozen gate checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-graph",
        type=Path,
        help="Original graph for anatomical parameter remapping to a denser --graph.",
    )
    parser.add_argument(
        "--freeze-remapped-anatomy",
        action="store_true",
        help="Train only neurons and edges added beyond --checkpoint-graph.",
    )
    parser.add_argument(
        "--new-visual-hemifields",
        action="store_true",
        help="Map only newly added L1 cells to fixed bilateral FPV hemifields.",
    )
    parser.add_argument(
        "--keep-remapped-roll-biases-frozen",
        action="store_true",
        help="Do not fine-tune old roll motor-neuron biases during graph expansion.",
    )
    parser.add_argument(
        "--new-pathways-roll-only",
        action="store_true",
        help="Train new-to-old boundary edges only when they enter roll motor pools.",
    )
    parser.add_argument(
        "--new-acceleration-pathways-only",
        action="store_true",
        help="Train only added edges on <=8-hop accelerometer-to-throttle paths.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "gate" / "latest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--rollout-steps", type=int, default=750)
    parser.add_argument("--bptt-window", type=int, default=150)
    parser.add_argument("--centered-imitation-iterations", type=int, default=15)
    parser.add_argument("--offset-imitation-iterations", type=int, default=20)
    parser.add_argument("--oblique-imitation-iterations", type=int, default=30)
    parser.add_argument("--centered-closed-loop-iterations", type=int, default=25)
    parser.add_argument("--offset-closed-loop-iterations", type=int, default=30)
    parser.add_argument("--closed-loop-iterations", type=int, default=50)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument(
        "--dagger-stage",
        choices=("centered", "offset", "paired", "oblique"),
        default="oblique",
    )
    parser.add_argument(
        "--evaluation-stage",
        choices=("centered", "offset", "paired", "oblique"),
        default=None,
        help="Override the final evaluation geometry; defaults to the checkpoint task.",
    )
    parser.add_argument("--dagger-collection-episodes", type=int, default=32)
    parser.add_argument("--dagger-teacher-episodes", type=int, default=8)
    parser.add_argument("--dagger-updates-per-round", type=int, default=50)
    parser.add_argument("--dagger-window-steps", type=int, default=140)
    parser.add_argument("--dagger-selection-interval", type=int, default=10)
    parser.add_argument(
        "--dagger-recent-probability",
        type=float,
        default=0.50,
        help="Probability of sampling the current student-state replay buffer.",
    )
    parser.add_argument(
        "--dagger-older-probability",
        type=float,
        default=0.25,
        help="Probability of sampling older student replay when one exists.",
    )
    parser.add_argument("--dagger-early-probability", type=float, default=0.50)
    parser.add_argument("--dagger-early-maximum-start", type=int, default=150)
    parser.add_argument(
        "--no-dagger-mirrored-pairs",
        action="store_false",
        dest="dagger_mirrored_pairs",
        help="Disable exact left/right mirror pairing in training collections.",
    )
    parser.add_argument(
        "--dagger-teacher-only",
        action="store_true",
        help="Replay only clean, exactly mirrored teacher trajectories.",
    )
    parser.add_argument(
        "--dagger-teacher-updates-per-student",
        type=int,
        default=0,
        help=(
            "Use deterministic anchored blocks: one current-student update followed "
            "by this many clean teacher updates; zero uses replay probabilities."
        ),
    )
    parser.add_argument(
        "--dagger-axis-priority",
        type=float,
        nargs=4,
        metavar=("ROLL", "PITCH", "YAW", "THROTTLE"),
        default=(2.0, 1.0, 0.25, 1.0),
    )
    parser.add_argument("--dagger-mirror-roll-weight", type=float, default=0.0)
    parser.add_argument("--recovery-rounds", type=int, default=0)
    parser.add_argument(
        "--recovery-stage",
        choices=("centered", "offset", "paired", "oblique"),
        default="paired",
    )
    parser.add_argument("--recovery-collection-episodes", type=int, default=96)
    parser.add_argument("--recovery-updates-per-round", type=int, default=30)
    parser.add_argument("--recovery-selection-interval", type=int, default=10)
    parser.add_argument("--recovery-window-steps", type=int, default=180)
    parser.add_argument("--recovery-early-emphasis-steps", type=int, default=50)
    parser.add_argument("--recovery-min-takeover-seconds", type=float, default=0.5)
    parser.add_argument("--recovery-max-takeover-seconds", type=float, default=2.5)
    parser.add_argument(
        "--recovery-clean-interval",
        type=int,
        default=3,
        help="Use one clean-teacher update at this interval; 3 gives a 2:1 recovery mix.",
    )
    parser.add_argument(
        "--recovery-roll-path-only",
        action="store_true",
        help="Fit only existing edges entering roll motor pools and anchor other outputs.",
    )
    parser.add_argument(
        "--recovery-throttle-path-only",
        action="store_true",
        help="Fit only throttle during recovery and anchor the other three outputs.",
    )
    parser.add_argument(
        "--recovery-visual-roll-path-hops",
        type=int,
        default=0,
        help=(
            "With roll-path fitting, train visual-to-roll path edges up to this "
            "length; zero trains only roll-pool inputs."
        ),
    )
    parser.add_argument("--trajectory-updates", type=int, default=0)
    parser.add_argument(
        "--trajectory-stage",
        choices=("centered", "offset", "paired", "oblique"),
        default="paired",
    )
    parser.add_argument("--trajectory-steps", type=int, default=800)
    parser.add_argument("--trajectory-selection-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument(
        "--roll-stick-gain",
        type=float,
        default=None,
        help="Fixed mechanical gain from the roll antagonist pair to the virtual stick.",
    )
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seconds", type=float, default=12.0)
    parser.add_argument(
        "--teacher-takeover-audit-seconds",
        type=float,
        nargs="*",
        default=(),
        metavar="SECONDS",
        help="Also evaluate teacher recovery after these student-controlled durations.",
    )
    parser.add_argument("--selection-episodes", type=int, default=24)
    parser.add_argument("--selection-seconds", type=float, default=10.0)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }


def controller_regularization(controller: ConnectomeController) -> torch.Tensor:
    return (
        2.0e-4 * (controller.edge_magnitude - controller.initial_edge_magnitude).square().mean()
        + 5.0e-5 * controller.bias.square().mean()
        + 2.0e-5
        * (controller.raw_time_constant - controller.initial_raw_time_constant).square().mean()
    )


def motor_imitation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    axis_weight = prediction.new_tensor((4.0, 4.0, 2.0, 4.0))
    return ((prediction - target).square() * axis_weight).mean()


def motor_contrastive_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Require observation-dependent action differences, not only a mean trajectory."""

    # Sequence tensors are [time, batch, axis].  Center each time slice across the
    # randomized batch so a shared open-loop time profile cannot satisfy this term.
    batch_dimension = -2
    return motor_imitation_loss(
        prediction - prediction.mean(dim=batch_dimension, keepdim=True),
        target - target.mean(dim=batch_dimension, keepdim=True),
    )


def curriculum_gate_config(stage: str) -> tuple[GateConfig, bool]:
    base = GateConfig()
    if stage == "centered":
        return (
            replace(
                base,
                minimum_distance=3.2,
                maximum_distance=3.8,
                minimum_lateral_offset=0.0,
                maximum_lateral_offset=0.10,
                minimum_obliquity_degrees=0.0,
                maximum_obliquity_degrees=0.0,
                minimum_height=1.00,
                maximum_height=1.15,
            ),
            False,
        )
    if stage == "offset":
        return (
            replace(
                base,
                minimum_distance=3.6,
                maximum_distance=4.5,
                minimum_lateral_offset=0.0,
                maximum_lateral_offset=0.85,
                minimum_obliquity_degrees=0.0,
                maximum_obliquity_degrees=5.0,
                minimum_height=0.95,
                maximum_height=1.20,
            ),
            False,
        )
    if stage == "paired":
        return (
            replace(
                base,
                minimum_distance=4.6,
                maximum_distance=4.6,
                minimum_lateral_offset=0.8,
                maximum_lateral_offset=0.8,
                minimum_obliquity_degrees=20.0,
                maximum_obliquity_degrees=20.0,
                minimum_height=1.1,
                maximum_height=1.1,
            ),
            True,
        )
    if stage == "oblique":
        return base, True
    raise ValueError(f"unknown curriculum stage: {stage}")


def initial_rollout(
    batch: int,
    *,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    strict: bool,
) -> tuple[QuadState, AnnularGate, torch.Tensor]:
    quad = DifferentiableQuad(hover_config).to(device)
    state = quad.initial_state(batch, device=device, dtype=torch.float32)
    gate = sample_annular_gates(
        batch,
        device=device,
        dtype=torch.float32,
        config=gate_config,
        strict=strict,
    )
    mass_scale = torch.empty(batch, device=device).uniform_(0.92, 1.08)
    return state, gate, mass_scale


def actor_teacher_gate_rc(
    controller: ConnectomeController | None,
    state: QuadState,
    gate: AnnularGate,
    hover_config: HoverConfig,
) -> torch.Tensor:
    """Choose the teacher whose inputs match the deployed actor interface."""

    if controller is not None and controller.uses_accelerometer:
        return teacher_gate_rc_with_accelerometer(state, gate, hover_config)
    return teacher_gate_rc(state, gate, hover_config)


def optimizer_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.5)
    optimizer.step()
    controller.project_parameters()


def imitation_rollout(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    batch: int,
    steps: int,
    bptt_window: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    strict: bool,
) -> dict[str, float]:
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    teacher_state, gate, mass_scale = initial_rollout(
        batch,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=strict,
    )
    teacher_sticks = sticks.initial_state(batch, device=device, dtype=torch.float32)
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    losses = []
    contrastive_losses = []
    chunk_predictions = []
    chunk_targets = []
    for step in range(steps):
        image = render_annular_gate(
            teacher_state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        predicted_motor, neural = controller(
            image,
            teacher_state.euler[:, :2],
            neural,
            teacher_state.specific_force,
        )
        with torch.no_grad():
            desired_rc = actor_teacher_gate_rc(controller, teacher_state, gate, hover_config)
            desired_motor = motor_target_for_rc(desired_rc, hover_config)
            measured_rc, teacher_sticks = sticks(desired_motor, teacher_sticks)
            teacher_state = quad(measured_rc, teacher_state, mass_scale)
        chunk_predictions.append(predicted_motor)
        chunk_targets.append(desired_motor)
        if (step + 1) % bptt_window == 0 or step + 1 == steps:
            prediction = torch.stack(chunk_predictions)
            target = torch.stack(chunk_targets)
            imitation = motor_imitation_loss(prediction, target)
            contrastive = motor_contrastive_loss(prediction, target)
            loss = 8.0 * imitation + 300.0 * contrastive + controller_regularization(controller)
            optimizer_step(controller, optimizer, loss)
            losses.append(float(imitation.detach()))
            contrastive_losses.append(float(contrastive.detach()))
            chunk_predictions = []
            chunk_targets = []
            neural = neural.detach()
    signed, lateral, vertical = gate_coordinates(teacher_state.position, gate)
    return {
        "imitation": float(np.mean(losses)),
        "contrastive_imitation": float(np.mean(contrastive_losses)),
        "teacher_signed_distance_m": float(signed.mean()),
        "teacher_radial_error_m": float(torch.sqrt(lateral.square() + vertical.square()).mean()),
    }


def closed_loop_rollout(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    batch: int,
    steps: int,
    bptt_window: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    strict: bool,
) -> dict[str, float]:
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = initial_rollout(
        batch,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=strict,
    )
    stick_state = sticks.initial_state(batch, device=device, dtype=torch.float32)
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    initial_signed, _, _ = gate_coordinates(state.position, gate)
    passed = torch.zeros(batch, dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    missed = torch.zeros_like(passed)
    imitation_values = []
    contrastive_values = []
    task_values = []
    chunk_predictions = []
    chunk_targets = []
    chunk_tasks = []
    for step in range(steps):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        motor, neural = controller(image, state.euler[:, :2], neural, state.specific_force)
        with torch.no_grad():
            desired_motor = motor_target_for_rc(
                actor_teacher_gate_rc(controller, state, gate, hover_config), hover_config
            )
        imitation = motor_imitation_loss(motor, desired_motor)
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, mass_scale)
        pass_now, collision_now, miss_now = classify_gate_crossing(
            previous_position, state.position, gate, gate_config
        )
        passed |= pass_now
        collision |= collision_now
        missed |= miss_now
        signed, lateral, vertical = gate_coordinates(state.position, gate)
        fraction = min(1.0, 1.15 * (step + 1) / steps)
        desired_signed = initial_signed + fraction * (0.8 - initial_signed)
        radial_error = lateral.square() + vertical.square()
        near_gate = torch.exp(-0.5 * signed.square())
        progress = ((signed - desired_signed) / 5.0).square().mean()
        centering = (near_gate * radial_error).mean()
        upright = state.euler[:, :2].square().mean()
        ground = torch.relu(0.15 - state.position[:, 2]).square().mean()
        task = 0.35 * progress + 1.8 * centering + 3.0 * upright + 0.5 * ground
        # On student trajectories the privileged teacher is the stable primary signal.
        # Long-horizon task gradients remain secondary so they cannot overwhelm the small
        # motor-pool losses and push the recurrent actor into unrecoverable views.
        chunk_predictions.append(motor)
        chunk_targets.append(desired_motor)
        chunk_tasks.append(task)
        imitation_values.append(float(imitation.detach()))
        task_values.append(float(task.detach()))
        if (step + 1) % bptt_window == 0 or step + 1 == steps:
            prediction = torch.stack(chunk_predictions)
            target = torch.stack(chunk_targets)
            direct = motor_imitation_loss(prediction, target)
            contrastive = motor_contrastive_loss(prediction, target)
            loss = (
                16.0 * direct
                + 300.0 * contrastive
                + 0.20 * torch.stack(chunk_tasks).mean()
                + controller_regularization(controller)
            )
            optimizer_step(controller, optimizer, loss)
            contrastive_values.append(float(contrastive.detach()))
            chunk_predictions = []
            chunk_targets = []
            chunk_tasks = []
            neural = neural.detach()
            state = state.detach()
            stick_state = stick_state.detach()
    signed, lateral, vertical = gate_coordinates(state.position, gate)
    clean_pass = passed & ~collision & ~missed
    return {
        "imitation": float(np.mean(imitation_values)),
        "contrastive_imitation": float(np.mean(contrastive_values)),
        "task": float(np.mean(task_values)),
        "signed_distance_m": float(signed.mean()),
        "radial_error_m": float(torch.sqrt(lateral.square() + vertical.square()).mean()),
        "tilt_degrees": float(
            torch.rad2deg(torch.linalg.vector_norm(state.euler[:, :2], dim=1)).mean()
        ),
        "clean_pass_rate": float(clean_pass.float().mean()),
        "ring_collision_rate": float(collision.float().mean()),
        "miss_rate": float(missed.float().mean()),
    }


def trajectory_tracking_rollout(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer | None,
    *,
    batch: int,
    steps: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    mirrored_pairs: bool,
) -> dict[str, float]:
    """Backpropagate once through a complete actor-and-plant trajectory.

    Teacher position and velocity are loss targets only.  The actor still receives just
    FPV pixels, roll, pitch, and its own connectome state.
    """

    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    teacher_state, gate, mass_scale = initial_rollout(
        batch,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    if mirrored_pairs:
        if batch % 2:
            raise ValueError("mirrored trajectory tracking requires an even batch")
        half = batch // 2
        paired_center = gate.center[:half].clone()
        paired_center[:, 1] *= -1.0
        gate = AnnularGate(
            center=torch.cat((gate.center[:half], paired_center)),
            yaw=torch.cat((gate.yaw[:half], -gate.yaw[:half])),
        )
        mass_scale = torch.cat((mass_scale[:half], mass_scale[:half]))
    student_state = QuadState(*(value.clone() for value in teacher_state.as_tuple()))
    teacher_sticks = sticks.initial_state(batch, device=device, dtype=torch.float32)
    student_sticks = sticks.initial_state(batch, device=device, dtype=torch.float32)
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    position_scale = torch.tensor((0.5, 0.5, 0.25), device=device, dtype=torch.float32)
    position_loss = torch.zeros((), device=device)
    velocity_loss = torch.zeros((), device=device)
    excessive_tilt_loss = torch.zeros((), device=device)
    passed = torch.zeros(batch, dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    missed = torch.zeros_like(passed)
    controller.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(steps):
        with torch.no_grad():
            teacher_motor = motor_target_for_rc(
                actor_teacher_gate_rc(controller, teacher_state, gate, hover_config),
                hover_config,
            )
            teacher_rc_command, teacher_sticks = sticks(teacher_motor, teacher_sticks)
            teacher_state = quad(teacher_rc_command, teacher_state, mass_scale)
        image = render_annular_gate(
            student_state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        motor, neural = controller(
            image,
            student_state.euler[:, :2],
            neural,
            student_state.specific_force,
        )
        student_rc_command, student_sticks = sticks(motor, student_sticks)
        previous_position = student_state.position
        student_state = quad(student_rc_command, student_state, mass_scale)
        normalized_position_error = (
            student_state.position - teacher_state.position
        ) / position_scale
        position_loss = position_loss + normalized_position_error.square().mean()
        velocity_loss = (
            velocity_loss + (student_state.velocity - teacher_state.velocity).square().mean()
        )
        tilt = torch.linalg.vector_norm(student_state.euler[:, :2], dim=1)
        excessive_tilt_loss = (
            excessive_tilt_loss + torch.relu(tilt - math.radians(35.0)).square().mean()
        )
        with torch.no_grad():
            pass_now, collision_now, miss_now = classify_gate_crossing(
                previous_position.detach(), student_state.position.detach(), gate, gate_config
            )
            passed |= pass_now
            collision |= collision_now
            missed |= miss_now

    position_loss = position_loss / steps
    velocity_loss = velocity_loss / steps
    excessive_tilt_loss = excessive_tilt_loss / steps
    loss = position_loss + 0.1 * velocity_loss + 0.25 * excessive_tilt_loss
    gradient_norm = torch.zeros((), device=device)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.5)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite trajectory gradient")
        optimizer.step()
        controller.project_parameters()
    signed, lateral, vertical = gate_coordinates(student_state.position.detach(), gate)
    clean_pass = passed & ~collision & ~missed
    return {
        "trajectory_loss": float(loss.detach()),
        "trajectory_position_loss": float(position_loss.detach()),
        "trajectory_velocity_loss": float(velocity_loss.detach()),
        "trajectory_excessive_tilt_loss": float(excessive_tilt_loss.detach()),
        "trajectory_gradient_norm": float(gradient_norm.detach()),
        "trajectory_peak_cuda_memory_bytes": (
            float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
        ),
        "trajectory_clean_pass_rate": float(clean_pass.float().mean()),
        "trajectory_signed_distance_m": float(signed.mean()),
        "trajectory_radial_error_m": float(torch.sqrt(lateral.square() + vertical.square()).mean()),
    }


@torch.no_grad()
def collect_dagger_trajectories(
    controller: ConnectomeController,
    *,
    episodes: int,
    steps: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    seed: int,
    teacher_driven: bool,
    mirrored_pairs: bool,
) -> DaggerTrajectories:
    """Collect one immutable sequence batch before any optimizer updates."""

    seed_everything(seed)
    controller.eval()
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = initial_rollout(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    if mirrored_pairs:
        if episodes % 2:
            raise ValueError("mirrored DAgger collection requires an even episode count")
        half = episodes // 2
        paired_center = gate.center[:half].clone()
        paired_center[:, 1] *= -1.0
        gate = AnnularGate(
            center=torch.cat((gate.center[:half], paired_center)),
            yaw=torch.cat((gate.yaw[:half], -gate.yaw[:half])),
        )
        mass_scale = torch.cat((mass_scale[:half], mass_scale[:half]))
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    images = []
    roll_pitch = []
    specific_force = []
    targets = []
    valid = []
    for _ in range(steps):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        desired_motor = motor_target_for_rc(
            actor_teacher_gate_rc(controller, state, gate, hover_config), hover_config
        )
        if teacher_driven:
            motor = desired_motor
        else:
            motor, neural = controller(image, state.euler[:, :2], neural, state.specific_force)
        images.append(image.to(dtype=torch.float16))
        roll_pitch.append(state.euler[:, :2])
        specific_force.append(state.specific_force)
        targets.append(desired_motor)
        valid.append(active.clone())
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, mass_scale)
        _, collision_now, miss_now = classify_gate_crossing(
            previous_position, state.position, gate, gate_config
        )
        active &= ~(collision_now | miss_now)
    return DaggerTrajectories(
        images=torch.stack(images),
        roll_pitch=torch.stack(roll_pitch),
        specific_force=torch.stack(specific_force),
        targets=torch.stack(targets),
        valid=torch.stack(valid),
        lateral_offset=gate.center[:, 1],
        teacher_driven=teacher_driven,
        mirrored_pairs=mirrored_pairs,
    )


@torch.no_grad()
def collect_takeover_recovery_trajectories(
    controller: ConnectomeController,
    *,
    episodes: int,
    steps: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    seed: int,
    minimum_takeover_seconds: float,
    maximum_takeover_seconds: float,
    recovery_window_steps: int,
    mirrored_pairs: bool,
) -> RecoveryTrajectories:
    """Collect recoveries after variable student-controlled prefixes.

    The teacher changes only the simulated stick commands after takeover.  Actor
    observations and recurrent state continue normally, and no teacher state is saved
    in the deployable checkpoint.
    """

    seed_everything(seed)
    controller.eval()
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = initial_rollout(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    minimum_step = round(minimum_takeover_seconds / hover_config.dt)
    maximum_step = round(maximum_takeover_seconds / hover_config.dt)
    sampled_episodes = episodes // 2 if mirrored_pairs else episodes
    takeover_steps = torch.randint(
        minimum_step,
        maximum_step + 1,
        (sampled_episodes,),
        device=device,
    )
    if mirrored_pairs:
        if episodes % 2:
            raise ValueError("mirrored recovery collection requires an even episode count")
        half = episodes // 2
        paired_center = gate.center[:half].clone()
        paired_center[:, 1] *= -1.0
        gate = AnnularGate(
            center=torch.cat((gate.center[:half], paired_center)),
            yaw=torch.cat((gate.yaw[:half], -gate.yaw[:half])),
        )
        mass_scale = torch.cat((mass_scale[:half], mass_scale[:half]))
        takeover_steps = torch.cat((takeover_steps, takeover_steps))

    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    passed = torch.zeros_like(active)
    collision = torch.zeros_like(active)
    missed = torch.zeros_like(active)
    images = []
    roll_pitch = []
    specific_force = []
    targets = []
    baseline_motor = []
    valid = []
    for step in range(steps):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        desired_motor = motor_target_for_rc(
            actor_teacher_gate_rc(controller, state, gate, hover_config), hover_config
        )
        student_motor, neural = controller(image, state.euler[:, :2], neural, state.specific_force)
        teacher_has_control = step >= takeover_steps
        motor = torch.where(teacher_has_control[:, None], desired_motor, student_motor)
        in_recovery_window = teacher_has_control & (step < takeover_steps + recovery_window_steps)
        images.append(image.to(dtype=torch.float16))
        roll_pitch.append(state.euler[:, :2])
        specific_force.append(state.specific_force)
        targets.append(desired_motor)
        baseline_motor.append(student_motor)
        valid.append(active & in_recovery_window)
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, mass_scale)
        pass_now, collision_now, miss_now = classify_gate_crossing(
            previous_position, state.position, gate, gate_config
        )
        passed |= pass_now
        collision |= collision_now
        missed |= miss_now
        active &= ~(collision_now | miss_now)

    teacher_recovery_clean_crossing = passed & ~collision & ~missed
    return RecoveryTrajectories(
        images=torch.stack(images),
        roll_pitch=torch.stack(roll_pitch),
        specific_force=torch.stack(specific_force),
        targets=torch.stack(targets),
        baseline_motor=torch.stack(baseline_motor),
        valid=torch.stack(valid),
        lateral_offset=gate.center[:, 1],
        takeover_steps=takeover_steps,
        mirrored_pairs=mirrored_pairs,
        teacher_recovery_clean_crossing_rate=float(teacher_recovery_clean_crossing.float().mean()),
    )


def masked_motor_imitation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    axis_weight: torch.Tensor,
) -> torch.Tensor:
    weighted_error = (prediction - target).square() * axis_weight
    weight = valid.to(dtype=prediction.dtype).unsqueeze(-1)
    return (weighted_error * weight).sum() / (4.0 * weight.sum()).clamp_min(1.0)


def recovery_replay_update(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source: RecoveryTrajectories,
    *,
    batch: int,
    window_steps: int,
    early_emphasis_steps: int,
    axis_weight: torch.Tensor,
    path_axis: int | None = None,
) -> dict[str, float]:
    """Fit teacher corrections immediately after per-episode takeover times."""

    _, source_episodes = source.valid.shape
    paired_selection = source.mirrored_pairs and batch % 2 == 0
    if paired_selection:
        source_half = source_episodes // 2
        candidates = torch.arange(source_half, device=source.images.device)
        candidates = candidates[
            source.valid[:, :source_half].any(dim=0) & source.valid[:, source_half:].any(dim=0)
        ]
        if not candidates.numel():
            raise RuntimeError("recovery replay has no valid mirrored episode pairs")
        base = candidates[
            torch.randint(candidates.numel(), (batch // 2,), device=candidates.device)
        ]
        selected = torch.cat((base, base + source_half))
    else:
        candidates = torch.nonzero(source.valid.any(dim=0), as_tuple=False).squeeze(-1)
        if not candidates.numel():
            raise RuntimeError("recovery replay has no valid episodes")
        selected = candidates[torch.randint(candidates.numel(), (batch,), device=candidates.device)]

    starts = source.takeover_steps[selected]
    maximum_window = int(source.valid.shape[0] - starts.max().item())
    window_steps = min(window_steps, maximum_window)
    neural = controller.initial_state(batch, device=source.images.device, dtype=torch.float32)
    controller.train()
    prefix_anchor_sum = torch.zeros((), device=source.images.device)
    prefix_anchor_count = torch.zeros((), device=source.images.device)
    if path_axis is not None:
        for step in range(int(starts.max().item())):
            prefix_motor, candidate_neural = controller(
                source.images[step, selected].float(),
                source.roll_pitch[step, selected],
                neural,
                source.specific_force[step, selected],
            )
            prefix_active = step < starts
            prefix_weight = prefix_active.to(dtype=prefix_motor.dtype)
            prefix_anchor_sum = (
                prefix_anchor_sum
                + (
                    (prefix_motor - source.baseline_motor[step, selected]).square()
                    * prefix_weight[:, None]
                ).sum()
            )
            prefix_anchor_count = prefix_anchor_count + 4.0 * prefix_weight.sum()
            neural = torch.where(prefix_active[:, None], candidate_neural, neural)
    else:
        with torch.no_grad():
            for step in range(int(starts.max().item())):
                _, candidate_neural = controller(
                    source.images[step, selected].float(),
                    source.roll_pitch[step, selected],
                    neural,
                    source.specific_force[step, selected],
                )
                neural = torch.where((step < starts)[:, None], candidate_neural, neural)
    if path_axis is None:
        neural = neural.detach()
    predictions = []
    targets = []
    baselines = []
    validity = []
    for relative_step in range(window_steps):
        time_index = starts + relative_step
        motor, neural = controller(
            source.images[time_index, selected].float(),
            source.roll_pitch[time_index, selected],
            neural,
            source.specific_force[time_index, selected],
        )
        predictions.append(motor)
        targets.append(source.targets[time_index, selected])
        baselines.append(source.baseline_motor[time_index, selected])
        validity.append(source.valid[time_index, selected])
    prediction = torch.stack(predictions)
    target = torch.stack(targets)
    baseline = torch.stack(baselines)
    valid = torch.stack(validity)
    emphasis = torch.ones(window_steps, 1, device=prediction.device, dtype=prediction.dtype)
    emphasis[: min(early_emphasis_steps, window_steps)] = 2.0
    sample_weight = valid.to(dtype=prediction.dtype) * emphasis
    if path_axis is not None:
        imitation = (
            (prediction[..., path_axis] - target[..., path_axis]).square() * sample_weight
        ).sum() / sample_weight.sum().clamp_min(1.0)
        other_axes = [axis for axis in range(4) if axis != path_axis]
        non_roll_anchor = (
            (prediction[..., other_axes] - baseline[..., other_axes]).square()
            * sample_weight[..., None]
        ).sum() / (3.0 * sample_weight.sum()).clamp_min(1.0)
        prefix_anchor = prefix_anchor_sum / prefix_anchor_count.clamp_min(1.0)
        pair_roll = prediction.new_zeros(())
        if paired_selection and path_axis == 0:
            half = batch // 2
            pair_weight = torch.minimum(sample_weight[:, :half], sample_weight[:, half:])
            prediction_difference = prediction[:, :half, 0] - prediction[:, half:, 0]
            target_difference = target[:, :half, 0] - target[:, half:, 0]
            pair_roll = (
                (prediction_difference - target_difference).square() * pair_weight
            ).sum() / pair_weight.sum().clamp_min(1.0)
    else:
        weighted_error = (prediction - target).square() * axis_weight
        imitation = (weighted_error * sample_weight[..., None]).sum() / (
            4.0 * sample_weight.sum()
        ).clamp_min(1.0)
        non_roll_anchor = prediction.new_zeros(())
        prefix_anchor = prediction.new_zeros(())
        pair_roll = prediction.new_zeros(())
    axis_mse = ((prediction - target).square() * sample_weight[..., None]).sum(
        dim=(0, 1)
    ) / sample_weight.sum().clamp_min(1.0)
    early_steps = min(early_emphasis_steps, window_steps)
    early_valid = valid[:early_steps].to(dtype=prediction.dtype)
    diagnostic_axis = path_axis if path_axis is not None else 0
    early_axis_mse = (
        (
            prediction[:early_steps, :, diagnostic_axis] - target[:early_steps, :, diagnostic_axis]
        ).square()
        * early_valid
    ).sum() / early_valid.sum().clamp_min(1.0)
    # The checkpoint is already a trained anatomical operating point.  The legacy
    # regularizer targets the graph's untrained initialization and would therefore
    # erase the warm-start policy during fine-tuning.
    loss = 12.0 * imitation + 12.0 * pair_roll + 12.0 * non_roll_anchor + 4.0 * prefix_anchor
    optimizer_step(controller, optimizer, loss)
    return {
        "recovery_imitation": float(imitation.detach()),
        "recovery_valid_fraction": float(valid.float().mean()),
        "recovery_early_fitted_axis_mse": float(early_axis_mse.detach()),
        "recovery_fitted_axis": float(diagnostic_axis),
        "recovery_non_roll_anchor_mse": float(non_roll_anchor.detach()),
        "recovery_prefix_anchor_mse": float(prefix_anchor.detach()),
        "recovery_pair_roll_mse": float(pair_roll.detach()),
        "recovery_roll_mse": float(axis_mse[0].detach()),
        "recovery_pitch_mse": float(axis_mse[1].detach()),
        "recovery_yaw_mse": float(axis_mse[2].detach()),
        "recovery_throttle_mse": float(axis_mse[3].detach()),
        "recovery_takeover_mean_seconds": float(starts.float().mean() * controller.neural_dt),
        "recovery_paired_selection": float(paired_selection),
    }


def calibrate_dagger_axis_weight(
    sources: tuple[DaggerTrajectories | RecoveryTrajectories, ...],
    priority_values: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze variance-normalized task priorities from balanced training rollouts."""

    samples = torch.cat([source.targets[source.valid] for source in sources])
    scale = samples.std(dim=0, unbiased=False).clamp_min(0.01)
    priority = scale.new_tensor(priority_values)
    weight = priority / scale.square()
    weight = weight * (3.5 / weight.mean())
    return scale, weight


def dagger_replay_update(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source: DaggerTrajectories,
    *,
    batch: int,
    window_steps: int,
    axis_weight: torch.Tensor,
    early_probability: float,
    early_maximum_start: int,
    mirror_roll_weight: float,
) -> dict[str, float]:
    """Rebuild current recurrent state, then fit a contiguous labelled window."""

    steps, source_episodes = source.valid.shape
    window_steps = min(window_steps, steps)
    maximum_start = steps - window_steps
    if maximum_start == 0:
        start = 0
    elif random.random() < early_probability:
        start = random.randint(0, min(maximum_start, early_maximum_start))
    else:
        approach_start = min(200, maximum_start)
        start = random.randint(approach_start, maximum_start)

    candidates = torch.nonzero(source.valid[start], as_tuple=False).squeeze(-1)
    if candidates.numel() == 0:
        start = 0
        candidates = torch.arange(source_episodes, device=source.valid.device)
    paired_selection = False
    if source.mirrored_pairs and batch % 2 == 0:
        source_half = source_episodes // 2
        paired_candidates = torch.arange(source_half, device=candidates.device)
        paired_candidates = paired_candidates[
            source.valid[start, paired_candidates]
            & source.valid[start, paired_candidates + source_half]
        ]
        if paired_candidates.numel():
            base = paired_candidates[
                torch.randint(
                    paired_candidates.numel(),
                    (batch // 2,),
                    device=candidates.device,
                )
            ]
            selected = torch.cat((base, base + source_half))
            paired_selection = True
        else:
            selected = candidates[
                torch.randint(candidates.numel(), (batch,), device=candidates.device)
            ]
    else:
        negative = candidates[source.lateral_offset[candidates] < 0.0]
        positive = candidates[source.lateral_offset[candidates] >= 0.0]
        if negative.numel() and positive.numel():
            negative_count = batch // 2
            selected = torch.cat(
                (
                    negative[
                        torch.randint(negative.numel(), (negative_count,), device=candidates.device)
                    ],
                    positive[
                        torch.randint(
                            positive.numel(),
                            (batch - negative_count,),
                            device=candidates.device,
                        )
                    ],
                )
            )
            selected = selected[torch.randperm(batch, device=candidates.device)]
        else:
            selected = candidates[
                torch.randint(candidates.numel(), (batch,), device=candidates.device)
            ]
    neural = controller.initial_state(batch, device=source.images.device, dtype=torch.float32)
    controller.train()
    with torch.no_grad():
        for step in range(start):
            _, neural = controller(
                source.images[step, selected].float(),
                source.roll_pitch[step, selected],
                neural,
                source.specific_force[step, selected],
            )
    neural = neural.detach()
    predictions = []
    for step in range(start, start + window_steps):
        motor, neural = controller(
            source.images[step, selected].float(),
            source.roll_pitch[step, selected],
            neural,
            source.specific_force[step, selected],
        )
        predictions.append(motor)
    prediction = torch.stack(predictions)
    target = source.targets[start : start + window_steps, selected]
    valid = source.valid[start : start + window_steps, selected]
    imitation = masked_motor_imitation_loss(prediction, target, valid, axis_weight=axis_weight)
    valid_weight = valid.to(dtype=prediction.dtype).unsqueeze(-1)
    axis_mse = ((prediction - target).square() * valid_weight).sum(
        dim=(0, 1)
    ) / valid_weight.sum().clamp_min(1.0)
    mirror_roll = prediction.new_zeros(())
    mirror_roll_mean = prediction.new_zeros(())
    if paired_selection:
        half = batch // 2
        pair_valid = valid[:, :half] & valid[:, half:]
        pair_prediction = prediction[:, :half, 0] - prediction[:, half:, 0]
        pair_target = target[:, :half, 0] - target[:, half:, 0]
        pair_prediction_mean = 0.5 * (prediction[:, :half, 0] + prediction[:, half:, 0])
        pair_target_mean = 0.5 * (target[:, :half, 0] + target[:, half:, 0])
        pair_weight = pair_valid.to(dtype=prediction.dtype)
        mirror_roll = (
            (pair_prediction - pair_target).square() * pair_weight
        ).sum() / pair_valid.sum().clamp_min(1)
        mirror_roll_mean = (
            (pair_prediction_mean - pair_target_mean).square() * pair_weight
        ).sum() / pair_valid.sum().clamp_min(1)
    applied_mirror_roll_weight = mirror_roll_weight if source.teacher_driven else 0.0
    loss = (
        12.0 * imitation
        + applied_mirror_roll_weight * mirror_roll
        + controller_regularization(controller)
    )
    optimizer_step(controller, optimizer, loss)
    return {
        "dagger_imitation": float(imitation.detach()),
        "dagger_valid_fraction": float(valid.float().mean()),
        "dagger_window_start": float(start),
        "dagger_teacher_source": float(source.teacher_driven),
        "dagger_roll_mse": float(axis_mse[0].detach()),
        "dagger_pitch_mse": float(axis_mse[1].detach()),
        "dagger_yaw_mse": float(axis_mse[2].detach()),
        "dagger_throttle_mse": float(axis_mse[3].detach()),
        "dagger_mirror_roll_mse": float(mirror_roll.detach()),
        "dagger_mirror_roll_mean_mse": float(mirror_roll_mean.detach()),
        "dagger_paired_selection": float(paired_selection),
        "dagger_mirror_roll_applied": float(applied_mirror_roll_weight > 0.0),
    }


def _stratum_rate(success: torch.Tensor, mask: torch.Tensor) -> float | None:
    return float(success[mask].float().mean()) if bool(mask.any()) else None


def gate_selection_score(metrics: dict[str, Any], *, balance_mass: bool = False) -> float:
    """Rank checkpoints by hard held-out flights; larger is better."""

    stratum_names = (
        ("lower_mass", "higher_mass")
        if balance_mass
        else ("negative_lateral_offset", "positive_lateral_offset")
    )
    rates = tuple(metrics["success_by_stratum"][name] for name in stratum_names)
    worst_stratum_rate = min(rate for rate in rates if rate is not None)
    return float(
        100.0 * metrics["success_rate"]
        + 100.0 * worst_stratum_rate
        + metrics["pass_rate"]
        + 0.5 * metrics["lift_off_rate"]
        - metrics["ring_collision_rate"]
        - 0.2 * metrics["miss_rate"]
    )


def roll_action_diagnostic(
    prediction: torch.Tensor, target: torch.Tensor, negative_offset: torch.Tensor
) -> dict[str, Any] | None:
    if prediction.numel() < 2:
        return None
    prediction_mean = prediction.mean()
    target_mean = target.mean()
    covariance = ((prediction - prediction_mean) * (target - target_mean)).mean()
    prediction_variance = (prediction - prediction_mean).square().mean()
    target_variance = (target - target_mean).square().mean()

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
        return float(values[mask].mean()) if bool(mask.any()) else None

    return {
        "samples": int(prediction.numel()),
        "prediction_mean": float(prediction_mean),
        "target_mean": float(target_mean),
        "prediction_std": float(torch.sqrt(prediction_variance)),
        "target_std": float(torch.sqrt(target_variance)),
        "correlation": float(
            covariance / torch.sqrt(prediction_variance * target_variance).clamp_min(1.0e-12)
        ),
        "regression_slope": float(covariance / target_variance.clamp_min(1.0e-12)),
        "meaningful_sign_accuracy": float(
            (torch.sign(prediction) == torch.sign(target)).float().mean()
        ),
        "prediction_mean_negative_offset": masked_mean(prediction, negative_offset),
        "target_mean_negative_offset": masked_mean(target, negative_offset),
        "prediction_mean_positive_offset": masked_mean(prediction, ~negative_offset),
        "target_mean_positive_offset": masked_mean(target, ~negative_offset),
    }


@torch.no_grad()
def evaluate_gate(
    controller: ConnectomeController | None,
    *,
    episodes: int,
    seconds: float,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    seed: int,
    teacher: bool = False,
    frozen_visual: bool = False,
    frozen_acceleration: bool = False,
    swapped_acceleration: bool = False,
    disabled_acceleration_channel: str | None = None,
    frozen_proprioception: bool = False,
    swapped_proprioception: bool = False,
    privileged_mass_trim: tuple[float, float] | None = None,
    shuffled_privileged_mass: bool = False,
    balanced_strata: bool = False,
    teacher_takeover_at_seconds: float | None = None,
) -> dict[str, Any]:
    if disabled_acceleration_channel not in (None, "above_1g", "below_1g"):
        raise ValueError("disabled acceleration channel must be above_1g or below_1g")
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = initial_rollout(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    if balanced_strata:
        if episodes % 8:
            raise ValueError("balanced mass/side/obliquity evaluation requires episodes % 8 == 0")
        codes = torch.arange(episodes, device=device) % 8
        mass_sign = torch.where(codes.bitwise_and(1).bool(), 1.0, -1.0)
        lateral_sign = torch.where(codes.bitwise_and(2).bool(), 1.0, -1.0)
        obliquity_sign = torch.where(codes.bitwise_and(4).bool(), 1.0, -1.0)
        center = gate.center.clone()
        center[:, 1] = center[:, 1].abs() * lateral_sign
        bearing = torch.atan2(center[:, 1], center[:, 0])
        sampled_bearing = torch.atan2(gate.center[:, 1], gate.center[:, 0])
        obliquity_magnitude = wrap_angle(gate.yaw - sampled_bearing).abs()
        gate = AnnularGate(center=center, yaw=bearing + obliquity_sign * obliquity_magnitude)
        mass_scale = 1.0 + mass_sign * (mass_scale - 1.0).abs()
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = (
        controller.initial_state(episodes, device=device, dtype=torch.float32)
        if controller is not None
        else None
    )
    initial_image = render_annular_gate(
        state,
        gate,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    step_count = round(seconds / hover_config.dt)
    passed = torch.zeros(episodes, dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    missed = torch.zeros_like(passed)
    lifted = torch.zeros_like(passed)
    recontact = torch.zeros_like(passed)
    cleared = torch.zeros_like(passed)
    pass_step = torch.full((episodes,), -1, dtype=torch.long, device=device)
    crossing_clearance = torch.full((episodes,), float("nan"), device=device)
    crossing_angle = torch.full((episodes,), float("nan"), device=device)
    crossing_radial = torch.full((episodes,), float("nan"), device=device)
    crossing_lateral = torch.full((episodes,), float("nan"), device=device)
    crossing_vertical = torch.full((episodes,), float("nan"), device=device)
    max_tilt = torch.zeros(episodes, device=device)
    saturation_steps = torch.zeros(episodes, device=device)
    diagnostic_prediction = []
    diagnostic_target = []
    diagnostic_negative_offset = []
    acceleration_permutation = torch.empty(episodes, dtype=torch.long, device=device)
    mass_order = torch.argsort(mass_scale)
    acceleration_permutation[mass_order] = torch.flip(mass_order, dims=(0,))
    privileged_mass_code = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    if shuffled_privileged_mass:
        if not balanced_strata:
            raise ValueError("shuffled mass oracle requires balanced geometry strata")
        shuffled_code = privileged_mass_code.clone()
        geometry_code = codes.bitwise_and(6)
        for stratum in (0, 2, 4, 6):
            indices = torch.nonzero(geometry_code == stratum, as_tuple=False).squeeze(-1)
            shuffled_code[indices] = privileged_mass_code[indices].roll(1)
        privileged_mass_code = shuffled_code
    for step in range(step_count):
        previous_position = state.position
        teacher_has_control = teacher or (
            teacher_takeover_at_seconds is not None
            and step >= round(teacher_takeover_at_seconds / hover_config.dt)
        )
        if teacher_has_control:
            desired_rc = actor_teacher_gate_rc(controller, state, gate, hover_config)
            motor = motor_target_for_rc(desired_rc, hover_config)
        else:
            assert controller is not None and neural is not None
            image = (
                initial_image
                if frozen_visual
                else render_annular_gate(
                    state,
                    gate,
                    resolution=resolution,
                    hover_config=hover_config,
                    gate_config=gate_config,
                )
            )
            specific_force = state.specific_force
            if frozen_acceleration:
                specific_force = torch.zeros_like(specific_force)
                specific_force[:, 2] = 9.81
            elif swapped_acceleration:
                specific_force = specific_force[acceleration_permutation]
            if disabled_acceleration_channel is not None:
                specific_force = specific_force.clone()
                if disabled_acceleration_channel == "above_1g":
                    specific_force[:, 2].clamp_(max=9.81)
                else:
                    specific_force[:, 2].clamp_(min=9.81)
            stick_position = stick_state.position
            if frozen_proprioception:
                stick_position = torch.zeros_like(stick_position)
                stick_position[:, 3] = -1.0
            elif swapped_proprioception:
                stick_position = stick_position[acceleration_permutation]
            privileged_bias = None
            if privileged_mass_trim is not None:
                intercept, slope = privileged_mass_trim
                privileged_bias = intercept + slope * privileged_mass_code
            motor, neural = controller(
                image,
                state.euler[:, :2],
                neural,
                specific_force,
                stick_position,
                privileged_throttle_pool_bias=privileged_bias,
            )
            if not frozen_visual and step < round(1.0 / hover_config.dt):
                target_motor = motor_target_for_rc(
                    actor_teacher_gate_rc(controller, state, gate, hover_config),
                    hover_config,
                )
                meaningful = target_motor[:, 0].abs() > 0.003
                diagnostic_prediction.append(motor[meaningful, 0])
                diagnostic_target.append(target_motor[meaningful, 0])
                diagnostic_negative_offset.append(gate.center[meaningful, 1] < 0.0)
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)
        pass_now, collision_now, miss_now = classify_gate_crossing(
            previous_position, state.position, gate, gate_config
        )
        new_crossing = (pass_now | collision_now | miss_now) & crossing_radial.isnan()
        new_pass = pass_now & ~passed
        if bool(new_crossing.any()):
            _, lateral, vertical = crossing_coordinates(previous_position, state.position, gate)
            radial = torch.sqrt(lateral.square() + vertical.square())
            crossing_radial[new_crossing] = radial[new_crossing]
            crossing_lateral[new_crossing] = lateral[new_crossing]
            crossing_vertical[new_crossing] = vertical[new_crossing]
            velocity_direction = state.velocity / torch.linalg.vector_norm(
                state.velocity, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            angle = torch.acos((velocity_direction * gate.normal).sum(dim=1).clamp(-1.0, 1.0))
            pass_step[new_pass] = step + 1
            crossing_clearance[new_pass] = (
                gate_config.inner_radius - gate_config.drone_radius - radial[new_pass]
            )
            crossing_angle[new_pass] = angle[new_pass]
        passed |= pass_now
        collision |= collision_now
        missed |= miss_now
        signed, _, _ = gate_coordinates(state.position, gate)
        cleared |= passed & (signed >= 0.4)
        lifted |= state.position[:, 2] > 0.15
        recontact |= lifted & (state.position[:, 2] <= 0.01)
        max_tilt = torch.maximum(max_tilt, torch.linalg.vector_norm(state.euler[:, :2], dim=1))
        saturation_steps += (stick_state.position.abs() > 0.98).any(dim=1)

    has_post_pass_second = (pass_step >= 0) & (
        pass_step <= step_count - round(1.0 / hover_config.dt)
    )
    sustained_saturation = saturation_steps / step_count > 0.25
    success = (
        passed
        & cleared
        & has_post_pass_second
        & ~collision
        & ~missed
        & ~recontact
        & (max_tilt <= math.radians(40.0))
        & ~sustained_saturation
    )
    centre_offset, obliquity = initial_gate_geometry(gate)
    low_offset = gate.center[:, 1].abs() < 0.8
    low_obliquity = obliquity < math.radians(20.0)
    negative_offset = gate.center[:, 1] < 0.0
    near_gate = gate.center[:, 0] < (
        0.5 * (gate_config.minimum_distance + gate_config.maximum_distance)
    )
    low_gate = gate.center[:, 2] < (0.5 * (gate_config.minimum_height + gate_config.maximum_height))
    valid_times = pass_step[passed].float() * hover_config.dt
    valid_clearance = crossing_clearance[passed]
    valid_angles = crossing_angle[passed]
    valid_radial = crossing_radial[~crossing_radial.isnan()]
    valid_lateral = crossing_lateral[~crossing_lateral.isnan()]
    valid_vertical = crossing_vertical[~crossing_vertical.isnan()]
    crossing_mask = ~crossing_vertical.isnan()
    if valid_vertical.numel() > 1:
        centered_mass = mass_scale[crossing_mask] - mass_scale[crossing_mask].mean()
        centered_vertical = valid_vertical - valid_vertical.mean()
        mass_vertical_correlation = float(
            (centered_mass * centered_vertical).mean()
            / (
                centered_mass.square().mean().sqrt() * centered_vertical.square().mean().sqrt()
            ).clamp_min(1.0e-12)
        )
    else:
        mass_vertical_correlation = None
    success_rate = float(success.float().mean())
    action_diagnostic = None
    if diagnostic_prediction:
        action_diagnostic = roll_action_diagnostic(
            torch.cat(diagnostic_prediction),
            torch.cat(diagnostic_target),
            torch.cat(diagnostic_negative_offset),
        )
    return {
        "label": (
            "privileged_teacher_through_foreleg_sticks"
            if teacher
            else f"connectome_then_teacher_at_{teacher_takeover_at_seconds}s"
            if teacher_takeover_at_seconds is not None
            else "frozen_initial_image"
            if frozen_visual
            else "constant_1g_accelerometer"
            if frozen_acceleration
            else "mass_rank_swapped_accelerometer"
            if swapped_acceleration
            else f"disabled_{disabled_acceleration_channel}_accelerometer_channel"
            if disabled_acceleration_channel is not None
            else "constant_initial_throttle_joint_position"
            if frozen_proprioception
            else "mass_rank_swapped_throttle_joint_position"
            if swapped_proprioception
            else "shuffled_privileged_mass_conditioned_throttle_trim"
            if privileged_mass_trim is not None and shuffled_privileged_mass
            else "privileged_mass_conditioned_throttle_trim"
            if privileged_mass_trim is not None
            else "connectome"
        ),
        "episodes": episodes,
        "seconds": seconds,
        "success_rate": success_rate,
        "pass_rate": float(passed.float().mean()),
        "clearance_rate": float(cleared.float().mean()),
        "plane_crossing_rate": float((~crossing_radial.isnan()).float().mean()),
        "ring_collision_rate": float(collision.float().mean()),
        "miss_rate": float(missed.float().mean()),
        "lift_off_rate": float(lifted.float().mean()),
        "ground_recontact_rate": float(recontact.float().mean()),
        "sustained_stick_saturation_rate": float(sustained_saturation.float().mean()),
        "max_tilt_mean_degrees": float(torch.rad2deg(max_tilt).mean()),
        "pass_time_mean_seconds": float(valid_times.mean()) if bool(passed.any()) else None,
        "pass_time_p95_seconds": (
            float(torch.quantile(valid_times, 0.95)) if bool(passed.any()) else None
        ),
        "crossing_clearance_mean_m": (
            float(valid_clearance.mean()) if bool(passed.any()) else None
        ),
        "crossing_angle_mean_degrees": (
            float(torch.rad2deg(valid_angles).mean()) if bool(passed.any()) else None
        ),
        "crossing_radial_quantiles_m": (
            {
                str(percentile): float(valid_radial.quantile(percentile / 100.0))
                for percentile in (50, 75, 90, 95, 99)
            }
            if valid_radial.numel()
            else None
        ),
        "crossing_radial_mean_m": (float(valid_radial.mean()) if valid_radial.numel() else None),
        "lateral_aperture_exceedance_rate": (
            float(
                (valid_lateral.abs() > gate_config.inner_radius - gate_config.drone_radius)
                .float()
                .mean()
            )
            if valid_lateral.numel()
            else None
        ),
        "crossing_error_components_m": (
            {
                "lateral_signed_mean": float(valid_lateral.mean()),
                "vertical_signed_mean": float(valid_vertical.mean()),
                "lateral_absolute_mean": float(valid_lateral.abs().mean()),
                "vertical_absolute_mean": float(valid_vertical.abs().mean()),
                "lateral_absolute_p90": float(valid_lateral.abs().quantile(0.90)),
                "vertical_absolute_p90": float(valid_vertical.abs().quantile(0.90)),
            }
            if valid_lateral.numel()
            else None
        ),
        "mass_vertical_error_correlation": mass_vertical_correlation,
        "initial_visual_centre_offset_min_degrees": float(torch.rad2deg(centre_offset).min()),
        "gate_obliquity_min_degrees": float(torch.rad2deg(obliquity).min()),
        "success_by_stratum": {
            "lower_offset": _stratum_rate(success, low_offset),
            "higher_offset": _stratum_rate(success, ~low_offset),
            "negative_lateral_offset": _stratum_rate(success, negative_offset),
            "positive_lateral_offset": _stratum_rate(success, ~negative_offset),
            "obliquity_10_to_20_degrees": _stratum_rate(success, low_obliquity),
            "obliquity_20_to_30_degrees": _stratum_rate(success, ~low_obliquity),
            "nearer_gate": _stratum_rate(success, near_gate),
            "farther_gate": _stratum_rate(success, ~near_gate),
            "lower_gate": _stratum_rate(success, low_gate),
            "higher_gate": _stratum_rate(success, ~low_gate),
            "lower_mass": _stratum_rate(success, mass_scale < 1.0),
            "higher_mass": _stratum_rate(success, mass_scale >= 1.0),
        },
        "early_roll_action_diagnostic": action_diagnostic,
        "goal_pass": success_rate >= 0.90,
    }


def save_checkpoint(
    path: Path,
    controller: ConnectomeController,
    args: argparse.Namespace,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    *,
    source_checkpoint_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "task_version": GATE_TASK_VERSION,
            "plant_model_version": PLANT_MODEL_VERSION,
            "controller": controller.state_dict(),
            "graph_sha256": file_sha256(args.graph),
            "hover_config": asdict(hover_config),
            "gate_config": asdict(gate_config),
            "retinal_receptive_field": RETINAL_RECEPTIVE_FIELD,
            "retinal_flip_x": RETINAL_FLIP_X,
            "image_resolution": args.resolution,
            "seed": args.seed,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "training_arguments": json_arguments(args),
        },
        path,
    )


def edges_on_short_paths(
    edge_pre: np.ndarray,
    edge_post: np.ndarray,
    node_count: int,
    sources: np.ndarray,
    targets: np.ndarray,
    max_hops: int,
) -> np.ndarray:
    """Select directed edges belonging to a source-to-target path within a hop cap."""

    unreachable = node_count + 1
    forward = np.full(node_count, unreachable, dtype=np.int32)
    backward = np.full(node_count, unreachable, dtype=np.int32)
    forward[sources] = 0
    backward[targets] = 0
    for _ in range(max_hops):
        previous = forward.copy()
        np.minimum.at(forward, edge_post, previous[edge_pre] + 1)
        previous = backward.copy()
        np.minimum.at(backward, edge_pre, previous[edge_post] + 1)
    return forward[edge_pre] + 1 + backward[edge_post] <= max_hops


def remap_anatomical_parameters(
    controller: ConnectomeController,
    checkpoint: dict[str, Any],
    checkpoint_graph: Path,
    destination_graph: Path,
    *,
    freeze_existing: bool,
    new_visual_hemifields: bool,
    keep_roll_biases_frozen: bool,
    new_pathways_roll_only: bool,
    new_acceleration_pathways_only: bool,
) -> dict[str, Any]:
    """Copy learned values only where source and destination anatomy are identical."""

    old_graph = np.load(checkpoint_graph)
    new_graph = np.load(destination_graph)
    old_node_ids = old_graph["node_ids"]
    saved_node_ids = checkpoint["controller"]["node_ids"].detach().cpu().numpy()
    if not np.array_equal(old_node_ids, saved_node_ids):
        raise SystemExit("--checkpoint-graph node IDs do not match checkpoint buffers")
    new_node_ids = controller.node_ids.detach().cpu().numpy()
    old_node_index = {int(body): index for index, body in enumerate(old_node_ids)}
    common_new_indices = []
    common_old_indices = []
    for new_index, body in enumerate(new_node_ids):
        old_index = old_node_index.get(int(body))
        if old_index is not None:
            common_new_indices.append(new_index)
            common_old_indices.append(old_index)

    old_edge_pre = old_graph["edge_pre"]
    old_edge_post = old_graph["edge_post"]
    old_edge_index = {
        (int(old_node_ids[pre]), int(old_node_ids[post])): index
        for index, (pre, post) in enumerate(zip(old_edge_pre, old_edge_post, strict=True))
    }
    new_edge_pre = controller.edge_pre.detach().cpu().numpy()
    new_edge_post = controller.edge_post.detach().cpu().numpy()
    old_node_set = set(map(int, old_node_ids))
    common_new_edges = []
    common_old_edges = []
    for new_index, (pre, post) in enumerate(zip(new_edge_pre, new_edge_post, strict=True)):
        old_index = old_edge_index.get((int(new_node_ids[pre]), int(new_node_ids[post])))
        if old_index is not None:
            common_new_edges.append(new_index)
            common_old_edges.append(old_index)

    old_visual_nodes = old_graph["visual_node_indices"]
    old_visual_index = {
        int(old_node_ids[node]): visual_index for visual_index, node in enumerate(old_visual_nodes)
    }
    new_visual_nodes = controller.visual_nodes.detach().cpu().numpy()
    new_visual_eye = new_graph["visual_eye"]
    common_new_visual = []
    common_old_visual = []
    for visual_index, node in enumerate(new_visual_nodes):
        old_index = old_visual_index.get(int(new_node_ids[node]))
        if old_index is not None:
            common_new_visual.append(visual_index)
            common_old_visual.append(old_index)

    device = controller.bias.device
    new_nodes = torch.tensor(common_new_indices, device=device, dtype=torch.long)
    old_nodes = torch.tensor(common_old_indices, device=device, dtype=torch.long)
    new_edges = torch.tensor(common_new_edges, device=device, dtype=torch.long)
    old_edges = torch.tensor(common_old_edges, device=device, dtype=torch.long)
    boundary_edges = torch.tensor(
        [
            edge
            for edge, (pre, post) in enumerate(zip(new_edge_pre, new_edge_post, strict=True))
            if int(new_node_ids[pre]) not in old_node_set
            and int(new_node_ids[post]) in old_node_set
        ],
        device=device,
        dtype=torch.long,
    )
    old_to_new_edges = torch.tensor(
        [
            edge
            for edge, (pre, post) in enumerate(zip(new_edge_pre, new_edge_post, strict=True))
            if int(new_node_ids[pre]) in old_node_set
            and int(new_node_ids[post]) not in old_node_set
        ],
        device=device,
        dtype=torch.long,
    )
    roll_pool_end = int(controller.pool_offsets[2].item())
    roll_motor_nodes = controller.pool_indices[:roll_pool_end]
    roll_motor_set = set(map(int, roll_motor_nodes.detach().cpu().tolist()))
    non_roll_boundary_edges = torch.tensor(
        [
            edge
            for edge in boundary_edges.tolist()
            if int(new_edge_post[edge]) not in roll_motor_set
        ],
        device=device,
        dtype=torch.long,
    )
    acceleration_path_mask = np.ones(len(new_edge_pre), dtype=bool)
    if new_acceleration_pathways_only:
        acceleration_nodes = new_graph.get("acceleration_node_indices", np.empty(0, dtype=np.int64))
        if not len(acceleration_nodes):
            raise SystemExit("--new-acceleration-pathways-only requires accelerometer nodes")
        throttle_begin = int(controller.pool_offsets[6].item())
        throttle_end = int(controller.pool_offsets[8].item())
        throttle_nodes = controller.pool_indices[throttle_begin:throttle_end].cpu().numpy()
        acceleration_path_mask = edges_on_short_paths(
            new_edge_pre,
            new_edge_post,
            len(new_node_ids),
            acceleration_nodes,
            throttle_nodes,
            max_hops=8,
        )
    saved = checkpoint["controller"]
    with torch.no_grad():
        controller.bias[new_nodes] = saved["bias"][old_nodes]
        controller.raw_time_constant[new_nodes] = saved["raw_time_constant"][old_nodes]
        controller.edge_magnitude[new_edges] = saved["edge_magnitude"][old_edges]
        controller.edge_magnitude[boundary_edges] = 0.0
        if new_acceleration_pathways_only:
            controller.edge_magnitude[old_to_new_edges] = 0.0
        if RETINAL_FLIP_X:
            controller.visual_grid[..., 0].mul_(-1.0)
        if new_visual_hemifields:
            added_visual = [
                index
                for index in range(len(new_visual_nodes))
                if index not in set(common_new_visual)
            ]
            added = torch.tensor(added_visual, device=device, dtype=torch.long)
            eye = torch.from_numpy(new_visual_eye[added_visual]).to(
                device=device, dtype=controller.visual_grid.dtype
            )
            overlap = 0.15
            u = controller.visual_grid[0, added, 0, 0]
            controller.visual_grid[0, added, 0, 0] = (
                0.5 * (1.0 + overlap) * u + 0.5 * (1.0 - overlap) * eye
            )
        # Expanded farthest-point samples can slightly change their min/max bounds.
        # Preserve every warm-started L1 cell's exact calibrated receptive field.
        controller.visual_grid[:, common_new_visual] = saved["visual_grid"][:, common_old_visual]
    if freeze_existing:
        node_gradient_mask = torch.ones_like(controller.bias)
        node_gradient_mask[new_nodes] = 0.0
        bias_gradient_mask = node_gradient_mask.clone()
        if not keep_roll_biases_frozen:
            bias_gradient_mask[roll_motor_nodes] = 1.0
        edge_gradient_mask = torch.ones_like(controller.edge_magnitude)
        edge_gradient_mask[new_edges] = 0.0
        if new_pathways_roll_only:
            edge_gradient_mask[non_roll_boundary_edges] = 0.0
        if new_acceleration_pathways_only:
            presynaptic_is_new = torch.tensor(
                [int(new_node_ids[pre]) not in old_node_set for pre in new_edge_pre],
                device=device,
                dtype=edge_gradient_mask.dtype,
            )
            edge_gradient_mask *= (
                torch.from_numpy(acceleration_path_mask).to(
                    device=device, dtype=edge_gradient_mask.dtype
                )
                * presynaptic_is_new
            )
            # A constant sensor must reproduce the baseline.  New biases or old-to-new
            # edges would otherwise provide sensor-independent controller capacity.
            bias_gradient_mask.zero_()
        controller.bias.register_hook(lambda gradient: gradient * bias_gradient_mask)
        controller.raw_time_constant.register_hook(lambda gradient: gradient * node_gradient_mask)
        controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_gradient_mask)
    if new_acceleration_pathways_only:
        trainable_boundary_edges = int(
            acceleration_path_mask[boundary_edges.detach().cpu().numpy()].sum()
        )
        common_edge_mask = np.zeros(len(new_edge_pre), dtype=bool)
        common_edge_mask[np.asarray(common_new_edges, dtype=np.int64)] = True
        presynaptic_is_new = np.asarray(
            [int(new_node_ids[pre]) not in old_node_set for pre in new_edge_pre]
        )
        trainable_acceleration_path_edges = int(
            (acceleration_path_mask & ~common_edge_mask & presynaptic_is_new).sum()
        )
    elif new_pathways_roll_only:
        trainable_boundary_edges = int(boundary_edges.numel() - non_roll_boundary_edges.numel())
        trainable_acceleration_path_edges = 0
    else:
        trainable_boundary_edges = int(boundary_edges.numel())
        trainable_acceleration_path_edges = 0
    return {
        "common_nodes": len(common_new_indices),
        "destination_nodes": len(new_node_ids),
        "common_edges": len(common_new_edges),
        "destination_edges": len(new_edge_pre),
        "zero_initialized_new_to_old_edges": int(boundary_edges.numel()),
        "zero_initialized_old_to_new_edges": (
            int(old_to_new_edges.numel()) if new_acceleration_pathways_only else 0
        ),
        "common_visual_nodes": len(common_new_visual),
        "destination_visual_nodes": len(new_visual_nodes),
        "existing_anatomy_frozen": freeze_existing,
        "unfrozen_existing_roll_motor_biases": (
            int(roll_motor_nodes.numel()) if freeze_existing and not keep_roll_biases_frozen else 0
        ),
        "new_visual_mapping": (
            "bilateral_hemifields_overlap_0.15"
            if new_visual_hemifields
            else "duplicated_full_frame"
        ),
        "trainable_new_to_old_boundary_edges": trainable_boundary_edges,
        "trainable_acceleration_path_edges": trainable_acceleration_path_edges,
    }


def load_controller(
    args: argparse.Namespace, device: torch.device
) -> tuple[ConnectomeController, HoverConfig, GateConfig, dict[str, Any], str]:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    destination_graph_sha = file_sha256(args.graph)
    remap_graph = checkpoint.get("graph_sha256") != destination_graph_sha
    if args.new_visual_hemifields and not remap_graph:
        raise SystemExit("--new-visual-hemifields requires anatomical graph remapping")
    if remap_graph:
        if args.evaluate_only:
            raise SystemExit("--evaluate-only does not permit graph remapping")
        if args.checkpoint_graph is None or not args.checkpoint_graph.is_file():
            raise SystemExit("checkpoint graph differs; provide its original --checkpoint-graph")
        if checkpoint.get("graph_sha256") != file_sha256(args.checkpoint_graph):
            raise SystemExit("--checkpoint-graph hash does not match the checkpoint")
    hover_config = HoverConfig(**checkpoint["hover_config"])
    gate_config = GateConfig(**checkpoint.get("gate_config", {}))
    if args.evaluate_only:
        if checkpoint.get("task_version") != GATE_TASK_VERSION:
            raise SystemExit("--evaluate-only requires an annular-gate-v1 checkpoint")
        if checkpoint.get("plant_model_version") != PLANT_MODEL_VERSION:
            raise SystemExit("checkpoint plant model does not match this evaluator")
        if checkpoint.get("image_resolution") != args.resolution:
            raise SystemExit("--resolution does not match the frozen gate checkpoint")
        if checkpoint.get("retinal_receptive_field") != RETINAL_RECEPTIVE_FIELD:
            raise SystemExit("retinal receptive field does not match the frozen checkpoint")
        if checkpoint.get("retinal_flip_x") != RETINAL_FLIP_X:
            raise SystemExit("retinal orientation does not match the frozen checkpoint")
    controller = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=RETINAL_RECEPTIVE_FIELD,
    ).to(device)
    if remap_graph:
        remap_summary = remap_anatomical_parameters(
            controller,
            checkpoint,
            args.checkpoint_graph,
            args.graph,
            freeze_existing=args.freeze_remapped_anatomy,
            new_visual_hemifields=args.new_visual_hemifields,
            keep_roll_biases_frozen=args.keep_remapped_roll_biases_frozen,
            new_pathways_roll_only=args.new_pathways_roll_only,
            new_acceleration_pathways_only=args.new_acceleration_pathways_only,
        )
        checkpoint = {**checkpoint, "graph_remap": remap_summary}
        print(json.dumps({"graph_remap": remap_summary}), flush=True)
    else:
        incompatible = controller.load_state_dict(checkpoint["controller"], strict=False)
        allowed_missing = {"initial_raw_time_constant"}
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise SystemExit(
                "checkpoint is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        saved_retinal_flip_x = bool(checkpoint.get("retinal_flip_x", False))
        if saved_retinal_flip_x != RETINAL_FLIP_X:
            # MaleCNS assigned hex coordinates do not define our camera-axis convention.
            # Calibrate only the fixed L1 sampling grid; FPV remains unmirrored.
            controller.visual_grid[..., 0].mul_(-1.0)
    return controller, hover_config, gate_config, checkpoint, file_sha256(args.checkpoint)


def constrain_to_roll_motor_inputs(
    controller: ConnectomeController, *, visual_path_hops: int = 0
) -> dict[str, int | str]:
    """Allow plasticity only on selected existing paths into roll motor pools."""

    roll_pool_end = int(controller.pool_offsets[2].item())
    roll_motor_nodes = controller.pool_indices[:roll_pool_end]
    if visual_path_hops:
        edge_pre = controller.edge_pre.detach().cpu().numpy()
        edge_post = controller.edge_post.detach().cpu().numpy()
        visual_nodes = set(map(int, controller.visual_nodes.detach().cpu().tolist()))
        roll_nodes = set(map(int, roll_motor_nodes.detach().cpu().tolist()))
        unreachable = controller.n_nodes + 1
        forward_distance = np.full(controller.n_nodes, unreachable, dtype=np.int32)
        backward_distance = np.full(controller.n_nodes, unreachable, dtype=np.int32)
        forward_distance[list(visual_nodes)] = 0
        backward_distance[list(roll_nodes)] = 0
        for _ in range(visual_path_hops):
            for pre, post in zip(edge_pre, edge_post, strict=True):
                forward_distance[post] = min(forward_distance[post], forward_distance[pre] + 1)
            for pre, post in zip(edge_pre, edge_post, strict=True):
                backward_distance[pre] = min(backward_distance[pre], backward_distance[post] + 1)
        selected_edges = (
            forward_distance[edge_pre] + 1 + backward_distance[edge_post] <= visual_path_hops
        )
        edge_mask = torch.from_numpy(selected_edges).to(
            device=controller.edge_magnitude.device,
            dtype=controller.edge_magnitude.dtype,
        )
        selection = f"visual_to_roll_paths_at_most_{visual_path_hops}_hops"
    else:
        edge_mask = torch.isin(controller.edge_post, roll_motor_nodes).to(
            dtype=controller.edge_magnitude.dtype
        )
        selection = "edges_entering_roll_motor_pools"
    controller.bias.requires_grad_(False)
    controller.raw_time_constant.requires_grad_(False)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_mask)
    return {
        "roll_motor_nodes": int(roll_motor_nodes.numel()),
        "trainable_incoming_edges": int(edge_mask.sum().item()),
        "frozen_edges": int(edge_mask.numel() - edge_mask.sum().item()),
        "trainable_biases": 0,
        "trainable_time_constants": 0,
        "selection": selection,
    }


def main() -> int:
    args = parse_args()
    if (
        args.dagger_rounds
        and min(
            args.dagger_collection_episodes,
            args.dagger_teacher_episodes,
            args.dagger_updates_per_round,
            args.dagger_window_steps,
            args.dagger_selection_interval,
        )
        <= 0
    ):
        raise SystemExit("all DAgger sizes and intervals must be positive")
    if (
        args.recovery_rounds
        and min(
            args.recovery_collection_episodes,
            args.recovery_updates_per_round,
            args.recovery_selection_interval,
            args.recovery_window_steps,
            args.recovery_clean_interval,
        )
        <= 0
    ):
        raise SystemExit("all recovery replay sizes and intervals must be positive")
    if (
        args.trajectory_updates
        and min(args.trajectory_steps, args.trajectory_selection_interval) <= 0
    ):
        raise SystemExit("trajectory steps and selection interval must be positive")
    if args.recovery_early_emphasis_steps < 0:
        raise SystemExit("--recovery-early-emphasis-steps cannot be negative")
    if not (0.0 <= args.recovery_min_takeover_seconds <= args.recovery_max_takeover_seconds):
        raise SystemExit("recovery takeover times must be nonnegative and ordered")
    maximum_recovery_step = (
        round(args.recovery_max_takeover_seconds / HoverConfig().dt) + args.recovery_window_steps
    )
    if args.recovery_rounds and maximum_recovery_step > args.rollout_steps:
        raise SystemExit("recovery takeover plus window must fit within --rollout-steps")
    if args.roll_stick_gain is not None and args.roll_stick_gain <= 0.0:
        raise SystemExit("--roll-stick-gain must be positive")
    if not 0.0 <= args.dagger_early_probability <= 1.0:
        raise SystemExit("--dagger-early-probability must be in [0, 1]")
    if args.dagger_early_maximum_start < 0:
        raise SystemExit("--dagger-early-maximum-start cannot be negative")
    if args.dagger_teacher_updates_per_student < 0:
        raise SystemExit("--dagger-teacher-updates-per-student cannot be negative")
    if any(priority <= 0.0 for priority in args.dagger_axis_priority):
        raise SystemExit("--dagger-axis-priority values must be positive")
    if args.dagger_mirror_roll_weight < 0.0:
        raise SystemExit("--dagger-mirror-roll-weight cannot be negative")
    if (
        args.dagger_recent_probability < 0.0
        or args.dagger_older_probability < 0.0
        or args.dagger_recent_probability + args.dagger_older_probability > 1.0
    ):
        raise SystemExit("DAgger replay probabilities must be nonnegative and sum to at most 1")
    if any(
        seconds < 0.0 or seconds >= args.evaluation_seconds
        for seconds in args.teacher_takeover_audit_seconds
    ):
        raise SystemExit(
            "teacher takeover times must be nonnegative and below --evaluation-seconds"
        )
    if args.dagger_mirrored_pairs and (
        args.dagger_collection_episodes % 2
        or args.dagger_teacher_episodes % 2
        or args.batch_size % 2
    ):
        raise SystemExit("mirrored DAgger episode and replay batch counts must be even")
    if (
        args.recovery_rounds
        and args.dagger_mirrored_pairs
        and (args.recovery_collection_episodes % 2 or args.batch_size % 2)
    ):
        raise SystemExit("mirrored recovery episode and replay batch counts must be even")
    if args.trajectory_updates and args.dagger_mirrored_pairs and args.batch_size % 2:
        raise SystemExit("mirrored trajectory tracking requires an even batch size")
    if args.recovery_roll_path_only and not args.recovery_rounds:
        raise SystemExit("--recovery-roll-path-only requires --recovery-rounds")
    if args.recovery_throttle_path_only and not args.recovery_rounds:
        raise SystemExit("--recovery-throttle-path-only requires --recovery-rounds")
    if args.recovery_roll_path_only and args.recovery_throttle_path_only:
        raise SystemExit("choose only one recovery path-only axis")
    if args.recovery_throttle_path_only and not args.new_acceleration_pathways_only:
        raise SystemExit("--recovery-throttle-path-only requires --new-acceleration-pathways-only")
    if args.recovery_visual_roll_path_hops < 0:
        raise SystemExit("--recovery-visual-roll-path-hops cannot be negative")
    if args.recovery_visual_roll_path_hops and not args.recovery_roll_path_only:
        raise SystemExit("--recovery-visual-roll-path-hops requires --recovery-roll-path-only")
    if (args.recovery_roll_path_only or args.recovery_throttle_path_only) and (
        args.dagger_rounds
        or args.trajectory_updates
        or args.centered_imitation_iterations
        or args.offset_imitation_iterations
        or args.oblique_imitation_iterations
        or args.centered_closed_loop_iterations
        or args.offset_closed_loop_iterations
        or args.closed_loop_iterations
    ):
        raise SystemExit("path-only recovery must be the only training stage")
    if args.new_pathways_roll_only and args.new_acceleration_pathways_only:
        raise SystemExit("choose only one new-pathway restriction")
    if args.new_acceleration_pathways_only and (
        not args.freeze_remapped_anatomy or args.checkpoint_graph is None
    ):
        raise SystemExit(
            "--new-acceleration-pathways-only requires a frozen remap and --checkpoint-graph"
        )
    if not args.graph.is_file() or not args.checkpoint.is_file():
        raise SystemExit("--graph and --checkpoint must both exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use scripts/run_gate_training.sh")
    seed_everything(args.seed)
    controller, hover_config, gate_config, source_checkpoint, source_sha = load_controller(
        args, device
    )
    if args.evaluation_stage is not None:
        gate_config, _ = curriculum_gate_config(args.evaluation_stage)
    if args.roll_stick_gain is not None:
        hover_config = replace(hover_config, roll_stick_gain=args.roll_stick_gain)
    if args.recovery_roll_path_only:
        args.recovery_plasticity = constrain_to_roll_motor_inputs(
            controller,
            visual_path_hops=args.recovery_visual_roll_path_hops,
        )
        print(json.dumps({"recovery_plasticity": args.recovery_plasticity}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "controller.pt"
    history: list[dict[str, Any]] = []
    started = perf_counter()
    if not args.evaluate_only:
        # Gradient masks below encode exact anatomical freezing.  Decoupled weight
        # decay would still move a masked tensor after its gradient is zeroed.
        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=args.learning_rate, weight_decay=0.0
        )
        stages = (
            ("centered", args.centered_imitation_iterations),
            ("offset", args.offset_imitation_iterations),
            ("oblique", args.oblique_imitation_iterations),
        )
        for stage, iterations in stages:
            if not iterations:
                continue
            stage_config, strict = curriculum_gate_config(stage)
            best_path = args.output_dir / f"{stage}-imitation-best.pt"
            baseline = evaluate_gate(
                controller,
                episodes=args.selection_episodes,
                seconds=args.selection_seconds,
                resolution=args.resolution,
                device=device,
                hover_config=hover_config,
                gate_config=stage_config,
                seed=args.seed + 30_000,
            )
            best_selection_score = gate_selection_score(baseline)
            save_checkpoint(
                best_path,
                controller,
                args,
                hover_config,
                gate_config,
                source_checkpoint_sha256=source_sha,
            )
            baseline_entry = {
                "stage": f"{stage}_imitation",
                "iteration": 0,
                "held_out_success_rate": baseline["success_rate"],
                "held_out_pass_rate": baseline["pass_rate"],
                "held_out_lift_off_rate": baseline["lift_off_rate"],
                "held_out_ring_collision_rate": baseline["ring_collision_rate"],
                "held_out_miss_rate": baseline["miss_rate"],
                "held_out_success_by_lateral_side": {
                    "negative": baseline["success_by_stratum"]["negative_lateral_offset"],
                    "positive": baseline["success_by_stratum"]["positive_lateral_offset"],
                },
                "held_out_selection_score": best_selection_score,
                "best_held_out_selection_score": best_selection_score,
            }
            history.append(baseline_entry)
            print(json.dumps(baseline_entry), flush=True)
            for iteration in range(iterations):
                metrics = imitation_rollout(
                    controller,
                    optimizer,
                    batch=args.batch_size,
                    steps=args.rollout_steps,
                    bptt_window=args.bptt_window,
                    resolution=args.resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=stage_config,
                    strict=strict,
                )
                should_select = (iteration + 1) % 5 == 0 or iteration + 1 == iterations
                selection = None
                if should_select:
                    selection = evaluate_gate(
                        controller,
                        episodes=args.selection_episodes,
                        seconds=args.selection_seconds,
                        resolution=args.resolution,
                        device=device,
                        hover_config=hover_config,
                        gate_config=stage_config,
                        seed=args.seed + 30_000,
                    )
                    selection_score = gate_selection_score(selection)
                    if selection_score > best_selection_score:
                        best_selection_score = selection_score
                        save_checkpoint(
                            best_path,
                            controller,
                            args,
                            hover_config,
                            gate_config,
                            source_checkpoint_sha256=source_sha,
                        )
                if iteration % 5 == 0 or should_select:
                    entry = {
                        "stage": f"{stage}_imitation",
                        "iteration": iteration + 1,
                        **metrics,
                    }
                    if selection is not None:
                        entry.update(
                            {
                                "held_out_success_rate": selection["success_rate"],
                                "held_out_pass_rate": selection["pass_rate"],
                                "held_out_lift_off_rate": selection["lift_off_rate"],
                                "held_out_ring_collision_rate": selection["ring_collision_rate"],
                                "held_out_miss_rate": selection["miss_rate"],
                                "held_out_selection_score": selection_score,
                                "best_held_out_selection_score": best_selection_score,
                            }
                        )
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
            best = torch.load(best_path, map_location=device, weights_only=True)
            controller.load_state_dict(best["controller"])
            optimizer = torch.optim.AdamW(
                controller.parameters(), lr=args.learning_rate, weight_decay=0.0
            )
        closed_stages = (
            ("centered", args.centered_closed_loop_iterations),
            ("offset", args.offset_closed_loop_iterations),
            ("oblique", args.closed_loop_iterations),
        )
        for stage, iterations in closed_stages:
            if not iterations:
                continue
            stage_config, strict = curriculum_gate_config(stage)
            best_path = args.output_dir / f"{stage}-closed-loop-best.pt"
            baseline = evaluate_gate(
                controller,
                episodes=args.selection_episodes,
                seconds=args.selection_seconds,
                resolution=args.resolution,
                device=device,
                hover_config=hover_config,
                gate_config=stage_config,
                seed=args.seed + 40_000,
            )
            best_selection_score = gate_selection_score(baseline)
            save_checkpoint(
                best_path,
                controller,
                args,
                hover_config,
                gate_config,
                source_checkpoint_sha256=source_sha,
            )
            baseline_entry = {
                "stage": f"{stage}_closed_loop",
                "iteration": 0,
                "held_out_success_rate": baseline["success_rate"],
                "held_out_pass_rate": baseline["pass_rate"],
                "held_out_lift_off_rate": baseline["lift_off_rate"],
                "held_out_ring_collision_rate": baseline["ring_collision_rate"],
                "held_out_miss_rate": baseline["miss_rate"],
                "held_out_success_by_lateral_side": {
                    "negative": baseline["success_by_stratum"]["negative_lateral_offset"],
                    "positive": baseline["success_by_stratum"]["positive_lateral_offset"],
                },
                "held_out_selection_score": best_selection_score,
                "best_held_out_selection_score": best_selection_score,
            }
            history.append(baseline_entry)
            print(json.dumps(baseline_entry), flush=True)
            for iteration in range(iterations):
                metrics = closed_loop_rollout(
                    controller,
                    optimizer,
                    batch=args.batch_size,
                    steps=args.rollout_steps,
                    bptt_window=args.bptt_window,
                    resolution=args.resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=stage_config,
                    strict=strict,
                )
                training_score = (
                    5.0 * (1.0 - metrics["clean_pass_rate"])
                    + metrics["imitation"]
                    + metrics["contrastive_imitation"]
                    + metrics["task"]
                    + max(0.0, 0.8 - metrics["signed_distance_m"])
                    + metrics["radial_error_m"]
                )
                should_select = (iteration + 1) % 5 == 0 or iteration + 1 == iterations
                selection = None
                if should_select:
                    selection = evaluate_gate(
                        controller,
                        episodes=args.selection_episodes,
                        seconds=args.selection_seconds,
                        resolution=args.resolution,
                        device=device,
                        hover_config=hover_config,
                        gate_config=stage_config,
                        seed=args.seed + 40_000,
                    )
                    selection_score = gate_selection_score(selection)
                    if selection_score > best_selection_score:
                        best_selection_score = selection_score
                        save_checkpoint(
                            best_path,
                            controller,
                            args,
                            hover_config,
                            gate_config,
                            source_checkpoint_sha256=source_sha,
                        )
                if iteration % 5 == 0 or should_select:
                    entry = {
                        "stage": f"{stage}_closed_loop",
                        "iteration": iteration + 1,
                        "training_score": training_score,
                        **metrics,
                    }
                    if selection is not None:
                        entry.update(
                            {
                                "held_out_success_rate": selection["success_rate"],
                                "held_out_pass_rate": selection["pass_rate"],
                                "held_out_lift_off_rate": selection["lift_off_rate"],
                                "held_out_ring_collision_rate": selection["ring_collision_rate"],
                                "held_out_miss_rate": selection["miss_rate"],
                                "held_out_selection_score": selection_score,
                                "best_held_out_selection_score": best_selection_score,
                            }
                        )
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
            best = torch.load(best_path, map_location=device, weights_only=True)
            controller.load_state_dict(best["controller"])
            optimizer = torch.optim.AdamW(
                controller.parameters(), lr=args.learning_rate, weight_decay=0.0
            )

        if args.dagger_rounds:
            dagger_gate_config, _ = curriculum_gate_config(args.dagger_stage)
            dagger_path = args.output_dir / "dagger-best.pt"
            selection_seed = args.seed + 50_000
            baseline = evaluate_gate(
                controller,
                episodes=args.selection_episodes,
                seconds=args.selection_seconds,
                resolution=args.resolution,
                device=device,
                hover_config=hover_config,
                gate_config=dagger_gate_config,
                seed=selection_seed,
            )
            best_selection_score = gate_selection_score(baseline)
            save_checkpoint(
                dagger_path,
                controller,
                args,
                hover_config,
                gate_config,
                source_checkpoint_sha256=source_sha,
            )
            baseline_entry = {
                "stage": f"{args.dagger_stage}_dagger",
                "round": 0,
                "held_out_success_rate": baseline["success_rate"],
                "held_out_pass_rate": baseline["pass_rate"],
                "held_out_lift_off_rate": baseline["lift_off_rate"],
                "held_out_ring_collision_rate": baseline["ring_collision_rate"],
                "held_out_miss_rate": baseline["miss_rate"],
                "held_out_success_by_lateral_side": {
                    "negative": baseline["success_by_stratum"]["negative_lateral_offset"],
                    "positive": baseline["success_by_stratum"]["positive_lateral_offset"],
                },
                "held_out_selection_score": best_selection_score,
                "best_held_out_selection_score": best_selection_score,
            }
            history.append(baseline_entry)
            print(json.dumps(baseline_entry), flush=True)
            teacher_replay = collect_dagger_trajectories(
                controller,
                episodes=args.dagger_teacher_episodes,
                steps=args.rollout_steps,
                resolution=args.resolution,
                device=device,
                hover_config=hover_config,
                gate_config=dagger_gate_config,
                seed=args.seed + 60_000,
                teacher_driven=True,
                mirrored_pairs=args.dagger_mirrored_pairs,
            )
            student_replay: list[DaggerTrajectories] = []
            dagger_axis_weight = None
            for dagger_round in range(args.dagger_rounds):
                recent = collect_dagger_trajectories(
                    controller,
                    episodes=args.dagger_collection_episodes,
                    steps=args.rollout_steps,
                    resolution=args.resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=dagger_gate_config,
                    seed=args.seed + 61_000 + dagger_round,
                    teacher_driven=False,
                    mirrored_pairs=args.dagger_mirrored_pairs,
                )
                student_replay.append(recent)
                if dagger_axis_weight is None:
                    dagger_axis_scale, dagger_axis_weight = calibrate_dagger_axis_weight(
                        (teacher_replay, recent), tuple(args.dagger_axis_priority)
                    )
                    args.dagger_axis_scale = dagger_axis_scale.detach().cpu().tolist()
                    args.dagger_axis_weight = dagger_axis_weight.detach().cpu().tolist()
                    calibration_entry = {
                        "stage": f"{args.dagger_stage}_dagger",
                        "round": dagger_round + 1,
                        "axis_target_scale": args.dagger_axis_scale,
                        "axis_loss_weight": args.dagger_axis_weight,
                    }
                    history.append(calibration_entry)
                    print(json.dumps(calibration_entry), flush=True)
                seed_everything(args.seed + 62_000 + dagger_round)
                source_counts = {"recent": 0, "older": 0, "teacher": 0}
                probe_metrics = []
                for update_index in range(args.dagger_updates_per_round):
                    choice = random.random()
                    if args.dagger_teacher_only:
                        source = teacher_replay
                        source_counts["teacher"] += 1
                    elif args.dagger_teacher_updates_per_student:
                        block_size = args.dagger_teacher_updates_per_student + 1
                        if update_index % block_size == 0:
                            source = recent
                            source_counts["recent"] += 1
                        else:
                            source = teacher_replay
                            source_counts["teacher"] += 1
                    elif choice < args.dagger_recent_probability:
                        source = recent
                        source_counts["recent"] += 1
                    elif (
                        choice < args.dagger_recent_probability + args.dagger_older_probability
                        and len(student_replay) > 1
                    ):
                        source = random.choice(student_replay[:-1])
                        source_counts["older"] += 1
                    else:
                        source = teacher_replay
                        source_counts["teacher"] += 1
                    replay_metric = dagger_replay_update(
                        controller,
                        optimizer,
                        source,
                        batch=args.batch_size,
                        window_steps=args.dagger_window_steps,
                        axis_weight=dagger_axis_weight,
                        early_probability=args.dagger_early_probability,
                        early_maximum_start=args.dagger_early_maximum_start,
                        mirror_roll_weight=args.dagger_mirror_roll_weight,
                    )
                    probe_metrics.append(replay_metric)
                    should_select = (
                        (update_index + 1) % args.dagger_selection_interval == 0
                        or update_index + 1 == args.dagger_updates_per_round
                    )
                    if not should_select:
                        continue
                    selection = evaluate_gate(
                        controller,
                        episodes=args.selection_episodes,
                        seconds=args.selection_seconds,
                        resolution=args.resolution,
                        device=device,
                        hover_config=hover_config,
                        gate_config=dagger_gate_config,
                        seed=selection_seed,
                    )
                    selection_score = gate_selection_score(selection)
                    if selection_score > best_selection_score:
                        best_selection_score = selection_score
                        save_checkpoint(
                            dagger_path,
                            controller,
                            args,
                            hover_config,
                            gate_config,
                            source_checkpoint_sha256=source_sha,
                        )
                    entry = {
                        "stage": f"{args.dagger_stage}_dagger",
                        "round": dagger_round + 1,
                        "update": update_index + 1,
                        "dagger_imitation": float(
                            np.mean([item["dagger_imitation"] for item in probe_metrics])
                        ),
                        "dagger_valid_fraction": float(
                            np.mean([item["dagger_valid_fraction"] for item in probe_metrics])
                        ),
                        "dagger_axis_mse": {
                            axis: float(
                                np.mean([item[f"dagger_{axis}_mse"] for item in probe_metrics])
                            )
                            for axis in ("roll", "pitch", "yaw", "throttle")
                        },
                        "dagger_mirror_roll_mse": float(
                            np.mean([item["dagger_mirror_roll_mse"] for item in probe_metrics])
                        ),
                        "dagger_mirror_roll_mean_mse": float(
                            np.mean([item["dagger_mirror_roll_mean_mse"] for item in probe_metrics])
                        ),
                        "dagger_paired_selection_rate": float(
                            np.mean([item["dagger_paired_selection"] for item in probe_metrics])
                        ),
                        "dagger_mirror_roll_applied_rate": float(
                            np.mean([item["dagger_mirror_roll_applied"] for item in probe_metrics])
                        ),
                        "replay_source_counts": source_counts.copy(),
                        "held_out_success_rate": selection["success_rate"],
                        "held_out_pass_rate": selection["pass_rate"],
                        "held_out_lift_off_rate": selection["lift_off_rate"],
                        "held_out_ring_collision_rate": selection["ring_collision_rate"],
                        "held_out_miss_rate": selection["miss_rate"],
                        "held_out_success_by_lateral_side": {
                            "negative": selection["success_by_stratum"]["negative_lateral_offset"],
                            "positive": selection["success_by_stratum"]["positive_lateral_offset"],
                        },
                        "held_out_selection_score": selection_score,
                        "best_held_out_selection_score": best_selection_score,
                    }
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
                    probe_metrics = []
                    # Evaluation deliberately reseeds global generators for a fixed
                    # development suite; resume replay from a unique training seed.
                    seed_everything(args.seed + 63_000 + 1_000 * dagger_round + update_index)
                best = torch.load(dagger_path, map_location=device, weights_only=True)
                controller.load_state_dict(best["controller"])
                optimizer = torch.optim.AdamW(
                    controller.parameters(), lr=args.learning_rate, weight_decay=0.0
                )

        if args.recovery_rounds:
            recovery_gate_config, _ = curriculum_gate_config(args.recovery_stage)
            recovery_path = args.output_dir / "recovery-best.pt"
            selection_seed = args.seed + 70_000
            baseline = evaluate_gate(
                controller,
                episodes=args.selection_episodes,
                seconds=args.selection_seconds,
                resolution=args.resolution,
                device=device,
                hover_config=hover_config,
                gate_config=recovery_gate_config,
                seed=selection_seed,
            )
            best_selection_score = gate_selection_score(
                baseline, balance_mass=args.recovery_throttle_path_only
            )
            save_checkpoint(
                recovery_path,
                controller,
                args,
                hover_config,
                recovery_gate_config,
                source_checkpoint_sha256=source_sha,
            )
            baseline_entry = {
                "stage": f"{args.recovery_stage}_recovery",
                "round": 0,
                "held_out_success_rate": baseline["success_rate"],
                "held_out_pass_rate": baseline["pass_rate"],
                "held_out_ring_collision_rate": baseline["ring_collision_rate"],
                "held_out_miss_rate": baseline["miss_rate"],
                "held_out_success_by_lateral_side": {
                    "negative": baseline["success_by_stratum"]["negative_lateral_offset"],
                    "positive": baseline["success_by_stratum"]["positive_lateral_offset"],
                },
                "held_out_success_by_mass": {
                    "lower": baseline["success_by_stratum"]["lower_mass"],
                    "higher": baseline["success_by_stratum"]["higher_mass"],
                },
                "held_out_selection_score": best_selection_score,
                "best_held_out_selection_score": best_selection_score,
            }
            history.append(baseline_entry)
            print(json.dumps(baseline_entry), flush=True)
            clean_replay = collect_dagger_trajectories(
                controller,
                episodes=args.recovery_collection_episodes,
                steps=args.rollout_steps,
                resolution=args.resolution,
                device=device,
                hover_config=hover_config,
                gate_config=recovery_gate_config,
                seed=args.seed + 71_000,
                teacher_driven=True,
                mirrored_pairs=args.dagger_mirrored_pairs,
            )
            recovery_axis_weight = None
            for recovery_round in range(args.recovery_rounds):
                recovery_replay = collect_takeover_recovery_trajectories(
                    controller,
                    episodes=args.recovery_collection_episodes,
                    steps=args.rollout_steps,
                    resolution=args.resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=recovery_gate_config,
                    seed=args.seed + 72_000 + recovery_round,
                    minimum_takeover_seconds=args.recovery_min_takeover_seconds,
                    maximum_takeover_seconds=args.recovery_max_takeover_seconds,
                    recovery_window_steps=args.recovery_window_steps,
                    mirrored_pairs=args.dagger_mirrored_pairs,
                )
                if recovery_axis_weight is None:
                    recovery_axis_scale, recovery_axis_weight = calibrate_dagger_axis_weight(
                        (recovery_replay,), tuple(args.dagger_axis_priority)
                    )
                    args.recovery_axis_scale = recovery_axis_scale.detach().cpu().tolist()
                    args.recovery_axis_weight = recovery_axis_weight.detach().cpu().tolist()
                    calibration_entry = {
                        "stage": f"{args.recovery_stage}_recovery",
                        "round": recovery_round + 1,
                        "axis_target_scale": args.recovery_axis_scale,
                        "axis_loss_weight": args.recovery_axis_weight,
                        "teacher_recovery_clean_crossing_rate": (
                            recovery_replay.teacher_recovery_clean_crossing_rate
                        ),
                    }
                    history.append(calibration_entry)
                    print(json.dumps(calibration_entry), flush=True)
                seed_everything(args.seed + 73_000 + recovery_round)
                source_counts = {"recovery": 0, "clean_teacher": 0}
                recovery_probe: list[dict[str, float]] = []
                clean_probe: list[dict[str, float]] = []
                for update_index in range(args.recovery_updates_per_round):
                    clean_update = (
                        not (args.recovery_roll_path_only or args.recovery_throttle_path_only)
                        and (update_index + 1) % args.recovery_clean_interval == 0
                    )
                    if clean_update:
                        clean_probe.append(
                            dagger_replay_update(
                                controller,
                                optimizer,
                                clean_replay,
                                batch=args.batch_size,
                                window_steps=args.recovery_window_steps,
                                axis_weight=recovery_axis_weight,
                                early_probability=1.0,
                                early_maximum_start=round(
                                    args.recovery_max_takeover_seconds / hover_config.dt
                                ),
                                mirror_roll_weight=0.0,
                            )
                        )
                        source_counts["clean_teacher"] += 1
                    else:
                        recovery_probe.append(
                            recovery_replay_update(
                                controller,
                                optimizer,
                                recovery_replay,
                                batch=args.batch_size,
                                window_steps=args.recovery_window_steps,
                                early_emphasis_steps=(args.recovery_early_emphasis_steps),
                                axis_weight=recovery_axis_weight,
                                path_axis=(
                                    0
                                    if args.recovery_roll_path_only
                                    else 3
                                    if args.recovery_throttle_path_only
                                    else None
                                ),
                            )
                        )
                        source_counts["recovery"] += 1
                    should_select = (
                        (update_index + 1) % args.recovery_selection_interval == 0
                        or update_index + 1 == args.recovery_updates_per_round
                    )
                    if not should_select:
                        continue
                    selection = evaluate_gate(
                        controller,
                        episodes=args.selection_episodes,
                        seconds=args.selection_seconds,
                        resolution=args.resolution,
                        device=device,
                        hover_config=hover_config,
                        gate_config=recovery_gate_config,
                        seed=selection_seed,
                    )
                    selection_score = gate_selection_score(
                        selection, balance_mass=args.recovery_throttle_path_only
                    )
                    if selection_score > best_selection_score:
                        best_selection_score = selection_score
                        save_checkpoint(
                            recovery_path,
                            controller,
                            args,
                            hover_config,
                            recovery_gate_config,
                            source_checkpoint_sha256=source_sha,
                        )
                    entry = {
                        "stage": f"{args.recovery_stage}_recovery",
                        "round": recovery_round + 1,
                        "update": update_index + 1,
                        "teacher_recovery_clean_crossing_rate": (
                            recovery_replay.teacher_recovery_clean_crossing_rate
                        ),
                        "replay_source_counts": source_counts.copy(),
                        "recovery_imitation": (
                            float(np.mean([item["recovery_imitation"] for item in recovery_probe]))
                            if recovery_probe
                            else None
                        ),
                        "recovery_early_fitted_axis_mse": (
                            float(
                                np.mean(
                                    [
                                        item["recovery_early_fitted_axis_mse"]
                                        for item in recovery_probe
                                    ]
                                )
                            )
                            if recovery_probe
                            else None
                        ),
                        "recovery_pair_roll_mse": (
                            float(
                                np.mean([item["recovery_pair_roll_mse"] for item in recovery_probe])
                            )
                            if recovery_probe
                            else None
                        ),
                        "recovery_non_roll_anchor_mse": (
                            float(
                                np.mean(
                                    [
                                        item["recovery_non_roll_anchor_mse"]
                                        for item in recovery_probe
                                    ]
                                )
                            )
                            if recovery_probe
                            else None
                        ),
                        "recovery_prefix_anchor_mse": (
                            float(
                                np.mean(
                                    [item["recovery_prefix_anchor_mse"] for item in recovery_probe]
                                )
                            )
                            if recovery_probe
                            else None
                        ),
                        "clean_teacher_imitation": (
                            float(np.mean([item["dagger_imitation"] for item in clean_probe]))
                            if clean_probe
                            else None
                        ),
                        "held_out_success_rate": selection["success_rate"],
                        "held_out_pass_rate": selection["pass_rate"],
                        "held_out_ring_collision_rate": selection["ring_collision_rate"],
                        "held_out_miss_rate": selection["miss_rate"],
                        "held_out_success_by_lateral_side": {
                            "negative": selection["success_by_stratum"]["negative_lateral_offset"],
                            "positive": selection["success_by_stratum"]["positive_lateral_offset"],
                        },
                        "held_out_success_by_mass": {
                            "lower": selection["success_by_stratum"]["lower_mass"],
                            "higher": selection["success_by_stratum"]["higher_mass"],
                        },
                        "held_out_selection_score": selection_score,
                        "best_held_out_selection_score": best_selection_score,
                    }
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
                    recovery_probe = []
                    clean_probe = []
                    source_counts = {"recovery": 0, "clean_teacher": 0}
                    seed_everything(args.seed + 74_000 + 1_000 * recovery_round + update_index)
                best = torch.load(recovery_path, map_location=device, weights_only=True)
                controller.load_state_dict(best["controller"])
                optimizer = torch.optim.AdamW(
                    controller.parameters(), lr=args.learning_rate, weight_decay=0.0
                )

        if args.trajectory_updates:
            trajectory_gate_config, _ = curriculum_gate_config(args.trajectory_stage)
            trajectory_path = args.output_dir / "trajectory-best.pt"
            selection_seed = args.seed + 80_000
            baseline = evaluate_gate(
                controller,
                episodes=args.selection_episodes,
                seconds=args.selection_seconds,
                resolution=args.resolution,
                device=device,
                hover_config=hover_config,
                gate_config=trajectory_gate_config,
                seed=selection_seed,
            )
            best_selection_score = gate_selection_score(baseline)
            save_checkpoint(
                trajectory_path,
                controller,
                args,
                hover_config,
                trajectory_gate_config,
                source_checkpoint_sha256=source_sha,
            )
            baseline_entry = {
                "stage": f"{args.trajectory_stage}_trajectory",
                "update": 0,
                "held_out_success_rate": baseline["success_rate"],
                "held_out_pass_rate": baseline["pass_rate"],
                "held_out_ring_collision_rate": baseline["ring_collision_rate"],
                "held_out_miss_rate": baseline["miss_rate"],
                "held_out_success_by_lateral_side": {
                    "negative": baseline["success_by_stratum"]["negative_lateral_offset"],
                    "positive": baseline["success_by_stratum"]["positive_lateral_offset"],
                },
                "held_out_selection_score": best_selection_score,
                "best_held_out_selection_score": best_selection_score,
            }
            history.append(baseline_entry)
            print(json.dumps(baseline_entry), flush=True)
            for update_index in range(args.trajectory_updates):
                training_seed = args.seed + 81_000 + update_index
                seed_everything(training_seed)
                metrics = trajectory_tracking_rollout(
                    controller,
                    optimizer,
                    batch=args.batch_size,
                    steps=args.trajectory_steps,
                    resolution=args.resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=trajectory_gate_config,
                    mirrored_pairs=args.dagger_mirrored_pairs,
                )
                seed_everything(training_seed)
                with torch.no_grad():
                    post_update_metrics = trajectory_tracking_rollout(
                        controller,
                        None,
                        batch=args.batch_size,
                        steps=args.trajectory_steps,
                        resolution=args.resolution,
                        device=device,
                        hover_config=hover_config,
                        gate_config=trajectory_gate_config,
                        mirrored_pairs=args.dagger_mirrored_pairs,
                    )
                should_select = (
                    (update_index + 1) % args.trajectory_selection_interval == 0
                    or update_index + 1 == args.trajectory_updates
                )
                if not should_select:
                    continue
                selection = evaluate_gate(
                    controller,
                    episodes=args.selection_episodes,
                    seconds=args.selection_seconds,
                    resolution=args.resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=trajectory_gate_config,
                    seed=selection_seed,
                )
                selection_score = gate_selection_score(selection)
                if selection_score > best_selection_score:
                    best_selection_score = selection_score
                    save_checkpoint(
                        trajectory_path,
                        controller,
                        args,
                        hover_config,
                        trajectory_gate_config,
                        source_checkpoint_sha256=source_sha,
                    )
                entry = {
                    "stage": f"{args.trajectory_stage}_trajectory",
                    "update": update_index + 1,
                    **metrics,
                    "post_update_trajectory_loss": post_update_metrics["trajectory_loss"],
                    "post_update_trajectory_position_loss": post_update_metrics[
                        "trajectory_position_loss"
                    ],
                    "post_update_trajectory_velocity_loss": post_update_metrics[
                        "trajectory_velocity_loss"
                    ],
                    "post_update_trajectory_clean_pass_rate": post_update_metrics[
                        "trajectory_clean_pass_rate"
                    ],
                    "post_update_trajectory_radial_error_m": post_update_metrics[
                        "trajectory_radial_error_m"
                    ],
                    "same_batch_trajectory_loss_delta": (
                        post_update_metrics["trajectory_loss"] - metrics["trajectory_loss"]
                    ),
                    "held_out_success_rate": selection["success_rate"],
                    "held_out_pass_rate": selection["pass_rate"],
                    "held_out_ring_collision_rate": selection["ring_collision_rate"],
                    "held_out_miss_rate": selection["miss_rate"],
                    "held_out_success_by_lateral_side": {
                        "negative": selection["success_by_stratum"]["negative_lateral_offset"],
                        "positive": selection["success_by_stratum"]["positive_lateral_offset"],
                    },
                    "held_out_selection_score": selection_score,
                    "best_held_out_selection_score": best_selection_score,
                }
                history.append(entry)
                print(json.dumps(entry), flush=True)
            best = torch.load(trajectory_path, map_location=device, weights_only=True)
            controller.load_state_dict(best["controller"])

    controller.eval()
    evaluation_seed = args.seed + 10_000
    teacher = evaluate_gate(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        resolution=args.resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=evaluation_seed,
        teacher=True,
    )
    nominal = evaluate_gate(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        resolution=args.resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=evaluation_seed,
    )
    frozen = evaluate_gate(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        resolution=args.resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=evaluation_seed,
        frozen_visual=True,
    )
    constant_acceleration = (
        evaluate_gate(
            controller,
            episodes=args.evaluation_episodes,
            seconds=args.evaluation_seconds,
            resolution=args.resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=evaluation_seed,
            frozen_acceleration=True,
        )
        if controller.uses_accelerometer
        else None
    )
    swapped_acceleration = (
        evaluate_gate(
            controller,
            episodes=args.evaluation_episodes,
            seconds=args.evaluation_seconds,
            resolution=args.resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=evaluation_seed,
            swapped_acceleration=True,
        )
        if controller.uses_accelerometer
        else None
    )
    teacher_takeover_audit = {
        str(seconds): evaluate_gate(
            controller,
            episodes=args.evaluation_episodes,
            seconds=args.evaluation_seconds,
            resolution=args.resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=evaluation_seed,
            teacher_takeover_at_seconds=seconds,
        )
        for seconds in args.teacher_takeover_audit_seconds
    }
    save_checkpoint(
        checkpoint_path,
        controller,
        args,
        hover_config,
        gate_config,
        source_checkpoint_sha256=source_sha,
    )
    report = {
        "passed": nominal["goal_pass"],
        "task_version": GATE_TASK_VERSION,
        "plant_model_version": PLANT_MODEL_VERSION,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "graph": str(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_task_version": source_checkpoint.get("task_version", "hover-v1"),
        "source_checkpoint_graph": (
            str(args.checkpoint_graph) if args.checkpoint_graph is not None else None
        ),
        "graph_remap": source_checkpoint.get("graph_remap"),
        "connectome_nodes": controller.n_nodes,
        "connectome_edges": int(controller.edge_pre.numel()),
        "actor_inputs": [
            f"{args.resolution}x{args.resolution}_monochrome_fpv",
            "roll",
            "pitch",
            *(["body_specific_force_z_centered_on_1g"] if controller.uses_accelerometer else []),
            "recurrent_neural_state",
        ],
        "actor_outputs": ["right_foreleg_roll_pitch", "left_foreleg_yaw_throttle"],
        "retinal_receptive_field": RETINAL_RECEPTIVE_FIELD,
        "retinal_flip_x": RETINAL_FLIP_X,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "evaluation_protocol": {
            "seed": evaluation_seed,
            "episodes": args.evaluation_episodes,
            "seconds": args.evaluation_seconds,
            "image_resolution": [args.resolution, args.resolution],
            "all_gates_initially_off_axis": True,
            "all_gate_normals_oblique_to_launch_displacement": True,
            "minimum_post_pass_flight_seconds": 1.0,
        },
        "training": {
            "performed_this_run": not args.evaluate_only,
            "arguments": json_arguments(args),
            "elapsed_seconds": perf_counter() - started,
            "history": history,
        },
        "teacher_baseline": teacher,
        "evaluation": nominal,
        "frozen_visual_ablation": frozen,
        "constant_1g_accelerometer_ablation": constant_acceleration,
        "mass_rank_swapped_accelerometer_ablation": swapped_acceleration,
        "teacher_takeover_audit": teacher_takeover_audit,
        "commands_from_measured_sticks_only": True,
        "external_actor_state_machine": False,
        "privileged_gate_geometry_given_to_actor": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"wrote {checkpoint_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
