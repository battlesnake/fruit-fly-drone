#!/usr/bin/env python3
"""Train a full MaleCNS graph for textured RGB airborne hover at 50 Hz."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.hover import (  # noqa: E402
    PLANT_MODEL_VERSION,
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
    motor_target_for_rc,
)
from flydrone.visual_hover import (  # noqa: E402
    DEFAULT_VISUAL_CAMERA,
    render_visual_hover_scene,
    sample_marker_heights,
)


@dataclass
class TemporalBatch:
    physical: QuadState
    sticks: StickState
    neural: torch.Tensor
    target_height: torch.Tensor
    mass_scale: torch.Tensor
    age: torch.Tensor
    handoff_step: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/visual-hover/pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--unroll", type=int, default=25)
    parser.add_argument("--episode-policy-steps", type=int, default=150)
    parser.add_argument("--static-iterations", type=int, default=120)
    parser.add_argument("--teacher-sequence-iterations", type=int, default=160)
    parser.add_argument("--handoff-iterations", type=int, default=320)
    parser.add_argument("--static-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--sequence-learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--evaluation-episodes", type=int, default=24)
    parser.add_argument("--evaluation-seconds", type=float, default=10.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--omit-attitude",
        action="store_true",
        help="train/evaluate with the estimated roll/pitch sensory channels zeroed",
    )
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


def observation_attitude(state: QuadState, omit_attitude: bool) -> torch.Tensor:
    attitude = state.euler[:, :2]
    return torch.zeros_like(attitude) if omit_attitude else attitude


def oracle_rc(
    state: QuadState,
    target_height: torch.Tensor,
    mass_scale: torch.Tensor,
    config: HoverConfig,
) -> torch.Tensor:
    """Privileged training/evaluation oracle with the exact steady hover base."""

    roll_rate = -3.5 * state.euler[:, 0] - 0.45 * state.rates[:, 0]
    pitch_rate = -3.5 * state.euler[:, 1] - 0.45 * state.rates[:, 1]
    roll = (roll_rate / config.max_roll_pitch_rate).clamp(-1.0, 1.0)
    pitch = (pitch_rate / config.max_roll_pitch_rate).clamp(-1.0, 1.0)
    yaw = (-0.4 * state.rates[:, 2] / config.max_yaw_rate).clamp(-1.0, 1.0)
    hover_thrust = mass_scale / config.thrust_to_weight
    throttle = (
        hover_thrust + 0.42 * (target_height - state.position[:, 2]) - 0.18 * state.velocity[:, 2]
    ).clamp(0.0, 1.0)
    return torch.stack((roll, pitch, yaw, throttle), dim=1)


def controller_regularization(controller: ConnectomeController) -> torch.Tensor:
    return (
        1.0e-4 * (controller.edge_magnitude - controller.initial_edge_magnitude).square().mean()
        + 2.0e-5 * controller.bias.square().mean()
        + 1.0e-5
        * (controller.raw_time_constant - controller.initial_raw_time_constant).square().mean()
    )


def motor_imitation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    axis_weight = prediction.new_tensor((4.0, 4.0, 2.0, 4.0))
    return ((prediction - target).square() * axis_weight).mean()


def random_physical_state(
    batch: int,
    device: torch.device,
    config: HoverConfig,
    *,
    airborne: bool,
    held_out_marker: bool = False,
) -> tuple[QuadState, torch.Tensor, torch.Tensor]:
    target = sample_marker_heights(batch, device=device, held_out=held_out_marker)
    position = torch.zeros(batch, 3, device=device)
    if airborne:
        position[:, 2] = (target + torch.empty(batch, device=device).uniform_(-0.28, 0.28)).clamp(
            0.35, 1.5
        )
    euler = torch.zeros(batch, 3, device=device)
    euler[:, :2] = torch.empty(batch, 2, device=device).uniform_(
        -math.radians(7.0), math.radians(7.0)
    )
    quad = DifferentiableQuad(config).to(device)
    state = quad.initial_state(
        batch, device=device, dtype=torch.float32, position=position, euler=euler
    )
    if airborne:
        state.velocity[:, 2] = torch.empty(batch, device=device).uniform_(-0.35, 0.35)
    mass_scale = torch.empty(batch, device=device).uniform_(0.9, 1.1)
    return state, target, mass_scale


def stick_state_for_rc(rc: torch.Tensor, config: HoverConfig) -> StickState:
    """Construct a physically consistent zero-velocity foreleg/stick state."""

    normalized_position = torch.cat((rc[:, :3], 2.0 * rc[:, 3:4] - 1.0), dim=1)
    sine_limit = math.sin(config.foreleg_joint_limit)
    joint_position = torch.asin((normalized_position * sine_limit).clamp(-1.0, 1.0))
    zeros = torch.zeros_like(normalized_position)
    return StickState(joint_position, zeros.clone(), normalized_position, zeros)


def static_imitation_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    batch: int,
    unroll: int,
    device: torch.device,
    config: HoverConfig,
    omit_attitude: bool,
) -> dict[str, float]:
    state, target, _ = random_physical_state(batch, device, config, airborne=True)
    # A single frozen image cannot reveal an independently sampled vertical velocity.
    # Static training teaches proportional visual error only; temporal lessons below
    # carry the motion-dependent label.
    state.velocity.zero_()
    nominal_mass = torch.ones(batch, device=device)
    image = render_visual_hover_scene(state, target, config=config)
    desired_motor = motor_target_for_rc(oracle_rc(state, target, nominal_mass, config), config)
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    losses = []
    for step in range(unroll):
        motor, neural = controller(image, observation_attitude(state, omit_attitude), neural)
        if step >= unroll // 2:
            losses.append(motor_imitation_loss(motor, desired_motor))
    direct = torch.stack(losses).mean()
    centered_prediction = motor - motor.mean(dim=0, keepdim=True)
    centered_target = desired_motor - desired_motor.mean(dim=0, keepdim=True)
    contrast = motor_imitation_loss(centered_prediction, centered_target)
    loss = 8.0 * direct + 30.0 * contrast + controller_regularization(controller)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
    optimizer.step()
    controller.project_parameters()
    return {
        "loss": float(loss.detach()),
        "direct": float(direct.detach()),
        "contrast": float(contrast.detach()),
        "gradient_norm": float(gradient_norm.detach()),
    }


def new_temporal_batch(
    controller: ConnectomeController,
    *,
    batch: int,
    episode_policy_steps: int,
    device: torch.device,
    config: HoverConfig,
    teacher_only: bool,
) -> TemporalBatch:
    physical, target, mass_scale = random_physical_state(batch, device, config, airborne=True)
    initial_rc = torch.zeros(batch, 4, device=device)
    initial_rc[:, 3] = mass_scale / config.thrust_to_weight
    sticks = stick_state_for_rc(initial_rc, config)
    physical.actuator[:, 0] = initial_rc[:, 3]
    if teacher_only:
        handoff = torch.full((batch,), episode_policy_steps + 1, device=device, dtype=torch.long)
    else:
        # Some flies take over immediately; all take over within the first second.
        handoff = torch.randint(0, min(episode_policy_steps, 50) + 1, (batch,), device=device)
    return TemporalBatch(
        physical=physical,
        sticks=sticks,
        neural=controller.initial_state(batch, device=device, dtype=torch.float32),
        target_height=target,
        mass_scale=mass_scale,
        age=torch.zeros(batch, device=device, dtype=torch.long),
        handoff_step=handoff,
    )


def temporal_imitation_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    rollout: TemporalBatch,
    *,
    unroll: int,
    physics_steps_per_policy: int,
    config: HoverConfig,
    omit_attitude: bool,
) -> tuple[dict[str, float], TemporalBatch]:
    quad = DifferentiableQuad(config).to(rollout.neural.device)
    sticks = ForelegStickPlant(config).to(rollout.neural.device)
    imitation_losses = []
    trim_losses = []
    applied_native = []
    final_prediction = None
    final_target = None

    for _ in range(unroll):
        image = render_visual_hover_scene(rollout.physical, rollout.target_height, config=config)
        prediction, rollout.neural = controller(
            image,
            observation_attitude(rollout.physical, omit_attitude),
            rollout.neural,
        )
        desired_rc = oracle_rc(rollout.physical, rollout.target_height, rollout.mass_scale, config)
        desired_motor = motor_target_for_rc(desired_rc, config)
        imitation_losses.append(motor_imitation_loss(prediction, desired_motor))

        true_hover_rc = torch.zeros_like(desired_rc)
        true_hover_rc[:, 3] = rollout.mass_scale / config.thrust_to_weight
        true_hover_motor = motor_target_for_rc(true_hover_rc, config)
        stable = (
            ((rollout.physical.position[:, 2] - rollout.target_height).abs() < 0.10)
            & (rollout.physical.velocity[:, 2].abs() < 0.20)
            & (torch.linalg.vector_norm(rollout.physical.euler[:, :2], dim=1) < math.radians(4.0))
            & (rollout.age >= 25)
        )
        if stable.any():
            trim_losses.append(
                (prediction[stable, 3] - true_hover_motor[stable, 3]).square().mean()
            )

        native = rollout.age >= rollout.handoff_step
        applied_native.append(native.float().mean())
        applied_motor = torch.where(native[:, None], prediction, desired_motor).detach()
        for _ in range(physics_steps_per_policy):
            measured_rc, rollout.sticks = sticks(applied_motor, rollout.sticks)
            rollout.physical = quad(measured_rc, rollout.physical, rollout.mass_scale)
        rollout.physical = rollout.physical.detach()
        rollout.sticks = rollout.sticks.detach()
        rollout.age = rollout.age + 1
        final_prediction, final_target = prediction, desired_motor

    assert final_prediction is not None and final_target is not None
    direct = torch.stack(imitation_losses).mean()
    centered_prediction = final_prediction - final_prediction.mean(dim=0, keepdim=True)
    centered_target = final_target - final_target.mean(dim=0, keepdim=True)
    contrast = motor_imitation_loss(centered_prediction, centered_target)
    trim = torch.stack(trim_losses).mean() if trim_losses else direct.new_zeros(())
    loss = 8.0 * direct + 12.0 * contrast + 2.0 * trim + controller_regularization(controller)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.7)
    optimizer.step()
    controller.project_parameters()
    rollout.neural = rollout.neural.detach()
    return (
        {
            "loss": float(loss.detach()),
            "direct": float(direct.detach()),
            "contrast": float(contrast.detach()),
            "hover_trim": float(trim.detach()),
            "stable_trim_samples": float(len(trim_losses)),
            "native_fraction": float(torch.stack(applied_native).mean()),
            "height_error_m": float(
                (rollout.physical.position[:, 2] - rollout.target_height).abs().mean()
            ),
            "gradient_norm": float(gradient_norm.detach()),
        },
        rollout,
    )


def reset_finished_rollout(
    controller: ConnectomeController,
    rollout: TemporalBatch,
    *,
    batch: int,
    episode_policy_steps: int,
    device: torch.device,
    config: HoverConfig,
    teacher_only: bool,
) -> TemporalBatch:
    if bool((rollout.age >= episode_policy_steps).all()):
        return new_temporal_batch(
            controller,
            batch=batch,
            episode_policy_steps=episode_policy_steps,
            device=device,
            config=config,
            teacher_only=teacher_only,
        )
    return rollout


@torch.no_grad()
def evaluate(
    controller: ConnectomeController,
    *,
    episodes: int,
    seconds: float,
    policy_hz: int,
    physics_steps_per_policy: int,
    device: torch.device,
    config: HoverConfig,
    seed: int,
    omit_attitude: bool,
    frozen_visual: bool = False,
    oracle: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    state, target, mass_scale = random_physical_state(
        episodes, device, config, airborne=True, held_out_marker=True
    )
    initial_rc = torch.zeros(episodes, 4, device=device)
    initial_rc[:, 3] = mass_scale / config.thrust_to_weight
    stick_state = stick_state_for_rc(initial_rc, config)
    state.actuator[:, 0] = initial_rc[:, 3]
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    frozen_image = render_visual_hover_scene(state, target, config=config)
    held_motor = torch.zeros(episodes, 4, device=device)
    heights = []
    vertical_velocities = []
    tilts = []
    throttles = []
    recontact = torch.zeros(episodes, dtype=torch.bool, device=device)
    policy_steps = round(seconds * policy_hz)
    for _ in range(policy_steps):
        if oracle:
            held_motor = motor_target_for_rc(oracle_rc(state, target, mass_scale, config), config)
        else:
            image = (
                frozen_image
                if frozen_visual
                else render_visual_hover_scene(state, target, config=config)
            )
            held_motor, neural = controller(
                image, observation_attitude(state, omit_attitude), neural
            )
        for _ in range(physics_steps_per_policy):
            rc, stick_state = sticks(held_motor, stick_state)
            state = quad(rc, state, mass_scale)
            heights.append(state.position[:, 2])
            vertical_velocities.append(state.velocity[:, 2])
            tilts.append(torch.linalg.vector_norm(state.euler[:, :2], dim=1))
            throttles.append(rc[:, 3])
            recontact |= state.position[:, 2] <= 0.005

    height = torch.stack(heights)
    vertical_velocity = torch.stack(vertical_velocities)
    tilt = torch.stack(tilts)
    throttle = torch.stack(throttles)
    final_steps = min(round(5.0 / config.dt), height.shape[0])
    final_error = height[-final_steps:] - target[None]
    episode_rmse = torch.sqrt(final_error.square().mean(dim=0))
    vertical_velocity_rms = torch.sqrt(vertical_velocity[-final_steps:].square().mean(dim=0))
    tilt_rms = torch.sqrt(tilt[-final_steps:].square().mean(dim=0))
    peak_to_peak = height[-final_steps:].max(dim=0).values - height[-final_steps:].min(dim=0).values
    observed_hover = throttle[-final_steps:].mean(dim=0)
    required_hover = mass_scale / config.thrust_to_weight
    centered_observed = observed_hover - observed_hover.mean()
    centered_required = required_hover - required_hover.mean()
    denominator = torch.sqrt(centered_observed.square().sum() * centered_required.square().sum())
    correlation = (centered_observed * centered_required).sum() / denominator.clamp_min(1.0e-8)
    per_episode_pass = (
        (episode_rmse <= 0.25)
        & (vertical_velocity_rms <= 0.50)
        & (tilt_rms <= math.radians(6.0))
        & ~recontact
    )
    return {
        "label": (
            "oracle"
            if oracle
            else "frozen_visual"
            if frozen_visual
            else "visual_only_attitude_ablation"
            if omit_attitude
            else "connectome"
        ),
        "episodes": episodes,
        "seconds": seconds,
        "policy_hz": policy_hz,
        "physics_hz": round(1.0 / config.dt),
        "success_rate": float(per_episode_pass.float().mean()),
        "altitude_rmse_final_5s_mean_m": float(episode_rmse.mean()),
        "altitude_rmse_final_5s_p95_m": float(torch.quantile(episode_rmse, 0.95)),
        "vertical_velocity_rms_final_5s_mean_mps": float(vertical_velocity_rms.mean()),
        "altitude_peak_to_peak_final_5s_mean_m": float(peak_to_peak.mean()),
        "tilt_rms_final_5s_mean_deg": float(torch.rad2deg(tilt_rms).mean()),
        "ground_contact_rate": float(recontact.float().mean()),
        "mean_observed_throttle_final_5s": float(observed_hover.mean()),
        "required_hover_throttle_mean": float(required_hover.mean()),
        "hover_throttle_rmse": float(torch.sqrt((observed_hover - required_hover).square().mean())),
        "hover_throttle_mass_correlation": float(correlation),
        "pass": float(per_episode_pass.float().mean()) >= 0.90,
    }


def save_checkpoint(
    path: Path,
    controller: ConnectomeController,
    args: argparse.Namespace,
    config: HoverConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "experiment": "full-visual-hover-v1",
            "controller": controller.state_dict(),
            "graph_sha256": file_sha256(args.graph),
            "hover_config": asdict(config),
            "camera": asdict(DEFAULT_VISUAL_CAMERA),
            "policy_hz": args.policy_hz,
            "omit_attitude": args.omit_attitude,
            "training_arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if args.evaluate_only and args.checkpoint is None:
        raise SystemExit("--evaluate-only requires --checkpoint")
    if not args.graph.is_file():
        raise SystemExit(
            f"full graph missing: run scripts/build_full_visual_connectome.py ({args.graph})"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy-hz must evenly divide the 100 Hz surrogate physics rate")
    physics_steps_per_policy = physics_hz // args.policy_hz
    seed_everything(args.seed)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if checkpoint["graph_sha256"] != file_sha256(args.graph):
            raise SystemExit("checkpoint graph hash does not match --graph")
        if checkpoint["policy_hz"] != args.policy_hz:
            raise SystemExit("checkpoint policy rate does not match --policy-hz")
        controller.load_state_dict(checkpoint["controller"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "controller.pt"
    history: list[dict[str, Any]] = []
    started = perf_counter()

    if not args.evaluate_only:
        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=args.static_learning_rate, weight_decay=1.0e-6
        )
        for iteration in range(args.static_iterations):
            metrics = static_imitation_step(
                controller,
                optimizer,
                batch=args.batch_size,
                unroll=args.unroll,
                device=device,
                config=config,
                omit_attitude=args.omit_attitude,
            )
            if iteration % 10 == 0 or iteration + 1 == args.static_iterations:
                entry = {"stage": "static", "iteration": iteration + 1, **metrics}
                history.append(entry)
                print(json.dumps(entry), flush=True)

        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=args.sequence_learning_rate, weight_decay=1.0e-6
        )
        for stage, iterations, teacher_only in (
            ("teacher_sequence", args.teacher_sequence_iterations, True),
            ("random_handoff", args.handoff_iterations, False),
        ):
            rollout = new_temporal_batch(
                controller,
                batch=args.batch_size,
                episode_policy_steps=args.episode_policy_steps,
                device=device,
                config=config,
                teacher_only=teacher_only,
            )
            for iteration in range(iterations):
                metrics, rollout = temporal_imitation_step(
                    controller,
                    optimizer,
                    rollout,
                    unroll=args.unroll,
                    physics_steps_per_policy=physics_steps_per_policy,
                    config=config,
                    omit_attitude=args.omit_attitude,
                )
                rollout = reset_finished_rollout(
                    controller,
                    rollout,
                    batch=args.batch_size,
                    episode_policy_steps=args.episode_policy_steps,
                    device=device,
                    config=config,
                    teacher_only=teacher_only,
                )
                if iteration % 10 == 0 or iteration + 1 == iterations:
                    entry = {"stage": stage, "iteration": iteration + 1, **metrics}
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
            save_checkpoint(checkpoint_path, controller, args, config)

    controller.eval()
    evaluation_seed = args.seed + 10_000
    nominal = evaluate(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        policy_hz=args.policy_hz,
        physics_steps_per_policy=physics_steps_per_policy,
        device=device,
        config=config,
        seed=evaluation_seed,
        omit_attitude=args.omit_attitude,
    )
    frozen_visual = evaluate(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        policy_hz=args.policy_hz,
        physics_steps_per_policy=physics_steps_per_policy,
        device=device,
        config=config,
        seed=evaluation_seed,
        omit_attitude=args.omit_attitude,
        frozen_visual=True,
    )
    without_attitude = None
    if not args.omit_attitude:
        without_attitude = evaluate(
            controller,
            episodes=args.evaluation_episodes,
            seconds=args.evaluation_seconds,
            policy_hz=args.policy_hz,
            physics_steps_per_policy=physics_steps_per_policy,
            device=device,
            config=config,
            seed=evaluation_seed,
            omit_attitude=True,
        )
    oracle_baseline = evaluate(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        policy_hz=args.policy_hz,
        physics_steps_per_policy=physics_steps_per_policy,
        device=device,
        config=config,
        seed=evaluation_seed,
        omit_attitude=False,
        oracle=True,
    )
    save_checkpoint(checkpoint_path, controller, args, config)
    report = {
        "passed": nominal["pass"],
        "experiment": "full-visual-hover-v1",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "plant_model_version": PLANT_MODEL_VERSION,
        "graph": str(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "connectome_nodes": controller.n_nodes,
        "connectome_edges": int(controller.edge_pre.numel()),
        "camera": asdict(DEFAULT_VISUAL_CAMERA),
        "policy_hz": args.policy_hz,
        "physics_hz": physics_hz,
        "actor_inputs": {
            "rgb_camera": True,
            "estimated_roll_pitch": not args.omit_attitude,
            "accelerometer": False,
            "mass": False,
            "hover_thrust": False,
            "external_history": False,
        },
        "training": {
            "performed_this_run": not args.evaluate_only,
            "privileged_hover_thrust_target": "mass_scale / thrust_to_weight",
            "target_available_at_deployment": False,
            "teacher_handoff_is_training_only": True,
            "elapsed_seconds": perf_counter() - started,
            "history": history,
        },
        "evaluation": nominal,
        "frozen_visual_ablation": frozen_visual,
        "attitude_ablation": without_attitude,
        "oracle_baseline": oracle_baseline,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"wrote {checkpoint_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
