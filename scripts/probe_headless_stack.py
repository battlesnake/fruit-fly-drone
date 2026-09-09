#!/usr/bin/env python3
"""Exercise the headless MuJoCo/FlyGym/Warp/PyTorch path used by this project."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

QUAD_MJCF = r"""
<mujoco model="headless_quad_probe">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="256" offheight="256"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="black_sky" type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0"
             width="8" height="8"/>
    <material name="floor_mat" rgba="0.35 0.35 0.35 1"/>
    <material name="gate_mat" rgba="1 0.08 0.03 1"/>
    <material name="quad_mat" rgba="0.1 0.1 0.1 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="1 1 1" specular="0 0 0"/>
    <geom name="floor" type="plane" size="30 30 0.1" material="floor_mat"/>
    <body name="gate" pos="0 4 1.5">
      <geom name="gate_left" type="box" pos="-1.6 0 0" size="0.12 0.12 1.5"
            material="gate_mat"/>
      <geom name="gate_right" type="box" pos="1.6 0 0" size="0.12 0.12 1.5"
            material="gate_mat"/>
      <geom name="gate_top" type="box" pos="0 0 1.4" size="1.72 0.12 0.12"
            material="gate_mat"/>
      <geom name="gate_bottom" type="box" pos="0 0 -1.4" size="1.72 0.12 0.12"
            material="gate_mat"/>
    </body>
    <body name="quad" pos="0 -2 1.5">
      <freejoint name="quad_free"/>
      <inertial pos="0 0 0" mass="0.035" diaginertia="0.00003 0.00003 0.00005"/>
      <geom name="quad_body" type="box" size="0.12 0.08 0.025" material="quad_mat"/>
      <camera name="fpv" pos="0 0.03 0" xyaxes="1 0 0 0 0 1" fovy="100"/>
    </body>
  </worldbody>
</mujoco>
"""


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def make_quad_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(QUAD_MJCF)


def probe_cpu_mujoco(steps: int) -> dict[str, Any]:
    import mujoco
    import numpy as np

    model = make_quad_model()
    quad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad")

    def rollout() -> np.ndarray:
        data = mujoco.MjData(model)
        for _ in range(steps):
            data.xfrc_applied[quad_id, 2] = 0.035 * 9.81
            mujoco.mj_step(model, data)
        return data.qpos.copy()

    started = perf_counter()
    first = rollout()
    second = rollout()
    elapsed = perf_counter() - started
    deterministic = bool(np.array_equal(first, second))
    finite = bool(np.isfinite(first).all())
    return {
        "passed": deterministic and finite,
        "steps_per_second": 2 * steps / elapsed,
        "deterministic": deterministic,
        "finite": finite,
        "final_qpos": first.tolist(),
    }


def probe_mjwarp(n_worlds: int, steps: int, resolution: int) -> dict[str, Any]:
    import mujoco_warp as mjw
    import torch
    import warp as wp

    wp.init()
    device = wp.get_device("cuda:0")
    model_cpu = make_quad_model()

    cold_startup_started = perf_counter()
    with wp.ScopedDevice(device):
        model = mjw.put_model(model_cpu)
        data = mjw.make_data(model_cpu, nworld=n_worlds)
        mjw.forward(model, data)
        render_context = mjw.create_render_context(
            model_cpu,
            nworld=n_worlds,
            cam_res=(resolution, resolution),
            render_rgb=True,
            render_depth=False,
            use_textures=False,
            use_shadows=False,
        )
        rgb = wp.zeros(
            (n_worlds, resolution, resolution),
            dtype=wp.vec3,
            device=device,
        )

    # Compile every kernel once before measuring steady-state throughput.
    mjw.step(model, data)
    mjw.refit_bvh(model, data, render_context)
    mjw.render(model, data, render_context)
    mjw.get_rgb(render_context, camera_index=0, rgb_out=rgb)
    wp.synchronize_device(device)
    cold_startup_seconds = perf_counter() - cold_startup_started

    step_started = perf_counter()
    for _ in range(steps):
        mjw.step(model, data)
    wp.synchronize_device(device)
    step_seconds = perf_counter() - step_started

    render_repeats = max(3, min(steps, 20))
    render_started = perf_counter()
    for _ in range(render_repeats):
        mjw.refit_bvh(model, data, render_context)
        mjw.render(model, data, render_context)
        mjw.get_rgb(render_context, camera_index=0, rgb_out=rgb)
    wp.synchronize_device(device)
    render_seconds = perf_counter() - render_started

    rgb_torch = wp.to_torch(rgb)
    torch.cuda.synchronize()
    pointer_shared = rgb_torch.data_ptr() == rgb.ptr
    image_min = float(rgb_torch.min().item())
    image_max = float(rgb_torch.max().item())
    image_mean = float(rgb_torch.float().mean().item())
    image_varies = math.isfinite(image_mean) and image_max > image_min
    cuda_resident = rgb_torch.device.type == "cuda"
    passed = pointer_shared and image_varies and cuda_resident

    return {
        "passed": passed,
        "device": str(device),
        "device_name": device.name,
        "n_worlds": n_worlds,
        "resolution": resolution,
        "cold_startup_seconds": cold_startup_seconds,
        "physics_steps_per_second": n_worlds * steps / step_seconds,
        "rendered_frames_per_second": n_worlds * render_repeats / render_seconds,
        "torch_shape": list(rgb_torch.shape),
        "torch_dtype": str(rgb_torch.dtype),
        "cuda_resident": cuda_resident,
        "zero_copy_pointer": pointer_shared,
        "image_min": image_min,
        "image_max": image_max,
        "image_mean": image_mean,
    }


def probe_flygym(n_worlds: int, steps: int) -> dict[str, Any]:
    import numpy as np
    import warp as wp
    from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
    from flygym.compose import (
        ActuatorType,
        KinematicPosePreset,
        NeuroMechFly,
        TetheredWorld,
    )
    from flygym.utils.math import Rotation3D
    from flygym.warp import GPUSimulation

    fly = NeuroMechFly()
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    actuated_dofs = fly.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(
        actuated_dofs,
        actuator_type=ActuatorType.POSITION,
        kp=50.0,
        neutral_input=KinematicPosePreset.NEUTRAL,
    )
    fly.colorize()

    world = TetheredWorld()
    world.add_fly(
        fly,
        spawn_position=(0, 0, 0.8),
        spawn_rotation=Rotation3D("quat", (1, 0, 0, 0)),
    )
    cold_startup_started = perf_counter()
    sim = GPUSimulation(world, n_worlds)
    sim.reset()
    controls = np.zeros((n_worlds, len(actuated_dofs)), dtype=np.float32)
    sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, controls)
    sim.step()
    wp.synchronize()
    cold_startup_seconds = perf_counter() - cold_startup_started

    step_started = perf_counter()
    for _ in range(steps):
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, controls)
        sim.step()
    wp.synchronize()
    step_seconds = perf_counter() - step_started

    joint_angles = sim.get_joint_angles(fly.name)
    joint_angles_np = joint_angles.numpy()
    finite = bool(np.isfinite(joint_angles_np).all())
    dof_names = [str(dof) for dof in actuated_dofs]
    front_dofs = [name for name in dof_names if "front" in name.lower() or "f_" in name.lower()]

    return {
        "passed": finite and len(actuated_dofs) > 0,
        "n_worlds": n_worlds,
        "n_actuated_dofs": len(actuated_dofs),
        "front_dof_names": front_dofs,
        "cold_startup_seconds": cold_startup_seconds,
        "physics_steps_per_second": n_worlds * steps / step_seconds,
        "finite_joint_angles": finite,
    }


def run_probe(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        result = function()
        result["status"] = "passed" if result.get("passed") else "failed"
        return result
    except Exception as exc:  # Probe reports are more useful than an early traceback.
        return {
            "passed": False,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "probe": name,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-worlds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--only",
        choices=("all", "cpu", "mjwarp", "flygym"),
        default="all",
        help="Run one probe or the full acceptance set.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_worlds < 1 or args.steps < 1 or args.resolution < 8:
        raise SystemExit("n-worlds and steps must be positive; resolution must be at least 8")

    selected = ("cpu", "mjwarp", "flygym") if args.only == "all" else (args.only,)
    probes: dict[str, dict[str, Any]] = {}
    if "cpu" in selected:
        probes["cpu_mujoco"] = run_probe("cpu_mujoco", lambda: probe_cpu_mujoco(args.steps))
    if "mjwarp" in selected:
        probes["mjwarp_rgb_torch"] = run_probe(
            "mjwarp_rgb_torch",
            lambda: probe_mjwarp(args.n_worlds, args.steps, args.resolution),
        )
    if "flygym" in selected:
        probes["flygym_gpu"] = run_probe(
            "flygym_gpu", lambda: probe_flygym(args.n_worlds, args.steps)
        )

    report = {
        "passed": all(probe["passed"] for probe in probes.values()),
        "versions": package_versions(
            ("flygym", "mujoco", "mujoco-warp", "warp-lang", "torch", "numpy")
        ),
        "parameters": {
            "n_worlds": args.n_worlds,
            "steps": args.steps,
            "resolution": args.resolution,
        },
        "probes": probes,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
