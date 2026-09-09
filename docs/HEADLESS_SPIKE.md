# Headless simulation acceptance spike

The first implementation milestone proves that the proposed simulation stack works under
WSL2 before environment or learning code depends on it.

## Pass criteria

1. Headless CPU MuJoCo advances a free-body quad model deterministically without creating
   a display or OpenGL context.
2. MJWarp advances multiple worlds on the RTX 5080 and renders an RGB image from a fixed
   camera using its GPU batch renderer.
3. The rendered Warp buffer is exposed to PyTorch without copying and remains on CUDA.
4. FlyGym advances multiple copies of a tethered fly with both front legs actuated.
5. Each probe emits machine-readable timing, version, device and memory information.

Measure complete sensor-to-action ticks rather than physics-only steps. The eventual
benchmark must include rendering, visual transduction, connectome inference, leg/stick
mechanics and quad dynamics.

## Environment

Create the local environment while reusing the CUDA-enabled PyTorch already installed in
WSL2:

```bash
uv venv --system-site-packages --python 3.12 .venv
aira confine -- uv pip install --python .venv/bin/python \
    --requirements pyproject.toml --extra sim --group dev
```

The simulation dependencies are deliberately isolated behind the `sim` extra. Training's
PyTorch dependency is a separate `train` extra so this probe reuses the CUDA-enabled build
already verified on the host instead of downloading a second CUDA runtime. FlyGym is
pinned because the 2.x API is new and version 2.1 changed its composition backend.

## Initial probes

The probes will live under `scripts/` and write JSON reports under the ignored
`runs/headless-probes/` directory. A successful CUDA import is not sufficient: WSL2 must
execute an actual MJWarp step, render RGB through the batch ray tracer and share that
buffer with PyTorch.

Run the complete small probe through the confined launcher:

```bash
scripts/run_headless_probe.sh \
    --n-worlds 8 --steps 20 --resolution 64 \
    --output runs/headless-probes/smoke.json
```

The launcher prioritizes `/usr/lib/wsl/lib` only when WSL's `libcuda.so` exists. This is
required on the current host because an ordinary Linux NVIDIA library under `/lib`
otherwise shadows the WSL driver for Warp, even though PyTorch finds the GPU.

## Initial result

The complete eight-world smoke test passed on 2026-09-09 with FlyGym 2.1.0, MuJoCo
3.9.0, MJWarp 3.9.0.1, Warp 1.14.0 and the host's PyTorch 2.12 development build:

| Probe | Result |
| --- | --- |
| CPU MuJoCo | Deterministic, finite free-body state without a renderer |
| MJWarp physics | Passed on RTX 5080 (`sm_120`) |
| MJWarp RGB | Eight headless 64×64 RGB images remained on CUDA |
| Warp → PyTorch | Identical allocation pointer; no copy |
| FlyGym GPU | Eight tethered flies, 42 actuated leg DoFs, finite state |
| Bilateral forelegs | Seven active DoFs detected on each front leg |

The cached smoke run observed roughly 370 batched RGB frames/s and 1,100 aggregate
FlyGym physics steps/s at only eight worlds. These are diagnostic numbers, not capacity
benchmarks: such a small batch under-occupies the GPU, timings vary while other AIRA jobs
share the device, and the first run spent about a minute compiling kernels. The next
benchmark must sweep batch size and retinal resolution after the actual stick and quad
models exist.
