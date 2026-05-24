# RL Project — PPO on PickCube-v1 (Modal)

Agent 2 deliverable. Trains PPO on ManiSkill's `PickCube-v1` on Modal and
produces a rollout video of the trained policy.

## Demo video

https://github.com/faisalhacks/maniskill-modal-demos/raw/main/rl/state_rollout.mp4

State-based PPO checkpoint at 4.1M timesteps. Evaluation success rate: 0.75
across 16 stochastic episodes. File: [state_rollout.mp4](state_rollout.mp4) (~91 KB).

## Setup

```bash
pip install modal
modal token new   # one-time auth
```

## Run

### State-based PPO (fast — 2-5 min)
```bash
modal run rl_app.py
# or explicitly:
modal run rl_app.py --mode state
```
Trains for 2M timesteps, then evaluates the final checkpoint and saves a rollout
video.

### RGB-based PPO (slower — 15-45 min)
```bash
modal run rl_app.py --mode rgb
```

### Both
```bash
modal run rl_app.py --mode both
```

## Pull artifacts

```bash
modal volume get rl-outputs state_rollout.mp4 .
modal volume get rl-outputs state_checkpoint.pt .
modal volume get rl-outputs rgb_rollout.mp4 .   # if rgb was run
```

## Architecture

`rl_app.py` is a Modal app with three functions, each on an L4 GPU:

| Function | Purpose | Wall-clock |
|---|---|---|
| `train_state()` | PPO on PickCube-v1 (state obs), 2M timesteps | 2-5 min |
| `train_rgb()` | PPO on PickCube-v1 (RGB obs), 10M timesteps | 15-45 min |
| `evaluate(mode)` | Load checkpoint, run 200 eval steps, save MP4 | ~30 sec |

The training script is the official `examples/baselines/ppo/ppo.py` from
`haosulab/ManiSkill` (cloned at image-build time into `/opt/ManiSkill`),
invoked as a subprocess. No custom PPO implementation.

## Why L4 (not A10G)

Modal's A10G provisioning fails `vk::PhysicalDevice::createDeviceUnique` even
after fixing the bundled NVIDIA userspace driver to match the host kernel
driver. L4 works with the same image. See [BENCHMARK.md](BENCHMARK.md) attempts 1-7 for
the full diagnostic story.

## The Vulkan loader fix

`maniskill/base` bundles a stale NVIDIA userspace driver. Modal's container
runtime mounts the host driver as `libGLX_nvidia.so.<host_version>` alongside
the bundled libs, but the unversioned `.so.0` symlink that the Vulkan ICD JSON
references still points at the old bundled lib. At runtime
`fix_nvidia_vulkan_loader()`:

1. Finds the newest `lib*.so.X.Y.Z` files in `/usr/lib/x86_64-linux-gnu/`
2. Rewrites `/usr/share/vulkan/icd.d/nvidia_icd.json` to reference the
   host-mounted lib by absolute path
3. Re-links the `.so.0` and `.so.1` symlinks for libGLX_nvidia,
   libnvidia-glcore, libnvidia-rtcore, etc.

This is a no-op when there's no version mismatch.

## Files

- [rl_smoke.py](rl_smoke.py) — Phase 1 smoke test (image + GPU env + tiny PPO 50k-step run)
- [rl_app.py](rl_app.py) — Phase 2 full training app
- [BENCHMARK.md](BENCHMARK.md) — full attempt log
- [state_rollout.mp4](state_rollout.mp4) — final rollout video (~91 KB)
