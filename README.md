# ManiSkill + Modal — Four Robotics Demos

End-to-end robotics demos on [ManiSkill](https://www.maniskill.ai/) running on
[Modal](https://modal.com/) serverless GPUs. Each demo is a self-contained
Modal app: image build → policy load → rollout → MP4 saved to a Modal volume.
Total Modal spend across all four: **~$3.80**.

---

## 1. VLM — Molmo points, ManiSkill picks

<video src="https://github.com/faisalhacks/maniskill-modal-demos/raw/main/vlm/rollout.mp4" controls width="640"></video>

Molmo-7B-D looks at a rendered RGB of a Franka Panda + red cube, returns
`<point x="..." y="...">` coordinates, the pixel is deprojected to a 3D
target via the depth buffer, and ManiSkill's motion planner executes the
grasp. L4 GPU, ~$1, pipeline PASS.

→ **[Read the full VLM demo writeup](vlm/README.md)** · [BENCHMARK.md](vlm/BENCHMARK.md) · [vlm_app.py](vlm/vlm_app.py)

---

## 2. RL — PPO from scratch on PickCube

<video src="https://github.com/faisalhacks/maniskill-modal-demos/raw/main/rl/state_rollout.mp4" controls width="640"></video>

PPO trained on `PickCube-v1` (Franka, state observations) for 4.1M timesteps
on a single L4. Evaluation success rate **0.75 across 16 stochastic episodes**.
Uses ManiSkill's official PPO baseline; ~$1.50 of Modal spend.

→ **[Read the full RL demo writeup](rl/README.md)** · [BENCHMARK.md](rl/BENCHMARK.md) · [rl_app.py](rl/rl_app.py)

---

## 3. Octo — 136.7M generalist policy on WidowX

<video src="https://github.com/faisalhacks/maniskill-modal-demos/raw/main/octo/rollout_0.mp4" controls width="640"></video>

Octo-Small 1.5 (T5 text encoder + transformer policy, 136.7M params total)
running zero-shot on the SimplerEnv WidowX `PutEggplantInBasketScene-v1`
digital-twin. Pipeline PASS — all 5 rollouts complete the 120-step horizon;
**0/5 task success** (action-space mismatch, detailed in BENCHMARK). L4, ~$0.60.

→ **[Read the full Octo demo writeup](octo/README.md)** · [BENCHMARK.md](octo/BENCHMARK.md) · [octo_app.py](octo/octo_app.py)

---

## 4. OpenVLA — 7B VLA on the same WidowX scene

<video src="https://github.com/faisalhacks/maniskill-modal-demos/raw/main/openvla/rollout_0.mp4" controls width="640"></video>

OpenVLA-7B in bf16 on Modal A100-40GB, same `PutEggplantInBasketScene-v1`
task as Octo. **1/5 task success** (episode 4 succeeded at step 59). ~247 ms
per inference step on A100; ~$0.70 of Modal spend.

→ **[Read the full OpenVLA demo writeup](openvla/README.md)** · [BENCHMARK.md](openvla/BENCHMARK.md) · [openvla_app.py](openvla/openvla_app.py)

---

## Summary table

| # | Demo | Policy | Task | Outcome | GPU | Cost |
|---|------|--------|------|---------|-----|------|
| 1 | [vlm/](vlm/) | Molmo-7B-D (VLM + motion planner) | `PickCube-v1` (Franka) | Pipeline PASS | L4 | ~$1 |
| 2 | [rl/](rl/) | PPO from scratch (state obs) | `PickCube-v1` (Franka) | state-success = 0.75 | L4 | ~$1.50 |
| 3 | [octo/](octo/) | Octo-Small (136.7M) | `PutEggplantInBasketScene-v1` (WidowX) | 0/5; pipeline PASS | L4 | ~$0.60 |
| 4 | [openvla/](openvla/) | OpenVLA-7B | `PutEggplantInBasketScene-v1` (WidowX) | 1/5 | A100-40GB | ~$0.70 |

## Headline finding

**Octo-Small (136.7M params) and OpenVLA-7B are within sampling noise of each
other on this benchmark slice (0/5 vs 1/5).** Both face the same action-space
mismatch between their BridgeV2 training distribution (`...delta_pos`) and the
only control mode `mani_skill==3.0.0b22` exposes for `PutEggplantInBasketScene-v1`
(`...align2_gripper_pd_joint_pos`). The 7B model isn't decisively better — the
benchmark is roughly a coin flip for both. See
[openvla/README.md](openvla/README.md) and
[octo/BENCHMARK.md](octo/BENCHMARK.md) for the full diagnosis.

## Critical infrastructure note (read first)

Three independent agents have now confirmed: **use `gpu="L4"`, not `gpu="A10G"`,
for any ManiSkill-on-Modal job that uses SAPIEN/Vulkan rendering.** A10G fails
`vk::PhysicalDevice::createDeviceUnique` even with the documented loader fix in
`maniskill/base`; L4 works out of the box. OpenVLA needs A100-40GB only because
of the 7B model weights, not the renderer.

Details in each per-demo `BENCHMARK.md`.

## Cross-agent intel (BridgeV2 digital-twin scenes)

If you're targeting any Bridge digital-twin scene on `mani_skill==3.0.0b22`,
these settings cost the OpenVLA agent four smoke attempts to discover:

- `control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"` (the *only* mode the `widowx250s_bridgedataset_sink` agent exposes)
- `obs_mode="rgb+segmentation"` (plain `"rgb"` is rejected by these scenes)
- Camera sensor key: `"3rd_view_camera"`
- Bake both assets into the Modal image build:
  - `python -m mani_skill.utils.download_asset bridge_v2_real2sim -y`
  - `python -m mani_skill.utils.download_asset widowx250s_bridgedataset_sink -y`
- Belt-and-suspenders: `mani_skill.utils.download_asset.prompt_yes_no = lambda *a, **kw: True` before `gym.make` catches any asset the build step missed.

## Repo layout

```
.
├── README.md          # this file
├── LICENSE            # MIT
├── vlm/               # Molmo + ManiSkill PickCube
├── rl/                # PPO on PickCube
├── octo/              # Octo-Small on SimplerEnv WidowX
└── openvla/           # OpenVLA-7B on SimplerEnv WidowX
```

Each demo folder contains a `README.md`, a `BENCHMARK.md` (full attempt
log), a Modal app script, a smoke-test script, and a rendered MP4.

## License

MIT — see [LICENSE](LICENSE).
