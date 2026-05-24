# Octo on SimplerEnv WidowX (Agent 3 deliverable)

Zero-shot inference of [Octo-Small 1.5](https://huggingface.co/rail-berkeley/octo-small-1.5)
on the SimplerEnv WidowX `PutEggplantInBasketScene-v1` task, running on Modal.

**Status:** Pipeline pass. 5/5 rollouts complete, 0/5 task success (action-space mismatch — see [BENCHMARK.md](BENCHMARK.md)).

## Demo video

https://github.com/faisalhacks/maniskill-modal-demos/raw/main/octo/rollout_0.mp4

Episode 0 of the `PutEggplantInBasketScene-v1` rollout — Octo-Small 1.5 executing
`"put eggplant in basket"` zero-shot. The arm completes the full 120-step horizon
without crashing; task success is 0/5 across all five episodes (analysis in
[BENCHMARK.md](BENCHMARK.md)). File: [rollout_0.mp4](rollout_0.mp4) (~258 KB).

## Files
- [octo_smoke.py](octo_smoke.py) — Phase 1 smoke test (JAX + Octo load + ManiSkill env reset).
- [octo_app.py](octo_app.py) — Phase 2 demo: Modal class that loads Octo and runs N rollouts, writing one MP4 per episode.
- [BENCHMARK.md](BENCHMARK.md) — Full 18-attempt iteration log with every dependency-resolution dead-end documented.
- [rollout_0.mp4](rollout_0.mp4) — Episode 0 (committed inline). The remaining four `rollout_{1..4}.mp4` files are available from the Modal volume.

## Running it

```bash
# Phase 1 smoke (~30s on warm cache, ~5min cold image build):
modal run octo_smoke.py

# Phase 2 rollouts (~60s warm, ~2min cold):
modal run octo_app.py --task put_eggplant_in_basket --n-rollouts 5
modal volume get octo-outputs rollout_0.mp4
```

The second task (`put_spoon_on_towel`) is also wired up but wasn't run for this deliverable:

```bash
modal run octo_app.py --task put_spoon_on_towel --n-rollouts 5
```

## What was hard

Every gotcha in the brief was real, plus several it didn't mention. The image
spec in `octo_app.py` is the product of 16 smoke attempts. The non-obvious ones:

1. **A10G ships a stale `libGLX_nvidia.so.0`** that fails SAPIEN's
   `vk::PhysicalDevice::createDeviceUnique`. Use **L4**. (Confirmed by Agent 2
   independently; no Vulkan env-var tweak rescues A10G.)
2. **maniskill/base is Python 3.9**, not 3.10 — the `add_python="3.10"` arg is
   shadowed by the base image's conda env. This caps JAX at 0.4.30.
3. **jax 0.4.28 wants cudnn 8, jax 0.4.30 wants cudnn 9.** Same library, two
   adjacent versions, opposite pins. With Python 3.9 we're stuck at 0.4.30,
   which forces torch 2.4.0 (cudnn 9), not the 2.3.x most JAX-cudnn-8 stacks
   pair with.
4. **`tensorflow-probability` auto-resolves to 0.25** (needs TF ≥ 2.18); pin
   `tensorflow-probability==0.23.0` to match `tensorflow-cpu==2.15.0`.
5. **Octo's `octo/utils/typing.py` references `jax.random.KeyArray`**, removed
   from JAX 0.4.30. `sed`-patch every Octo `.py`, **then delete every
   `__pycache__`** — pip pre-compiles bytecode at install time and Modal's
   reproducible-build mtimes don't invalidate it on a simple `sed -i`.
6. **`pip install --no-deps`** is required to install Octo (its pyproject pins
   incompatible JAX/TF), which means **you have to bring `transformers` and
   `sentencepiece` yourself** (they're only needed by the T5 language head and
   aren't in Octo's runtime imports until `load_pretrained` is called).
7. **ManiSkill prompts via `input()`** to download scene assets on first env
   build. Modal's non-interactive shell → `EOFError`. Pre-download scenes in
   the image build (`yes | python -m mani_skill.utils.download_asset ...`) AND
   monkeypatch `mani_skill.utils.download_asset.prompt_yes_no` at runtime to
   auto-accept the *robot URDF* prompt (which the scene CLI doesn't cover).
8. **`obs_mode="rgb"` is rejected** by `bridge_dataset_eval` envs — only
   `rgb+segmentation` is allowed.
9. **The control-mode string in the brief doesn't exist** in
   `mani_skill==3.0.0b22`. The only supported mode is
   `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`. This is also the
   most likely cause of the 0% task success (Octo trained on a different
   action space).
10. **ManiSkill 3 returns sensor RGB as a torch tensor of shape
    `(1, H, W, 3)`** (not numpy, not `(H, W, 3)`) — needs
    `.cpu().numpy().squeeze(0)` before Octo or imageio touches it.

## Result summary

| Metric | Value |
|---|---|
| Smoke attempts | 16 |
| Rollout attempts | 2 |
| Final rollouts | 5/5 complete, 0/5 task success |
| Final rollout sim steps each | 120 (env's truncation horizon) |
| Per-rollout MP4 size | 222–281 KB |
| Total Modal spend | ~$0.60 / $5.00 budget |

For per-rollout numbers and every individual attempt's traceback, see
[BENCHMARK.md](BENCHMARK.md).
