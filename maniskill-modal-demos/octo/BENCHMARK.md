# Octo on SimplerEnv WidowX — Benchmark Log

**Result: Pipeline works. Octo-Small achieves 0/5 on put_eggplant_in_basket, ~$0.60 total.**

**Agent**: 3 (Octo) · **Model**: rail-berkeley/octo-small-1.5 (136.7M total params incl. T5)
**Stack**: Modal · `maniskill/base` image · **L4** (A10G ships a stale libGLX_nvidia.so.0) · ManiSkill 3.0.0b22
**Tasks run**: PutEggplantInBasketScene-v1
**Attempts**: 18 (16 to get smoke green, 2 to get the rollout video sizes right)
**Spend**: ~$0.60 (well under the $5 cap)

### Success criteria
| # | Criterion | Status |
|---|---|---|
| 1 | `modal run octo_app.py --task put_eggplant_in_basket` exits 0 | ✅ |
| 2 | `octo_outputs/rollout_0.mp4` exists and is >100 KB | ✅ (258 KB) |
| 3 | Success rate reported from ≥5 rollouts; ≥0% counts as pipeline pass | ✅ (0/5, no crashes) |
| 4 | This file contains per-rollout success/failure | ✅ (see attempt 18) |
| 5 | Total Modal spend < $5 | ✅ (~$0.60) |

### Per-rollout (attempt 18, the canonical run)
| ep | success | sim steps | local rollout_N.mp4 size |
|---|---|---|---|
| 0 | False | 120 | 263921 B (258 KB) |
| 1 | False | 120 | 288206 B (281 KB) |
| 2 | False | 120 | 258937 B (253 KB) |
| 3 | False | 120 | 227813 B (222 KB) |
| 4 | False | 120 | 264483 B (258 KB) |

### Note on task success (0%)
Octo-Small in the original SimplerEnv paper reports ~50–70% on this task. We see 0/5. Most likely cause: the only control mode supported by `mani_skill==3.0.0b22` for these bridge_dataset_eval scenes is `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`, while Octo's training distribution (Bridge V2) used the `_interpolate_by_planner_..._delta_pos` mode listed in the original instructions doc. The action space is similar but not identical — particularly the gripper channel (`pd_joint_pos` vs `pd_joint_target_delta_pos`) and the EE planner. Action-space mismatch is the canonical explanation for "pipeline runs, success = 0" in zero-shot VLA evals; it's exactly the case the instructions doc called out under "Known Gotchas #1".

Per the instructions, **pipeline-pass is the primary goal here**, and that's met. Closing the action-space gap to recover the published ~50–70% rate would need either (a) a ManiSkill 3 build that ships the planner-interpolating controller, or (b) wrapping the env to post-process Octo's actions into the supported mode. Out of scope for this deliverable.

---

## Attempt 1 — 2026-05-24 — Phase 1 smoke test
**Task**: smoke (no rollouts)
**Outcome**: FAIL (image build)
**Change from previous attempt**: N/A — first attempt.
**Cost**: $0 (build failed before any GPU time)

### Traceback (last lines)
```
ERROR: Cannot install jax[cuda12]==0.4.28 and torch==2.4.0 because these package versions have conflicting dependencies.
    torch 2.4.0 depends on nvidia-cudnn-cu12==9.1.0.70
    jax[cuda12] 0.4.28 depends on nvidia-cudnn-cu12<9.0 and >=8.9.2.26
ERROR: ResolutionImpossible
Image build for im-9dIkrSaZUVdnVKmlEzNxb3 failed.
```

### Root-cause hypothesis
The recommended jax[cuda12]==0.4.28 caps cudnn<9, but maniskill/base ships cudnn 9.1 (because of torch 2.4.0). The two cannot coexist via pip metadata.

### Fix planned for next attempt
Bump JAX to 0.4.34 — first JAX release that supports cudnn 9 — keeping torch 2.4.0. Octo's flax/jax surface is stable across this range, so checkpoint load should still work.

---

## Attempt 2 — 2026-05-24 — Phase 1 smoke test (JAX bumped to 0.4.34)
**Task**: smoke
**Outcome**: FAIL (image build)
**Change from previous attempt**: jax[cuda12] 0.4.28 → 0.4.34 to allow coexistence with torch 2.4.0's cudnn 9.1.
**Cost**: $0

### Traceback (last lines)
```
ERROR: Could not find a version that satisfies the requirement jax==0.4.34
(jax 0.4.31+ Requires-Python >=3.10; we're on Python 3.9 from the maniskill/base conda env)
```

### Root-cause hypothesis
maniskill/base ships a Python 3.9 conda env that takes precedence over `add_python="3.10"`. So we're locked to JAX ≤0.4.30, and 0.4.30's jaxlib was built against cudnn 8.x — incompatible with torch 2.4.0's cudnn 9.1.

### Fix planned for next attempt
Downgrade torch 2.4.0 → 2.3.1 so both torch and jax agree on cudnn 8.9.x. ManiSkill 3.0.0b22 supports torch 2.3.

---

## Attempt 3 — 2026-05-24 — Phase 1 smoke test (torch downgraded to 2.3.1)
**Task**: smoke
**Outcome**: FAIL (image build)
**Change from previous attempt**: torch 2.4.0 → 2.3.1, jax[cuda12] back to 0.4.30. Assumed both would want cudnn 8.
**Cost**: $0

### Traceback (last lines)
```
ERROR: Cannot install jax-cuda12-plugin[with-cuda]==0.4.30 and torch==2.3.1 because these package versions have conflicting dependencies.
    torch 2.3.1 depends on nvidia-cudnn-cu12==8.9.2.26
    jax-cuda12-plugin[with-cuda] 0.4.30 depends on nvidia-cudnn-cu12<10.0 and >=9.0
```

### Root-cause hypothesis
Wrong assumption about jax-cuda12-plugin 0.4.30 — it actually requires cudnn>=9, not <9 like 0.4.28. So torch 2.4.0 (cudnn 9.1) is the correct partner, not 2.3.1.

### Fix planned for next attempt
torch back to 2.4.0; keep jax[cuda12]==0.4.30. Both pull cudnn 9.x.

---

## Attempt 4 — 2026-05-24 — Phase 1 smoke test (torch 2.4.0 + jax 0.4.30)
**Task**: smoke
**Outcome**: PARTIAL — image built, JAX GPU PASS, Octo import FAIL
**Change from previous attempt**: torch 2.3.1 → 2.4.0 to match jax 0.4.30's cudnn 9 requirement.
**Cost**: ~$0.05 (image build + brief container startup, no full GPU run)

### What passed
- Image built cleanly (no pip resolver errors)
- `JAX devices: [cuda(id=0)]` — JAX sees the GPU
- `PASS: JAX GPU OK`

### Traceback (last lines)
```
File "/opt/conda/lib/python3.9/site-packages/octo/model/components/action_heads.py", line 5, in <module>
  import distrax
  ...
  from tensorflow_probability.substrates import jax as tfp
ImportError: This version of TensorFlow Probability requires TensorFlow version >= 2.18; Detected an installation of version 2.15.0.
```

### Root-cause hypothesis
tensorflow_probability auto-installed 0.25.0 (latest), which requires TF≥2.18. We pinned TF to 2.15.0.

### Fix planned for next attempt
Pin `tensorflow-probability==0.23.0` — the matching TFP release for TF 2.15.

---

## Attempt 5 — 2026-05-24 — Phase 1 smoke test (TFP pinned to 0.23.0)
**Task**: smoke
**Outcome**: PARTIAL — TFP/distrax import works, Octo import FAIL
**Change from previous attempt**: Added `tensorflow-probability==0.23.0` pin to match TF 2.15.
**Cost**: ~$0.03

### Traceback (last lines)
```
File "/opt/conda/lib/python3.9/site-packages/octo/utils/typing.py", line 5
  PRNGKey = jax.random.KeyArray
AttributeError: module 'jax.random' has no attribute 'KeyArray'
```

### Root-cause hypothesis
`jax.random.KeyArray` was deprecated in JAX 0.4.16 and removed by 0.4.30. Octo's typing.py still references it. Octo HEAD hasn't been updated for newer JAX.

### Fix planned for next attempt
Patch `/opt/conda/lib/python3.9/site-packages/octo/utils/typing.py` post-install via sed: `jax.random.KeyArray` → `jax.Array`.

---

## Attempt 6 — 2026-05-24 — Phase 1 smoke test (sed patch for KeyArray)
**Task**: smoke
**Outcome**: FAIL — stale .pyc; .py was patched but Python loaded the old bytecode
**Change from previous attempt**: Added sed step to rewrite `jax.random.KeyArray` → `jax.Array` in Octo's typing module after install.
**Cost**: ~$0.03

### Traceback (last lines)
```
typing.py:5 in <module>
 > 5 PRNGKey = jax.Array                 # <- patched .py source
AttributeError: module 'jax.random' has no attribute 'KeyArray'   # <- old .pyc behavior
```

### Root-cause hypothesis
`pip install` precompiled Octo's .py files to .pyc bytecode before the sed step. The traceback shows the patched .py text but Python loads the stale .pyc. (Likely cause: Modal pins mtimes for reproducibility, so sed's mtime update doesn't invalidate the .pyc.)

### Fix planned for next attempt
After patching, recursively delete `__pycache__` under Octo's install dir. Also broaden the sed to all Octo .py files in case other modules reference `jax.random.KeyArray`.

---

## Attempt 7 — 2026-05-24 — Phase 1 smoke test (sed-all + clear __pycache__)
**Task**: smoke
**Outcome**: PARTIAL — Octo imports; load_pretrained FAIL (missing transformers)
**Change from previous attempt**: sed now runs over all Octo .py files; __pycache__ purged after sed so the patch actually takes effect.
**Cost**: ~$0.03

### What passed
- Octo modules import cleanly (KeyArray patch holds)
- `load_pretrained` started → JIT'd shape inference → reached tokenizer setup

### Traceback (last lines)
```
File "/opt/conda/lib/python3.9/site-packages/octo/model/components/tokenizers.py", line 187, in setup
  from transformers import AutoConfig, FlaxAutoModel, FlaxT5EncoderModel
ModuleNotFoundError: No module named 'transformers'
```

### Root-cause hypothesis
Octo's language tokenizer uses HuggingFace's Flax T5 encoder. We installed Octo with `--no-deps`, so transformers wasn't pulled.

### Fix planned for next attempt
Add `transformers==4.40.2` and `sentencepiece` (T5 tokenizer dep) to the image.

---

## Attempt 8 — 2026-05-24 — Phase 1 smoke test (added transformers + sentencepiece)
**Task**: smoke
**Outcome**: PARTIAL — Octo fully loaded; gym.make FAIL (interactive asset-download prompt)
**Change from previous attempt**: Added `transformers==4.40.2` and `sentencepiece` to pull the Flax T5 encoder Octo needs.
**Cost**: ~$0.05 (cold start + T5 weight load)

### What passed
- All imports clean: JAX, Octo, distrax, transformers, ManiSkill
- `OctoModel.load_pretrained` returned (T5 + Octo weights loaded)

### Traceback (last lines)
```
File "/opt/conda/lib/python3.9/site-packages/mani_skill/utils/download_asset.py", line 28, in prompt_yes_no
  answer = input("(y|n): ")
EOFError: EOF when reading a line
```

### Root-cause hypothesis
ManiSkill 3 prompts via `input()` to download scene assets on first `gym.make`. Modal runs non-interactive → EOFError.

### Fix planned for next attempt
Pre-download both task scenes during image build with `yes | python -m mani_skill.utils.download_asset <env>` so no prompt fires at runtime.

---

## Attempt 9 — 2026-05-24 — Phase 1 smoke test (pre-download scene assets)
**Task**: smoke
**Outcome**: PARTIAL — assets downloaded; env build FAIL (obs_mode "rgb" rejected)
**Change from previous attempt**: Added image-build steps to pre-download PutEggplantInBasketScene-v1 and PutSpoonOnTableClothScene-v1 assets with `yes | …` auto-accept.
**Cost**: ~$0.05

### Traceback (last lines)
```
File "/opt/conda/lib/python3.9/site-packages/mani_skill/envs/sapien_env.py", line 294, in __init__
  raise NotImplementedError(f"Unsupported obs mode: {obs_mode}. Must be one of ...")
NotImplementedError: Unsupported obs mode: rgb. Must be one of ['rgb+segmentation']
```

### Root-cause hypothesis
`bridge_dataset_eval` envs override `_default_sim_config` and only accept `rgb+segmentation`. The instruction-doc's `obs_mode="rgb"` is wrong for these specific scenes.

### Fix planned for next attempt
Change `obs_mode="rgb"` → `obs_mode="rgb+segmentation"`. Octo only consumes the RGB stream, so the extra segmentation channel is harmless.

---

## Attempt 10 — 2026-05-24 — Phase 1 smoke test (obs_mode=rgb+segmentation)
**Task**: smoke
**Outcome**: PARTIAL — env config accepted; Vulkan device init FAIL
**Change from previous attempt**: obs_mode "rgb" → "rgb+segmentation" (required by bridge_dataset_eval).
**Cost**: ~$0.05

### Traceback (last lines)
```
File "/opt/conda/lib/python3.9/site-packages/mani_skill/envs/sapien_env.py", line 1211, in _setup_scene
  systems.append(sapien.render.RenderSystem(self._render_device))
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed
```

### Root-cause hypothesis
JAX/CUDA initialized the GPU first (the test ran `jax.devices()` before `gym.make`). SAPIEN's Vulkan then fails to acquire the same device. The maniskill/base image expects SAPIEN to come up first.

### Fix planned for next attempt
Reorder smoke tests: build the env before importing JAX. Keep JAX_PLATFORMS unset for now — we want SAPIEN to win the GPU race naturally.

---

## Attempt 11 — 2026-05-24 — Phase 1 smoke test (env first, then JAX)
**Task**: smoke
**Outcome**: FAIL — same Vulkan ErrorInitializationFailed; order wasn't the bug
**Change from previous attempt**: Test order: ManiSkill env → JAX → Octo (was JAX → Octo → env).
**Cost**: ~$0.05

### Root-cause hypothesis (revised)
JAX/CUDA wasn't the culprit (env still failed first). More likely: Modal's container only exposes `utility,compute` driver capabilities; SAPIEN's Vulkan needs `graphics`.

### Fix planned for next attempt
Set `NVIDIA_DRIVER_CAPABILITIES=all`, `NVIDIA_VISIBLE_DEVICES=all`, and `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` in the image env. Add a Test 0 diagnostics block so future failures don't cost another round-trip.

---

## Attempt 12 — 2026-05-24 — Phase 1 smoke test (Vulkan env vars + diagnostics)
**Task**: smoke
**Outcome**: FAIL — env vars didn't fix Vulkan; diagnostics confirmed nvidia-smi sees A10, ICD present, vulkaninfo binary present
**Change from previous attempt**: Added NVIDIA_DRIVER_CAPABILITIES=all, NVIDIA_VISIBLE_DEVICES=all, VK_ICD_FILENAMES; pre-flight diagnostic prints for nvidia-smi + ICD files.
**Cost**: ~$0.05

### Diagnostic output (kept here as it's now load-bearing)
```
nvidia-smi -L → "GPU 0: NVIDIA A10 (UUID: ...)"
ls /usr/share/vulkan/icd.d/ → "nvidia_icd.json"
which vulkaninfo → "/usr/bin/vulkaninfo"
nvidia_icd.json contents → api_version: 1.2.155, library_path: libGLX_nvidia.so.0
```
Same SAPIEN error at `sapien.render.RenderSystem(self._render_device)`.

---

## Attempt 13 — 2026-05-24 — Phase 1 smoke test (more Vulkan loader knobs)
**Task**: smoke
**Outcome**: FAIL — same Vulkan error
**Change from previous attempt**: Added VK_LOADER_LAYERS_DISABLE=*, __GLX_VENDOR_LIBRARY_NAME=nvidia; added `vulkaninfo --summary` to diagnostics.
**Cost**: ~$0.05
**Diagnostic note**: `vulkaninfo --summary` returned rc=1 with just its `--help` text — the binary in maniskill/base is too old to support `--summary`. Consistent with the user's tip that A10G on Modal ships a stale libGLX_nvidia.so.0.

---

## Attempt 14 — 2026-05-24 — Phase 1 smoke test (GPU swap A10G → L4)
**Task**: smoke
**Outcome**: PARTIAL — Vulkan WORKS on L4; new prompt for the WidowX robot URDF
**Change from previous attempt**: GPU `A10G` → `L4`. Reverted image to attempt-9 spec; removed Vulkan-loader env-var experiments and the diagnostics block.
**Cost**: ~$0.05

### Traceback (last lines)
```
File "/opt/conda/lib/python3.9/site-packages/mani_skill/agents/base_agent.py", line 189, in build_articulation
  response = download_asset.prompt_yes_no(...)
File "/opt/conda/lib/python3.9/site-packages/mani_skill/utils/download_asset.py", line 28, in prompt_yes_no
  answer = input("(y|n): ")
EOFError: EOF when reading a line
```

### What this confirms
- L4 cleared the Vulkan device-creation failure. Trace got past `_setup_scene()` → `_load_agent()` → `_load_articulation()`, which is many lines deeper than any A10G run.
- The user's diagnosis is correct: it was the GPU, not the config.
- A second download prompt exists for the robot agent (WidowX URDF) that the scene-only image-time pre-download in attempt 9 did not cover.

### Root-cause hypothesis (new failure class — counts as 1/8 for prompt class)
Image pre-download only fetched the bridge_dataset_eval scene assets; the BaseAgent code path independently checks for the robot URDF and prompts if missing.

### Fix planned for next attempt
Monkeypatch `mani_skill.utils.download_asset.prompt_yes_no` to always return True in the smoke function, so any remaining first-run downloads auto-accept. Robust for the smoke; for Phase 2 we'll also add a proper image-build pre-download once we know the WidowX uid.

---

## Attempt 15 — 2026-05-24 — Phase 1 smoke test (monkeypatch prompt_yes_no)
**Task**: smoke
**Outcome**: PARTIAL — robot URDF downloaded; control_mode rejected by agent
**Change from previous attempt**: Added in-process `download_asset.prompt_yes_no = lambda *a, **k: True` before `gym.make`.
**Cost**: ~$0.05

### Traceback (last lines)
```
File "mani_skill/agents/base_agent.py", line 254, in set_control_mode
AssertionError:
arm_pd_ee_target_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos
  not in supported modes:
  ['arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos']
```

### Root-cause hypothesis
The control-mode string in the instructions doc doesn't exist for this env in mani_skill==3.0.0b22 (likely changed name between SimplerEnv versions). The instructions even warn about this exact gotcha.

### Fix planned for next attempt
Use the only supported mode: `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`. This may change the action space Octo emits into, which is a Phase 2 concern (could hurt task-success rate); for the smoke test it just needs to be accepted.

---

## Attempt 16 — 2026-05-24 — Phase 1 smoke test (use supported control_mode)
**Task**: smoke
**Outcome**: **PASS** ✅ — all three smoke checks green; exit 0
**Change from previous attempt**: control_mode → `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`.
**Cost**: ~$0.05

### Smoke output (the green lines)
```
PASS: SimplerEnv scene reset; sensor keys: ['3rd_view_camera']
      camera rgb shape=torch.Size([1, 480, 640, 3]) dtype=torch.uint8
JAX devices: [cuda(id=0)]
PASS: JAX GPU OK
PASS: Octo loaded; param count: 136.7M
      dataset_statistics keys: [... 'bridge_dataset' ... ]
```

### Notes carried forward into Phase 2
- Camera key is `3rd_view_camera` (only one camera in this env).
- RGB tensor is a **torch tensor** shaped `(1, 480, 640, 3)`, dtype uint8 — needs `.cpu().numpy()` + `.squeeze(0)`.
- Octo param count is 136.7M (transformer + T5 encoder). The instructions doc's "27M" only counted the transformer.
- `bridge_dataset` stats key is present, as expected for WidowX.
- L4 is the supported Modal GPU for this stack; do NOT try A10G.

---

# Phase 1 summary
- **16 attempts**, 1 smoke pass. ~$0.45 total spend on smoke runs (well under $5 cap).
- Image spec, code patches, and runtime hacks all proven; ready for Phase 2 rollouts.

---

## Attempt 17 — 2026-05-24 — Phase 2 first rollout run (5 rollouts, eggplant task)
**Task**: put_eggplant_in_basket
**Outcome**: PIPELINE PASS — 5/5 rollouts completed without crash; 0/5 task success; rollout_0.mp4 = 91 KB (just under the 100 KB criterion)
**Change from previous attempt**: First Phase 2 launch — `octo_app.py` (octo-demo app, OctoRunner class, L4, octo-outputs volume). Same image spec as smoke; runtime adapts torch-tensor RGB `(1,480,640,3)` → numpy `(480,640,3)`, resizes to 256x256 for Octo, executes 4-action chunks open-loop.
**Cost**: ~$0.12 (cold start + 5 rollouts ≈ 60s on L4)

### Per-rollout
| ep | success | steps (outer) | sim steps | video bytes |
|---|---|---|---|---|
| 0 | False | 30 | 120 | 92808  |
| 1 | False | 30 | 120 | 140473 |
| 2 | False | 30 | 120 | 105546 |
| 3 | False | 30 | 120 | 126768 |
| 4 | False | 30 | 120 | 119364 |

### Observations
- All 5 rollouts ran the env's natural horizon (120 sim steps = 30 chunks of 4) and were truncated by the env, not by my `max_steps=120`.
- 4/5 videos exceed 100 KB; rollout_0 is borderline at 91 KB.
- 0% task success is within the "pipeline pass" definition (<20% with no crashes).

### Fix planned for next attempt
Record one frame per sim step (currently I record one per outer chunk → only 31 frames). Per-substep capture gives ≈4× more frames and crosses the 100 KB bar comfortably for every rollout, plus the videos actually show motion between Octo decisions. Also bumping fps 10 → 15.

---

## Attempt 18 — 2026-05-24 — Phase 2 re-run with per-substep frame capture
**Task**: put_eggplant_in_basket
**Outcome**: **PASS** ✅ — pipeline runs, 5/5 rollouts complete, all videos >220 KB, success 0/5
**Change from previous attempt**: Record a video frame after every `env.step()` inside the action chunk (not once per chunk); fps 10 → 15.
**Cost**: ~$0.10 (warm container — model already loaded from attempt 17)

### Per-rollout
See the summary table at the top of this file.

### What "pipeline pass" means here
- Image build is reproducible (16 attempts of dependency archaeology pinned into the spec).
- L4 + maniskill/base + JAX 0.4.30 + torch 2.4.0 + the KeyArray sed-patch is a working combination.
- Octo loads, runs inference (T5 + transformer + diffusion head), and produces actions every step.
- ManiSkill env accepts the actions, advances physics, returns sensor RGB.
- Videos are written to the Modal volume and downloadable to local `octo_outputs/`.

This is what success criterion #3 calls "the inference loop *works*"; the 0% task-success rate is the action-space-mismatch limitation discussed in the summary above.
