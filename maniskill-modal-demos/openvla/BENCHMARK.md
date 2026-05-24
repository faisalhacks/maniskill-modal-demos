# OpenVLA Project — Modal Attempt Log

**Pipeline works. OpenVLA-7B 1/5 on put_eggplant_in_basket, $0.70 total.**

6 attempts (5 smoke iterations + 1 successful full demo). Compare against Agent 3 (Octo) results in PROJECT_STATUS.md.

## ⚠️ Cross-agent intel for Agent 3 (Octo)
Found that the control_mode string in the original spec was wrong. Working control_mode is `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` — the only mode the `widowx250s_bridgedataset_sink` agent exposes in `mani_skill==3.0.0b22`. If Agent 3 uses the spec's `arm_pd_ee_target_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos`, env construction will assert-fail at `_load_agent → set_control_mode`. Also: `obs_mode` must be `"rgb+segmentation"` (plain `"rgb"` is rejected by these digital-twin scenes), camera key is `"3rd_view_camera"`, and BOTH `bridge_v2_real2sim` (scene mesh) AND `widowx250s_bridgedataset_sink` (robot URDF) must be downloaded in the image build to avoid the interactive `input("(y|n): ")` EOF in Modal.

Owner: Agent 4 (OpenVLA)
Goal: Zero-shot OpenVLA-7B on SimplerEnv `PutEggplantInBasketScene-v1`, save `rollout_*.mp4`. Pipeline success even if task success rate is low (OpenVLA is known weak on these scenes; ~10% published).

Constraints: `maniskill/base` image, `mani_skill==3.0.0b22`, `A100-40GB`, cost cap $5, 8-retry budget.

---

## Attempt 1 — 2026-05-24 — Phase 1 smoke test
**Outcome**: PARTIAL (steps 1–4 PASS, step 5 FAIL)
**Duration**: image build ~7 min + function ~2 min
**Cost**: ~$0.15 (image build is CPU; ~2 min on A100-40GB @ $3.40/hr ≈ $0.11)
**Change from previous attempt**: initial run.

### What passed
- A100-40GB visible: 42.4 GB total, 42.0 GB free
- Image build with `mani_skill==3.0.0b22 + torch==2.4.0 + transformers==4.40.1` succeeded (no Vulkan/SAPIEN conflict Agent 1 hit — different deps and different GPU)
- `AutoProcessor` from local `/root/openvla-7b` loaded
- 7B model loaded in bf16: **7.54 B params, 15.1 GB GPU mem**
- `predict_action` returned `(7,)` action `[0.0036, 0.0129, 0.0161, 0.0254, 0.0032, 0.0112, …]` — sane magnitudes for ee-delta + gripper

### Traceback (last lines)
```
Could not find asset bridge_v2_real2sim at /root/.maniskill/data/tasks/bridge_v2_real2sim_dataset
Environment PutEggplantInBasketScene-v1 requires asset(s) bridge_v2_real2sim which could not be found. Would you like to download them now?
(y|n): EOFError: EOF when reading a line
```

### Root-cause hypothesis
`PutEggplantInBasketScene-v1` triggers an interactive `input("(y|n): ")` on first use because the BridgeV2 real2sim asset bundle isn't shipped. Modal containers have no stdin so the prompt explodes.

### Fix planned for next attempt
Bake the asset download into the image: add `.run_commands("python -m mani_skill.utils.download_asset bridge_v2_real2sim -y")` after the pip install step. This caches the assets into the image layer so cold starts don't pay the download.

---

## Attempt 2 — 2026-05-24 — Phase 1 smoke test
**Outcome**: PARTIAL (steps 1–4 PASS, step 5 FAIL on env config)
**Duration**: image build ~4 min (asset download added 78 MB, HF download 16 GB cached) + function ~30 sec
**Cost**: ~$0.08 (function exited quickly after the env error)
**Change from previous attempt**: added `.run_commands("python -m mani_skill.utils.download_asset bridge_v2_real2sim -y")` to image build.

### What passed
- Asset download cached (78.2 MB bridge_v2_real2sim zip)
- All model load + inference assertions passed identically to Attempt 1

### Traceback (last lines)
```
NotImplementedError: Unsupported obs mode: rgb. Must be one of ['rgb+segmentation']
  File ".../mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/base_env.py", line 202
```

### Root-cause hypothesis
The digital-twin Bridge eval envs (subclass `BaseDigitalTwinEnv`) override `SUPPORTED_OBS_MODES` to only `["rgb+segmentation"]` — the spec's recommended `obs_mode="rgb"` is wrong for this scene class. The segmentation channel is required for the success-checker to identify the basket interior.

### Fix planned for next attempt
Change `obs_mode="rgb"` → `obs_mode="rgb+segmentation"`. The model still only sees the RGB channel via `obs["sensor_data"][cam_key]["rgb"]`; segmentation is parallel-stored under the same dict and ignored by the VLA.

---

## Attempt 3 — 2026-05-24 — Phase 1 smoke test
**Outcome**: PARTIAL (steps 1–4 PASS, step 5 FAIL on second asset)
**Duration**: ~1.5 min function (image already cached)
**Cost**: ~$0.08
**Change from previous attempt**: `obs_mode="rgb"` → `obs_mode="rgb+segmentation"`.

### Traceback (last lines)
```
Robot widowx250s_bridgedataset_sink definition file not found at
  /root/.maniskill/data/robots/widowx/wx250s.urdf
Robot widowx250s_bridgedataset_sink has assets available for download. Would you like to download them now?
(y|n): EOFError: EOF when reading a line
```

### Root-cause hypothesis
Got past the scene asset, hit the robot asset prompt next. The Bridge digital-twin scene uses a custom robot variant (`widowx250s_bridgedataset_sink` — a WidowX 250 6-DoF arm pinned in front of a Bridge-data sink scene) whose URDF isn't part of the scene asset bundle.

### Fix planned for next attempt
Two-pronged:
1. Add `python -m mani_skill.utils.download_asset widowx250s_bridgedataset_sink -y || true` to the image build (cache it in the layer if the CLI knows the UID).
2. Belt-and-suspenders: monkey-patch `mani_skill.utils.download_asset.prompt_yes_no` to return True before `gym.make`, so any asset the CLI doesn't recognise still auto-downloads at runtime (cold-start penalty but no crash).

---

## Attempt 4 — 2026-05-24 — Phase 1 smoke test
**Outcome**: PARTIAL (steps 1–4 PASS, step 5 FAIL on control_mode)
**Duration**: image rebuild ~30 sec + function ~1 min
**Cost**: ~$0.05
**Change from previous attempt**: added `widowx250s_bridgedataset_sink` to image build asset list; monkey-patched `prompt_yes_no` for runtime safety.

### Traceback (last lines)
```
AssertionError: arm_pd_ee_target_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos not in supported modes:
  ['arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos']
```

### Root-cause hypothesis
The spec's control_mode string is from an older / different ManiSkill version; in `mani_skill==3.0.0b22`, the `widowx250s_bridgedataset_sink` agent only exposes ONE control mode: `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`. This is the BridgeV2-compatible EE-delta-pose + gripper joint-pos mode that OpenVLA's BridgeV2 head was trained for, so semantically equivalent — just renamed.

### Fix planned for next attempt
Use the only supported control mode: `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`.

---

## Attempt 5 — 2026-05-24 — Phase 1 smoke test
**Outcome**: **PASS** — all five checks green
**Duration**: function ~30 sec (warm image)
**Cost**: ~$0.03
**Change from previous attempt**: control_mode → `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`.

### What passed (full log)
- CUDA available, A100-40GB, 42.0 GB free
- Processor + 7.54 B model loaded in bf16, 15.1 GB GPU mem
- `predict_action` → `(7,)` action, sensible magnitudes
- `PutEggplantInBasketScene-v1` reset OK
- Camera sensor key: **`'3rd_view_camera'`** (single third-person view, no wrist cam)

Harmless warnings (no action needed):
- Mimic joint controller config for `widowx250s_bridgedataset_sink` (gripper paired joint)
- No initial poses for `eggplant` / `dummy_sink_target_plane` (sim speed only)

**Phase 1 → DONE. Total Phase 1 spend ≈ $0.39 / $5 cap.**

---

## Attempt 6 — 2026-05-24 — Phase 2 first full rollout
**Task**: put_eggplant_in_basket
**Outcome**: **PASS (pipeline) + 1/5 task success (20%)**
**Rollouts**: 5 total, 1 successful → success_rate = 0.20 (above the published ~10% OpenVLA baseline on this scene class)
**Per-step inference time**: ~247 ms (consistent across all rollouts — first-call warmup amortised over 120 steps)
**Total time**: model load 7.4 s + 5 rollouts in ~2.5 min wall time + 30 s asset/volume sync ≈ 3.5 min
**Cost**: ~$0.30 (A100-40GB @ ~$3.40/hr × 3.5 min ≈ $0.20 + $0.10 cold-start overhead)
**Change from previous attempt**: promoted smoke into `openvla_app.py`, switched to `@modal.cls` with `@modal.enter` model load, attached `openvla-outputs` volume.

### Per-rollout results
| ep | success | steps | wall | mean inf | video |
|---|---|---|---|---|---|
| 0 | False | 120 | 34.8 s | 257 ms | 236 KB |
| 1 | False | 120 | 33.9 s | 247 ms | 277 KB |
| 2 | False | 120 | 33.4 s | 247 ms | 240 KB |
| 3 | False | 120 | 33.1 s | 248 ms | 233 KB |
| 4 | **True** | **59** | **16.1 s** | **244 ms** | **178 KB** |

Rollout 4 terminated early because `info["success"]` flipped to True at step 59 — the gripper actually picked up the eggplant and dropped it in the basket. Videos pulled locally to `openvla_outputs/` and confirmed >100 KB.

### What worked
- Single model load via `@modal.enter` (container stays warm across all 5 rollouts; saves ~10 s per cold start)
- `unnorm_key="bridge_orig"` matched the BridgeV2 norm_stats embedded in the OpenVLA checkpoint
- The supposedly-wrong control mode `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` is actually the BridgeV2-compatible one in this mani_skill version
- No flash-attn, no int8 — vanilla bf16 SDPA was enough on A100-40GB (peak 15.1 GB / 42.4 GB)

### Cumulative cost: ~$0.70 / $5 cap. Stopping here per spec — pipeline goal met.

---

## Final summary
- **All 5 success criteria met.**
- 6 attempts total (5 smoke iterations + 1 successful full demo).
- Pipeline is reproducible: `modal run openvla_app.py --task put_eggplant_in_basket --n-rollouts 5` from a cold cache.
- The 1/5 success rate is genuinely above OpenVLA's published WidowX baseline; treat it as a lucky sample (N=5 is tiny). The deliverable is the *pipeline*, not the success rate.





