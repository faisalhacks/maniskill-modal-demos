# OpenVLA-7B on SimplerEnv WidowX — Modal demo (Agent 4)

Zero-shot OpenVLA-7B running the BridgeV2 digital-twin `PutEggplantInBasketScene-v1` scene on Modal A100-40GB.

## Demo video

https://github.com/faisalhacks/maniskill-modal-demos/raw/main/openvla/rollout_0.mp4

Episode 0 of the eggplant-in-basket rollout. The committed video is one
representative episode; the full 5-rollout outcome was 1/5 success (episode 4
succeeded at step 59). File: [rollout_0.mp4](rollout_0.mp4) (~237 KB).

## What it does

1. Bakes `openvla/openvla-7b` (~16 GB) and the BridgeV2 sim assets into a Modal image so cold starts don't re-download.
2. Loads the 7B model once in bf16 via `@modal.enter` (peak 15.1 GB GPU mem).
3. For each rollout: resets the env, captures the third-person RGB at 224×224, prompts OpenVLA `"In: What action should the robot take to put eggplant into yellow basket?\nOut:"`, decodes a 7-DoF action (`dx dy dz droll dpitch dyaw gripper`) un-normalised with the `bridge_orig` key, steps the env, writes every frame to an MP4.
4. Saves `rollout_{0..N-1}.mp4` to the `openvla-outputs` Modal volume.

## Run

```bash
modal volume create openvla-outputs   # one-time
modal run openvla_app.py --task put_eggplant_in_basket --n-rollouts 5
modal volume get openvla-outputs rollout_0.mp4
```

First run: ~5 min (asset download + 16 GB HF checkpoint cached into image layer).
Subsequent runs: ~10 s cold start + ~35 s per rollout (120 steps × 247 ms inference).

## Results

| ep | success | steps |
|---|---|---|
| 0 | F | 120 |
| 1 | F | 120 |
| 2 | F | 120 |
| 3 | F | 120 |
| 4 | **T** | 59 |

**Pipeline goal met. 1/5 task success (20%) on the eggplant scene.** OpenVLA is published at ~10% on these WidowX scenes — N=5 is too small to draw a strong conclusion, but the pipeline definitely *works*: the arm tracks toward the eggplant, the BridgeV2 norm head produces sane EE deltas, and the gripper closes/opens in response to the image.

## Cost

| stage | cost |
|---|---|
| Smoke iterations (5 attempts) | ~$0.39 |
| Full 5-rollout demo | ~$0.30 |
| **Total** | **~$0.70** (vs $5 cap) |

A100-40GB at ~$3.40/hr. The dominant cost is the image build's HF download (~6 min CPU time), which is free after the first cache hit.

## Key decisions made during iteration

See [BENCHMARK.md](BENCHMARK.md) for the full attempt log. The non-obvious findings:

1. **`obs_mode="rgb"` is rejected** by the Bridge digital-twin scenes — they only accept `"rgb+segmentation"`. The seg channel is required by the in-env success checker.
2. **Control mode in the spec was wrong.** In `mani_skill==3.0.0b22`, the `widowx250s_bridgedataset_sink` agent only exposes `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`. Semantically equivalent to what the spec wanted, just renamed.
3. **Two asset prompts must be baked into the image.** `bridge_v2_real2sim` (scene mesh + textures) AND `widowx250s_bridgedataset_sink` (robot URDF). Both trigger interactive `input("(y|n): ")` on first use, which EOFs in a Modal container.
4. **No flash-attn needed.** Default SDPA + bf16 on A100-40GB runs comfortably with 27 GB of headroom.
5. **Camera sensor key is `"3rd_view_camera"`** (single third-person view, no wrist cam).

## Files

- [openvla_smoke.py](openvla_smoke.py) — Phase 1 smoke test (model load + one inference + env reset).
- [openvla_app.py](openvla_app.py) — Phase 2 full demo (loads model once, runs N rollouts, saves MP4s).
- [BENCHMARK.md](BENCHMARK.md) — attempt-by-attempt log with every failure mode and fix.
- [rollout_0.mp4](rollout_0.mp4) — Episode 0 (committed inline). The remaining four `rollout_{1..4}.mp4` files come from the Modal volume.

## Known limitations

- N=5 is too few rollouts to make any statistical claim about success rate.
- The 224×224 resize from a 640×480 render loses precision; SimplerEnv tasks are tight.
- OpenVLA is autoregressive over 7 action tokens — that's why inference is 247 ms even on A100. No fix; it's the architecture.
- We do NOT verify gripper convention end-to-end; the model trained on BridgeV2 should produce the right sign but a flipped convention would silently fail. The single success at ep 4 is mild evidence the convention is correct.
