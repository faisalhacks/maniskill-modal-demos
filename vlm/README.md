# VLM Pick-Cube Demo on Modal

A Franka Panda in a ManiSkill `PickCube-v1` scene, told to pick up the
red cube. Molmo-7B-D looks at the rendered RGB, points at the cube,
the pixel is deprojected to 3D with the depth buffer + camera
intrinsics, and ManiSkill's motion planner executes the grasp. The
trajectory is saved as an MP4 to a Modal volume and pulled locally.

## Demo video

https://github.com/faisalhacks/maniskill-modal-demos/raw/main/vlm/rollout.mp4

(If the player above doesn't render, the file is at [rollout.mp4](rollout.mp4) — ~161 KB.)

## Files

| File | Purpose |
|------|---------|
| [vlm_app.py](vlm_app.py) | Full demo (image build → Molmo load → ManiSkill rollout → MP4 save) |
| [vlm_smoke.py](vlm_smoke.py) | Phase 1 smoke test: prove the image + GPU class can init SAPIEN |
| [vlm_molmo_test.py](vlm_molmo_test.py) | Phase 2 sanity test: prove Molmo loads and the `<point>` regex parses |
| [BENCHMARK.md](BENCHMARK.md) | Full attempt log (18 attempts, with the critical infra note) |
| [ESCALATION.md](ESCALATION.md) | Mid-run escalation doc (now superseded by the L4 fix in BENCHMARK) |
| [rollout.mp4](rollout.mp4) | Final video (~161 KB) |

## Run it

```bash
# (One-time) authenticate the modal CLI:
modal token new

# Full demo (image build the first time is slow because Molmo weights
# are baked in — ~15 GB. Subsequent runs hit the image cache.)
modal run vlm_app.py --prompt "Pick up the red cube"

# Pull the video locally
modal volume get vlm-outputs rollout.mp4 vlm_outputs/rollout.mp4
```

## ⚠️ Critical infra note (read this FIRST for any future ManiSkill-on-Modal project)

**Use `gpu="L4"`, not `gpu="A10G"`.** Three independent agents have now
reproduced this:

```
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed
```

…from SAPIEN on A10G, regardless of the base image, libvulkan loader
version, ICD JSON contents, or `NVIDIA_DRIVER_CAPABILITIES`. A10G on
Modal is configured compute-only — no graphics device exposed.

Switching the single line `gpu="A10G"` → `gpu="L4"` on the otherwise
identical setup fixes it. See [BENCHMARK.md](BENCHMARK.md) for
the 12-attempt rediscovery.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Modal L4 container                                       │
│   ├─ nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04         │
│   │  + libvulkan1, libegl1, libgl1 (apt)                  │
│   │  + libglib2.0-0, ffmpeg (cv2 + MP4 deps)              │
│   ├─ Python 3.10                                          │
│   │  + mani_skill==3.0.0b19, torch==2.4.0,                │
│   │    transformers==4.45.2, mplib, transforms3d          │
│   └─ /root/.cache/huggingface/  ← Molmo-7B-D-0924         │
│       (baked at image build via run_function snapshot     │
│        so @modal.enter only loads from disk)              │
│                                                            │
│  @app.cls(gpu="L4") class Robot:                          │
│     @modal.enter() load()  ← Molmo from disk              │
│     @modal.method() run_demo(prompt):                     │
│        env = gym.make("PickCube-v1",                      │
│              obs_mode="rgb+depth+segmentation",           │
│              control_mode="pd_joint_pos",                 │
│              render_mode="rgb_array")                     │
│        rgb, depth = obs["sensor_data"][cam]               │
│        (u, v) = molmo.point_at(rgb, "red cube")           │
│        target_xyz = deproject(u, v, depth,                │
│                               intrinsic, extrinsic)       │
│        planner = PandaArmMotionPlanningSolver(env)        │
│        planner.{open_gripper, move(approach), move(grasp),│
│                  close_gripper, move(lift)}               │
│        # env.step is monkey-patched to record frames      │
│        imageio.mimsave("/outputs/rollout.mp4", frames)    │
│        volume.commit()                                     │
└──────────────────────────────────────────────────────────┘
```

## Key implementation details (gotchas you may hit)

1. **GPU class**: Use `gpu="L4"`. A10G fails (see above).
2. **`transformers==4.45.2` pinned**: newer transformers imports
   `torch.distributed.tensor.device_mesh`, which requires `torch>=2.5`,
   but ManiSkill 3.0.0b19 needs `torch==2.4.0`. 4.45.2 is Molmo's
   tested version per the model card.
3. **`generate_from_batch` API**: pass a `GenerationConfig` object as
   the second arg, NOT `eos_token_id`. Molmo's signature is
   `(batch, generation_config, tokenizer=...)`.
4. **`control_mode="pd_joint_pos"`**: `PandaArmMotionPlanningSolver`
   sends 15-D joint commands. If you set `pd_ee_delta_pose` (7-D),
   `planner.open_gripper()` will fail an action-shape assertion.
5. **Monkey-patch `env.step` to capture motion frames**: the motion
   planner steps the env internally; without a hook your video will
   be ~21 frames (75 KB). With the hook it's ~150 frames (~161 KB).
6. **Molmo's `<point>` format**: matches the regex
   `r'x="([\d.]+)"\s+y="([\d.]+)"'`. Coordinates are image-percent
   (0-100), not pixels.
7. **Bake Molmo weights at image build**: a `run_function` that calls
   `huggingface_hub.snapshot_download(MODEL_ID)` saves ~3 min of cold
   start time per call (and keeps the @modal.enter to a disk load).
8. **`obs_mode="rgb+depth+segmentation"`**: depth is needed for the
   pixel→3D deprojection. ManiSkill 3 returns batched tensors; call
   `.squeeze(0).cpu().numpy()` before passing to PIL.
9. **Don't trust just-deprojected XY from a single Molmo point.**
   Molmo's `<point>` typically lands 1-2 cm off the object center,
   which is enough to make a Panda parallel-jaw graze a 4 cm cube's
   edge. Snap (u, v) to the centroid of the cube's pixel mask in
   `obs["sensor_data"][cam]["segmentation"]` (sample the seg ID at
   the VLM point, then centroid all pixels with that ID). Depth-
   tolerance windows DO NOT work as a substitute — same-camera-depth
   != same-world-Z under a tilted camera, so you can end up
   centroiding the table.
10. **Post-grasp idle frames must NOT use `np.zeros(action_space.shape)`
    under `pd_joint_pos`.** That action commands every joint to position
    0, which opens the gripper and dumps the cube. Re-issue the lift
    pose via `planner.move_to_pose_with_screw(lift)` to hold instead.
11. **Verify the task succeeded with a programmatic check, not just
    exit code + file size.** A passing pipeline can produce a valid
    >100 KB MP4 that shows the gripper closing on empty air. Add
    `final_cube_z = env.unwrapped.cube.pose.p[2]; assert final_cube_z > 0.10`
    (or use the env's `is_success` flag) before claiming success.

## Cost

~$0.95 total Modal spend across 18 attempts during development.
The fully cached path (image already built, weights baked) is
roughly one $0.05-$0.08 GPU-minute per `modal run`.

## Verifying the final result

```bash
# Size sanity check (must be > 100 KB)
ls -la rollout.mp4

# Play it
ffplay rollout.mp4   # or vlc/mpv/QuickTime
```

The video opens on the resting scene, shows the arm approaching the
red cube, closing the gripper, and lifting. The motion planner's
internal env.step calls are captured via the hook in `run_demo`.
