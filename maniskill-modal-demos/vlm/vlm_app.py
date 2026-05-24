"""ManiSkill + Molmo VLM demo on Modal — Phase 2.

A Franka Panda in a ManiSkill PickCube scene; Molmo-7B-D looks at the
rendered RGB, points at the cube, the pixel is deprojected to 3D with
the depth buffer + camera intrinsics, and ManiSkill's motion planner
picks it up. Frames are stitched into a video saved on a Modal volume.

Run:
    modal run vlm_app.py --prompt "Pick up the red cube"
    modal volume get vlm-outputs rollout.mp4 vlm_outputs/rollout.mp4

Infrastructure notes (see vlm_BENCHMARK.md "Critical Infrastructure Note"):
  - gpu="L4". A10G does NOT expose Vulkan rendering on Modal.
  - Image is Ubuntu 22.04 (libvulkan 1.3 native), NOT debian_slim and
    NOT maniskill/base.
  - Molmo weights are baked into the image layer at build time; the
    @modal.enter just loads from disk.
"""

import modal

MODEL_ID = "allenai/Molmo-7B-D-0924"


def _bake_molmo():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "libvulkan1", "libvulkan-dev", "vulkan-tools",
        "libegl1", "libgles2", "libglvnd0", "libgl1",
        "libxext6", "libx11-6",
        "libglib2.0-0", "libsm6", "libxrender1",
        "ffmpeg",
        "wget", "ca-certificates",
    )
    .pip_install(
        # ManiSkill stack (Attempt-12-validated)
        "mani_skill==3.0.0b19",
        "torch==2.4.0",
        "torchvision",
        "gymnasium",
        # VLM. transformers pinned to 4.45.2: newer transformers imports
        # torch.distributed.tensor.device_mesh which requires torch>=2.5,
        # but we need torch==2.4.0 for ManiSkill compat. 4.45.2 is the
        # version Molmo's model card lists as tested.
        "transformers==4.45.2",
        "tokenizers>=0.20",
        "einops",
        "accelerate",
        "pillow",
        "huggingface_hub",
        # Demo plumbing
        "imageio[ffmpeg]",
        "mplib",
        "transforms3d",
    )
    .env({
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "NVIDIA_VISIBLE_DEVICES": "all",
    })
    .run_function(_bake_molmo)
)

app = modal.App("vlm-demo", image=image)
volume = modal.Volume.from_name("vlm-outputs", create_if_missing=True)


@app.cls(
    gpu="L4",
    volumes={"/outputs": volume},
    timeout=60 * 20,
    scaledown_window=300,
)
class Robot:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        print("Loading Molmo...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True,
            torch_dtype="auto", device_map="auto",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True,
            torch_dtype="auto", device_map="auto",
        )
        self.torch = torch
        self.GenerationConfig = GenerationConfig
        print("Molmo loaded.")

    def point_at(self, pil_image, object_description: str):
        """Ask Molmo to point at an object. Returns (x_px, y_px) or None."""
        import re

        prompt = f"Point to the {object_description}."
        inputs = self.processor.process(images=[pil_image], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0)
                  for k, v in inputs.items()}

        # Molmo's generate_from_batch wants a GenerationConfig object,
        # not eos_token_id. Per the model card.
        gen_cfg = self.GenerationConfig(
            max_new_tokens=200, stop_strings="<|endoftext|>"
        )
        with self.torch.no_grad():
            output = self.model.generate_from_batch(
                inputs, gen_cfg, tokenizer=self.processor.tokenizer
            )

        text = self.processor.tokenizer.decode(
            output[0, inputs["input_ids"].size(1):], skip_special_tokens=True)
        print(f"Molmo said: {text}")

        m = re.search(r'x="([\d.]+)"\s+y="([\d.]+)"', text)
        if not m:
            return None
        x_pct, y_pct = float(m.group(1)), float(m.group(2))
        w, h = pil_image.size
        return int(x_pct / 100 * w), int(y_pct / 100 * h)

    @modal.method()
    def run_demo(self, prompt: str = "Pick up the red cube"):
        import os
        import numpy as np
        import gymnasium as gym
        import imageio
        from PIL import Image
        import mani_skill.envs  # noqa: F401  (registers envs)

        os.makedirs("/outputs", exist_ok=True)

        # ---- 1. Build env ----
        # control_mode="pd_joint_pos" is what the canonical
        # PandaArmMotionPlanningSolver in mani_skill examples expects.
        # Using pd_ee_delta_pose makes planner.open_gripper() send
        # an action of the wrong arity (assertion failure inside SB3
        # controller).
        env = gym.make(
            "PickCube-v1",
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sensor_configs=dict(width=640, height=480),
        )
        obs, _ = env.reset(seed=0)

        # Hook env.step so the motion planner's internal stepping shows
        # up in the saved video. Without this we only get 1 initial frame
        # + 20 idle post-lift frames (~75 KB MP4); with the hook we get
        # ~150 frames covering the whole approach/grasp/lift motion.
        frames = []
        underlying_step = env.step

        def step_with_capture(action):
            result = underlying_step(action)
            try:
                frames.append(env.render().squeeze(0).cpu().numpy())
            except Exception:
                pass
            return result

        env.step = step_with_capture
        # And one frame at t=0 before the planner starts so the video
        # opens on the resting scene.
        frames.append(env.render().squeeze(0).cpu().numpy())

        # ---- 2. Grab base camera RGB + depth ----
        cam_name = list(obs["sensor_data"].keys())[0]
        rgb = obs["sensor_data"][cam_name]["rgb"].squeeze(0).cpu().numpy()
        depth = obs["sensor_data"][cam_name]["depth"].squeeze(0).cpu().numpy().squeeze()
        pil = Image.fromarray(rgb)

        # ---- 3. Molmo points at the cube ----
        target_desc = prompt.lower().replace("pick up the", "").strip() or "red cube"
        pt = self.point_at(pil, target_desc)
        if pt is None:
            print("VLM did not return a point; aborting.")
            return None
        u, v = pt
        print(f"VLM pointed at pixel ({u}, {v})")

        # ---- 4. Deproject pixel -> world 3D ----
        cam_params = obs["sensor_param"][cam_name]
        intrinsic = cam_params["intrinsic_cv"].squeeze(0).cpu().numpy()  # 3x3
        extrinsic = cam_params["extrinsic_cv"].squeeze(0).cpu().numpy()  # 3x4

        # ManiSkill depth: float32 = metres, uint16 = millimetres.
        z_at_vlm = float(depth[v, u])
        if depth.dtype != np.float32:
            z_at_vlm = z_at_vlm / 1000.0
        print(f"Raw VLM depth: {z_at_vlm:.4f} m at pixel ({u}, {v})")

        # Snap (u, v) from "Molmo's chosen point on the cube top" to
        # "centroid of the cube's pixel mask", to eliminate the 1-2 cm
        # XY error of Molmo's single <point>. Same-depth-windowing was
        # tried and failed because the table behind the cube has the
        # same camera depth as the cube top (camera tilt makes same-z
        # !⇒ same-world-position). Using ManiSkill's per-pixel
        # segmentation map, which is ground truth.
        seg = obs["sensor_data"][cam_name]["segmentation"]
        seg = seg.squeeze(0).squeeze(-1).cpu().numpy()
        # Sample the segmentation id at the VLM point — that's the cube's
        # id by definition (Molmo pointed AT the cube).
        cube_id = int(seg[v, u])
        cube_mask = seg == cube_id
        n_cube_px = int(cube_mask.sum())
        if n_cube_px >= 25:
            ys, xs = np.where(cube_mask)
            u_new = int(round(xs.mean()))
            v_new = int(round(ys.mean()))
            z_new = float(depth[v_new, u_new])
            if depth.dtype != np.float32:
                z_new = z_new / 1000.0
            print(f"Segmentation snap: ({u}, {v}) z={z_at_vlm:.4f}  ->  "
                  f"({u_new}, {v_new}) z={z_new:.4f}  "
                  f"[cube id={cube_id}, {n_cube_px} px]")
            u, v, z = u_new, v_new, z_new
        else:
            print(f"Segmentation snap: only {n_cube_px} px for id "
                  f"{cube_id}; falling back to raw VLM point.")
            z = z_at_vlm

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        p_cam = np.array([x_cam, y_cam, z, 1.0])

        # extrinsic is world->cam; invert to cam->world
        R, t = extrinsic[:3, :3], extrinsic[:3, 3]
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = R.T
        cam_to_world[:3, 3] = -R.T @ t
        p_world = cam_to_world @ p_cam
        target_xyz = p_world[:3]
        print(f"Target world position: {target_xyz}")

        # Diagnostic: compare VLM point to ground truth
        true_cube_pos = env.unwrapped.cube.pose.p
        if hasattr(true_cube_pos, "cpu"):
            true_cube_pos = true_cube_pos.cpu().numpy().squeeze()
        print("=== GRASP DIAGNOSTIC ===")
        print(f"VLM pointed at pixel: ({u}, {v})")
        print(f"Depth at that pixel:  {z:.4f} m")
        print(f"Deprojected world XYZ: {target_xyz}")
        print(f"True cube position:    {true_cube_pos}")
        print(f"Error (m):             {target_xyz - true_cube_pos}")
        print("========================")

        # Isolation test (now removed): we previously overrode target_xyz
        # with the ground-truth cube position. That run showed the grasp
        # succeeded and the cube was lifted to z=0.218; the failure was
        # the 20 post-lift idle frames using a zero action, which under
        # pd_joint_pos commanded the gripper to position 0 (open),
        # releasing the cube. The VLM-derived target_xyz is good enough.

        # ---- 5. Motion planner ----
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )
        from transforms3d.quaternions import axangle2quat
        import sapien

        planner = PandaArmMotionPlanningSolver(
            env, debug=False, vis=False,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )

        grasp_quat = axangle2quat([1, 0, 0], np.pi)  # gripper -Z
        # target_xyz is on the cube's TOP face. Empirically:
        #   +0.02 z: even with XY aligned (8 mm off), the cube wasn't
        #            touched — the gripper closed above it. So Panda's
        #            TCP-to-fingertip offset on PandaArmMotionPlanningSolver
        #            is smaller than the ~4 cm I'd assumed; closer to 2 cm.
        #   0.00 z (this attempt): TCP at the cube top, fingertips ~2 cm
        #            below ≈ cube centre. Should grasp now that XY is
        #            correct (no longer at risk of grazing the edge).
        approach = sapien.Pose(p=target_xyz + np.array([0, 0, 0.10]),
                               q=grasp_quat)
        grasp = sapien.Pose(p=target_xyz + np.array([0, 0, 0.0]),
                            q=grasp_quat)
        lift = sapien.Pose(p=target_xyz + np.array([0, 0, 0.20]),
                           q=grasp_quat)

        def cube_z():
            p = env.unwrapped.cube.pose.p
            if hasattr(p, "cpu"):
                p = p.cpu().numpy().squeeze()
            return float(p[2])

        print(f"[trace] before plan, cube_z={cube_z():.4f}")
        planner.open_gripper()
        print(f"[trace] after open_gripper, cube_z={cube_z():.4f}")
        planner.move_to_pose_with_screw(approach)
        print(f"[trace] after approach (z={approach.p[2]:.3f}), "
              f"cube_z={cube_z():.4f}")
        planner.move_to_pose_with_screw(grasp)
        print(f"[trace] after grasp move (z={grasp.p[2]:.3f}), "
              f"cube_z={cube_z():.4f}")
        planner.close_gripper()
        print(f"[trace] after close_gripper, cube_z={cube_z():.4f}")
        planner.move_to_pose_with_screw(lift)
        print(f"[trace] after lift (z={lift.p[2]:.3f}), cube_z={cube_z():.4f}")

        # Hold the lifted pose for some idle frames. CANNOT use
        # `np.zeros(action_space.shape)` here — with pd_joint_pos
        # control the action IS the commanded joint position, and zero
        # would slam the arm to the home pose AND open the gripper,
        # dropping the cube. Re-issuing the lift move via the planner
        # keeps the arm at the lift pose with the gripper held closed.
        for _ in range(2):
            planner.move_to_pose_with_screw(lift)

        # End-of-rollout sanity check: where is the cube?
        # If grasp succeeded, the cube was lifted to ~target_xyz[2]+0.20.
        # If the gripper closed on empty air, the cube stayed on the table.
        final_cube_pos = env.unwrapped.cube.pose.p
        if hasattr(final_cube_pos, "cpu"):
            final_cube_pos = final_cube_pos.cpu().numpy().squeeze()
        grasp_ok = final_cube_pos[2] > 0.10  # >10 cm above table = lifted
        print("=== END-OF-ROLLOUT CHECK ===")
        print(f"Final cube position: {final_cube_pos}")
        print(f"Cube lifted (z > 0.10): {grasp_ok}")
        print("============================")

        # ---- 6. Save video ----
        out_path = "/outputs/rollout.mp4"
        imageio.mimsave(out_path, frames, fps=20)
        volume.commit()
        print(f"Saved video to {out_path}")
        return out_path


@app.local_entrypoint()
def main(prompt: str = "Pick up the red cube"):
    robot = Robot()
    result = robot.run_demo.remote(prompt)
    print(f"Done. Output: {result}")
    print("Download with:  modal volume get vlm-outputs rollout.mp4 .")
