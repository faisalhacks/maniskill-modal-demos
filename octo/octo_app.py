"""Phase 2: zero-shot Octo-Small 1.5 rollouts on a SimplerEnv WidowX scene.

This file is the production version of octo_smoke.py. Image spec is identical
(every pin/patch was validated by smoke attempts 1-16 in octo_BENCHMARK.md).

Run:
  modal run octo_app.py --task put_eggplant_in_basket --n-rollouts 5
  modal volume get octo-outputs rollout_0.mp4
"""
import modal

image = (
    modal.Image.from_registry("maniskill/base", add_python="3.10")
    .pip_install(
        # maniskill/base ships Python 3.9 (conda env wins over add_python),
        # which caps JAX at 0.4.30. jax-cuda12-plugin 0.4.30 needs cudnn>=9 and
        # so does torch 2.4.0 — these are the only compatible pair.
        "mani_skill==3.0.0b22",
        "torch==2.4.0",
        "jax[cuda12]==0.4.30",
        "flax==0.8.5",
        "distrax==0.1.5",
        "einops",
        "tensorflow-cpu==2.15.0",
        # tensorflow-probability auto-selects 0.25 (needs TF>=2.18) without a
        # pin; 0.23.0 is the matching TFP for TF 2.15.
        "tensorflow-probability==0.23.0",
        "tensorflow_hub",
        "dlimp @ git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972",
        # transformers/sentencepiece are needed by Octo's Flax T5 language head
        # (we install Octo with --no-deps below, so they aren't pulled
        # automatically).
        "transformers==4.40.2",
        "sentencepiece",
        "imageio[ffmpeg]",
    )
    .run_commands(
        "pip install --no-deps git+https://github.com/octo-models/octo.git",
        # Octo references jax.random.KeyArray (removed in JAX 0.4.30) in a few
        # places. Patch + clear stale .pyc so Python actually reads the patched
        # source.
        "find /opt/conda/lib/python3.9/site-packages/octo -name '*.py' "
        "-exec sed -i 's|jax\\.random\\.KeyArray|jax.Array|g' {} +",
        "find /opt/conda/lib/python3.9/site-packages/octo -name '__pycache__' "
        "-type d -exec rm -rf {} +",
        # Pre-download the two task scenes so first gym.make() doesn't prompt.
        # The robot URDF is downloaded at runtime via the prompt_yes_no
        # monkeypatch below (we don't know the right CLI uid offhand).
        "yes | python -m mani_skill.utils.download_asset PutEggplantInBasketScene-v1 || true",
        "yes | python -m mani_skill.utils.download_asset PutSpoonOnTableClothScene-v1 || true",
    )
    .env({"TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_CPP_MIN_LOG_LEVEL": "3"})
)

app = modal.App("octo-demo", image=image)
volume = modal.Volume.from_name("octo-outputs", create_if_missing=True)


# task -> (env_id, language instruction, Octo dataset_statistics key)
# WidowX scenes use bridge_dataset stats (NOT fractal — that's Google Robot).
TASKS = {
    "put_eggplant_in_basket": (
        "PutEggplantInBasketScene-v1",
        "put the eggplant in the basket",
        "bridge_dataset",
    ),
    "put_spoon_on_towel": (
        "PutSpoonOnTableClothScene-v1",
        "put the spoon on the towel",
        "bridge_dataset",
    ),
}

# The only control_mode supported by these bridge_dataset_eval scenes in
# mani_skill==3.0.0b22 — the long string in the instructions doc does not exist.
WIDOWX_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


@app.cls(
    gpu="L4",  # A10G ships a stale libGLX_nvidia.so.0 that breaks SAPIEN's Vulkan.
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 30,
    scaledown_window=300,
)
class OctoRunner:
    @modal.enter()
    def load(self):
        import jax
        # Auto-accept any first-run ManiSkill asset prompts (robot URDF etc).
        from mani_skill.utils import download_asset as _da
        _da.prompt_yes_no = lambda *a, **k: True
        from octo.model.octo_model import OctoModel
        print(f"JAX devices: {jax.devices()}")
        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
        self.jax = jax
        print("Octo loaded.")

    @modal.method()
    def rollout(self, task: str = "put_eggplant_in_basket",
                n_rollouts: int = 5, max_steps: int = 120):
        import os
        from collections import deque
        import numpy as np
        import gymnasium as gym
        import imageio
        from PIL import Image as PILImage
        import mani_skill.envs  # noqa: F401  (registers envs)

        env_id, instruction, stats_key = TASKS[task]
        os.makedirs("/outputs", exist_ok=True)

        # Cache the language embedding once — instruction doesn't change between
        # rollouts, and create_tasks runs T5 (slow).
        task_emb = self.model.create_tasks(texts=[instruction])
        unnorm_stats = self.model.dataset_statistics[stats_key]["action"]

        per_rollout = []
        successes = 0

        for ep in range(n_rollouts):
            env = gym.make(
                env_id,
                obs_mode="rgb+segmentation",
                control_mode=WIDOWX_CONTROL_MODE,
                render_mode="rgb_array",
            )
            obs, info = env.reset(seed=42 + ep)
            cam_key = next(iter(obs["sensor_data"].keys()))
            frames = []
            obs_window = deque(maxlen=2)
            success = False
            step = 0

            def to_hwc(rgb_t):
                # ManiSkill 3 returns sensor RGB as torch.Tensor (1,H,W,3) uint8.
                arr = rgb_t.cpu().numpy() if hasattr(rgb_t, "cpu") else np.asarray(rgb_t)
                return np.squeeze(arr, axis=0).astype(np.uint8) if arr.ndim == 4 else arr.astype(np.uint8)

            # Capture the initial frame.
            frames.append(to_hwc(obs["sensor_data"][cam_key]["rgb"]))

            for step in range(max_steps):
                rgb = to_hwc(obs["sensor_data"][cam_key]["rgb"])
                # Octo-Small 1.5 expects 256x256 RGB.
                rgb_resized = np.array(PILImage.fromarray(rgb).resize((256, 256)))
                obs_window.append(rgb_resized)
                # On the very first step pad the window by duplicating the
                # single observation — Octo needs T=2.
                while len(obs_window) < 2:
                    obs_window.appendleft(obs_window[0])

                octo_obs = {
                    "image_primary": np.stack(list(obs_window))[None],  # (1,2,256,256,3)
                    "timestep_pad_mask": np.array([[True, True]]),
                }

                actions = self.model.sample_actions(
                    octo_obs,
                    task_emb,
                    unnormalization_statistics=unnorm_stats,
                    rng=self.jax.random.PRNGKey(step),
                )
                # actions: (1, 4, 7) — chunk of 4 future actions, 7-DoF.
                actions = np.asarray(actions[0])

                # Execute the chunk open-loop (recommended by Octo). Record a
                # frame after every sim step so the video has real motion (≈4x
                # more frames than recording per-chunk).
                for a in actions:
                    obs, reward, terminated, truncated, info = env.step(a)
                    frames.append(to_hwc(obs["sensor_data"][cam_key]["rgb"]))
                    succ = info.get("success", False) if hasattr(info, "get") else False
                    if hasattr(succ, "item"):
                        succ = bool(succ.item())
                    success = success or bool(succ)
                    if terminated or truncated or success:
                        break
                # Convert torch booleans coming back from ManiSkill 3.
                if hasattr(terminated, "item"):
                    terminated = bool(terminated.item())
                if hasattr(truncated, "item"):
                    truncated = bool(truncated.item())
                if success or terminated or truncated:
                    break

            video_path = f"/outputs/rollout_{ep}.mp4"
            imageio.mimsave(video_path, frames, fps=15)
            size = os.path.getsize(video_path)
            per_rollout.append({"ep": ep, "success": bool(success),
                                "steps": step + 1, "video_bytes": size})
            successes += int(success)
            env.close()
            print(f"Rollout {ep}: success={success}, steps={step + 1}, "
                  f"video={size/1024:.1f} KB")

        volume.commit()
        summary = {
            "task": task,
            "n_rollouts": n_rollouts,
            "successes": successes,
            "success_rate": successes / n_rollouts,
            "per_rollout": per_rollout,
        }
        print(summary)
        return summary


@app.local_entrypoint()
def main(task: str = "put_eggplant_in_basket", n_rollouts: int = 5):
    print(OctoRunner().rollout.remote(task=task, n_rollouts=n_rollouts))
    print("Pull videos: modal volume get octo-outputs rollout_0.mp4")
