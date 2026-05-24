"""Phase 2 OpenVLA demo (Agent 4).

Zero-shot OpenVLA-7B on SimplerEnv `PutEggplantInBasketScene-v1` (BridgeV2
digital-twin). Loads the 7B model once via @modal.enter, then runs N rollouts
of up to max_steps each, saving each rollout as an MP4 to the openvla-outputs
volume.

Pipeline goal is the deliverable: OpenVLA on SimplerEnv WidowX is published
at ~10% task success, so success-rate-of-zero is acceptable here. What we
need is the arm responding to images, video saved, no crashes.

Run: modal run openvla_app.py --task put_eggplant_in_basket --n-rollouts 5
Pull: modal volume get openvla-outputs rollout_0.mp4
"""

import modal


def _download_openvla():
    from huggingface_hub import snapshot_download
    snapshot_download("openvla/openvla-7b", local_dir="/root/openvla-7b")


image = (
    modal.Image.from_registry("maniskill/base", add_python="3.10")
    .pip_install(
        "mani_skill==3.0.0b22",
        "torch==2.4.0",
        "torchvision",
        "transformers==4.40.1",
        "tokenizers==0.19.1",
        "timm==0.9.10",
        "accelerate==0.32.1",
        "huggingface_hub",
        "Pillow",
        "imageio[ffmpeg]",
    )
    # Cache both Bridge digital-twin assets in image layers so cold starts
    # don't re-download or hit the interactive y/n prompt. See smoke attempts
    # 1–4 for the failure modes these guard against.
    .run_commands(
        "python -m mani_skill.utils.download_asset bridge_v2_real2sim -y",
        "python -m mani_skill.utils.download_asset widowx250s_bridgedataset_sink -y || true",
    )
    .run_function(_download_openvla, timeout=60 * 30)
)

app = modal.App("openvla-demo", image=image)
volume = modal.Volume.from_name("openvla-outputs", create_if_missing=True)

TASKS = {
    "put_eggplant_in_basket": (
        "PutEggplantInBasketScene-v1",
        "put eggplant into yellow basket",
    ),
    "put_spoon_on_towel": (
        "PutSpoonOnTableInScene-v1",
        "put the spoon on the towel",
    ),
}

CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


@app.cls(
    gpu="A100-40GB",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 30,
    scaledown_window=300,
)
class OpenVLARunner:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # Belt-and-suspenders: same monkey-patch the smoke test used in case
        # any asset slipped past the image build.
        from mani_skill.utils import download_asset
        download_asset.prompt_yes_no = lambda *a, **kw: True

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            "/root/openvla-7b", trust_remote_code=True
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            "/root/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda")
        self.vla.eval()
        print(
            f"Loaded OpenVLA-7B; GPU mem: "
            f"{torch.cuda.memory_allocated()/1e9:.1f} GB"
        )

    def _grab_rgb(self, obs):
        """Extract uint8 (H, W, 3) RGB from the single SimplerEnv camera."""
        import numpy as np
        cam = obs["sensor_data"]["3rd_view_camera"]["rgb"]
        if hasattr(cam, "cpu"):
            cam = cam.cpu().numpy()
        cam = np.asarray(cam).squeeze()
        # Mani_skill returns (H, W, 3) for vectorised num_envs=1; double-check
        if cam.ndim == 3 and cam.shape[-1] != 3 and cam.shape[0] == 3:
            cam = cam.transpose(1, 2, 0)
        return cam.astype(np.uint8)

    @modal.method()
    def rollout(
        self,
        task: str = "put_eggplant_in_basket",
        n_rollouts: int = 5,
        max_steps: int = 120,
    ):
        import os
        import time
        import numpy as np
        import gymnasium as gym
        import imageio
        from PIL import Image as PILImage
        import mani_skill.envs  # noqa: F401

        env_id, instruction = TASKS[task]
        os.makedirs("/outputs", exist_ok=True)
        prompt_tmpl = "In: What action should the robot take to {}?\nOut:"

        per_rollout = []
        successes = 0

        for ep in range(n_rollouts):
            env = gym.make(
                env_id,
                obs_mode="rgb+segmentation",
                control_mode=CONTROL_MODE,
                render_mode="rgb_array",
            )
            obs, _ = env.reset(seed=42 + ep)
            frames = []
            success = False
            step = 0
            t0 = time.perf_counter()
            inf_times = []

            for step in range(max_steps):
                rgb = self._grab_rgb(obs)
                frames.append(rgb)

                pil = PILImage.fromarray(rgb).resize((224, 224))
                prompt = prompt_tmpl.format(instruction)
                inputs = self.processor(prompt, pil).to(
                    "cuda", dtype=self.torch.bfloat16
                )
                ti = time.perf_counter()
                with self.torch.no_grad():
                    action = self.vla.predict_action(
                        **inputs, unnorm_key="bridge_orig", do_sample=False
                    )
                inf_times.append(time.perf_counter() - ti)
                action = np.asarray(action, dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)

                sval = info.get("success", False) if hasattr(info, "get") else False
                if hasattr(sval, "item"):
                    try:
                        sval = bool(sval.item())
                    except Exception:
                        sval = bool(np.asarray(sval).any())
                elif hasattr(sval, "any"):
                    sval = bool(np.asarray(sval).any())
                if sval:
                    success = True

                done = False
                for x in (terminated, truncated):
                    if hasattr(x, "any"):
                        if bool(np.asarray(x).any()):
                            done = True
                    elif bool(x):
                        done = True
                if done or success:
                    break

            # Final frame
            try:
                frames.append(self._grab_rgb(obs))
            except Exception:
                pass

            elapsed = time.perf_counter() - t0
            mean_inf_ms = (sum(inf_times) / len(inf_times) * 1000) if inf_times else 0.0

            video_path = f"/outputs/rollout_{ep}.mp4"
            imageio.mimsave(video_path, frames, fps=5)
            size_kb = os.path.getsize(video_path) / 1024

            per_rollout.append(
                {
                    "ep": ep,
                    "success": bool(success),
                    "steps": step + 1,
                    "elapsed_s": round(elapsed, 1),
                    "mean_inf_ms": round(mean_inf_ms, 1),
                    "video_kb": round(size_kb, 1),
                }
            )
            successes += int(success)
            env.close()
            print(
                f"Rollout {ep}: success={success}, steps={step+1}, "
                f"elapsed={elapsed:.1f}s, inf={mean_inf_ms:.0f}ms, "
                f"video={size_kb:.0f}KB"
            )

        volume.commit()
        summary = {
            "task": task,
            "n_rollouts": n_rollouts,
            "successes": successes,
            "success_rate": successes / n_rollouts,
            "per_rollout": per_rollout,
        }
        print("SUMMARY:", summary)
        return summary


@app.local_entrypoint()
def main(task: str = "put_eggplant_in_basket", n_rollouts: int = 5):
    print(OpenVLARunner().rollout.remote(task=task, n_rollouts=n_rollouts))
    print("Pull videos: modal volume get openvla-outputs rollout_0.mp4")
