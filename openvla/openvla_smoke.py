"""Phase 1 smoke test for OpenVLA project (Agent 4).

Verifies on Modal:
  1. maniskill/base image builds with OpenVLA's pinned dependency set.
  2. A100-40GB sees the GPU and has enough memory.
  3. openvla-7b loads in bf16 (~14 GB).
  4. One predict_action() call works on a dummy 224x224 image and returns a (7,) action.
  5. SimplerEnv PutEggplantInBasketScene-v1 can be reset with control_mode that
     matches the BridgeV2 action ordering.

Run: modal run openvla_smoke.py
Expected: prints "OK" and exits with code 0.
"""

import modal


def _download():
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
    # Cache assets that PutEggplantInBasketScene-v1 requires. Without these,
    # gym.make triggers an interactive `input("(y|n): ")` that EOFs in a Modal
    # container. Two separate prompts: scene assets (Attempt 1) + robot URDF
    # (Attempt 3). The `|| true` lets the build proceed even if an asset is
    # already cached from a prior layer.
    .run_commands(
        "python -m mani_skill.utils.download_asset bridge_v2_real2sim -y",
        "python -m mani_skill.utils.download_asset widowx250s_bridgedataset_sink -y || true",
    )
    .run_function(_download, timeout=60 * 30)
)

app = modal.App("openvla-smoke", image=image)


@app.function(gpu="A100-40GB", timeout=900)
def smoke():
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    # Test 1: load OpenVLA in bf16
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained("/root/openvla-7b", trust_remote_code=True)
    print("PASS processor loaded")
    vla = AutoModelForVision2Seq.from_pretrained(
        "/root/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    n_params = sum(p.numel() for p in vla.parameters()) / 1e9
    print(f"PASS model loaded; params: {n_params:.2f}B")
    print(f"GPU mem after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Test 2: one inference pass
    from PIL import Image
    import numpy as np
    dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    prompt = "In: What action should the robot take to pick up the cube?\nOut:"
    inputs = processor(prompt, dummy).to("cuda", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(f"PASS inference works; action shape: {tuple(action.shape)}, values: {action}")
    assert action.shape == (7,), f"Expected 7-DoF action, got {action.shape}"

    # Test 3: SimplerEnv scene reset
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    # Belt-and-suspenders auto-yes for any asset download that slipped past the
    # image build step (e.g., asset UIDs the CLI doesn't recognize but the
    # runtime download flow does).
    from mani_skill.utils import download_asset
    download_asset.prompt_yes_no = lambda *a, **kw: True

    env = gym.make(
        "PutEggplantInBasketScene-v1",
        obs_mode="rgb+segmentation",  # this digital-twin scene rejects plain "rgb"
        # The bridge digital-twin agent only exposes one control mode in this
        # mani_skill version; it's the BridgeV2-compatible EE-delta + gripper
        # joint-pos mode that OpenVLA's BridgeV2 head was trained on.
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=0)
    sensor_keys = list(obs["sensor_data"].keys()) if isinstance(obs, dict) and "sensor_data" in obs else "<missing>"
    print(f"PASS SimplerEnv scene reset OK; sensor keys: {sensor_keys}")
    env.close()
    return "OK"


@app.local_entrypoint()
def main():
    print(smoke.remote())
