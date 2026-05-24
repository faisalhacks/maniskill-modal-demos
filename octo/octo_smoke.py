"""Phase 1 smoke test for Octo on Modal.

Verifies three things end-to-end:
  1. JAX with CUDA 12 sees the GPU.
  2. Octo-Small 1.5 loads from HuggingFace.
  3. SimplerEnv WidowX scene (ManiSkill 3) instantiates with Vulkan rendering.

Run: modal run octo_smoke.py
"""
import modal

image = (
    modal.Image.from_registry("maniskill/base", add_python="3.10")
    .pip_install(
        "mani_skill==3.0.0b22",
        # Python 3.9 (from maniskill/base conda env) caps JAX at 0.4.30.
        # jax-cuda12-plugin 0.4.30 pulls cudnn>=9.0,<10.0 (NOT cudnn 8 like
        # 0.4.28); torch 2.4.0 pulls cudnn 9.1.0.70. Both agree on cudnn 9.
        # See attempts 1–3 in octo_BENCHMARK.md.
        "torch==2.4.0",
        "jax[cuda12]==0.4.30",
        "flax==0.8.5",
        "distrax==0.1.5",
        "einops",
        "tensorflow-cpu==2.15.0",
        # distrax 0.1.5 imports tensorflow_probability, which auto-selects 0.25
        # (requires TF≥2.18) without an explicit pin. 0.23.0 is the matching
        # TFP release for TF 2.15.
        "tensorflow-probability==0.23.0",
        "tensorflow_hub",
        "dlimp @ git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972",
        # Octo's tokenizers.py uses transformers' FlaxT5EncoderModel for the
        # language head. We installed Octo with --no-deps so it isn't pulled
        # automatically. T5's tokenizer also needs sentencepiece.
        "transformers==4.40.2",
        "sentencepiece",
    )
    .run_commands(
        "pip install --no-deps git+https://github.com/octo-models/octo.git",
        # Octo references jax.random.KeyArray (removed in JAX 0.4.30) in a few
        # places. Patch every occurrence to jax.Array (current PRNGKey type),
        # then clear pip's pre-compiled bytecode so the patched .py files are
        # actually read on import.
        "find /opt/conda/lib/python3.9/site-packages/octo -name '*.py' "
        "-exec sed -i 's|jax\\.random\\.KeyArray|jax.Array|g' {} +",
        "find /opt/conda/lib/python3.9/site-packages/octo -name '__pycache__' "
        "-type d -exec rm -rf {} +",
        # ManiSkill 3 prompts via input() to download scene assets the first
        # time an env is built; that fails with EOFError under Modal's
        # non-interactive shell. Pre-download both task scenes here with `yes`
        # piped in to auto-accept every prompt.
        "yes | python -m mani_skill.utils.download_asset PutEggplantInBasketScene-v1 || true",
        "yes | python -m mani_skill.utils.download_asset PutSpoonOnTableClothScene-v1 || true",
    )
    .env({"TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_CPP_MIN_LOG_LEVEL": "3"})
)

app = modal.App("octo-smoke", image=image)


@app.function(gpu="L4", timeout=600)
def smoke():
    # Test 1 (run first): Vulkan + ManiSkill SimplerEnv scene.
    # SAPIEN's Vulkan must grab the GPU before JAX/CUDA monopolizes it; running
    # JAX first reliably yields ErrorInitializationFailed in createDeviceUnique.
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers envs)
    # ManiSkill prompts via input() for the WidowX URDF too (the image build only
    # pre-downloaded the *scene* assets). Auto-accept any remaining prompts.
    from mani_skill.utils import download_asset as _da
    _da.prompt_yes_no = lambda *a, **k: True
    env = gym.make(
        "PutEggplantInBasketScene-v1",
        obs_mode="rgb+segmentation",
        # The control mode in the original instructions
        # ("...align_interpolate_by_planner_gripper_pd_joint_target_delta_pos")
        # does not exist in mani_skill==3.0.0b22 for this env — only
        # "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos" is supported.
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=0)
    sensor_keys = list(obs["sensor_data"].keys())
    print(f"PASS: SimplerEnv scene reset; sensor keys: {sensor_keys}")
    first_cam = obs["sensor_data"][sensor_keys[0]]["rgb"]
    print(f"      camera rgb shape={getattr(first_cam, 'shape', None)} dtype={getattr(first_cam, 'dtype', None)}")
    env.close()

    # Test 2: JAX sees the GPU
    import jax
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    assert any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices), \
        f"JAX didn't find GPU: {devices}"
    print("PASS: JAX GPU OK")

    # Test 3: Octo loads
    from octo.model.octo_model import OctoModel
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(model.params)) / 1e6
    print(f"PASS: Octo loaded; param count: {n_params:.1f}M")
    print(f"      dataset_statistics keys: {list(model.dataset_statistics.keys())}")
    return "OK"


@app.local_entrypoint()
def main():
    print(smoke.remote())
