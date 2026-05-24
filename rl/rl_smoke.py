"""Phase 1 smoke test for RL project (Agent 2).

Verifies:
  1. maniskill/base image boots with ManiSkill 3.0.0b22.
  2. GPU vectorized PickCube-v1 env can be created and reset (num_envs=4).
  3. The official ppo.py script can be imported and run for ~50k timesteps end-to-end.

Run: modal run rl_smoke.py
Expected: prints "OK" and exits with code 0.
"""

import modal

image = (
    modal.Image.from_registry("maniskill/base", add_python="3.10")
    .env(
        {
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
        }
    )
    .pip_install(
        "mani_skill==3.0.0b22",
        "torch==2.4.0",
        "tyro",
        "tensorboard",
        "wandb",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/haosulab/ManiSkill.git /opt/ManiSkill"
    )
)

app = modal.App("rl-smoke", image=image)


def fix_nvidia_vulkan_loader():
    """Modal mounts the host NVIDIA driver as libGLX_nvidia.so.<host_version>,
    but the .so.0 symlink (referenced by /usr/share/vulkan/icd.d/nvidia_icd.json)
    still points at the OLD bundled userspace lib from maniskill/base. Result:
    vkCreateDevice fails with ErrorInitializationFailed because userspace and
    kernel driver versions are mismatched.

    Fix: find the host-mounted versioned lib, rewrite the ICD JSON to point
    directly at its absolute path, and (defensively) re-link the .so.0 symlinks
    to the same versioned files. Idempotent."""
    import glob, json, os, re, subprocess

    lib_dir = "/usr/lib/x86_64-linux-gnu"
    versioned = glob.glob(f"{lib_dir}/libGLX_nvidia.so.[0-9]*.[0-9]*")
    versioned = [v for v in versioned if re.search(r"\.so\.\d+\.\d+", v)]
    if not versioned:
        print("fix_nvidia_vulkan_loader: no versioned libGLX_nvidia.so found, skipping")
        return
    # Sort by version, pick newest
    def vkey(p):
        m = re.search(r"\.so\.([\d.]+)$", p)
        return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)
    target = sorted(versioned, key=vkey)[-1]
    version = re.search(r"\.so\.([\d.]+)$", target).group(1)
    print(f"fix_nvidia_vulkan_loader: host NVIDIA driver lib = {target} (v{version})")

    # Rewrite ICD to point at absolute path of versioned lib
    icd_path = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    icd = {
        "file_format_version": "1.0.0",
        "ICD": {"library_path": target, "api_version": "1.3.289"},
    }
    with open(icd_path, "w") as f:
        json.dump(icd, f, indent=4)
    print(f"fix_nvidia_vulkan_loader: rewrote {icd_path} → library_path={target}")

    # Re-link .so.0 for any matching nvidia user lib (libGLX_nvidia,
    # libnvidia-glcore, libnvidia-rtcore, etc.) to the host versioned file.
    for lib in ["libGLX_nvidia", "libnvidia-glcore", "libnvidia-rtcore",
                "libnvidia-glvkspirv", "libnvidia-eglcore",
                "libnvidia-ptxjitcompiler", "libnvidia-cbl"]:
        cands = [p for p in glob.glob(f"{lib_dir}/{lib}.so.[0-9]*.[0-9]*")
                 if re.search(r"\.so\.\d+\.\d+", p)]
        if not cands:
            continue
        newest = sorted(cands, key=vkey)[-1]
        for symlink in [f"{lib_dir}/{lib}.so.0", f"{lib_dir}/{lib}.so.1"]:
            if os.path.lexists(symlink):
                try:
                    os.remove(symlink)
                except OSError:
                    pass
            try:
                os.symlink(newest, symlink)
                print(f"  linked {os.path.basename(symlink)} → {os.path.basename(newest)}")
            except OSError as e:
                print(f"  WARN: could not link {symlink}: {e}")
    subprocess.run(["ldconfig"], check=False)


def _common_render_probe():
    import subprocess, os
    print("==== /dev/nvidia* ====")
    r = subprocess.run("ls -la /dev/nvidia* /dev/nvidia-caps/* 2>&1 | head -40",
                       shell=True, capture_output=True, text=True)
    print(r.stdout)

    print("==== device extensions via VK_LOADER_DEBUG ====")
    test = r"""
import os
os.environ['VK_LOADER_DEBUG'] = 'error,warn'
import sapien
try:
    rs = sapien.render.RenderSystem()
    print('SAPIEN_RENDER_OK')
except Exception as e:
    print(f'SAPIEN_RENDER_FAIL: {e}')
"""
    r = subprocess.run(["python3", "-c", test], capture_output=True, text=True,
                       env={**os.environ, "VK_LOADER_DEBUG": "error,warn"})
    print("STDOUT:", r.stdout)
    print("STDERR last 3000:", r.stderr[-3000:])


@app.function(gpu="A10G", timeout=300)
def diagnose_a10g():
    fix_nvidia_vulkan_loader()
    _common_render_probe()
    return "DIAG_A10G OK"


@app.function(gpu="L4", timeout=300)
def diagnose_l4():
    fix_nvidia_vulkan_loader()
    _common_render_probe()
    return "DIAG_L4 OK"


@app.function(gpu="T4", timeout=300)
def diagnose_t4():
    fix_nvidia_vulkan_loader()
    _common_render_probe()
    return "DIAG_T4 OK"


@app.function(gpu="L4", timeout=600)
def smoke():
    fix_nvidia_vulkan_loader()

    import subprocess
    import sys

    # Test 1: GPU vectorized env reset
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers envs

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        num_envs=4,
        sim_backend="gpu",
    )
    obs, _ = env.reset(seed=0)
    print(
        f"PASS GPU vectorized env works; obs type={type(obs).__name__}, "
        f"shape={getattr(obs, 'shape', 'n/a')}"
    )
    env.close()

    # Test 2: Tiny PPO end-to-end run (~50k timesteps, should finish in <3 min)
    cmd = [
        sys.executable,
        "/opt/ManiSkill/examples/baselines/ppo/ppo.py",
        "--env_id=PickCube-v1",
        "--num_envs=64",
        "--num_eval_envs=4",
        "--total_timesteps=50000",
        "--num-steps=20",
        "--update_epochs=2",
        "--num_minibatches=4",
        "--no-track",
        "--no-save_model",
        "--no-capture_video",
        "--exp-name=smoke",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd="/opt/ManiSkill/examples/baselines/ppo",
        capture_output=True,
        text=True,
        timeout=300,
    )
    print("STDOUT (last 2000 chars):", result.stdout[-2000:])
    print("STDERR (last 2000 chars):", result.stderr[-2000:])
    assert result.returncode == 0, (
        f"PPO smoke run failed with code {result.returncode}"
    )
    print("PASS PPO smoke run completed")
    return "OK"


@app.local_entrypoint()
def main(diag: bool = False, gpu: str = "a10g"):
    if diag:
        if gpu == "l4":
            print(diagnose_l4.remote())
        elif gpu == "t4":
            print(diagnose_t4.remote())
        else:
            print(diagnose_a10g.remote())
    else:
        print(smoke.remote())
