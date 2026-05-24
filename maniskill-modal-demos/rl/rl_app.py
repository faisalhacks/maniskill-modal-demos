"""Phase 2 RL training app for Agent 2.

Trains PPO on PickCube-v1 on Modal (L4) using the official baseline script from
haosulab/ManiSkill, then evaluates the trained policy and saves a rollout video.

Run:
  modal run rl_app.py                 # state-mode training + eval
  modal run rl_app.py --mode rgb      # rgb-mode training + eval
  modal run rl_app.py --mode both
"""

import modal

# A10G fails vkCreateDevice on Modal regardless of loader fixes; L4 works.
# Loader fix is still applied because the bundled NVIDIA userspace in
# maniskill/base is older than the host driver Modal mounts.
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
        "torchvision",
        "tyro",
        "tensorboard",
        "wandb",
        "imageio[ffmpeg]",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/haosulab/ManiSkill.git /opt/ManiSkill"
    )
)

app = modal.App("rl-demo", image=image)
volume = modal.Volume.from_name("rl-outputs", create_if_missing=True)


def fix_nvidia_vulkan_loader():
    """See rl_smoke.py for full rationale. Modal mounts host NVIDIA libs as
    versioned .so.X.Y.Z but the .so.0 symlinks (referenced by the Vulkan ICD)
    still point at the bundled stale userspace lib. Re-link them at runtime
    and rewrite the ICD JSON to use the absolute path of the host lib."""
    import glob, json, os, re, subprocess

    lib_dir = "/usr/lib/x86_64-linux-gnu"
    versioned = [
        v for v in glob.glob(f"{lib_dir}/libGLX_nvidia.so.[0-9]*.[0-9]*")
        if re.search(r"\.so\.\d+\.\d+", v)
    ]
    if not versioned:
        print("fix_nvidia_vulkan_loader: no versioned libGLX_nvidia found, skipping")
        return

    def vkey(p):
        m = re.search(r"\.so\.([\d.]+)$", p)
        return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)

    target = sorted(versioned, key=vkey)[-1]
    print(f"fix_nvidia_vulkan_loader: using {target}")

    icd = {
        "file_format_version": "1.0.0",
        "ICD": {"library_path": target, "api_version": "1.3.289"},
    }
    with open("/usr/share/vulkan/icd.d/nvidia_icd.json", "w") as f:
        json.dump(icd, f, indent=4)

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
            except OSError:
                pass
    subprocess.run(["ldconfig"], check=False)


@app.function(
    gpu="L4",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 60,
)
def train_state():
    """State-based PPO on PickCube. Bumped from 2M (under-trained, return
    still climbing) → 5M. Early-stop if eval_success_once_mean ≥ 0.50 once
    we've passed 3M timesteps."""
    fix_nvidia_vulkan_loader()
    import subprocess, sys, shutil, os, time, signal, re
    os.makedirs("/outputs/runs", exist_ok=True)
    cmd = [
        sys.executable, "-u",  # unbuffered, so we see eval lines in real time
        "/opt/ManiSkill/examples/baselines/ppo/ppo.py",
        "--env_id=PickCube-v1",
        "--num_envs=1024",
        "--update_epochs=8",
        "--num_minibatches=32",
        "--total_timesteps=5000000",
        "--num-steps=50",
        "--num-eval-steps=50",
        "--num_eval_envs=16",
        "--eval_freq=10",
        "--exp-name=state_pickcube",
        "--no-track",
    ]
    print("Running:", " ".join(cmd))
    t0 = time.time()

    proc = subprocess.Popen(
        cmd, cwd="/opt/ManiSkill/examples/baselines/ppo",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    EARLY_STOP_STEPS = 3_000_000
    EARLY_STOP_SUCCESS = 0.50
    last_global_step = 0
    early_stopped = False
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            m_step = re.search(r"global_step=(\d+)", line)
            if m_step:
                last_global_step = int(m_step.group(1))
            m_succ = re.search(r"eval_success_once_mean=([\d.]+)", line)
            if m_succ:
                succ = float(m_succ.group(1))
                print(
                    f"[early-stop check] global_step={last_global_step} "
                    f"eval_success_once_mean={succ}",
                    flush=True,
                )
                if last_global_step >= EARLY_STOP_STEPS and succ >= EARLY_STOP_SUCCESS:
                    print(
                        f"[early-stop] success={succ} ≥ {EARLY_STOP_SUCCESS} at "
                        f"step={last_global_step} ≥ {EARLY_STOP_STEPS} — "
                        "terminating PPO subprocess",
                        flush=True,
                    )
                    proc.terminate()
                    early_stopped = True
                    break
    finally:
        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()

    elapsed = time.time() - t0
    print(f"Training elapsed: {elapsed:.1f}s, return code: {rc}, early_stopped={early_stopped}")
    if rc != 0 and not early_stopped:
        raise RuntimeError(f"PPO training exited with code {rc}")

    runs_dir = "/opt/ManiSkill/examples/baselines/ppo/runs"
    if os.path.exists(runs_dir):
        shutil.copytree(runs_dir, "/outputs/runs/state", dirs_exist_ok=True)
    volume.commit()
    return (
        f"state training done in {elapsed:.1f}s "
        f"(early_stopped={early_stopped}, last_step={last_global_step})"
    )


@app.function(
    gpu="L4",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 60,
)
def train_rgb():
    """RGB-based PPO on PickCube; 15-45 min on L4."""
    fix_nvidia_vulkan_loader()
    import subprocess, sys, shutil, os, time
    os.makedirs("/outputs/runs", exist_ok=True)
    cmd = [
        sys.executable, "/opt/ManiSkill/examples/baselines/ppo/ppo_rgb.py",
        "--env_id=PickCube-v1",
        "--num_envs=256",
        "--update_epochs=8",
        "--num_minibatches=8",
        "--total_timesteps=10000000",
        "--exp-name=rgb_pickcube",
        "--no-track",
    ]
    print("Running:", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/opt/ManiSkill/examples/baselines/ppo", check=False)
    elapsed = time.time() - t0
    print(f"RGB training elapsed: {elapsed:.1f}s, return code: {proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"PPO RGB training exited with code {proc.returncode}")

    runs_dir = "/opt/ManiSkill/examples/baselines/ppo/runs"
    if os.path.exists(runs_dir):
        shutil.copytree(runs_dir, "/outputs/runs/rgb", dirs_exist_ok=True)
    volume.commit()
    return f"rgb training done in {elapsed:.1f}s"


@app.function(
    gpu="L4",
    image=image,
    volumes={"/outputs": volume},
    timeout=600,
)
def evaluate(mode: str = "state"):
    """Replay trained checkpoint, save MP4 to /outputs/<mode>_rollout.mp4.

    Eval videos go to runs/<exp>/test_videos/, not runs/<exp>/videos/.
    First try to use the existing test_videos from the training run; only
    re-run ppo.py --evaluate if none exist."""
    fix_nvidia_vulkan_loader()
    import subprocess, sys, os, glob, shutil
    script = "ppo.py" if mode == "state" else "ppo_rgb.py"

    ckpts = glob.glob(f"/outputs/runs/{mode}/*/final_ckpt.pt")
    if not ckpts:
        ckpts = glob.glob(f"/outputs/runs/{mode}/*/ckpt_*.pt")
    if not ckpts:
        raise RuntimeError(f"No checkpoint found under /outputs/runs/{mode}")
    ckpt = sorted(ckpts)[-1]
    print(f"Using checkpoint: {ckpt}")

    if mode == "state":
        shutil.copy(ckpt, "/outputs/state_checkpoint.pt")

    # If the training run already saved test_videos (the final post-training
    # eval pass), use the last one as the rollout — it's already the trained
    # policy. Cheaper than re-running an eval.
    existing_test_vids = sorted(
        glob.glob(f"/outputs/runs/{mode}/*/test_videos/*.mp4")
    )
    if existing_test_vids:
        latest = existing_test_vids[-1]
        out = f"/outputs/{mode}_rollout.mp4"
        shutil.copy(latest, out)
        print(f"Saved rollout video to {out} (from training's post-eval: {latest})")
        volume.commit()
        return f"eval {mode} done (used existing test video)"

    # Fallback: run ppo.py --evaluate to produce fresh test_videos
    cmd = [
        sys.executable, "-u", f"/opt/ManiSkill/examples/baselines/ppo/{script}",
        "--env_id=PickCube-v1",
        "--evaluate",
        f"--checkpoint={ckpt}",
        "--num_eval_envs=4",
        "--num-eval-steps=200",
        f"--exp-name=eval_{mode}",
        "--no-track",
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/opt/ManiSkill/examples/baselines/ppo", check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Eval exited with code {proc.returncode}")

    # ppo.py writes eval videos to test_videos/, not videos/
    for sub in ["test_videos", "videos"]:
        vids = sorted(glob.glob(
            f"/opt/ManiSkill/examples/baselines/ppo/runs/eval_{mode}/{sub}/*.mp4"
        ))
        if vids:
            shutil.copy(vids[-1], f"/outputs/{mode}_rollout.mp4")
            print(f"Saved rollout video to /outputs/{mode}_rollout.mp4 (from {sub})")
            volume.commit()
            return f"eval {mode} done"

    print("WARNING: no video file produced — listing eval dir for debug:")
    for p in glob.glob(f"/opt/ManiSkill/examples/baselines/ppo/runs/eval_{mode}/**/*",
                       recursive=True):
        print("  ", p)
    volume.commit()
    return f"eval {mode} done (NO VIDEO)"


@app.local_entrypoint()
def main(mode: str = "state"):
    """mode = 'state', 'rgb', or 'both'"""
    if mode in ("state", "both"):
        print(train_state.remote())
        print(evaluate.remote("state"))
    if mode in ("rgb", "both"):
        print(train_rgb.remote())
        print(evaluate.remote("rgb"))
    print("Done. Pull videos with: modal volume get rl-outputs state_rollout.mp4 .")
