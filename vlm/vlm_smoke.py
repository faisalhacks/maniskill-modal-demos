"""Phase 1 smoke test (Attempt 11) — find and wire up libGLX_nvidia.

Findings from Attempt 10 on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`:
- No nvidia_icd.json in image (only Mesa ICDs: lvp, intel, radeon)
- Vulkan loader: "Could not get vkCreateInstance via vk_icdGetInstanceProcAddr
  for ICD libGLX_nvidia.so.0" — i.e. libGLX_nvidia.so.0 not on linker path
- SAPIEN warns: "Failed to find glvnd ICD file"
- Modal injects driver libs to /usr/local/nvidia/lib* IFF
  NVIDIA_DRIVER_CAPABILITIES includes "graphics" or "all"

Try: same base image but heavy diagnostics; locate libGLX_nvidia.so.0 at
runtime and write a JSON pointing to its absolute path; then run gym.make.
"""

import modal

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
        "mani_skill==3.0.0b19",
        "torch==2.4.0",
        "torchvision",
        "gymnasium",
    )
    .env({
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "NVIDIA_VISIBLE_DEVICES": "all",
    })
)

app = modal.App("vlm-smoke", image=image)


@app.function(gpu="L4", timeout=300)
def smoke():
    import glob
    import json
    import os
    import subprocess

    def run(cmd, ok_fail=False):
        print(f"\n--- $ {' '.join(cmd)} ---")
        try:
            print(subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode())
        except subprocess.CalledProcessError as e:
            print(f"[exit {e.returncode}]\n{e.output.decode(errors='replace')}")
            if not ok_fail:
                raise
        except FileNotFoundError as e:
            print(f"[not found] {e}")

    run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"])

    print("\n=== Modal NVIDIA-injected lib dirs (FULL listing) ===")
    for d in ["/usr/local/nvidia/lib", "/usr/local/nvidia/lib64",
              "/usr/local/nvidia"]:
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            print(f"  {d}: {len(files)} entries")
            for f in files[:40]:
                print(f"    {f}")
            if len(files) > 40:
                print(f"    ... +{len(files) - 40} more")
        else:
            print(f"  {d}: <does not exist>")

    print("\n=== /dev/nvidia* and /dev/dri ===")
    for pattern in ["/dev/nvidia*", "/dev/dri/*"]:
        for p in sorted(glob.glob(pattern)):
            print(f"  {p}")

    print("\n=== libGLX_nvidia search ===")
    candidates = []
    for pattern in [
        "/usr/local/nvidia/lib*/libGLX_nvidia.so*",
        "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*",
        "/usr/lib64/libGLX_nvidia.so*",
    ]:
        candidates.extend(glob.glob(pattern))
    for p in candidates:
        print(f"  {p} -> {os.readlink(p) if os.path.islink(p) else '(file)'}")

    print(f"\nLD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '<unset>')}")

    # Pick the .so to point the ICD JSON at (prefer Modal-injected nvidia lib)
    libglx = None
    for p in candidates:
        if not os.path.islink(p):  # the real .so.VERSION file
            libglx = p
            break
    if libglx is None and candidates:
        libglx = candidates[0]

    if libglx is None:
        print("\nFAIL: libGLX_nvidia.so not found anywhere; "
              "Modal didn't inject the graphics driver libs")
        # Print everything Modal mounted under /usr/local/nvidia to debug
        for d in ["/usr/local/nvidia", "/usr/local/nvidia/lib",
                  "/usr/local/nvidia/lib64"]:
            if os.path.isdir(d):
                print(f"\n--- ls {d} ---")
                for f in sorted(os.listdir(d))[:50]:
                    print(f"  {f}")
        raise RuntimeError("libGLX_nvidia missing")

    print(f"\nUsing libGLX_nvidia at: {libglx}")

    # Write ICD JSON with absolute path so loader doesn't need linker path
    icd_dir = "/usr/share/vulkan/icd.d"
    os.makedirs(icd_dir, exist_ok=True)
    icd_path = f"{icd_dir}/nvidia_icd.json"
    icd_data = {
        "file_format_version": "1.0.0",
        "ICD": {"library_path": libglx, "api_version": "1.3.296"},
    }
    with open(icd_path, "w") as f:
        json.dump(icd_data, f, indent=4)
    print(f"\nWrote {icd_path}:")
    with open(icd_path) as f:
        print(f.read())

    os.environ["VK_ICD_FILENAMES"] = icd_path

    run(["vulkaninfo", "--summary"], ok_fail=True)

    print("\n=== gym.make PickCube-v1 ===")
    import sapien
    print(f"SAPIEN: {sapien.__version__}")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make("PickCube-v1", obs_mode="state", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    print(f"PASS PickCube-v1 reset; obs type: {type(obs).__name__}")
    env.close()
    return "OK"


@app.local_entrypoint()
def main():
    print(smoke.remote())
