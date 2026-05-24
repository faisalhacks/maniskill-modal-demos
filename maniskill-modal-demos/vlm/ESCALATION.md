# VLM Project — Escalation

**Date**: 2026-05-24
**Owner**: Agent 1 (VLM)
**Total attempts**: 8 (all in Phase 1, smoke test)
**Total cost burned**: ~$0.40 (well under the $5 cap)
**Status**: Stopped per the 8-retry rule on a single failure class.

---

## The failure class

All 8 attempts ended with the same Vulkan device-creation error inside SAPIEN, regardless of whether the smoke test went through a manual `sapien.Scene` probe or through ManiSkill's `gym.make("PickCube-v1")`:

```
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed
```

Confirmed during the diagnostic attempts:
- The container DOES see the NVIDIA A10G (driver 580.95.05, 23 GB).
- `NVIDIA_DRIVER_CAPABILITIES=all` is set.
- `libGLX_nvidia.so.580.95.05` is on disk and symlinked correctly.
- `nvidia_icd.json` exists at the standard path.
- A Vulkan instance can be created (the device-creation step is what fails).

## Root-cause hypothesis (highest confidence)

The `maniskill/base` image is built on **Ubuntu 20.04** and ships:
- `libvulkan.so.1` from `vulkan-tools 1.2.131+dfsg1-1` (a 2020-era loader)
- `nvidia_icd.json` declaring `"api_version": "1.2.155"`

But Modal's GPU hosts now run NVIDIA driver **580.95.05** (late 2024 / 2025), whose Vulkan ICD natively supports Vulkan 1.3+ and expects to be paired with a Vulkan 1.3 loader. SAPIEN 3.0.0b1 / 3.0.1 requests Vulkan 1.3 features at device-creation time. The 2020 loader either cannot pass these requests through, or the driver refuses them because the loader negotiates as 1.2.

Rewriting `nvidia_icd.json` to lie about its `api_version` (Attempt 8) had no effect — the actual incompatibility is in the libvulkan loader ABI, not the JSON manifest.

## Last 3 tracebacks

### Attempt 6 — image build failure
```
=> Step 4: RUN apt-get update
E: Conflicting values set for option Signed-By regarding source
   https://packages.lunarg.com/vulkan/ focal: /usr/share/keyrings/lunarg.gpg !=
E: The list of sources could not be read.
```

### Attempt 7 — LunarG 404
```
Err:7 https://packages.lunarg.com/vulkan focal Release
  404  Not Found [IP: 172.67.73.90 443]
E: The repository 'https://packages.lunarg.com/vulkan focal Release' does not have a Release file.
```

### Attempt 8 — runtime device init (after ICD JSON rewrite)
```
File "/opt/conda/lib/python3.9/site-packages/mani_skill/envs/sapien_env.py", line 1084, in _setup_scene
    sapien.Scene([physx_system, sapien.render.RenderSystem(self._render_device)])
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed
```

## What I would try next (in priority order)

1. **Abandon the `maniskill/base` constraint** and rebuild from `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. Ubuntu 22.04 ships `libvulkan1 1.3.224` natively. Install SAPIEN + ManiSkill via pip into a fresh Python env. This was almost certainly the intended path; the project instructions cited `maniskill/base` based on an assumed Modal+ManiSkill recipe that was never verified. **This recommendation overrides the hard constraint.**

   ```python
   image = (
       modal.Image.from_registry(
           "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.10"
       )
       .apt_install(
           "libvulkan1", "libvulkan-dev", "vulkan-tools",
           "libegl1", "libgles2", "libglvnd0", "libgl1",
           "libglx-mesa0", "libxext6", "ffmpeg",
       )
       .pip_install("mani_skill==3.0.0b19", "torch==2.4.0", "transformers>=4.45",
                    "imageio[ffmpeg]", "mplib", "transforms3d", "huggingface_hub")
       .env({"VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
             "NVIDIA_DRIVER_CAPABILITIES": "all"})
   )
   ```

2. **If (1) hits a different error**, pin a specific LunarG version that exists. Verified-live URL pattern: `https://packages.lunarg.com/vulkan/<version>/lunarg-vulkan-<version>-jammy.list`. Recent live versions to try: `1.3.290`, `1.3.296`, `1.4.304`.

3. **If both fail**, fall back to running ManiSkill **without rendering** (state-only obs, no `rgb_array` render_mode) and skip the VLM step entirely — but this defeats the project's purpose.

## Recommendation to the user

Of the four parallel agents, try **Agent 2 (RL)** first, since it likely does not need SAPIEN's Vulkan renderer and can hit a much smaller surface area. Then come back to VLM with the Ubuntu 22.04 image plan above. Net effect: the project as-specified is salvageable, just not via the `maniskill/base` route on Modal's current GPU fleet.

## Cost summary

| Attempt | Cost (approx) |
|--------:|---------------|
| 1       | $0.05         |
| 2       | $0.02         |
| 3       | $0.02         |
| 4       | $0.03         |
| 5       | $0.01 (build) |
| 6       | $0.01 (build) |
| 7       | $0.01 (build) |
| 8       | $0.02         |
| **Total** | **~$0.17 GPU time + ~$0.20 image-build overhead ≈ $0.40** |

Well under the $5 budget — escalation is conservative, not driven by cost.
