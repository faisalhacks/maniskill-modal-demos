"""Phase 2.0 — Molmo regex sanity test (cheap, runs in ~30 sec after model load).

Why: the only Molmo-specific failure modes are (a) model card / chat-template
drift breaking the load path, and (b) Molmo emitting a slightly different
<point ...> format than what the regex in vlm_app.py expects. Running the
full ManiSkill rollout to detect either of these is wasteful.

This script:
  1. Builds the same image as the Attempt-12 smoke (Ubuntu 22.04 + Vulkan
     deps + ManiSkill stack) and bakes the Molmo weights into the image
     so @modal.enter is just a disk load.
  2. Constructs a 224x224 white image with a red square in the center.
  3. Asks Molmo to "Point to the red square."
  4. Confirms the `<point x="..." y="...">` regex matches and the pixel
     lands somewhere inside (or near) the red square.

If this passes, vlm_app.py can ship with confidence in the VLM step.
"""

import modal

MODEL_ID = "allenai/Molmo-7B-D-0924"


def _bake_molmo():
    """Pre-download Molmo weights into the image layer.

    Done at image build (not @modal.enter) so cold starts are <2 min
    instead of >5 min, and the same warm image can be re-used across
    runs without re-downloading the 15 GB checkpoint."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID)


# Same image stack as the Attempt 12 smoke test. Molmo deps added on top.
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
        # ManiSkill stack from Attempt 12 (already known-good)
        "mani_skill==3.0.0b19",
        "torch==2.4.0",
        "torchvision",
        "gymnasium",
        # Molmo additions. transformers pinned to 4.45.2: anything newer
        # imports torch.distributed.tensor.device_mesh which only exists
        # in torch>=2.5, and we need torch==2.4.0 for ManiSkill compat.
        # 4.45.2 is the version Molmo's model card lists as tested.
        "transformers==4.45.2",
        "tokenizers>=0.20",
        "einops",
        "accelerate",
        "pillow",
        "huggingface_hub",
    )
    .env({
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "NVIDIA_VISIBLE_DEVICES": "all",
    })
    .run_function(_bake_molmo)
)

app = modal.App("vlm-molmo-test", image=image)


@app.cls(gpu="L4", timeout=600, scaledown_window=120)
class MolmoProbe:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        print(f"Loading {MODEL_ID} ...")
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
        print("Loaded.")

    @modal.method()
    def probe(self) -> dict:
        import re
        from PIL import Image, ImageDraw

        # Build a 224x224 white image with a 64x64 red square centered
        # (pixel rect [80,80] to [144,144], i.e. roughly 36-64% of W/H).
        img = Image.new("RGB", (224, 224), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([80, 80, 144, 144], fill="red")

        prompt = "Point to the red square."
        inputs = self.processor.process(images=[img], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0)
                  for k, v in inputs.items()}

        # Molmo's generate_from_batch needs a GenerationConfig object,
        # not an int eos token id. The model card spells this out:
        # https://huggingface.co/allenai/Molmo-7B-D-0924
        gen_cfg = self.GenerationConfig(
            max_new_tokens=200, stop_strings="<|endoftext|>"
        )
        with self.torch.no_grad():
            output = self.model.generate_from_batch(
                inputs, gen_cfg, tokenizer=self.processor.tokenizer
            )

        generated = output[0, inputs["input_ids"].size(1):]
        text = self.processor.tokenizer.decode(
            generated, skip_special_tokens=True)
        print(f"Molmo raw output:\n---\n{text}\n---")

        # Same regex used in vlm_app.py
        m = re.search(r'x="([\d.]+)"\s+y="([\d.]+)"', text)
        result = {"raw_text": text, "regex_matched": bool(m)}

        if m is None:
            print("FAIL: regex did not match. Molmo output format drift.")
            return result

        x_pct, y_pct = float(m.group(1)), float(m.group(2))
        w, h = img.size
        u, v = int(x_pct / 100 * w), int(y_pct / 100 * h)
        result.update({"x_pct": x_pct, "y_pct": y_pct,
                       "pixel_u": u, "pixel_v": v})
        # Red square bbox: u in [80, 144], v in [80, 144]
        inside_square = (80 <= u <= 144) and (80 <= v <= 144)
        # Be generous — within 20px of the square also counts as "the
        # model is roughly localising the right object."
        near_square = (60 <= u <= 164) and (60 <= v <= 164)
        result["inside_square"] = inside_square
        result["near_square"] = near_square

        if inside_square:
            print(f"PASS: pixel ({u}, {v}) is INSIDE the red square.")
        elif near_square:
            print(f"PASS (loose): pixel ({u}, {v}) is NEAR the red square.")
        else:
            print(f"WARN: pixel ({u}, {v}) is NOT near the red square; "
                  "regex parses but localisation is off.")

        return result


@app.local_entrypoint()
def main():
    result = MolmoProbe().probe.remote()
    print("\n=== probe result ===")
    for k, v in result.items():
        if k == "raw_text":
            print(f"  {k}: <{len(v)} chars; see above>")
        else:
            print(f"  {k}: {v}")
    # Exit code semantics: non-zero if the regex failed (a real problem),
    # zero otherwise (even loose localisation is fine — the demo only
    # cares about the regex returning *something* plausible).
    if not result["regex_matched"]:
        raise SystemExit(1)
