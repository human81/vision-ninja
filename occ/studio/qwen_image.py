"""Local Qwen-Image-Edit (4-bit GGUF · stable-diffusion.cpp · Metal) — $0 image edit/gen.

The open (Apache-2.0) local replacement for the cloud image models: instruction image
EDITING (the nano_banana / virtual_try_on path) and text-to-image GENERATION (the imagen
path), running 4-bit on Apple-Silicon Metal via the `sd-cli` binary from
stable-diffusion.cpp — out-of-process, like litert-lm for Gemma. Selected per-task in the
image_edit / image_gen dropdowns (a `qwen-image*` id routes here); cloud stays the default.

Bring it up:
  • build sd.cpp:  cd vendor/stable-diffusion.cpp && cmake -B build -DSD_METAL=ON \
                     -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
  • weights (~21GB) into out/studio/qwen/:  Qwen_Image_Edit-Q4_K_M.gguf (QuantStack),
    split_files/vae/qwen_image_vae.safetensors + split_files/text_encoders/
    qwen_2.5_vl_7b_fp8_scaled.safetensors (Comfy-Org/Qwen-Image_ComfyUI).
Env overrides: STUDIO_SD_BIN, STUDIO_QWEN_DIR, STUDIO_QWEN_STEPS.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bin() -> str:
    return os.environ.get("STUDIO_SD_BIN",
                          os.path.join(_ROOT, "vendor/stable-diffusion.cpp/build/bin/sd-cli"))


def _dir() -> str:
    return os.environ.get("STUDIO_QWEN_DIR", os.path.join(_ROOT, "out/studio/qwen"))


_LORA_SUBDIR = "lora/Qwen-Image-Edit-2509"
_LIGHTNING_LORA = "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16"   # 4-step few-step distill


def _weights():
    d = _dir()
    return {
        "diffusion": os.path.join(d, "Qwen_Image_Edit-Q4_K_M.gguf"),
        "vae": os.path.join(d, "split_files/vae/qwen_image_vae.safetensors"),
        "llm": os.path.join(d, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    }


def _lightning():
    """(lora_dir, lora_name) if the 4-step Lightning LoRA is present, else None — enables
    fast few-step (4 steps, cfg 1.0) instead of the slow 20-step path."""
    ld = os.path.join(_dir(), _LORA_SUBDIR)
    if os.path.exists(os.path.join(ld, _LIGHTNING_LORA + ".safetensors")):
        return ld, _LIGHTNING_LORA
    return None


def is_qwen_model(model) -> bool:
    return bool(model) and "qwen-image" in str(model).lower()


def available() -> bool:
    """Is the sd-cli binary built AND all three weight files present? (No work done.)"""
    w = _weights()
    return (os.path.exists(_bin()) and os.access(_bin(), os.X_OK)
            and all(os.path.exists(p) and os.path.getsize(p) > 1_000_000 for p in w.values()))


def status() -> dict:
    w = _weights()
    return {"engine": "stable-diffusion.cpp (Metal)", "bin": os.path.exists(_bin()),
            "weights": {k: os.path.exists(p) for k, p in w.items()}, "ready": available(),
            "lightning": bool(_lightning()), "steps": 4 if _lightning() else 20}


def _run(prompt: str, base_cfg: float, args: list, timeout: int) -> bytes | None:
    w = _weights()
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    # The diffusion transformer runs on Metal; the text-encoder + VAE go on CPU because
    # current sd.cpp Metal lacks the 'PAD' op they use (else it aborts).
    lit = _lightning()
    extra = []
    if lit:                                              # few-step Lightning: 4 steps, cfg 1.0 — fast
        ld, name = lit
        prompt = f"{prompt} <lora:{name}:1.0>"
        steps, cfg = os.environ.get("STUDIO_QWEN_STEPS", "4"), "1.0"
        extra = ["--lora-model-dir", ld]
    else:                                                # base path: ~20 steps (fewer renders blank)
        steps, cfg = os.environ.get("STUDIO_QWEN_STEPS", "20"), str(base_cfg)
    cmd = [_bin(), "-M", "img_gen",
           "--diffusion-model", w["diffusion"], "--vae", w["vae"], "--llm", w["llm"],
           "--offload-to-cpu", "--diffusion-fa", "--backend", "te=cpu,vae=cpu",
           "--sampling-method", "euler", "--flow-shift", "3", "--cfg-scale", cfg,
           "--steps", steps, "-p", prompt, "-o", out] + extra + args
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        if os.path.exists(out) and os.path.getsize(out) > 100:
            with open(out, "rb") as f:
                return f.read()
    except Exception:
        return None
    finally:
        try:
            os.remove(out)
        except Exception:
            pass
    return None


def edit(image_jpg: bytes, prompt: str, timeout: int = 900) -> bytes | None:
    """Instruction EDIT: reference image + prompt → PNG bytes ($0, local Metal)."""
    if not available() or not image_jpg:
        return None
    ref = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    try:
        with open(ref, "wb") as f:
            f.write(image_jpg)
        return _run(prompt, 2.5, ["-r", ref, "--img-cfg-scale", "1.0"], timeout)
    finally:
        try:
            os.remove(ref)
        except Exception:
            pass


def generate(prompt: str, width: int = 1024, height: int = 1024, timeout: int = 900) -> bytes | None:
    """Text-to-image GENERATION → PNG bytes ($0, local Metal)."""
    if not available():
        return None
    return _run(prompt, 4.0, ["-W", str(int(width)), "-H", str(int(height))], timeout)
