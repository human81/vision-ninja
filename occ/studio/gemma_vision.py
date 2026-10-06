"""Local GENERAL vision — google/gemma-3-4b-it (multimodal Gemma) on-device, $0.

The litert-lm Gemma brain is text-only, so to let Gemma actually SEE you locally we run a
multimodal Gemma through transformers/MPS — exactly how medgemma.py runs MedGemma. Pick
`gemma-3-4b-it` in the Vision dropdown and describe_image / analyze route here (general
scene/object/colour/text understanding), $0 and private. Loaded lazily, only when selected.

Gated: accept the license at https://huggingface.co/google/gemma-3-4b-it + `hf auth login`.
First use downloads ~8GB. Env: STUDIO_GEMMA_VISION_MODEL.
"""

from __future__ import annotations

import io
import os
import threading

_LOCK = threading.Lock()
_STATE = None            # (model, processor, device) | "off" | None

_SYS = ("You are a sharp, helpful visual assistant. Describe the image clearly and "
        "concisely: the main objects and people, clothing and colours, any readable text, "
        "and what's happening. Answer the user's question directly.")


def model_id() -> str:
    return os.environ.get("STUDIO_GEMMA_VISION_MODEL", "google/gemma-3-4b-it")


def is_gemma_vision(model) -> bool:
    """A multimodal-Gemma vision id (gemma-3*). Distinct from the text-only litert brain
    (gemma-4-12b-it) and from MedGemma."""
    m = str(model or "").lower()
    return "gemma-3" in m


def unload():
    """Free the model (and its GPU memory). Next use reloads it."""
    global _STATE
    with _LOCK:
        if isinstance(_STATE, tuple):
            _STATE = None
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _load():
    global _STATE
    if _STATE is None:
        import torch
        if torch.cuda.is_available():
            # L4 = 24GB: Gemma 3 + MedGemma (~8.6GB each) + LocateAnything-3B (~7GB) + YOLO
            # don't all fit, and only one vision model is ever selected — swap, don't stack.
            # (Called before taking our lock, so two concurrent loads can't deadlock.)
            from . import medgemma
            medgemma.unload()
    with _LOCK:
        if _STATE is not None:
            return _STATE if _STATE != "off" else None
        from .settings import local_models_enabled
        if not local_models_enabled():          # CPU cloud build: ~8GB model would OOM the box
            return None
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            from .offline import offline
            lfo = offline()                                  # cache-only when wifi is off
            from ..device import pick_device
            dev = pick_device("auto")                        # mps (Mac) / cuda (L4) / cpu
            dt = torch.bfloat16 if dev != "cpu" else torch.float32
            proc = AutoProcessor.from_pretrained(model_id(), local_files_only=lfo)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id(), dtype=dt, local_files_only=lfo).to(dev).eval()
            _STATE = (model, proc, dev)
        except Exception:
            _STATE = "off"
            return None
        return _STATE


def available() -> bool:
    """True if loadable. NOTE: first call triggers the (cached) load + ~8GB download —
    only call when the user actually selected it."""
    return _load() is not None


def loaded() -> bool:
    return isinstance(_STATE, tuple)


def describe(jpg: bytes, prompt: str = "") -> str:
    """Describe an image (jpg/png bytes) → text. '' if unavailable."""
    if not jpg:
        return ""
    st = _load()
    if st is None:
        return ""
    model, proc, dev = st
    try:
        import torch
        from PIL import Image
        img = Image.open(io.BytesIO(jpg)).convert("RGB")
        q = prompt or "Describe this image."
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _SYS}]},
            {"role": "user", "content": [{"type": "text", "text": q},
                                         {"type": "image", "image": img}]},
        ]
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(dev)
        in_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return proc.decode(out[0][in_len:], skip_special_tokens=True).strip()
    except Exception as e:
        return f"(gemma vision error: {e})"
