"""Local MedGemma 4B — specialised MEDICAL-image understanding, on-device, $0.

google/medgemma-4b-it (Gemma-3 multimodal, image-text-to-text) is tuned for medical
imaging — radiology, pathology, dermatology, ophthalmology, histology. There is NO LiteRT
build, so (unlike the Gemma-4 text brain) it does not go through litert-lm; it runs
IN-PROCESS on Apple Silicon via transformers/MPS. Pick `medgemma-4b-it` in the Vision
dropdown and describe_image routes here.

Gated model: accept the license at https://huggingface.co/google/medgemma-4b-it, then
`hf auth login` (or HF_TOKEN). First use downloads ~8GB. Loaded lazily, only when selected.

⚠️ Decision-support only — the model is told to defer to a qualified clinician, never to
diagnose. Keep that framing in any UI copy.
"""

from __future__ import annotations

import io
import os
import threading

_LOCK = threading.Lock()
_STATE = None            # (model, processor, device) | "off" | None(unloaded)

_MED_SYS = (
    "You are a careful medical-imaging assistant. Describe the salient findings in this "
    "image precisely: note the modality and anatomy if identifiable, and any notable "
    "features. This is DECISION-SUPPORT, not a diagnosis — explicitly recommend review by "
    "a qualified clinician and avoid definitive claims.")


def model_id() -> str:
    return os.environ.get("STUDIO_MEDGEMMA_MODEL", "google/medgemma-4b-it")


def is_medgemma(model) -> bool:
    return bool(model) and "medgemma" in str(model).lower()


def _load():
    global _STATE
    with _LOCK:
        if _STATE is not None:
            return _STATE if _STATE != "off" else None
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            dt = torch.bfloat16 if dev == "mps" else torch.float32
            proc = AutoProcessor.from_pretrained(model_id())
            model = AutoModelForImageTextToText.from_pretrained(
                model_id(), torch_dtype=dt).to(dev).eval()
            _STATE = (model, proc, dev)
        except Exception:
            _STATE = "off"
            return None
        return _STATE


def available() -> bool:
    """True if MedGemma is loaded/loadable. NOTE: triggers the (cached) load + ~8GB
    download on first call — only call when the user actually selected it."""
    return _load() is not None


def describe(jpg: bytes, prompt: str = "") -> str:
    """Analyse a medical image (jpg/png bytes) → findings text. '' if unavailable."""
    if not jpg:                     # short-circuit BEFORE loading (so '' never triggers an 8GB pull)
        return ""
    st = _load()
    if st is None:
        return ""
    model, proc, dev = st
    try:
        import torch
        from PIL import Image
        img = Image.open(io.BytesIO(jpg)).convert("RGB")
        q = prompt or "Describe the findings in this medical image."
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _MED_SYS}]},
            {"role": "user", "content": [{"type": "text", "text": q},
                                         {"type": "image", "image": img}]},
        ]
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(dev)
        in_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        return proc.decode(out[0][in_len:], skip_special_tokens=True).strip()
    except Exception as e:
        return f"(medgemma error: {e})"
