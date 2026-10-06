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


# The out-of-box "powers" surfaced in the Medical dashboard (key → (label, prompt)).
# vqa uses the user's own question; the rest are one-tap presets.
TASKS = {
    "findings": ("🩻 Findings",
                 "Describe the salient findings in this medical image. Note the imaging "
                 "modality and anatomy if identifiable."),
    "report": ("📋 Structured report",
               "Write a structured radiology-style report with two sections: FINDINGS "
               "(systematic) and IMPRESSION (concise)."),
    "modality": ("🔬 Modality & anatomy",
                 "Identify the imaging modality, the body region / anatomy, and the view "
                 "or projection if applicable."),
    "abnormal": ("⚠️ Abnormalities",
                 "Are there any abnormalities? List each one with its approximate "
                 "location, or state clearly if the image appears within normal limits."),
    "differential": ("🧠 Differential",
                     "Give a brief, cautious differential — what could explain these "
                     "findings? Rank the most likely first."),
    "measure": ("📐 Quantify",
                "Note any measurable or gradable features (sizes, ratios, severity "
                "grades) that are visible, with the usual caveats."),
    "vqa": ("💬 Ask a question", ""),
}

# one-click EXAMPLE images per sub-expertise (public-domain, resolved from Wikimedia
# Commons + cached on disk) — so each power is obvious with real medical imagery.
EXAMPLE_QUERIES = {
    "cxr": ("🫁 Chest X-ray", "chest radiograph"),
    "ct": ("🧠 Brain CT/MRI", "computed tomography brain"),
    "derm": ("🩹 Skin lesion", "melanoma skin lesion"),
    "fundus": ("👁 Retina (fundus)", "fundus photograph retina"),
    "histo": ("🧫 Histopathology", "histopathology micrograph"),
}
_EX_DIR = "out/studio/medical_examples"


def example_image(key: str):
    """Public-domain example image for a sub-expertise (Commons search → cached jpg bytes)."""
    import urllib.parse
    import urllib.request
    import json as _json
    q = EXAMPLE_QUERIES.get(key)
    if not q:
        return None
    os.makedirs(_EX_DIR, exist_ok=True)
    cache = os.path.join(_EX_DIR, key + ".jpg")
    if os.path.exists(cache) and os.path.getsize(cache) > 1000:
        return open(cache, "rb").read()
    ua = {"User-Agent": "occ-studio-medgemma/1.0 (research; jeanlaboratories@gmail.com)"}
    api = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search"
           "&gsrsearch=" + urllib.parse.quote(q[1]) + "&gsrnamespace=6&gsrlimit=10"
           "&prop=imageinfo&iiprop=url|mime|size")
    try:
        j = _json.loads(urllib.request.urlopen(urllib.request.Request(api, headers=ua), timeout=20).read())
        pages = [p for p in (j.get("query", {}).get("pages") or {}).values() if p.get("imageinfo")]
        pages.sort(key=lambda p: p["imageinfo"][0].get("width", 0), reverse=True)
        import cv2
        import numpy as np
        for p in pages:
            ii = p["imageinfo"][0]
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            raw = urllib.request.urlopen(urllib.request.Request(ii["url"], headers=ua), timeout=30).read()
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            h, w = arr.shape[:2]
            s = 1024 / max(h, w)
            if s < 1:
                arr = cv2.resize(arr, (int(w * s), int(h * s)))
            jpg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            open(cache, "wb").write(jpg)
            return jpg
    except Exception:
        return None
    return None


# specialty presets prime the question box for common domains
SPECIALTIES = {
    "cxr": ("🫁 Chest X-ray", "Read this chest radiograph."),
    "derm": ("🩹 Dermatology", "Describe this skin lesion (morphology, colour, borders)."),
    "fundus": ("👁 Retina / fundus", "Assess this fundus image for retinopathy features."),
    "histo": ("🧫 Histopathology", "Describe this histopathology field (tissue, features)."),
    "ct": ("🧠 CT / MRI", "Describe this cross-sectional (CT/MRI) slice."),
}


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
            from . import gemma_vision
            gemma_vision.unload()
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
    """True if MedGemma is loaded/loadable. NOTE: triggers the (cached) load + ~8GB
    download on first call — only call when the user actually selected it."""
    return _load() is not None


def loaded() -> bool:
    """Is the model resident in memory right now? Never triggers a load."""
    return isinstance(_STATE, tuple)


def _cached() -> bool:
    """Are the weights already downloaded? (No load.)"""
    try:
        from huggingface_hub import constants
        d = os.path.join(constants.HF_HUB_CACHE, "models--" + model_id().replace("/", "--"))
        return os.path.isdir(d) and any(os.scandir(d))
    except Exception:
        return False


def status() -> dict:
    """Dashboard status WITHOUT forcing a load: loaded / ready (downloaded) /
    needs-download / error, plus the task + specialty menus."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForImageTextToText  # noqa: F401
        deps = True
    except Exception:
        deps = False
    if loaded():
        state = "loaded"
    elif _STATE == "off":
        state = "error"
    elif not deps:
        state = "needs-deps"
    elif _cached():
        state = "ready"
    else:
        state = "needs-download"
    return {"model": model_id(), "state": state, "loaded": loaded(), "deps": deps,
            "tasks": [{"key": k, "label": v[0]} for k, v in TASKS.items()],
            "specialties": [{"key": k, "label": v[0], "hint": v[1]} for k, v in SPECIALTIES.items()],
            "examples": [{"key": k, "label": v[0]} for k, v in EXAMPLE_QUERIES.items()]}


def analyze(jpg: bytes, task: str = "findings", question: str = "") -> str:
    """Run a preset 'power' (or a free-text question) on a medical image."""
    label, preset = TASKS.get(task, TASKS["findings"])
    q = question.strip()
    if task == "vqa":
        prompt = q or "What can you tell me about this medical image?"
    elif q:
        prompt = preset + " Also: " + q
    else:
        prompt = preset
    return describe(jpg, prompt)


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
