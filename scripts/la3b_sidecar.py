#!/usr/bin/env python
"""Run nvidia/LocateAnything-3B locally on Apple Silicon (MPS), image grounding only.

Two tweaks make it work off-CUDA:
  • STUB decord — the processor does a top-level `import decord`, but only uses it in the
    video reader (never for images), so we inject a fake module so the import succeeds.
  • attn_implementation="sdpa" — flash-attn isn't available on Mac; the model's import is
    guarded and ships Sdpa attention classes, so SDPA runs on MPS/CPU.

Must run in the dedicated .venv-la3b (transformers==4.57.1):
    .venv-la3b/bin/python scripts/la3b_sidecar.py <image.jpg> "<prompt>"
"""
from __future__ import annotations

import importlib.machinery
import re
import sys
import types

# ---- 1) stub decord (image grounding never touches the video path) ----
# Needs a real __spec__ — transformers calls importlib.util.find_spec("decord"), which
# raises if a sys.modules entry has __spec__ = None.
_dec = types.ModuleType("decord")
_dec.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
class _NoVideo:                                    # raises only if something tries video
    def __init__(self, *a, **k): raise RuntimeError("decord stub: video unsupported on the Mac sidecar")
_dec.VideoReader = _NoVideo
_dec.cpu = _dec.gpu = lambda *a, **k: None
_dec.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
sys.modules["decord"] = _dec

import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = "nvidia/LocateAnything-3B"


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "assets/test/face.jpg"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "person"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    print(f"[la3b] loading {MODEL} (~6GB on first run) → {device}/{dtype} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, dtype=dtype,
        attn_implementation="sdpa").to(device).eval()
    print("[la3b] loaded. grounding…", flush=True)

    frame = cv2.imread(img_path)
    h0, w0 = frame.shape[:2]                        # cap long side — MoonViT cost grows with res
    cap = 1024
    if max(h0, w0) > cap:
        sc = cap / max(h0, w0)
        frame = cv2.resize(frame, (round(w0 * sc), round(h0 * sc)), interpolation=cv2.INTER_AREA)
        print(f"[la3b] resized {w0}x{h0} -> {frame.shape[1]}x{frame.shape[0]} for grounding")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"Locate all {prompt}."}]}]
    text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(messages)
    inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to(device)

    import time
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            pixel_values=inputs["pixel_values"].to(dtype),
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"), tokenizer=tok,
            max_new_tokens=1024, generation_mode="hybrid", do_sample=False, use_cache=True)
    ans = out[0] if isinstance(out, (list, tuple)) else out
    if not isinstance(ans, str):
        ans = tok.decode(ans[0] if hasattr(ans, "__len__") else ans, skip_special_tokens=True)
    boxes = re.findall(r"<box>(?:<(\d+)>){4}</box>", ans) or re.findall(r"<box>.*?</box>", ans)
    print(f"[la3b] done in {time.time()-t0:.1f}s on {device}")
    print(f"[la3b] grounded '{prompt}': {len(re.findall(r'<box>', ans))} box(es)")
    print("[la3b] raw answer (first 600 chars):")
    print(ans[:600])


if __name__ == "__main__":
    main()
