#!/usr/bin/env python
"""Persistent LocateAnything-3B grounding sidecar (Apple Silicon / MPS).

Runs in the dedicated .venv-la3b (transformers==4.57.1) so its version pins stay isolated
from the studio. Loads the ~6GB model ONCE and serves it over HTTP, so the studio can call
open-vocab grounding without the per-call reload. Image grounding only (decord stubbed).

    .venv-la3b/bin/python scripts/la3b_server.py            # → http://127.0.0.1:9393
    curl -s localhost:9393/health
    POST /ground {image: "<data-uri or base64 jpg>", prompt: "forklift", max_side: 1024}
        → {boxes: [[x1,y1,x2,y2], ...]  (normalized 0..1), n, secs}
"""
from __future__ import annotations

import base64
import importlib.machinery
import os
import re
import sys
import time
import types

# ---- stub decord (top-level import in the processor; used only for video) ----
_dec = types.ModuleType("decord")
_dec.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
_dec.VideoReader = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("video unsupported"))
_dec.cpu = _dec.gpu = lambda *a, **k: None
_dec.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
sys.modules["decord"] = _dec

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = os.environ.get("LA3B_MODEL", "nvidia/LocateAnything-3B")
PORT = int(os.environ.get("LA3B_PORT", "9393"))
app = FastAPI()
_S: dict = {}


def _load():
    if _S:
        return _S
    device = ("mps" if torch.backends.mps.is_available()          # Mac
              else "cuda" if torch.cuda.is_available() else "cpu")  # Cloud Run L4 / fallback
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"[la3b] loading {MODEL} → {device}/{dtype} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype,
                                      attn_implementation="sdpa").to(device).eval()
    _S.update(tok=tok, proc=proc, model=model, device=device, dtype=dtype)
    print("[la3b] ready.", flush=True)
    return _S


@app.get("/health")
def health():
    return {"ready": bool(_S), "model": MODEL, "device": _S.get("device", "?")}


@app.post("/ground")
async def ground(req: Request):
    b = await req.json()
    s = _load()
    prompt = (b.get("prompt") or "object").strip()
    try:
        raw = base64.b64decode((b.get("image") or "").split(",")[-1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"error": "bad image"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"decode: {e}"}, status_code=400)
    cap = int(b.get("max_side", 1024))
    h0, w0 = frame.shape[:2]
    if max(h0, w0) > cap:                              # MoonViT cost ∝ resolution → cap it
        sc = cap / max(h0, w0)
        frame = cv2.resize(frame, (round(w0 * sc), round(h0 * sc)), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"Locate all {prompt}."}]}]
    text = s["proc"].py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = s["proc"].process_vision_info(messages)
    inputs = s["proc"](text=[text], images=images, videos=videos, return_tensors="pt").to(s["device"])
    t0 = time.time()
    with torch.no_grad():
        out = s["model"].generate(
            pixel_values=inputs["pixel_values"].to(s["dtype"]),
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"), tokenizer=s["tok"],
            max_new_tokens=int(b.get("max_new_tokens", 1024)),
            generation_mode="hybrid", do_sample=False, use_cache=True)
    ans = out[0] if isinstance(out, (list, tuple)) else out
    if not isinstance(ans, str):
        ans = s["tok"].decode(ans, skip_special_tokens=True)
    boxes = [[int(v) / 1000.0 for v in m.groups()]            # 0..1000 markup → normalized 0..1
             for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", ans)]
    return {"boxes": boxes, "n": len(boxes), "secs": round(time.time() - t0, 1), "prompt": prompt}


if __name__ == "__main__":
    _load()                                            # warm before serving
    uvicorn.run(app, host="127.0.0.1", port=PORT)
