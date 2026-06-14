"""LocateAnything-3B grounder (NVIDIA, HF transformers, trust_remote_code).

Weights (~6 GB) download lazily on first ground(). Runs on MPS if possible,
falls back to CPU. dtype configurable (bf16 recommended; float32 safest on MPS).
Output boxes use <box><x1><y1><x2><y2></box> markup at 0..1000 → parsed to
normalized 0..1 by base.parse_locate_anything_boxes.

GCP path: set grounding.endpoint to offload generate() to a remote server with the
same prompt/parse contract — this class stays the local implementation.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .base import parse_locate_anything_boxes, GroundBox

_DTYPE = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


class LocateAnythingGrounder:
    name = "locate_anything"

    def __init__(self, cfg):
        g = cfg.section("grounding")
        self.model_path = g.get("checkpoint", "nvidia/LocateAnything-3B")
        self.device_pref = g.get("device", "mps")
        self.dtype_name = g.get("dtype", "float32")    # safest default on MPS
        self.gen_mode = g.get("generation_mode", "hybrid")
        self.max_new_tokens = int(g.get("max_new_tokens", 2048))
        self._model = self._proc = self._tok = None     # lazy

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        dtype = getattr(torch, _DTYPE.get(self.dtype_name, "float32"))
        device = ("mps" if self.device_pref == "mps" and torch.backends.mps.is_available()
                  else "cuda" if self.device_pref == "cuda" and torch.cuda.is_available()
                  else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._proc = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
        self._device, self._dtype, self._torch = device, dtype, torch

    def ground(self, frame: np.ndarray, prompt: str) -> list[GroundBox]:
        self._ensure_loaded()
        torch = self._torch
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Locate all {prompt}."}]}]
        text = self._proc.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self._proc.process_vision_info(messages)
        inputs = self._proc(text=[text], images=images, videos=videos,
                            return_tensors="pt").to(self._device)
        with torch.no_grad():
            response = self._model.generate(
                pixel_values=inputs["pixel_values"].to(self._dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self._tok,
                max_new_tokens=self.max_new_tokens,
                generation_mode=self.gen_mode,
                do_sample=False)
        answer = response[0] if isinstance(response, tuple) else response
        if not isinstance(answer, str):
            answer = self._tok.decode(answer[0] if hasattr(answer, "__len__")
                                      else answer, skip_special_tokens=True)
        return parse_locate_anything_boxes(answer, label=prompt)
