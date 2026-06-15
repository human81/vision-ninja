"""OWLv2 open-vocabulary grounder — runs on Apple Silicon (MPS).

Unlike the big generative VLMs (LocateAnything/Molmo2, whose remote code needs
decord and won't install on macOS-arm64), OWLv2 is a plain transformers detection
model: text queries -> boxes, ~600MB, MPS-friendly. This is the Mac-native path for
open-vocab zone setup. Lazy-loads on first ground(); CPU fallback if an op is missing.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .base import GroundBox


class Owlv2Grounder:
    name = "owlv2"

    def __init__(self, cfg):
        g = cfg.section("grounding")
        self.ckpt = g.get("owlv2_checkpoint", "google/owlv2-base-patch16-ensemble")
        self.device_pref = g.get("device", "mps")
        self.thr = float(g.get("threshold", 0.1))
        self._proc = self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Owlv2ForObjectDetection
        self._proc = AutoProcessor.from_pretrained(self.ckpt)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.ckpt)
        dev = ("mps" if self.device_pref == "mps" and torch.backends.mps.is_available()
               else "cuda" if self.device_pref == "cuda" and torch.cuda.is_available()
               else "cpu")
        self._model.to(dev).eval()
        self._device, self._torch = dev, torch

    def ground(self, frame: np.ndarray, prompt: str) -> list[GroundBox]:
        self._load()
        torch = self._torch
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        queries = [q.strip() for q in prompt.split(",") if q.strip()] or [prompt]
        inputs = self._proc(text=[queries], images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            try:
                outputs = self._model(**inputs)
            except Exception:                       # MPS op gap → CPU
                if self._device != "cpu":
                    self._device = "cpu"
                    self._model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    outputs = self._model(**inputs)
                else:
                    raise
        W, H = image.size
        ts = torch.tensor([[H, W]]).to(self._device)
        try:
            res = self._proc.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=ts, threshold=self.thr,
                text_labels=[queries])[0]
            labels = res.get("text_labels") or [queries[i] for i in res["labels"].tolist()]
        except Exception:
            res = self._proc.post_process_object_detection(
                outputs=outputs, target_sizes=ts, threshold=self.thr)[0]
            labels = [queries[i] if i < len(queries) else prompt
                      for i in res["labels"].tolist()]
        out = []
        for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            x1, y1, x2, y2 = box
            out.append(GroundBox(max(0, x1 / W), max(0, y1 / H),
                                 min(1, x2 / W), min(1, y2 / H),
                                 label=str(lab), score=float(score)))
        return out
