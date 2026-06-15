"""OWLv2 open-vocabulary DETECTOR — type any classes, get boxes (runs on MPS).

Returns supervision.Detections like the other detectors, so tracking / zones /
lines / counts all work on whatever classes you type. It's a grounding model, so
it's much slower than YOLO/RF-DETR — pair it with a high `runtime.detect_every`
(the tracker holds boxes between detections, keeping the stream responsive).
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
from PIL import Image


class Owlv2Detector:
    name = "owlv2"

    def __init__(self, cfg):
        d = cfg.section("detector")
        self.ckpt = d.get("owlv2_checkpoint", "google/owlv2-base-patch16-ensemble")
        self.device_pref = d.get("device", "mps")
        self.thr = float(d.get("owlv2_threshold", 0.1))
        prompt = d.get("owlv2_prompt", "person, car, truck")
        self.queries = [q.strip() for q in str(prompt).split(",") if q.strip()] or ["object"]
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
        self.device, self._torch = dev, torch

    @property
    def names(self):
        return {i: q for i, q in enumerate(self.queries)}

    def detect(self, frame: np.ndarray) -> sv.Detections:
        self._load()
        torch = self._torch
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self._proc(text=[self.queries], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            try:
                outputs = self._model(**inputs)
            except Exception:
                if self.device != "cpu":
                    self.device = "cpu"
                    self._model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    outputs = self._model(**inputs)
                else:
                    raise
        W, H = image.size
        ts = torch.tensor([[H, W]]).to(self.device)
        try:
            res = self._proc.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=ts, threshold=self.thr,
                text_labels=[self.queries])[0]
            labels = res.get("text_labels") or [self.queries[i] for i in res["labels"].tolist()]
        except Exception:
            res = self._proc.post_process_object_detection(
                outputs=outputs, target_sizes=ts, threshold=self.thr)[0]
            labels = [self.queries[i] if i < len(self.queries) else "object"
                      for i in res["labels"].tolist()]
        boxes = res["boxes"].cpu().numpy()
        if len(boxes) == 0:
            return sv.Detections.empty()
        ids = np.array([self.queries.index(l) if l in self.queries else 0 for l in labels])
        return sv.Detections(
            xyxy=boxes.astype(float),
            confidence=res["scores"].cpu().numpy().astype(float),
            class_id=ids.astype(int),
            data={"class_name": np.array([str(l) for l in labels])})
