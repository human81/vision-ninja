"""RF-DETR detector (HuggingFace transformers) — the accuracy option.

DINOv2 backbone + shallow DETR decoder; beats YOLO on mAP at similar latency.
Filters by class *name* (RF-DETR's label ids differ from YOLO's COCO-80 ids).
Tries MPS, falls back to CPU automatically if an op isn't supported there.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from ..device import pick_device


class RfDetrDetector:
    name = "rfdetr"

    def __init__(self, cfg):
        d = cfg.section("detector")
        ckpt = d.get("rfdetr_checkpoint", "Roboflow/rf-detr-nano")
        self.processor = AutoImageProcessor.from_pretrained(ckpt)
        self.model = AutoModelForObjectDetection.from_pretrained(ckpt)
        self.conf = float(d.get("conf", 0.25))
        self.id2label = self.model.config.id2label
        names = cfg.resolve_class_names()
        self.keep = names                       # set[str] | None
        self.device = self._pick_device(d.get("device", "mps"))
        self.model.to(self.device).eval()

    @staticmethod
    def _pick_device(pref: str) -> str:
        return pick_device(pref)

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> sv.Detections:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        try:
            outputs = self.model(**inputs)
        except Exception:                        # MPS op gap → fall back to CPU
            if self.device != "cpu":
                self.device = "cpu"
                self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                outputs = self.model(**inputs)
            else:
                raise
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=self.conf)[0]
        det = sv.Detections.from_transformers(
            {k: v.cpu() for k, v in results.items()}, id2label=self.id2label)
        if self.keep is not None and len(det):
            names = det.data.get("class_name")
            if names is not None:
                mask = np.array([n in self.keep for n in names], dtype=bool)
                det = det[mask]
        return det
