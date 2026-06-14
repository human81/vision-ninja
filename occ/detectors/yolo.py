"""Ultralytics YOLO detector (YOLO11/12), MPS-accelerated.

Cost knobs (all from config): model tier (n/s/m/l/x), imgsz, half precision,
device, and class filtering done in-engine so non-target classes never leave the GPU.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO


class YoloDetector:
    name = "yolo"

    def __init__(self, cfg):
        d = cfg.section("detector")
        self.model = YOLO(d.get("model", "yolo11n.pt"))
        self.device = d.get("device", "mps")
        self.half = bool(d.get("half", True))
        self.imgsz = int(d.get("imgsz", 640))
        self.conf = float(d.get("conf", 0.25))
        self.iou = float(d.get("iou", 0.7))
        self.class_ids = cfg.resolve_class_ids()   # None => all classes
        # human-readable names for rendering / labels
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.class_ids,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)
