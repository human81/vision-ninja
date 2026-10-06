"""Ultralytics YOLO detector (YOLO11/12), MPS-accelerated — all tasks.

The Ultralytics model auto-detects its task from the weights:
  detect (yolo11n.pt) · segment (-seg) · pose (-pose) · obb (-obb) · classify (-cls).
All return `supervision.Detections` so the tracker stays detector-agnostic; the
task-specific extras (masks, keypoints, oriented corners, the classify label) ride
along on `.mask` / `.data["kpts"]` / `.data["xyxyxyxy"]` / `self.cls_label`.

Cost knobs (all from config): model tier (n/s/m/l/x), imgsz, half precision,
device, and class filtering done in-engine so non-target classes never leave the GPU.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from ..device import pick_device


class YoloDetector:
    name = "yolo"

    def __init__(self, cfg):
        d = cfg.section("detector")
        self.model = YOLO(d.get("model", "yolo11n.pt"))
        self.task = getattr(self.model, "task", "detect")  # detect|segment|pose|obb|classify
        self.device = pick_device(d.get("device", "mps"))      # mps → cuda → cpu fallback
        self.half = bool(d.get("half", True)) and self.device != "cpu"   # no FP16 on CPU
        # Ultralytics ≥8.4.1xx replaced `half` with `quantize` and warns on EVERY predict
        # call if `half` is passed (log spam on Cloud Run). Use whichever this version knows.
        if not self.half:
            self._precision = {}
        elif "quantize" in DEFAULT_CFG_DICT:
            self._precision = {"quantize": 16}
        else:
            self._precision = {"half": True}
        self.imgsz = int(d.get("imgsz", 640))
        self.conf = float(d.get("conf", 0.25))
        self.iou = float(d.get("iou", 0.7))
        self.class_ids = cfg.resolve_class_ids()   # None => all classes
        # human-readable names for rendering / labels
        self.names = self.model.names
        self.cls_label: tuple[str, float] | None = None    # last classify result

    def detect(self, frame: np.ndarray) -> sv.Detections:
        # class filtering only applies to the COCO box/mask/pose tasks; obb (DOTA)
        # and classify (ImageNet) have their own label spaces.
        classes = self.class_ids if self.task in ("detect", "segment") else None
        result = self.model.predict(
            frame,
            device=self.device,
            **self._precision,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=classes,
            verbose=False,
        )[0]

        if self.task == "classify":
            top1 = int(result.probs.top1)
            self.cls_label = (self.names.get(top1, str(top1)),
                              float(result.probs.top1conf))
            return sv.Detections.empty()

        det = sv.Detections.from_ultralytics(result)   # boxes (+mask for seg, +obb corners)
        if self.task == "pose" and result.keypoints is not None \
                and result.keypoints.data is not None and len(det):
            try:
                det.data["kpts"] = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            except Exception:
                pass
        return det
