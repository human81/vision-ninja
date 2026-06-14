"""Pluggable detectors. Each returns supervision.Detections so the tracker is
detector-agnostic (YOLO today, RF-DETR in Phase 3 — one flag swaps them)."""

from .base import Detector
from .yolo import YoloDetector


def build_detector(cfg) -> Detector:
    backend = cfg.get("detector.backend", "yolo").lower()
    if backend == "yolo":
        return YoloDetector(cfg)
    if backend == "rfdetr":
        from .rfdetr import RfDetrDetector  # Phase 3, optional dep
        return RfDetrDetector(cfg)
    raise ValueError(f"unknown detector backend: {backend!r}")


__all__ = ["Detector", "YoloDetector", "build_detector"]
