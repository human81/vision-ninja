"""Detector interface. Implementations take a config and return
supervision.Detections for a single BGR frame."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import supervision as sv


class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on one BGR frame, return supervision.Detections
        (xyxy in pixel coords, class_id, confidence, data['class_name'])."""
        ...
