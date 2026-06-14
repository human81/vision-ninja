"""Tracker factory over roboflow/trackers.

All four algorithms share `update(detections: sv.Detections, frame) -> sv.Detections`
(with `.tracker_id` populated). They take *different* constructor params, so we
signature-filter the config's `params` dict — unknown keys are dropped with a warning
instead of raising, keeping the config a permissive superset.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import supervision as sv
from trackers import (
    ByteTrackTracker,
    BoTSORTTracker,
    OCSORTTracker,
    SORTTracker,
)

_REGISTRY = {
    "bytetrack": ByteTrackTracker,
    "botsort": BoTSORTTracker,
    "ocsort": OCSORTTracker,
    "sort": SORTTracker,
}


def build_tracker(algorithm: str, params: dict | None = None):
    """Instantiate a tracker, passing only params its constructor accepts."""
    algo = algorithm.lower()
    if algo not in _REGISTRY:
        raise ValueError(
            f"unknown tracker {algorithm!r}; choose from {sorted(_REGISTRY)}")
    cls = _REGISTRY[algo]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    params = params or {}
    used = {k: v for k, v in params.items() if k in accepted}
    ignored = set(params) - accepted
    if ignored:
        warnings.warn(
            f"{cls.__name__} ignores params {sorted(ignored)}", stacklevel=2)
    return cls(**used)


class Tracker:
    """Thin wrapper so the pipeline calls one stable interface regardless of algo.

    On detector-skip frames, call `update` with `sv.Detections.empty()` and the
    underlying Kalman filter predicts positions for still-alive tracks."""

    def __init__(self, algorithm: str, params: dict | None = None):
        self.algorithm = algorithm
        self._impl = build_tracker(algorithm, params)

    def update(self, detections: sv.Detections,
               frame: np.ndarray | None = None) -> sv.Detections:
        return self._impl.update(detections, frame)
