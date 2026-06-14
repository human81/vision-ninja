"""Homography-based speed estimation.

A 4-point ground calibration maps image pixels to real-world coordinates (meters).
Per track we project the bottom-center anchor to the ground plane and differentiate
its world position over a short time window — robust to the perspective foreshortening
that makes naive pixel-speed useless for traffic.

Calibration (per camera, in config):
    calibration:
      image_points: [[x,y], ...]   # 4 points, normalized 0..1, in the image
      world_points: [[X,Y], ...]   # same 4 points in meters on the ground
      units: "kmh"                 # kmh | mph | ms
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import cv2

from .geometry import _anchors

_UNIT = {"ms": 1.0, "kmh": 3.6, "mph": 2.2369362920544}


class SpeedEstimator:
    def __init__(self, image_points, world_points, units: str = "kmh",
                 window: float = 0.5, max_history: int = 30,
                 max_step_ms: float = 70.0, min_samples: int = 3):
        self.units = units
        self.factor = _UNIT.get(units, 3.6)
        self.window = window
        self.max_step_ms = max_step_ms      # >this m/s between frames = track discontinuity
        self.min_samples = min_samples      # need this many points before reporting
        self.img_pts = np.array(image_points, dtype=np.float32)   # normalized
        self.world_pts = np.array(world_points, dtype=np.float32)
        self._H = None
        self._hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))

    @classmethod
    def from_config(cls, cfg) -> "SpeedEstimator | None":
        c = cfg.section("calibration")
        if not c or not c.get("image_points") or not c.get("world_points"):
            return None
        return cls(c["image_points"], c["world_points"],
                   units=c.get("units", "kmh"),
                   window=float(c.get("window", 0.5)))

    def _homography(self, w: int, h: int):
        if self._H is None:
            src = self.img_pts * np.array([w, h], np.float32)
            self._H = cv2.getPerspectiveTransform(
                src.astype(np.float32), self.world_pts)
        return self._H

    def _to_world(self, pts_px: np.ndarray, w: int, h: int) -> np.ndarray:
        H = self._homography(w, h)
        p = pts_px.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(p, H).reshape(-1, 2)

    def update(self, det, w: int, h: int, t: float) -> dict[int, float]:
        """Return {track_id: speed} for tracks with enough history."""
        speeds: dict[int, float] = {}
        if det.tracker_id is None or len(det) == 0:
            return speeds
        anchors = _anchors(det)
        world = self._to_world(anchors, w, h)
        live = set()
        for i, tid in enumerate(det.tracker_id):
            tid = int(tid)
            if tid < 0:
                continue
            live.add(tid)
            hist = self._hist[tid]
            # discontinuity guard: a huge one-frame jump = ID switch / bad box → restart
            if hist:
                pt, pp = hist[-1]
                step_dt = t - pt
                if step_dt > 1e-3:
                    step_ms = float(np.linalg.norm(world[i] - pp)) / step_dt
                    if step_ms > self.max_step_ms:
                        hist.clear()
            hist.append((t, world[i]))
            if len(hist) < self.min_samples:
                continue
            # oldest sample within the window → baseline for the speed estimate
            t0, p0 = hist[0]
            for tt, pp in hist:
                if t - tt <= self.window:
                    t0, p0 = tt, pp
                    break
            dt = t - t0
            if dt >= self.window * 0.5:
                dist = float(np.linalg.norm(world[i] - p0))
                speeds[tid] = dist / dt * self.factor
        for tid in [k for k in self._hist if k not in live]:
            del self._hist[tid]
        return speeds
