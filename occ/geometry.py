"""Geometry engine: track IDs -> occupancy analytics.

For each tracked object we use its **bottom-center anchor** (ground contact) and:
  * zones      — point-in-polygon → instantaneous per-class counts
  * lines      — sign of cross(v1-v0, p-v0) flips as the anchor crosses; a flip from
                 negative→positive side is a **positive** (right-hand-rule) crossing,
                 positive→negative is **negative**. Counts accumulate.
  * dwell      — time the anchor stays inside a zone; emitted once it exceeds min_dwell.
  * track_info — first-seen time per track id.
Counts are keyed by class name. State is pruned when a track disappears.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

from .annotations import AnnotationSet, Annotation


@dataclass
class GeoStats:
    full_frame: Counter                                   # class_name -> count
    zone_counts: dict[str, Counter]                       # zone_id -> class -> count
    line_counts: dict[str, dict[str, Counter]]            # line_id -> {pos,neg} -> class -> count
    dwell: list[tuple[str, str, float, float]]            # (track_id, zone_id, start, end)
    track_start: dict[int, float] = field(default_factory=dict)


def _anchors(det: sv.Detections) -> np.ndarray:
    """Bottom-center of each box, pixel coords."""
    xyxy = det.xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    by = xyxy[:, 3]
    return np.stack([cx, by], axis=1)


def _class_names(det: sv.Detections) -> list[str]:
    names = det.data.get("class_name") if det.data else None
    if names is not None:
        return list(names)
    if det.class_id is not None:
        return [str(c) for c in det.class_id]
    return ["object"] * len(det)


def _side(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class GeometryEngine:
    def __init__(self, annotations: AnnotationSet, min_dwell: float = 1.0):
        self.ann = annotations
        self.min_dwell = min_dwell
        # cumulative line crossing counts
        self._line_counts: dict[str, dict[str, Counter]] = {
            ln.id: {"positive": Counter(), "negative": Counter()}
            for ln in annotations.lines()}
        self._prev_side: dict[tuple[str, int], float] = {}
        self._zone_enter: dict[tuple[str, int], float] = {}
        self._track_start: dict[int, float] = {}

    @staticmethod
    def _to_px(verts, w, h) -> np.ndarray:
        return np.array([[x * w, y * h] for x, y in verts], dtype=np.float32)

    def update(self, det: sv.Detections, w: int, h: int, t: float) -> GeoStats:
        anchors = _anchors(det) if len(det) else np.empty((0, 2))
        names = _class_names(det)
        tids = (det.tracker_id if det.tracker_id is not None
                else np.full(len(det), -1))

        full = Counter(names)
        zone_counts: dict[str, Counter] = {z.id: Counter() for z in self.ann.zones()}
        dwell: list[tuple[str, str, float, float]] = []
        live: set[int] = set()

        zones_px = {z.id: self._to_px(z.vertices, w, h) for z in self.ann.zones()}
        lines_px = {ln.id: self._to_px(ln.vertices, w, h) for ln in self.ann.lines()}

        for i in range(len(det)):
            tid = int(tids[i])
            if tid < 0:
                continue
            live.add(tid)
            cls = names[i]
            p = anchors[i]
            self._track_start.setdefault(tid, t)

            # --- lines ---
            for ln in self.ann.lines():
                a, b = lines_px[ln.id][0], lines_px[ln.id][1]
                s = _side(a, b, p)
                key = (ln.id, tid)
                prev = self._prev_side.get(key)
                if prev is not None and prev != 0 and s != 0 and (prev > 0) != (s > 0):
                    if prev < 0 < s:
                        self._line_counts[ln.id]["positive"][cls] += 1
                    else:
                        self._line_counts[ln.id]["negative"][cls] += 1
                self._prev_side[key] = s

            # --- zones ---
            for z in self.ann.zones():
                inside = cv2.pointPolygonTest(
                    zones_px[z.id], (float(p[0]), float(p[1])), False) >= 0
                key = (z.id, tid)
                if inside:
                    zone_counts[z.id][cls] += 1
                    start = self._zone_enter.setdefault(key, t)
                    if t - start >= self.min_dwell:
                        dwell.append((str(tid), z.id, start, t))
                else:
                    self._zone_enter.pop(key, None)

        # prune state for tracks that disappeared
        for d in (self._zone_enter, self._prev_side):
            for k in [k for k in d if k[1] not in live]:
                del d[k]
        for tid in [t_ for t_ in self._track_start if t_ not in live]:
            del self._track_start[tid]

        return GeoStats(
            full_frame=full,
            zone_counts=zone_counts,
            line_counts={lid: {k: Counter(v) for k, v in d.items()}
                         for lid, d in self._line_counts.items()},
            dwell=dwell,
            track_start=dict(self._track_start),
        )
