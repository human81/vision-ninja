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
    dwell: list[tuple[str, str, float, float]]            # (track_id, zone_id, start, end) — over threshold
    track_start: dict[int, float] = field(default_factory=dict)
    active_dwell: list[dict] = field(default_factory=list)  # LIVE dwell for the on-frame timer ring


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


def _segments_cross(p0, p1, a, b) -> bool:
    """True iff the MOVEMENT segment p0->p1 actually intersects the line SEGMENT a->b
    (proper straddle test), NOT merely the infinite line. This is the correctness fix:
    the old code counted a crossing whenever the anchor flipped side of the *infinite*
    line, so an object passing the line's extension (or anywhere on the far side) was
    miscounted. Both segments must straddle each other for a real crossing."""
    d1 = _side(a, b, p0)
    d2 = _side(a, b, p1)
    if (d1 > 0) == (d2 > 0):                 # both movement ends on the same side → no cross
        return False
    d3 = _side(p0, p1, a)
    d4 = _side(p0, p1, b)
    return (d3 > 0) != (d4 > 0)              # AND the line's endpoints straddle the movement


def _seg_dist2(a, b, p) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return (p[0] - (a[0] + t * dx)) ** 2 + (p[1] - (a[1] + t * dy)) ** 2


def _poly_side(verts, p) -> float:
    """Signed side of a polyline: the cross-product sign of the *nearest* segment.
    For a 2-point line this is exactly `_side(v0, v1, p)` — counting is unchanged."""
    best_d, best_s = float("inf"), 0.0
    for i in range(len(verts) - 1):
        a, b = verts[i], verts[i + 1]
        d = _seg_dist2(a, b, p)
        if d < best_d:
            best_d, best_s = d, _side(a, b, p)
    return best_s


class _DwellTracker:
    """Robust per-occupant dwell timing that survives tracker ID CHANGES and brief
    OCCLUSIONS / detection dropouts.

    The bug with keying dwell on the raw track id: when the tracker drops a person for a
    frame (occlusion, fast motion, a missed detection) it usually comes back with a NEW id,
    so the timer resets to zero and the ring flickers. Here each occupant is a persistent
    record re-associated every frame by, in order of strength:
      1. same track id in the same zone (normal continuity), else
      2. the nearest velocity-PREDICTED record in that zone, within a tolerance (this catches
         the id swap — the person is still where physics says they should be).
    Records LINGER for `grace_lost` seconds after they were last seen, so a lost-then-
    reacquired person continues the SAME timer instead of restarting; the ring is held for
    `ring_hold` seconds so it doesn't blink during a one-frame gap.
    """

    def __init__(self, grace_lost: float = 2.0, ring_hold: float = 0.5, match_frac: float = 0.10,
                 gap_grow: float = 0.16, smooth: float = 0.5, confirm: int = 2):
        self.grace_lost = grace_lost      # keep a record alive this long after last seen (occlusion)
        self.ring_hold = ring_hold        # show the ring this long through a brief gap
        self.match_frac = match_frac      # base re-assoc radius, fraction of frame diagonal
        self.gap_grow = gap_grow          # radius GROWS with the occlusion gap (uncertainty)
        self.smooth = smooth              # EMA on the anchor → steady ring + clean velocity
        self.confirm = confirm            # frames a record must be seen before it's shown
        self._rec: dict[int, dict] = {}
        self._next = 1

    def update(self, t: float, inside: list[dict], w: int, h: int):
        """`inside` = [{tid, zone, thresh, anchor, bw, strict}] for every (track, zone) the
        anchor is within (or just inside the hysteresis band). Associate each to a record via
        a global cost assignment (predicted position + size + id continuity)."""
        for did in [d for d, r in self._rec.items() if t - r["last"] > self.grace_lost]:
            del self._rec[did]
        diag = (w * w + h * h) ** 0.5
        # candidate (record, detection) pairs with a combined cost; assign globally best-first
        pairs = []
        for idx, d in enumerate(inside):
            for did, r in self._rec.items():
                if r["zone"] != d["zone"]:
                    continue
                gap = max(0.0, t - r["last"])
                rad = (self.match_frac + self.gap_grow * gap) * diag   # grows with the gap
                px, py = self._predict(r, t)
                dist = ((px - d["anchor"][0]) ** 2 + (py - d["anchor"][1]) ** 2) ** 0.5
                if dist > rad:
                    continue
                cost = dist / rad
                if r["bw"] > 1 and d["bw"] > 1:                        # people keep ~constant size
                    cost += 0.6 * abs(r["bw"] - d["bw"]) / max(r["bw"], d["bw"])
                if r["tid"] == d["tid"]:                               # same id = strong continuity
                    cost -= 0.5
                pairs.append((cost, did, idx))
        pairs.sort(key=lambda x: x[0])
        used_r, used_d = set(), set()
        for cost, did, idx in pairs:
            if did in used_r or idx in used_d:
                continue
            used_r.add(did); used_d.add(idx); self._touch(did, t, inside[idx])
        for idx, d in enumerate(inside):                              # new record only if CLEARLY inside
            if idx not in used_d and d.get("strict", True):
                self._new(t, d)

    def _new(self, t, d):
        self._rec[self._next] = {"zone": d["zone"], "start": t, "last": t, "thresh": d["thresh"],
                                 "pos": d["anchor"], "prev": d["anchor"], "prev_t": t,
                                 "tid": d["tid"], "bw": d["bw"], "hits": 1}
        self._next += 1

    def _touch(self, did, t, d):
        r = self._rec[did]
        sx = self.smooth * d["anchor"][0] + (1 - self.smooth) * r["pos"][0]   # EMA-smoothed anchor
        sy = self.smooth * d["anchor"][1] + (1 - self.smooth) * r["pos"][1]
        if t > r["last"]:
            r["prev"], r["prev_t"] = r["pos"], r["last"]
        r["pos"], r["last"], r["tid"] = (sx, sy), t, d["tid"]
        r["thresh"] = d["thresh"]
        r["bw"] = 0.6 * r["bw"] + 0.4 * d["bw"]
        r["hits"] += 1

    def _predict(self, r, t):
        dt = r["last"] - r["prev_t"]
        if dt > 1e-3:
            gap = min(max(0.0, t - r["last"]), 0.7)        # extrapolate a bit further through gaps
            vx = (r["pos"][0] - r["prev"][0]) / dt
            vy = (r["pos"][1] - r["prev"][1]) / dt
            return (r["pos"][0] + vx * gap, r["pos"][1] + vy * gap)
        return r["pos"]

    def active(self, t: float) -> list[dict]:
        """Records to score/draw now — seen within ring_hold and CONFIRMED (steady, no ghosts)."""
        out = []
        for did, r in self._rec.items():
            if t - r["last"] <= self.ring_hold and r["hits"] >= self.confirm:
                sec = t - r["start"]
                out.append({"track": did, "zone": r["zone"], "seconds": round(sec, 2),
                            "threshold": round(r["thresh"], 2), "anchor": r["pos"],
                            "over": sec >= r["thresh"]})
        return out


class GeometryEngine:
    def __init__(self, annotations: AnnotationSet, min_dwell: float = 1.0):
        self.ann = annotations
        self.min_dwell = min_dwell
        # cumulative line crossing counts
        self._line_counts: dict[str, dict[str, Counter]] = {
            ln.id: {"positive": Counter(), "negative": Counter()}
            for ln in annotations.lines()}
        self._prev_pos: dict[int, tuple[float, float]] = {}   # track id -> last anchor (px)
        self._dwell = _DwellTracker()      # id-change / occlusion-robust per-occupant dwell
        self._track_start: dict[int, float] = {}

    def reset_dwell(self):
        """Clear all running dwell timers (every occupant's counter restarts from zero)."""
        self._dwell = _DwellTracker()

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
        active_dwell: list[dict] = []
        inside_pairs: list[dict] = []      # (track, zone) anchor-inside pairs for dwell timing
        live: set[int] = set()

        zones_px = {z.id: self._to_px(z.vertices, w, h) for z in self.ann.zones()}
        lines_px = {ln.id: self._to_px(ln.vertices, w, h) for ln in self.ann.lines()}
        # a single detection step can't move a real object across most of the frame —
        # a jump that large is a tracker id-swap/teleport, not a crossing, so ignore it.
        max_jump2 = 0.25 * (w * w + h * h)
        diag = (w * w + h * h) ** 0.5
        enter_band = 0.004 * diag      # must be clearly INSIDE to start a dwell (hysteresis)
        leave_band = 0.02 * diag       # keep dwelling until clearly OUTSIDE — no edge flicker

        for i in range(len(det)):
            tid = int(tids[i])
            if tid < 0:
                continue
            live.add(tid)
            cls = names[i]
            p = anchors[i]
            self._track_start.setdefault(tid, t)

            # --- lines (2-pt or polyline): count a crossing ONLY when the object's
            #     movement (prev anchor -> current anchor) actually intersects the line
            #     segment, with right-hand-rule direction from the side it ends on. ---
            p0 = self._prev_pos.get(tid)
            if p0 is not None and (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2 <= max_jump2:
                for ln in self.ann.lines():
                    verts = lines_px[ln.id]
                    for j in range(len(verts) - 1):
                        a, b = verts[j], verts[j + 1]
                        if _segments_cross(p0, p, a, b):
                            if _side(a, b, p) > 0:               # ended on the positive side
                                self._line_counts[ln.id]["positive"][cls] += 1
                            else:
                                self._line_counts[ln.id]["negative"][cls] += 1
                            break                                # one crossing per line per step
            self._prev_pos[tid] = (float(p[0]), float(p[1]))

            # --- zones: occupancy + dwell candidates (with edge hysteresis & box size) ---
            bw = float(det.xyxy[i][2] - det.xyxy[i][0])
            for z in self.ann.zones():
                sd = cv2.pointPolygonTest(zones_px[z.id], (float(p[0]), float(p[1])), True)  # signed dist
                if sd >= 0:
                    zone_counts[z.id][cls] += 1
                if sd >= -leave_band:          # within the keep-dwelling band → a candidate
                    inside_pairs.append({"tid": tid, "zone": z.id,
                                         "thresh": (getattr(z, "dwell", 0.0) or self.min_dwell),
                                         "anchor": (float(p[0]), float(p[1])),
                                         "bw": bw, "strict": sd >= enter_band})

        # robust per-occupant dwell — re-associates the SAME person across id changes and
        # brief occlusions, so the timer keeps counting instead of resetting/flickering.
        self._dwell.update(t, inside_pairs, w, h)
        active_dwell = self._dwell.active(t)
        dwell = [(str(a["track"]), a["zone"], round(t - a["seconds"], 2), round(t, 2))
                 for a in active_dwell if a["over"]]

        # prune line-crossing state for tracks that disappeared
        for tid in [t_ for t_ in self._prev_pos if t_ not in live]:
            del self._prev_pos[tid]
        for tid in [t_ for t_ in self._track_start if t_ not in live]:
            del self._track_start[tid]

        return GeoStats(
            full_frame=full,
            zone_counts=zone_counts,
            line_counts={lid: {k: Counter(v) for k, v in d.items()}
                         for lid, d in self._line_counts.items()},
            dwell=dwell,
            track_start=dict(self._track_start),
            active_dwell=active_dwell,
        )
