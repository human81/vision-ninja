"""Overlay renderer — boxes, per-track labels, motion trails, zones/lines, counts.

Everything scales with frame resolution (base 1080p) so text and lines stay legible
at 1080p *and* 4K, and text sits on solid/translucent backgrounds for contrast.
All elements toggle from the `render` config section.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Renderer:
    def __init__(self, cfg):
        r = cfg.section("render")
        self.cfg = r
        self.base_thick = int(r.get("thickness", 2))
        self.trail = int(r.get("trail_length", 30))
        self._cache: dict[int, dict] = {}

    # resolution scale factor (1.0 at 1080p, ~2.0 at 4K)
    def _scale(self, h: int) -> float:
        return max(0.8, h / 1080.0)

    def _ann(self, h: int) -> dict:
        if h not in self._cache:
            s = self._scale(h)
            t = max(2, round(self.base_thick * s))
            self._cache[h] = {
                "s": s, "t": t,
                "box": sv.BoxAnnotator(thickness=t),
                "label": sv.LabelAnnotator(
                    text_scale=max(0.5, 0.6 * s),
                    text_thickness=max(1, round(1.4 * s)),
                    text_padding=int(5 * s),
                    text_position=sv.Position.TOP_LEFT),
                "trace": sv.TraceAnnotator(trace_length=self.trail, thickness=t),
            }
        return self._cache[h]

    @staticmethod
    def _text(img, text, org, scale, color=(255, 255, 255), thick=1,
              bg=(0, 0, 0), pad=4):
        (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
        x, y = int(org[0]), int(org[1])
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), bg, -1)
        cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

    def _labels(self, det: sv.Detections, speeds=None) -> list[str]:
        names = det.data.get("class_name") if det.data else None
        out = []
        for i in range(len(det)):
            cls = names[i] if names is not None else str(det.class_id[i])
            tid = det.tracker_id[i] if det.tracker_id is not None else None
            conf = det.confidence[i] if det.confidence is not None else None
            parts = [cls]
            if tid is not None:
                parts.append(f"#{tid}")
            if conf is not None:
                parts.append(f"{conf:.2f}")
            if speeds and tid is not None and int(tid) in speeds:
                parts.append(f"{speeds[int(tid)]:.0f}km/h")
            out.append(" ".join(parts))
        return out

    def draw(self, frame: np.ndarray, det: sv.Detections, speeds=None) -> np.ndarray:
        out = frame.copy()
        if len(det) == 0:
            return out
        a = self._ann(frame.shape[0])
        if self.cfg.get("draw_trails", True) and det.tracker_id is not None:
            out = a["trace"].annotate(out, det)
        if self.cfg.get("draw_boxes", True):
            out = a["box"].annotate(out, det)
        if self.cfg.get("draw_labels", True):
            out = a["label"].annotate(out, det, labels=self._labels(det, speeds))
        return out

    # ---- zones / lines / counts ----
    def draw_annotations(self, frame, ann, geo=None):
        if not self.cfg.get("draw_zones", True):
            return frame
        h, w = frame.shape[:2]
        s = self._scale(h)
        t = max(2, round(self.base_thick * s))
        for z in ann.zones():
            pts = np.array([[int(x * w), int(y * h)] for x, y in z.vertices], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 255))
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 180, 255), t, cv2.LINE_AA)
            name = z.display_name or z.id
            if geo is not None:
                name = f"{name}: {sum(geo.zone_counts.get(z.id, {}).values())}"
            self._text(frame, name, (pts[0][0], pts[0][1] - int(6 * s)),
                       0.7 * s, (0, 220, 255), max(1, round(1.4 * s)))
        for ln in ann.lines():
            (x0, y0), (x1, y1) = ln.vertices
            a = (int(x0 * w), int(y0 * h)); b = (int(x1 * w), int(y1 * h))
            cv2.line(frame, a, b, (0, 255, 0), t, cv2.LINE_AA)
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            nx, ny = dy / L, -dx / L
            tip = (int(mx + nx * 36 * s), int(my + ny * 36 * s))
            cv2.arrowedLine(frame, (int(mx), int(my)), tip, (0, 255, 0),
                            t, cv2.LINE_AA, tipLength=0.4)
            label = ln.display_name or ln.id
            if geo is not None and ln.id in geo.line_counts:
                pos = sum(geo.line_counts[ln.id]["positive"].values())
                neg = sum(geo.line_counts[ln.id]["negative"].values())
                label = f"{label}  +{pos}/-{neg}"
            self._text(frame, label, (a[0], a[1] - int(8 * s)),
                       0.7 * s, (0, 255, 0), max(1, round(1.4 * s)))
        return frame

    def draw_counts(self, frame, geo):
        if not self.cfg.get("draw_counts", True) or geo is None:
            return frame
        h, w = frame.shape[:2]
        s = self._scale(h)
        fs = 0.62 * s
        th = max(1, round(1.4 * s))
        rows = ["FULL FRAME"] + [f"  {k}: {v}" for k, v in geo.full_frame.items()]
        for lid, d in geo.line_counts.items():
            rows.append(f"LINE {lid}: +{sum(d['positive'].values())}"
                        f" / -{sum(d['negative'].values())}")
        line_h = int(28 * s)
        pad = int(10 * s)
        tw = max((cv2.getTextSize(r, FONT, fs, th)[0][0] for r in rows), default=0)
        x0, y0 = int(10 * s), int(64 * s)
        x1 = x0 + tw + 2 * pad
        y1 = y0 - line_h + line_h * len(rows) + pad
        ov = frame.copy()
        cv2.rectangle(ov, (x0 - pad, y0 - line_h), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        y = y0
        for i, r in enumerate(rows):
            col = (0, 220, 255) if r == "FULL FRAME" or r.startswith("LINE") else (255, 255, 255)
            cv2.putText(frame, r, (x0, y), FONT, fs, col, th, cv2.LINE_AA)
            y += line_h
        return frame

    def hud(self, frame: np.ndarray, text: str) -> np.ndarray:
        h, w = frame.shape[:2]
        s = self._scale(h)
        bar = int(34 * s)
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, bar), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (int(12 * s), int(23 * s)), FONT,
                    0.62 * s, (0, 255, 120), max(1, round(1.4 * s)), cv2.LINE_AA)
        return frame
