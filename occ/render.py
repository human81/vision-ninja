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

# COCO-17 pose skeleton (keypoint index pairs).
_SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
             (2, 4), (0, 5), (0, 6)]
_PALETTE = [(54, 215, 224), (60, 255, 170), (60, 180, 255), (230, 90, 230),
            (40, 150, 255), (90, 230, 90), (170, 120, 255), (60, 220, 250)]


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

    def _palette(self, i):
        return _PALETTE[int(i) % len(_PALETTE)]

    def draw_task(self, frame, det, task, cls_label=None):
        """Render the non-detect YOLO task outputs: segment masks, pose skeletons,
        oriented (obb) boxes, or the classify label."""
        h, w = frame.shape[:2]
        s = self._scale(h)
        t = max(1, round(2 * s))
        if task == "segment" and det is not None and det.mask is not None and len(det):
            ov = frame.copy()
            for i in range(len(det)):
                ov[det.mask[i]] = self._palette(i)
            cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
            for i in range(len(det)):
                cnts, _ = cv2.findContours(det.mask[i].astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, cnts, -1, self._palette(i), t, cv2.LINE_AA)
        elif task == "pose" and det is not None and det.data and "kpts" in det.data:
            for kp in det.data["kpts"]:
                for a, b in _SKELETON:
                    if kp[a][2] > 0.3 and kp[b][2] > 0.3:
                        cv2.line(frame, (int(kp[a][0]), int(kp[a][1])),
                                 (int(kp[b][0]), int(kp[b][1])), (90, 255, 90), t, cv2.LINE_AA)
                for x, y, cf in kp:
                    if cf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), max(2, round(3 * s)),
                                   (60, 220, 250), -1, cv2.LINE_AA)
        elif task == "obb" and det is not None and det.data and "xyxyxyxy" in det.data:
            names = det.data.get("class_name")
            for i, corners in enumerate(det.data["xyxyxyxy"]):
                pts = np.asarray(corners, np.int32)
                cv2.polylines(frame, [pts], True, self._palette(i), t, cv2.LINE_AA)
                if names is not None:
                    self._text(frame, str(names[i]), tuple(pts[0]), 0.5 * s,
                               self._palette(i), max(1, round(1.3 * s)))
        elif task == "classify" and cls_label:
            self._text(frame, f"{cls_label[0]}  {cls_label[1]:.2f}",
                       (int(12 * s), int(46 * s)), 0.9 * s, (60, 220, 250),
                       max(2, round(2 * s)))
        return frame

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
            pts = [(int(x * w), int(y * h)) for x, y in ln.vertices]
            cv2.polylines(frame, [np.array(pts, np.int32)], False, (0, 255, 0),
                          t, cv2.LINE_AA)
            for a, b in zip(pts[:-1], pts[1:]):       # +dir arrow per segment
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                nx, ny = dy / L, -dx / L
                tip = (int(mx + nx * 36 * s), int(my + ny * 36 * s))
                cv2.arrowedLine(frame, (int(mx), int(my)), tip, (0, 255, 0),
                                t, cv2.LINE_AA, tipLength=0.4)
            a = pts[0]
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
