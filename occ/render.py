"""Overlay renderer. Draws boxes, per-track labels, and motion trails on the
frame. All elements toggle from the `render` config section. Zones/lines/count-HUD
hooks land in Phase 2.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


class Renderer:
    def __init__(self, cfg):
        r = cfg.section("render")
        self.cfg = r
        thickness = int(r.get("thickness", 2))
        self.box = sv.BoxAnnotator(thickness=thickness)
        self.label = sv.LabelAnnotator(
            text_scale=float(r.get("font_scale", 0.5)),
            text_thickness=1,
            text_padding=3,
        )
        self.trace = sv.TraceAnnotator(
            trace_length=int(r.get("trail_length", 30)),
            thickness=thickness,
        )

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
                parts.append(f"{speeds[int(tid)]:.0f}")
            out.append(" ".join(parts))
        return out

    def draw(self, frame: np.ndarray, det: sv.Detections, speeds=None) -> np.ndarray:
        out = frame.copy()
        if len(det) == 0:
            return out
        if self.cfg.get("draw_trails", True) and det.tracker_id is not None:
            out = self.trace.annotate(out, det)
        if self.cfg.get("draw_boxes", True):
            out = self.box.annotate(out, det)
        if self.cfg.get("draw_labels", True):
            out = self.label.annotate(out, det, labels=self._labels(det, speeds))
        return out

    # ---- Phase 2: zones / lines / counts ----
    def draw_annotations(self, frame, ann, geo=None):
        if not self.cfg.get("draw_zones", True):
            return frame
        h, w = frame.shape[:2]
        for z in ann.zones():
            pts = np.array([[int(x * w), int(y * h)] for x, y in z.vertices], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 255))
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 180, 255), 2, cv2.LINE_AA)
            name = z.display_name or z.id
            if geo is not None:
                tot = sum(geo.zone_counts.get(z.id, {}).values())
                name = f"{name}: {tot}"
            cv2.putText(frame, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 180, 255), 2, cv2.LINE_AA)
        for ln in ann.lines():
            (x0, y0), (x1, y1) = ln.vertices
            a = (int(x0 * w), int(y0 * h))
            b = (int(x1 * w), int(y1 * h))
            cv2.line(frame, a, b, (0, 255, 0), 2, cv2.LINE_AA)
            # positive-direction arrow: normal pointing to the side where side()>0
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            # right-hand-rule positive side direction = rotate (dx,dy) so that side>0
            nx, ny = dy / L, -dx / L          # unit normal toward side()>0
            tip = (int(mx + nx * 28), int(my + ny * 28))
            cv2.arrowedLine(frame, (int(mx), int(my)), tip, (0, 255, 0), 2,
                            cv2.LINE_AA, tipLength=0.4)
            label = ln.display_name or ln.id
            if geo is not None and ln.id in geo.line_counts:
                pos = sum(geo.line_counts[ln.id]["positive"].values())
                neg = sum(geo.line_counts[ln.id]["negative"].values())
                label = f"{label}  +{pos}/-{neg}"
            cv2.putText(frame, label, (a[0], a[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def draw_counts(self, frame, geo):
        if not self.cfg.get("draw_counts", True) or geo is None:
            return frame
        lines = ["FULL FRAME:"] + [f"  {k}: {v}" for k, v in geo.full_frame.items()]
        for lid, d in geo.line_counts.items():
            p = sum(d["positive"].values()); n = sum(d["negative"].values())
            lines.append(f"LINE {lid}: +{p} / -{n}")
        y = 50
        for txt in lines:
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
        return frame

    @staticmethod
    def hud(frame: np.ndarray, text: str) -> np.ndarray:
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        return frame
