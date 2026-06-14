"""Pipeline: source → detect-every-N → track-every-frame → render.

Detect/skip cadence is the main real-time lever: the detector runs every
`runtime.detect_every` frames; on the in-between frames the tracker updates with
empty detections so its Kalman filter predicts positions, keeping IDs and overlays
smooth at full frame rate for a fraction of the detector cost.
"""

from __future__ import annotations

import time

import cv2
import supervision as sv

from .analytics import AnalyticsSink
from .annotations import AnnotationSet
from .config import Config
from .detectors import build_detector
from .emit import ResultWriter, build_result
from .geometry import GeometryEngine
from .render import Renderer
from .sources import open_source
from .speed import SpeedEstimator
from .tracking import Tracker


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.source = open_source(cfg)
        self.detector = build_detector(cfg)
        tcfg = cfg.section("tracker")
        # default frame_rate param to the source fps if unset
        params = dict(tcfg.get("params", {}))
        params.setdefault("frame_rate", round(self.source.fps))
        self.tracker = Tracker(tcfg.get("algorithm", "bytetrack"), params)
        self.renderer = Renderer(cfg)
        self.detect_every = max(1, int(cfg.get("runtime.detect_every", 1)))
        self.max_fps = float(cfg.get("runtime.max_fps", 0) or 0)

        # Phase 2: annotations + geometry
        ann_path = cfg.get("annotations")
        self.annotations = (AnnotationSet.load(ann_path) if ann_path
                            else AnnotationSet())
        self.geometry = GeometryEngine(
            self.annotations, min_dwell=float(cfg.get("geometry.min_dwell", 1.0)))
        self.speed = SpeedEstimator.from_config(cfg)   # None unless calibrated

    def _build_analytics(self, csv_path, json_path) -> AnalyticsSink | None:
        a = self.cfg.section("analytics")
        csv_path = csv_path or a.get("csv")
        json_path = json_path or a.get("json")
        if not (a.get("enabled") or csv_path or json_path):
            return None
        return AnalyticsSink(
            interval_seconds=float(a.get("interval_seconds", 900)),
            csv_path=csv_path, json_path=json_path)

    def run(self, show: bool | None = None, out_path: str | None = None,
            emit_proto: str | None = None, analytics_csv: str | None = None,
            analytics_json: str | None = None, max_frames: int | None = None):
        show = self.cfg.get("render.show", True) if show is None else show
        writer = None
        pb_writer = ResultWriter(emit_proto) if emit_proto else None
        analytics = self._build_analytics(analytics_csv, analytics_json)
        win = "occupancy"
        ema_fps = None
        n = 0
        last_tracked = sv.Detections.empty()
        fps = self.source.fps or 30.0
        t_wall0 = time.perf_counter()
        min_dt = 1.0 / self.max_fps if self.max_fps else 0.0

        try:
            for frame in self.source.frames():
                t0 = time.perf_counter()
                h, w = frame.shape[:2]
                # timestamp: media time for files, wall-clock for live sources
                t = (time.time() if getattr(self.source, "is_live", False)
                     else n / fps)

                # detect every N frames; on skip frames HOLD the last tracked result
                # (feeding empty detections would make ByteTrack drop every track).
                if n % self.detect_every == 0:
                    tracked = self.tracker.update(self.detector.detect(frame), frame)
                    last_tracked = tracked
                else:
                    tracked = last_tracked if n else sv.Detections.empty()

                geo = self.geometry.update(tracked, w, h, t)
                if pb_writer is not None:
                    pb_writer.write(build_result(
                        tracked, geo, self.annotations, w, h, t, pts=n))
                if analytics is not None:
                    analytics.update(t, geo)
                speeds = (self.speed.update(tracked, w, h, t)
                          if self.speed is not None else None)

                vis = self.renderer.draw(frame, tracked, speeds=speeds)
                vis = self.renderer.draw_annotations(vis, self.annotations, geo)
                vis = self.renderer.draw_counts(vis, geo)
                # true wall-clock throughput (honest under detect_every>1)
                ema_fps = (n + 1) / max(time.perf_counter() - t_wall0, 1e-6)
                n_tracks = 0 if tracked.tracker_id is None else len(tracked)
                self.renderer.hud(
                    vis, f"{ema_fps:4.1f} fps | tracks {n_tracks} | "
                         f"det every {self.detect_every} | {self.detector.name}")

                if out_path:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(
                            out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                            self.source.fps, (w, h))
                    writer.write(vis)
                if show:
                    cv2.imshow(win, vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                n += 1
                if max_frames and n >= max_frames:
                    break
                if min_dt:
                    sleep = min_dt - (time.perf_counter() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
        finally:
            self.source.release()
            if writer is not None:
                writer.release()
            if pb_writer is not None:
                pb_writer.close()
            if analytics is not None:
                analytics.close()
            if show:
                cv2.destroyAllWindows()
        return {"frames": n, "fps": ema_fps}
