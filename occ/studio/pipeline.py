"""StudioPipeline — the live CV loop the agent drives.

Self-contained (does NOT import occ.web, which would auto-start a server). Runs
source -> detect -> track -> geometry -> render, then executes the agent's
dynamic OVERLAYS, meters compute into the ledger, and folds live stats into the
Vision Brain. Live-reconfigurable; exposes the latest clean + annotated frames
and tracked detections so tools (analyze_scene, snapshot, ffmpeg) can act.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import supervision as sv

from ..annotations import Annotation, AnnotationSet
from ..config import Config, _set_dotted
from ..detectors import build_detector
from ..geometry import GeometryEngine
from ..render import Renderer
from ..sources import open_source
from ..tracking import Tracker

_METER_EVERY = 60          # frames between compute-point meterings
_OBSERVE_EVERY = 90        # frames between brain observations


class StudioPipeline:
    def __init__(self, cfg: Config, overlays=None, ledger=None, brain=None,
                 settings=None):
        self.cfg = cfg
        self.overlays = overlays
        self.ledger = ledger
        self.brain = brain
        self.settings = settings
        self.annotations = AnnotationSet()
        self.ann_path: str | None = None
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._frame_vis = None             # latest annotated frame
        self._frame_clean = None           # latest raw frame
        self._tracked = sv.Detections.empty()
        self._geo = None
        self._t = 0.0                      # geo time of the latest frame (for proto)
        self._stats = {"full_frame": {}, "lines": {}, "zones": {}, "tracks": 0}
        self._fps = 0.0
        self._status = "idle"
        self._ann_version = 0
        self._run = threading.Event()
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._thread: threading.Thread | None = None
        self._rec = False
        self._rec_name: str | None = None
        self._rec_frames = 0
        self._writer = None

    # ---- annotations ----
    def set_annotations(self, items: list[dict]):
        anns = [Annotation(id=i.get("id") or f"a{n}", type=i["type"],
                           vertices=[tuple(v) for v in i["vertices"]],
                           display_name=i.get("display_name", ""))
                for n, i in enumerate(items)]
        with self._lock:
            self.annotations = AnnotationSet(annotations=anns)
            self._ann_version += 1

    def add_annotation(self, item: dict):
        cur = self.annotation_dicts()
        cur.append(item)
        self.set_annotations(cur)

    def load_annotations(self, path: str):
        s = AnnotationSet.load(path)
        with self._lock:
            self.annotations = s
            self._ann_version += 1

    def annotation_dicts(self) -> list[dict]:
        with self._lock:
            return [{"id": a.id, "type": a.type, "display_name": a.display_name,
                     "vertices": [[x, y] for x, y in a.vertices]}
                    for a in self.annotations.annotations]

    def save_annotations(self, path: str | None = None) -> str:
        p = path or self.ann_path or "configs/studio_annotations.json"
        with self._lock:
            self.annotations.source = str(self.cfg.get("source.uri", ""))
            self.annotations.save(p)
        return p

    # ---- config ----
    def reconfigure(self, updates: dict):
        with self._lock:
            for k, v in updates.items():
                _set_dotted(self.cfg.data, k, v)
            self.cfg._apply_source_overrides(set(updates))
            self._status = "loading"
        self._dirty.set()

    def set_render_flags(self, **flags) -> dict:
        """Toggle renderer draw flags LIVE (no rebuild/flash) — the Renderer reads the
        same render dict each frame."""
        with self._lock:
            r = self.cfg.data.setdefault("render", {})
            for k, v in flags.items():
                r[k] = bool(v)
            return {k: r.get(k) for k in flags}

    def current_config(self) -> dict:
        g = self.cfg.get
        return {"source.uri": g("source.uri"), "detector.backend": g("detector.backend"),
                "detector.model": g("detector.model"),
                "detector.rfdetr_checkpoint": g("detector.rfdetr_checkpoint"),
                "detector.owlv2_prompt": g("detector.owlv2_prompt"),
                "tracker.algorithm": g("tracker.algorithm"),
                "detector.conf": g("detector.conf"), "detector.imgsz": g("detector.imgsz"),
                "runtime.detect_every": g("runtime.detect_every"),
                "detector.classes": g("detector.classes")}

    # ---- lifecycle ----
    def start(self):
        self._run.set()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._run.clear()

    def shutdown(self):
        self._stop.set()

    # ---- recording ----
    def start_recording(self) -> str:
        name = "rec_" + time.strftime("%Y%m%d_%H%M%S") + ".mp4"
        with self._lock:
            self._rec, self._rec_name, self._rec_frames = True, name, 0
        return name

    def stop_recording(self):
        with self._lock:
            self._rec = False
        for _ in range(60):
            if self._writer is None:
                break
            time.sleep(0.05)
        with self._lock:
            return self._rec_name, self._rec_frames

    def recording_path(self) -> str | None:
        with self._lock:
            return f"out/recordings/{self._rec_name}" if self._rec_name else None

    def _record_frame(self, vis, fps):
        if self._rec:
            if self._writer is None and self._rec_name:
                os.makedirs("out/recordings", exist_ok=True)
                h, w = vis.shape[:2]
                self._writer = cv2.VideoWriter(
                    f"out/recordings/{self._rec_name}",
                    cv2.VideoWriter_fourcc(*"mp4v"), max(1, round(fps)), (w, h))
            if self._writer is not None and self._writer.isOpened():
                self._writer.write(vis)
                self._rec_frames += 1
        elif self._writer is not None:
            self._writer.release()
            self._writer = None

    # ---- build + loop ----
    def _build(self):
        with self._lock:
            cfg = self.cfg
        src = open_source(cfg)
        det = build_detector(cfg)
        tcfg = cfg.section("tracker")
        trk = Tracker(tcfg.get("algorithm", "bytetrack"),
                      {**tcfg.get("params", {}), "frame_rate": round(src.fps)})
        return src, det, trk, Renderer(cfg), int(cfg.get("runtime.detect_every", 1) or 1)

    def _meter(self, model, window, every, detected):
        if not self.ledger:
            return
        infers = max(1, window // max(every, 1))
        active = self.overlays.active_count() if self.overlays else 0
        self.ledger.record("detect", model=model, units={"inferences": infers},
                           label="live detection", persist=False)
        self.ledger.record("track", units={"frames": window}, persist=False)
        if active:
            self.ledger.record("overlay", units={"overlay_frames": active * window},
                               label=f"{active} overlay(s)", persist=False)
        self.ledger.record("render", units={"frames": window}, persist=True)

    def _loop(self):
        while not self._stop.is_set():
            self._dirty.clear()
            uri = str(self.cfg.get("source.uri", ""))
            try:
                src, det, trk, renderer, every = self._build()
            except Exception as e:
                # never freeze: show a clear card, then retry (a new switch wins via dirty)
                self._set_placeholder("cannot open source", os.path.basename(uri))
                with self._lock:
                    self._status = f"error: {e}"
                for _ in range(10):                     # ~1s, but bail early on a new switch
                    if self._dirty.is_set() or self._stop.is_set():
                        break
                    time.sleep(0.1)
                continue
            fps = src.fps or 30.0
            model = str(self.cfg.get("detector.model", "yolo11m.pt"))
            with self._lock:
                self._status = "running"
            self._set_placeholder("connecting…", os.path.basename(uri))
            geo_engine = GeometryEngine(self.annotations)
            seen_ann = self._ann_version
            last_tracked = sv.Detections.empty()
            t_prev, n, meter_base = None, 0, 0
            for frame in src.frames():
                if self._stop.is_set() or self._dirty.is_set():
                    break
                if not self._run.is_set():
                    time.sleep(0.05)
                    continue
                with self._lock:
                    ann = self.annotations
                    version = self._ann_version
                if version != seen_ann:
                    geo_engine = GeometryEngine(ann)
                    seen_ann = version
                if n % every == 0:
                    last_tracked = trk.update(det.detect(frame), frame)
                tracked = last_tracked
                geo = geo_engine.update(tracked, frame.shape[1], frame.shape[0], n / fps)
                vis = renderer.draw(frame, tracked)
                vis = renderer.draw_annotations(vis, ann, geo)
                if self.overlays:                       # the agent's dynamic overlays
                    self.overlays.run(vis, tracked, geo, n)
                vis = renderer.draw_counts(vis, geo)
                self._record_frame(vis, fps)
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 72])
                now = time.perf_counter()
                inst = 1.0 / max(now - t_prev, 1e-6) if t_prev else self._fps
                t_prev = now
                stats = {
                    "full_frame": dict(geo.full_frame),
                    "lines": {lid: {"positive": sum(d["positive"].values()),
                                    "negative": sum(d["negative"].values())}
                              for lid, d in geo.line_counts.items()},
                    "zones": {zid: sum(c.values()) for zid, c in geo.zone_counts.items()},
                    "dwell": [{"track": tid, "zone": zid, "seconds": round(end - start, 1)}
                              for (tid, zid, start, end) in geo.dwell],
                    "tracks": 0 if tracked.tracker_id is None else len(tracked)}
                with self._lock:
                    if ok:
                        self._jpeg = buf.tobytes()
                    self._frame_vis = vis
                    self._frame_clean = frame
                    self._tracked = tracked
                    self._geo = geo
                    self._t = n / fps
                    self._stats = stats
                    self._fps = 0.9 * self._fps + 0.1 * inst
                if n - meter_base >= _METER_EVERY:
                    self._meter(model, n - meter_base, every,
                                stats["tracks"])
                    meter_base = n
                if self.brain and n % _OBSERVE_EVERY == 0:
                    self.brain.observe({**stats, "fps": round(self._fps, 1)},
                                       source=str(self.cfg.get("source.uri", "")),
                                       resolution=(frame.shape[1], frame.shape[0]),
                                       fps=fps)
                n += 1
            src.release()
            if not (self._dirty.is_set() or self._stop.is_set()):
                time.sleep(0.2)

    def _set_placeholder(self, title: str, sub: str = ""):
        """Render a status card so the canvas shows state, never a frozen frame."""
        img = np.full((360, 640, 3), 22, np.uint8)
        cv2.putText(img, title, (28, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (90, 200, 255), 2, cv2.LINE_AA)
        if sub:
            cv2.putText(img, sub[:54], (28, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (170, 170, 170), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img)
        with self._lock:
            if ok:
                self._jpeg = buf.tobytes()
            self._frame_vis = img

    # ---- reads for tools / server ----
    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def snapshot_clean(self):
        with self._lock:
            return None if self._frame_clean is None else self._frame_clean.copy()

    def snapshot_vis(self):
        with self._lock:
            return None if self._frame_vis is None else self._frame_vis.copy()

    def scene(self) -> dict:
        """Structured snapshot of what's on screen right now (for analyze_scene)."""
        with self._lock:
            det, geo, stats = self._tracked, self._geo, dict(self._stats)
            stats["fps"] = round(self._fps, 1)
            res = (0, 0) if self._frame_vis is None else (
                self._frame_vis.shape[1], self._frame_vis.shape[0])
        tracks = []
        if det is not None and len(det):
            names = (list(det.data["class_name"]) if det.data and "class_name" in det.data
                     else ["object"] * len(det))
            ids = det.tracker_id if det.tracker_id is not None else [-1] * len(det)
            for i in range(len(det)):
                x1, y1, x2, y2 = det.xyxy[i].astype(int)
                tracks.append({"id": int(ids[i]), "cls": names[i],
                               "box": [int(x1), int(y1), int(x2), int(y2)],
                               "anchor": [int((x1 + x2) / 2), int(y2)]})
        return {"resolution": list(res), "fps": stats.get("fps"),
                "counts": stats.get("full_frame", {}), "tracks": stats.get("tracks", 0),
                "zones": stats.get("zones", {}), "lines": stats.get("lines", {}),
                "dwell": stats.get("dwell", []), "objects": tracks[:60]}

    def emit_proto(self, path: str | None = None) -> dict:
        """Write the CURRENT frame's OccupancyCountingPredictionResult (the original
        occupancy-analytics contract) as a length-delimited .pb — replayable."""
        from ..emit import build_result, ResultWriter
        with self._lock:
            det, geo, ann, frame, t = (self._tracked, self._geo, self.annotations,
                                       self._frame_clean, self._t)
        if det is None or geo is None or frame is None:
            return {"status": "error", "error": "no frame yet"}
        h, w = frame.shape[:2]
        res = build_result(det, geo, ann, w, h, t)
        p = path or f"out/studio/exports/result_{time.strftime('%H%M%S')}.pb"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        wr = ResultWriter(p); wr.write(res); wr.close()
        return {"status": "success", "output": p, "boxes": len(res.identified_boxes),
                "lines": len(res.stats.crossing_line_counts),
                "zones": len(res.stats.active_zone_counts),
                "dwell": len(res.dwell_time_info), "tracks": len(res.track_info)}

    def stats(self) -> dict:
        with self._lock:
            return {**self._stats, "fps": round(self._fps, 1), "status": self._status,
                    "recording": self._rec, "rec_frames": self._rec_frames,
                    "rec_name": self._rec_name}

    # ---- still-image detection (for analyze_image on a shared photo) ----
    def detect_still(self, frame):
        """Run the configured detector on ONE still image; return (det, annotated)."""
        from collections import Counter
        with self._lock:
            cfg = self.cfg
        key = (cfg.get("detector.backend"), cfg.get("detector.model"),
               cfg.get("detector.owlv2_prompt"))
        if getattr(self, "_still_det", None) is None or getattr(self, "_still_key", None) != key:
            self._still_det = build_detector(cfg)
            self._still_key = key
        det = self._still_det.detect(frame)
        vis = Renderer(cfg).draw(frame.copy(), det)
        names = (list(det.data["class_name"]) if det.data and "class_name" in det.data
                 else ["object"] * len(det))
        return det, vis, dict(Counter(names))
