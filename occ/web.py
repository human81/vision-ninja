"""Browser UI over the pipeline — rich, interactive control panel.

Live-reconfigurable: switch source / detector / model / tracker / conf / classes /
frame-skip from the browser and the worker rebuilds without a restart. Draw zones &
lines on the canvas, see live per-class / per-line / per-zone counts, Save to JSON.

  GET  /                 the control-panel page
  GET  /options          available sources/detectors/models/trackers/classes
  GET  /config           current settings        POST /config   apply settings (dotted keys)
  GET  /stream.mjpg      annotated MJPEG          GET  /stats    live counts + fps + status
  GET  /annotations      current shapes           POST /annotations   replace shapes
  POST /save             persist shapes to JSON   POST /start /stop    worker control

Run:  OCC_SOURCE=assets/videos/vehicles-2.mp4 uvicorn occ.web:app --port 8000
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
import supervision as sv
from fastapi import FastAPI, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

from .annotations import Annotation, AnnotationSet
from .config import Config, _set_dotted
from .detectors import build_detector
from .geometry import GeometryEngine
from .render import Renderer
from .sources import open_source
from .tracking import Tracker

_UI = (Path(__file__).resolve().parent / "web_ui.html")

MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
RFDETR = ["Roboflow/rf-detr-nano", "Roboflow/rf-detr-small",
          "Roboflow/rf-detr-medium", "Roboflow/rf-detr-base"]
TRACKERS = ["bytetrack", "botsort", "ocsort", "sort"]
CLASS_OPTS = ["person", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle"]


class WebPipeline:
    """Runs source→detect→track→geometry→render in a thread; live-reconfigurable."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.annotations = AnnotationSet()
        self.ann_path: str | None = None
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stats: dict = {"full_frame": {}, "lines": {}, "zones": {}, "tracks": 0}
        self._fps = 0.0
        self._status = "idle"
        self._ann_version = 0
        self._run = threading.Event()
        self._stop = threading.Event()
        self._dirty = threading.Event()       # config changed → rebuild
        self._thread: threading.Thread | None = None
        self._rec = False                     # recording the annotated stream?
        self._rec_name: str | None = None
        self._rec_frames = 0
        self._writer = None

    # ---- annotations ----
    def set_annotations(self, items: list[dict]):
        anns = [Annotation(id=i.get("id") or f"a{n}", type=i["type"],
                           vertices=[tuple(v) for v in i["vertices"]],
                           display_name=i.get("display_name", "")) for n, i in enumerate(items)]
        with self._lock:
            self.annotations = AnnotationSet(annotations=anns)
            self._ann_version += 1

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
        p = path or self.ann_path or "configs/annotations.json"
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

    def current_config(self) -> dict:
        g = self.cfg.get
        return {"source.uri": g("source.uri"),
                "detector.backend": g("detector.backend"),
                "detector.model": g("detector.model"),
                "detector.rfdetr_checkpoint": g("detector.rfdetr_checkpoint"),
                "tracker.algorithm": g("tracker.algorithm"),
                "detector.conf": g("detector.conf"),
                "detector.imgsz": g("detector.imgsz"),
                "runtime.detect_every": g("runtime.detect_every"),
                "detector.classes": g("detector.classes")}

    # ---- worker ----
    def start(self):
        self._run.set()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._run.clear()

    def shutdown(self):
        self._stop.set()

    # ---- recording the annotated stream ----
    def start_recording(self) -> str:
        name = "rec_" + time.strftime("%Y%m%d_%H%M%S") + ".mp4"
        with self._lock:
            self._rec, self._rec_name, self._rec_frames = True, name, 0
        return name

    def stop_recording(self):
        with self._lock:
            self._rec = False
        for _ in range(60):                 # wait for the loop to flush + close
            if self._writer is None:
                break
            time.sleep(0.05)
        with self._lock:
            return self._rec_name, self._rec_frames

    def _record_frame(self, vis, fps):
        """Called inside the loop thread only — all VideoWriter ops live here."""
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

    def _build(self):
        with self._lock:
            cfg = self.cfg
        src = open_source(cfg)
        det = build_detector(cfg)
        tcfg = cfg.section("tracker")
        trk = Tracker(tcfg.get("algorithm", "bytetrack"),
                      {**tcfg.get("params", {}), "frame_rate": round(src.fps)})
        return src, det, trk, Renderer(cfg), int(cfg.get("runtime.detect_every", 1) or 1)

    def _loop(self):
        while not self._stop.is_set():
            self._dirty.clear()
            try:
                src, det, trk, renderer, every = self._build()
            except Exception as e:
                with self._lock:
                    self._status = f"error: {e}"
                time.sleep(1.0)
                continue
            fps = src.fps or 30.0
            with self._lock:
                self._status = "running"
            geo_engine = GeometryEngine(self.annotations)
            seen_ann = self._ann_version
            last_tracked = sv.Detections.empty()
            t_prev = None
            n = 0
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
                vis = renderer.draw_counts(vis, geo)
                self._record_frame(vis, fps)
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                now = time.perf_counter()
                inst = 1.0 / max(now - t_prev, 1e-6) if t_prev else self._fps
                t_prev = now
                stats = {
                    "full_frame": dict(geo.full_frame),
                    "lines": {lid: {"positive": sum(d["positive"].values()),
                                    "negative": sum(d["negative"].values())}
                              for lid, d in geo.line_counts.items()},
                    "zones": {zid: sum(c.values()) for zid, c in geo.zone_counts.items()},
                    "tracks": 0 if tracked.tracker_id is None else len(tracked)}
                with self._lock:
                    if ok:
                        self._jpeg = buf.tobytes()
                    self._stats = stats
                    self._fps = 0.9 * self._fps + 0.1 * inst
                n += 1
            src.release()
            if not (self._dirty.is_set() or self._stop.is_set()):
                time.sleep(0.2)

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def stats(self) -> dict:
        with self._lock:
            return {**self._stats, "fps": round(self._fps, 1), "status": self._status,
                    "recording": self._rec, "rec_frames": self._rec_frames,
                    "rec_name": self._rec_name}


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config.load(overrides=[
        f"source.uri={os.environ.get('OCC_SOURCE', 'assets/videos/vehicles-2.mp4')}",
        "source.loop=true"])
    pipe = WebPipeline(cfg)
    ann_path = (os.environ.get("OCC_ANNOTATIONS") or cfg.get("annotations")
                or "configs/highway_lines.json")
    pipe.ann_path = ann_path
    if ann_path and Path(ann_path).exists():
        pipe.load_annotations(ann_path)
    pipe.start()
    sources = sorted(str(p) for p in Path("assets/videos").glob("*.mp4"))

    app = FastAPI()
    app.state.pipe = pipe

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _UI.read_text()

    @app.get("/options")
    def options():
        return {"sources": sources, "detectors": ["yolo", "rfdetr"],
                "models": MODELS, "rfdetr_checkpoints": RFDETR,
                "trackers": TRACKERS, "classes": CLASS_OPTS}

    @app.get("/config")
    def get_config():
        return JSONResponse(pipe.current_config())

    @app.post("/config")
    async def set_config(req: Request):
        pipe.reconfigure(await req.json())
        return JSONResponse(pipe.current_config())

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while not pipe._stop.is_set():
                j = pipe.latest_jpeg()
                if j:
                    yield boundary + j + b"\r\n"
                time.sleep(0.05)
        return StreamingResponse(gen(),
            media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stats")
    def stats():
        return JSONResponse(pipe.stats())

    @app.get("/annotations")
    def get_annotations():
        return JSONResponse(pipe.annotation_dicts())

    @app.post("/annotations")
    async def annotations(req: Request):
        pipe.set_annotations(await req.json())
        return {"ok": True, "count": len(pipe.annotations.annotations)}

    @app.post("/save")
    def save():
        return {"saved": pipe.save_annotations(), "count": len(pipe.annotation_dicts())}

    @app.post("/record/start")
    def rec_start():
        return {"recording": True, "file": pipe.start_recording()}

    @app.post("/record/stop")
    def rec_stop():
        name, n = pipe.stop_recording()
        return {"recording": False, "file": name, "frames": n}

    @app.get("/download/{name}")
    def download(name: str):
        p = Path("out/recordings") / Path(name).name      # prevent path traversal
        if not p.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(p), media_type="video/mp4", filename=p.name)

    @app.post("/start")
    def start():
        pipe.start()
        return {"running": True}

    @app.post("/stop")
    def stop():
        pipe.stop()
        return {"running": False}

    return app


app = create_app()
