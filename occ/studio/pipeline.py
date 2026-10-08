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
        self._detect_on = True             # YOLO detect + track + occupancy; OFF when you go live
        from .gestures import GestureBrowser
        self.gestures = GestureBrowser()   # hands-free store browsing (lazy mediapipe)
        # feature-anchored annotation stabilization: pin zones/lines to the SCENE so they
        # re-localize (right place + size) when the camera angle changes.
        from ..stabilize import SceneStabilizer
        self.stabilizer = SceneStabilizer()
        self._stabilize = False
        self._stab_every = 4               # re-register every N frames; hold H between
        self._anchor_path = "out/studio/anchor.jpg"
        self._anchor_meta = "out/studio/anchor.json"
        self._cam_sim = "none"             # simulated camera move (flip/rotate) to test redraw
        self._cam_sim_deg = 0.0
        self._min_dwell = 1.0              # default dwell threshold (s) for zones w/o their own
        self._dwell_reset = False          # one-shot flag: clear all running dwell timers
        self._geo_engine = None            # live ref so dwell settings apply without a rebuild

    def set_dwell_default(self, seconds: float) -> dict:
        """Set the DEFAULT dwell threshold (seconds) — used by any zone without its own."""
        self._min_dwell = max(0.0, float(seconds or 0))
        if self._geo_engine is not None:
            self._geo_engine.min_dwell = self._min_dwell
        return {"default_dwell": self._min_dwell}

    def reset_dwell(self) -> dict:
        """Clear all running dwell timers — everyone's counter restarts from zero."""
        self._dwell_reset = True
        return {"ok": True, "default_dwell": self._min_dwell}

    def set_camera_sim(self, mode: str = "none", degrees: float = 0.0) -> dict:
        """Simulate a camera move on the live stream (flip ↕/↔, rotate 180, or rotate N°) so
        you can SEE zones/lines redraw to the right place. modes: none|flipv|fliph|rotate180|
        rotate (with `degrees`)."""
        self._cam_sim = (mode or "none").lower()
        self._cam_sim_deg = float(degrees or 0.0)
        return {"camera_sim": self._cam_sim, "degrees": self._cam_sim_deg}

    # ---- annotation stabilization (feature anchoring) ----
    def anchor_scene(self, detector: str | None = None) -> dict:
        """Snapshot the CURRENT view as the reference the zones/lines are pinned to. Call
        after drawing (or re-drawing) annotations so they track the camera from here on."""
        if detector:
            self.stabilizer.set_detector(detector)
        with self._lock:
            frame = None if self._frame_clean is None else self._frame_clean.copy()
            src = str(self.cfg.get("source.uri", ""))
        if frame is None:
            return {"ok": False, "error": "no live frame yet"}
        ok = self.stabilizer.anchor(frame)
        if ok:
            try:
                import os
                import json as _json
                os.makedirs("out/studio", exist_ok=True)
                cv2.imwrite(self._anchor_path, frame)
                with open(self._anchor_meta, "w") as f:
                    _json.dump({"source": src, "detector": self.stabilizer.detector,
                                "size": list(self.stabilizer.ref["size"])}, f)
            except Exception:
                pass
        return {"ok": ok, **self.stabilizer.status()}

    def set_stabilize(self, on: bool, detector: str | None = None) -> dict:
        """Turn feature anchoring on/off. Turning ON anchors to the current view (reusing a
        saved anchor for the same source if present), so existing zones stay pinned."""
        if detector:
            self.stabilizer.set_detector(detector)
        if on and not self.stabilizer.anchored():
            if not self._load_anchor():
                self.anchor_scene()
        self._stabilize = bool(on)
        return {"enabled": self._stabilize, **self.stabilizer.status()}

    def _load_anchor(self) -> bool:
        """Re-anchor from the saved reference image IF it belongs to the current source."""
        try:
            import json as _json
            import os
            if not (os.path.exists(self._anchor_path) and os.path.exists(self._anchor_meta)):
                return False
            meta = _json.load(open(self._anchor_meta))
            if meta.get("source") != str(self.cfg.get("source.uri", "")):
                return False
            img = cv2.imread(self._anchor_path)
            if img is None:
                return False
            self.stabilizer.set_detector(meta.get("detector", "orb"))
            return self.stabilizer.anchor(img)
        except Exception:
            return False

    def tracking_status(self) -> dict:
        return {"enabled": self._stabilize, **self.stabilizer.status()}

    # ---- annotations ----
    def set_annotations(self, items: list[dict]):
        anns = [Annotation(id=i.get("id") or f"a{n}", type=i["type"],
                           vertices=[tuple(v) for v in i["vertices"]],
                           display_name=i.get("display_name", ""),
                           dwell=float(i.get("dwell", 0) or 0))
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
                     "dwell": a.dwell, "vertices": [[x, y] for x, y in a.vertices]}
                    for a in self.annotations.annotations]

    def save_annotations(self, path: str | None = None) -> str:
        p = path or self.ann_path or "configs/studio_annotations.json"
        with self._lock:
            self.annotations.source = str(self.cfg.get("source.uri", ""))
            self.annotations.save(p)
        return p

    # ---- config ----
    def reconfigure(self, updates: dict):
        new_uri = str(updates.get("source.uri", ""))
        if new_uri and not new_uri.startswith("push:") and getattr(self, "_prev_source", None):
            # Leaving camera mode for a real scene: camera mode switched analysis OFF (it's a
            # selfie feed) — restore what was on before, or the scene plays with no detections.
            self._detect_on = getattr(self, "_prev_detect", True)
            self._prev_source = None
        with self._lock:
            for k, v in updates.items():
                _set_dotted(self.cfg.data, k, v)
            self.cfg._apply_source_overrides(set(updates))
            self._status = "loading"
        self._dirty.set()
        self._interrupt_source()

    def _interrupt_source(self):
        """The loop only sees `_dirty` between frames. A source that waits for frames without
        yielding (camera push after the browser stops sending; a stalled/underrun live stream)
        would hold it forever — the switch was saved but never applied. Wake it."""
        src = getattr(self, "_src", None)
        if src is not None and hasattr(src, "interrupt"):
            try:
                src.interrupt()
            except Exception:
                pass

    def set_render_flags(self, **flags) -> dict:
        """Toggle renderer draw flags LIVE (no rebuild/flash) — the Renderer reads the
        same render dict each frame."""
        with self._lock:
            r = self.cfg.data.setdefault("render", {})
            for k, v in flags.items():
                r[k] = bool(v)
            return {k: r.get(k) for k in flags}

    # ---- BIDI live: browser-pushed webcam/screen frames ----
    def push_camera_frame(self, jpeg: bytes) -> bool:
        """Feed one JPEG frame from the browser-capture websocket into the live
        push source. The normal _loop then detects/tracks/overlays it and it
        republishes to /stream.mjpg — so the agent's overlays land on *you*."""
        import numpy as np
        from ..sources import push_frame
        arr = np.frombuffer(jpeg, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        push_frame(frame)
        return True

    def set_detection(self, on: bool) -> bool:
        """Turn the YOLO detect + track + occupancy analysis ON/OFF live. OFF = the
        feed passes through clean (face filters/overlays still run; no boxes, no
        counts, no compute). Returns the new state."""
        self._detect_on = bool(on)
        return self._detect_on

    def set_gesture_browse(self, on=None, store=None) -> dict:
        """Toggle hands-free gesture browsing of the sponsored stores. When on, a
        product carousel is drawn on the live video and your hand drives it (move to
        browse, fist to try on, V to switch store, thumbs-down to clear)."""
        st = self.gestures.toggle(on=on, store=store)
        if self.gestures.active:
            self._detect_on = False          # browsing is a clean-feed experience
            self.set_render_flags(draw_boxes=False, draw_labels=False, draw_counts=False)
        return st

    def set_gesture_current(self, img="", store=None) -> dict:
        """Move the browse cursor to a clicked item so the rails + strip stay in sync."""
        return self.gestures.set_current(img, store)

    def step_gesture(self, d: int) -> dict:
        """Move the browse cursor by ±N (keyboard / UI Prev-Next), returning the new state."""
        self.gestures.step(int(d))
        return self.gestures.state()

    def start_camera(self) -> str:
        """Switch the live source to the browser push feed, remembering the prior
        source so stop_camera() can restore it. Detection/tracking/occupancy default
        OFF when you go live (it's a selfie feed, not a scene to analyze). Returns
        the prior source.uri."""
        from ..sources import PUSH
        prev = str(self.cfg.get("source.uri", ""))
        with self._lock:
            self._prev_source = getattr(self, "_prev_source", None) or prev
            self._prev_detect = self._detect_on
        self._detect_on = False                     # go live = analysis off by default
        PUSH.clear()
        if not prev.startswith("push:"):
            self.reconfigure({"source.uri": "push://camera"})
        return self._prev_source

    def stop_camera(self):
        """Restore the source (and detection state) from before BIDI camera capture."""
        prev = getattr(self, "_prev_source", None)
        from ..sources import PUSH
        PUSH.clear()
        self._detect_on = getattr(self, "_prev_detect", True)
        if prev and not str(prev).startswith("push:"):
            self.reconfigure({"source.uri": prev})
        self._prev_source = None

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
        self._interrupt_source()

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
                self._src = src
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
            geo_engine = GeometryEngine(self.annotations, min_dwell=self._min_dwell)
            self._geo_engine = geo_engine
            seen_ann = self._ann_version
            last_tracked = sv.Detections.empty()
            last_raw = sv.Detections.empty()
            task = getattr(det, "task", "detect")          # detect|segment|pose|obb|classify
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
                    geo_engine = GeometryEngine(ann, min_dwell=self._min_dwell)
                    self._geo_engine = geo_engine
                    seen_ann = version
                # simulated camera move (flip/rotate the stream) — transform the frame BEFORE
                # detection so boxes/tracking align with the flipped/rotated view, and redraw
                # the zones/lines by the SAME exact transform: a ground-truth demo of
                # re-localization. (The feature stabilizer below handles REAL camera moves.)
                ann_draw = ann
                if self._cam_sim != "none":
                    from ..stabilize import sim_transform, warp_annset
                    Hh, Ww = frame.shape[0], frame.shape[1]
                    T = sim_transform(self._cam_sim, Ww, Hh, self._cam_sim_deg)
                    if T is not None:
                        frame = cv2.warpPerspective(frame, T, (Ww, Hh))
                        ann_draw = warp_annset(ann, T, Ww, Hh, Ww, Hh)
                detect_on = self._detect_on
                if not detect_on:             # going live: no YOLO detect/track/occupancy
                    last_raw = sv.Detections.empty()
                    last_tracked = sv.Detections.empty()
                elif n % every == 0:
                    last_raw = det.detect(frame)
                    if task != "classify":
                        last_tracked = trk.update(last_raw, frame)
                tracked = (last_tracked if (detect_on and task != "classify")
                           else sv.Detections.empty())
                # feature-anchored stabilization: re-register against the reference (every
                # _stab_every frames; H holds between) and warp the zones/lines into the
                # current view, so BOTH counting and drawing use the camera-corrected geometry.
                if self._stabilize and self.stabilizer.anchored():
                    if n % self._stab_every == 0:
                        self.stabilizer.register(frame)
                    if not self.stabilizer.lost:
                        ann_draw = self.stabilizer.warp(ann, frame.shape[1], frame.shape[0])
                geo_engine.ann = ann_draw          # same ids → per-track crossing state persists
                if self._dwell_reset:                       # one-shot: clear running dwell timers
                    geo_engine.reset_dwell()
                    self._dwell_reset = False
                geo = geo_engine.update(tracked, frame.shape[1], frame.shape[0], n / fps)
                vis = renderer.draw(frame, tracked)
                if detect_on and task != "detect":   # masks / keypoints / obb / classify label
                    vis = renderer.draw_task(vis, last_raw, task,
                                             getattr(det, "cls_label", None))
                vis = renderer.draw_annotations(vis, ann_draw, geo)
                if self.overlays:                       # the agent's dynamic overlays
                    self.overlays.run(vis, tracked, geo, n, raw=last_raw, clean=frame)
                if self.gestures.active:                 # hands-free store browsing
                    try:
                        self.gestures.process(vis, frame)
                    except Exception:
                        pass
                if detect_on:                            # no count overlay on a clean live/voice feed
                    vis = renderer.draw_dwell(vis, geo)  # live dwell-timer rings
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
        """Render a status card so the canvas shows state, never a frozen frame. Match the
        last live frame's dimensions so a card<->video transition never changes the stream's
        aspect ratio (which makes the displayed image jump vertically under object-fit)."""
        with self._lock:
            last = self._frame_vis
        if last is not None and getattr(last, "ndim", 0) == 3 and last.shape[0] > 1:
            h, w = int(last.shape[0]), int(last.shape[1])     # match the live frame
        else:
            h, w = 720, 1280                                  # neutral 16:9 before any frame
        img = np.full((h, w, 3), 22, np.uint8)
        fs = max(0.6, w / 1100.0)
        cv2.putText(img, title, (int(w * 0.05), int(h * 0.47)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (90, 200, 255), max(1, round(fs * 2)), cv2.LINE_AA)
        if sub:
            cv2.putText(img, sub[:54], (int(w * 0.05), int(h * 0.47 + fs * 42)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (170, 170, 170),
                        max(1, round(fs)), cv2.LINE_AA)
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
                    "rec_name": self._rec_name, "detect_on": self._detect_on,
                    "gesture": self.gestures.state()}

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
