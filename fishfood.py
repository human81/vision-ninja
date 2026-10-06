#!/usr/bin/env python
"""Fishfood — incremental, exhaustive end-to-end dogfood of the whole pipeline.

Climbs through levels (each a superset of the last), exercising every component on
the real test clips. Never stops on failure — it records every result so you see
total coverage, writes an annotated frame per clip to out/fishfood/, and prints a
pass/fail matrix at the end.

    .venv/bin/python fishfood.py            # all levels
    .venv/bin/python fishfood.py --level 2  # up to level 2 only
    .venv/bin/python fishfood.py --frames 80

Levels:
  1  smoke      imports · proto contract · one clip detect→track→render
  2  per-source detect→track→geometry→emit on ALL clips + gallery frames
  3  matrix     every tracker · analytics CSV/JSON · speed · RF-DETR
  4  exhaustive proto round-trip per clip · detect-every variants · VLM logic ·
                protobuf field-coverage assertions
  5  browser e2e Playwright fully drives the web UI: draws a zone+line, starts,
                asserts live stats (headless Chromium)
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from occ.config import Config
from occ.annotations import AnnotationSet, Annotation, ZONE, LINE
from occ.detectors import build_detector
from occ.detectors.yolo import YoloDetector
from occ.geometry import GeometryEngine
from occ.render import Renderer
from occ.sources import FileSource
from occ.tracking import Tracker
from occ.emit import build_result, ResultWriter
from occ.speed import SpeedEstimator

CLIPS = ["vehicles", "vehicles-2", "people-walking",
         "market-square", "grocery-store", "subway"]
HIGHWAY = ["vehicles", "vehicles-2"]
GALLERY = Path("out/fishfood")
RESULTS: list[dict] = []

# shared, built once (YOLO load is expensive); cached per (imgsz, conf, model)
_CFG = Config.load(overrides=["source.max_long_side=1280"])
_RENDER = Renderer(_CFG)
_DET_CACHE: dict = {}


def detector_for(cfg: Config) -> YoloDetector:
    key = (cfg.get("detector.imgsz"), cfg.get("detector.conf"), cfg.get("detector.model"))
    if key not in _DET_CACHE:
        _DET_CACHE[key] = YoloDetector(cfg)
    return _DET_CACHE[key]


def yolo() -> YoloDetector:
    return detector_for(_CFG)


def auto_annotations() -> AnnotationSet:
    s = AnnotationSet()
    s.annotations.append(Annotation("line1", LINE, [(0.05, 0.80), (0.95, 0.80)], "count"))
    s.annotations.append(Annotation("zone1", ZONE,
                                    [(0.2, 0.6), (0.8, 0.6), (0.8, 0.95), (0.2, 0.95)], "roi"))
    return s


def clip_path(name: str) -> str:
    return f"assets/videos/{name}.mp4"


def record(level: int, name: str, status: str, detail: str = "", dur: float = 0.0,
           cmd: str = ""):
    RESULTS.append({"level": level, "name": name, "status": status,
                    "detail": detail, "dur": dur, "cmd": cmd})
    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "·"}.get(status, "?")
    print(f"  [{level}] {icon} {name:34} {detail}  ({dur:.1f}s)")
    if cmd:
        print(f"        ↳ {cmd}")


def run(level: int, name: str, fn, cmd: str = ""):
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        record(level, name, "PASS", detail, time.perf_counter() - t0, cmd)
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        record(level, name, "FAIL", f"{type(e).__name__}: {e} | {tb}",
               time.perf_counter() - t0, cmd)


def process_clip(name: str, ann: AnnotationSet, n_frames: int, algo: str = "bytetrack",
                 detector=None, speed: SpeedEstimator | None = None,
                 emit_path: str | None = None, detect_every: int = 1,
                 gallery_png: str | None = None) -> dict:
    """Core dogfood loop; returns metrics and optionally writes an annotated frame."""
    # per-clip config so source_overrides (e.g. market-square imgsz) take effect
    ccfg = Config.load(overrides=[f"source.uri={clip_path(name)}", "source.loop=false",
                                  "source.max_long_side=1280"])
    detector = detector or detector_for(ccfg)
    src = FileSource(clip_path(name), loop=False,
                     max_long_side=int(ccfg.get("source.max_long_side") or 0))
    trk = Tracker(algo, {"frame_rate": round(src.fps), "minimum_consecutive_frames": 2})
    geo = GeometryEngine(ann)
    pb = ResultWriter(emit_path) if emit_path else None
    fps = src.fps or 30.0
    n = 0
    unique = set()
    last_vis = None
    t0 = time.perf_counter()
    import supervision as sv
    last_tracked = sv.Detections.empty()
    for frame in src.frames():
        h, w = frame.shape[:2]
        if n % detect_every == 0:
            tracked = trk.update(detector.detect(frame), frame)
            last_tracked = tracked
        else:
            tracked = last_tracked
        g = geo.update(tracked, w, h, n / fps)
        if tracked.tracker_id is not None:
            unique.update(int(t) for t in tracked.tracker_id if int(t) >= 0)
        spd = speed.update(tracked, w, h, n / fps) if speed else None
        if pb:
            pb.write(build_result(tracked, g, ann, w, h, n / fps, pts=n))
        if gallery_png:
            vis = _RENDER.draw(frame, tracked, speeds=spd)
            vis = _RENDER.draw_annotations(vis, ann, g)
            vis = _RENDER.draw_counts(vis, g)
            last_vis = vis
        n += 1
        if n >= n_frames:
            break
    src.release()
    if pb:
        pb.close()
    if gallery_png and last_vis is not None:
        GALLERY.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(gallery_png, last_vis)
    wall = time.perf_counter() - t0
    pos = sum(sum(d["positive"].values()) for d in geo._line_counts.values())
    neg = sum(sum(d["negative"].values()) for d in geo._line_counts.values())
    return {"frames": n, "fps": n / max(wall, 1e-6), "tracks": len(unique),
            "crossings": pos + neg, "speeds": spd}


# ---------------- Level 1: smoke ----------------
def level1(frames):
    run(1, "proto round-trip",
        lambda: __import__("test_proto_roundtrip").build_sample() and "contract OK",
        cmd=".venv/bin/python test_proto_roundtrip.py")
    def smoke():
        m = process_clip("vehicles-2", auto_annotations(), min(frames, 60),
                         gallery_png=str(GALLERY / "_smoke.png"))
        assert m["tracks"] > 0
        return f"{m['fps']:.0f} fps, {m['tracks']} tracks"
    run(1, "detect→track→render (vehicles-2)", smoke,
        cmd="python run.py run --source assets/videos/vehicles-2.mp4 --no-show --max-frames 60")


# ---------------- Level 2: per-source ----------------
def level2(frames):
    for clip in CLIPS:
        def f(clip=clip):
            ann = auto_annotations()
            m = process_clip(clip, ann, frames, emit_path=f"out/fishfood/{clip}.pb",
                             gallery_png=str(GALLERY / f"{clip}.png"))
            assert m["frames"] == frames
            return f"{m['fps']:.0f}fps tracks={m['tracks']} cross={m['crossings']} → gallery"
        run(2, f"source:{clip}", f,
            cmd=f"python run.py run --source assets/videos/{clip}.mp4 "
                f"--annotations configs/highway_lines.json --emit-proto out/{clip}.pb --no-show")


# ---------------- Level 3: matrix ----------------
def level3(frames):
    for algo in ["bytetrack", "botsort", "ocsort", "sort"]:
        def f(algo=algo):
            m = process_clip("vehicles-2", auto_annotations(), frames, algo=algo)
            assert m["tracks"] > 0
            return f"{m['fps']:.0f}fps tracks={m['tracks']}"
        run(3, f"tracker:{algo}", f,
            cmd=f"python run.py run --source assets/videos/vehicles-2.mp4 --tracker {algo} --no-show")

    def analytics():
        from occ.pipeline import Pipeline
        ann = auto_annotations(); ann.save("out/fishfood/_ann.json")
        cfg = Config.load(overrides=[
            "source.uri=assets/videos/vehicles-2.mp4", "source.loop=false",
            "annotations=out/fishfood/_ann.json",
            "analytics.enabled=true", "analytics.interval_seconds=2"])
        Pipeline(cfg).run(show=False, analytics_csv="out/fishfood/counts.csv",
                          analytics_json="out/fishfood/summary.json", max_frames=frames)
        import csv, json
        rows = list(csv.DictReader(open("out/fishfood/counts.csv")))
        metrics = {r["metric"] for r in rows}
        assert {"line_crossing", "zone_occupancy", "full_frame"} <= metrics
        s = json.load(open("out/fishfood/summary.json"))
        return f"{len(rows)} rows, metrics={len(metrics)}, crossings={sum(s['line_crossings'].values())}"
    run(3, "analytics CSV/JSON", analytics,
        cmd="python run.py run --config configs/highway_example.yaml "
            "--analytics-csv out/counts.csv --analytics-json out/summary.json --no-show")

    def speed_check():
        spd = SpeedEstimator([[0.35, 0.62], [0.65, 0.62], [0.95, 0.95], [0.05, 0.95]],
                             [[0, 40], [12, 40], [12, 0], [0, 0]], units="kmh")
        peak = {}
        src = FileSource(clip_path("vehicles-2"), loop=False, max_long_side=1280)
        trk = Tracker("bytetrack", {"frame_rate": round(src.fps)})
        n = 0
        for frame in src.frames():
            h, w = frame.shape[:2]
            d = trk.update(yolo().detect(frame), frame)
            for tid, v in spd.update(d, w, h, n / (src.fps or 30)).items():
                peak[tid] = max(peak.get(tid, 0), v)
            n += 1
            if n >= frames:
                break
        src.release()
        vals = [v for v in peak.values() if v > 1]
        assert vals and all(0 < v < 300 for v in peak.values())
        return f"{len(peak)} tracks, median {np.median(vals):.0f} km/h, max {max(peak.values()):.0f}"
    run(3, "speed estimation", speed_check,
        cmd="python run.py run --config configs/highway_example.yaml --no-show  # calibration→speed")

    def rfdetr():
        cfg = Config.load(overrides=["detector.backend=rfdetr",
                                     "detector.rfdetr_checkpoint=Roboflow/rf-detr-nano",
                                     "detector.conf=0.3", "source.max_long_side=1280"])
        det = build_detector(cfg)
        m = process_clip("vehicles-2", auto_annotations(), min(frames, 40), detector=det)
        assert m["tracks"] >= 0
        return f"device={det.device} {m['fps']:.1f}fps tracks={m['tracks']}"
    run(3, "detector:rfdetr", rfdetr,
        cmd="python run.py run --source assets/videos/vehicles-2.mp4 --detector rfdetr --no-show")


# ---------------- Level 4: exhaustive ----------------
def level4(frames):
    def roundtrip(clip):
        def f():
            path = f"out/fishfood/{clip}.pb"
            if not Path(path).exists():
                process_clip(clip, auto_annotations(), min(frames, 60), emit_path=path)
            data = open(path, "rb").read()
            from proto import OccupancyCountingPredictionResult as R
            i = msgs = 0
            while i < len(data):
                shift = ln = 0
                while True:
                    b = data[i]; i += 1; ln |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                R.FromString(data[i:i + ln]); i += ln; msgs += 1
            assert msgs > 0
            return f"{msgs} protobuf msgs valid"
        run(4, f"proto round-trip:{clip}", f,
            cmd=f"python run.py run --source assets/videos/{clip}.mp4 --emit-proto out/{clip}.pb --no-show")
    for clip in CLIPS:
        roundtrip(clip)

    for every in [2, 3, 5]:
        def f(every=every):
            m = process_clip("vehicles-2", auto_annotations(), frames, detect_every=every)
            return f"detect_every={every}: {m['fps']:.0f}fps tracks={m['tracks']}"
        run(4, f"detect_every:{every}", f,
            cmd=f"python run.py run --source assets/videos/vehicles-2.mp4 --detect-every {every} --no-show")

    def field_coverage():
        # assert the emitted protobuf exercises every major field group
        from proto import OccupancyCountingPredictionResult as R
        ann = auto_annotations()
        path = "out/fishfood/_cov.pb"
        process_clip("vehicles-2", ann, frames, emit_path=path, speed=None)
        data = open(path, "rb").read()
        i = 0; seen = set()
        while i < len(data):
            shift = ln = 0
            while True:
                b = data[i]; i += 1; ln |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            r = R.FromString(data[i:i + ln]); i += ln
            if r.identified_boxes: seen.add("boxes")
            if r.stats.full_frame_count: seen.add("full_frame")
            if r.stats.crossing_line_counts: seen.add("lines")
            if r.stats.active_zone_counts: seen.add("zones")
            if r.track_info: seen.add("track_info")
            if r.dwell_time_info: seen.add("dwell")
        need = {"boxes", "full_frame", "lines", "zones", "track_info"}
        missing = need - seen
        assert not missing, f"missing field groups: {missing}"
        return f"field groups present: {sorted(seen)}"
    run(4, "protobuf field coverage", field_coverage,
        cmd="python run.py run --config configs/highway_example.yaml --emit-proto out/run.pb --no-show")

    def grounding_logic():
        from occ.grounding import (build_grounder, parse_locate_anything_boxes,
                                   suggest_zone_from_boxes)
        b = parse_locate_anything_boxes("<box><100><200><300><400></box>")
        z = suggest_zone_from_boxes(b, name="g")
        g = build_grounder(Config.load(overrides=["grounding.backend=locate_anything"]))
        assert len(b) == 1 and z and g._model is None
        return "parse+suggest+lazy-construct OK (no download)"
    run(4, "VLM grounding logic", grounding_logic,
        cmd="python run.py ground --source assets/videos/vehicles-2.mp4 --prompt 'cars' --suggest-zone roi")


def summary(max_level: int):
    print("\n" + "=" * 64)
    by_level: dict[int, list] = {}
    for r in RESULTS:
        by_level.setdefault(r["level"], []).append(r)
    total_pass = total = 0
    for lvl in sorted(by_level):
        ps = sum(1 for r in by_level[lvl] if r["status"] == "PASS")
        n = len(by_level[lvl])
        total_pass += ps; total += n
        print(f"  level {lvl}: {ps}/{n} passed")
    print("-" * 64)
    fails = [r for r in RESULTS if r["status"] == "FAIL"]
    if fails:
        print("  FAILURES:")
        for r in fails:
            print(f"    ✗ [{r['level']}] {r['name']}: {r['detail']}")
    gallery = sorted(GALLERY.glob("*.png")) if GALLERY.exists() else []
    print(f"  gallery: {len(gallery)} annotated frames in {GALLERY}/")
    print("=" * 64)
    print(f"  {'ALL PASS' if total_pass == total else 'SOME FAILED'} — "
          f"{total_pass}/{total} checks across levels 1–{max_level}")
    return total_pass == total


# ---------------- Level 5: browser e2e (Playwright) ----------------
def level5(frames):
    def e2e():
        import test_e2e_web
        test_e2e_web.main()
        return "browser drew zone+line, started run, stats verified"
    run(5, "e2e browser (Playwright)", e2e,
        cmd=".venv/bin/python test_e2e_web.py")


def level6(frames):
    """Studio AR try-on — face landmarks, every filter, sponsored stylist + shape
    search (test_studio_tryon's OFFLINE phase, no server). SKIPs without [face].
    Also the studio auth gate (test_studio_auth, fake Firebase) and the agent-code
    policy (test_studio_codepolicy) — both always run."""
    try:
        import test_studio_auth
        for name, ok, detail in test_studio_auth.collect():
            record(6, f"studio auth: {name}", "PASS" if ok else "FAIL", detail,
                   cmd=".venv/bin/python test_studio_auth.py")
    except Exception as e:
        record(6, "studio auth", "FAIL", f"{type(e).__name__}: {e}")
    try:
        import test_studio_codepolicy
        for name, ok, detail in test_studio_codepolicy.collect():
            record(6, f"agent-code policy: {name}", "PASS" if ok else "FAIL", detail,
                   cmd=".venv/bin/python test_studio_codepolicy.py")
    except Exception as e:
        record(6, "agent-code policy", "FAIL", f"{type(e).__name__}: {e}")
    try:
        import mediapipe  # noqa: F401  (the [face] extra)
    except Exception:
        record(6, "studio AR try-on", "SKIP", "mediapipe not installed ([face] extra)")
        return
    try:
        import test_studio_tryon as st
        results = st.collect_offline()
    except Exception as e:
        record(6, "studio AR try-on offline", "FAIL", f"{type(e).__name__}: {e}")
        return
    for name, ok, detail in results:
        record(6, name, "PASS" if ok else "FAIL", detail,
               cmd=".venv/bin/python test_studio_tryon.py")


def write_runbook(path="out/fishfood/RUNBOOK.md"):
    """Standalone command to reproduce each check, grouped by level."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Fishfood runbook — reproduce each check standalone\n",
             "Uses all 6 test clips in `assets/videos/`.\n"]
    seen = set()
    for lvl in sorted({r["level"] for r in RESULTS}):
        lines.append(f"\n## Level {lvl}\n")
        for r in RESULTS:
            if r["level"] == lvl and r["cmd"] and r["cmd"] not in seen:
                seen.add(r["cmd"])
                lines.append(f"# {r['name']}  [{r['status']}]")
                lines.append(f"{r['cmd']}\n")
    Path(path).write_text("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6, help="max level to run (1-6)")
    ap.add_argument("--frames", type=int, default=120, help="frames per clip")
    args = ap.parse_args()
    print(f"FISHFOOD — levels 1–{args.level}, {args.frames} frames/clip\n")
    levels = [level1, level2, level3, level4, level5, level6]
    for i, fn in enumerate(levels[:args.level], start=1):
        print(f"── Level {i} ──")
        fn(args.frames)
    ok = summary(args.level)
    print(f"  runbook (commands to run each): {write_runbook()}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
