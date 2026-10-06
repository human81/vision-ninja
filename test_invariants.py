"""Core behavioural invariants — offline, deterministic, synthetic-input regression.

The clip-driven suites prove the pipeline *runs* (e.g. "crossings > 0"); this file proves it
counts *correctly*, with hand-built `supervision.Detections` whose answers are known exactly:

  1. Line crossing — segment∩segment test, right-hand-rule direction, bottom-center anchor:
     one-way motion counts +1 in exactly one direction; reverse motion counts the other;
     passing the line's INFINITE extension counts nothing; touching / jittering on one side
     never double counts; unconfirmed tracks and teleports are ignored.
  2. GeometryEngine must persist across frames (crossing needs the previous anchor).
  3. Normalized (0..1) annotations are resolution-independent: the same scene at 640x360,
     1920x1080 and 3840x2160 gives identical zone / line / dwell results.
  4. Dwell (`_DwellTracker`): accrues while inside; survives tracker-id changes, id swaps
     between two people and short occlusions (< grace_lost); restarts after a long absence;
     edge hysteresis (enter band / leave band); per-zone thresholds.
  5. detect_every>1 HOLDS the last tracked result on skip frames, in all three loops
     (occ/pipeline.py, occ/web.py, occ/studio/pipeline.py), driven with stub
     source/detector/tracker/renderer — plus the reason: real ByteTrack fed an empty frame
     returns zero tracks.
  6. Config `source_overrides`: substring match applies; an explicit override wins.

No video, models, network, GPU or servers. Global state touched is restored.

    .venv/bin/python test_invariants.py
"""

from __future__ import annotations

import time
import types
import warnings
from pathlib import Path

import numpy as np
import supervision as sv

from occ.annotations import LINE, ZONE, Annotation, AnnotationSet
from occ.config import Config, _set_dotted
from occ.geometry import GeometryEngine, _side

ROOT = Path(__file__).resolve().parent
W = H = 1000                                 # default synthetic frame (px)


# ---------------------------------------------------------------- synthetic inputs
def person(ax, ay, tid, w=40, h=80, cls="person"):
    """One box whose BOTTOM-CENTER anchor is exactly (ax, ay) px."""
    return (ax - w / 2, ay - h, ax + w / 2, ay, tid, cls)


def dets(*objs) -> sv.Detections:
    """Tracked detections from `person(...)` tuples (tracker_id < 0 = unconfirmed)."""
    if not objs:
        return sv.Detections.empty()
    xyxy = np.array([o[:4] for o in objs], dtype=float)
    return sv.Detections(xyxy=xyxy, confidence=np.full(len(objs), 0.9),
                         class_id=np.zeros(len(objs), int),
                         tracker_id=np.array([o[4] for o in objs], int),
                         data={"class_name": np.array([o[5] for o in objs])})


def hline(y=0.5, x0=0.2, x1=0.8, lid="L") -> Annotation:
    return Annotation(lid, LINE, [(x0, y), (x1, y)])


def box_zone(x0, y0, x1, y1, zid="Z", dwell=0.0) -> Annotation:
    return Annotation(zid, ZONE, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dwell=dwell)


def counts(stats, lid="L") -> tuple[int, int]:
    d = stats.line_counts[lid]
    return sum(d["positive"].values()), sum(d["negative"].values())


def run_track(engine, ys, x=500, tid=1, dt=0.1, w=W, h=H):
    """Walk one track's anchor through `ys` (px), one frame each; return the final stats."""
    s = None
    for k, y in enumerate(ys):
        s = engine.update(dets(person(x, y, tid)), w, h, k * dt)
    return s


def collect() -> list[tuple[str, bool, str]]:
    R: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        R.append((name, bool(cond), detail))

    _lines(check)
    _persistence(check)
    _resolution(check)
    _dwell(check)
    _detect_every(check)
    _config(check)
    return R


# ---------------------------------------------------------------- 1. line crossing
def _lines(check):
    ann = AnnotationSet([hline()])        # (200,500) -> (800,500); +side = y > 500 (image down)
    a, b = (200, 500), (800, 500)

    s = run_track(GeometryEngine(ann), [450, 550])
    check("line: downward crossing → +1 positive, 0 negative", counts(s) == (1, 0), str(counts(s)))
    s = run_track(GeometryEngine(ann), [550, 450])
    check("line: upward (reverse) crossing → +1 negative, 0 positive", counts(s) == (0, 1),
          str(counts(s)))

    rev = AnnotationSet([Annotation("L", LINE, [(0.8, 0.5), (0.2, 0.5)])])
    s = run_track(GeometryEngine(rev), [450, 550])
    check("line: reversing v0→v1 flips the direction (right-hand rule)", counts(s) == (0, 1),
          str(counts(s)))

    # crosses the INFINITE line (side flips) but at x=900, beyond the segment's end (x=800)
    naive_flip = (_side(a, b, (900, 450)) > 0) != (_side(a, b, (900, 550)) > 0)
    s = run_track(GeometryEngine(ann), [450, 550], x=900)
    check("line: crossing only the infinite extension counts nothing",
          naive_flip and counts(s) == (0, 0), f"side flip={naive_flip}, counts={counts(s)}")
    s = run_track(GeometryEngine(ann), [450, 550], x=100)
    check("line: ... on the other end too (x < v0)", counts(s) == (0, 0), str(counts(s)))

    # anchor = bottom-center: a box whose TOP crosses the line but whose feet never do
    g = GeometryEngine(ann)
    for k, top in enumerate([400, 520]):  # bottom stays at y=480 (above) — h shrinks
        g.update(sv.Detections(xyxy=np.array([[480, top, 520, 480.0]]),
                               tracker_id=np.array([1]),
                               data={"class_name": np.array(["person"])}), W, H, k * 0.1)
    s = g.update(dets(), W, H, 0.2)
    check("line: uses the bottom-center anchor (box top crossing ≠ crossing)",
          counts(s) == (0, 0), str(counts(s)))

    s = run_track(GeometryEngine(ann), [400, 450, 490, 510, 550, 600])
    check("line: slow multi-frame approach + crossing counts exactly once",
          counts(s) == (1, 0), str(counts(s)))
    s = run_track(GeometryEngine(ann), [480, 492, 485, 499, 490, 498])
    check("line: jitter near the line without crossing → 0", counts(s) == (0, 0), str(counts(s)))
    s = run_track(GeometryEngine(ann), [480, 500, 480, 500, 480])
    check("line: touch the line from the − side and retreat → 0", counts(s) == (0, 0),
          str(counts(s)))
    s = run_track(GeometryEngine(ann), [520, 500, 520, 500, 520])
    check("line: touch the line from the + side and retreat → 0  [BUG if ✗]",
          counts(s) == (0, 0), f"(pos, neg)={counts(s)}; anchor exactly ON the line is "
          "treated as the − side, so + → on → + counts a − AND a + crossing each touch")
    s = run_track(GeometryEngine(ann), [480, 500, 520])
    check("line: − → on-line → + counts exactly one positive", counts(s) == (1, 0), str(counts(s)))
    s = run_track(GeometryEngine(ann), [495, 505, 495, 505])
    check("line: real back-and-forth crossings each count once (net flow +1)",
          counts(s) == (2, 1), str(counts(s)))

    s = run_track(GeometryEngine(ann), [450, 550], tid=-1)
    check("line: unconfirmed track (tracker_id −1) never counts", counts(s) == (0, 0),
          str(counts(s)))
    # a 1-frame jump > half the frame diagonal is a tracker id-swap/teleport, not a crossing
    s = run_track(GeometryEngine(AnnotationSet([hline(x0=0.0, x1=1.0)])), [5, 995])
    check("line: teleport (> ½ diagonal in one step) is ignored", counts(s) == (0, 0),
          str(counts(s)))

    g = GeometryEngine(ann)
    g.update(dets(person(400, 450, 1), person(600, 550, 2)), W, H, 0.0)
    s = g.update(dets(person(400, 550, 1), person(600, 450, 2)), W, H, 0.1)
    check("line: two tracks crossing opposite ways in one frame → (1, 1)",
          counts(s) == (1, 1), str(counts(s)))
    s = g.update(dets(person(400, 560, 1), person(600, 440, 2)), W, H, 0.2)
    check("line: counts are cumulative and don't re-fire once across", counts(s) == (1, 1),
          str(counts(s)))


# ---------------------------------------------------------------- 2. persistence
def _persistence(check):
    ann = AnnotationSet([hline()])
    ys = [450, 490, 510, 550]
    fresh = []
    for k, y in enumerate(ys):            # a NEW engine every frame (the web-UI bug pattern)
        fresh.append(counts(GeometryEngine(ann).update(dets(person(500, y, 1)), W, H, k * 0.1)))
    persistent = counts(run_track(GeometryEngine(ann), ys))
    check("persist: fresh engine per frame sees 0 crossings",
          all(c == (0, 0) for c in fresh), str(fresh))
    check("persist: one engine across frames sees the crossing", persistent == (1, 0),
          str(persistent))
    # the track must be seen on consecutive updates: a frame without it prunes its history
    g = GeometryEngine(ann)
    g.update(dets(person(500, 450, 1)), W, H, 0.0)
    g.update(dets(), W, H, 0.1)
    s = g.update(dets(person(500, 550, 1)), W, H, 0.2)
    check("persist: a frame with NO tracks prunes crossing state (why skip frames must "
          "HOLD tracks)", counts(s) == (0, 0), str(counts(s)))


# ---------------------------------------------------------------- 3. resolution independence
def _scene_at(w, h):
    """The same 12-frame normalized scene rendered at w×h. Returns a comparable digest."""
    ann = AnnotationSet([box_zone(0.10, 0.55, 0.45, 0.95, "Z"), hline(0.5, 0.2, 0.8, "L")])
    g = GeometryEngine(ann)

    def p(nx, ny, tid):                     # normalized anchor + normalized box size
        return person(nx * w, ny * h, tid, w=0.03 * w, h=0.12 * h)

    zone_seq, dwell_seq = [], []
    s = None
    for k in range(12):
        objs = [p(0.50, 0.40 + 0.02 * k, 1),          # walks down across the line
                p(0.30, 0.75, 2),                     # stands inside the zone
                p(0.90, 0.60 - 0.02 * k, 3)]          # walks up across the line's EXTENSION
        if k >= 4:
            objs.append(p(0.12 + 0.05 * (k - 4), 0.70, 4))   # walks into then out of the zone
        s = g.update(dets(*objs), w, h, k * 0.25)
        zone_seq.append(sum(s.zone_counts["Z"].values()))
        dwell_seq.append(sorted((a["track"], a["zone"], a["seconds"]) for a in s.active_dwell))
    return zone_seq, counts(s, "L"), dwell_seq, [d[:2] for d in s.dwell]


def _resolution(check):
    lo, hi, uhd = _scene_at(640, 360), _scene_at(1920, 1080), _scene_at(3840, 2160)
    check("res: zone counts identical at 640x360 / 1080p / 4K",
          lo[0] == hi[0] == uhd[0], f"{lo[0]} | {hi[0]} | {uhd[0]}")
    check("res: zone occupancy actually varies (non-trivial scene)",
          len(set(lo[0])) > 1, str(lo[0]))
    check("res: line counts identical (and = 1 positive; extension walker ignored)",
          lo[1] == hi[1] == uhd[1] == (1, 0), f"{lo[1]} | {hi[1]} | {uhd[1]}")
    check("res: live dwell timers identical across resolutions",
          lo[2] == hi[2] == uhd[2] and lo[3] == hi[3] == uhd[3] and lo[3],
          f"{lo[3]} | {hi[3]}")


# ---------------------------------------------------------------- 4. dwell
def _approx(a, b, tol=0.011):
    return abs(a - b) <= tol


def _dwell(check):
    zone = AnnotationSet([box_zone(0.25, 0.25, 0.75, 0.75)])   # 250..750 px at 1000x1000

    # -- accrual + confirm + threshold
    g = GeometryEngine(zone, min_dwell=1.0)
    s0 = g.update(dets(person(500, 500, 1)), W, H, 0.0)
    check("dwell: first sighting is not shown yet (confirm=2 frames)", s0.active_dwell == [],
          str(s0.active_dwell))
    seen, over_at = [], None
    for k in range(1, 31):
        s = g.update(dets(person(500, 500, 1)), W, H, k * 0.1)
        seen.append(s.active_dwell[0]["seconds"])
        if s.dwell and over_at is None:
            over_at = round(k * 0.1, 2)
    check("dwell: timer accrues monotonically from first sighting (3.0 s at t=3.0)",
          seen == sorted(seen) and _approx(seen[-1], 3.0), f"last={seen[-1]}")
    check("dwell: reported as dwell exactly once it reaches min_dwell (1.0 s)",
          over_at == 1.0, f"first over at t={over_at}")
    check("dwell: reported interval starts at first sighting",
          s.dwell and _approx(s.dwell[0][2], 0.0) and _approx(s.dwell[0][3], 3.0), str(s.dwell))

    # -- tracker id change mid-stay (re-id): timer continues, same occupant record
    g = GeometryEngine(zone, min_dwell=1.0)
    for k in range(0, 16):
        s = g.update(dets(person(500, 500, 1)), W, H, k * 0.1)
    rec_before = s.active_dwell[0]["track"]
    for k in range(16, 31):
        s = g.update(dets(person(502, 501, 7)), W, H, k * 0.1)    # NEW tracker id 7
    ad = s.active_dwell
    check("dwell: id change 1→7 mid-stay keeps ONE occupant, same record",
          len(ad) == 1 and ad[0]["track"] == rec_before, str(ad))
    check("dwell: ... and the timer continues (3.0 s, not 1.5 s)",
          ad and _approx(ad[0]["seconds"], 3.0), str(ad))
    check("dwell: (contrast) raw track_start for the new id restarted at 1.6 s",
          _approx(s.track_start.get(7, -1), 1.6), str(s.track_start))

    # -- two occupants whose tracker ids SWAP: each timer follows the person, not the id
    g = GeometryEngine(AnnotationSet([box_zone(0.05, 0.05, 0.95, 0.95)]))
    for k in range(0, 21):
        objs = [person(300, 500, 1)] + ([person(700, 500, 2)] if k >= 10 else [])
        g.update(dets(*objs), W, H, k * 0.1)
    for k in range(21, 31):
        s = g.update(dets(person(300, 500, 2), person(700, 500, 1)), W, H, k * 0.1)
    by_x = {round(a["anchor"][0]): a["seconds"] for a in s.active_dwell}
    check("dwell: id swap between two people → timers stay with the people",
          len(by_x) == 2 and _approx(by_x.get(300, 0), 3.0) and _approx(by_x.get(700, 0), 2.0),
          str(by_x))

    # -- occlusion shorter than grace_lost (2 s): ring holds 0.5 s, then the SAME timer resumes
    g = GeometryEngine(zone, min_dwell=1.0)
    for k in range(0, 11):
        s = g.update(dets(person(500, 500, 1)), W, H, k * 0.1)
    rec = s.active_dwell[0]["track"]
    held = g.update(dets(), W, H, 1.3).active_dwell            # 0.3 s gap ≤ ring_hold
    gone = g.update(dets(), W, H, 1.8).active_dwell            # 0.8 s gap > ring_hold
    s = g.update(dets(person(505, 498, 9)), W, H, 2.2)         # back, new id, 1.2 s gap
    check("dwell: ring held through a brief (0.3 s) gap", len(held) == 1, str(held))
    check("dwell: ring hidden after ring_hold (0.8 s gap)", gone == [], str(gone))
    check("dwell: re-acquired after 1.2 s occlusion with a new id → same timer (2.2 s)",
          len(s.active_dwell) == 1 and s.active_dwell[0]["track"] == rec
          and _approx(s.active_dwell[0]["seconds"], 2.2), str(s.active_dwell))

    # -- absence longer than grace_lost: genuinely a new visit → timer restarts
    g = GeometryEngine(zone, min_dwell=1.0)
    for k in range(0, 11):
        g.update(dets(person(500, 500, 1)), W, H, k * 0.1)
    g.update(dets(), W, H, 1.5)
    g.update(dets(person(500, 500, 1)), W, H, 3.5)             # 2.5 s gap > grace_lost
    s = g.update(dets(person(500, 500, 1)), W, H, 3.6)
    check("dwell: absence > grace_lost (2.5 s) restarts the timer",
          len(s.active_dwell) == 1 and _approx(s.active_dwell[0]["seconds"], 0.1)
          and not s.dwell, str(s.active_dwell))

    # -- edge hysteresis. diag≈1414 → enter band ≈5.7 px inside, leave band ≈28.3 px outside
    g = GeometryEngine(zone, min_dwell=1.0)
    for k, y in enumerate([700] * 10 + [720, 740, 760] + [760] * 10):   # ends 10 px OUTSIDE
        s = g.update(dets(person(500, y, 1)), W, H, k * 0.1)
    check("dwell: just outside the edge (10 px) → zone count 0 …",
          sum(s.zone_counts["Z"].values()) == 0, str(s.zone_counts))
    check("dwell: … but the dwell keeps counting (leave band, no edge flicker)",
          len(s.active_dwell) == 1 and _approx(s.active_dwell[0]["seconds"], 2.2),
          str(s.active_dwell))
    for k, y in enumerate([790, 820, 820, 820, 820, 820, 820], start=23):  # clearly outside
        s = g.update(dets(person(500, y, 1)), W, H, k * 0.1)
    check("dwell: clearly outside (> leave band) → dwell ends", s.active_dwell == [],
          str(s.active_dwell))
    g = GeometryEngine(zone, min_dwell=1.0)
    for k in range(20):                                      # 3 px inside < enter band
        s = g.update(dets(person(500, 747, 1)), W, H, k * 0.1)
    check("dwell: hovering 3 px inside the edge counts as occupancy but never starts a dwell",
          sum(s.zone_counts["Z"].values()) == 1 and s.active_dwell == [], str(s.active_dwell))

    # -- per-zone threshold overrides min_dwell; unconfirmed tracks never dwell
    g = GeometryEngine(AnnotationSet([box_zone(0.25, 0.25, 0.75, 0.75, dwell=2.0)]), min_dwell=1.0)
    over = {}
    for k in range(0, 26):
        s = g.update(dets(person(500, 500, 1)), W, H, k * 0.1)
        over[round(k * 0.1, 1)] = bool(s.dwell)
    check("dwell: per-zone threshold (2.0 s) overrides min_dwell (1.0 s)",
          not over[1.5] and not over[1.9] and over[2.0], f"1.5→{over[1.5]} 2.0→{over[2.0]}")
    g = GeometryEngine(zone)
    for k in range(20):
        s = g.update(dets(person(500, 500, -1)), W, H, k * 0.1)
    check("dwell: unconfirmed track (id −1) never dwells or occupies",
          s.active_dwell == [] and sum(s.zone_counts["Z"].values()) == 0)


# ---------------------------------------------------------------- 5. detect_every hold
EVERY, NFRAMES = 3, 10
FRAME = np.zeros((360, 640, 3), np.uint8)


class _Source:
    fps = 30.0

    def __init__(self, on_release=None):
        self.on_release = on_release

    def frames(self):
        for _ in range(NFRAMES):
            yield FRAME.copy()

    def release(self):
        if self.on_release:
            self.on_release()


class _Detector:
    name = "stub"

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return dets(person(100, 200, -1), person(400, 300, -1))


class _Tracker:
    """Records what it is fed; each output is tagged with its generation number."""

    def __init__(self):
        self.fed: list[int] = []

    def update(self, d, frame=None):
        self.fed.append(len(d))
        gen = len(self.fed) - 1
        out = dets(person(100 + gen, 200, 1), person(400, 300, 2))
        out.data["gen"] = np.array([gen, gen])
        return out


class _Renderer:
    def __getattr__(self, name):           # draw / draw_annotations / draw_counts / hud / …
        return lambda img=None, *a, **k: img


def _spy_engine(log):
    class SpyGeometry(GeometryEngine):
        def update(self, det, w, h, t):
            log.append(det)
            return super().update(det, w, h, t)
    return SpyGeometry


def _assert_hold(check, tag, det, trk, seen):
    gens = [int(d.data["gen"][0]) if len(d) and "gen" in d.data else None for d in seen]
    want = [n // EVERY for n in range(NFRAMES)]
    check(f"hold[{tag}]: detector runs only on every {EVERY}rd frame (0, 3, 6, 9)",
          det.calls == len(range(0, NFRAMES, EVERY)), f"calls={det.calls}")
    check(f"hold[{tag}]: tracker is never fed empty detections on skip frames",
          trk.fed and all(n > 0 for n in trk.fed), f"fed={trk.fed}")
    check(f"hold[{tag}]: geometry gets the LAST tracked result on every skip frame",
          len(seen) == NFRAMES and gens == want and all(len(d) == 2 for d in seen),
          f"gens={gens} want={want}")


def _detect_every(check):
    with warnings.catch_warnings():      # trackers warns that ByteTrack ignores `frame`
        warnings.filterwarnings("ignore", message=".*does not use it.*")
        _detect_every_impl(check)


def _detect_every_impl(check):
    # why: real ByteTrack drops every track the moment it's fed an empty frame
    from occ.tracking import Tracker
    bt = Tracker("bytetrack", {"frame_rate": 30})
    raw = dets(person(100, 200, -1), person(400, 300, -1))
    for _ in range(5):
        live = bt.update(raw, None)
    dropped = bt.update(sv.Detections.empty(), None)
    check("hold[why]: ByteTrack with 2 confirmed tracks fed an empty frame → 0 tracks",
          len(live) == 2 and (live.tracker_id >= 0).all() and len(dropped) == 0,
          f"live={len(live)} after-empty={len(dropped)}")

    # (a) occ/pipeline.py — Pipeline.run with stubbed collaborators (bypass __init__ I/O)
    from occ.pipeline import Pipeline
    p = Pipeline.__new__(Pipeline)
    seen: list = []
    p.cfg = Config.load()
    p.source, p.detector, p.tracker, p.renderer = _Source(), _Detector(), _Tracker(), _Renderer()
    p.detect_every, p.max_fps, p.speed = EVERY, 0.0, None
    p.annotations = AnnotationSet()
    p.geometry = _spy_engine(seen)(p.annotations)
    p.run(show=False)
    _assert_hold(check, "occ/pipeline.py", p.detector, p.tracker, seen)

    # ...and end-to-end with the REAL ByteTrack: tracks stay alive through skip frames
    p2 = Pipeline.__new__(Pipeline)
    seen2: list = []
    p2.cfg, p2.source, p2.detector = Config.load(), _Source(), _Detector()
    p2.tracker, p2.renderer = Tracker("bytetrack", {"frame_rate": 30}), _Renderer()
    p2.detect_every, p2.max_fps, p2.speed, p2.annotations = EVERY, 0.0, None, AnnotationSet()
    p2.geometry = _spy_engine(seen2)(p2.annotations)
    p2.run(show=False)
    alive = [int((d.tracker_id >= 0).sum()) if d.tracker_id is not None else 0 for d in seen2]
    check("hold[occ/pipeline.py + real ByteTrack]: 2 confirmed tracks on every frame once "
          "confirmed", alive[EVERY:] == [2] * (NFRAMES - EVERY), f"alive={alive}")

    # (b) occ/web.py — importing it runs `app = create_app()` (opens a clip, starts a worker
    # thread), so execute the module source minus that one line in a private namespace and
    # drive the REAL WebPipeline._loop with stubbed _build.
    src = (ROOT / "occ" / "web.py").read_text()
    tail = "\napp = create_app()"
    if src.count(tail) != 1:
        check("hold[occ/web.py]: module shape as expected", False, "`app = create_app()` moved")
    else:
        web = types.ModuleType("occ._web_under_test")
        web.__dict__.update(__package__="occ", __file__=str(ROOT / "occ" / "web.py"))
        exec(compile(src.replace(tail, "\n"), str(ROOT / "occ" / "web.py"), "exec"),
             web.__dict__)
        seen_w: list = []
        web.GeometryEngine = _spy_engine(seen_w)
        wp = web.WebPipeline(Config.load())
        det_w, trk_w = _Detector(), _Tracker()
        wp._build = lambda: (_Source(wp._stop.set), det_w, trk_w, _Renderer(), EVERY)
        wp._run.set()
        t0 = time.perf_counter()
        wp._loop()                           # returns once the stub source's release() stops it
        _assert_hold(check, "occ/web.py", det_w, trk_w, seen_w)
        check("hold[occ/web.py]: /stats reports 2 tracks on the last (skip) frame",
              wp.stats()["tracks"] == 2 and time.perf_counter() - t0 < 5, str(wp.stats()))

    # (c) occ/studio/pipeline.py — real StudioPipeline._loop, stubbed _build
    import occ.studio.pipeline as SP
    saved_ge = SP.GeometryEngine
    try:
        seen_s: list = []
        SP.GeometryEngine = _spy_engine(seen_s)
        sp = SP.StudioPipeline(Config.load())
        det_s, trk_s = _Detector(), _Tracker()
        sp._build = lambda: (_Source(sp._stop.set), det_s, trk_s, _Renderer(), EVERY)
        sp._run.set()
        sp._loop()
        _assert_hold(check, "occ/studio/pipeline.py", det_s, trk_s, seen_s)
        check("hold[occ/studio/pipeline.py]: scene() exposes the held tracks",
              sp.stats()["tracks"] == 2 and len(sp.scene()["objects"]) == 2, str(sp.stats()))
    finally:
        SP.GeometryEngine = saved_ge


# ---------------------------------------------------------------- 6. config precedence
def _config(check):
    base = Config.load()
    block = (base.data.get("source_overrides") or {}).get("market-square") or {}
    check("config: default.yaml ships a market-square source override (imgsz 1280)",
          block.get("detector.imgsz") == 1280, str(block))
    default_imgsz = base.get("detector.imgsz")

    c = Config.load(overrides=["source.uri=rtsp://cam/market-square-east"])
    check("config: substring-matched source override applies",
          all(c.get(k) == v for k, v in block.items()),
          f"{ {k: c.get(k) for k in block} }")
    c = Config.load(overrides=["source.uri=assets/videos/vehicles-2.mp4"])
    check("config: non-matching source keeps defaults",
          c.get("detector.imgsz") == default_imgsz != 1280, str(c.get("detector.imgsz")))
    c = Config.load(overrides=["source.uri=assets/videos/market-square.mp4",
                               "detector.imgsz=960"])
    check("config: explicit --set wins over the source override …",
          c.get("detector.imgsz") == 960, str(c.get("detector.imgsz")))
    check("config: … while the override's other keys still apply",
          c.get("detector.conf") == block.get("detector.conf"), str(c.get("detector.conf")))
    c = Config.load(overrides=["detector.imgsz=960",
                               "source.uri=assets/videos/market-square.mp4"])
    check("config: precedence is independent of --set order",
          c.get("detector.imgsz") == 960, str(c.get("detector.imgsz")))

    # live reconfigure path (web.py / studio): switch source → override applies,
    # except for keys in the same update
    c = Config.load(overrides=["source.uri=assets/videos/vehicles-2.mp4"])
    upd = {"source.uri": "assets/videos/market-square.mp4", "detector.conf": 0.4}
    for k, v in upd.items():
        _set_dotted(c.data, k, v)
    c._apply_source_overrides(set(upd))
    check("config: live source switch applies override but keeps keys set in the same update",
          c.get("detector.imgsz") == 1280 and c.get("detector.conf") == 0.4,
          f"imgsz={c.get('detector.imgsz')} conf={c.get('detector.conf')}")
    check("config: Config.load never mutates the default (fresh load is clean)",
          Config.load().get("detector.imgsz") == default_imgsz)


def main():
    t0 = time.perf_counter()
    results = collect()
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({detail})" if detail and not ok else ""))
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed  "
          f"({time.perf_counter() - t0:.1f}s)")
    assert not bad, bad


if __name__ == "__main__":
    main()
