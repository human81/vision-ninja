"""The OpenCV ninja — a hot-loadable dynamic-overlay engine.

This is the heart of "create ANY computer-vision overlay to achieve a visual
task." The agent authors a small `draw(ctx)` function (or instantiates a
built-in preset); the engine compiles it and runs it on every rendered frame.
`ctx` hands the agent the frame plus rich, vectorized helpers (boxes, anchors,
class masks, rings, heatmaps, a palette, and a persistent per-overlay `state`
for trails/accumulators) so overlay code is a few expressive lines, not a loop.

Safety note: overlay code is the user's own agent authoring cv2/numpy for a
LOCAL, single-user tool. Builtins are restricted to a safe subset and each
overlay runs inside try/except so a bad overlay can never crash the stream — but
this is not a hardened sandbox. It is creative compute the user opted into.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

COLORS = {
    "red": (60, 60, 240), "green": (90, 230, 90), "blue": (240, 160, 60),
    "yellow": (60, 220, 250), "cyan": (230, 230, 60), "magenta": (230, 90, 230),
    "orange": (40, 150, 255), "white": (240, 240, 240), "amber": (60, 180, 255),
    "lime": (60, 255, 170), "pink": (170, 120, 255), "teal": (200, 200, 60),
}
_PALETTE = [COLORS[k] for k in ("cyan", "lime", "amber", "magenta", "orange",
                                "green", "pink", "yellow", "red", "teal")]


class OverlayCtx:
    """Per-frame drawing context handed to each overlay's draw(ctx)."""

    def __init__(self, frame, det, geo, t, state, params):
        self.frame = frame
        self.det = det
        self.geo = geo
        self.h, self.w = frame.shape[:2]
        self.t = t                      # frame index
        self.state = state              # persistent dict for THIS overlay
        self.params = params or {}
        self.cv2 = cv2
        self.np = np
        self.COLORS = COLORS

    # ---- vectorized accessors over the tracked detections ----
    @property
    def n(self):
        return 0 if self.det is None else len(self.det)

    @property
    def boxes(self):
        return (self.det.xyxy.astype(int) if self.n else np.empty((0, 4), int))

    @property
    def centers(self):
        b = self.boxes
        if not len(b):
            return np.empty((0, 2), int)
        return np.stack([(b[:, 0] + b[:, 2]) // 2, (b[:, 1] + b[:, 3]) // 2], 1)

    @property
    def anchors(self):
        """Bottom-center ground anchors (what geometry counts on)."""
        b = self.boxes
        if not len(b):
            return np.empty((0, 2), int)
        return np.stack([(b[:, 0] + b[:, 2]) // 2, b[:, 3]], 1)

    @property
    def names(self):
        if self.n and self.det.data and "class_name" in self.det.data:
            return list(self.det.data["class_name"])
        return ["object"] * self.n

    @property
    def ids(self):
        return (self.det.tracker_id if self.n and self.det.tracker_id is not None
                else np.full(self.n, -1))

    @property
    def confs(self):
        return (self.det.confidence if self.n and self.det.confidence is not None
                else np.ones(self.n))

    def mask(self, *classes):
        names = self.names
        want = {c.lower() for c in classes}
        return np.array([n.lower() in want for n in names], bool)

    def palette(self, i):
        return _PALETTE[int(i) % len(_PALETTE)]

    # ---- drawing helpers (all anti-aliased) ----
    def ring(self, center, r, color="cyan", thick=2, glow=False):
        c = COLORS.get(color, color); p = (int(center[0]), int(center[1]))
        if glow:
            cv2.circle(self.frame, p, int(r) + 4, c, thick + 4, cv2.LINE_AA)
        cv2.circle(self.frame, p, int(r), c, thick, cv2.LINE_AA)

    def box(self, b, color="cyan", thick=2):
        c = COLORS.get(color, color)
        cv2.rectangle(self.frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      c, thick, cv2.LINE_AA)

    def line(self, a, b, color="cyan", thick=2):
        c = COLORS.get(color, color)
        cv2.line(self.frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 c, thick, cv2.LINE_AA)

    def arrow(self, a, b, color="lime", thick=2):
        c = COLORS.get(color, color)
        cv2.arrowedLine(self.frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                        c, thick, cv2.LINE_AA, tipLength=0.3)

    def poly(self, pts, color="amber", fill_alpha=0.0, thick=2):
        c = COLORS.get(color, color)
        arr = np.array(pts, np.int32).reshape(-1, 1, 2)
        if fill_alpha > 0:
            ov = self.frame.copy(); cv2.fillPoly(ov, [arr], c)
            cv2.addWeighted(ov, fill_alpha, self.frame, 1 - fill_alpha, 0, self.frame)
        cv2.polylines(self.frame, [arr], True, c, thick, cv2.LINE_AA)

    def text(self, s, xy, color="white", scale=0.6, bg=(0, 0, 0), thick=1):
        c = COLORS.get(color, color)
        x, y = int(xy[0]), int(xy[1])
        (tw, th), bl = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if bg is not None:
            cv2.rectangle(self.frame, (x - 3, y - th - 3), (x + tw + 3, y + bl + 3), bg, -1)
        cv2.putText(self.frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c,
                    thick, cv2.LINE_AA)

    def heat(self, points, radius=60, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """Blend a density heatmap from a set of points onto the frame."""
        if len(points) == 0:
            return
        acc = np.zeros((self.h, self.w), np.float32)
        for px, py in points:
            cv2.circle(acc, (int(px), int(py)), int(radius), 1.0, -1, cv2.LINE_AA)
        acc = cv2.GaussianBlur(acc, (0, 0), radius / 2.0)
        if acc.max() > 1e-6:
            acc /= acc.max()
        hm = cv2.applyColorMap((acc * 255).astype(np.uint8), colormap)
        m = (acc > 0.05)[..., None]
        blended = (self.frame * (1 - alpha) + hm * alpha).astype(np.uint8)
        np.copyto(self.frame, np.where(m, blended, self.frame))


@dataclass
class Overlay:
    name: str
    intent: str
    code: str
    enabled: bool = True
    builtin: bool = False
    fn: object = None
    state: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    error: str = ""
    errors: int = 0


_SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict)
                  else getattr(__builtins__, k)
                  for k in ("abs", "min", "max", "len", "range", "enumerate",
                            "zip", "sorted", "sum", "round", "int", "float",
                            "str", "list", "dict", "tuple", "set", "bool",
                            "map", "filter", "any", "all", "reversed")}


def compile_overlay(code: str):
    """Compile agent code that defines `def draw(ctx): ...`; return the function."""
    g = {"__builtins__": _SAFE_BUILTINS, "cv2": cv2, "np": np, "COLORS": COLORS}
    exec(compile(code, "<overlay>", "exec"), g)  # noqa: S102 (intentional)
    fn = g.get("draw")
    if not callable(fn):
        raise ValueError("overlay code must define a function `draw(ctx)`")
    return fn


class OverlayEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self.overlays: dict[str, Overlay] = {}

    def add(self, name: str, intent: str, code: str, builtin: bool = False) -> Overlay:
        fn = compile_overlay(code)            # raises on bad code (caught by caller)
        ov = Overlay(name=name, intent=intent, code=code, fn=fn, builtin=builtin)
        with self._lock:
            self.overlays[name] = ov
        return ov

    def add_builtin(self, key: str) -> Overlay:
        if key not in BUILTINS:
            raise KeyError(f"unknown preset {key!r}; have {list(BUILTINS)}")
        intent, code = BUILTINS[key]
        return self.add(key, intent, code, builtin=True)

    def remove(self, name: str) -> bool:
        with self._lock:
            return self.overlays.pop(name, None) is not None

    def toggle(self, name: str, on: bool | None = None) -> bool:
        with self._lock:
            ov = self.overlays.get(name)
            if not ov:
                return False
            ov.enabled = (not ov.enabled) if on is None else bool(on)
            return ov.enabled

    def list(self) -> list[dict]:
        with self._lock:
            return [{"name": o.name, "intent": o.intent, "enabled": o.enabled,
                     "builtin": o.builtin, "error": o.error} for o in self.overlays.values()]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for o in self.overlays.values() if o.enabled and not o.error)

    def run(self, frame, det, geo, t: int) -> int:
        """Execute all enabled overlays on `frame` in place. Returns # executed."""
        with self._lock:
            items = list(self.overlays.values())
        ran = 0
        for ov in items:
            if not ov.enabled or ov.fn is None:
                continue
            try:
                ctx = OverlayCtx(frame, det, geo, t, ov.state, ov.params)
                ov.fn(ctx)
                ov.error = ""
                ran += 1
            except Exception as e:                     # never let one overlay kill the stream
                ov.errors += 1
                ov.error = f"{type(e).__name__}: {e}"
                if ov.errors >= 30:                    # quarantine a persistently broken overlay
                    ov.enabled = False
        return ran


# ---- built-in presets the agent can instantiate instantly ----------------
BUILTINS: dict[str, tuple[str, str]] = {
    "density_heatmap": (
        "Heatmap of where objects are densest (ground anchors).",
        "def draw(ctx):\n"
        "    ctx.heat(ctx.anchors, radius=70, alpha=0.55)\n"
        "    ctx.text(f'density: {ctx.n} objs', (12, ctx.h-16), 'white', 0.6)\n"),
    "track_trails": (
        "Fading motion trail behind every tracked object.",
        "def draw(ctx):\n"
        "    hist = ctx.state.setdefault('h', {})\n"
        "    live = set()\n"
        "    for (cx, cy), tid in zip(ctx.centers, ctx.ids):\n"
        "        tid = int(tid)\n"
        "        if tid < 0: continue\n"
        "        live.add(tid)\n"
        "        pts = hist.setdefault(tid, [])\n"
        "        pts.append((int(cx), int(cy)))\n"
        "        if len(pts) > 30: del pts[0]\n"
        "        for i in range(1, len(pts)):\n"
        "            ctx.line(pts[i-1], pts[i], ctx.palette(tid), 2)\n"
        "    for tid in [k for k in hist if k not in live]: del hist[tid]\n"),
    "class_highlight": (
        "Pulsing ring around every object of params['class'] (default 'car').",
        "def draw(ctx):\n"
        "    import math\n"
        "    want = ctx.params.get('class', 'car')\n"
        "    m = ctx.mask(want)\n"
        "    r = 24 + int(6*math.sin(ctx.t*0.2))\n"
        "    for (cx, cy) in ctx.anchors[m]:\n"
        "        ctx.ring((cx, cy-20), r, 'magenta', 2, glow=True)\n"
        "    ctx.text(f'{want}: {int(m.sum())}', (12, 60), 'magenta', 0.7)\n"),
    "count_badge": (
        "Big live per-class tally in the corner.",
        "def draw(ctx):\n"
        "    from collections import Counter\n"
        "    c = Counter(ctx.names)\n"
        "    y = 90\n"
        "    for k, v in c.most_common():\n"
        "        ctx.text(f'{k}: {v}', (12, y), 'cyan', 0.8); y += 30\n"),
    "speed_vectors": (
        "Per-track velocity arrows from frame-to-frame motion.",
        "def draw(ctx):\n"
        "    prev = ctx.state.setdefault('p', {})\n"
        "    cur = {}\n"
        "    for (cx, cy), tid in zip(ctx.centers, ctx.ids):\n"
        "        tid = int(tid)\n"
        "        if tid < 0: continue\n"
        "        cur[tid] = (int(cx), int(cy))\n"
        "        if tid in prev:\n"
        "            px, py = prev[tid]\n"
        "            ctx.arrow((cx, cy), (cx + (cx-px)*4, cy + (cy-py)*4), 'lime', 2)\n"
        "    ctx.state['p'] = cur\n"),
}
