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

    def __init__(self, frame, det, geo, t, state, params, raw=None, faces_fn=None):
        self.frame = frame
        self.det = det
        self.geo = geo
        self.raw = raw if raw is not None else det     # raw detection (carries kpts/mask)
        self.h, self.w = frame.shape[:2]
        self.t = t                      # frame index
        self.state = state              # persistent dict for THIS overlay
        self.params = params or {}
        self.cv2 = cv2
        self.np = np
        self.COLORS = COLORS
        self._faces_fn = faces_fn       # memoized facemesh provider (one detect/frame)

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

    @property
    def kpts(self):
        """Pose keypoints (N, 17, 3) = [x, y, conf] in COCO order — only present in
        pose mode (set_task('pose')). 0 nose, 5/6 shoulders, 9/10 wrists, 15/16 ankles."""
        d = self.raw.data if (self.raw is not None and self.raw.data) else {}
        return d.get("kpts", np.empty((0, 17, 3)))

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

    # ---- AR / face try-on: dense landmarks + realistic asset compositing ----
    @property
    def faces(self):
        """List of detected faces (478 px landmarks + blendshapes + head pose).
        Lazily detected ONCE per frame (shared across overlays). Empty if no face
        or mediapipe unavailable. See occ.studio.facemesh.Face for the anchor API."""
        return self._faces_fn() if self._faces_fn else []

    def asset(self, name: str):
        """Fetch a cached RGBA (HxWx4) try-on asset (procedural or bundled PNG)."""
        from .face_filters import get_asset
        return get_asset(name)

    def warp(self, rgba, dst_quad):
        """Perspective-warp an RGBA asset so its [TL,TR,BR,BL] corners land on the
        4 frame points `dst_quad`, then alpha-composite. This is what makes try-on
        realistic: the quad comes from face landmarks, so the asset follows real
        head yaw / pitch / roll and scale — not a flat pasted sticker."""
        if rgba is None or len(rgba) == 0:
            return
        h, w = rgba.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32(dst_quad)
        try:
            M = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            return
        warped = cv2.warpPerspective(rgba, M, (self.w, self.h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
        self._alpha_over(warped)

    def sticker(self, rgba, center, width, angle=0.0):
        """Place an RGBA asset centered at `center`, scaled to pixel `width`,
        rotated by `angle` degrees (pass face.roll to follow head tilt)."""
        if rgba is None or len(rgba) == 0:
            return
        h, w = rgba.shape[:2]
        hh = width * h / w
        a = np.radians(angle); ca, sa = np.cos(a), np.sin(a)
        ex = np.array([ca, sa]); ey = np.array([-sa, ca])
        c = np.array([float(center[0]), float(center[1])])
        hw, ht = width / 2.0, hh / 2.0
        quad = [c - ex * hw - ey * ht, c + ex * hw - ey * ht,
                c + ex * hw + ey * ht, c - ex * hw + ey * ht]
        self.warp(rgba, quad)

    def paste(self, rgba, top_left):
        """Alpha-paste an RGBA asset at its native size, top-left at `top_left`."""
        if rgba is None or len(rgba) == 0:
            return
        x, y = int(top_left[0]), int(top_left[1])
        h, w = rgba.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
        a = sub[:, :, 3:4].astype(np.float32) / 255.0
        roi = self.frame[y0:y1, x0:x1].astype(np.float32)
        self.frame[y0:y1, x0:x1] = (sub[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)

    def _alpha_over(self, rgba_full):
        """Alpha-composite a frame-sized RGBA image over the frame (in place)."""
        a = rgba_full[:, :, 3:4].astype(np.float32) / 255.0
        if a.max() <= 0:
            return
        self.frame[:] = (rgba_full[:, :, :3].astype(np.float32) * a +
                         self.frame.astype(np.float32) * (1 - a)).astype(np.uint8)

    def fill_poly(self, pts, color="black", alpha=1.0):
        """Filled (optionally translucent) polygon; `color` name or BGR tuple."""
        c = COLORS.get(color, color) if isinstance(color, str) else color
        arr = np.array(pts, np.int32).reshape(-1, 1, 2)
        if alpha >= 1.0:
            cv2.fillPoly(self.frame, [arr], c, cv2.LINE_AA)
        else:
            ov = self.frame.copy(); cv2.fillPoly(ov, [arr], c, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, self.frame, 1 - alpha, 0, self.frame)

    def pixelate(self, pts, blocks=14):
        """Pixelate (anonymize) the convex region around `pts`."""
        arr = np.array(pts, np.int32)
        x, y, w, h = cv2.boundingRect(arr)
        x, y = max(0, x), max(0, y); w = min(self.w - x, w); h = min(self.h - y, h)
        if w < 4 or h < 4:
            return
        roi = self.frame[y:y + h, x:x + w]
        small = cv2.resize(roi, (max(1, blocks), max(1, blocks)),
                           interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(arr - [x, y]), 255)
        roi[mask > 0] = pix[mask > 0]


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


# Builtins the agent's cv2 code may use. The overlay sandbox is for avoiding
# accidents (no open/eval/exec/__import__), not real security — the agent is
# trusted to author arbitrary cv2, so include the everyday introspection/number/
# sequence helpers it reaches for (hasattr/getattr/isinstance were missing and
# raised NameError at render time, so overlays silently never drew).
_SAFE_BUILTIN_NAMES = (
    "abs", "min", "max", "len", "range", "enumerate", "zip", "sorted", "sum",
    "round", "int", "float", "str", "list", "dict", "tuple", "set", "frozenset",
    "bool", "bytes", "bytearray", "map", "filter", "any", "all", "reversed",
    "hasattr", "getattr", "setattr", "isinstance", "issubclass", "callable",
    "divmod", "pow", "chr", "ord", "hex", "bin", "oct", "format", "repr",
    "slice", "type", "iter", "next", "print", "hash", "complex",
)
_SAFE_BUILTINS = {k: (__builtins__[k] if isinstance(__builtins__, dict)
                      else getattr(__builtins__, k))
                  for k in _SAFE_BUILTIN_NAMES}


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
        if key in FACE_FILTERS:
            return self.add_native(key)
        if key not in BUILTINS:
            raise KeyError(f"unknown preset {key!r}; have {self.preset_names()}")
        intent, code = BUILTINS[key]
        return self.add(key, intent, code, builtin=True)

    def add_native(self, key: str) -> Overlay:
        """Register a built-in NATIVE filter (a real Python draw(ctx), e.g. the AR
        face filters) — not sandbox-compiled. Its source shows in the Code panel."""
        if key not in FACE_FILTERS:
            raise KeyError(f"unknown filter {key!r}; have {list(FACE_FILTERS)}")
        intent, fn, src = FACE_FILTERS[key]
        ov = Overlay(name=key, intent=intent, code=src, fn=fn, builtin=True)
        with self._lock:
            self.overlays[key] = ov
        return ov

    @staticmethod
    def preset_names() -> list[str]:
        return list(BUILTINS) + list(FACE_FILTERS)

    def remove(self, name: str) -> bool:
        with self._lock:
            return self.overlays.pop(name, None) is not None

    def clear(self) -> int:
        """Remove ALL overlays. Returns how many were removed."""
        with self._lock:
            n = len(self.overlays)
            self.overlays = {}
            return n

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
                     "builtin": o.builtin, "error": o.error, "code": o.code}
                    for o in self.overlays.values()]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for o in self.overlays.values() if o.enabled and not o.error)

    def run(self, frame, det, geo, t: int, raw=None, clean=None) -> int:
        """Execute all enabled overlays on `frame` in place. Returns # executed.
        `raw` is the pre-tracker detection (carries pose keypoints / masks).
        `clean` is the un-annotated frame used for face-landmark detection (so a
        filter doesn't try to find landmarks in already-drawn-on pixels)."""
        with self._lock:
            items = list(self.overlays.values())
        # one facemesh pass per FRAME, shared across every overlay that asks.
        face_src = clean if clean is not None else frame
        cache = {}
        def faces_fn():
            if "f" not in cache:
                try:
                    from .facemesh import detect_faces
                    cache["f"] = detect_faces(face_src)
                except Exception:
                    cache["f"] = []
            return cache["f"]
        ran = 0
        for ov in items:
            if not ov.enabled or ov.fn is None:
                continue
            try:
                ctx = OverlayCtx(frame, det, geo, t, ov.state, ov.params, raw=raw,
                                 faces_fn=faces_fn)
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
    "raised_hands": (
        "Ring people raising a hand (a wrist above the head). Needs pose mode.",
        "def draw(ctx):\n"
        "    for kp in ctx.kpts:\n"
        "        nose = kp[0]\n"
        "        if nose[2] < 0.3: continue\n"
        "        up = (kp[9][2] > 0.3 and kp[9][1] < nose[1]) or \\\n"
        "             (kp[10][2] > 0.3 and kp[10][1] < nose[1])\n"
        "        if up:\n"
        "            ctx.ring((int(nose[0]), int(nose[1])), 42, 'lime', 3, glow=True)\n"
        "            ctx.text('hand up', (int(nose[0]) - 34, int(nose[1]) - 52), 'lime', 0.6)\n"),
    "pose_glow": (
        "Glowing skeleton + joint dots for every person. Needs pose mode.",
        "def draw(ctx):\n"
        "    E = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),\n"
        "         (11,13),(13,15),(12,14),(14,16)]\n"
        "    for i, kp in enumerate(ctx.kpts):\n"
        "        col = ctx.palette(i)\n"
        "        for a, b in E:\n"
        "            if kp[a][2] > 0.3 and kp[b][2] > 0.3:\n"
        "                ctx.line((kp[a][0], kp[a][1]), (kp[b][0], kp[b][1]), col, 3)\n"
        "        for x, y, c in kp:\n"
        "            if c > 0.3: ctx.ring((x, y), 3, col, 2)\n"),
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


# Native AR face filters (glasses / ninja_mask / dog / … ) — real Python draw(ctx)
# functions registered alongside the string BUILTINS. Imported last to avoid a
# cycle (face_filters depends on nothing here; it duck-types ctx).
from .face_filters import FACE_FILTERS  # noqa: E402
