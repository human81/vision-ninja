"""AR face filters / virtual try-on — landmark-accurate, head-pose-following.

Native `draw(ctx)` filters (not sandbox-compiled) that use the dense 478-point
FaceLandmarker (ctx.faces) and the realistic compositor (ctx.warp/sticker — a
perspective warp onto landmark-derived quads, so assets follow real head yaw /
pitch / roll and scale). Assets are generated PROCEDURALLY at high resolution
with 4x supersampling for clean anti-aliased edges; any RGBA PNG dropped in
`assets/filters/` overrides a procedural asset of the same name.

Each filter is registered in FACE_FILTERS = {name: (intent, draw_fn, source)};
the source string surfaces in the studio's live Code panel.
"""

from __future__ import annotations

import inspect
import os

import cv2
import numpy as np

_S = 4                                     # supersample factor for asset AA
_ASSET_CACHE: dict[str, np.ndarray] = {}
_ASSET_DIR = "assets/filters"


# ============================== asset generation ==============================
def get_asset(name: str) -> np.ndarray:
    """Return a cached RGBA (HxWx4 uint8) asset. A PNG in assets/filters/<name>.png
    (with alpha) wins; otherwise a procedural generator draws it."""
    if name in _ASSET_CACHE:
        return _ASSET_CACHE[name]
    png = os.path.join(_ASSET_DIR, f"{name}.png")
    img = None
    if os.path.exists(png):
        loaded = cv2.imread(png, cv2.IMREAD_UNCHANGED)
        if loaded is not None and loaded.ndim == 3 and loaded.shape[2] == 4:
            img = loaded
    if img is None:
        gen = _GENERATORS.get(name)
        img = gen() if gen else np.zeros((4, 4, 4), np.uint8)
    _ASSET_CACHE[name] = img
    return img


def _canvas(w, h):
    return np.zeros((h * _S, w * _S, 4), np.uint8)


def _down(big, w, h):
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def _ell(c, center, axes, color, thick=-1, ang=0):
    cv2.ellipse(c, (int(center[0] * _S), int(center[1] * _S)),
                (int(axes[0] * _S), int(axes[1] * _S)), ang, 0, 360, color,
                thick if thick < 0 else thick * _S, cv2.LINE_AA)


def _line(c, a, b, color, thick):
    cv2.line(c, (int(a[0] * _S), int(a[1] * _S)), (int(b[0] * _S), int(b[1] * _S)),
             color, thick * _S, cv2.LINE_AA)


def _poly(c, pts, color, thick=-1):
    arr = (np.array(pts, np.float32) * _S).astype(np.int32).reshape(-1, 1, 2)
    if thick < 0:
        cv2.fillPoly(c, [arr], color, cv2.LINE_AA)
    else:
        cv2.polylines(c, [arr], True, color, thick * _S, cv2.LINE_AA)


def _gen_sunglasses():
    W, H = 660, 240
    c = _canvas(W, H)
    frame = (28, 28, 30, 255); lens = (35, 28, 22, 215); glint = (210, 200, 190, 130)
    lc = [(0.275 * W, 0.5 * H), (0.725 * W, 0.5 * H)]   # lens centers at 27.5% / 72.5%
    rx, ry = 0.165 * W, 0.30 * H
    for cx, cy in lc:
        _ell(c, (cx, cy), (rx, ry), lens)               # tinted lens
        _ell(c, (cx - rx * 0.3, cy - ry * 0.35), (rx * 0.34, ry * 0.22), glint)  # highlight
        _ell(c, (cx, cy), (rx, ry), frame, thick=4)     # frame
    # bridge (brow bar) + nose bridge
    _line(c, (lc[0][0] + rx * 0.82, 0.40 * H), (lc[1][0] - rx * 0.82, 0.40 * H), frame, 6)
    # temple arms to the edges
    _line(c, (lc[0][0] - rx, 0.44 * H), (0.02 * W, 0.40 * H), frame, 6)
    _line(c, (lc[1][0] + rx, 0.44 * H), (0.98 * W, 0.40 * H), frame, 6)
    return _down(c, W, H)


def _gen_glasses():
    W, H = 660, 230
    c = _canvas(W, H)
    frame = (45, 38, 30, 255); lens = (245, 235, 225, 40)
    lc = [(0.275 * W, 0.5 * H), (0.725 * W, 0.5 * H)]
    rx, ry = 0.165 * W, 0.27 * H
    for cx, cy in lc:
        _ell(c, (cx, cy), (rx, ry), lens)
        _ell(c, (cx, cy), (rx, ry), frame, thick=5)
    _line(c, (lc[0][0] + rx * 0.8, 0.42 * H), (lc[1][0] - rx * 0.8, 0.42 * H), frame, 7)
    _line(c, (lc[0][0] - rx, 0.45 * H), (0.02 * W, 0.40 * H), frame, 6)
    _line(c, (lc[1][0] + rx, 0.45 * H), (0.98 * W, 0.40 * H), frame, 6)
    return _down(c, W, H)


def _gen_mustache():
    W, H = 420, 200
    c = _canvas(W, H)
    col = (30, 25, 22, 255)
    # two curved halves meeting in the middle, tapering to points
    for s in (-1, 1):
        pts = [(0.5 * W, 0.30 * H), (0.5 * W + s * 0.10 * W, 0.22 * H),
               (0.5 * W + s * 0.30 * W, 0.20 * H), (0.5 * W + s * 0.46 * W, 0.30 * H),
               (0.5 * W + s * 0.49 * W, 0.5 * H), (0.5 * W + s * 0.40 * W, 0.42 * H),
               (0.5 * W + s * 0.22 * W, 0.55 * H), (0.5 * W + s * 0.08 * W, 0.55 * H),
               (0.5 * W, 0.45 * H)]
        _poly(c, pts, col)
    return _down(c, W, H)


def _gen_dog_ear():
    W, H = 200, 300
    c = _canvas(W, H)
    brown = (40, 70, 130, 255); inner = (120, 160, 220, 255)
    _poly(c, [(0.5 * W, 0.05 * H), (0.05 * W, 0.5 * H), (0.35 * W, 0.98 * H),
              (0.75 * W, 0.85 * H), (0.62 * W, 0.30 * H)], brown)
    _poly(c, [(0.5 * W, 0.22 * H), (0.30 * W, 0.5 * H), (0.42 * W, 0.82 * H),
              (0.60 * W, 0.62 * H)], inner)
    return _down(c, W, H)


def _gen_dog_nose():
    W, H = 220, 170
    c = _canvas(W, H)
    _ell(c, (0.5 * W, 0.45 * H), (0.40 * W, 0.32 * H), (30, 25, 22, 255))
    _ell(c, (0.40 * W, 0.38 * H), (0.07 * W, 0.05 * H), (120, 120, 120, 160))  # glint
    return _down(c, W, H)


def _gen_cat_ear():
    W, H = 200, 260
    c = _canvas(W, H)
    black = (25, 25, 28, 255); pink = (150, 150, 240, 255)
    _poly(c, [(0.5 * W, 0.04 * H), (0.08 * W, 0.95 * H), (0.92 * W, 0.95 * H)], black)
    _poly(c, [(0.5 * W, 0.32 * H), (0.30 * W, 0.85 * H), (0.70 * W, 0.85 * H)], pink)
    return _down(c, W, H)


def _gen_crown():
    W, H = 460, 260
    c = _canvas(W, H)
    gold = (40, 195, 240, 255); edge = (20, 120, 170, 255); gem = (210, 90, 90, 255)
    base = 0.92 * H; peak = 0.12 * H
    pts = [(0.04 * W, base)]
    xs = [0.04, 0.2, 0.36, 0.5, 0.64, 0.8, 0.96]
    ys = [base, peak, base * 0.62, peak * 0.7, base * 0.62, peak, base]
    for x, y in zip(xs, ys):
        pts.append((x * W, y))
    pts.append((0.96 * W, base))
    _poly(c, pts, gold); _poly(c, pts, edge, thick=4)
    for x in (0.2, 0.5, 0.8):
        _ell(c, (x * W, peak * (0.7 if x == 0.5 else 1.0)), (0.03 * W, 0.045 * H), gem)
    return _down(c, W, H)


def _gen_clown_nose():
    W, H = 200, 200
    c = _canvas(W, H)
    _ell(c, (0.5 * W, 0.5 * H), (0.42 * W, 0.42 * H), (60, 60, 235, 255))
    _ell(c, (0.38 * W, 0.38 * H), (0.12 * W, 0.10 * H), (160, 160, 255, 170))
    return _down(c, W, H)


_GENERATORS = {
    "sunglasses": _gen_sunglasses, "glasses": _gen_glasses, "mustache": _gen_mustache,
    "dog_ear": _gen_dog_ear, "dog_nose": _gen_dog_nose, "cat_ear": _gen_cat_ear,
    "crown": _gen_crown, "clown_nose": _gen_clown_nose,
}


# ============================== placement helpers ==============================
def _frame_axes(face):
    """Right (ex) and down (ey) unit vectors in the face plane (encode roll)."""
    ex = face.p("temple_r") - face.p("temple_l")
    n = np.linalg.norm(ex) or 1.0
    ex = ex / n
    ey = np.array([-ex[1], ex[0]])               # +90° → points toward chin
    return ex, ey


def _eyes_quad(face, wscale=2.25, hscale=1.0, down=0.08):
    """A roll-aware quad over the eyes for glasses (lenses land on the pupils:
    asset lens centers are at ±27.5% ↔ ±0.55·IPD here)."""
    ex, ey = _frame_axes(face)
    c = face.eyes_center + ey * (face.eye_dist * down)
    half_w = ex * (face.eye_dist * wscale / 2.0)
    half_h = ey * (face.eye_dist * wscale * (230 / 660) * hscale / 2.0)
    return [c - half_w - half_h, c + half_w - half_h,
            c + half_w + half_h, c - half_w + half_h]


def trim_arms(rgba):
    """Crop residual temple-arm stubs from a front-on glasses cutout: keep only the
    column range spanned by the TALL lens block (arms are thin strips at the far
    left/right). No-op when the front already fills the width."""
    a = rgba[:, :, 3]
    op = a > 40
    cols = np.where(op.any(0))[0]
    if len(cols) < 2:
        return rgba
    ys = np.where(op, np.arange(a.shape[0])[:, None], -1)
    top = np.where(op.any(0), op.argmax(0), 0)
    bot = a.shape[0] - 1 - np.where(op[::-1].any(0), op[::-1].argmax(0), 0)
    span = np.where(op.any(0), bot - top, 0)
    keep = np.where(span > 0.5 * span.max())[0]
    if len(keep) < 2:
        return rgba
    x0, x1 = int(keep.min()), int(keep.max())
    if x0 < a.shape[1] * 0.06 and x1 > a.shape[1] * 0.94:
        return rgba                                  # already armless — leave it
    pad = int((x1 - x0) * 0.02)
    return np.ascontiguousarray(rgba[:, max(0, x0 - pad):min(a.shape[1], x1 + pad + 1)])


def _eyewear_quad(face, rgba, wscale=1.9, down=0.06):
    """Place a front-on (ARMLESS) glasses cutout on the eyes, preserving its aspect
    ratio so frames aren't distorted. Lenses land on the pupils; seats on the eye
    line and follows head roll/yaw."""
    h, w = rgba.shape[:2]
    ex, ey = _frame_axes(face)
    c = face.eyes_center + ey * (face.eye_dist * down)
    W = face.eye_dist * wscale
    H = W * (h / max(1, w))
    hw = ex * (W / 2.0); hh = ey * (H / 2.0)
    return [c - hw - hh, c + hw - hh, c + hw + hh, c - hw + hh]


def glassify(rgba):
    """OpenCV realism pass for eyewear: turn an enclosed 'clear lens' (white on the
    studio backdrop) into SEE-THROUGH glass — eyes show through — with a cool tint
    and a diagonal sheen. Dark/tinted (sun) lenses are detected and left opaque."""
    rgba = rgba.copy()
    bgr = rgba[:, :, :3]; a = rgba[:, :, 3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lens = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 195) & (a > 180)).astype(np.uint8) * 255
    lens = cv2.morphologyEx(lens, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lm = lens > 0
    if not lm.any():
        return rgba                              # sunglasses / tinted → keep opaque
    a[lm] = 64                                    # see-through
    bgr[lm] = (bgr[lm] * 0.35 + np.array([62, 47, 38]) * 0.65).astype(np.uint8)  # cool glass
    h, w = a.shape                               # diagonal sheen across the lens band
    yy, xx = np.mgrid[0:h, 0:w]
    diag = (xx + yy).astype(np.float32); diag /= max(1.0, diag.max())
    sheen = ((diag > 0.40) & (diag < 0.47)).astype(np.float32)
    sheen = cv2.GaussianBlur(sheen, (0, 0), max(2, w // 200)) * lm
    bgr[:] = np.clip(bgr.astype(np.float32) + sheen[..., None] * 230, 0, 255).astype(np.uint8)
    a[:] = np.clip(a.astype(np.float32) + sheen * 130, 0, 255).astype(np.uint8)
    return rgba


# ---- live eyewear try-on: warp the actual selected product onto the face ----
import threading as _threading  # noqa: E402
_EYEWEAR = {"rgba": None, "label": ""}
_EYEWEAR_LOCK = _threading.Lock()


def _knockout_bg(bgr, existing_alpha=None):
    """Remove the studio background by flood-filling near-white from the borders —
    so only the OUTER background is cut (interior white, e.g. white frames or lens
    glare, is preserved). Combined with any real alpha already present."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 185)).astype(np.uint8)
    ff = white.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for sx, sy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
                   (w // 2, 0), (w // 2, h - 1)):
        if ff[sy, sx]:
            cv2.floodFill(ff, mask, (sx, sy), 2)
    alpha = np.where(ff == 2, 0, 255).astype(np.uint8)
    if existing_alpha is not None:
        alpha = np.minimum(alpha, existing_alpha)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)        # feather for clean compositing
    return alpha


def load_eyewear_rgba(raw: bytes):
    """Decode a product image to a tight RGBA cutout with a clean transparent
    background (handles both real-alpha PNGs and opaque white-bg product shots)."""
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        bgr, a0 = img[:, :, :3], img[:, :, 3]
        # if the alpha is effectively opaque, the "transparency" is fake → knock bg
        alpha = a0 if int(a0.min()) < 200 else _knockout_bg(bgr, a0)
    else:
        bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = _knockout_bg(bgr)
    rgba = np.dstack([bgr, alpha])
    ys, xs = np.where(alpha > 12)                       # tight-crop to the frame
    if len(xs):
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgba


def set_current_eyewear(rgba, label=""):
    with _EYEWEAR_LOCK:
        _EYEWEAR["rgba"] = rgba
        _EYEWEAR["label"] = label


def filter_eyewear(ctx):
    """Real-time try-on of the SELECTED eyewear product (set_current_eyewear) —
    warped onto each face's eyes with the dense landmarks, following head pose,
    with a soft contact shadow on the nose/cheeks for depth (so it reads as WORN,
    not pasted)."""
    with _EYEWEAR_LOCK:
        rgba = _EYEWEAR["rgba"]
    if rgba is None:
        return
    for f in ctx.faces:
        q = _eyewear_quad(f, rgba)
        ex, ey = _frame_axes(f)
        # soft contact shadow: a dark, slightly-down-offset copy of the frame alpha
        shadow = np.zeros_like(rgba)
        shadow[:, :, 3] = (rgba[:, :, 3].astype(np.float32) * 0.42).astype(np.uint8)
        off = ey * (f.eye_dist * 0.06)
        ctx.warp(shadow, [p + off for p in q])
        ctx.warp(rgba, q)


# ================================ the filters =================================
def filter_sunglasses(ctx):
    for f in ctx.faces:
        ctx.warp(ctx.asset("sunglasses"), _eyes_quad(f))


def filter_glasses(ctx):
    for f in ctx.faces:
        ctx.warp(ctx.asset("glasses"), _eyes_quad(f))


def filter_mustache(ctx):
    for f in ctx.faces:
        center = (f.p("nose_bottom") + f.p("lip_top")) / 2
        ctx.sticker(ctx.asset("mustache"), center, f.eye_dist * 1.15, angle=f.roll)


def filter_clown_nose(ctx):
    for f in ctx.faces:
        ctx.sticker(ctx.asset("clown_nose"), f.p("nose_tip"), f.eye_dist * 0.6,
                    angle=f.roll)


def filter_crown(ctx):
    for f in ctx.faces:
        ex, ey = _frame_axes(f)
        center = f.p("forehead") - ey * (f.face_h * 0.30)
        ctx.sticker(ctx.asset("crown"), center, f.face_w * 1.05, angle=f.roll)


def filter_dog(ctx):
    for f in ctx.faces:
        ex, ey = _frame_axes(f)
        up = -ey
        # floppy ears at the temples, angled outward
        lc = f.p("temple_l") + up * (f.face_h * 0.10) - ex * (f.face_w * 0.10)
        rc = f.p("temple_r") + up * (f.face_h * 0.10) + ex * (f.face_w * 0.10)
        ear = ctx.asset("dog_ear")
        ctx.sticker(ear, lc, f.face_w * 0.42, angle=f.roll - 28)
        ctx.sticker(cv2.flip(ear, 1), rc, f.face_w * 0.42, angle=f.roll + 28)
        ctx.sticker(ctx.asset("dog_nose"), f.p("nose_tip"), f.eye_dist * 0.7, angle=f.roll)
        # tongue out when the mouth opens
        if f.mouth_open > 0.28:
            t = (f.p("lip_bot") + f.p("chin")) / 2
            ctx.fill_poly([f.p("mouth_l"), f.p("mouth_r"),
                           t + ex * 12, t, t - ex * 12], color=(150, 150, 240), alpha=0.95)


def filter_cat(ctx):
    for f in ctx.faces:
        ex, ey = _frame_axes(f); up = -ey
        lc = f.p("temple_l") + up * (f.face_h * 0.22) - ex * (f.face_w * 0.02)
        rc = f.p("temple_r") + up * (f.face_h * 0.22) + ex * (f.face_w * 0.02)
        ear = ctx.asset("cat_ear")
        ctx.sticker(ear, lc, f.face_w * 0.34, angle=f.roll - 12)
        ctx.sticker(ear, rc, f.face_w * 0.34, angle=f.roll + 12)
        # whiskers + pink nose
        nose = f.p("nose_tip")
        ctx.sticker(ctx.asset("clown_nose"), nose, f.eye_dist * 0.34, angle=f.roll)
        for s in (-1, 1):
            base = nose + ex * (s * f.eye_dist * 0.25)
            for dy in (-0.10, 0.0, 0.10):
                tip = base + ex * (s * f.eye_dist * 0.9) + ey * (f.eye_dist * dy)
                ctx.line(base, tip, color=(245, 245, 245), thick=2)


def filter_face_mesh(ctx):
    """The classic dense tesselation — a vivid demo that landmarks are tracking."""
    conns = _mesh_connections()
    for f in ctx.faces:
        lm = f.lm.astype(int)
        if conns is not None:
            for a, b in conns:
                cv2.line(ctx.frame, tuple(lm[a]), tuple(lm[b]),
                         (180, 255, 120), 1, cv2.LINE_AA)
        for (x, y) in lm[::3]:
            cv2.circle(ctx.frame, (int(x), int(y)), 1, (60, 230, 250), -1, cv2.LINE_AA)


def filter_anonymize(ctx):
    """Privacy: pixelate every detected face (mainstream/clinical use)."""
    for f in ctx.faces:
        hull = cv2.convexHull(f.oval())
        pad = (hull.reshape(-1, 2) - f.eyes_center) * 1.12 + f.eyes_center
        ctx.pixelate(pad.astype(int), blocks=12)


def filter_ninja_mask(ctx):
    """🥷 The Vision Ninja hood: black balaclava over forehead/cheeks/jaw with an
    eye slit, plus a red headband. Built from the face oval + landmarks."""
    for f in ctx.faces:
        ex, ey = _frame_axes(f); up = -ey
        oval = f.oval().reshape(-1, 2).astype(np.float32)
        cen = oval.mean(0)
        hood = ((oval - cen) * 1.16 + cen)                  # enlarge to cover edges
        # extend the top of the hood up over the forehead/hair
        topmask = hood[:, 1] < f.p("forehead")[1]
        hood[topmask] += up * (f.face_h * 0.34)
        # eye slit: keep the original pixels across the eyes (brow → mid-nose)
        slit_h = f.eye_dist * 0.62
        sc = f.eyes_center + ey * (f.eye_dist * 0.02)
        hw = ex * (f.face_w * 0.60); hh = ey * (slit_h / 2)
        slit = np.array([sc - hw - hh, sc + hw - hh, sc + hw + hh, sc - hw + hh])
        # save slit pixels, paint hood, restore slit (rounded)
        x, y, w, h = cv2.boundingRect(slit.astype(np.int32))
        x, y = max(0, x), max(0, y); w = min(ctx.w - x, w); h = min(ctx.h - y, h)
        keep = ctx.frame[y:y + h, x:x + w].copy() if (w > 0 and h > 0) else None
        ctx.fill_poly(hood.astype(int), color=(18, 18, 20), alpha=1.0)
        if keep is not None:
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, (slit.astype(np.int32) - [x, y]), 255)
            m = cv2.GaussianBlur(m, (0, 0), 6)[..., None].astype(np.float32) / 255.0
            roi = ctx.frame[y:y + h, x:x + w].astype(np.float32)
            ctx.frame[y:y + h, x:x + w] = (keep * m + roi * (1 - m)).astype(np.uint8)
        # red headband across the forehead with two trailing tails
        bc = f.p("forehead") + up * (f.face_h * 0.06)
        bw = ex * (f.face_w * 0.62); bh = ey * (f.eye_dist * 0.14)
        ctx.fill_poly([bc - bw - bh, bc + bw - bh, bc + bw + bh, bc - bw + bh],
                      color=(40, 40, 200), alpha=1.0)
        knot = bc + bw
        tail = knot + ex * (f.face_w * 0.28) + ey * (f.eye_dist * 0.5)
        ctx.fill_poly([knot - bh * 0.6, knot + bh * 0.6, tail + bh, tail - bh * 0.3],
                      color=(36, 36, 180), alpha=1.0)


def filter_heart_eyes(ctx):
    """Hearts pop over the eyes when you smile (blendshape-reactive)."""
    for f in ctx.faces:
        if f.smile < 0.35:
            continue
        s = min(1.0, f.smile) * f.eye_dist * 0.7
        for eye in (f.eye_l, f.eye_r):
            _draw_heart(ctx.frame, eye, s, (90, 90, 240))


def _draw_heart(img, center, size, color):
    cx, cy = float(center[0]), float(center[1]); s = size / 2
    pts = []
    for th in np.linspace(0, 2 * np.pi, 40):
        x = 16 * np.sin(th) ** 3
        y = -(13 * np.cos(th) - 5 * np.cos(2 * th) - 2 * np.cos(3 * th) - np.cos(4 * th))
        pts.append((cx + x / 16 * s, cy + y / 16 * s))
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


_MESH_CONN = None


def _mesh_connections():
    global _MESH_CONN
    if _MESH_CONN is not None:
        return _MESH_CONN if _MESH_CONN else None
    try:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
        conns = []
        for grp in ("FACE_LANDMARKS_TESSELATION",):
            for e in getattr(C, grp):
                conns.append((e.start, e.end))
        _MESH_CONN = conns
    except Exception:
        _MESH_CONN = []
    return _MESH_CONN if _MESH_CONN else None


def _src(fn):
    try:
        return inspect.getsource(fn)
    except Exception:
        return f"# native filter {fn.__name__}"


FACE_FILTERS = {
    "eyewear": ("Live try-on of the selected eyewear product, warped to your eyes.",
                filter_eyewear, _src(filter_eyewear)),
    "ninja_mask": ("🥷 Black ninja hood with eye slit + red headband (Vision Ninja).",
                   filter_ninja_mask, _src(filter_ninja_mask)),
    "sunglasses": ("Realistic sunglasses tracked to your eyes (follows head pose).",
                   filter_sunglasses, _src(filter_sunglasses)),
    "glasses": ("Clear eyeglasses tried on, tracked to your eyes.",
                filter_glasses, _src(filter_glasses)),
    "dog": ("Dog filter: floppy ears + nose, tongue out when you open your mouth.",
            filter_dog, _src(filter_dog)),
    "cat": ("Cat filter: ears, pink nose, whiskers.", filter_cat, _src(filter_cat)),
    "mustache": ("A curly mustache on your lip.", filter_mustache, _src(filter_mustache)),
    "crown": ("A golden crown above your head.", filter_crown, _src(filter_crown)),
    "clown_nose": ("A red clown nose.", filter_clown_nose, _src(filter_clown_nose)),
    "heart_eyes": ("Hearts over your eyes when you smile (blendshape-reactive).",
                   filter_heart_eyes, _src(filter_heart_eyes)),
    "face_mesh": ("The dense 478-point face mesh tesselation.",
                  filter_face_mesh, _src(filter_face_mesh)),
    "anonymize": ("Privacy: pixelate every detected face.",
                  filter_anonymize, _src(filter_anonymize)),
}
