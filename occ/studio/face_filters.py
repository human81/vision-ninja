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


# named lens tints (BGR). "clear" → see-through glass; others → coloured sunglasses.
_LENS_TINTS = {
    "clear": None, "smoke": (42, 42, 48), "grey": (70, 70, 72), "gray": (70, 70, 72),
    "dark": (22, 22, 26), "black": (15, 15, 15), "brown": (28, 52, 92),
    "amber": (22, 95, 152), "blue": (152, 92, 42), "green": (46, 92, 46),
    "rose": (120, 92, 175), "purple": (140, 70, 120), "gold": (45, 175, 215),
    "mirror": (205, 205, 205), "silver": (180, 180, 185),
}


_GLASS_TINT = (60, 46, 36)          # faint cool glass (BGR) for see-through lenses
_CLEAR_A = 46                       # see-through alpha (low → the live eye shows through)
_SUN_A = 200                        # tinted/sun lens alpha (slightly translucent)


def _ellipse(k):
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _lens_regions(rgba):
    """Heavy-duty COLOUR-AWARE lens isolation → ([(interior_mask, (cx,cy)), …≤2], holes).
    The lens is the smooth low-detail area INSIDE the frame; the frame/bridge is the
    coloured or dark material. We treat the frame as separator — which naturally SPLITS
    the two lenses (the coloured bridge between them) even when the whole front is one
    opaque silhouette (the case morphology alone can't split). `holes` = enclosed
    transparent pixels (true clear openings)."""
    a = rgba[:, :, 3]; bgr = rgba[:, :, :3]; h, w = a.shape
    binm = (a > 40).astype(np.uint8)
    ff = binm.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)   # outer bg → 1
    filled = (binm | (ff == 0)).astype(np.uint8)                       # silhouette
    holes = ((ff != 1) & (binm == 0)).astype(np.uint8)                # enclosed transparent
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    # FRAME material = coloured (saturated) OR dark, where opaque. Everything else
    # inside the silhouette is lens (clear/white) or a transparent opening.
    frame = ((sat > 55) | (val < 105)) & (binm > 0)
    inside = cv2.erode(filled, _ellipse(max(3, int(0.02 * h)))) > 0     # drop the outer rim
    lens_mask = ((inside & ~frame) | (holes > 0)).astype(np.uint8)
    lens_mask = cv2.morphologyEx(lens_mask, cv2.MORPH_OPEN,             # de-speckle only
                                 _ellipse(max(4, int(0.04 * h))))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(lens_mask, 8)
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.012 * h * w]
    cand = sorted(cand, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]
    # tuck a couple px UNDER the inner rim so the lens meets the frame with NO gap ring
    tuck = _ellipse(max(2, int(0.018 * h)))
    smooth = _ellipse(max(3, int(0.02 * h)))
    out = []
    for i in cand:
        comp = (lbl == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        sol = np.zeros((h, w), np.uint8)                               # EXACT lens shape (not a hull)
        cv2.drawContours(sol, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
        sol = cv2.morphologyEx(sol, cv2.MORPH_CLOSE, smooth)           # smooth, continuous edge
        sol = cv2.dilate(sol, tuck) & filled                          # meet the rim, no gap
        interior = sol > 0
        if interior.any():
            out.append((interior, (float(cent[i][0]), float(cent[i][1]))))
    out.sort(key=lambda r: r[1][0])                                    # left → right
    return out, holes


def lens_centers_norm(rgba):
    """The two lens centres in normalised (x,y) ∈ [0,1], left→right, for REGISTRATION
    onto the pupils. GEOMETRIC + colour-independent (works on clear, sun, AND patterned
    novelty lenses, where colour-based detection fails): find the nose BRIDGE as the
    narrowest silhouette column near centre, then take the CENTROID of the glasses
    silhouette on each side of it = the two lens centres. Validated for plausibility;
    None → symmetric fallback."""
    a = rgba[:, :, 3]; h, w = a.shape
    binm = (a > 40).astype(np.uint8)
    ff = binm.copy(); cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    silh = ((binm | (ff == 0)) > 0)
    if int(silh.sum()) < max(200, 0.01 * h * w):
        return None
    # Use the LENS BAND only — the columns where the silhouette is TALL (the lenses) —
    # ignoring thin temple stubs, so the centre is robust and front-on-symmetric. (The
    # old 'narrowest column' bridge search failed on solid flag-lens blocks → left shift.)
    cols = silh.sum(0).astype(np.float32)
    rows = silh.sum(1).astype(np.float32)
    tall = np.where(cols > 0.5 * cols.max())[0]
    wide = np.where(rows > 0.5 * rows.max())[0]
    if len(tall) < 10 or len(wide) < 5:
        return None
    cx = 0.5 * (int(tall.min()) + int(tall.max()))        # centre of the lens band
    cy = 0.5 * (int(wide.min()) + int(wide.max()))        # vertical centre of the lens band
    midi = int(round(cx))

    def half(x0, x1):                                      # centroid of TALL columns on a side
        cc = tall[(tall >= x0) & (tall < x1)]
        return None if len(cc) < 5 else float(cc.mean())

    lxm, rxm = half(0, midi), half(midi, w)
    if lxm is None or rxm is None:
        return None
    d = 0.5 * ((cx - lxm) + (rxm - cx))                   # symmetric half-separation
    lx, rx = (cx - d) / w, (cx + d) / w
    ly = ry = cy / h
    dx = rx - lx
    ok = (0.20 < dx < 0.80 and 0.32 < (lx + rx) / 2 < 0.68
          and 0.06 < lx < 0.46 and 0.54 < rx < 0.94 and 0.20 < ly < 0.80)
    return [(lx, ly), (rx, ry)] if ok else None


def clean_lenses(rgba, tint=None, opacity=None):
    """Rebuild the lenses as REAL see-through glass — and GUARANTEE no opaque white
    blob survives anywhere in the lens area (the bug that left white marks). Each lens
    interior is repainted: clear frames → low-alpha cool glass (the live eye shows
    through), tinted/sun frames → their own colour, slightly translucent. A hard
    safety net then forces ANY near-white enclosed (lens-area) pixel to see-through, so
    studio-backdrop bleed can never read as a white lens. Frame/rim pixels (incl. white
    acetate frames) are untouched.
      tint    — None=match original; 'clear'; a name in _LENS_TINTS; or a BGR tuple.
      opacity — None=auto (clear≈46, tinted≈200); else 1-255 (low = more see-through)."""
    rgba = rgba.copy(); a = rgba[:, :, 3]; bgr = rgba[:, :, :3]; h, w = a.shape
    regions, holes = _lens_regions(rgba)
    if not regions:
        return glassify(rgba, tint, opacity)        # couldn't isolate lenses → safe fallback
    # resolve any explicit tint override
    clear_force, forced = False, None
    if isinstance(tint, str):
        t = tint.lower().strip()
        if t == "clear":
            clear_force = True
        elif t in _LENS_TINTS and _LENS_TINTS[t] is not None:
            forced = _LENS_TINTS[t]
    elif tint is not None:
        forced = tuple(int(c) for c in tint)

    hsv0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lens_union = np.zeros((h, w), bool)
    for interior, _c in regions:
        frac_hole = float((holes[interior] > 0).mean())          # was it a transparent opening?
        med = np.median(bgr[interior].reshape(-1, 3), 0)
        v = float(np.median(hsv0[:, :, 2][interior])); s = float(np.median(hsv0[:, :, 1][interior]))
        is_clear = clear_force or (forced is None and (frac_hole > 0.30 or (v > 150 and s < 60)))
        if not is_clear and forced is None and max(med) > 232 and (max(med) - min(med)) < 24:
            is_clear = True                                      # near-white "tint" = clear lens
        idx = interior
        if forced is not None:                                   # explicit tint override → flat
            a[idx] = int(opacity) if opacity else _SUN_A
            bgr[idx] = np.array(forced, np.uint8)
        elif is_clear:
            # KEEP the aligned lens pixels, just make them SEE-THROUGH (low alpha). A light
            # cool-glass blend stops a pure-white studio backdrop reading as a milky veil,
            # while preserving the real lens texture/reflections.
            a[idx] = int(opacity) if opacity else _CLEAR_A
            bgr[idx] = (bgr[idx] * 0.62 + np.array(_GLASS_TINT, np.float32) * 0.38).astype(np.uint8)
        else:
            # tinted / sun: KEEP the real lens colour, just slightly translucent.
            a[idx] = int(opacity) if opacity else _SUN_A
        lens_union |= interior

    # ---- HARD SAFETY NET: no opaque white may survive ANYWHERE inside the frame ----
    # (lens area + enclosed gaps like the nose-bridge cutout). We lower the ALPHA of any
    # near-white pixel in the DEEP interior — protecting thin frame rims (incl. white
    # acetate, which sits on the silhouette boundary and is eroded away here).
    binm = (a > 40).astype(np.uint8)
    ff = binm.copy(); cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    filled = (binm | (ff == 0)).astype(np.uint8)
    deep = cv2.erode(filled, _ellipse(max(4, int(0.05 * h)))) > 0     # interior, not the rim
    hsv1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    near_white = (hsv1[:, :, 2] > 224) & (hsv1[:, :, 1] < 36) & (a > 110)
    enclosed = (holes > 0) | lens_union | deep                       # lens + enclosed gaps
    enclosed = cv2.dilate(enclosed.astype(np.uint8), _ellipse(max(3, int(0.03 * h)))) > 0
    kill = near_white & enclosed
    if kill.any():
        a[kill] = int(opacity) if (opacity and clear_force) else _CLEAR_A
        bgr[kill] = (bgr[kill] * 0.62 + np.array(_GLASS_TINT, np.float32) * 0.38).astype(np.uint8)
        lens_union |= kill

    # subtle per-lens glint (small soft highlight — NOT a full-width diagonal streak)
    for interior, c in regions:
        ys, xs = np.where(interior)
        if len(xs) < 30:
            continue
        diam = np.sqrt(len(xs))
        gx = int(c[0] - 0.18 * diam); gy = int(c[1] - 0.22 * diam)
        g = np.zeros((h, w), np.float32)
        cv2.circle(g, (gx, gy), max(2, int(0.16 * diam)), 1.0, -1)
        g = cv2.GaussianBlur(g, (0, 0), max(1.5, diam / 14)) * interior
        if g.max() > 0:
            g /= g.max()
            bgr[:] = np.clip(bgr + g[..., None] * 70, 0, 255).astype(np.uint8)
            a[:] = np.clip(a + (g * 55).astype(np.uint8), 0, 255).astype(np.uint8)
    return rgba


def remove_arms(rgba):
    """Mask a front-on glasses cutout down to JUST the two lens discs + the bridge
    between them — dropping temple arms/hinge stubs entirely (they look unnatural
    laid flat across the face). Robust to clear OR sunglasses lenses:
      fill lens holes → morphological OPEN (thin arms+bridge vanish, lens cores
      remain) → keep the 2 largest blobs (lenses) → keep = dilated-lenses ∪ the
      column band between the two lens centres (re-includes the bridge/brow bar,
      excludes the outer arms)."""
    a = rgba[:, :, 3]
    h, w = a.shape
    binm = (a > 40).astype(np.uint8)
    ff = binm.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)   # background → 1
    filled = (binm | (ff == 0)).astype(np.uint8)                        # + interior holes
    k = max(9, int(0.22 * h))
    cores = cv2.morphologyEx(filled, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(cores, 8)
    if n < 3:
        return trim_arms(rgba)                       # couldn't split lenses → fall back
    idx = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]
    keepcore = np.isin(lbl, idx).astype(np.uint8)
    cx = sorted(int(cent[i][0]) for i in idx)
    dk = max(5, int(0.09 * h))
    dil = cv2.dilate(keepcore, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)))
    band = np.zeros_like(a); band[:, cx[0]:cx[1] + 1] = 1
    keep = (dil > 0) | (band > 0)
    out = rgba.copy()
    out[:, :, 3] = np.where(keep, a, 0)
    cols = np.where((out[:, :, 3] > 20).any(0))[0]
    rows = np.where((out[:, :, 3] > 20).any(1))[0]
    if len(cols) and len(rows):
        out = out[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return np.ascontiguousarray(out)


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


def _eyewear_quad(face, rgba, lens_centers=None, down=0.06):
    """REGISTER the frames to the face: solve the 2-point similarity (scale+roll+
    translation) that maps the asset's two LENS CENTRES onto the user's two PUPILS, then
    transform the asset rectangle's corners through it. This lands each lens exactly on
    its eye and sizes the frame to the real interpupillary distance — far more accurate
    than a fixed box. Aspect ratio is preserved (uniform scale → no stretch). Falls back
    to a symmetric 27/73% lens assumption if the asset's lens centres are unknown."""
    h, w = rgba.shape[:2]
    if lens_centers is None:
        lens_centers = [(0.27, 0.46), (0.73, 0.46)]
    aL = np.array([lens_centers[0][0] * w, lens_centers[0][1] * h], float)
    aR = np.array([lens_centers[1][0] * w, lens_centers[1][1] * h], float)
    pL = np.array(face.eye_l, float); pR = np.array(face.eye_r, float)
    da = aR - aL; dp = pR - pL
    na = np.linalg.norm(da) or 1.0
    ipd = np.linalg.norm(dp) or na
    scale = ipd / na
    # CLAMP: the full frame width = scale*w must stay a sane multiple of the IPD, so a
    # mis-detected lens separation can't blow the frame up or shrink it to nothing.
    fw = scale * w
    scale *= float(np.clip(fw, 1.9 * ipd, 3.4 * ipd) / max(fw, 1e-6))
    # ROTATION: follow the head roll (from the pupils), not the asset's tilt — keeps the
    # frames level even if the detected lens centres aren't perfectly aligned.
    ang = np.arctan2(dp[1], dp[0])
    cs, sn = np.cos(ang) * scale, np.sin(ang) * scale
    R = np.array([[cs, -sn], [sn, cs]])
    # anchor on the asset's lens MIDPOINT → the pupils' midpoint (robust to per-lens noise)
    aM = (aL + aR) / 2.0; pM = (pL + pR) / 2.0
    def tf(p):
        return pM + R @ (np.array(p, float) - aM)
    _, ey = _frame_axes(face)                        # seat slightly down the nose
    off = ey * (face.eye_dist * down)
    return [tf((0, 0)) + off, tf((w, 0)) + off, tf((w, h)) + off, tf((0, h)) + off]


def glassify(rgba, tint=None, opacity=None):
    """Safe fallback when the two lenses can't be isolated: turn any enclosed near-white
    lens pixels into SEE-THROUGH glass (eye shows through) and force ANY enclosed white
    blob transparent — so no white mark is left. Tinted/sun lenses stay as-is. NO
    diagonal sheen (that was the source of the white streak)."""
    rgba = rgba.copy()
    bgr = rgba[:, :, :3]; a = rgba[:, :, 3]; h, w = a.shape
    binm = (a > 40).astype(np.uint8)
    ff = binm.copy(); cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    holes = ((ff != 1) & (binm == 0))                 # enclosed transparent = clear openings
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 42) & (hsv[:, :, 2] > 200) & (a > 150)
    # only treat white that is ENCLOSED (lens area), never the outer rim / white frames
    enclosed = cv2.dilate(holes.astype(np.uint8), _ellipse(max(5, int(0.06 * h)))) > 0
    lens = cv2.morphologyEx((white & enclosed).astype(np.uint8),
                            cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    if not lens.any():
        return rgba                                   # no white lens to fix → leave it
    a[lens] = int(opacity) if opacity else _CLEAR_A
    bgr[lens] = _GLASS_TINT
    return rgba


# ---- live eyewear try-on: warp the actual selected product onto the face ----
import threading as _threading  # noqa: E402
_EYEWEAR = {"rgba": None, "armless": None, "label": "", "color": (45, 45, 45),
            "tint": None, "opacity": None}
_EYEWEAR_LOCK = _threading.Lock()


def frame_color(rgba):
    """Median colour of the FRAME (opaque, non-lens, non-white pixels) — used to
    paint the synthesized temple arms so they match the frame."""
    a = rgba[:, :, 3]; bgr = rgba[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rim = (a > 200) & ~((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 190))
    if int(rim.sum()) < 20:
        return (45, 45, 45)
    return tuple(int(x) for x in np.median(bgr[rim].reshape(-1, 3), 0))


def _knockout_bg(bgr, existing_alpha=None):
    """Remove the studio background = the LIGHT region connected to the image border.
    Robust to off-white / gradient / vignetted backgrounds (the wide Nano-Pro renders
    whose corners weren't pure white, which left an opaque rectangle): we flood-fill the
    light-and-low-saturation mask from MANY border seeds, so the whole outer background
    is cut while interior light (white frames / lens glare) stays. Then floor faint
    alpha to fully kill any translucent halo."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    light = ((hsv[:, :, 1] < 62) & (hsv[:, :, 2] > 165)).astype(np.uint8)
    ff = light.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    sx = max(1, w // 60); sy = max(1, h // 60)
    seeds = ([(x, 0) for x in range(0, w, sx)] + [(x, h - 1) for x in range(0, w, sx)] +
             [(0, y) for y in range(0, h, sy)] + [(w - 1, y) for y in range(0, h, sy)])
    for px, py in seeds:
        if ff[py, px] == 1:
            cv2.floodFill(ff, mask, (px, py), 2)
    alpha = np.where(ff == 2, 0, 255).astype(np.uint8)
    if existing_alpha is not None:
        alpha = np.minimum(alpha, existing_alpha)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)        # feather for clean compositing
    alpha[alpha < 30] = 0                               # floor: no translucent halo rectangle
    return alpha


def _keep_glasses(bgr, alpha):
    """SEGMENTATION safety net — guarantee NOTHING outside the actual frame survives
    (no stray rectangle / blob, ever). Full arsenal:
      1) foreground = the cutout alpha, FUSED with Canny EDGES (catches thin metal rims
         the colour cut can miss) and morphologically closed,
      2) keep only the GLASSES connected component(s): the largest blob plus any other
         blob ≥18% of it (the second lens of a rimless/semi-rimless pair),
      3) zero alpha everywhere else.
    This is true silhouette segmentation, not just a background flood-fill."""
    h, w = alpha.shape
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    edges = cv2.dilate(edges, _ellipse(max(2, int(0.012 * h))))
    fg = (((alpha > 45).astype(np.uint8) | (edges > 0)).astype(np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _ellipse(max(3, int(0.02 * h))))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    big = order[0]; a0 = areas[big]
    ty0 = stats[big + 1, cv2.CC_STAT_TOP]; th0 = stats[big + 1, cv2.CC_STAT_HEIGHT]
    keep = np.zeros((h, w), np.uint8); keep[lbl == (big + 1)] = 1     # the frame
    for i in order[1:4]:
        ty = stats[i + 1, cv2.CC_STAT_TOP]; th = stats[i + 1, cv2.CC_STAT_HEIGHT]
        ov = max(0, min(ty0 + th0, ty + th) - max(ty0, ty))          # vertical overlap
        # a 2nd glasses part (lens) shares the frame's vertical band; a stray bg blob doesn't
        if areas[i] >= 0.12 * a0 and ov > 0.35 * min(th0, th):
            keep[lbl == (i + 1)] = 1
    keep = cv2.dilate(keep, _ellipse(max(2, int(0.012 * h))))        # don't clip the frame edge
    return np.where(keep > 0, alpha, 0).astype(np.uint8)


_REMBG_SESSION = None
_REMBG_ON = os.environ.get("STUDIO_REMBG", "1") != "0"
_CUTOUT_CACHE = "out/studio/cache/cutout"


def _rembg_alpha(bgr):
    """A real SEGMENTATION/matting model (rembg · U2Net) → a clean alpha matte of the
    glasses, robust to any studio background. Process-wide session (loaded once).
    Returns the alpha channel, or None if rembg is unavailable / disabled."""
    global _REMBG_SESSION
    if not _REMBG_ON:
        return None
    try:
        from rembg import remove, new_session
        if _REMBG_SESSION is None:
            _REMBG_SESSION = new_session(os.environ.get("STUDIO_REMBG_MODEL", "u2net"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = remove(rgb, session=_REMBG_SESSION)
        out = np.asarray(out)
        if out.ndim == 3 and out.shape[2] == 4:
            return out[:, :, 3]
    except Exception as e:
        globals()["_REMBG_ON"] = False                # disable after a failure (graceful)
        print(f"[rembg] disabled: {type(e).__name__}: {e}")
    return None


def load_eyewear_rgba(raw: bytes):
    """Decode a product image to a tight RGBA cutout with a clean transparent
    background. Primary path: the rembg/U2Net SEGMENTATION matte (robust to any studio
    background); cached on disk by content hash. Falls back to the classical
    flood-fill + connected-component segmentation if rembg is unavailable."""
    import hashlib
    key = hashlib.md5(raw).hexdigest()
    cp = os.path.join(_CUTOUT_CACHE, key + ".png")
    if os.path.exists(cp) and os.path.getsize(cp) > 800:
        c = cv2.imdecode(np.frombuffer(open(cp, "rb").read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if c is not None and c.ndim == 3 and c.shape[2] == 4:
            return c
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    bgr = (img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    a0 = img[:, :, 3] if (img.ndim == 3 and img.shape[2] == 4) else None
    alpha = _rembg_alpha(bgr)                           # the segmentation model (preferred)
    if alpha is None:                                  # classical fallback
        if a0 is not None and float((a0 > 200).mean()) > 0.04:
            alpha = np.minimum(a0, _knockout_bg(bgr))
        else:
            alpha = _knockout_bg(bgr)
        alpha = _keep_glasses(bgr, alpha)              # safety net (the matte needs none)
    rgba = np.dstack([bgr, alpha])
    ys, xs = np.where(alpha > 12)                       # tight-crop to the frame
    if len(xs):
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    try:
        os.makedirs(_CUTOUT_CACHE, exist_ok=True)
        cv2.imwrite(cp, rgba)
    except Exception:
        pass
    return rgba


def set_current_eyewear(rgba, label="", color=None, tint=None, opacity=None):
    """`rgba` is the ARMLESS front (remove_arms output). We store it so the lenses
    can be re-tinted live, and paint the lenses now via clean_lenses(tint, opacity).
    Pass None to clear the current eyewear."""
    if rgba is None:
        with _EYEWEAR_LOCK:
            _EYEWEAR.update({"rgba": None, "armless": None, "label": "",
                             "reflection": None, "refl_layer": None})
        return
    regions, _ = _lens_regions(rgba)
    lm = np.zeros(rgba.shape[:2], np.uint8)
    for it, _c in regions:
        lm[it] = 1
    with _EYEWEAR_LOCK:
        _EYEWEAR["armless"] = rgba
        _EYEWEAR["tint"] = tint
        _EYEWEAR["opacity"] = opacity
        _EYEWEAR["rgba"] = clean_lenses(rgba, tint, opacity)
        _EYEWEAR["label"] = label
        _EYEWEAR["color"] = color or frame_color(rgba)
        _EYEWEAR["lens_centers"] = lens_centers_norm(rgba)   # for pupil registration
        _EYEWEAR["lensmask"] = lm                            # exact glass region (asset space)
        _EYEWEAR["reflection"] = None                        # a new pair starts with NO reflection
        _EYEWEAR["refl_layer"] = None


def set_lens_reflection(scene_bgr) -> bool:
    """Set (or clear with None) a STATIC scene that REALLY reflects in the see-through
    lenses of the current glasses — eyes still show through. Computational photography,
    not a video paste. The reflection LAYER is built ONCE here (it's static) and cached,
    so the live loop just warps it (fast)."""
    with _EYEWEAR_LOCK:
        if _EYEWEAR.get("armless") is None:
            return False
        _EYEWEAR["reflection"] = scene_bgr
        lm = _EYEWEAR.get("lensmask")
        _EYEWEAR["refl_layer"] = (_build_reflection(_EYEWEAR["armless"].shape, lm, scene_bgr)
                                  if (scene_bgr is not None and lm is not None) else None)
        return True


def _build_reflection(shape, lensmask, scene, strength=0.78):
    """Compose a photoreal glass REFLECTION layer (RGBA, asset space) of `scene` inside
    the lens region — 2026 computational-photography pass:
      • the scene is mapped per-lens with a slight BARREL warp (curved-glass refraction),
      • a FRESNEL gradient (brighter toward the top/sky) and a RIM glow,
      • the centre stays dim so the EYE shows through (real transparency),
      • a luminous diagonal SPECULAR streak (the classic glass glint),
      • a faint chromatic rim fringe. Eyes remain visible; nothing reads as opaque."""
    h, w = shape[:2]
    s = cv2.resize(scene, (w, h)).astype(np.float32)
    s = np.clip(s * 1.16 + 14, 0, 255)                          # glassy luminance lift
    refl = np.zeros((h, w, 4), np.uint8)
    m = lensmask.astype(np.float32)
    if not m.any():
        return refl
    # per-lens barrel warp: pull the scene toward each lens centre a touch (refraction)
    regs = cv2.connectedComponents(lensmask.astype(np.uint8))[1]
    mapx, mapy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for li in range(1, regs.max() + 1):
        ys, xs = np.where(regs == li)
        if len(xs) < 30:
            continue
        cx, cy = xs.mean(), ys.mean(); rad = max(np.ptp(xs), np.ptp(ys)) / 2 + 1
        dxm = (mapx - cx) / rad; dym = (mapy - cy) / rad
        r2 = dxm * dxm + dym * dym
        k = 0.14                                                # barrel strength
        sel = regs == li
        mapx[sel] = (cx + (mapx - cx) * (1 - k * r2))[sel]
        mapy[sel] = (cy + (mapy - cy) * (1 - k * r2))[sel]
    s = cv2.remap(s, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # specular diagonal streak (bright glass glint)
    diag = (mapx + mapy); diag = diag / (diag.max() or 1.0)
    streak = np.exp(-((diag - 0.46) ** 2) / (2 * 0.018 ** 2)) * 150.0
    s = np.clip(s + streak[..., None], 0, 255)
    refl[:, :, :3] = s.astype(np.uint8)
    # alpha shaping: fresnel (top) × rim glow × strength, centre kept see-through
    ys = np.where(m.any(1))[0]; y0, y1 = int(ys.min()), int(ys.max())
    vg = np.clip(1.0 - (np.arange(h) - y0) / max(1, (y1 - y0)), 0.30, 1.0)[:, None]
    dist = cv2.distanceTransform(lensmask.astype(np.uint8), cv2.DIST_L2, 5)
    dist = dist / (dist.max() or 1.0)                           # 0 rim → 1 centre
    edge = 1.0 - 0.62 * dist
    a = m * vg * edge * strength * 255.0
    a = np.maximum(a, m * (streak * 0.6))                       # the glint stays bright
    refl[:, :, 3] = np.clip(a, 0, 235).astype(np.uint8)
    return refl


def set_lens_tint(tint=None, opacity=None) -> bool:
    """Re-paint the CURRENT glasses' lenses live (tint/opacity) without re-fetching —
    re-runs clean_lenses on the stored armless front. Returns False if nothing's on."""
    with _EYEWEAR_LOCK:
        if _EYEWEAR["armless"] is None:
            return False
        _EYEWEAR["tint"] = tint
        _EYEWEAR["opacity"] = opacity
        _EYEWEAR["rgba"] = clean_lenses(_EYEWEAR["armless"], tint, opacity)
        return True


def _draw_temple_arms(ctx, f, quad, color):
    """Synthesize the temple arms (legs) from each lens hinge BACK to the ears,
    using the dense landmarks, in the frame's colour — so the glasses read as truly
    worn. Drawn BEFORE the lens front so the rim covers the hinge join cleanly."""
    TL, TR, BR, BL = [np.array(p, float) for p in quad]
    ex, ey = _frame_axes(f)
    ec = f.eyes_center
    # connect near the TOP-outer corner of the frame, tucked well INWARD so the rim
    # covers the flat hinge end (arms are drawn BEFORE the front) — no visible stub.
    tuck = f.eye_dist * 0.11
    hinges = [TL * 0.72 + BL * 0.28 + ex * tuck,        # left: top-outer corner
              TR * 0.72 + BR * 0.28 - ex * tuck]        # right
    ears = [ec - ex * (f.face_w * 0.52) + ey * (f.eye_dist * 0.05),   # face-edge, ear height
            ec + ex * (f.face_w * 0.52) + ey * (f.eye_dist * 0.05)]
    th = max(3, int(f.eye_dist * 0.11))
    hi = tuple(min(255, c + 50) for c in color)
    dk = tuple(max(0, c - 30) for c in color)
    for hinge, ear in zip(hinges, ears):
        d = ear - hinge; n = np.linalg.norm(d) or 1.0
        p = np.array([-d[1], d[0]]) / n                    # perpendicular
        mid = (hinge + ear) / 2 + ey * (f.eye_dist * 0.04)  # slight downward bow to the ear
        poly = np.array([hinge + p * (th / 2), mid + p * (th * 0.40), ear + p * (th * 0.30),
                         ear - p * (th * 0.30), mid - p * (th * 0.40), hinge - p * (th / 2)],
                        np.int32)
        cv2.fillPoly(ctx.frame, [poly], color, cv2.LINE_AA)
        cv2.polylines(ctx.frame, [poly], True, dk, 1, cv2.LINE_AA)
        cv2.line(ctx.frame, tuple(hinge.astype(int)), tuple(ear.astype(int)), hi, 1, cv2.LINE_AA)


def filter_eyewear(ctx):
    """Real-time try-on of the SELECTED eyewear product (set_current_eyewear) —
    warped onto each face's eyes with the dense landmarks, following head pose, with
    SYNTHESIZED temple arms (frame-coloured, from the hinges back to the ears) and a
    soft contact shadow, so it reads as truly WORN, not pasted."""
    with _EYEWEAR_LOCK:
        rgba = _EYEWEAR["rgba"]; color = _EYEWEAR["color"]
        lc = _EYEWEAR.get("lens_centers")
        refl = _EYEWEAR.get("refl_layer")          # prebuilt once (static) → fast warp
    if rgba is None:
        return
    for f in ctx.faces:
        q = _eyewear_quad(f, rgba, lens_centers=lc)
        ex, ey = _frame_axes(f)
        # SOFT contact shadow: only the FRAME casts it (lenses are see-through), heavily
        # blurred + low alpha + a small offset, so it reads as a diffuse shadow — not a
        # hard dark duplicate of the frame (the old 'lens shadow' artifact).
        sa = rgba[:, :, 3].astype(np.float32)
        sa = np.where(sa > 150, sa, sa * 0.25)            # drop the faint see-through lens
        sa = cv2.GaussianBlur(sa, (0, 0), max(2.0, rgba.shape[0] * 0.05))
        shadow = np.zeros_like(rgba)
        shadow[:, :, 3] = np.clip(sa * 0.16, 0, 70).astype(np.uint8)
        off = ey * (f.eye_dist * 0.03)
        ctx.warp(shadow, [p + off for p in q])
        ctx.warp(rgba, q)
        if refl is not None:                              # REAL scene reflection in the glass
            ctx.warp(refl, q)


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
