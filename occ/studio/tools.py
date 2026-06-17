"""The Vision Ninja's tools — pipeline control, the OpenCV overlay ninja, ffmpeg,
the Vision Brain, and UI driving. Each is a plain function (ADK introspects the
docstring + type hints) that does real work and meters itself into the ledger.

The same functions back the real ADK agent AND the deterministic SimRunner, so
the studio behaves identically with or without an API key.
"""

from __future__ import annotations

import base64
import glob
import os
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np

from . import ffmpeg_ops as ff
from .overlays import BUILTINS
from .runtime import ctx


def _meter(checkpoint, **kw):
    c = ctx()
    fake = c.settings.is_node_fake(checkpoint.split(":", 1)[0]) if c.settings else False
    return c.ledger.record(checkpoint, simulated=fake, **kw) if c.ledger else None


def _has_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _lib(res, caption="", tags=None, source=""):
    """Auto-register a produced artifact into the media library."""
    c = ctx()
    if (isinstance(res, dict) and res.get("status") == "success"
            and res.get("output") and c.library):
        kind = res.get("kind", "image")
        c.library.add("image" if kind == "display" else kind, res["output"],
                      caption=caption or res.get("caption", ""), tags=tags or [kind],
                      source=source, meta={k: res[k] for k in ("counts", "resolution",
                      "seconds") if res.get(k) is not None})
    return res


# ---------- planning + UI driving ----------
def plan(steps: list) -> dict:
    """Lay out THIS turn's work as a short ordered checklist BEFORE doing it.
    Use when the task has >2 steps. Args: steps — list of one-line next actions."""
    s = [str(x) for x in (steps or [])][:10]
    return {"status": "success", "steps": s}


def drive_ui(action: str, target: str = "", title: str = "", caption: str = "",
             value: str = "") -> dict:
    """Drive the SCREEN to guide the user. $0, pure UI. Call a short sequence.
    actions: 'toast' (caption=message), 'spotlight' (target=CSS selector),
    'focus' (target = panel id: live|chat|meter|brain|overlays),
    'pulse_node' (target = neuron id, e.g. detect/overlay/agent_brain),
    'set_tab' (value = meter|brain|overlays)."""
    return {"status": "success", "action": action, "target": target,
            "title": title, "caption": caption, "value": value}


# ---------- pipeline control ----------
def set_source(source: str) -> dict:
    """Switch the live video source. `source` is a path under assets/videos/ or any
    file/RTSP URI. The live view updates immediately."""
    ctx().pipe.reconfigure({"source.uri": source})
    if ctx().brain:
        ctx().brain.observe({}, source=source)
    return {"status": "success", "source": source, "ui": "config"}


def set_detector(backend: str = "yolo", model: str = "", classes: str = "",
                 prompt: str = "", imgsz: int = 0, conf: float = 0.0) -> dict:
    """Configure detection. backend: yolo|rfdetr|owlv2. model: e.g. yolo11m.pt.
    classes: comma list to keep (e.g. 'car,truck'). prompt: open-vocab classes for
    owlv2 (e.g. 'forklift,stroller'). imgsz: inference size (640/960/1280).
    conf: confidence 0-1. owlv2 is the open-vocab path (slower)."""
    up = {"detector.backend": backend}
    if model:
        up["detector.model"] = model
    if prompt:
        up["detector.owlv2_prompt"] = prompt
    if imgsz:
        up["detector.imgsz"] = int(imgsz)
    if conf:
        up["detector.conf"] = float(conf)
    if classes:
        up["detector.classes"] = [c.strip() for c in classes.split(",") if c.strip()]
    ctx().pipe.reconfigure(up)
    return {"status": "success", "config": up, "ui": "config"}


_TASK_MODELS = {"detect": "yolo11n.pt", "segment": "yolo11n-seg.pt",
                "pose": "yolo11n-pose.pt", "obb": "yolo11n-obb.pt",
                "classify": "yolo11n-cls.pt"}


def set_task(task: str) -> dict:
    """Switch the YOLO vision TASK on the live video:
    detect (boxes) | segment (instance masks) | pose (human keypoints/skeleton) |
    obb (oriented bounding boxes, aerial/DOTA classes) | classify (whole-image label).
    'semantic' maps to segment. Loads the matching yolo11n model (downloads on 1st use)."""
    task = task.lower().strip()
    if task in ("semantic", "semantic_segmentation", "semantic segmentation"):
        task = "segment"
    if task not in _TASK_MODELS:
        return {"status": "error", "error": f"task must be one of {list(_TASK_MODELS)}"}
    ctx().pipe.reconfigure({"detector.backend": "yolo", "detector.model": _TASK_MODELS[task]})
    return {"status": "success", "task": task, "model": _TASK_MODELS[task], "ui": "config"}


def set_tracker(algorithm: str) -> dict:
    """Set the tracker: bytetrack|botsort|ocsort|sort."""
    ctx().pipe.reconfigure({"tracker.algorithm": algorithm})
    return {"status": "success", "tracker": algorithm, "ui": "config"}


def set_detect_every(n: int) -> dict:
    """Run detection every Nth frame (1=every frame, higher=cheaper/faster).
    Tracking holds objects between detections."""
    ctx().pipe.reconfigure({"runtime.detect_every": max(1, int(n))})
    return {"status": "success", "detect_every": max(1, int(n)), "ui": "config"}


def set_detection(on: bool = True) -> dict:
    """Turn object DETECTION + TRACKING + OCCUPANCY analysis ON/OFF on the live video.
    It is OFF by default once the user's camera is live (a selfie feed isn't a scene
    to analyze). Turn it ON when the user wants to detect / track / COUNT objects
    ('count the people', 'ring the cars', 'occupancy'). OFF = a clean pass-through —
    face filters / AR try-on / overlays still run, but no boxes, counts or compute."""
    state = ctx().pipe.set_detection(bool(on))
    return {"status": "success", "detect_on": state, "ui": "config"}


def set_render(boxes: bool = True, labels: bool = True, trails: bool = True,
               counts: bool = True) -> dict:
    """Control what is DRAWN on the live frame. Pass false to HIDE: `boxes` =
    detection boxes, `labels` = per-track id/class/conf text, `trails` = motion
    trails, `counts` = the FULL-FRAME count HUD. To clear the detection boxes /
    tracking the user sees, call set_render(boxes=false, labels=false, trails=false).
    The detector + tracker keep running underneath (counts/zones still update)."""
    ctx().pipe.set_render_flags(draw_boxes=boxes, draw_labels=labels,
                                draw_trails=trails, draw_counts=counts)
    return {"status": "success",
            "render": {"boxes": bool(boxes), "labels": bool(labels),
                       "trails": bool(trails), "counts": bool(counts)}, "ui": "config"}


# ---------- annotations (zones / lines) ----------
def _xy_pairs(points) -> list:
    out = []
    for p in points or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
    return out


def draw_zone(points: list, name: str = "") -> dict:
    """Add an active ZONE (polygon) for occupancy counting. points: list of >=3
    [x,y] pairs NORMALIZED 0..1 (resolution-independent). name: optional label."""
    pts = _xy_pairs(points)
    if len(pts) < 3:
        return {"status": "error", "error": "need >=3 normalized [x,y] points"}
    item = {"id": name or f"zone{int(time.time()) % 1000}", "type": "active_zone",
            "display_name": name, "vertices": pts}
    ctx().pipe.add_annotation(item)
    return {"status": "success", "annotation": item, "ui": "annotations"}


def draw_line(points: list, name: str = "") -> dict:
    """Add a crossing LINE (polyline, 2+ points) for directional counting. points:
    list of [x,y] pairs NORMALIZED 0..1. The right-hand side of v0->v1 is positive."""
    pts = _xy_pairs(points)
    if len(pts) < 2:
        return {"status": "error", "error": "need >=2 normalized [x,y] points"}
    item = {"id": name or f"line{int(time.time()) % 1000}", "type": "crossing_line",
            "display_name": name, "vertices": pts}
    ctx().pipe.add_annotation(item)
    return {"status": "success", "annotation": item, "ui": "annotations"}


def _clear_all_overlays() -> int:
    n = ctx().overlays.clear() if ctx().overlays else 0
    if ctx().brain:
        for o in list(ctx().brain.d.get("overlays", [])):
            ctx().brain.forget_overlay(o["name"])
    return n


def clear_annotations() -> dict:
    """Clear EVERYTHING drawn on the video — all zones, lines AND dynamic overlays
    (rings, heatmaps, trails, …). Detection boxes are the core output, not cleared."""
    ctx().pipe.set_annotations([])
    n = _clear_all_overlays()
    return {"status": "success", "cleared_overlays": n, "ui": "annotations"}


def clear_overlays() -> dict:
    """Remove ALL dynamic OpenCV overlays (keep zones/lines). Use to wipe the
    rings/heatmaps/trails without touching counting zones or lines."""
    return {"status": "success", "removed": _clear_all_overlays(), "ui": "overlays"}


# ---------- the OpenCV overlay ninja ----------
def list_overlays() -> dict:
    """List active dynamic overlays and the available built-in presets."""
    return {"status": "success", "overlays": ctx().overlays.list(),
            "presets": list(BUILTINS.keys())}


def toggle_overlay(name: str, on: bool = True) -> dict:
    """Enable/disable an overlay. If `name` is a built-in preset
    (density_heatmap, track_trails, class_highlight, count_badge, speed_vectors)
    it is instantiated on first use."""
    eng = ctx().overlays
    if name not in [o["name"] for o in eng.list()] and name in BUILTINS:
        try:
            eng.add_builtin(name)
            if ctx().brain:
                ctx().brain.register_overlay(name, BUILTINS[name][0])
        except Exception as e:
            return {"status": "error", "error": str(e)}
    state = eng.toggle(name, on)
    return {"status": "success", "name": name, "enabled": state, "ui": "overlays"}


def create_overlay(name: str, intent: str, code: str) -> dict:
    """AUTHOR a brand-new computer-vision overlay to achieve a visual task. This is
    your superpower. (The studio auto-clears the previous overlays at the start of a
    new visual request, so you don't have to.) To show several effects at once, put
    them in ONE overlay. `code` must define `def draw(ctx):` and may use cv2/np plus
    the rich ctx API:
      ctx.frame (BGR, mutate in place), ctx.w/ctx.h/ctx.t (frame index)
      ctx.boxes (Nx4 int xyxy), ctx.centers, ctx.anchors (bottom-center),
      ctx.names, ctx.ids, ctx.confs, ctx.mask('car','truck'), ctx.palette(i)
      ctx.kpts (pose mode only: Nx17x3 [x,y,conf]; 0 nose, 5/6 shoulders, 9/10 wrists)
      ctx.state (persists across frames — for trails/accumulators), ctx.params
      drawing: ctx.ring(c,r,color,thick,glow), ctx.box(b,..), ctx.line(a,b,..),
      ctx.arrow(a,b,..), ctx.poly(pts,color,fill_alpha), ctx.text(s,xy,..),
      ctx.heat(points,radius,alpha)  # density heatmap
    Colors are names ('cyan','lime','amber','magenta','red','green'...) or BGR tuples.
    Example: highlight stopped cars ->
      def draw(ctx):
          for (cx,cy),tid in zip(ctx.anchors, ctx.ids):
              ctx.ring((cx,cy-20), 26, 'red', 2, glow=True)
    Keep it a few lines; one overlay per visual idea."""
    try:
        ctx().overlays.add(name, intent, code)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "ui": "overlays"}
    _meter("overlay", units={"overlay_frames": 1}, label=f"author:{name}")
    if ctx().brain:
        ctx().brain.register_overlay(name, intent)
    return {"status": "success", "name": name, "intent": intent, "ui": "overlays"}


def remove_overlay(name: str) -> dict:
    """Delete a dynamic overlay by name."""
    ok = ctx().overlays.remove(name)
    if ctx().brain:
        ctx().brain.forget_overlay(name)
    return {"status": "success" if ok else "error", "name": name, "ui": "overlays"}


FACE_FILTERS_LIST = ["ninja_mask", "sunglasses", "glasses", "dog", "cat",
                     "mustache", "crown", "clown_nose", "heart_eyes",
                     "face_mesh", "anonymize"]
_FILTER_ALIASES = {
    "ninja": "ninja_mask", "mask": "ninja_mask", "ninja_hood": "ninja_mask",
    "shades": "sunglasses", "sun_glasses": "sunglasses",
    "eyeglasses": "glasses", "spectacles": "glasses", "specs": "glasses",
    "dog_filter": "dog", "puppy": "dog", "doggy": "dog",
    "kitty": "cat", "cat_ears": "cat", "cat_filter": "cat", "kitten": "cat",
    "king": "crown", "queen": "crown", "royal": "crown",
    "hearts": "heart_eyes", "heart": "heart_eyes", "love": "heart_eyes",
    "mesh": "face_mesh", "wireframe": "face_mesh", "landmarks": "face_mesh",
    "blur": "anonymize", "pixelate": "anonymize", "privacy": "anonymize",
    "clown": "clown_nose", "red_nose": "clown_nose", "moustache": "mustache",
}


def apply_face_filter(name: str) -> dict:
    """Apply an AR FACE FILTER / face try-on to faces in the live video — ideal in
    Live Voice mode with the camera on ("put a ninja mask on me", "give me
    sunglasses"). Options: ninja_mask, sunglasses, glasses, dog, cat, mustache,
    crown, clown_nose, heart_eyes (smile-reactive), face_mesh, anonymize. Uses
    dense 478-point facial landmarks so the asset tracks your head pose. The studio
    auto-clears the prior look, so this replaces it. For trying on CLOTHES/garments
    use virtual_try_on instead."""
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    key = _FILTER_ALIASES.get(key, key)
    if key not in FACE_FILTERS_LIST:
        return {"status": "error", "error": f"unknown filter {name!r}",
                "available": FACE_FILTERS_LIST, "ui": "overlays"}
    try:
        ctx().overlays.clear()                 # a filter is a LOOK — replace, don't stack
        ctx().overlays.add_builtin(key)
        # clean selfie-filter look: hide detection boxes / labels / count badge
        ctx().pipe.set_render_flags(draw_boxes=False, draw_labels=False, draw_counts=False)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "ui": "overlays"}
    _meter("overlay", units={"overlay_frames": 1}, label=f"face_filter:{key}")
    if ctx().brain:
        ctx().brain.register_overlay(key, f"face filter: {key}")
    return {"status": "success", "name": key, "filter": key, "ui": "overlays"}


def _fetch_bytes(s: str) -> bytes | None:
    """Raw bytes from a data: URI / local path / http(s) URL (keeps PNG alpha)."""
    try:
        if s.startswith("data:"):
            b64 = s.split(",", 1)[1] if "," in s else s
            return base64.b64decode(b64 + "=" * (-len(b64) % 4), validate=False)
        if s.startswith("http"):
            req = urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=15).read()         # noqa: S310
        if os.path.exists(s):
            return open(s, "rb").read()
    except Exception:
        return None
    return None


import functools
import json as _json
import re as _re

_EYEWEAR_WORDS = ("glass", "sunglass", "shade", "frame", "eyewear", "aviator",
                  "optical", "specs", "spectacle", "lens", "wayfarer", "cat-eye",
                  "cateye", "browline", "clubmaster", "rimless")
_APPAREL_WORDS = ("polo", "shirt", "tee", "t-shirt", "tshirt", "jersey", "jacket",
                  "coat", "dress", "pant", "jean", "chino", "sweater", "hoodie",
                  "knit", "garment", "outfit", "blazer", "suit", "wear", "top")

# spoken shape -> canonical frame-shape tag (set on eyewear.json by the vision tagger)
_SHAPE_SYNS = {
    "aviator": "aviator", "pilot": "aviator", "teardrop": "aviator",
    "round": "round", "circular": "round", "circle": "round",
    "rectangle": "rectangle", "rectangular": "rectangle",
    "square": "square", "cat-eye": "cat-eye", "cateye": "cat-eye", "cat eye": "cat-eye",
    "oval": "oval", "wayfarer": "wayfarer", "browline": "browline", "clubmaster": "browline",
    "geometric": "geometric", "hexagonal": "hexagonal", "hexagon": "hexagonal",
    "sport": "sport", "sporty": "sport", "wrap": "sport", "wraparound": "sport",
    "rimless": "rimless", "oversized": "oversized", "oversize": "oversized",
}


def _query_shape(query: str):
    q = query.lower()
    for word, tag in _SHAPE_SYNS.items():
        if word in q:
            return tag
    return None


@functools.lru_cache(maxsize=1)
def _load_catalogs():
    base = os.path.dirname(__file__)
    out = {}
    for store, key, fn in (("apparel", "garments", "garments.json"),
                           ("eyewear", "eyewear", "eyewear.json")):
        try:
            d = _json.loads(open(os.path.join(base, fn)).read())
            items = d.get(key, [])
            for it in items:
                it["store"] = store
            out[store] = items
        except Exception:
            out[store] = []
    return out


def _guess_store(query: str):
    q = query.lower()
    if _query_shape(query) or any(w in q for w in _EYEWEAR_WORDS):
        return "eyewear"
    if any(w in q for w in _APPAREL_WORDS):
        return "apparel"
    return None                              # search both


def _search_catalog(query: str, store: str | None = None, limit: int = 8):
    toks = [t for t in _re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1]
    want_shape = _query_shape(query)         # shape-accurate eyewear search
    cats = _load_catalogs()
    pool = cats.get(store, []) if store else (cats["apparel"] + cats["eyewear"])
    scored = []
    for it in pool:
        title = (it.get("title", "") or "").lower()
        hay = " ".join(str(it.get(k, "")) for k in
                       ("title", "brand", "type", "store", "gender", "shape")).lower()
        score = sum(hay.count(t) for t in toks) + 2 * sum(1 for t in toks if t in title)
        if want_shape:
            if it.get("shape") == want_shape:
                score += 6                   # strong boost for the exact frame shape
            elif it.get("store") == "eyewear" and it.get("shape"):
                score -= 1                   # demote other eyewear shapes
        if score > 0:
            scored.append((score, it))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:limit]]


def _slim(it: dict) -> dict:
    return {"title": it.get("title", ""), "price": it.get("price", ""),
            "brand": it.get("brand", ""), "img": it.get("img", ""),
            "store": it.get("store", ""), "ar": bool(it.get("ar"))}


def shop_search(query: str, store: str = "") -> dict:
    """SHOP the sponsored stores — Mode Marco (apparel) + Ralba Optical (eyewear) —
    for products matching `query` (e.g. 'navy polo', 'hugo aviator', 'haiti jersey').
    Returns ranked matches with title, brand and PRICE so you can recommend options
    like a stylist, then call try_product to put one on the user. store: optional
    'apparel' | 'eyewear'."""
    matches = _search_catalog(query, store=store or _guess_store(query))
    return {"status": "success", "query": query,
            "matches": [_slim(m) for m in matches],
            "store": (matches[0]["store"] if matches else ""), "ui": None}


def try_product(query: str = "", image: str = "") -> dict:
    """SELL + TRY ON anything: find the product in the sponsored stores that best
    matches `query` (brand/colour/style, e.g. 'hugo aviator sunglasses', 'light
    blue polo', 'haiti jersey') and TRY IT ON the live person — real-time AR for
    eyewear, generative for apparel. Pass `image` directly to try a specific
    product URL. Then tell the user the brand + price (you're the store's stylist)."""
    if image and not query:
        return try_eyewear(image=image)
    store = _guess_store(query)
    matches = _search_catalog(query, store=store) or _search_catalog(query)
    if not matches:
        return {"status": "error", "error": f"no product matched '{query}'",
                "ui": "overlays"}
    best = matches[0]
    if best["store"] == "eyewear":
        res = try_eyewear(image=best["img"], label=best["title"])   # one unified glasses try-on
    else:
        res = virtual_try_on(garment=best["img"], which="frame")
    res["product"] = _slim(best)
    res["alternatives"] = [_slim(m) for m in matches[1:5]]
    res["catalog"] = {"store": best["store"], "matches": [_slim(m) for m in matches[:12]]}
    return res


_CANON_DIR = "out/studio/cache/eyewear_canon"
_CANON_VER = "v4-cleanlens"       # bump to invalidate cached renders when the pipeline changes
_CANON_PROMPT = (
    "Show ONLY the FRONT of these eyeglasses — the two lens rims joined by the nose "
    "bridge (and the brow bar if it has one), matching their exact colour, pattern "
    "and shape. The temple arms / legs MUST be ENTIRELY ABSENT — do not render them "
    "at all, not even folded, not as stubs, and NOTHING of an arm visible THROUGH or "
    "BEHIND the lenses. The lenses must be perfectly clean, showing only what is "
    "behind them. Perfectly FRONT-ON and symmetric, both lenses equal, lenses "
    "transparent (see-through) unless they are sunglasses, sharp even studio "
    "lighting, on a PURE WHITE seamless background, centered, the frame front "
    "filling ~85% of the width. No face, no arms, no shadow, no text.")
# final nano-banana correction pass — polish the OpenCV-armless front into a clean,
# symmetric, flawless asset just before it becomes the live-AR overlay.
_CORRECT_PROMPT = (
    "Polish this into a FLAWLESS front-on eyeglasses product image: perfectly "
    "symmetric lens rims joined by the nose bridge (and brow bar if present), smooth "
    "clean frame edges, absolutely NO temple arms or hinges anywhere, and NOTHING of "
    "an arm visible THROUGH or BEHIND the lenses — the lenses must be perfectly clean "
    "and clear. No rough or cut edges. Lenses transparent and see-through unless they "
    "are sunglasses. Keep the EXACT frame colour, pattern and shape. PURE WHITE "
    "seamless background, centered. No face, no arms, no text.")


def _flatten_white_jpg(rgba) -> bytes:
    bgr = rgba[:, :, :3].copy()
    bgr[rgba[:, :, 3] < 128] = (255, 255, 255)
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def _canonical_eyewear(raw: bytes, url: str):
    """Render a canonical, ARMLESS, polished FRONT-ON version of the frames so the AR
    warp sits right. Three layers, cached by URL on disk (generated ONCE per product):
      1) image model → front-on render,
      2) OpenCV remove_arms → drop temple arms/hinges,
      3) nano-banana correction pass → polish into a flawless clean asset.
    Returns png bytes or None."""
    import hashlib
    os.makedirs(_CANON_DIR, exist_ok=True)
    key = hashlib.md5(((url or "") + "|" + _CANON_VER).encode()).hexdigest()
    cache = os.path.join(_CANON_DIR, key + ".png")
    if os.path.exists(cache) and os.path.getsize(cache) > 1000:
        return open(cache, "rb").read()
    if not _has_key():
        return None
    from . import genmedia
    from .face_filters import load_eyewear_rgba, remove_arms
    model = ctx().settings.model_for("image_edit")
    try:
        p1, _, _ = genmedia.edit_image([raw], _CANON_PROMPT, model=model)   # 1) front-on
    except Exception:
        return None
    if not p1:
        return None
    try:                                                                    # 2) drop arms
        armless = remove_arms(load_eyewear_rgba(p1))
        p2, _, _ = genmedia.edit_image([_flatten_white_jpg(armless)],       # 3) nano polish
                                       _CORRECT_PROMPT, model=model)
    except Exception:
        p2 = None
    out = p2 or p1
    open(cache, "wb").write(out)
    ctx().ledger.record("agent_brain", model=model, input_tokens=700,
                        output_tokens=2600, label="eyewear_canonical")
    return out


def _eyewear_asset(image: str):
    """(rgba, fitted) — the realistic try-on asset for a product image: a canonical
    front-on render (generative) refined into see-through glass (OpenCV), with a
    graceful fallback to the raw cutout if generation is unavailable."""
    raw = _fetch_bytes(image)
    if not raw:
        return None, False
    from .face_filters import load_eyewear_rgba, clean_lenses, remove_arms
    canon = _canonical_eyewear(raw, image)
    rgba = load_eyewear_rgba(canon or raw)
    if rgba is None:
        return None, False
    # remove_arms drops the OUTER arms; clean_lenses rebuilds each lens interior so
    # any arm crossing BEHIND/THROUGH the lens is painted over (eye shows clean).
    return clean_lenses(remove_arms(rgba)), bool(canon)


def prefetch_eyewear(image: str = "") -> dict:
    """Warm the cache: generate + cache a product's canonical render WITHOUT
    applying it (called on hover so the actual try-on is instant)."""
    raw = _fetch_bytes(image)
    if not raw:
        return {"status": "error"}
    _canonical_eyewear(raw, image)
    return {"status": "success"}


def try_eyewear(image: str = "", label: str = "") -> dict:
    """LIVE EYEWEAR TRY-ON (Ralba Optical): put an actual glasses PRODUCT on the
    person in the live video — real-time, tracked to dense facial landmarks and
    following head pose. The frames are first rendered FRONT-ON by the image model
    and refined into realistic see-through glass (so they sit like they're worn,
    not pasted), then warped live. `image` = product image URL / path / data URI.
    For a single photoreal still instead, use virtual_try_on(which='frame')."""
    rgba, fitted = _eyewear_asset(image)
    if rgba is None:
        return {"status": "error", "error": "could not load eyewear image", "ui": "overlays"}
    from .face_filters import set_current_eyewear
    set_current_eyewear(rgba, label or "eyewear")
    try:
        ctx().overlays.clear()
        ctx().overlays.add_builtin("eyewear")
        ctx().pipe.set_render_flags(draw_boxes=False, draw_labels=False, draw_counts=False)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "ui": "overlays"}
    _meter("overlay", units={"overlay_frames": 1}, label="try_eyewear")
    return {"status": "success", "name": "eyewear", "label": label,
            "fitted": fitted, "ui": "overlays"}


# ---------- perception / the agent's eyes ----------
def analyze_scene() -> dict:
    """Look at the live frame RIGHT NOW: per-class counts, track count, zone/line
    counts, and up to 60 objects with class + box + ground anchor. Use this to
    decide what overlay or zone to create."""
    sc = ctx().pipe.scene()
    _meter("vision_brain", units={"synthesis": 1}, label="analyze_scene")
    if ctx().brain:
        ctx().brain.observe({"full_frame": sc.get("counts", {}),
                             "tracks": sc.get("tracks", 0), "fps": sc.get("fps")})
    return {"status": "success", **sc}


# ---------- the canvas: show images / videos, ingest a shared photo ----------
def _load_image(s: str):
    """Decode an image from a data: URI, a local path, or an http(s) URL."""
    try:
        if s.startswith("data:"):
            b64 = s.split(",", 1)[1] if "," in s else s
            b64 = b64[: len(b64) - (len(b64) % 4)]              # tolerate truncation
            raw = base64.b64decode(b64, validate=False)
        elif s.startswith("http"):
            raw = urllib.request.urlopen(s, timeout=10).read()  # noqa: S310
        elif os.path.exists(s):
            raw = open(s, "rb").read()
        else:
            return None
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            return arr
        from PIL import Image, ImageFile               # fallback for truncated JPEG
        import io
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def test_image(label: str = "") -> dict:
    """Render a synthetic TEST CARD (color bars + crosshair + grid) and show it on the
    canvas. Use to demonstrate the canvas or sanity-check image display."""
    w, h = 640, 360
    img = np.zeros((h, w, 3), np.uint8)
    bars = [(192, 192, 192), (0, 255, 255), (255, 255, 0), (0, 255, 0),
            (255, 0, 255), (0, 0, 255), (255, 0, 0), (24, 24, 24)]
    bw = w // len(bars)
    for i, c in enumerate(bars):
        img[:h * 2 // 3, i * bw:(i + 1) * bw] = c[::-1]            # RGB->BGR
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h - h * 2 // 3, 1))
    img[h * 2 // 3:, :, 0] = grad; img[h * 2 // 3:, :, 1] = grad; img[h * 2 // 3:, :, 2] = grad
    cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (0, h // 2), (w, h // 2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, h // 2), 70, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, label or "VISION NINJA - TEST CARD", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, time.strftime("%Y-%m-%d %H:%M:%S"), (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs("out/studio/exports", exist_ok=True)
    out = f"out/studio/exports/testcard_{time.strftime('%H%M%S')}.png"
    cv2.imwrite(out, img)
    _meter("snapshot", units={"ops": 1}, label="test_image")
    return _lib({"status": "success", "kind": "image", "output": out,
                 "caption": label or "test card", "resolution": [w, h]},
                tags=["test"])


def display_media(kind: str, src: str = "", caption: str = "") -> dict:
    """Show something on the center CANVAS. kind: 'image' | 'video' | 'live'.
    src: a file produced by an export (path under out/studio or out/recordings),
    an http URL, or empty to return to the live feed. Use this to move the canvas
    between the live stream and an artifact based on what the user needs to see."""
    return {"status": "success", "kind": "display", "mode": kind,
            "src": src, "caption": caption}


def analyze_image(image: str, caption: str = "") -> dict:
    """Ingest a STILL image the user shared and SHOW it on the canvas, annotated with
    detections. `image` is a data: URI (data:image/...;base64,...), a file path, or an
    http URL. Use this whenever the user pastes/links an image or asks about a picture.
    Runs the configured detector on the still and reports per-class counts."""
    frame = _load_image(image)
    if frame is None:
        return {"status": "error", "error": "could not decode the image"}
    try:
        det, vis, counts = ctx().pipe.detect_still(frame)
    except Exception as e:
        return {"status": "error", "error": f"detect failed: {e}"}
    os.makedirs("out/studio/exports", exist_ok=True)
    out = f"out/studio/exports/analyzed_{time.strftime('%H%M%S')}.png"
    cv2.imwrite(out, vis)
    _meter("detect", model=str(ctx().pipe.cfg.get("detector.model", "")),
           units={"inferences": 1}, label="analyze_image")
    _meter("snapshot", units={"ops": 1}, label="render still")
    n = int(len(det))
    summary = ", ".join(f"{v} {k}" for k, v in counts.items()) or "no known objects"
    return _lib({"status": "success", "kind": "image", "output": out, "objects": n,
                 "counts": counts, "resolution": [frame.shape[1], frame.shape[0]],
                 "caption": caption or f"shared image — {summary}", "summary": summary},
                tags=["analysis", "detection"], source="shared")


# ---------- visual analysis + generative media (Gemini) ----------
def _frame_jpg(which: str = "frame", image: str = ""):
    """Resolve a frame to (jpg_bytes, bgr_frame): a shared image, or the live frame."""
    frame = _load_image(image) if image else None
    if frame is None and which in ("frame", "live", "scene", "", None):
        frame = ctx().pipe.snapshot_clean()
    if frame is None:
        return None, None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes(), frame


def _save_png(frame_or_bytes, stem: str) -> str:
    os.makedirs("out/studio/exports", exist_ok=True)
    out = f"out/studio/exports/{stem}_{time.strftime('%H%M%S')}.png"
    if isinstance(frame_or_bytes, (bytes, bytearray)):
        open(out, "wb").write(frame_or_bytes)
    else:
        cv2.imwrite(out, frame_or_bytes)
    return out


def describe_image(question: str = "", which: str = "frame", image: str = "") -> dict:
    """SEMANTIC visual analysis with Gemini Vision — describe or answer a question
    about the live frame (which='frame') or a shared image (data URI / path / URL).
    Goes beyond detection: scene, mood, text/OCR, identity of objects. Use when the
    user asks 'what is this / what's happening / read the sign'."""
    jpg, frame = _frame_jpg(which, image)
    if jpg is None:
        return {"status": "error", "error": "no frame/image to analyze"}
    out = _save_png(frame, "look")
    q = question or ("Describe this image in 1-2 sentences. Note notable objects, "
                     "any text, colors, and activity.")
    if _has_key():
        from . import genmedia
        try:
            model = ctx().settings.model_for("vision")
            text, r = genmedia.describe(jpg, q, model=model)
            um = getattr(r, "usage_metadata", None)
            ctx().ledger.record("agent_brain", model=model,
                                input_tokens=getattr(um, "prompt_token_count", 320) or 320,
                                output_tokens=getattr(um, "candidates_token_count", 90) or 90,
                                label="vision")
        except Exception as e:
            text = f"(vision error: {e})"
    else:
        _det, _vis, counts = ctx().pipe.detect_still(frame)
        text = ("I detect " + (", ".join(f"{v} {k}" for k, v in counts.items())
                or "no known objects") + ". (Add a Gemini key for full visual analysis.)")
        _meter("vision_brain", units={"synthesis": 1}, label="describe (fallback)")
    if ctx().brain:
        ctx().brain.remember("last_look", text[:160])
    return _lib({"status": "success", "kind": "image", "output": out, "text": text,
                 "caption": text[:140]}, tags=["vision"], source=which)


def nano_banana(prompt: str, which: str = "frame", image: str = "") -> dict:
    """NANO-BANANA image edit: transform a frame with Gemini's image model — restyle,
    add/remove things, change time-of-day/weather, recolor. Works on the live frame
    (which='frame') or a shared image. e.g. 'make it night with wet roads'."""
    if not _has_key():
        return {"status": "error", "error": "nano-banana needs a Gemini key"}
    jpg, frame = _frame_jpg(which, image)
    if jpg is None:
        return {"status": "error", "error": "no frame/image"}
    from . import genmedia
    try:
        out_bytes, text, r = genmedia.edit_image(
            [jpg], prompt, model=ctx().settings.model_for("image_edit"))
    except Exception as e:
        return {"status": "error", "error": str(e)}
    if not out_bytes:
        return {"status": "error", "error": text or "no image returned"}
    out = _save_png(out_bytes, "nano")
    ctx().ledger.record("agent_brain", model=ctx().settings.model_for("image_edit"),
                        input_tokens=300, output_tokens=1300, label="nano_banana")
    return _lib({"status": "success", "kind": "image", "output": out,
                 "caption": prompt, "text": text}, tags=["nano-banana", "edit"], source=which)


def virtual_try_on(garment: str, which: str = "frame", image: str = "") -> dict:
    """VIRTUAL TRY-ON: put a garment (image path / URL / data URI) onto the person in
    the frame, preserving their pose, identity and background (Gemini image model)."""
    if not _has_key():
        return {"status": "error", "error": "try-on needs a Gemini key"}
    jpg, frame = _frame_jpg(which, image)
    g = _load_image(garment)
    if jpg is None or g is None:
        return {"status": "error", "error": "need a person frame + a garment image"}
    ok, gbuf = cv2.imencode(".jpg", g)
    from . import genmedia
    prompt = (
        "Virtual try-on. Put the EXACT garment shown in the second image onto the "
        "person in the first image — match its colour, pattern, logos and texture "
        "faithfully. Render a natural, photoreal head-and-shoulders / upper-body "
        "portrait so the garment is clearly visible worn on their chest and "
        "shoulders; if the first photo is cropped tight on the face, zoom out "
        "slightly to reveal the upper body wearing it. Preserve the person's face, "
        "identity, skin tone, hair, lighting and background. Output only the image.")
    try:
        out_bytes, text, r = genmedia.edit_image(
            [jpg, gbuf.tobytes()], prompt, model=ctx().settings.model_for("image_edit"))
    except Exception as e:
        return {"status": "error", "error": str(e)}
    if not out_bytes:
        return {"status": "error", "error": text or "no image returned"}
    out = _save_png(out_bytes, "tryon")
    ctx().ledger.record("agent_brain", model=ctx().settings.model_for("image_edit"),
                        input_tokens=400, output_tokens=1300, label="virtual_try_on")
    return _lib({"status": "success", "kind": "image", "output": out,
                 "caption": "virtual try-on"}, tags=["try-on", "edit"], source=which)


def generate_video(prompt: str, from_frame: bool = True) -> dict:
    """VEO: generate a short video from a prompt, optionally seeded by the current
    frame (from_frame=True). SLOW — Veo takes minutes — and costs real money. Use
    sparingly, only when the user explicitly asks to generate/animate a video."""
    if not _has_key():
        return {"status": "error", "error": "Veo needs a Gemini key"}
    from . import genmedia
    img = _frame_jpg("frame", "")[0] if from_frame else None
    data, err = genmedia.generate_video(prompt, image_jpg=img,
                                        model=ctx().settings.model_for("video"))
    if not data:
        return {"status": "error", "error": err or "veo failed"}
    os.makedirs("out/studio/exports", exist_ok=True)
    out = f"out/studio/exports/veo_{time.strftime('%H%M%S')}.mp4"
    open(out, "wb").write(data)
    ctx().ledger.record("agent_brain", model=ctx().settings.model_for("video"),
                        input_tokens=400, output_tokens=4000, label="veo")
    return _lib({"status": "success", "kind": "clip", "output": out, "caption": prompt,
                 "seconds": ff.probe(out).get("duration", 0)}, tags=["veo", "video"],
                source="generated")


def extend_video(prompt: str = "continue the scene naturally", which: str = "recording") -> dict:
    """VEO video extension: take a clip's LAST frame as a seed, generate a continuation
    with Veo, and stitch it on. SLOW + costs money."""
    if not _has_key():
        return {"status": "error", "error": "Veo needs a Gemini key"}
    src = _resolve_target(which)
    if not src:
        return {"status": "error", "error": f"no {which} file"}
    dur = ff.probe(src).get("duration", 1)
    fr = ff.frame_at(src, max(0, dur - 0.1))
    if fr.get("status") != "success":
        return fr
    frame = cv2.imread(fr["output"])
    ok, buf = cv2.imencode(".jpg", frame)
    from . import genmedia
    data, err = genmedia.generate_video(prompt, image_jpg=buf.tobytes(),
                                        model=ctx().settings.model_for("video"))
    if not data:
        return {"status": "error", "error": err or "veo failed"}
    cont = f"out/studio/exports/veo_ext_{time.strftime('%H%M%S')}.mp4"
    open(cont, "wb").write(data)
    ctx().ledger.record("agent_brain", model=ctx().settings.model_for("video"),
                        input_tokens=400, output_tokens=4000, label="veo extend")
    joined = ff.concat([src, cont])
    res = joined if joined.get("status") == "success" else {
        "status": "success", "kind": "clip", "output": cont, "caption": prompt}
    return _lib(res, tags=["veo", "extension"], source=which)


# ---------- the OpenCV coder: arbitrary CV on a frame / image / video ----------
from .overlays import _SAFE_BUILTINS as _CV_SAFE


def _resolve_image(target: str):
    if target in ("frame", "live", "scene", "", None):
        return ctx().pipe.snapshot_clean()
    cand = target
    if not (target.startswith(("data:", "http")) or os.path.exists(target)):
        cand = _media_path(os.path.basename(target)) or target
    return _load_image(cand)


def run_cv_code(code: str, target: str = "frame") -> dict:
    """Run ARBITRARY OpenCV/numpy code to solve ANY computer-vision task on one image
    and show the result on the canvas. This is your full-power CV coding tool — edges,
    contours, optical flow (two frames), feature matching, thresholding, morphology,
    FFT, Hough, homography, segmentation, color spaces — anything OpenCV does.
    `target`: 'frame' (the live frame), a library image name, a path, a data: URI, or
    an http URL. Your `code` gets `img` (BGR np.ndarray) + `cv2`,`np`; set `out` to a
    result image (BGR or gray) OR `result` to a dict/number/string.
    Example: `out = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)`."""
    img = _resolve_image(target)
    if img is None:
        return {"status": "error", "error": f"could not load target {target!r}"}
    g = {"__builtins__": _CV_SAFE, "cv2": cv2, "np": np, "img": img,
         "out": None, "result": None}
    try:
        exec(compile(code, "<cv_code>", "exec"), g)        # noqa: S102 (the ninja)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    out = g.get("out")
    if isinstance(out, np.ndarray):
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        if out.dtype != np.uint8:
            out = (np.clip(out, 0, 255) if out.max() > 1 else out * 255).astype(np.uint8)
        path = _save_png(out, "cv")
        _meter("overlay", units={"overlay_frames": 1}, label="run_cv_code")
        return _lib({"status": "success", "kind": "image", "output": path,
                     "caption": code.strip().splitlines()[-1][:60]}, tags=["cv-code"],
                    source=target)
    res = g.get("result")
    return {"status": "success", "kind": "data", "result": res,
            "text": str(res)[:400] if res is not None else "ran (no image output)"}


def run_cv_video(code: str, which: str = "source", max_seconds: float = 10.0) -> dict:
    """Apply ARBITRARY OpenCV code to EVERY frame of a clip -> a new video. `code` gets
    `img` (BGR) per frame and must set `out` (BGR or gray). which: 'source'|'recording'.
    e.g. `out = cv2.Canny(img, 80, 160)` edge-detects the whole clip."""
    src = _resolve_target(which)
    if not src:
        return {"status": "error", "error": f"no {which} file"}
    try:
        fn = compile(code, "<cv_video>", "exec")
    except Exception as e:
        return {"status": "error", "error": f"compile: {e}"}
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return {"status": "error", "error": "cannot open clip"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    os.makedirs("out/studio/exports", exist_ok=True)
    out_path = f"out/studio/exports/cvvid_{time.strftime('%H%M%S')}.mp4"
    writer, n, limit = None, 0, int(max_seconds * fps)
    while n < limit:
        ok, frame = cap.read()
        if not ok:
            break
        g = {"__builtins__": _CV_SAFE, "cv2": cv2, "np": np, "img": frame, "out": None}
        try:
            exec(fn, g)
        except Exception as e:
            cap.release()
            return {"status": "error", "error": f"frame {n}: {type(e).__name__}: {e}"}
        out = g.get("out")
        if not isinstance(out, np.ndarray):
            out = frame
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        if out.dtype != np.uint8:
            out = (np.clip(out, 0, 255) if out.max() > 1 else out * 255).astype(np.uint8)
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1, round(fps)), (w, h))
        writer.write(out)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    if n == 0:
        return {"status": "error", "error": "no frames processed"}
    _meter("overlay", units={"overlay_frames": n}, label="run_cv_video")
    return _lib({"status": "success", "kind": "clip", "output": out_path,
                 "seconds": round(n / fps, 1), "caption": "cv: " + code[:40]},
                tags=["cv-code", "video"], source=which)


def emit_proto() -> dict:
    """Write the CURRENT frame's OccupancyCountingPredictionResult protobuf — the
    ORIGINAL occupancy-analytics contract: identified boxes, full-frame/line/zone
    counts, DWELL times, and track info. Saved as a replayable .pb (parse_occupancy.py)."""
    res = ctx().pipe.emit_proto()
    if res.get("status") == "success":
        _meter("snapshot", units={"ops": 1}, label="emit_proto")
        res["ui"] = "library"
        res["kind"] = "proto"
    return res


# ---------- audio (TTS narration; Lyria music gated) ----------
def narrate(text: str, voice: str = "Puck") -> dict:
    """Generate spoken NARRATION (Gemini TTS) as a WAV and add it to the library —
    use it as an audio clip in the NLE timeline. voice: Puck|Charon|Kore|Fenrir|…"""
    if not _has_key():
        return {"status": "error", "error": "TTS needs a Gemini key"}
    from . import genmedia
    wav, err = genmedia.tts(text, voice)
    if not wav:
        return {"status": "error", "error": err}
    os.makedirs("out/studio/exports", exist_ok=True)
    out = f"out/studio/exports/narration_{time.strftime('%H%M%S')}.wav"
    open(out, "wb").write(wav)
    ctx().ledger.record("agent_brain", model="gemini-2.5-flash-preview-tts",
                        input_tokens=len(text) // 4 + 20, output_tokens=200, label="tts")
    if ctx().library:
        ctx().library.add("audio", out, caption=f"narration: {text[:50]}",
                          tags=["narration", "audio"])
    return {"status": "success", "kind": "audio", "output": out,
            "caption": text[:60], "ui": "library"}


def generate_music(prompt: str) -> dict:
    """Generate instrumental background MUSIC (Lyria on Vertex AI, ~30s) and add it to
    the media library — use it on the NLE MUSIC track. Needs GCP creds
    (gcloud auth application-default login) + GOOGLE_CLOUD_PROJECT."""
    from . import genmedia
    wav, err = genmedia.generate_music(prompt)
    if not wav:
        return {"status": "error", "error": err or "Lyria failed"}
    os.makedirs("out/studio/exports", exist_ok=True)
    out = f"out/studio/exports/music_{time.strftime('%H%M%S')}.wav"
    open(out, "wb").write(wav)
    ctx().ledger.record("agent_brain", model="lyria-002", input_tokens=200,
                        output_tokens=2000, label="lyria music")
    if ctx().library:
        ctx().library.add("audio", out, caption=f"music: {prompt[:50]}",
                          tags=["music", "lyria"])
    return {"status": "success", "kind": "audio", "output": out,
            "caption": prompt[:60], "ui": "library"}


# ---------- media library + search ----------
def search_library(query: str = "") -> dict:
    """Search the media library in natural language (over captions/tags/source).
    Empty query lists everything recent."""
    lib = ctx().library
    items = lib.search(query) if query else lib.list()
    return {"status": "success", "query": query, "items": items[:40], "ui": "library"}


def list_library() -> dict:
    """List recent media library items (snapshots, clips, gifs, analyses, …)."""
    return {"status": "success", "items": ctx().library.list(), "ui": "library"}


def show_media(item_id: int = 0, query: str = "") -> dict:
    """Show a library item on the canvas — by id, or the best match for `query`."""
    lib = ctx().library
    item = lib.get(item_id) if item_id else ((lib.search(query)[:1] or [None])[0])
    if not item:
        return {"status": "error", "error": "no matching media"}
    mode = "video" if item["name"].lower().endswith((".mp4", ".mov", ".webm")) else "image"
    return {"status": "success", "kind": "display", "mode": mode,
            "src": "/download/" + item["name"], "caption": item["caption"]}


def save_to_library(caption: str = "", tags: str = "") -> dict:
    """Save the CURRENT annotated frame (with overlays) to the media library."""
    vis = ctx().pipe.snapshot_vis()
    if vis is None:
        return {"status": "error", "error": "no frame yet"}
    out = _save_png(vis, "saved")
    _meter("snapshot", units={"ops": 1}, label="save_to_library")
    item = ctx().library.add("image", out, caption=caption or "saved frame",
                             tags=[t.strip() for t in tags.split(",") if t.strip()] or ["saved"],
                             source=str(ctx().pipe.cfg.get("source.uri", "")))
    return {"status": "success", "kind": "image", "output": out, "item": item, "ui": "library"}


# ---------- YouTube ingestion ----------
def load_youtube(url: str, seconds: int = 30) -> dict:
    """Download a YouTube (or any yt-dlp-supported) video and play it as the live
    source — then you can detect/overlay/analyze it. `seconds` caps the download."""
    os.makedirs("out/studio/cache", exist_ok=True)
    base = f"out/studio/cache/yt_{int(time.time())}"
    out = base + ".mp4"
    args = [sys.executable, "-m", "yt_dlp", "-f",
            "mp4[height<=720]/best[height<=720]/best", "-o", out,
            "--no-playlist", "--quiet", "--no-warnings", "--force-overwrites"]
    if seconds and int(seconds) > 0:
        args += ["--download-sections", f"*0-{int(seconds)}"]
    args.append(url)
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=300)
    except Exception as e:
        return {"status": "error", "error": f"yt-dlp: {e}"}
    produced = out if os.path.exists(out) else next(iter(glob.glob(base + "*")), None)
    if p.returncode != 0 or not produced or not os.path.exists(produced):
        return {"status": "error", "error": (p.stderr or p.stdout or "download failed")[-300:]}
    ctx().pipe.reconfigure({"source.uri": produced})
    _meter("ffmpeg:youtube", units={"out_seconds": int(seconds or 0), "ops": 1},
           label="youtube")
    ctx().library.add("video", produced, caption=f"YouTube: {url}",
                      tags=["youtube", "source"], source=url)
    if ctx().brain:
        ctx().brain.observe({}, source=produced)
    return {"status": "success", "output": produced, "source": produced, "url": url,
            "ui": "config"}


# ---------- ffmpeg export ----------
def _resolve_target(which: str) -> str | None:
    if which == "recording":
        recs = sorted(glob.glob("out/recordings/*.mp4"), key=os.path.getmtime)
        return recs[-1] if recs else None
    uri = str(ctx().pipe.cfg.get("source.uri", ""))
    return uri if os.path.exists(uri) else None


def _ff_meter(res):
    if res.get("status") == "success":
        _meter("ffmpeg:" + res.get("kind", "op"),
               units={"out_seconds": float(res.get("seconds", 0)), "ops": 1},
               label=res.get("kind", "ffmpeg"))
        out = str(res.get("output", ""))
        if out and ctx().library:
            kind = ("video" if out.lower().endswith((".mp4", ".mov", ".webm"))
                    else "audio" if out.lower().endswith(".mp3") else "image")
            ctx().library.add(kind, out, caption=res.get("kind", "clip"),
                              tags=[res.get("kind", "ffmpeg"), "export"])
    return res


def edit_video(op: str, which: str = "recording", text: str = "", look: str = "grayscale",
               factor: float = 2.0, degrees: int = 90, seconds: float = 4.0,
               x: int = 0, y: int = 0, w: int = 640, h: int = 360) -> dict:
    """The ffmpeg multitool. op = filter | crop | rotate | flip | reverse | boomerang
    | fade | caption | loop | speed | extract_audio | frame. which = 'recording' |
    'source'. Params by op: look (grayscale/sepia/invert/sharpen/blur/vintage/edges),
    text (caption), factor (speed/loop), degrees (rotate), x/y/w/h (crop), seconds
    (frame time). Output auto-saves to the media library + canvas."""
    src = _resolve_target(which)
    if not src:
        return {"status": "error", "error": f"no {which} file"}
    o = op.lower().strip()
    table = {
        "filter": lambda: ff.vfilter(src, look), "crop": lambda: ff.crop(src, x, y, w, h),
        "rotate": lambda: ff.rotate(src, degrees), "flip": lambda: ff.flip(src, "h"),
        "mirror": lambda: ff.flip(src, "h"), "reverse": lambda: ff.reverse(src),
        "boomerang": lambda: ff.boomerang(src), "fade": lambda: ff.fade(src),
        "caption": lambda: ff.overlay_text(src, text or "Vision Ninja"),
        "text": lambda: ff.overlay_text(src, text or "Vision Ninja"),
        "loop": lambda: ff.loop(src, int(factor) or 3),
        "speed": lambda: ff.speed_ramp(src, factor),
        "timelapse": lambda: ff.speed_ramp(src, max(2.0, factor)),
        "slowmo": lambda: ff.speed_ramp(src, min(0.5, 1.0 / max(factor, 1))),
        "extract_audio": lambda: ff.extract_audio(src), "audio": lambda: ff.extract_audio(src),
        "frame": lambda: ff.frame_at(src, seconds), "grab": lambda: ff.frame_at(src, seconds),
    }
    if o not in table:
        return {"status": "error", "error": f"unknown op {op!r}; have {list(table)}"}
    return _ff_meter(table[o]())


def export_gif(which: str = "source", start: float = 0, duration: float = 4,
               fps: int = 12, width: int = 480) -> dict:
    """Make a shareable GIF from 'source' (the live file) or 'recording' (latest
    recorded clip). start/duration in seconds."""
    src = _resolve_target(which)
    if not src:
        return {"status": "error", "error": f"no {which} file available"}
    return _ff_meter(ff.make_gif(src, start, duration, fps, width))


def export_clip(which: str = "recording", start: float = 0, duration: float = 8) -> dict:
    """Cut an MP4 clip from 'recording' or 'source'."""
    src = _resolve_target(which)
    if not src:
        return {"status": "error", "error": f"no {which} file available"}
    return _ff_meter(ff.clip(src, start, duration))


def export_contact_sheet(which: str = "source", cols: int = 4, rows: int = 3) -> dict:
    """Tile evenly-sampled frames into one PNG — a visual summary of the clip."""
    src = _resolve_target(which)
    if not src:
        return {"status": "error", "error": f"no {which} file available"}
    return _ff_meter(ff.contact_sheet(src, cols, rows))


def speed_ramp(which: str = "recording", factor: float = 4.0) -> dict:
    """Timelapse (factor>1) or slow-mo (factor<1) a clip with ffmpeg."""
    src = _resolve_target(which)
    if not src:
        return {"status": "error", "error": f"no {which} file available"}
    return _ff_meter(ff.speed_ramp(src, factor))


def probe_media(which: str = "source") -> dict:
    """Report duration/resolution/fps/codec/size of 'source' or 'recording'."""
    src = _resolve_target(which)
    if not src:
        return {"status": "error", "error": f"no {which} file available"}
    return ff.probe(src)


# ---------- recording + snapshot ----------
def record_start() -> dict:
    """Start recording the annotated live stream to an MP4."""
    name = ctx().pipe.start_recording()
    return {"status": "success", "file": name, "recording": True}


def record_stop() -> dict:
    """Stop recording; returns the file + frame count, and adds it to the library."""
    name, n = ctx().pipe.stop_recording()
    path = f"out/recordings/{name}" if name else ""
    if path and os.path.exists(path) and ctx().library:
        ctx().library.add("video", path, caption="recording", tags=["recording"],
                          source=str(ctx().pipe.cfg.get("source.uri", "")))
    return {"status": "success", "file": name, "frames": n, "recording": False}


def snapshot() -> dict:
    """Save a PNG of the current annotated frame (with overlays)."""
    vis = ctx().pipe.snapshot_vis()
    if vis is None:
        return {"status": "error", "error": "no frame yet"}
    path = _save_png(vis, "snap")
    _meter("snapshot", units={"ops": 1}, label="snapshot")
    return _lib({"status": "success", "output": path, "kind": "snapshot"},
                caption="annotated snapshot", tags=["snapshot"],
                source=str(ctx().pipe.cfg.get("source.uri", "")))


# ---------- Vision Brain ----------
def remember(key: str, text: str) -> dict:
    """Save a durable fact to the Vision Brain (recalled in later turns)."""
    if ctx().brain:
        ctx().brain.remember(key, text)
    return {"status": "success", "key": key, "ui": "brain"}


def recall() -> dict:
    """Read the Vision Brain: source, observed classes/peaks, zones, overlays, notes."""
    return {"status": "success", "brain": ctx().brain.to_dict() if ctx().brain else {}}


# ---------- settings ----------
def set_simulation(level: str) -> dict:
    """Set the simulation level: live (everything real) | simulated (brain real,
    heavy ops metered as sim) | zero ($0, all fake)."""
    if level not in ("live", "simulated", "zero"):
        return {"status": "error", "error": "level must be live|simulated|zero"}
    ctx().settings.update({"simulation": level})
    return {"status": "success", "simulation": level, "ui": "settings"}


# ---- registry: name -> function, and the list ADK gets ----
# ---------- live sources library (RTSP cameras / YouTube live) ----------
def save_source(url: str, name: str = "") -> dict:
    """Save a LIVE source — an RTSP url (rtsp://…) or a YouTube live URL — to the
    live-sources library so you can switch to it anytime."""
    kind = ("youtube_live" if ("youtube.com" in url or "youtu.be" in url)
            else "rtsp" if url.startswith("rtsp") else "stream")
    item = ctx().sources.add(kind, url, caption=name or url, tags=["live", kind], source=url)
    return {"status": "success", "item": item, "ui": "sources"}


def list_sources() -> dict:
    """List saved live sources (RTSP cameras / YouTube live)."""
    return {"status": "success", "items": ctx().sources.list(), "ui": "sources"}


def use_source(item_id: int = 0, query: str = "") -> dict:
    """Switch the live pipeline to a saved live source by id or name. Resolves a
    YouTube-live URL to a playable stream via yt-dlp."""
    lib = ctx().sources
    item = lib.get(item_id) if item_id else ((lib.search(query)[:1] or [None])[0])
    if not item:
        return {"status": "error", "error": "no matching live source"}
    url = item["path"]
    if item["kind"] == "youtube_live":
        stream, err = _resolve_youtube(url)
        if not stream:
            return {"status": "error", "error": err}
        url = stream
    ctx().pipe.reconfigure({"source.uri": url})
    return {"status": "success", "source": url, "name": item["caption"], "ui": "config"}


def _resolve_youtube(url: str) -> tuple[str, str]:
    """Resolve a YouTube watch URL to a direct stream/HLS URL via yt-dlp, trying a
    few strategies (plain, then browser cookies which get past the bot-check).
    Returns (stream_url, "") on success or ("", honest_error) on failure — the
    real yt-dlp error is surfaced so a YouTube bot-check / geo-block / missing JS
    runtime isn't masked as a generic 'not live' message."""
    fmt = "best[height<=720]/best"
    # yt-dlp's modern YouTube path needs (a) a JS runtime (deno) + the EJS remote
    # challenge-solver script to compute the n-signature, and (b) browser cookies
    # to clear the "confirm you're not a bot" gate. Try cheap → heavy.
    ejs = ["--remote-components", "ejs:github"]
    attempts = [
        ["-g", "-f", fmt, *ejs, url],
        ["-g", "-f", fmt, *ejs, "--cookies-from-browser", "chrome", url],
        ["-g", "-f", fmt, *ejs, "--cookies-from-browser", "safari", url],
    ]
    last = ""
    for extra in attempts:
        try:
            p = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", *extra],
                               capture_output=True, text=True, timeout=60)
        except Exception as e:
            last = str(e); continue
        for ln in reversed((p.stdout or "").strip().splitlines()):
            ln = ln.strip()
            if ln.startswith("http") and "youtube.com/watch" not in ln and "youtu.be" not in ln:
                return ln, ""
        # capture the real reason (last ERROR line) for an honest message
        for ln in reversed((p.stderr or "").strip().splitlines()):
            if "ERROR" in ln:
                last = ln.split("ERROR:", 1)[-1].strip(); break
    hint = ""
    low = last.lower()
    if "not a bot" in low or "sign in to confirm" in low:
        hint = (" — YouTube is rate-limiting/bot-checking this host. Wait a few "
                "minutes, or install a JS runtime (`brew install deno`) so yt-dlp "
                "can solve the challenge.")
    elif "no video formats" in low:
        hint = " — yt-dlp needs a JS runtime to decode formats (`brew install deno`)."
    elif "geo" in low or "not available in your" in low:
        hint = " — the stream looks geo-blocked."
    return "", (f"could not resolve YouTube stream: {last}{hint}"
                if last else "could not resolve a playable stream from that YouTube URL "
                             "(it may not be live, or geo-blocked).")


def _media_path(name: str) -> str:
    for d in ("out/studio/exports", "out/recordings", "out/studio/cache"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return ""


def co_direct(brief: str, max_scenes: int = 4, narration: str = "") -> dict:
    """CO-DIRECTOR (Izumi): assemble a rough edit for `brief`. Gathers the best video
    clips from the media library (recordings, exports, Veo, nano/edit clips), orders
    them into scenes, optionally lays a TTS narration track, and LOADS it into the NLE
    Editor for you to refine + Export. The OpenCV-annotated clips are the material —
    record/snapshot/export first, or it falls back to cutting the current source."""
    lib = ctx().library
    seen, chosen = set(), []
    for it in lib.list(200):
        if it["name"].lower().endswith((".mp4", ".mov", ".webm")) and it["name"] not in seen:
            seen.add(it["name"]); chosen.append(it["name"])
            if len(chosen) >= max_scenes:
                break
    if not chosen:                                   # fall back: cut the live source
        src = _resolve_target("source")
        if src:
            for s in (0, 3):
                c = ff.clip(src, s, 3)
                if c.get("status") == "success":
                    chosen.append(os.path.basename(c["output"]))
    if not chosen:
        return {"status": "error", "error": "no clips to assemble — record or export first"}
    video, t0 = [], 0.0
    for name in chosen:
        path = _media_path(name)
        dur = min(6.0, (ff.probe(path).get("duration", 4) or 4)) if path else 4.0
        video.append({"src": name, "t0": round(t0, 2), "offset": 0, "duration": round(dur, 2)})
        t0 += dur
    audio = []
    if narration and _has_key():
        nar = narrate(narration)
        if nar.get("status") == "success":
            audio.append({"src": os.path.basename(nar["output"]), "t0": 0, "offset": 0,
                          "duration": round(t0, 2), "gain": 1.0})
    if ctx().brain:
        ctx().brain.remember("last_edit_brief", brief[:120])
    return {"status": "success", "timeline": {"video": video, "audio": audio},
            "scenes": len(video), "brief": brief, "ui": "edit"}


def direct_story(brief: str, scenes: int = 3, mode: str = "fast",
                 narrate_scenes: bool = True) -> dict:
    """CO-DIRECTOR generative storytelling (Co-Director / Izumi style): PLAN scenes for
    `brief` with Gemini, GENERATE each scene, add TTS narration, and load the result
    into the NLE Editor. mode='fast' = nano-banana keyframe + Ken-Burns motion (cheap,
    seconds); mode='veo' = Veo text->video (slow, minutes, costs real money). Use when
    the user wants to GENERATE footage for a story/reel (not stitch existing clips)."""
    if not _has_key():
        return {"status": "error", "error": "generative storytelling needs a Gemini key"}
    from . import genmedia
    plan, _ = genmedia.plan_story(brief, scenes, model=ctx().settings.model_for("vision"))
    if not plan:
        return {"status": "error", "error": "could not plan scenes"}
    video, audio, t0, made = [], [], 0.0, 0
    for i, sc in enumerate(plan):
        clip, dur = None, 3.5
        if mode == "veo":
            data, _err = genmedia.generate_video(sc["prompt"],
                                                 model=ctx().settings.model_for("video"))
            if data:
                out = f"out/studio/exports/story_{int(time.time())}_{i}.mp4"
                open(out, "wb").write(data)
                clip = out; dur = min(8.0, ff.probe(out).get("duration", 6) or 6)
        else:
            img, _txt, _r = genmedia.image(sc["prompt"],
                                           model=ctx().settings.model_for("image_edit"))
            if img:
                png = f"out/studio/exports/key_{int(time.time())}_{i}.png"
                open(png, "wb").write(img)
                ctx().library.add("image", png, caption=sc["prompt"][:60],
                                  tags=["story", "keyframe"])
                cl = ff.still_to_clip(png, 3.5)
                if cl.get("status") == "success":
                    clip = cl["output"]
        if not clip:
            continue
        ctx().ledger.record("agent_brain",
                            model=ctx().settings.model_for("image_edit" if mode == "fast" else "video"),
                            input_tokens=300, output_tokens=1300, label="story scene")
        ctx().library.add("video", clip, caption=sc["prompt"][:60], tags=["story", "scene"])
        video.append({"src": os.path.basename(clip), "t0": round(t0, 2), "offset": 0,
                      "duration": round(dur, 2), "prompt": sc.get("prompt", ""),
                      "narration": sc.get("narration", "")})
        if narrate_scenes and sc.get("narration"):
            nar = narrate(sc["narration"])
            if nar.get("status") == "success":
                audio.append({"src": os.path.basename(nar["output"]), "t0": round(t0, 2),
                              "offset": 0, "duration": round(dur, 2), "gain": 1.0})
        t0 += dur; made += 1
    if not made:
        return {"status": "error", "error": "scene generation failed"}
    if ctx().brain:
        ctx().brain.remember("last_story", brief[:120])
    return {"status": "success", "timeline": {"video": video, "audio": audio},
            "scenes": made, "brief": brief, "mode": mode, "ui": "edit"}


ALL_TOOLS = [
    plan, drive_ui, set_source, set_detector, set_task, set_tracker, set_detect_every, set_render, set_detection,
    draw_zone, draw_line, clear_annotations,
    list_overlays, toggle_overlay, create_overlay, remove_overlay, clear_overlays,
    apply_face_filter, try_eyewear, shop_search, try_product,
    analyze_scene, analyze_image, describe_image, display_media, test_image,
    run_cv_code, run_cv_video, emit_proto,
    nano_banana, virtual_try_on, generate_video, extend_video, narrate, generate_music,
    search_library, list_library, show_media, save_to_library,
    load_youtube, save_source, list_sources, use_source, co_direct, direct_story,
    export_gif, export_clip, export_contact_sheet, speed_ramp, probe_media,
    edit_video, record_start, record_stop, snapshot,
    remember, recall, set_simulation,
]
TOOLS_BY_NAME = {f.__name__: f for f in ALL_TOOLS}
