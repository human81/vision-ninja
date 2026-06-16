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


def set_tracker(algorithm: str) -> dict:
    """Set the tracker: bytetrack|botsort|ocsort|sort."""
    ctx().pipe.reconfigure({"tracker.algorithm": algorithm})
    return {"status": "success", "tracker": algorithm, "ui": "config"}


def set_detect_every(n: int) -> dict:
    """Run detection every Nth frame (1=every frame, higher=cheaper/faster).
    Tracking holds objects between detections."""
    ctx().pipe.reconfigure({"runtime.detect_every": max(1, int(n))})
    return {"status": "success", "detect_every": max(1, int(n)), "ui": "config"}


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


def clear_annotations() -> dict:
    """Remove all zones and lines."""
    ctx().pipe.set_annotations([])
    return {"status": "success", "ui": "annotations"}


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
    your superpower. `code` must define `def draw(ctx):` and may use cv2/np plus the
    rich ctx API:
      ctx.frame (BGR, mutate in place), ctx.w/ctx.h/ctx.t (frame index)
      ctx.boxes (Nx4 int xyxy), ctx.centers, ctx.anchors (bottom-center),
      ctx.names, ctx.ids, ctx.confs, ctx.mask('car','truck'), ctx.palette(i)
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
    prompt = ("Dress the person in the first image with the garment shown in the second "
              "image. Keep the person's identity, pose, lighting and background.")
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
        try:
            p = subprocess.run([sys.executable, "-m", "yt_dlp", "-g", "-f",
                                "best[height<=720]/best", url],
                               capture_output=True, text=True, timeout=60)
            lines = (p.stdout or "").strip().splitlines()
            if lines:
                url = lines[-1]
        except Exception:
            pass
    ctx().pipe.reconfigure({"source.uri": url})
    return {"status": "success", "source": url, "name": item["caption"], "ui": "config"}


ALL_TOOLS = [
    plan, drive_ui, set_source, set_detector, set_tracker, set_detect_every,
    draw_zone, draw_line, clear_annotations,
    list_overlays, toggle_overlay, create_overlay, remove_overlay,
    analyze_scene, analyze_image, describe_image, display_media, test_image,
    nano_banana, virtual_try_on, generate_video, extend_video,
    search_library, list_library, show_media, save_to_library,
    load_youtube, save_source, list_sources, use_source,
    export_gif, export_clip, export_contact_sheet, speed_ramp, probe_media,
    edit_video, record_start, record_stop, snapshot,
    remember, recall, set_simulation,
]
TOOLS_BY_NAME = {f.__name__: f for f in ALL_TOOLS}
