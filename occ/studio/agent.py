"""The Vision Ninja agent — ADK brain with a deterministic SimRunner fallback.

Real mode (a Gemini key in env, simulation != 'zero'): a Google ADK Agent with
the full tool belt reasons and acts, streaming NDJSON frames the browser applies.
No-key / sim mode: a rule-based planner runs the SAME tools and emits the SAME
frames — so the studio is fully alive at $0, and lights up the moment a key
appears. Mirrors Momentum's NDJSON event contract + tool-result dispatch.
"""

from __future__ import annotations

import asyncio
import glob
import os
import re

from . import tools as T
from .runtime import ctx

SYSTEM_PROMPT = """You are the VISION NINJA — a master computer-vision agent that
drives a live OpenCV studio through tools and conversation. You can reconfigure
the detection/tracking pipeline, draw zones/lines, AUTHOR arbitrary cv2 overlays
to achieve any visual task, export with ffmpeg, and narrate what you do.

Operating principles:
- Be decisive and autonomous. When asked for a visual outcome, achieve it: pick
  the detector, then `analyze_scene` to see what's there, then create or toggle
  the overlay(s) that deliver the look. Prefer authoring a precise `create_overlay`
  over vague words.
- Drive the UI as you go: `plan` first for multi-step work; `drive_ui` to toast /
  spotlight / pulse the relevant neuron so the user follows along.
- Keep overlays a few expressive lines using the rich `ctx` API. One visual idea
  per overlay. Name them clearly.
- Overlays don't compound: the studio auto-clears the previous overlays at the start
  of each new visual request, so just author the new look. To combine several effects,
  put them in ONE overlay. Use `clear_overlays` to wipe them all.
- FACE FILTERS / TRY-ON (great in Live Voice with the camera on): when the user
  asks to put something on their FACE — "ninja mask", "sunglasses", "glasses",
  "dog/cat filter", "mustache", "crown", "clown nose", "heart eyes", "face mesh",
  "blur/anonymize my face" — call `apply_face_filter(name)`. It uses dense 478-pt
  landmarks and follows head pose. For CLOTHES/GARMENTS ("try this jacket/shirt on
  me") call `virtual_try_on(garment=<image path/URL/data URI>, which='frame')` — it
  dresses the person in the live frame and shows the result on the canvas.
- YOU ARE ALSO THE STORE'S STYLIST — you can SELL anything in the sponsored shops
  (Mode Marco apparel + Ralba Optical eyewear). When the user asks to try on or see
  a product ("try the HUGO aviators on me", "show me a navy polo", "put the Haiti
  jersey on me", "what sunglasses suit me"): call `try_product(query)` to find the
  best match and put it on them live (real-time AR for eyewear, generative for
  clothes), or `shop_search(query)` to recommend a few options. ALWAYS mention the
  brand and PRICE, suggest a tasteful alternative, and be a warm, concise salesperson.
  When the user says "reset", "clear the canvas", "go back to live", "take it/them
  off", or "remove the glasses/jacket/filter" → call `go_live` (clears the try-on +
  returns to the live video).
  GLASSES are ONE unified live try-on (try_product/try_eyewear render the frames
  front-on and fit them to the face — it takes a few seconds while it tailors, and
  the screen shows a 'tailoring your fit' loader). Say something warm meanwhile,
  like "let me tailor these to you". To change the LENS colour/tint on the glasses
  they're wearing ("make them clear / darker / blue / mirror / gold sunglasses")
  call `set_lens_tint(tint, opacity)`.
- TOUCHLESS SHOPPING (be its co-pilot). In live voice with the camera on, the user's
  SELF-VIEW shows on-screen icon buttons — Prev, Next, Try on, Store, Clear — that
  they press by pointing their hand and holding. It is the SAME engine you drive with
  `live_control` (one unified path, never a parallel one). So you can do everything
  the buttons do, hands-free: `live_control('next'|'prev')` to step the catalogue,
  `live_control('try')` to try the current item on (this runs a 3-2-1 'STRIKE A POSE'
  countdown, then fits it — tell them "okay, strike a pose!"), `live_control('store')`
  to switch eyewear⇄apparel, `live_control('clear')` to wipe everything back to clean
  live. INSTANT SEARCH — whenever the user asks for ANYTHING ("show me red aviators",
  "got navy polos?", "Haiti jersey", "round tortoise frames"), call
  `live_control(query=...)`: it searches the stores and surfaces the matches into BOTH
  the live browser (Prev/Next/Try now browse exactly those results) AND the store strip,
  auto-switching store to fit. Then say what you found and offer to 'try' the top one.
  Be aware of what's selected (the tool returns the current item + price) and SIMPLIFY:
  if they seem stuck, just do it for them and narrate ("I've got the next pair up —
  want to try them? strike a pose in 3…"). Prefer `live_control` for the live shopping
  flow so you and the buttons stay perfectly in sync.
- DETECTION/TRACKING/OCCUPANCY is OFF by default once the user's CAMERA is live
  (a selfie feed isn't a scene to analyze). If they ask to detect / track / COUNT
  objects, find/highlight things, or do occupancy — call `set_detection(True)`
  first, then proceed. Face filters / AR try-on do NOT need it.
- Save durable facts to the brain with `remember`. Use ffmpeg tools to export.
- THE CANVAS: the center area shows EITHER the live feed OR an artifact. When the
  user shares or asks about an image, call `analyze_image` (it ingests a data: URI /
  path / URL, runs detection, and shows the annotated still). When you export a gif /
  clip / contact sheet, it auto-shows. Use `display_media('live')` to return to the
  feed. Move the canvas to whatever the task needs the user to see.
- Narrate briefly in plain language. Cost is metered in compute points; you don't
  pay per word, but be efficient.

LIVE VISION BRAIN:
{brain}
"""

_HAS_KEY = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_GENAI_API_KEY"))

# COCO-80 names YOLO knows; anything else routes to OWLv2 open-vocab.
_COCO = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
         "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow",
         "backpack", "umbrella", "handbag", "suitcase", "bottle", "cup",
         "chair", "couch", "tv", "laptop", "cell phone", "book", "clock"}


# tools whose work makes the user WAIT (Gemini) → show the courteous fitting loader
_FITTING_TOOLS = {
    "try_eyewear": ("Tailoring your fit", "fitting your frames to your face…"),
    "try_product": ("Finding & tailoring your fit", "one moment…"),
    "virtual_try_on": ("Styling your look", "dressing you in the look…"),
}


def fitting_on(name: str):
    if name in _FITTING_TOOLS:
        t, s = _FITTING_TOOLS[name]
        return {"type": "fitting", "data": {"on": True, "title": t, "sub": s}}
    return None


# ---------------- NDJSON frame dispatch ----------------
def _ui_action(res: dict) -> dict:
    data = {k: v for k, v in res.items()
            if k in ("action", "target", "title", "caption", "value") and v != ""}
    return {"type": "ui_action", "data": data}


_LIVE_TOOLS = {"set_source", "set_detector", "set_task", "set_tracker", "set_detect_every", "set_render",
               "draw_zone", "draw_line", "clear_annotations", "toggle_overlay",
               "create_overlay", "remove_overlay", "clear_overlays", "go_live", "use_source",
               "load_youtube"}


def _file_url(path: str) -> str:
    return ("/download/" + os.path.basename(path)) if path else ""


def _canvas_frame(name: str, res: dict):
    """Decide what the center canvas should show after this tool — the studio
    'moves between modes' here: artifacts -> media canvas, pipeline work -> live."""
    kind = res.get("kind")
    if name in ("display_media", "show_media"):
        return {"type": "canvas", "data": {"mode": res.get("mode", "live"),
                                           "src": res.get("src", ""),
                                           "caption": res.get("caption", "")}}
    if name == "analyze_image" or kind in ("image", "snapshot", "sheet", "gif"):
        return {"type": "canvas", "data": {"mode": "image",
                                           "src": _file_url(res.get("output", "")),
                                           "caption": res.get("caption") or kind or ""}}
    if kind in ("clip", "speed", "concat", "transcode"):
        return {"type": "canvas", "data": {"mode": "video",
                                           "src": _file_url(res.get("output", "")),
                                           "caption": kind}}
    if name in _LIVE_TOOLS:
        return {"type": "canvas", "data": {"mode": "live"}}
    return None


def frames_for(name: str, res) -> list[dict]:
    """Map a tool result to NDJSON frames the UI applies."""
    out: list[dict] = []
    if not isinstance(res, dict):
        return [{"type": "log", "data": {"text": str(res)}}]
    ok = res.get("status") == "success"
    if name == "plan":
        out.append({"type": "plan", "data": {"steps": res.get("steps", [])}})
    elif name == "drive_ui":
        out.append(_ui_action(res))
    elif name == "analyze_scene":
        out.append({"type": "scene", "data": res})
    if res.get("kind") in ("gif", "clip", "sheet", "transcode", "speed", "concat",
                           "snapshot", "image", "audio", "scene", "edit", "proto"):
        out.append({"type": "export", "data": res})
    # production receipt: every scene of an assembled/generated edit, in the chat
    tl = res.get("timeline")
    if isinstance(tl, dict):
        for c in (tl.get("video") or [])[:8]:
            if c.get("src"):
                out.append({"type": "export", "data": {"output": c["src"], "kind": "scene"}})
    if ok:
        cf = _canvas_frame(name, res)
        if cf:
            out.append(cf)
    if isinstance(res.get("timeline"), dict):
        out.append({"type": "timeline", "data": {"op": "set", **res["timeline"]}})
    if name in ("search_library", "list_library"):
        out.append({"type": "library", "data": {"items": res.get("items", []),
                                                "query": res.get("query", "")}})
    if name in ("list_sources", "save_source", "use_source"):
        srcs = ctx().sources.list() if ctx().sources else []
        out.append({"type": "sources", "data": {"items": srcs}})
    if name in _FITTING_TOOLS:                  # close the fitting loader
        out.append({"type": "fitting", "data": {"on": False}})
    out.append({"type": "tool", "data": {"name": name, "ok": ok,
                                         "detail": res.get("error", "")}})
    if res.get("ui"):
        out.append({"type": "refresh", "data": {"panel": res["ui"]}})
    if name in ("create_overlay", "toggle_overlay", "remove_overlay", "clear_overlays",
                "clear_annotations", "apply_face_filter", "try_eyewear", "try_product"):
        out.append({"type": "refresh", "data": {"panel": "overlays"}})
    # the ninja-as-stylist: show shopped products in the try-on strip
    if name == "shop_search":
        out.append({"type": "catalog", "data": {"store": res.get("store", ""),
                                                "matches": res.get("matches", [])}})
    if isinstance(res.get("catalog"), dict):
        out.append({"type": "catalog", "data": res["catalog"]})
    if name in ("nano_banana", "virtual_try_on", "describe_image", "analyze_image",
                "save_to_library", "generate_video", "extend_video"):
        out.append({"type": "refresh", "data": {"panel": "library"}})
    return out


async def _call(name: str, **kwargs):
    """Invoke a tool, yield its log + result frames. Returns the raw result."""
    fn = T.TOOLS_BY_NAME[name]
    res = fn(**kwargs)
    return res


# ---------------- the deterministic SimRunner (no-key path) ----------------
class SimRunner:
    """A rule-based Vision Ninja: parses intent, runs the real tools, narrates."""

    async def stream(self, message: str):
        m = message.lower().strip()
        yield {"type": "log", "data": {"text": "thinking…"}}
        await asyncio.sleep(0.1)

        steps, actions = self._route(m, message)
        if len(steps) > 1:
            res = T.plan(steps)
            for f in frames_for("plan", res):
                yield f
            await asyncio.sleep(0.12)

        # brain meter (simulated tokens) — the brain "thought"
        if ctx().ledger:
            ctx().ledger.record("agent_brain", model="gemini-2.5-flash",
                                input_tokens=len(message) // 4 + 700,
                                output_tokens=240, simulated=True, label="sim turn")

        narration = []
        for name, kwargs, say in actions:
            yield {"type": "log", "data": {"text": f"{name} …"}}
            await asyncio.sleep(0.12)
            try:
                res = T.TOOLS_BY_NAME[name](**kwargs)
            except Exception as e:
                res = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            for f in frames_for(name, res):
                yield f
            if res.get("status") == "success":
                if say:
                    narration.append(say)
                elif name == "analyze_image":
                    narration.append(f"That image shows {res.get('summary', '—')} "
                                     f"({res.get('objects', 0)} object(s)) — I put the "
                                     f"annotated version on the canvas.")
            elif res.get("status") == "error":
                narration.append(f"({name} failed: {res.get('error')})")
            await asyncio.sleep(0.1)

        text = " ".join(narration) or "Done."
        for chunk in re.findall(r"\S+\s*", text):
            yield {"type": "text_delta", "data": {"text": chunk}}
            await asyncio.sleep(0.015)
        yield {"type": "final", "data": {"text": text}}
        yield {"type": "done", "data": {}}

    # ---- intent routing: message -> (plan steps, [ (tool, kwargs, narration) ]) ----
    def _route(self, m: str, original: str):
        acts: list[tuple] = []
        steps: list[str] = []

        def add(tool, kwargs, say, step):
            acts.append((tool, kwargs, say)); steps.append(step)

        # shared image FIRST — pull the data: URI / image path out so its base64
        # can't pollute the keyword routing below.
        mimg = re.search(r"data:image/[^\s\"']+", original)
        img = mimg.group(0) if mimg else ""
        if not img:
            mf = re.search(r"(https?://\S+\.(?:png|jpe?g|webp)|[\w./\-]+\.(?:png|jpe?g|webp))",
                           original, re.I)
            img = mf.group(1) if mf else ""
        if img:
            m = m.replace(img.lower(), " ")
            add("analyze_image", {"image": img, "caption": "shared image"}, None,
                "ingest + analyze the image")

        # reset / back to the live video (take off the try-on)
        # touchless live-shopping — the SAME engine the on-screen buttons drive
        if any(p in m for p in ("next item", "next one", "next pair", "next look",
                                "show me the next", "go next", "scroll right")):
            add("live_control", {"action": "next"}, "Here's the next one.", "next item")
            return steps, acts
        if any(p in m for p in ("previous item", "previous one", "go back one", "last one",
                                "previous pair", "go prev", "scroll left", "the one before")):
            add("live_control", {"action": "prev"}, "Back one.", "previous item")
            return steps, acts
        if any(p in m for p in ("switch store", "switch the store", "other store",
                                "show me glasses instead", "show me clothes instead",
                                "switch to eyewear", "switch to apparel", "change store")):
            add("live_control", {"action": "store"}, "Switched the store.", "switch store")
            return steps, acts
        if any(p in m for p in ("try this on", "try that on", "try these on", "try it on me",
                                "put this on me", "fit this", "try the current", "strike a pose")):
            add("live_control", {"action": "try"}, "Okay — strike a pose!", "try it on")
            return steps, acts

        if any(p in m for p in ("reset", "go back to live", "back to live", "go live",
                                "take it off", "take them off", "remove the glass",
                                "remove the jacket", "remove the shirt", "remove the filter",
                                "remove the mask", "clear the canvas", "live video")):
            add("go_live", {}, "Back to the live video.", "reset to live")
            return steps, acts

        # clear annotations / overlays
        if any(p in m for p in ("clear all", "clear annotation", "clear overlay",
                                "clear everything", "remove all overlay", "wipe the",
                                "clear the annotation", "clear the overlay", "reset the overlay")):
            if "overlay" in m and not any(w in m for w in ("annotation", "all", "everything")):
                add("clear_overlays", {}, "Removed all overlays.", "clear overlays")
            else:
                add("clear_annotations", {}, "Cleared all zones, lines and overlays.", "clear all")

        # ---- AR face filters (ninja mask, sunglasses, dog…) + garment try-on ----
        _FILT = [("ninja", "ninja_mask"), ("sunglass", "sunglasses"), ("shades", "sunglasses"),
                 ("eyeglass", "glasses"), ("spectacle", "glasses"), ("glasses", "glasses"),
                 ("puppy", "dog"), ("dog", "dog"), ("kitt", "cat"), ("cat", "cat"),
                 ("mustache", "mustache"), ("moustache", "mustache"), ("crown", "crown"),
                 ("king", "crown"), ("queen", "crown"), ("clown", "clown_nose"),
                 ("heart eye", "heart_eyes"), ("face mesh", "face_mesh"),
                 ("wireframe", "face_mesh"), ("anonymi", "anonymize"), ("pixelate", "anonymize")]
        fname = next((v for k, v in _FILT if k in m), None)
        garment = any(w in m for w in ("jacket", "shirt", "t-shirt", "tshirt", "dress",
                      "outfit", "hoodie", "coat", "sweater", "garment", "clothes", "wear "))
        trigger = any(w in m for w in ("filter", "mask", "on me", "on my", "put ", "give me",
                                       "apply", "try on", "tryon", "try it"))
        if garment and (img or "http" in original):
            g = img or (re.search(r"https?://\S+", original) or [None])
            g = g if isinstance(g, str) else (g.group(0) if g else "")
            add("virtual_try_on", {"garment": g, "which": "frame"},
                "Trying that garment on you…", "virtual try-on")
            return steps, acts
        if fname and (trigger or "filter" in m):
            add("apply_face_filter", {"name": fname},
                f"Applying the {fname.replace('_', ' ')} look.", f"face filter: {fname}")
            return steps, acts

        # ---- ninja-as-stylist: shop & try REAL products by voice ----
        shop_trigger = any(w in m for w in ("try on", "try the", "put the", "put on",
            "wear", "show me", "do you have", "recommend", "sell me", "i want",
            "looking for", "what about", "suits me", "suit me"))
        prod_word = any(w in m for w in (
            "glass", "sunglass", "shade", "frame", "eyewear", "aviator", "optical",
            "polo", "shirt", "tee", "jersey", "jacket", "coat", "dress", "jean",
            "sweater", "hoodie", "knit", "blazer", "hugo", "carrera", "calvin klein",
            "coach", "gant", "under armour", "ralba", "mode marco", "haiti"))
        if shop_trigger and prod_word:
            if any(w in m for w in ("show me", "do you have", "recommend", "options",
                                    "what about", "looking for", "browse", "any ",
                                    "find me", "got any")):
                # instant search → surfaced to the live browser AND the store strip
                add("live_control", {"query": original},
                    "Here's what I found — browse them on your self-view or say 'try it on'.",
                    "search the stores")
            else:
                add("try_product", {"query": original},
                    "Finding that and trying it on you…", "try it on live")
            return steps, acts

        # hide / show the detection boxes + tracking
        if any(p in m for p in ("detection box", "detection boxes", "hide the box",
                                "hide box", "clear the box", "clear box", "remove the box",
                                "hide detection", "clear detection", "hide tracking",
                                "no boxes", "without boxes", "turn off detection",
                                "hide the label")):
            add("set_render", {"boxes": False, "labels": False, "trails": False,
                               "counts": True}, "Hid the detection boxes, labels and trails.",
                "hide detections")
        elif any(p in m for p in ("show detection", "show the box", "show boxes",
                                  "turn on detection", "bring back the box", "show tracking",
                                  "boxes back")):
            add("set_render", {"boxes": True, "labels": True, "trails": True,
                               "counts": True}, "Detection boxes are back on.",
                "show detections")

        # Co-Director generative storytelling (generate footage)
        ms = re.search(r"(?:direct a story|generate (?:a )?(?:video|story|reel|footage)|"
                       r"make (?:a )?(?:story|reel)|create (?:a )?reel|story about|video story)"
                       r"\s*:?\s*(.*)", m)
        if ms:
            add("direct_story", {"brief": (ms.group(1).strip() or original), "mode": "fast"},
                "Planned, generated and assembled the story into the Editor.",
                "direct a generated story")
        # Co-Director / auto-edit (stitch existing footage)
        mc = re.search(r"co-?direct(?:or)?\s*:?\s*(.*)", original, re.I)
        if not ms and (mc or any(p in m for p in ("make an edit", "build the edit",
                "highlight reel", "auto edit", "auto-edit", "cut a reel", "assemble a"))):
            brief = (mc.group(1).strip() if mc and mc.group(1).strip() else original)
            add("co_direct", {"brief": brief or "a short reel"},
                "Assembled an edit and loaded it into the Editor — refine and Export.",
                "co-direct the edit")

        # YouTube ingest (URL anywhere in the message)
        myt = re.search(r"(https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+)", original)
        if myt:
            add("load_youtube", {"url": myt.group(1), "seconds": 30},
                "Pulled the YouTube video and set it as the live source.", "load youtube")

        # save an RTSP camera to live sources
        mrtsp = re.search(r"(rtsp://\S+)", original)
        if mrtsp:
            add("save_source", {"url": mrtsp.group(1)},
                "Saved that RTSP camera to live sources.", "save live source")

        # semantic visual analysis (collision-free phrases)
        if any(p in m for p in ("what is this", "what's this", "whats this",
                                "what's happening", "whats happening", "describe the",
                                "describe this", "read the", "what's in the image",
                                "whats in the image", "who is", "what do you see in")):
            add("describe_image", {}, None, "describe with Gemini vision")

        # nano-banana frame edit (visual-restyle phrases that don't collide with overlays)
        if any(p in m for p in ("make it look", "make it night", "at night", "make night",
                                "restyle", "cartoon", "anime", "neon", "cyberpunk",
                                "make it rain", "make it snow", "snowy", "recolor the",
                                "turn the sky", "vintage look", "make the scene", "repaint")):
            add("nano_banana", {"prompt": original}, None, "nano-banana edit")

        # search the media library
        if "library" in m or re.search(r"(?:search|find).*(?:clip|gif|recording|snapshot|media)", m):
            q = re.sub(r".*(?:search|find|show me)\s+", "", m)
            q = re.sub(r"\b(?:in (?:the )?library|the library|library)\b.*", "", q).strip()
            add("search_library", {"query": q}, None, "search library")

        # save current frame to the library
        if any(p in m for p in ("save this frame", "save the frame", "save to library",
                                "save to the library", "bookmark this", "keep this frame")):
            add("save_to_library", {}, "Saved the current frame to the library.",
                "save to library")

        # ffmpeg edits on a clip
        for op, words, look in [("reverse", ("reverse",), ""), ("boomerang", ("boomerang",), ""),
                                ("fade", ("fade ",), ""), ("filter", ("grayscale", "black and white", "b&w"), "grayscale"),
                                ("filter", ("sepia",), "sepia"), ("filter", ("invert",), "invert")]:
            if any(w in m for w in words):
                kw = {"op": op, "which": "recording"}
                if look:
                    kw["look"] = look
                add("edit_video", kw, f"Applied {look or op} with ffmpeg.", f"{look or op} clip")

        # YOLO task switch: segment / pose / obb / classify
        mt = None
        if any(w in m for w in ("segment", "segmentation", "instance mask", " masks")):
            mt = "segment"
        elif any(w in m for w in ("pose", "skeleton", "keypoint", "key point")):
            mt = "pose"
        elif any(w in m for w in ("oriented box", "obb", "rotated box")):
            mt = "obb"
        elif any(w in m for w in ("classify", "classification", "classify the")):
            mt = "classify"
        if mt:
            add("set_task", {"task": mt}, f"Switched the YOLO task to {mt}.", f"task: {mt}")

        # test card / sample image
        if any(p in m for p in ("test image", "test card", "test pattern",
                                "sample image", "show me an image", "show an image",
                                "example image", "show a test", "show me a test")):
            add("test_image", {}, "Rendered a test card on the canvas.",
                "render test card")

        # canvas mode commands
        if any(p in m for p in ("back to live", "live view", "live feed", "go live",
                                "show the live", "show me live", "back to the feed")):
            add("display_media", {"kind": "live"}, "Back to the live feed.", "show live")

        # source switch: explicit "...vehicles-2.mp4" / "switch to market", OR a bare
        # "switch the video" with no name -> cycle to the next available clip.
        msrc = re.search(r"([\w\-/]+\.mp4)", m)
        mname = re.search(r"(?:switch|change|use|play|load|go) to (\w[\w\- ]*)", m)
        wants_switch = any(p in m for p in (
            "switch the video", "switch video", "change the video", "change video",
            "switch source", "change source", "next video", "another video",
            "different video", "switch the source", "other video", "new video",
            "switch the clip", "next clip", "change the clip", "rotate the video",
            "switch the feed", "change the feed"))
        if msrc or mname or wants_switch:
            vids = sorted(glob.glob("assets/videos/*.mp4"))
            cur = os.path.basename(str(ctx().pipe.cfg.get("source.uri", "")))
            target = None
            if msrc:
                p = msrc.group(1)
                target = p if "/" in p else f"assets/videos/{p}"
            elif mname:                          # fuzzy: "switch to market" -> market-square.mp4
                key = mname.group(1).strip().replace(" ", "-")
                target = next((v for v in vids if key in os.path.basename(v).lower()), None)
            if not target and vids:              # bare "switch the video" -> next in the list
                i = next((k for k, v in enumerate(vids)
                          if os.path.basename(v) == cur), -1)
                target = vids[(i + 1) % len(vids)]
            if target:
                add("set_source", {"source": target},
                    f"Switched to {os.path.basename(target)}.",
                    f"switch to {os.path.basename(target)}")

        classes = [c for c in ("person", "people", "car", "truck", "bus", "bicycle",
                               "motorcycle", "dog", "cat") if c in m]
        nicewords = self._novel_class(m)

        # open-vocab (non-COCO) target -> OWLv2
        if nicewords and not classes:
            add("set_detector", {"backend": "owlv2", "prompt": nicewords},
                f"Pointed OWLv2 open-vocab at '{nicewords}'.",
                f"set OWLv2 to find {nicewords}")
            add("create_overlay", self._highlight_overlay(nicewords.split(",")[0].strip()),
                f"Authored a ring overlay around every {nicewords.split(',')[0].strip()}.",
                "author highlight overlay")

        # heatmap / density / busiest / crowd
        if any(w in m for w in ("heatmap", "heat map", "density", "busiest",
                                "crowd", "hot spot", "hotspot", "congest")):
            add("analyze_scene", {}, None, "look at the scene")
            add("toggle_overlay", {"name": "density_heatmap", "on": True},
                "Lit up a density heatmap over the ground anchors.", "enable heatmap")

        # trails / paths
        if any(w in m for w in ("trail", "path", "trajector", "where they go", "route")):
            add("toggle_overlay", {"name": "track_trails", "on": True},
                "Turned on fading motion trails per track.", "enable trails")

        # speed / velocity
        if any(w in m for w in ("speed", "velocit", "how fast", "moving fast")):
            add("toggle_overlay", {"name": "speed_vectors", "on": True},
                "Added per-track velocity arrows.", "enable speed vectors")

        # counts / tally
        if any(w in m for w in ("count", "tally", "how many", "number of")):
            add("toggle_overlay", {"name": "count_badge", "on": True},
                "Pinned a live per-class tally.", "enable count badge")

        # highlight a COCO class
        if classes and not nicewords:
            cls = "person" if classes[0] == "people" else classes[0]
            keep = "person" if cls == "person" else cls
            add("set_detector", {"backend": "yolo", "classes": keep},
                f"Filtered detection to {keep}.", f"detect only {keep}")
            add("create_overlay", self._highlight_overlay(cls),
                f"Ringed every {cls} in magenta.", "author highlight overlay")

        # crossing line
        if any(w in m for w in ("crossing", "cross line", "entering", "count line",
                                "tripwire")):
            add("draw_line", {"points": [[0.05, 0.6], [0.95, 0.6]], "name": "countline"},
                "Dropped a horizontal counting line across the road.", "draw count line")

        # zone
        if "zone" in m or "region" in m or "area" in m:
            add("draw_zone", {"points": [[0.2, 0.55], [0.8, 0.55], [0.8, 0.95],
                                         [0.2, 0.95]], "name": "roi"},
                "Marked a region-of-interest zone.", "draw zone")

        # ffmpeg exports
        if "gif" in m:
            add("export_gif", {"which": "source", "duration": 4},
                "Rendered a GIF with ffmpeg.", "export gif")
        if "timelapse" in m or "speed up" in m or "time lapse" in m:
            add("speed_ramp", {"which": "recording", "factor": 6.0},
                "Made a 6x timelapse.", "speed ramp")
        if "contact sheet" in m or "thumbnail" in m or "summary image" in m or "storyboard" in m:
            add("export_contact_sheet", {"which": "source"},
                "Built a contact-sheet summary.", "contact sheet")
        if "clip" in m or ("cut" in m and "video" in m):
            add("export_clip", {"which": "recording", "duration": 8},
                "Cut an MP4 clip.", "export clip")
        if "snapshot" in m or "screenshot" in m or "grab a frame" in m:
            add("snapshot", {}, "Saved a PNG snapshot.", "snapshot")

        # remember
        mr = re.search(r"remember (?:that )?(.+)", original, re.I)
        if mr:
            add("remember", {"key": "note", "text": mr.group(1).strip()},
                "Noted that in the Vision Brain.", "remember note")

        # pure scene question / fallback
        if not acts:
            add("analyze_scene", {}, None, "look at the scene")
            sc = ctx().pipe.scene()
            counts = ", ".join(f"{v} {k}" for k, v in (sc.get("counts") or {}).items()) or "nothing yet"
            acts[-1] = ("analyze_scene", {},
                        f"Right now I see {counts} across {sc.get('tracks', 0)} tracks "
                        f"at {sc.get('fps')} fps. I can: switch the video, highlight a "
                        f"class, add a heatmap / trails / speed vectors, draw a zone or "
                        f"counting line, analyze an image you paste, show a test card, or "
                        f"export a gif/clip. What look do you want?")

        # always finish by focusing the live view
        add("drive_ui", {"action": "focus", "target": "live",
                         "caption": "driving the live view"}, None, "focus live view")
        return steps, acts

    def _novel_class(self, m: str) -> str:
        """Pull a 'find X' target that isn't a COCO class -> OWLv2 prompt."""
        mt = re.search(r"(?:find|detect|spot|highlight|locate|track)\s+(?:the\s+|all\s+|every\s+)?"
                       r"([a-z][a-z \-]{2,30}?)(?:\s+(?:in|on|that|which|near|and)\b|[.?!,]|$)", m)
        if not mt:
            return ""
        term = mt.group(1).strip().rstrip("s")
        head = term.split()[-1]
        if head in _COCO or term in _COCO or head in ("object", "thing", "everything",
                                                      "one", "it", "them"):
            return ""
        return term

    def _highlight_overlay(self, cls: str) -> dict:
        code = (
            "def draw(ctx):\n"
            "    import math\n"
            f"    m = ctx.mask('{cls}')\n"
            "    r = 24 + int(6*math.sin(ctx.t*0.2))\n"
            "    for (cx, cy) in ctx.anchors[m]:\n"
            "        ctx.ring((cx, cy-20), r, 'magenta', 2, glow=True)\n"
            f"    ctx.text(f'{cls}: '+str(int(m.sum())), (12, 60), 'magenta', 0.7)\n")
        return {"name": f"highlight_{cls}", "intent": f"ring every {cls}", "code": code}


# ---------------- the ADK runner (real path) ----------------
class ADKRunner:
    def __init__(self, settings):
        from google.adk.agents import Agent
        from google.adk.runners import InMemoryRunner
        self.settings = settings
        brain = ctx().brain.prompt_context() if ctx().brain else ""
        self.agent = Agent(name="vision_ninja", model=settings.agent_model,
                           instruction=SYSTEM_PROMPT.format(brain=brain),
                           tools=list(T.ALL_TOOLS))
        self.runner = InMemoryRunner(agent=self.agent, app_name="studio")
        self.user_id = "local"
        self.session_id = "studio-session"
        self._session_ready = False

    async def _ensure_session(self):
        if not self._session_ready:
            await self.runner.session_service.create_session(
                app_name="studio", user_id=self.user_id, session_id=self.session_id)
            self._session_ready = True

    async def stream(self, message: str):
        from google.adk.agents.run_config import (RunConfig, StreamingMode,
                                                  ToolThreadPoolConfig)
        from google.genai import types
        await self._ensure_session()
        new_message = types.Content(role="user",
                                    parts=[types.Part.from_text(text=message)])
        streamed: list[str] = []          # live token deltas (partial events)
        final_text = ""                   # complete aggregate (final event)
        async for event in self.runner.run_async(
                user_id=self.user_id, session_id=self.session_id,
                new_message=new_message,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE,
                                     tool_thread_pool_config=ToolThreadPoolConfig(max_workers=4))):
            for fc in event.get_function_calls() or []:
                f = fitting_on(fc.name)
                if f:
                    yield f
                yield {"type": "log", "data": {"text": f"{fc.name} …"}}
            for fr in event.get_function_responses() or []:
                for f in frames_for(fr.name, fr.response):
                    yield f
            if event.content and event.content.parts:
                text = "".join(p.text for p in event.content.parts
                               if getattr(p, "text", None))
                if text:
                    if getattr(event, "partial", False):
                        streamed.append(text)
                        yield {"type": "text_delta", "data": {"text": text}}
                    elif event.is_final_response():
                        final_text = text             # the full message (don't re-stream)
                    elif not streamed:                # non-streaming model: emit once
                        streamed.append(text)
                        yield {"type": "text_delta", "data": {"text": text}}
            um = getattr(event, "usage_metadata", None)
            if um and ctx().ledger:
                ctx().ledger.record(
                    "agent_brain", model=self.settings.agent_model,
                    input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                    label="adk turn")
        yield {"type": "final", "data": {"text": "".join(streamed) or final_text}}
        yield {"type": "done", "data": {}}


class StudioAgent:
    def __init__(self, settings):
        self.settings = settings
        self._adk = None
        self._sim = SimRunner()

    def mode(self) -> str:
        if _HAS_KEY and self.settings.simulation != "zero":
            return "adk"
        return "sim"

    @staticmethod
    def _maybe_clear_overlays(message: str):
        """Don't compound overlays: at the START of a NEW visual/overlay request,
        wipe the current overlays once — so the new look replaces, not stacks. Skip
        only when the user explicitly wants to keep/add to the current look."""
        m = message.lower()
        overlay_req = any(w in m for w in (
            "ring", "highlight", "heatmap", "heat map", "trail", "overlay", "skeleton",
            "glow", "speed vector", "outline", "mask", "circle the", "box the",
            "draw a", "draw the", "color the", "colour the", "mark the", "densest",
            "lane", "pose", "segment"))
        keep = any(p in m for p in (
            "keep the", "in addition", "on top of the current", "as well as the current",
            "don't clear", "do not clear", "add to the current", "leave the current",
            "keep the current", "also keep", "without clearing"))
        try:
            if overlay_req and not keep and ctx().overlays and ctx().overlays.active_count():
                from .tools import _clear_all_overlays
                _clear_all_overlays()
        except Exception:
            pass

    async def stream(self, message: str):
        if ctx().brain:
            task = message if not message.startswith("data:") else "[shared image]"
            ctx().brain.set_task(task[:200])
        self._maybe_clear_overlays(message)
        if self.mode() == "adk":
            try:
                if self._adk is None:
                    self._adk = ADKRunner(self.settings)
                async for f in self._adk.stream(message):
                    yield f
                return
            except Exception as e:                     # graceful fallback (Momentum-style)
                yield {"type": "log", "data": {"text": f"brain fell back to sim: {e}"}}
        async for f in self._sim.stream(message):
            yield f
