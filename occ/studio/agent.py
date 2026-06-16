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


# ---------------- NDJSON frame dispatch ----------------
def _ui_action(res: dict) -> dict:
    data = {k: v for k, v in res.items()
            if k in ("action", "target", "title", "caption", "value") and v != ""}
    return {"type": "ui_action", "data": data}


_LIVE_TOOLS = {"set_source", "set_detector", "set_tracker", "set_detect_every",
               "draw_zone", "draw_line", "clear_annotations", "toggle_overlay",
               "create_overlay", "remove_overlay", "use_source", "load_youtube"}


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
    out.append({"type": "tool", "data": {"name": name, "ok": ok,
                                         "detail": res.get("error", "")}})
    if res.get("ui"):
        out.append({"type": "refresh", "data": {"panel": res["ui"]}})
    if name in ("create_overlay", "toggle_overlay", "remove_overlay"):
        out.append({"type": "refresh", "data": {"panel": "overlays"}})
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
        from google.adk.agents.run_config import RunConfig, StreamingMode
        from google.genai import types
        await self._ensure_session()
        new_message = types.Content(role="user",
                                    parts=[types.Part.from_text(text=message)])
        streamed: list[str] = []          # live token deltas (partial events)
        final_text = ""                   # complete aggregate (final event)
        async for event in self.runner.run_async(
                user_id=self.user_id, session_id=self.session_id,
                new_message=new_message,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE)):
            for fc in event.get_function_calls() or []:
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

    async def stream(self, message: str):
        if ctx().brain:
            task = message if not message.startswith("data:") else "[shared image]"
            ctx().brain.set_task(task[:200])
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
