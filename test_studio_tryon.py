"""Regression test for the Live-Voice AR try-on stack — proves the 'magic' works
WITHOUT clicking the frontend.

Two phases:
  • OFFLINE (always): face landmarks, every filter draws, eyewear warp, the
    sponsored catalogs, shape-accurate search, store routing, tool registry.
  • ONLINE (only if the studio is up at $STUDIO_URL): the REST endpoints, and the
    real path browser→/ws/cam→pipeline→/stream.mjpg with filters + AR eyewear.
  • UI (only if the studio is up AND playwright is installed): a deterministic
    frontend smoke — store switch, filter chips, showCatalog — asserting 0 JS errors.

Run:  .venv/bin/python test_studio_tryon.py
      STUDIO_URL=http://127.0.0.1:8011 .venv/bin/python test_studio_tryon.py   # + online/UI
"""

import os
import sys
import time
import json
import inspect
import urllib.request

import cv2
import numpy as np

STUDIO_URL = os.environ.get("STUDIO_URL", "http://127.0.0.1:8011")
FACE = "assets/test/face.jpg"

_results = []
_VERBOSE = True


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    if _VERBOSE:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


def collect_offline():
    """Run the OFFLINE checks silently and return [(name, ok, detail), …].
    Lets fishfood fold these in as a level without the standalone printing."""
    global _VERBOSE
    _VERBOSE = False
    _results.clear()
    try:
        offline()
    except Exception as e:
        _results.append(("studio try-on offline crashed", False, repr(e)))
    _VERBOSE = True
    return list(_results)


# ----------------------------- OFFLINE -----------------------------
def offline():
    print("\n[OFFLINE] core logic + drawing (no server, no network)")
    import supervision as sv
    from occ.studio import tools as T
    from occ.studio.overlays import OverlayEngine
    from occ.studio.face_filters import (FACE_FILTERS, get_asset,
                                         set_current_eyewear)
    from occ.studio.facemesh import detect_faces

    # --- catalogs ---
    cats = T._load_catalogs()
    g, e = cats["apparel"], cats["eyewear"]
    check("apparel catalog loaded (>400)", len(g) > 400, f"{len(g)} items")
    check("eyewear catalog loaded (>300)", len(e) > 300, f"{len(e)} frames")
    check("every eyewear frame has a shape tag",
          all(it.get("shape") for it in e), f"{sum(bool(i.get('shape')) for i in e)}/{len(e)}")
    haiti = [it for it in g if it.get("cat") == "haiti"]
    check("Haiti World Cup category present", len(haiti) >= 1, f"{len(haiti)} jerseys")
    check("Mode Marco sponsor",
          json.loads(open("occ/studio/garments.json").read())["sponsor"]["name"] == "Mode Marco")
    check("Ralba Optical sponsor",
          json.loads(open("occ/studio/eyewear.json").read())["sponsor"]["name"] == "Ralba Optical")

    # --- shape-accurate search ---
    for shape in ("aviator", "round", "wayfarer", "cat-eye", "rectangle", "browline"):
        res = T._search_catalog(shape, store="eyewear")[:5]
        m = sum(1 for r in res if r.get("shape") == shape)
        check(f"search '{shape}' → {shape} frames", m >= 4, f"{m}/5")
    check("query_shape parses 'aviator sunglasses'", T._query_shape("aviator sunglasses") == "aviator")

    # --- store routing ---
    check("route 'navy polo' → apparel", T._guess_store("navy polo") == "apparel")
    check("route 'aviator sunglasses' → eyewear", T._guess_store("aviator sunglasses") == "eyewear")
    check("route 'round frames' → eyewear", T._guess_store("round frames") == "eyewear")
    check("search 'haiti jersey' finds Haiti",
          "haiti" in (T._search_catalog("haiti jersey", "apparel")[:1] or [{}])[0].get("title", "").lower())
    polo = T._search_catalog("navy polo", "apparel")[:1]
    check("search 'navy polo' → a navy polo",
          bool(polo) and "navy" in (polo[0].get("title", "").lower()))

    # --- tool registry ---
    names = {f.__name__ for f in T.ALL_TOOLS}
    for t in ("apply_face_filter", "try_eyewear", "try_product", "shop_search", "virtual_try_on"):
        check(f"tool registered: {t}", t in names)
    check("filter aliases resolve (ninja→ninja_mask)",
          T._FILTER_ALIASES.get("ninja") == "ninja_mask")

    # --- face landmarks ---
    face = cv2.imread(FACE)
    check("test face fixture present", face is not None, FACE)
    if face is None:
        return
    faces = detect_faces(face)
    ok_face = len(faces) >= 1 and faces[0].lm.shape == (478, 2)
    check("FaceLandmarker detects 478 pts", ok_face,
          f"{len(faces)} face(s)" + (f", {faces[0].lm.shape}" if faces else ""))
    if faces:
        f0 = faces[0]
        check("blendshapes + head-pose matrix present",
              len(f0.blend) > 10 and f0.matrix is not None,
              f"{len(f0.blend)} blendshapes")

    # --- every filter draws without error and changes the frame ---
    # mean-over-frame is dominated by background, so count notably-changed pixels.
    # heart_eyes is blendshape-reactive (only on a smile) → neutral face draws
    # nothing, which is CORRECT; for it we only require no error.
    det = sv.Detections.empty()
    set_current_eyewear(get_asset("sunglasses"), "test")   # so 'eyewear' has a product
    REACTIVE = {"heart_eyes"}
    for name in FACE_FILTERS:
        eng = OverlayEngine(); eng.add_native(name)
        img = face.copy()
        eng.run(img, det, None, 0, clean=face.copy())
        ov = list(eng.overlays.values())[0]
        changed_px = int((np.abs(img.astype(int) - face.astype(int)).max(2) > 18).sum())
        drew = changed_px > 200 or name in REACTIVE
        check(f"filter '{name}' draws", (not ov.error) and drew,
              (ov.error or (f"{changed_px}px" + (" (reactive)" if name in REACTIVE else ""))))

    # --- BIDI transcript de-dup (no chat bubbles twice during live voice) ---
    # Gemini Live streams the transcript as deltas AND re-sends the full aggregate
    # at turn end; LiveBridge._emit_event must drop that duplicate so the bubble
    # renders once. Feed synthetic events through it with a fake websocket.
    import asyncio
    from types import SimpleNamespace
    from occ.studio.live import LiveBridge

    class _FakeWS:
        def __init__(self): self.sent = []
        async def send_text(self, s): self.sent.append(json.loads(s))

    def _ev(out=None, inp=None, done=False):
        def _tr(t): return SimpleNamespace(text=t) if t is not None else None
        return SimpleNamespace(
            content=None, output_transcription=_tr(out), input_transcription=_tr(inp),
            get_function_calls=lambda: [], get_function_responses=lambda: [],
            interrupted=False, turn_complete=done)

    async def _drive(events):
        br = LiveBridge.__new__(LiveBridge)
        br._accum = {"user": "", "model": ""}
        ws = _FakeWS()
        for e in events:
            await br._emit_event(ws, e)
        return ws.sent

    def _run(coro):  # own loop; don't null the global (online() needs get_event_loop)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # deltas then a final aggregate that repeats the whole turn
    frames = _run(_drive([
        _ev(out="hello there"), _ev(out=" friend."),
        _ev(out="hello there friend."), _ev(done=True)]))
    model_txt = "".join(f["text"] for f in frames if f.get("type") == "transcript"
                        and f.get("role") == "model")
    check("BIDI model transcript not doubled", model_txt == "hello there friend.",
          repr(model_txt))

    # user side: deltas + aggregate must also collapse to one
    frames = _run(_drive([
        _ev(inp="put on"), _ev(inp=" the aviators"),
        _ev(inp="put on the aviators"), _ev(done=True)]))
    user_txt = "".join(f["text"] for f in frames if f.get("type") == "transcript"
                       and f.get("role") == "user")
    check("BIDI user transcript not doubled", user_txt == "put on the aviators",
          repr(user_txt))

    # --- OpenAI Realtime backend (the Creole-strong Live Voice) — offline logic ---
    from occ.studio import live_openai as LO
    sch = LO.tools_payload()
    check("OpenAI tool schemas built for every tool", len(sch) == len(T.ALL_TOOLS)
          and all(s.get("name") and s["parameters"]["type"] == "object" for s in sch),
          f"{len(sch)} schemas")
    # event translation + de-dup (deltas then the .done aggregate → ONE bubble)
    acc = {"user": "", "model": ""}
    fr = []
    for ev in [{"type": "response.output_audio_transcript.delta", "delta": "Bonjou"},
               {"type": "response.output_audio_transcript.delta", "delta": " zanmi"},
               {"type": "response.output_audio_transcript.done", "transcript": "Bonjou zanmi"},
               {"type": "response.output_audio.delta", "delta": "QUJD"},
               {"type": "conversation.item.input_audio_transcription.completed",
                "transcript": "mete linèt yo"},
               {"type": "response.done", "response": {"output": [{"type": "message"}]}}]:
        fr += LO.event_to_frames(ev, acc)
    mtxt = "".join(f["text"] for f in fr if f.get("type") == "transcript" and f["role"] == "model")
    utxt = "".join(f["text"] for f in fr if f.get("type") == "transcript" and f["role"] == "user")
    check("OpenAI model transcript not doubled", mtxt == "Bonjou zanmi", repr(mtxt))
    check("OpenAI Creole user transcript flows", utxt == "mete linèt yo", repr(utxt))
    check("OpenAI audio + turn_complete emitted",
          sum(1 for f in fr if f["type"] == "audio") == 1
          and sum(1 for f in fr if f["type"] == "turn_complete") == 1)
    # a tool-only response.done is intermediate → must NOT close the turn
    tool_only = LO.event_to_frames(
        {"type": "response.done", "response": {"output": [{"type": "function_call"}]}},
        {"user": "", "model": ""})
    check("OpenAI tool-only turn stays open (no premature turn_complete)",
          not any(f["type"] == "turn_complete" for f in tool_only))
    # 16k browser mic → 24k OpenAI input
    out = LO.resample_pcm16(bytes(1600 * 2), 16000, 24000)
    check("OpenAI 16k→24k resample", len(out) // 2 == 2400, f"{len(out)//2} samples")

    # --- multilingual narration TTS routing (gpt-4o-mini-tts ↔ Gemini) ---
    from occ.studio import genmedia as GM
    check("TTS provider: gpt-* → openai", GM.tts_provider("gpt-4o-mini-tts") == "openai")
    check("TTS provider: gemini → gemini",
          GM.tts_provider("gemini-2.5-flash-preview-tts") == "gemini")
    check("narrate exposes a model arg (Creole TTS)",
          "model" in inspect.signature(T.narrate).parameters)

    # --- gesture browsing logic (deterministic; no camera / mediapipe needed) ---
    from occ.studio.gestures import GestureBrowser, _HOLD_FRAMES
    from types import SimpleNamespace as NS
    def _hand(x, y=0.5):                       # 21 landmarks; only [8]=index tip matters
        return [NS(x=x, y=y, z=0) for _ in range(21)]
    gb = GestureBrowser()
    gb.active = True; gb.store = "eyewear"
    gb.items = [{"title": f"g{i}", "img": ""} for i in range(10)]
    gb._drive("Open_Palm", _hand(0.82), 1000, 1000)
    far_right = gb.idx
    gb._drive("Open_Palm", _hand(0.18), 1000, 1000)
    check("gesture: hand position scrubs the carousel", far_right == 9 and gb.idx == 0,
          f"right={far_right} left={gb.idx}")
    calls = []
    gb._select = lambda: calls.append("select")
    for _ in range(_HOLD_FRAMES + 3):          # fist must HOLD to fire, then latch
        gb._drive("Closed_Fist", _hand(0.5), 1000, 1000)
    check("gesture: fist holds then fires try-on exactly once", calls == ["select"], calls)
    gb._drive("Open_Palm", _hand(0.5), 1000, 1000)  # release
    gb._cool = 0; gb._switch_store = lambda: calls.append("switch")
    gb._drive("Victory", _hand(0.5), 1000, 1000)
    gb._cool = 0; gb._clear = lambda: calls.append("clear")
    gb._drive("Thumb_Down", _hand(0.5), 1000, 1000)
    check("gesture: V switches store, thumbs-down clears",
          calls == ["select", "switch", "clear"], calls)
    check("gesture_browse tool registered", "gesture_browse" in
          {f.__name__ for f in T.ALL_TOOLS})


# ----------------------------- ONLINE -----------------------------
def _get(path):
    return json.loads(urllib.request.urlopen(STUDIO_URL + path, timeout=8).read())


def _post(path, body):
    req = urllib.request.Request(STUDIO_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _server_up():
    try:
        urllib.request.urlopen(STUDIO_URL + "/", timeout=3)
        return True
    except Exception:
        return False


def online():
    print(f"\n[ONLINE] endpoints + live pipeline @ {STUDIO_URL}")
    import asyncio
    import websockets

    check("GET /garments", _get("/garments")["sponsor"]["name"] == "Mode Marco")
    ec = _get("/eyewear")
    check("GET /eyewear", ec["sponsor"]["name"] == "Ralba Optical" and len(ec["eyewear"]) > 300)

    # apply a face filter via the endpoint, confirm it's active
    r = _post("/filter", {"name": "ninja_mask"})
    ov = _get("/overlays").get("overlays", [])
    check("POST /filter ninja_mask → active", r.get("status") == "success"
          and any(o["name"] == "ninja_mask" for o in ov))

    # the real path: push a face → /ws/cam → pipeline → apply → /stream.mjpg changes
    face = cv2.imread(FACE)
    jpg = cv2.imencode(".jpg", cv2.resize(face, (640, 640)))[1].tobytes()

    def grab():
        req = urllib.request.urlopen(STUDIO_URL + "/stream.mjpg", timeout=8); buf = b""; t0 = time.time()
        while time.time() - t0 < 5:
            buf += req.read(8192); a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2)
            if a >= 0 and b >= 0:
                req.close(); return cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)

    async def live_path():
        async with websockets.connect(STUDIO_URL.replace("http", "ws") + "/ws/cam", max_size=None) as ws:
            async def pump():
                while True:
                    await ws.send(jpg); await asyncio.sleep(0.05)
            t = asyncio.create_task(pump()); await asyncio.sleep(1.5)
            _post("/filter", {"name": "sunglasses"}); await asyncio.sleep(1.2)
            sun = grab()
            _post("/filter", {"name": "clear"}); await asyncio.sleep(1.2)
            clean = grab()
            t.cancel()
            return sun, clean

        return None, None

    sun, clean = asyncio.get_event_loop().run_until_complete(live_path())
    if sun is not None and clean is not None:
        px = int((np.abs(sun.astype(int) - clean.astype(int)).max(2) > 18).sum())
        check("push→pipeline→/stream.mjpg: sunglasses visibly drawn", px > 200, f"{px}px")
    else:
        check("live pipeline frame grab", False, "no frame")

    # OpenAI Realtime LIVE turn — opt-in (bills OpenAI), so off by default.
    # STUDIO_TEST_OPENAI=1 .venv/bin/python test_studio_tryon.py
    if os.environ.get("STUDIO_TEST_OPENAI") == "1" and _get("/agent/mode").get("has_openai"):
        async def creole_turn():
            uri = STUDIO_URL.replace("http", "ws") + "/ws/live?backend=openai"
            async with websockets.connect(uri, max_size=None) as ws:
                await asyncio.sleep(0.3)
                await ws.send(json.dumps({"type": "text",
                    "text": "Pale Kreyòl. Di m yon ti bonjou kout epi mete yon mask ninja."}))
                model, audio, tools, t0 = [], 0, [], time.time()
                while time.time() - t0 < 35:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=35))
                    if m.get("type") == "transcript" and m["role"] == "model":
                        model.append(m["text"])
                    elif m.get("type") == "audio":
                        audio += 1
                    elif m.get("type") == "tool":
                        tools.append(m.get("data", {}).get("name"))
                    elif m.get("type") == "turn_complete" and audio:
                        break
                return "".join(model), audio, tools
        txt, audio, tools = asyncio.get_event_loop().run_until_complete(creole_turn())
        check("OpenAI Realtime live: spoke + tool-called", audio > 0 and "apply_face_filter" in tools,
              f"{audio} audio, tools={tools}, said={txt[:60]!r}")


def ui():
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("\n[UI] skipped (playwright not installed)")
        return
    print("\n[UI] deterministic frontend smoke (playwright)")
    errs = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_context(viewport={"width": 1500, "height": 950}).new_page()
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(STUDIO_URL + "/", wait_until="domcontentloaded"); pg.wait_for_timeout(1800)
        pg.click("#modeswitch button[data-m=voice]"); pg.wait_for_timeout(800)
        # pure canvas: panels are icon-driven now — the canvas stays clean until tapped
        check("filter panel hidden until its icon is tapped", not pg.is_visible("#filterbar"))
        pg.click("#vfilters"); pg.wait_for_timeout(300)
        check("face-filter panel opens on icon", pg.is_visible("#filterbar"))
        check("12 face-filter chips", pg.eval_on_selector_all("#filterbar button", "e=>e.length") == 12)
        pg.click("#vshop"); pg.wait_for_timeout(400)          # open the shop panel
        check("shop panel opens on icon", pg.is_visible("#tryonbar"))
        pg.click("#store-eyewear"); pg.wait_for_timeout(1200)
        check("store switch → Ralba sponsor", "Ralba Optical" in (pg.text_content("#sponsor") or ""))
        # one unified glasses try-on with a courteous fitting loader
        pg.evaluate("()=>vShowFitting('Tailoring your fit','fitting…')")
        pg.wait_for_timeout(150)
        check("fitting loader shows", "on" in (pg.get_attribute("#fitloader", "class") or ""))
        pg.evaluate("()=>vHideFitting()")
        pg.wait_for_timeout(150)
        check("fitting loader hides", "on" not in (pg.get_attribute("#fitloader", "class") or ""))
        # showCatalog renders agent results deterministically
        pg.evaluate("""()=>showCatalog({store:'apparel',matches:[
          {title:'POLO Navy',price:'150',img:'x',store:'apparel'},
          {title:'POLO Blue',price:'149',img:'y',store:'apparel'}]})""")
        pg.wait_for_timeout(300)
        check("showCatalog renders stylist results",
              pg.eval_on_selector_all("#garments .gcard", "e=>e.length") == 2)
        # collapsible panels → maximize collapses both side columns so the canvas grows
        cw0 = pg.eval_on_selector("#col-center", "e=>e.getBoundingClientRect().width")
        pg.click("#tgl-max"); pg.wait_for_timeout(450)
        cls = pg.get_attribute("#grid", "class") or ""
        cw1 = pg.eval_on_selector("#col-center", "e=>e.getBoundingClientRect().width")
        check("maximize collapses both panels", "cl" in cls.split() and "cr" in cls.split())
        check("canvas column grows when maximized", cw1 > cw0 + 100, f"{cw0:.0f}→{cw1:.0f}px")
        pg.click("#tgl-max"); pg.wait_for_timeout(450)  # restore
        check("un-maximize restores panels",
              (lambda c: "cl" not in c.split() and "cr" not in c.split())(pg.get_attribute("#grid", "class") or ""))
        # Live Voice backend selector (Gemini ↔ OpenAI Realtime for Creole)
        check("voice backend selector present",
              pg.eval_on_selector_all("#vbackend option", "e=>e.map(o=>o.value).join(',')")
              == "gemini,openai")
        check("gesture-browse button present", pg.is_visible("#vgest"))
        # Meet-style extras: fill-the-canvas toggle + self-view PiP
        pg.click("#fillbtn"); pg.wait_for_timeout(150)
        check("fill toggle fills the canvas", "fill" in (pg.get_attribute("#wrap", "class") or ""))
        pg.click("#fillbtn"); pg.wait_for_timeout(100)
        check("self-view PiP element present", pg.query_selector("#selfview") is not None)
        b.close()
    check("no uncaught JS errors", not errs, "; ".join(errs[:3]))


def main():
    print("=" * 60)
    print("  Vision Ninja — AR try-on regression test")
    print("=" * 60)
    offline()
    if _server_up():
        online()
        ui()
    else:
        print(f"\n[ONLINE/UI] skipped — studio not reachable at {STUDIO_URL}")
        print("  (start it: .venv/bin/python run.py studio, then re-run)")
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 60)
    print(f"  {'ALL PASS' if passed == total else 'FAILURES'} — {passed}/{total} checks")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
