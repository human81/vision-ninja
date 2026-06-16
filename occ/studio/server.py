"""Studio server — FastAPI hosting the live CV view + the agent's NDJSON stream.

  uvicorn occ.studio.server:app --port 8011
  OCC_SOURCE=assets/videos/market-square.mp4 uvicorn occ.studio.server:app --port 8011

Routes mirror Momentum's seam: a streaming /agent/chat (NDJSON) the browser
applies as UI mutations, plus REST panels (/usage, /brain, /graph, /overlays,
/settings, /scene) the agent refreshes by emitting `refresh` frames.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Load a gitignored .env (GEMINI_API_KEY etc.) FIRST — before importing .agent,
# which reads the key at import time. Mirrors Momentum's load_dotenv()-first rule.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response,
                               StreamingResponse)

from ..config import Config
from . import STUDIO_DIR, neurons
from .agent import StudioAgent, _HAS_KEY
from .brain import VisionBrain
from .ledger import Ledger
from .library import MediaLibrary
from .overlays import OverlayEngine
from .pipeline import StudioPipeline
from .runtime import StudioContext, set_context
from .settings import StudioSettings

_UI = Path(__file__).with_name("studio_ui.html")
MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
TRACKERS = ["bytetrack", "botsort", "ocsort", "sort"]


def create_studio_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config.load(overrides=[
        f"source.uri={os.environ.get('OCC_SOURCE', 'assets/videos/vehicles-2.mp4')}",
        "source.loop=true"])

    overlays = OverlayEngine()
    ledger = Ledger()
    settings = StudioSettings.load()
    brain = VisionBrain()
    library = MediaLibrary()
    sources = MediaLibrary(f"{STUDIO_DIR}/sources.json")     # live RTSP/YT-live + examples
    # Clean fake/unplayable test stubs, then seed REAL playable local examples.
    for it in list(sources.items):
        if any(h in str(it.get("path", "")) for h in
               ("rtsp://demo/", "rtsp://x/", "demo:demo@ipvmdemo", "demo/cam", "://x/y")):
            sources.remove(it["id"])
    _have = {Path(i["path"]).name for i in sources.items}
    for _v in sorted(Path("assets/videos").glob("*.mp4")):
        if _v.name not in _have:
            sources.add("file", str(_v), caption=_v.stem.replace("-", " "),
                        tags=["example", "file"])
    pipe = StudioPipeline(cfg, overlays=overlays, ledger=ledger, brain=brain,
                          settings=settings)
    set_context(StudioContext(pipe=pipe, overlays=overlays, ledger=ledger,
                              brain=brain, settings=settings, library=library,
                              sources=sources))
    agent = StudioAgent(settings)
    from . import genmedia
    genmedia.prewarm()                 # create the genai client on the main thread
    pipe.start()

    video_files = sorted(str(p) for p in Path("assets/videos").glob("*.mp4"))
    app = FastAPI(title="Vision Ninja Studio")
    app.state.pipe = pipe
    app.state.agent = agent
    app.state.ledger = ledger
    app.state.overlays = overlays
    app.state.brain = brain
    app.state.settings = settings

    # ---------- page + live view ----------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return _UI.read_text()

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while not pipe._stop.is_set():
                j = pipe.latest_jpeg()
                if j:
                    yield boundary + j + b"\r\n"
                time.sleep(0.03)
        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stats")
    def stats():
        return JSONResponse(pipe.stats())

    @app.get("/scene")
    def scene():
        return JSONResponse(pipe.scene())

    @app.get("/emit.pb")
    def emit_pb():
        res = pipe.emit_proto()
        if res.get("status") != "success":
            return JSONResponse(res, status_code=503)
        p = res["output"]
        return FileResponse(p, media_type="application/octet-stream",
                            filename=Path(p).name)

    # ---------- config + annotations ----------
    @app.get("/options")
    def options():
        return {"sources": video_files, "models": MODELS, "trackers": TRACKERS,
                "detectors": ["yolo", "rfdetr", "owlv2"]}

    @app.get("/config")
    def get_config():
        return JSONResponse(pipe.current_config())

    @app.post("/config")
    async def set_config(req: Request):
        pipe.reconfigure(await req.json())
        return JSONResponse(pipe.current_config())

    @app.get("/annotations")
    def get_ann():
        return JSONResponse(pipe.annotation_dicts())

    @app.post("/annotations")
    async def set_ann(req: Request):
        pipe.set_annotations(await req.json())
        return {"ok": True, "count": len(pipe.annotation_dicts())}

    @app.post("/annotations/save")
    def save_ann():
        return {"saved": pipe.save_annotations()}

    # ---------- overlays ----------
    @app.get("/overlays")
    def get_overlays():
        from .overlays import BUILTINS
        return {"overlays": overlays.list(), "presets": list(BUILTINS.keys())}

    @app.post("/overlays/toggle")
    async def toggle_overlay(req: Request):
        b = await req.json()
        from .tools import toggle_overlay as tt
        return tt(b.get("name", ""), bool(b.get("on", True)))

    @app.post("/overlays/remove")
    async def remove_overlay(req: Request):
        b = await req.json()
        return {"ok": overlays.remove(b.get("name", ""))}

    # ---------- meter / brain / graph / settings ----------
    @app.get("/usage")
    def usage():
        return JSONResponse(ledger.summary())

    @app.get("/brain")
    def get_brain():
        return JSONResponse(brain.to_dict())

    @app.get("/graph")
    def graph():
        g = neurons.graph_for_ui()
        g["activity"] = ledger.summary()["by_node"]
        return JSONResponse(g)

    @app.get("/settings")
    def get_settings():
        d = settings.to_dict()
        d["agent_mode"] = agent.mode()
        d["has_key"] = _HAS_KEY
        return JSONResponse(d)

    @app.post("/settings")
    async def post_settings(req: Request):
        settings.update(await req.json())
        ledger.set_budget(settings.points_budget)
        return JSONResponse(settings.to_dict())

    # ---------- media library + live sources ----------
    @app.get("/library")
    def library_list(q: str = ""):
        return {"items": (library.search(q) if q else library.list())}

    @app.post("/library/remove")
    async def library_remove(req: Request):
        return {"ok": library.remove((await req.json()).get("id"))}

    @app.post("/library/clean")
    def library_clean():
        return {"removed": library.prune()}

    @app.get("/thumb")
    def thumb(name: str):
        name = Path(name).name
        src = next((str(Path(b) / name) for b in
                    ("out/studio/exports", "out/recordings", "out/studio/cache")
                    if (Path(b) / name).exists()), None)
        if not src:
            return JSONResponse({"error": "not found"}, status_code=404)
        ext = Path(src).suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".gif"):
            return FileResponse(src)
        tdir = Path("out/studio/thumbs"); tdir.mkdir(parents=True, exist_ok=True)
        out = tdir / (Path(name).stem + (".png" if ext in (".wav", ".mp3", ".m4a",
                                                           ".aac", ".ogg") else ".jpg"))
        if not out.exists():
            import subprocess
            if ext in (".mp4", ".mov", ".webm"):
                args = ["ffmpeg", "-y", "-ss", "1", "-i", src, "-frames:v", "1",
                        "-vf", "scale=260:-1", str(out)]
            else:                                       # audio -> waveform image
                args = ["ffmpeg", "-y", "-i", src, "-filter_complex",
                        "showwavespic=s=260x70:colors=#36d7e0", str(out)]
            try:
                subprocess.run(args, capture_output=True, timeout=30)
            except Exception:
                pass
        return FileResponse(str(out)) if out.exists() else \
            JSONResponse({"error": "thumb failed"}, status_code=500)

    @app.get("/sources")
    def sources_list():
        return {"items": sources.list()}

    @app.post("/sources")
    async def sources_add(req: Request):
        from .tools import save_source
        b = await req.json()
        return save_source(b.get("url", ""), b.get("name", ""))

    @app.post("/sources/use")
    async def sources_use(req: Request):
        from .tools import use_source
        b = await req.json()
        return use_source(int(b.get("id", 0) or 0), b.get("query", ""))

    @app.post("/sources/remove")
    async def sources_remove(req: Request):
        return {"ok": sources.remove((await req.json()).get("id"))}

    # ---------- NLE timeline render + creative-studio narrate ----------
    @app.post("/timeline/render")
    async def timeline_render(req: Request):
        from . import nle
        res = nle.render(await req.json())
        if res.get("status") == "success" and res.get("output"):
            library.add("video", res["output"], caption="timeline edit",
                        tags=["edit", "nle"])
        return JSONResponse(res)

    @app.post("/timeline/narrate")
    async def timeline_narrate(req: Request):
        from .tools import narrate
        b = await req.json()
        return JSONResponse(narrate(b.get("text", ""), b.get("voice", "Puck")))

    @app.post("/timeline/music")
    async def timeline_music(req: Request):
        from .tools import generate_music
        return JSONResponse(generate_music((await req.json()).get("prompt", "")))

    @app.post("/timeline/image")
    async def timeline_image(req: Request):
        from . import genmedia
        prompt = (await req.json()).get("prompt", "")
        if not genmedia.has_key():
            return JSONResponse({"status": "error", "error": "needs a Gemini key"})
        img, txt, _ = genmedia.image(prompt)
        if not img:
            return JSONResponse({"status": "error", "error": txt or "no image"})
        os.makedirs("out/studio/exports", exist_ok=True)
        outp = f"out/studio/exports/gen_{time.strftime('%H%M%S')}.png"
        open(outp, "wb").write(img)
        library.add("image", outp, caption=prompt[:50], tags=["generated", "keyframe"])
        return JSONResponse({"status": "success", "output": outp})

    # ---------- recording + snapshot ----------
    @app.post("/record/start")
    def rec_start():
        return {"recording": True, "file": pipe.start_recording()}

    @app.post("/record/stop")
    def rec_stop():
        name, n = pipe.stop_recording()
        return {"recording": False, "file": name, "frames": n}

    @app.get("/snapshot.png")
    def snapshot():
        vis = pipe.snapshot_vis()
        if vis is None:
            return JSONResponse({"error": "no frame"}, status_code=503)
        ok, buf = cv2.imencode(".png", vis)
        return Response(content=buf.tobytes(), media_type="image/png")

    @app.get("/download/{name}")
    def download(name: str):
        for base in ("out/studio/exports", "out/recordings"):
            p = Path(base) / Path(name).name
            if p.exists():
                return FileResponse(str(p), filename=p.name)
        return JSONResponse({"error": "not found"}, status_code=404)

    @app.get("/file")
    def file(path: str):
        p = Path(path)
        if p.exists() and "out/studio" in str(p.resolve()):
            return FileResponse(str(p), filename=p.name)
        return JSONResponse({"error": "not found"}, status_code=404)

    # ---------- the agent stream ----------
    @app.get("/agent/mode")
    def agent_mode():
        return {"mode": agent.mode(), "has_key": _HAS_KEY,
                "simulation": settings.simulation}

    @app.post("/agent/chat")
    async def agent_chat(req: Request):
        body = await req.json()
        message = (body.get("message") or "").strip()

        async def gen():
            if not message:
                yield json.dumps({"type": "error",
                                  "data": {"text": "empty message"}}) + "\n"
                return
            try:
                async for frame in agent.stream(message):
                    yield json.dumps(frame) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error",
                                  "data": {"text": f"{type(e).__name__}: {e}"}}) + "\n"
                yield json.dumps({"type": "done", "data": {}}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app


app = create_studio_app()
