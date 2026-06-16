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
    sources = MediaLibrary(f"{STUDIO_DIR}/sources.json")     # live RTSP/YT-live
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
