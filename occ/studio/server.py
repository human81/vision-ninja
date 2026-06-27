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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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

    import secrets
    boot_id = secrets.token_hex(4)        # changes on every server (re)start

    @app.get("/stats")
    def stats():
        return JSONResponse({**pipe.stats(), "boot": boot_id})

    @app.post("/render")
    async def set_render_route(req: Request):
        b = await req.json()
        return pipe.set_render_flags(**{"draw_" + k: v for k, v in b.items()
                                        if k in ("boxes", "labels", "trails", "counts", "zones")})

    @app.post("/detection")
    async def set_detection_route(req: Request):
        """Toggle detection+tracking+occupancy (off by default once the camera is live)."""
        b = await req.json()
        return JSONResponse({"detect_on": pipe.set_detection(bool(b.get("on", True)))})

    @app.post("/gestures")
    async def gestures_route(req: Request):
        """Toggle hands-free gesture browsing (UI button; the agent can also do this)."""
        b = await req.json()
        on = b.get("on")
        return JSONResponse(pipe.set_gesture_browse(
            on=None if on is None else bool(on), store=b.get("store") or None))

    @app.get("/gesture")
    async def gesture_state():
        """Live browse state for the canvas side-rails (polled fast while active)."""
        return JSONResponse(pipe.gestures.state())

    @app.post("/gesture/select")
    async def gesture_select(req: Request):
        """Move the browse cursor to a clicked catalogue item (img) so the rails + the
        bottom strip stay in sync. Optional `store` switches stores if needed."""
        b = await req.json()
        return JSONResponse(pipe.set_gesture_current(b.get("img", ""), b.get("store") or None))

    @app.post("/gesture/step")
    async def gesture_step(req: Request):
        """Move the browse cursor by ±N (keyboard / UI Prev-Next)."""
        b = await req.json()
        return JSONResponse(pipe.step_gesture(int(b.get("d", 1))))

    @app.post("/filter")
    async def filter_route(req: Request):
        """Apply an AR face filter directly (UI chip; the agent can also do this)."""
        from .tools import apply_face_filter, FACE_FILTERS_LIST
        b = await req.json()
        name = (b.get("name") or "").strip()
        if name in ("clear", "none", ""):
            ctx_overlays = app.state.overlays
            return JSONResponse({"status": "success", "cleared": ctx_overlays.clear()})
        res = apply_face_filter(name)
        res["available"] = FACE_FILTERS_LIST
        return JSONResponse(res)

    @app.post("/jewelry")
    async def jewelry_route(req: Request):
        """Try jewelry on the live face (UI chip; the agent can also via try_jewelry).
        {kind: nose_ring|earrings|studs|septum|lip_ring|necklace, image?}."""
        from .tools import try_jewelry
        b = await req.json()
        return JSONResponse(try_jewelry(kind=b.get("kind", "nose_ring"),
                                        image=b.get("image", ""), label=b.get("label", "")))

    @app.get("/garments")
    def garments():
        """Default VTO catalog (sponsored). Falls back to an empty set."""
        p = Path(__file__).with_name("garments.json")
        try:
            return JSONResponse(json.loads(p.read_text()))
        except Exception:
            return JSONResponse({"sponsor": {}, "garments": []})

    @app.get("/eyewear")
    def eyewear():
        """Sponsored eyewear catalog (Ralba Optical) for live AR try-on."""
        p = Path(__file__).with_name("eyewear.json")
        try:
            return JSONResponse(json.loads(p.read_text()))
        except Exception:
            return JSONResponse({"sponsor": {}, "eyewear": []})

    @app.post("/eyewear/try")
    async def eyewear_try(req: Request):
        """Real-time AR try-on: warp a product frame onto the live face. The first fit
        of a product runs a slow Gemini render — do it OFF the event loop so the live
        feed / audio never freeze."""
        import asyncio as _aio
        import functools as _ft
        from .tools import try_eyewear
        b = await req.json()
        res = await _aio.get_event_loop().run_in_executor(None, _ft.partial(
            try_eyewear, image=b.get("image") or b.get("url") or "", label=b.get("label", "")))
        return JSONResponse(res)

    @app.post("/eyewear/tint")
    async def eyewear_tint(req: Request):
        """Recolour the live glasses' lenses (auto/clear/named tint + opacity)."""
        from .tools import set_lens_tint
        b = await req.json()
        return JSONResponse(set_lens_tint(tint=b.get("tint", "auto"),
                                          opacity=int(b.get("opacity", 0) or 0)))

    @app.post("/eyewear/prefetch")
    async def eyewear_prefetch(req: Request):
        """Warm the canonical render in the background (called on hover) so the
        actual try-on is instant. Returns immediately; generation runs in a thread."""
        import asyncio as _aio
        from .tools import prefetch_eyewear
        b = await req.json()
        img = b.get("image") or b.get("url") or ""
        if img:
            _aio.get_event_loop().run_in_executor(None, prefetch_eyewear, img)
        return JSONResponse({"status": "queued"})

    # ---------- Try-on Lab: flag bad try-ons, inspect the pipeline frame-by-frame ----------
    @app.post("/eyewear/flag")
    async def eyewear_flag(req: Request):
        """DEV: capture the current live try-on (face frame + every pipeline stage) into the
        bad-case dataset. Body: {note, title, brand, src?}. src falls back to the live pair."""
        import asyncio as _aio
        from . import eyewear_lab
        from .face_filters import _EYEWEAR
        b = await req.json()
        src = b.get("src") or _EYEWEAR.get("src")
        if not src:
            return JSONResponse({"status": "error", "error": "try a pair on first"}, status_code=400)
        clean = pipe.snapshot_clean(); vis = pipe.snapshot_vis()
        title = b.get("title") or _EYEWEAR.get("label") or ""
        has_3d = bool(b.get("ar") or b.get("has_3d"))
        try:
            fid = await _aio.get_event_loop().run_in_executor(
                None, lambda: eyewear_lab.save_flag(src, title=title, brand=b.get("brand", ""),
                                                    note=b.get("note", ""), clean_bgr=clean,
                                                    vis_bgr=vis, has_3d=has_3d))
        except Exception as e:
            return JSONResponse({"status": "error", "error": f"{type(e).__name__}: {e}"}, status_code=500)
        return JSONResponse({"status": "success", "id": fid,
                             "count": len(eyewear_lab.list_flags()), "lab": "/eyewear/lab"})

    @app.get("/eyewear/flags")
    def eyewear_flags():
        from . import eyewear_lab
        return JSONResponse({"flags": eyewear_lab.list_flags()})

    @app.post("/eyewear/flag/delete")
    async def eyewear_flag_delete(req: Request):
        from . import eyewear_lab
        b = await req.json()
        return JSONResponse({"deleted": eyewear_lab.delete_flag(b.get("id", ""))})

    @app.get("/eyewear/lab/img/{fid}/{fn}")
    def eyewear_lab_img(fid: str, fn: str):
        from . import eyewear_lab
        p = eyewear_lab.flag_image_path(fid, fn)
        if not p:
            return Response(status_code=404)
        return FileResponse(p, media_type="image/png")

    @app.get("/eyewear/lab/report/{fid}")
    def eyewear_lab_report(fid: str):
        """The coding-assistant REPORT.md for a flag, as raw markdown."""
        from . import eyewear_lab
        p = eyewear_lab.flag_image_path(fid, "REPORT.md")
        if not p:
            return Response(status_code=404)
        return Response(open(p).read(), media_type="text/markdown; charset=utf-8")

    @app.post("/eyewear/inspect")
    async def eyewear_inspect(req: Request):
        """CASCADE analysis: run the asset pipeline on a candidate image (upload/URL) WITHOUT
        saving and return every stage (base64) + stats — so you can see how a new image flows
        through before committing to it."""
        import asyncio as _aio
        from . import eyewear_lab
        b = await req.json()
        img = b.get("image") or b.get("url") or ""
        if not img:
            return JSONResponse({"status": "error", "error": "no image"}, status_code=400)
        res = await _aio.get_event_loop().run_in_executor(None, eyewear_lab.inspect_candidate, img)
        return JSONResponse({"status": "success", **res})

    @app.post("/eyewear/refix")
    async def eyewear_refix(req: Request):
        """FIX THE IMAGE: force a fresh Nano Banana Pro canonical render for a product and
        re-apply it live (requires an API key). Returns whether a new canonical was produced."""
        import asyncio as _aio
        from . import eyewear_lab
        from .face_filters import _EYEWEAR
        b = await req.json()
        src = b.get("src") or b.get("image") or _EYEWEAR.get("src")
        if not src:
            return JSONResponse({"status": "error", "error": "no source"}, status_code=400)
        ok = await _aio.get_event_loop().run_in_executor(None, eyewear_lab.regenerate_canonical, src)
        if ok and b.get("apply", True):
            from .tools import try_eyewear
            await _aio.get_event_loop().run_in_executor(
                None, lambda: try_eyewear(image=src, label=b.get("title", "")))
        return JSONResponse({"status": "success" if ok else "error",
                             "regenerated": ok,
                             "error": None if ok else "no API key / generation failed"})

    @app.post("/eyewear/lab/enhance")
    async def eyewear_lab_enhance(req: Request):
        """Re-image a pipeline STAGE with Nano Banana Pro (stage-tuned for optimal try-on) and
        cascade the downstream pipeline on the result. Body: {src, stage}."""
        import asyncio as _aio
        from . import eyewear_lab
        from .face_filters import _EYEWEAR
        b = await req.json()
        src = b.get("src") or _EYEWEAR.get("src")
        stage = b.get("stage", "")
        if not src or not stage:
            return JSONResponse({"status": "error", "error": "src + stage required"}, status_code=400)
        res = await _aio.get_event_loop().run_in_executor(
            None, lambda: eyewear_lab.enhance_stage(src, stage))
        if res.get("error"):
            return JSONResponse({"status": "error", **res}, status_code=502)
        return JSONResponse({"status": "success", **res})

    @app.post("/eyewear/lab/apply")
    async def eyewear_lab_apply(req: Request):
        """Apply an enhanced asset (data URI) as the live try-on, bypassing the canonical step."""
        import asyncio as _aio
        from . import eyewear_lab
        b = await req.json()
        uri = b.get("uri") or ""
        if not uri:
            return JSONResponse({"status": "error", "error": "no uri"}, status_code=400)
        ok = await _aio.get_event_loop().run_in_executor(
            None, lambda: eyewear_lab.apply_enhanced(uri, b.get("label", "enhanced")))
        return JSONResponse({"status": "success" if ok else "error", "applied": ok})

    @app.get("/eyewear/lab", response_class=HTMLResponse)
    def eyewear_lab_page():
        return (Path(__file__).with_name("eyewear_lab.html")).read_text()

    # ---------- Medical dashboard: MedGemma 4B (on-device, $0) ----------
    @app.get("/medical", response_class=HTMLResponse)
    def medical_page():
        return (Path(__file__).with_name("medgemma_ui.html")).read_text()

    @app.get("/medgemma/status")
    def medgemma_status():
        from . import medgemma as MG
        return JSONResponse(MG.status())                  # never forces a load

    @app.get("/medgemma/example/{key}")
    async def medgemma_example(key: str):
        """A public-domain example image for a sub-expertise (cached). For the dashboard."""
        import asyncio as _aio
        from . import medgemma as MG
        jpg = await _aio.get_event_loop().run_in_executor(None, MG.example_image, key)
        if not jpg:
            return Response(status_code=404)
        return Response(jpg, media_type="image/jpeg")

    @app.post("/medgemma/analyze")
    async def medgemma_analyze(req: Request):
        """Run a MedGemma 'power' on an uploaded/URL image or the live frame. Body:
        {image?, task, question?, which?}. The SAME engine the agent's medical_image tool uses."""
        import asyncio as _aio
        from . import medgemma as MG
        from .tools import _frame_jpg, _save_png
        b = await req.json()
        jpg, frame = _frame_jpg(b.get("which", "frame"), b.get("image", ""))
        if jpg is None:
            return JSONResponse({"status": "error", "error": "no image"}, status_code=400)
        if not await _aio.get_event_loop().run_in_executor(None, MG.available):
            return JSONResponse({"status": "error", "state": MG.status()["state"],
                "error": "MedGemma not ready — accept the license + install [med] (first use ~8GB)."},
                status_code=503)
        out = _save_png(frame, "medical")
        text = await _aio.get_event_loop().run_in_executor(
            None, lambda: MG.analyze(jpg, task=b.get("task", "findings"), question=b.get("question", "")))
        try:
            ctx().ledger.record("agent_brain", model="medgemma-4b-it",
                                input_tokens=300, output_tokens=140, label="medical")
        except Exception:
            pass
        return JSONResponse({"status": "success", "text": text, "image": "/download/" + Path(out).name,
                             "disclaimer": "Decision-support only — not a diagnosis. Consult a clinician."})

    @app.post("/tryon")
    async def tryon_route(req: Request):
        """Garment virtual try-on on the live frame (you). garment = URL / data URI."""
        from .tools import virtual_try_on
        b = await req.json()
        garment = b.get("garment") or b.get("url") or b.get("image") or ""
        if not garment:
            return JSONResponse({"status": "error", "error": "no garment"}, status_code=400)
        import asyncio as _aio
        import functools as _ft
        res = await _aio.get_event_loop().run_in_executor(None, _ft.partial(
            virtual_try_on, garment, which="frame"))      # off the event loop (~8s Gemini)
        out = res.get("output", "")
        if out:
            res["url"] = "/download/" + Path(out).name
        return JSONResponse(res)

    # ---------- agentic checkout: Stripe Checkout (test mode) ----------
    @app.post("/checkout")
    async def checkout_route(req: Request):
        """Create a Stripe Checkout Session for the tried-on product (or {query}).
        Returns the hosted pay URL — the shopper enters card + shipping on Stripe."""
        from .tools import checkout as _checkout
        import asyncio as _aio
        import functools as _ft
        b = await req.json()
        res = await _aio.get_event_loop().run_in_executor(None, _ft.partial(
            _checkout, query=b.get("query", ""), quantity=int(b.get("quantity", 1) or 1)))
        return JSONResponse(res, status_code=200 if res.get("status") == "success" else 400)

    @app.get("/checkout/orders")
    def checkout_orders():
        from . import commerce
        return JSONResponse({"orders": commerce.list_orders(), "ready": commerce.ready()[0]})

    @app.post("/stripe/webhook")
    async def stripe_webhook(req: Request):
        """Stripe → us: mark the order paid on checkout.session.completed."""
        from . import commerce
        payload = await req.body()
        sig = req.headers.get("stripe-signature", "")
        secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        try:
            import stripe
            if secret:
                event = stripe.Webhook.construct_event(payload, sig, secret)
            else:                                   # no secret configured → trust body (dev only)
                event = json.loads(payload)
        except Exception as e:
            return JSONResponse({"error": f"bad webhook: {e}"}, status_code=400)
        etype = event["type"] if isinstance(event, dict) else event.type
        obj = (event["data"]["object"] if isinstance(event, dict) else event.data.object)
        if etype == "checkout.session.completed":
            ship = (obj.get("shipping_details") or obj.get("customer_details") or {}) \
                if isinstance(obj, dict) else {}
            commerce.mark_paid(obj["id"] if isinstance(obj, dict) else obj.id, shipping=ship)
        return JSONResponse({"received": True})

    @app.get("/checkout/return", response_class=HTMLResponse)
    def checkout_return(session_id: str = "", status: str = ""):
        """Landing page Stripe redirects to after pay/cancel; confirms + records the order."""
        from . import commerce
        paid, item, amount, ccy = False, "", 0, ""
        if status == "cancel":
            msg = "Checkout cancelled — nothing was charged."
        elif session_id:
            try:
                import stripe
                stripe.api_key = commerce.api_key()
                s = stripe.checkout.Session.retrieve(session_id)
                paid = s.get("payment_status") == "paid"
                amount, ccy = (s.get("amount_total") or 0), (s.get("currency") or "")
                ship = s.get("shipping_details") or s.get("customer_details") or {}
                if paid:
                    commerce.mark_paid(session_id, shipping=ship)
                rec = next((o for o in commerce.list_orders() if o["session_id"] == session_id), {})
                item = "; ".join(i.get("title", "") for i in rec.get("items", [])) or "your order"
                msg = (f"✅ Payment received — {item}." if paid
                       else "Payment not completed yet.")
            except Exception as e:
                msg = f"Could not confirm the order: {e}"
        else:
            msg = "No checkout session."
        amt = f"{amount/100:.2f} {ccy.upper()}" if amount else ""
        return ("<!doctype html><meta charset=utf-8><title>Order</title>"
                "<style>body{font:16px/1.6 system-ui;background:#0b0b0d;color:#eee;"
                "display:grid;place-items:center;height:100vh;margin:0;text-align:center}"
                ".c{max-width:460px;padding:32px;background:#16161a;border-radius:16px}"
                "a{color:#7cc4ff}</style><div class=c><h2>🥷 Vision Ninja Studio</h2>"
                f"<p>{msg}</p>{('<p style=\"opacity:.7\">'+amt+'</p>') if amt else ''}"
                f"{'<p style=\"opacity:.6;font-size:13px\">TEST mode — no real charge.</p>' if commerce.test_mode() else ''}"
                "<p><a href=\"/\">← Back to the studio</a></p></div>")

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

    @app.post("/overlays/clear")
    def clear_overlays_route():
        n = overlays.clear()
        for o in list(brain.d.get("overlays", [])):
            brain.forget_overlay(o["name"])
        return {"removed": n}

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
        from .offline import offline as _offline
        d["offline"] = _offline()      # UI locks Live Voice to the local backend when True
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
        from .live import LIVE_MODEL
        from .live_openai import OPENAI_REALTIME_MODEL
        return {"mode": agent.mode(), "has_key": _HAS_KEY,
                "has_openai": bool(os.environ.get("OPENAI_API_KEY")),
                "live_models": {"gemini": LIVE_MODEL, "openai": OPENAI_REALTIME_MODEL},
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

    # ---------- BIDI live: camera/screen frame ingest ----------
    @app.websocket("/ws/cam")
    async def ws_cam(ws: WebSocket):
        """Browser pushes webcam/screen JPEG frames (binary) here; they become the
        live source so the pipeline runs detection + the agent's overlays on them
        and republishes to /stream.mjpg (overlays on *you*)."""
        await ws.accept()
        pipe.start_camera()
        try:
            while True:
                data = await ws.receive_bytes()
                if data:
                    pipe.push_camera_frame(data)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            pipe.stop_camera()

    # ---------- BIDI live: Gemini Live voice + tool-calling ----------
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        backend = (ws.query_params.get("backend") or "gemini").lower()
        from .offline import offline as _offline
        if _offline() and backend != "local":
            # Airplane mode: cloud backends (Gemini/OpenAI) need DNS → gaierror offline.
            # Force the on-device backend regardless of what the (cached) page requested.
            await ws.send_text(json.dumps({"type": "log", "data": {"text":
                "offline mode → using 🦙 Local voice (cloud BIDI needs internet)"}}))
            backend = "local"
        if backend == "local":
            from .voice_local import LocalVoiceBridge       # $0 — whisper + local Gemma, no key
            bridge = LocalVoiceBridge(settings)
        elif backend == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                await ws.send_text(json.dumps({"type": "error",
                    "message": "OpenAI Realtime needs OPENAI_API_KEY"}))
                await ws.close()
                return
            from .live_openai import OpenAIRealtimeBridge
            bridge = OpenAIRealtimeBridge(settings)
        else:
            if not _HAS_KEY:
                await ws.send_text(json.dumps({"type": "error",
                    "message": "live voice needs a Gemini API key (GEMINI_API_KEY)"}))
                await ws.close()
                return
            from .live import LiveBridge
            bridge = LiveBridge(settings)
        try:
            await bridge.run(ws)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await ws.send_text(json.dumps({"type": "error",
                    "message": f"{type(e).__name__}: {e}"}))
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    return app


app = create_studio_app()
