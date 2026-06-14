"""Browser UI over the pipeline — so it can be fully driven (and Playwright-tested)
without an OpenCV window.

  GET  /                 control page (MJPEG view + canvas zone editor + live stats)
  GET  /stream.mjpg      annotated frames as multipart MJPEG
  GET  /stats            live counts JSON (full-frame, per-line +/- , per-zone, tracks)
  POST /annotations      replace zones/lines (normalized verts); geometry rebuilds live
  POST /start /stop      control the worker

Run:  OCC_SOURCE=assets/videos/vehicles-2.mp4 uvicorn occ.web:app --port 8000
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
import supervision as sv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .annotations import Annotation, AnnotationSet
from .config import Config
from .detectors import build_detector
from .geometry import GeometryEngine
from .render import Renderer
from .sources import open_source
from .tracking import Tracker


class WebPipeline:
    """Runs source→detect→track→geometry→render in a thread; serves latest frame+stats."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.annotations = AnnotationSet()
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stats: dict = {"full_frame": {}, "lines": {}, "zones": {}, "tracks": 0}
        self._ann_version = 0
        self.ann_path: str | None = None
        self._run = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_annotations(self, items: list[dict]):
        anns = [Annotation(id=i.get("id") or f"a{n}", type=i["type"],
                           vertices=[tuple(v) for v in i["vertices"]],
                           display_name=i.get("display_name", "")) for n, i in enumerate(items)]
        with self._lock:
            self.annotations = AnnotationSet(annotations=anns)
            self._ann_version += 1

    def load_annotations(self, path: str):
        s = AnnotationSet.load(path)
        with self._lock:
            self.annotations = s
            self._ann_version += 1

    def annotation_dicts(self) -> list[dict]:
        with self._lock:
            return [{"id": a.id, "type": a.type, "display_name": a.display_name,
                     "vertices": [[x, y] for x, y in a.vertices]}
                    for a in self.annotations.annotations]

    def save_annotations(self, path: str | None = None) -> str:
        p = path or self.ann_path or "configs/annotations.json"
        with self._lock:
            self.annotations.source = str(self.cfg.get("source.uri", ""))
            self.annotations.save(p)
        return p

    def start(self):
        self._run.set()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._run.clear()

    def shutdown(self):
        self._stop.set()

    def _loop(self):
        src = open_source(self.cfg)
        det = build_detector(self.cfg)
        tcfg = self.cfg.section("tracker")
        trk = Tracker(tcfg.get("algorithm", "bytetrack"),
                      {**tcfg.get("params", {}), "frame_rate": round(src.fps)})
        renderer = Renderer(self.cfg)
        fps = src.fps or 30.0
        n = 0
        geo_engine = GeometryEngine(self.annotations)
        seen_version = self._ann_version
        for frame in src.frames():
            if self._stop.is_set():
                break
            if not self._run.is_set():
                time.sleep(0.05)
                continue
            with self._lock:
                ann = self.annotations
                version = self._ann_version
            # rebuild geometry ONLY when annotations change — otherwise the per-track
            # line-crossing side history must persist across frames.
            if version != seen_version:
                geo_engine = GeometryEngine(ann)
                seen_version = version
            tracked = trk.update(det.detect(frame), frame)
            geo = geo_engine.update(tracked, frame.shape[1], frame.shape[0], n / fps)
            vis = renderer.draw(frame, tracked)
            vis = renderer.draw_annotations(vis, ann, geo)
            vis = renderer.draw_counts(vis, geo)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
            stats = {
                "full_frame": dict(geo.full_frame),
                "lines": {lid: {"positive": sum(d["positive"].values()),
                                "negative": sum(d["negative"].values())}
                          for lid, d in geo.line_counts.items()},
                "zones": {zid: sum(c.values()) for zid, c in geo.zone_counts.items()},
                "tracks": 0 if tracked.tracker_id is None else len(tracked),
            }
            with self._lock:
                if ok:
                    self._jpeg = buf.tobytes()
                self._stats = stats
            n += 1

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Occupancy</title>
<style>body{font-family:system-ui;margin:16px}#wrap{position:relative;display:inline-block}
#cv{position:absolute;left:0;top:0;cursor:crosshair}#stats{font-family:monospace;white-space:pre}
button{margin-right:8px;padding:6px 10px}</style></head><body>
<h3>Occupancy / Traffic — browser control</h3>
<div><button id=zone>Add zone</button><button id=line>Add line</button>
<button id=finish>Finish shape</button><button id=start>Start</button>
<button id=stop>Stop</button><button id=save>Save</button><span id=mode>running</span></div>
<div id=wrap><img id=img src=/stream.mjpg width=640><canvas id=cv width=640 height=360></canvas></div>
<pre id=stats>stats…</pre>
<script>
let img=document.getElementById('img'),cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let mode='idle',draft=[],shapes=[];
function resize(){cv.width=img.clientWidth;cv.height=img.clientHeight;draw();}
img.onload=resize;window.onload=resize;
document.getElementById('zone').onclick=()=>{mode='zone';draft=[];setmode()};
document.getElementById('line').onclick=()=>{mode='line';draft=[];setmode()};
function setmode(){document.getElementById('mode').textContent='mode: '+mode;}
cv.onclick=e=>{let r=cv.getBoundingClientRect();
 draft.push([(e.clientX-r.left)/cv.width,(e.clientY-r.top)/cv.height]);
 if(mode==='line'&&draft.length===2)finish();draw();};
document.getElementById('finish').onclick=finish;
function finish(){if(mode==='zone'&&draft.length>=3||mode==='line'&&draft.length===2){
 shapes.push({type:mode==='zone'?'active_zone':'crossing_line',id:mode+shapes.length,vertices:draft});
 fetch('/annotations',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(shapes)});}
 draft=[];mode='idle';setmode();draw();}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);ctx.lineWidth=2;
 for(let s of shapes){ctx.strokeStyle=s.type==='active_zone'?'#ffb400':'#00ff00';ctx.beginPath();
  s.vertices.forEach((v,i)=>{let x=v[0]*cv.width,y=v[1]*cv.height;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  if(s.type==='active_zone')ctx.closePath();ctx.stroke();}
 ctx.strokeStyle='#ff00ff';ctx.beginPath();
 draft.forEach((v,i)=>{let x=v[0]*cv.width,y=v[1]*cv.height;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
document.getElementById('start').onclick=()=>fetch('/start',{method:'POST'});
document.getElementById('stop').onclick=()=>fetch('/stop',{method:'POST'});
document.getElementById('save').onclick=async()=>{let r=await(await fetch('/save',{method:'POST'})).json();
 document.getElementById('mode').textContent='saved '+r.count+' → '+r.saved;};
async function poll(){try{let s=await(await fetch('/stats')).json();
 document.getElementById('stats').textContent=JSON.stringify(s,null,2);}catch(e){}
 setTimeout(poll,500);}poll();
async function loadShapes(){try{let a=await(await fetch('/annotations')).json();
 shapes=a.map(s=>({type:s.type,id:s.id,vertices:s.vertices}));draw();}catch(e){}}
loadShapes();
</script></body></html>"""


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config.load(overrides=[
        f"source.uri={os.environ.get('OCC_SOURCE', 'assets/videos/vehicles-2.mp4')}",
        "source.loop=true"])
    pipe = WebPipeline(cfg)
    # pre-load zones/lines so counts appear immediately (env > config > default file)
    ann_path = (os.environ.get("OCC_ANNOTATIONS") or cfg.get("annotations")
                or "configs/highway_lines.json")
    pipe.ann_path = ann_path
    if ann_path and Path(ann_path).exists():
        pipe.load_annotations(ann_path)
    pipe.start()                 # show the live annotated stream by default
    app = FastAPI()
    app.state.pipe = pipe

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _PAGE

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while not pipe._stop.is_set():
                j = pipe.latest_jpeg()
                if j:
                    yield boundary + j + b"\r\n"
                time.sleep(0.05)
        return StreamingResponse(gen(),
            media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stats")
    def stats():
        return JSONResponse(pipe.stats())

    @app.get("/annotations")
    def get_annotations():
        return JSONResponse(pipe.annotation_dicts())

    @app.post("/annotations")
    async def annotations(req: Request):
        pipe.set_annotations(await req.json())
        return {"ok": True, "count": len(pipe.annotations.annotations)}

    @app.post("/start")
    def start():
        pipe.start()
        return {"running": True}

    @app.post("/stop")
    def stop():
        pipe.stop()
        return {"running": False}

    @app.post("/save")
    def save():
        path = pipe.save_annotations()
        return {"saved": path, "count": len(pipe.annotation_dicts())}

    return app


app = create_app()
