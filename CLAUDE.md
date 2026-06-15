# CLAUDE.md

Guidance for working in this repo. Local, real-time occupancy & traffic analytics on
Apple Silicon (M5 Pro / MPS): capture → detect → track → spatial math → emit the
Vertex AI Vision `OccupancyCountingPredictionResult` protobuf + annotated overlays +
traffic CSV. Runs $0-marginal locally. See `PLAN.md` for the phased build and `README.md`
for usage.

## Environment (read first)

- **Always use `.venv/bin/python`**, never bare `python`. System Python is 3.14, which has
  no torch/ultralytics/transformers wheels. The venv is CPython **3.12** (created with `uv`).
- Setup from scratch:
  ```bash
  uv venv --python 3.12 .venv
  uv pip install -e ".[cv]"          # core; extras: [rfdetr] [vlm] [web] [build]
  bash assets/fetch_assets.sh        # 6 test clips → assets/videos/ (gitignored)
  ```
- `uv pip install -e ".[web]" && .venv/bin/playwright install chromium` for the browser UI + e2e.
- Models auto-download on first use (YOLO `.pt`, HF checkpoints). Weights/clips/`out/` are gitignored.

## Commands

```bash
# real-time (OpenCV window by default; --no-show for headless)
.venv/bin/python run.py run --source assets/videos/vehicles-2.mp4
.venv/bin/python run.py run --config configs/highway_example.yaml --emit-proto out/run.pb \
    --analytics-csv out/counts.csv
# override ANY config field: --set detector.model=yolo11s.pt --set runtime.detect_every=3
# shortcuts: --detector {yolo,rfdetr,owlv2} --tracker {bytetrack,botsort,ocsort,sort} --detect-every N

.venv/bin/python run.py edit --source ... --annotations configs/cam.json   # draw zones/lines (GUI)
.venv/bin/python run.py ground --source ... --prompt "forklifts" --suggest-zone bay  # OWLv2 open-vocab

# browser dashboard (live-reconfig, draw/save, snapshot, record, timeline chart)
OCC_SOURCE=assets/videos/vehicles-2.mp4 .venv/bin/uvicorn occ.web:app --port 8001

# tests
.venv/bin/python fishfood.py            # full dogfood, all clips, levels 1–5 (the green gate; expect 27/27)
.venv/bin/python test_proto_roundtrip.py test_phase2.py test_phase3.py test_phase4.py test_e2e_web.py
```

After any change, run `fishfood.py` and expect **ALL PASS — 27/27**. It also writes
`out/fishfood/RUNBOOK.md` (a command per check) and a gallery of annotated frames.

## Architecture (`occ/`)

| Module | Role |
|---|---|
| `config.py` | One YAML drives all. `Config.load(overrides=["a.b=v"])`; dotted `--set`; `source_overrides` (substring-keyed, CLI wins); class-group + COCO-name resolution. |
| `sources.py` | `FileSource` + newest-frame-wins `StreamSource` (RTSP/cam, auto-reconnect). |
| `detectors/` | `build_detector(cfg)`; `yolo` (Ultralytics, MPS), `rfdetr` (HF transformers, MPS), `owlv2` (open-vocab, MPS). **All return `supervision.Detections`** so the tracker is detector-agnostic. |
| `tracking.py` | Factory over **roboflow/trackers**; signature-filters params (config is a permissive superset). ByteTrack/BoT-SORT/OC-SORT/SORT. |
| `geometry.py` | `GeometryEngine`: zones (point-in-polygon), line crossing (right-hand-rule sign flip), dwell, full-frame counts. Uses each track's **bottom-center anchor**. |
| `emit.py` | `build_result()` → `OccupancyCountingPredictionResult`; `ResultWriter` (varint length-delimited `.pb` stream). |
| `analytics.py` | Per-interval CSV (line crossings as deltas; zone/full-frame avg+peak) + JSON summary. |
| `speed.py` | Homography speed (4-pt ground calibration) with discontinuity rejection. |
| `render.py` | Overlays; **scales with resolution** (legible at 1080p & 4K), text on backgrounds. |
| `editor.py` | Interactive OpenCV zone/line editor (mouse + keyboard + on-screen help). |
| `grounding/` | Open-vocab VLM grounders: `owlv2` (Mac/MPS default), `locate_anything`/`molmo2` (GCP/Linux). `run.py ground`. |
| `web.py` + `web_ui.html` | FastAPI dashboard: live-reconfig, MJPEG, canvas draw/save, stats, timeline chart, snapshot/record. Live-reconfigurable via a dirty-flag rebuild loop. |
| `proto/` | Vendored, wire-identical `OccupancyCountingPredictionResult` (`visionai_annotations.proto` → `_pb2.py`). Regenerate: `python -m grpc_tools.protoc -Iproto --python_out=proto proto/visionai_annotations.proto`. |

Cost levers (💰 in `configs/default.yaml`): model tier, `detect_every`, `imgsz`, `half`, `max_long_side`.

## Conventions & gotchas (learned the hard way)

- **New detector** = implement `detect(frame) -> sv.Detections` and register in `detectors/__init__.py`.
  YOLO filters classes by id (in-engine); RF-DETR/OWLv2 filter by **name** (their label ids differ from YOLO COCO-80).
- **`detect_every>1` must HOLD the last tracked result on skip frames** — feeding empty detections makes
  ByteTrack drop every track (tracks=0). Done in `pipeline.py` and `web.py`; preserve this.
- **Only confirmed tracks (`tracker_id >= 0`) are counted.** ByteTrack confirms after
  `minimum_consecutive_frames` (2), so small/distant/fast objects stay uncounted. Place counting
  lines where stable tracks are (near camera).
- **Per-source tuning** goes in `source_overrides` (e.g. market-square needs `detector.imgsz: 1280`
  because tiny people vanish at 640 — NOT a downscale/conf issue). Explicit `--set` always wins.
- **VLMs on M5**: LocateAnything-3B / Molmo2 need `decord` (no Apple-Silicon wheel) → run them on
  GCP/Linux behind the Grounder interface. **OWLv2 is the Mac-native open-vocab path (MPS).**
- **Web UI**: persist the `GeometryEngine` across frames; rebuild only on annotation-version change
  (rebuilding each frame wipes line-crossing side history → 0 crossings). MJPEG `<img>` never fires
  page `load`, so Playwright must use `wait_until="domcontentloaded"`.
- Keep the web element ids (`#zone #line #finish #start #cv` …) and `/stats` schema stable — the
  Playwright e2e (fishfood level 5) depends on them.
- Annotations are normalized (0..1) so they're resolution-independent; `configs/highway_lines.json`
  is the default pre-loaded set for the web UI (overridable via `OCC_ANNOTATIONS`).

## Git

Private repo `jeanlaboratories/occupancy_analysis` (origin/main). Commit identity is passed inline
(`-c user.name=... -c user.email=jeanlaboratories@gmail.com`) so global git config is untouched.
Run `fishfood.py` green before committing.
