# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

Local, real-time occupancy & traffic analytics on
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
- Extra groups: `[face]` (MediaPipe 478-pt landmarks), `[seg]` (rembg U²-Net segmentation),
  `[voice]` (faster-whisper STT), `[med]` (MedGemma), `[pay]` (Stripe checkout).
- Models auto-download on first use (YOLO `.pt`, HF checkpoints). Weights/clips/`out/` are gitignored.

## Commands

```bash
# real-time (OpenCV window by default; --no-show for headless)
.venv/bin/python run.py run --source assets/videos/vehicles-2.mp4
.venv/bin/python run.py run --config configs/highway_example.yaml --emit-proto out/run.pb \
    --analytics-csv out/counts.csv
# override ANY config field: --set detector.model=yolo26s.pt --set runtime.detect_every=3
# shortcuts: --detector {yolo,rfdetr,owlv2} --tracker {bytetrack,botsort,ocsort,sort} --detect-every N

.venv/bin/python run.py edit --source ... --annotations configs/cam.json   # draw zones/lines (GUI)
.venv/bin/python run.py ground --source ... --prompt "forklifts" --suggest-zone bay  # OWLv2 open-vocab

# browser dashboard (live-reconfig, draw/save, snapshot, record, timeline chart)
OCC_SOURCE=assets/videos/vehicles-2.mp4 .venv/bin/uvicorn occ.web:app --port 8001

# AGENTIC studio: an ADK "Vision Ninja" drives the pipeline + UI via chat + tools.
# `run.py studio` also boots the optional sidecars — litert-lm (:9379, local 🦙 voice) and the
# LocateAnything-3B grounding server (:9393, 🔍 locate) — and stops them on exit; idempotent
# (skips ones already up). --no-voice / --no-locate to skip. Logs → out/studio/logs/.
.venv/bin/python run.py studio                 # → http://127.0.0.1:8011  ([studio] extra) + sidecars
OCC_SOURCE=assets/videos/market-square.mp4 .venv/bin/uvicorn occ.studio.server:app --port 8011  # studio only

# tests
.venv/bin/python fishfood.py            # full dogfood, all clips, levels 1–6 (the green gate; expect 169/169)
.venv/bin/python test_proto_roundtrip.py test_phase2.py test_phase3.py test_phase4.py test_e2e_web.py
# Level 6 folds in the AR try-on / face-filter / stylist regression (offline phase).
# Run it standalone for the FULL stack (+endpoints/WS/UI) once the studio is up:
STUDIO_URL=http://127.0.0.1:8011 .venv/bin/python test_studio_tryon.py   # full offline+online+UI regression
```

After any change, run `fishfood.py` and expect **ALL PASS — 169/169** (levels 1–6;
level 6 = studio AR try-on, SKIPs without the `[face]` extra). It also writes
`out/fishfood/RUNBOOK.md` (a command per check) and a gallery of annotated frames.

## Architecture (`occ/`)

| Module | Role |
|---|---|
| `pipeline.py` | **Core glue.** `Pipeline`: source → detect-every-N → tracker (Kalman predicts on skip frames) → `GeometryEngine` → `Renderer` → `ResultWriter` + `AnalyticsSink`. Start here for any new end-to-end feature. |
| `config.py` | One YAML drives all. `Config.load(overrides=["a.b=v"])`; dotted `--set`; `source_overrides` (substring-keyed, CLI wins); class-group + COCO-name resolution. |
| `annotations.py` | `AnnotationSet` / `Annotation` dataclasses. Zones and lines are stored **normalized (0..1)** — resolution-independent. Serializes to/from JSON and to the `StreamAnnotation` proto. |
| `sources.py` | `FileSource` + newest-frame-wins `StreamSource` (RTSP/cam, auto-reconnect) + `PushSource` (for studio camera injection). |
| `capture_worker.py` | **Subprocess-isolated capture.** FFMPEG can SIGSEGV (inside `av_log`) on CDN host-rotation for YT Live; a crash there would take down the whole server. This module runs the capture in a child process and streams JPEG frames back via stdout. The studio server respawns it on crash. |
| `detectors/` | `build_detector(cfg)`; `yolo` (Ultralytics, MPS), `rfdetr` (HF transformers, MPS), `owlv2` (open-vocab, MPS). **All return `supervision.Detections`** so the tracker is detector-agnostic. |
| `tracking.py` | Factory over **roboflow/trackers**; signature-filters params (config is a permissive superset). ByteTrack/BoT-SORT/OC-SORT/SORT. |
| `geometry.py` | `GeometryEngine`: zones (point-in-polygon), line crossing (**segment∩segment** test — right-hand-rule direction, NOT infinite-line side-flip), **robust per-occupant dwell** (`_DwellTracker`: re-associates the same person across id-changes/occlusions via cost assignment + velocity prediction + edge hysteresis), full-frame counts. Uses each track's **bottom-center anchor**. |
| `stabilize.py` | `SceneStabilizer`: pin zones/lines to the SCENE — ORB/SIFT/AKAZE → RANSAC homography (EMA-smoothed) re-localizes annotations when the camera pans/tilts/rotates; `sim_transform`/`warp_annset` flip/rotate the stream to test redraw. |
| `emit.py` | `build_result()` → `OccupancyCountingPredictionResult`; `ResultWriter` (varint length-delimited `.pb` stream). |
| `analytics.py` | Per-interval CSV (line crossings as deltas; zone/full-frame avg+peak) + JSON summary. |
| `speed.py` | Homography speed (4-pt ground calibration) with discontinuity rejection. |
| `render.py` | Overlays; **scales with resolution** (legible at 1080p & 4K), text on backgrounds. |
| `editor.py` | Interactive OpenCV zone/line editor (mouse + keyboard + on-screen help). |
| `grounding/` | Open-vocab VLM grounders: `owlv2` (Mac/MPS default), `locate_anything`/`molmo2` (GCP/Linux). `run.py ground`. |
| `web.py` + `web_ui.html` | FastAPI dashboard: live-reconfig, MJPEG, canvas draw/save, stats, timeline chart, snapshot/record. Live-reconfigurable via a dirty-flag rebuild loop. |
| `proto/` (repo root, not under `occ/`) | Vendored, wire-identical `OccupancyCountingPredictionResult` (`visionai_annotations.proto` → `_pb2.py`). Regenerate: `python -m grpc_tools.protoc -Iproto --python_out=proto proto/visionai_annotations.proto`. |
| `studio/` | **Agentic layer** (Google ADK). Self-contained (never imports `occ.web`). Tools share state via `runtime.py`'s module-level `StudioContext` (set once at agent-build time, reached by all tool functions). Core: `neurons.py`+`ledger.py` (compute-points meter / neuron graph), `brain.py` (persistent scene memory), `settings.py` (sim axis `live/simulated/zero`), `overlays.py` (hot-loads agent-authored `def draw(ctx)` cv2 code into the render loop), `tools.py`, `agent.py` (ADK + deterministic **SimRunner** — no API key needed), `server.py`+`studio_ui.html`. Local/$0 model stack: `local_llm.py` (Gemma via LiteRT-LM :9379), `gemma_vision.py`/`medgemma.py` (MPS vision + medical), `voice_local.py` (Whisper STT + local TTS), `offline.py` (`STUDIO_OFFLINE=1`), `la3b.py` (LocateAnything-3B :9393). Live voice: `live.py` (Gemini BIDI, `GEMINI_API_KEY`), `live_openai.py` (OpenAI Realtime, `OPENAI_API_KEY` — better for Haitian Creole). Gesture browse: `gestures.py` (`GestureBrowser` — hand *position* drives store browsing, not noisy gesture classification). Media: `library.py` (auto-registers every snapshot/recording/gif with caption+tags; agent searches in natural language; persisted to `out/studio/library.json`). NLE: `nle.py` (ffmpeg `filter_complex` timeline assembly). Try-on/commerce: `face_filters.py`+`facemesh.py` (eyewear AR), `jewelry.py` (nose ring/earrings/necklace), `genmedia.py`/`qwen_image.py` (Nano Banana / local image), `commerce.py` (Stripe checkout), `eyewear_lab.py` (flag-based bad-case dataset under `out/studio/tryon_flags/`). Runs `$0` in sim mode; uses Gemini when `GOOGLE_API_KEY`/`GEMINI_API_KEY` is set. `run.py studio` boots the studio + litert + la3b sidecars together. **Stop the studio server (and sidecars) before running `fishfood.py`** — CV pipelines contend for MPS and flake the level-5 e2e. |

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
- **`capture_worker.py` isolation**: FFMPEG can SIGSEGV in native `av_log` on CDN host-rotation for
  YT Live streams. A native crash in-process takes down the entire server and can't be caught in
  Python. The studio server spawns `capture_worker` as a child process and respawns it on death —
  never inline the capture loop into the server for live stream sources.
- **Studio env vars**: `GOOGLE_API_KEY`/`GEMINI_API_KEY` (Gemini brain + Live voice),
  `OPENAI_API_KEY` (OpenAI Realtime voice backend), `STRIPE_SECRET_KEY` + `STRIPE_WEBHOOK_SECRET`
  (checkout), `STUDIO_OFFLINE=1` (airplane mode). Loaded from a gitignored `.env` via python-dotenv.

## Git

Private repo `jeanlaboratories/occupancy_analysis` (origin/main). Commit identity is passed inline
(`-c user.name=... -c user.email=jeanlaboratories@gmail.com`) so global git config is untouched.
Run `fishfood.py` green before committing.
