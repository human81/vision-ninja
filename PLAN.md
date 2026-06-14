# Occupancy & Traffic Analysis — Implementation Plan

Real-time, cost-effective occupancy/traffic analytics running **locally** on Apple
Silicon (M5 Pro, 48 GB, Metal 4). Detect → track → spatial math → emit the Vertex AI
Vision `OccupancyCountingPredictionResult` protobuf + render overlays. Optimized for
**RTSP real-time** and **vehicle/traffic** workloads.

## Goals & constraints

- **Optimal real-time on RTSP** — newest-frame-wins capture, no latency buildup, smooth full-FPS overlays even when detection runs slower.
- **Cost-effective** — everything runs locally on the M5 Pro by default. No per-frame cloud calls. VLMs are optional and off the hot path.
- **Traffic analysis** — vehicles first (car/truck/bus/motorcycle/bicycle) + people; directional line-crossing → turning-movement / approach counts; zone occupancy / queue length; optional speed estimation.
- **Protobuf-compatible** — emit the exact `OccupancyCountingPredictionResult` so downstream tooling (incl. the existing `parse_occupancy.py`) is unchanged.
- **Swappable** — `Detector`, `Tracker`, `Grounder` are interfaces. Detector default chosen per-run via CLI flag.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Detector | **Both, runtime flag** — `--detector yolo` (default, real-time) / `--detector rfdetr` (accuracy) |
| Tracker | **roboflow/trackers** — ByteTrack default, BoT-SORT for moving/PTZ cams |
| VLMs | **Local MPS first** — LocateAnything-3B / Molmo2, off the hot path, GCP-able later behind the interface |
| Runtime env | **Python 3.12 venv** (3.14 has no torch/ultralytics/transformers wheels yet) |
| Platform | Local Apple Silicon, MPS + VideoToolbox HW decode |

## Architecture

```
            ┌─────────────── capture thread (newest-frame-wins) ───────────────┐
RTSP/file/cam → VideoCapture(FFMPEG, rtsp_transport=tcp, buffer=1, HW decode) → latest-frame slot
            └───────────────────────────────────────────────────────────────────┘
                                          │ (latest frame only)
                                          ▼
   Detector (YOLO11 | RF-DETR, MPS, FP16)  — runs every N frames
                                          ▼
   Tracker (roboflow/trackers; Kalman predict fills the in-between frames → full-FPS IDs)
                                          ▼
   Geometry engine:  point-in-zone · line-cross (right-hand rule) · dwell-by-track-id · (opt) speed
                                          ▼
        ┌── Protobuf emitter → OccupancyCountingPredictionResult (3 FPS cadence, matches model)
        ├── Analytics sink   → CSV/JSON time series (per-class, per-direction, per-interval)
        └── OpenCV renderer  → boxes, IDs, trails, zones, lines, live counts (full FPS)

   Grounder (LocateAnything / Molmo2, async, MPS) — open-vocab zone setup + sampled validation, never blocks
   Zone/Line editor — interactive OpenCV window, mouse + keyboard, persist as StreamAnnotation
```

## The real-time RTSP core (the hard part)

This is where "optimal real-time" is won or lost:

1. **Newest-frame-wins capture.** Dedicated thread runs `cap.grab()` in a tight loop and only `retrieve()`s the latest frame on demand. RTSP/network jitter never accumulates as latency. Single-slot buffer (latest overwrites).
2. **Transport + decode.** `OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"` (reliable) or `udp` (lowest latency) — configurable. macOS **VideoToolbox** hardware decode to keep CPU free.
3. **Auto-reconnect** with backoff when the stream drops (traffic cams are flaky).
4. **Detect-every-N + track-between.** Detector runs every N frames; the tracker's Kalman filter predicts positions on the skipped frames. IDs and overlays stay smooth at full display FPS while inference cost drops ~N×. N is adaptive to keep up with the stream — the key cost/latency lever.
5. **Three decoupled stages** (capture / inference / render) so a slow detector never stalls capture or display.
6. **Resolution & model tier** knobs (`yolo11n/s/m`, input size) to trade accuracy for FPS per camera.

## Traffic-analysis features

- **Classes:** COCO-pretrained YOLO/RF-DETR already cover car, truck, bus, motorcycle, bicycle, person — no training needed to start.
- **Directional line crossing → turning movements / approach volumes.** Right-hand-rule sign matches the model's positive/negative direction semantics exactly. Per-class, per-direction tallies.
- **Zone occupancy & queue length.** Point-in-polygon per zone; sustained occupancy = congestion/queue metric.
- **Dwell / stopped-vehicle detection.** Track-id timestamps → stopped vehicles, queue dwell, illegal-stop candidates.
- **Speed estimation (optional, phase 4).** Pixel→ground homography from a 4-point calibration; speed from track displacement. Off by default (needs per-camera calibration).
- **Aggregation sink.** Rolling per-interval counts (e.g. 15-min bins) to CSV/JSON alongside the protobuf, for traffic reporting.

## Package layout

```
occupancy_analysis/
├── pyproject.toml              # 3.12, pinned deps
├── proto/
│   ├── occupancy.proto         # OccupancyCountingPredictionResult + StreamAnnotation (vendored)
│   └── occupancy_pb2.py        # generated
├── occ/
│   ├── sources.py              # RtspSource / FileSource / CameraSource (newest-frame-wins)
│   ├── detectors/  base.py · yolo.py · rfdetr.py
│   ├── tracking.py             # roboflow/trackers wrapper → supervision.Detections
│   ├── geometry.py             # zones, line-crossing, dwell, (opt) speed
│   ├── emit.py                 # build OccupancyCountingPredictionResult
│   ├── analytics.py            # CSV/JSON time-series sink
│   ├── render.py               # OpenCV overlays
│   ├── grounding/  base.py · locate_anything.py · molmo2.py   # async, MPS
│   ├── editor.py               # interactive zone/line drawing → StreamAnnotation
│   └── pipeline.py             # threaded orchestration
├── configs/                    # per-camera: source, zones/lines, detector, N, classes
├── parse_occupancy.py          # already present
└── run.py                      # CLI entrypoint
```

## CLI sketch

```bash
# draw zones/lines once, save config
python run.py edit --source rtsp://cam/stream --out configs/intersection7.json

# run real-time
python run.py run --config configs/intersection7.json \
    --detector yolo --model yolo11s --every 3 --classes vehicle,person \
    --emit-proto out/%Y%m%d.pb --emit-csv out/counts.csv --show
```

## Phased milestones

- **Phase 0 — Env & contract.** 3.12 venv, deps, vendor `occupancy.proto`, compile `_pb2`, verify round-trip with `parse_occupancy.py`. *(no CV yet — fixes the data contract first)*
- **Phase 1 — Real-time spine. ✅ DONE.** Newest-frame-wins RTSP/file/cam source + YOLO11(MPS) + roboflow/trackers + OpenCV render of boxes/IDs/trails, all config-driven. Proven: 79 fps @ 1080p (ByteTrack), tracker swap (BoT-SORT/OC-SORT/SORT) + detect-every cost lever working. `python run.py run ...`
- **Phase 2 — Geometry + protobuf + interactive editor. ✅ DONE.** Zones (point-in-polygon), directional line-crossing (right-hand rule), dwell, full-frame counts → emits wire-valid `OccupancyCountingPredictionResult` stream; interactive mouse+keyboard editor with on-screen guidance. Verified: directional counts, zone occupancy, 400-msg protobuf round-trip. `python run.py edit ...` / `run.py run --annotations ... --emit-proto ...`
  - **Interactive OpenCV editor (required):** runs on a frozen/live frame in an OpenCV window. Fully driven by **mouse + keyboard** with **on-screen guidance** (persistent help overlay + status line + step prompts):
    - **Zones:** left-click to drop polygon vertices, close to commit; a filled preview tracks the cursor.
    - **Lines:** click two endpoints; an **arrow renders the positive (right-hand-rule) direction** so the user sees "in vs out" while drawing, with a key to flip it.
    - **Keyboard:** `z` new zone · `l` new line · `Enter`/`c` commit · `Backspace`/`u` undo last vertex · `n` name the item (typed inline) · `f` flip line direction · `Tab` cycle/select items · `d` delete selected · `s` save → JSON `StreamAnnotation` · `r` reset · `h` toggle help · `q`/`Esc` quit.
    - Always-visible HUD: current mode, item count, active item name, and the next expected action ("click 2nd line point", "press Enter to close zone", etc.).
    - Per-source persistence keyed by resolution; reload + edit existing annotations.
- **Phase 3 — Traffic analytics + RF-DETR. ✅ DONE.** Per-class/per-direction interval aggregation → tidy CSV + JSON summary (line crossings, zone occupancy avg/peak, full-frame avg/peak); RF-DETR behind `--detector rfdetr` (HF transformers, runs on MPS); BoT-SORT already available. Verified: directional+per-class counts over interval bins, RF-DETR detecting on MPS. `run.py run --detector rfdetr --analytics-csv ...`
- **Phase 4 — VLMs (local MPS) + speed. ✅ DONE.** Homography speed estimation (4-point ground calibration, discontinuity-robust, km/h on overlay) + LocateAnything-3B grounder for open-vocab zone setup (`run.py ground --prompt ... --suggest-zone`); Molmo2 stub. Grounder loads lazily (no download until used), MPS/CPU/GCP-able. Verified: plausible speeds (median peak 102 km/h), VLM parse + zone suggestion. Weights are an explicit opt-in pull.

## Cost posture

Default config = **$0 marginal cost**: local YOLO11 + ByteTrack + OpenCV on the M5 Pro, no cloud. RF-DETR and local VLMs add compute but no spend. GCP only enters if VLM throughput ever needs to scale beyond one box — and it's already abstracted for that.

## Resolved scope

1. **Sources:** both RTSP **and** file, with a diverse verified test set (below). Functional RTSP via a local loopback (`mediamtx` + ffmpeg) — public RTSP test streams are dead/unreliable.
2. **Classes:** people **and** vehicles together from day one.
3. **Speed estimation:** deferred to Phase 4.

## Test footage (verified live + decodable, 2026-06-14)

`assets/fetch_assets.sh` downloads these; `assets/rtsp_sim.sh` loops any of them as RTSP.

| File | Res / FPS | Orientation | Scene |
|---|---|---|---|
| vehicles.mp4 | 4K / 25 | landscape | highway vehicles |
| vehicles-2.mp4 | 1080p / 30 | landscape | road vehicles |
| people-walking.mp4 | 1080p / 25 | landscape | pedestrian street |
| market-square.mp4 | 4K / 60 | vertical | crowded plaza (people + vehicles) |
| grocery-store.mp4 | 4K / 30 | landscape | indoor retail (people) |
| subway.mp4 | 4K / 30 | vertical | transit (people) |

Spread: 1080p→4K, 25/30/60 fps, landscape + portrait, outdoor traffic + dense crowds + indoor. Functional RTSP:

```bash
brew install mediamtx          # once
mediamtx                       # terminal A — RTSP server on :8554
./assets/rtsp_sim.sh assets/videos/vehicles.mp4 cam   # terminal B
# app target → rtsp://localhost:8554/cam
```
```
