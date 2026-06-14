# Occupancy & Traffic Analysis

Real-time, local, cost-effective occupancy/traffic analytics on Apple Silicon.
Capture (RTSP / file / camera) → detect → track → spatial math (zones, directional
line-crossing, dwell, speed) → emit the Vertex AI Vision `OccupancyCountingPredictionResult`
protobuf + annotated overlays + traffic CSV. Runs $0-marginal-cost on the local GPU.

## Setup

```bash
uv venv --python 3.12 .venv           # 3.14 has no torch/ultralytics wheels yet
uv pip install -e ".[cv]"             # core: torch, opencv, ultralytics, trackers, supervision
uv pip install -e ".[rfdetr]"         # optional: RF-DETR accuracy detector
bash assets/fetch_assets.sh           # 6 diverse test clips
```

## Run

```bash
# real-time on a video file (window)
python run.py run --source assets/videos/vehicles-2.mp4

# live RTSP (functional local test stream: brew install mediamtx; mediamtx; ./assets/rtsp_sim.sh ...)
python run.py run --source rtsp://localhost:8554/cam

# full traffic pipeline: counts + zone + speed + analytics
python run.py run --config configs/highway_example.yaml \
    --emit-proto out/run.pb --analytics-csv out/counts.csv --analytics-json out/summary.json
```

Any config field is overridable: `--set detector.model=yolo11s.pt --set runtime.detect_every=2`,
or shortcuts `--detector {yolo,rfdetr}`, `--tracker {bytetrack,botsort,ocsort,sort}`, `--detect-every N`.

## Draw zones & lines (interactive)

```bash
python run.py edit --source assets/videos/vehicles-2.mp4 --annotations configs/mycam.json
```
Mouse + keyboard with on-screen guidance: `z` zone · `l` line · click vertices · `Enter`
commit · `u` undo · `n` name · `f` flip line direction · `Tab` select · `d` delete · `s`
save · `h` help · `q` quit. A green arrow shows each line's positive (right-hand-rule) direction.

## Open-vocab zone setup (VLM, optional)

```bash
python run.py ground --source ... --prompt "forklifts" --suggest-zone loading_bay \
    --annotations configs/mycam.json
```
First run downloads LocateAnything-3B (~6 GB) to MPS. Off the real-time hot path.

## Layout

| Path | Role |
|---|---|
| `occ/sources.py` | newest-frame-wins capture (RTSP/file/cam) |
| `occ/detectors/` | YOLO11 / RF-DETR → `supervision.Detections` |
| `occ/tracking.py` | roboflow/trackers factory (ByteTrack/BoT-SORT/OC-SORT/SORT) |
| `occ/geometry.py` | zones, line-crossing, dwell |
| `occ/speed.py` | homography speed estimation |
| `occ/emit.py` | `OccupancyCountingPredictionResult` protobuf stream |
| `occ/analytics.py` | per-interval traffic CSV + JSON |
| `occ/editor.py` | interactive zone/line editor |
| `occ/grounding/` | VLM open-vocab grounding (off hot path) |
| `proto/` | vendored, wire-identical occupancy protobuf |
| `configs/default.yaml` | every knob (💰 = cost levers) |

## Cost levers (`configs/default.yaml`)

Model tier (`yolo11n…x`), `detect_every` (detect every N frames, track between),
`imgsz`, `half` (FP16), `max_long_side` (downscale 4K→1080p). Default config is real-time
and free on an M5 Pro.

## Browser UI (Playwright-driveable)

A thin web UI mirrors the pipeline so it can be driven (and automated) from a browser —
useful where an OpenCV window can't be (CI, headless, remote).

```bash
uv pip install -e ".[web]" && .venv/bin/playwright install chromium
OCC_SOURCE=assets/videos/vehicles-2.mp4 .venv/bin/uvicorn occ.web:app --port 8000
# open http://localhost:8000 → Add zone / Add line (click canvas) → Start → live counts
```

## Fishfood — incremental, exhaustive dogfood

One harness drives the whole system over **all 6 clips**, climbing from a smoke check
to a full feature matrix, then a Playwright browser e2e. Writes a gallery of annotated
frames + a runbook of the command to reproduce each check.

```bash
.venv/bin/python fishfood.py                 # all levels (1–5), all clips
.venv/bin/python fishfood.py --level 2       # quick: per-source coverage only
# → out/fishfood/*.png (gallery) + out/fishfood/RUNBOOK.md (commands to run each)
```

## Tests

```bash
.venv/bin/python test_proto_roundtrip.py   # protobuf contract
.venv/bin/python test_phase2.py            # geometry + editor + emit
.venv/bin/python test_phase3.py            # analytics + RF-DETR
.venv/bin/python test_phase4.py            # speed + VLM logic
.venv/bin/python test_e2e_web.py           # Playwright browser end-to-end
.venv/bin/python fishfood.py               # everything, all clips, levels 1–5
```
