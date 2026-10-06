# Architecture & design decisions

This is the "why" behind the code: the problems each piece solves and the trade-offs taken.
For usage, see the [README](../README.md).

## 1. The real-time core

**Problem:** live RTSP streams arrive faster than a detector can always keep up, network
jitter is constant, and traffic cameras drop. A naive `read() → infer → show` loop builds
latency without bound.

**Decisions**

- **Newest-frame-wins capture** (`occ/sources.py`). A capture thread keeps only the latest
  frame in a single slot; the processing loop always takes the freshest one. Latency stays
  bounded by one frame plus inference time, whatever the network does.
- **Detect every N frames, track in between** (`runtime.detect_every`). The tracker's Kalman
  filter predicts positions on skipped frames, so IDs and overlays stay at full frame rate
  while inference cost drops roughly N×. It's the main cost lever, alongside model tier,
  `imgsz`, FP16 and `max_long_side` (marked 💰 in `configs/default.yaml`).
- **On skip frames, hold the last tracked result.** Feeding the tracker *empty* detections
  instead makes ByteTrack conclude every object vanished and drop all tracks. All three loops
  (`occ/pipeline.py`, `occ/web.py`, `occ/studio/pipeline.py`) preserve this.
- **Capture in a subprocess for live streams** (`occ/capture_worker.py`). FFmpeg can segfault
  inside native `av_log` when a YouTube Live CDN rotates hosts. A native crash can't be caught
  in Python and would take down the whole server, so capture runs in a child process that
  streams JPEG frames back and is respawned when it dies.

## 2. Counting correctly

The value of an occupancy system is in counts people can trust, so the geometry is explicit
about its contract (`occ/geometry.py`):

- **Ground anchor.** Each track is placed at its bounding box's **bottom-center** — where the
  object touches the ground — not its center, which drifts with perspective and box height.
- **Line crossing is a segment∩segment test,** not "which side of the infinite line am I on".
  Side-flip tests count an object passing the line's *extension* far from the drawn segment,
  and double-count jitter around the line. Direction follows the right-hand rule relative to
  the drawn line, so "in" and "out" are stable and can be flipped in the editor.
- **The engine is stateful across frames.** Crossing needs the previous anchor; rebuilding
  the engine per frame silently yields zero crossings. The web loops rebuild it only when the
  annotation version changes.
- **Dwell survives ID switches.** Occlusions make trackers reassign IDs, which naively resets
  a person's dwell time to zero. `_DwellTracker` re-associates occupants across ID changes with
  a cost assignment, velocity prediction and edge hysteresis.
- **Only confirmed tracks count.** ByteTrack confirms a track after a few consecutive
  detections; unconfirmed boxes (id −1 on screen) are drawn but never counted.
- **Annotations are normalized (0..1).** Zones and lines are resolution-independent, so the
  same configuration works across 720p, 1080p and 4K sources. `occ/stabilize.py` can also pin
  them to the *scene* (feature matching + RANSAC homography) when a camera pans.

## 3. Output contract

The system emits the wire-identical Vertex AI Vision `OccupancyCountingPredictionResult`
(vendored in `proto/`), written as a varint length-delimited stream. Existing consumers keep
working unchanged; `test_proto_roundtrip.py` and fishfood level 4 assert field coverage and
exact round-trips.

## 4. Models behind interfaces

Every detector implements `detect(frame) → supervision.Detections`, so trackers and geometry
never know which model ran. Two consequences worth knowing:

- YOLO filters classes **in-engine by COCO id** (non-target classes never leave the GPU);
  RF-DETR and OWLv2 have different label spaces, so they filter **by name**.
- `occ/device.py` resolves the device mps → cuda → cpu. The same config (`device: mps`) runs on
  a Mac, an L4 and a CPU-only container, and FP16 is switched off where it isn't supported.

## 5. The agentic studio

`occ/studio/` puts a Google ADK agent in front of the live pipeline.

- **Tools, not prompts, are the API.** The agent acts through typed Python tools (set
  detector, draw zone, record, analyze the scene, edit a clip…) that operate on one
  `StudioContext`. The browser applies the agent's streamed NDJSON frames as UI updates.
- **Works with no API key.** A deterministic `SimRunner` handles common requests with the
  same tools, which keeps the studio demo-able and testable offline. With a key, Gemini runs
  the agent; with LiteRT-LM it runs on a local Gemma 4 at $0.
- **Everything is metered.** Each tool call records compute points in a ledger, so cost is
  visible per neuron (detect, track, render, brain…), and a simulation axis
  (`live / simulated / zero`) lets expensive nodes be faked during development.
- **Scene memory.** `brain.py` keeps what the agent learned about a camera across sessions.

## 6. Security model

The agent has powerful tools, so every default fails closed:

- **Sign-in on by default.** Firebase ID token → httpOnly session cookie (cookies, because the
  MJPEG `<img>` and WebSockets can't send headers). One **pure-ASGI** middleware checks every
  HTTP route *and* WebSocket. That's deliberately not per-route decorators, so a new route is
  protected without anyone remembering to protect it. Public routes are an explicit allow-list.
- **Misconfiguration fails closed.** With auth switched off, the server only answers loopback
  clients and refuses anything carrying proxy headers, so an "off" deployment on Cloud Run
  serves nobody rather than everybody.
- **Agent-authored code is a local-only feature.** The overlay sandbox restricts builtins but
  still exposes `numpy`/`cv2`, which can read files, so it isn't a security boundary. It's off
  unless `STUDIO_AGENT_CODE=on`, and regression tests prove the refusal happens *before* any
  code runs. Built-in presets ship with the repo and always work.
- **No splicing of user text into code or commands.** Chat-derived labels are sanitized and
  `repr()`-quoted before entering generated source; URLs go to yt-dlp after `--`, so a "URL"
  like `--exec=…` can't become an option.

## 7. Cloud Run

- **One image, two variants.** `VARIANT=cpu` uses CPU torch wheels (no 2 GB of CUDA). On the
  GPU, `VARIANT=gpu` runs the same process tree as `run.py studio` on a laptop: the studio plus
  LocateAnything-3B and LiteRT-LM on loopback, sharing the L4.
- **Single always-on instance.** The pipeline and agent live in process memory, so the
  service is pinned to one instance with CPU always allocated. That's an explicit trade-off for
  a single-tenant studio; multi-tenancy would mean moving `StudioContext` out of module scope.
- **GPU memory is budgeted.** The L4 has 24 GB; Gemma 3 vision and MedGemma (~8.6 GB each),
  LocateAnything-3B (~7 GB) and YOLO don't all fit, and only one vision model is ever selected,
  so loading one unloads the other.
- **Weights are seeded, not downloaded at boot.** Cloud Run's shared egress IPs get rate-limited
  by Hugging Face, and gated models need a token. `scripts/seed_models.sh` uploads the laptop's
  exact cached weights to the service's bucket, and the service runs cache-only.
- **Org-policy friendly.** The organization forbids `allUsers` IAM grants (domain-restricted
  sharing), so the services disable Cloud Run's invoker check instead and rely on the app's own
  sign-in gate.
- **CPU service hides local models.** On a CPU instance an 8 GB vision model would exhaust
  memory for every user, so `STUDIO_LOCAL_MODELS=off` hides those models from the UI, maps
  saved choices to Gemini, and makes the loaders refuse.

## 8. Testing strategy

- **`fishfood.py`** is the gate: a leveled harness from a smoke test up to every tracker,
  analytics, speed, RF-DETR, protobuf field coverage, a Playwright browser end-to-end and the
  studio suites, over six real clips. It writes an annotated gallery and a runbook with the
  command to reproduce each check.
- **Offline suites** run without clips, models or network: auth (against a fake Firebase),
  agent-code policy, cloud readiness (simulating no-MPS and CUDA hosts) and geometry
  invariants. CI runs these on every push.
- **Known gaps:** the studio's ~85 routes are only partly covered offline, and the Stripe flow
  has no offline tests yet.
