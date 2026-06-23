"""occ.studio — the agentic computer-vision studio.

A Google ADK agent ("the Vision Ninja") drives the live CV pipeline and the
browser UI through tools + conversation. Mirrors the Momentum architecture:

  * neurons.py / ledger.py  — a data-flow graph of charge points + a usage
    ledger that meters every op in compute *points* (and real USD for the
    Gemini brain).  [Momentum: billing/neurons + usage ledger]
  * settings.py             — typed settings with a simulation axis
    (live / simulated / zero) and per-node modes, so it runs with NO API key.
  * brain.py                — the Vision Brain: persistent scene/task memory.
    [Momentum: Brand Brain / Brand Soul]
  * overlays.py             — the OpenCV ninja: hot-loads agent-authored cv2
    overlay code into the render loop to achieve any visual task.
  * ffmpeg.py / tools.py    — the agent's hands (export + pipeline control).
  * agent.py                — ADK agent + a deterministic SimRunner fallback.
  * server.py               — FastAPI: NDJSON agent stream + the live view.
"""

STUDIO_DIR = "out/studio"

# Honour STUDIO_OFFLINE=1 before any transformers / huggingface_hub import, so local
# model loads are cache-only (no network round-trip that hangs when wifi is off).
from . import offline as _offline  # noqa: E402
_offline.apply()
