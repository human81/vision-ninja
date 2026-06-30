"""Client for the LocateAnything-3B grounding sidecar (scripts/la3b_server.py).

The model pins transformers==4.57.1 (vs the studio's 5.x) and needs a decord stub + sdpa, so
it can't be imported in-process — it runs in .venv-la3b behind a small HTTP server. This is
the thin client the studio calls. Start the sidecar:  .venv-la3b/bin/python scripts/la3b_server.py
Env: STUDIO_LA3B_URL (default http://127.0.0.1:9393).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request


def url() -> str:
    return os.environ.get("STUDIO_LA3B_URL", "http://127.0.0.1:9393").rstrip("/")


def health(timeout: float = 2.0) -> bool:
    try:
        r = json.loads(urllib.request.urlopen(url() + "/health", timeout=timeout).read())
        return bool(r.get("ready"))
    except Exception:
        return False


def ground(jpg_bytes: bytes, prompt: str, max_side: int = 1024, timeout: float = 180) -> dict:
    """POST the frame + prompt → {boxes: [[x1,y1,x2,y2] normalized 0..1], n, secs, prompt}."""
    body = json.dumps({"image": base64.b64encode(jpg_bytes).decode(),
                       "prompt": prompt, "max_side": max_side}).encode()
    req = urllib.request.Request(url() + "/ground", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())
