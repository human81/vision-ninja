"""Gemini generative media — the studio's imagination.

Thin helpers over google-genai for: semantic vision (describe a frame),
nano-banana image editing (gemini-2.5-flash-image), and Veo video generation.
All need a Gemini key; callers gate on `has_key()` and fall back gracefully.
Each returns plain bytes/text so tools can save + library + meter them.
"""

from __future__ import annotations

import os
import threading
import time

VISION_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"      # "nano-banana"
VEO_MODEL = "veo-3.0-fast-generate-001"

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def has_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _client():
    # A SINGLE reused client — a fresh genai.Client() per call dies in ADK's tool
    # worker threads ("client has been closed"). Pre-warm() it on the main thread.
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                from google import genai
                _CLIENT = genai.Client()
    return _CLIENT


def prewarm():
    """Create the client on the main thread at startup (if a key is present)."""
    if has_key():
        try:
            _client()
        except Exception:
            pass


def describe(jpg: bytes, prompt: str, model: str = VISION_MODEL):
    """Return (text, response) — semantic description/answer about an image."""
    from google.genai import types
    r = _client().models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=jpg, mime_type="image/jpeg"), prompt])
    return (r.text or "").strip(), r


def edit_image(jpgs: list[bytes], prompt: str, model: str = IMAGE_MODEL):
    """nano-banana: edit/compose from one or more input images. Returns
    (out_png_bytes | None, text, response)."""
    from google.genai import types
    parts = [prompt] + [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in jpgs]
    r = _client().models.generate_content(model=model, contents=parts)
    out, text = None, ""
    for cand in (r.candidates or []):
        for p in (getattr(cand.content, "parts", None) or []):
            inline = getattr(p, "inline_data", None)
            if inline and getattr(inline, "data", None):
                out = inline.data
            elif getattr(p, "text", None):
                text += p.text
    return out, text.strip(), r


def generate_video(prompt: str, image_jpg: bytes | None = None,
                   model: str = VEO_MODEL, poll_s: int = 8, max_wait: int = 240):
    """Veo: text(+image)->video. Blocks (Veo takes minutes). Returns
    (mp4_bytes | None, error)."""
    from google.genai import types
    client = _client()
    kwargs = {}
    if image_jpg:
        kwargs["image"] = types.Image(image_bytes=image_jpg, mime_type="image/jpeg")
    op = client.models.generate_videos(model=model, prompt=prompt, **kwargs)
    waited = 0
    while not op.done and waited < max_wait:
        time.sleep(poll_s); waited += poll_s
        op = client.operations.get(op)
    if not op.done:
        return None, f"veo timed out after {max_wait}s"
    resp = getattr(op, "response", None) or getattr(op, "result", None)
    vids = getattr(resp, "generated_videos", None) if resp else None
    if not vids:
        return None, "veo returned no video"
    vfile = vids[0].video
    try:
        client.files.download(file=vfile)
        data = getattr(vfile, "video_bytes", None)
    except Exception:
        data = getattr(vfile, "video_bytes", None)
    return (data, "") if data else (None, "veo gave no bytes")
