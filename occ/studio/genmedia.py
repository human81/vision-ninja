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

# Narration TTS. Provider is inferred from the model id (gpt-* → OpenAI, else
# Gemini) — both are multilingual; OpenAI's gpt-4o-mini-tts is the strongest at
# non-English accents (Haitian Creole). Default via env (a "setting").
TTS_MODEL = os.environ.get("STUDIO_TTS_MODEL", "gemini-2.5-flash-preview-tts")
# Default voice per provider (Gemini prebuilt vs OpenAI named voices).
_OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable",
                  "onyx", "nova", "sage", "shimmer"}


def tts_provider(model: str) -> str:
    return "openai" if (model or "").lower().startswith("gpt") else "gemini"

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


def _vertex_token_project():
    """ADC token + (project, location) for Vertex AI. Raises with a clear message
    if credentials need a refresh (run `gcloud auth application-default login`)."""
    import google.auth
    import google.auth.transport.requests
    creds, proj = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or proj
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    return creds.token, project, location


def generate_music(prompt: str, negative_prompt: str = "", model: str = "lyria-002"):
    """Lyria instrumental music on VERTEX AI (not the Gemini Developer API).
    Returns (wav_bytes | None, error). ~30s, 48kHz."""
    import base64
    import json
    import urllib.request
    try:
        token, project, location = _vertex_token_project()
    except Exception as e:
        return None, ("Vertex auth needs a refresh — run "
                      "`gcloud auth application-default login`. (" + str(e)[:120] + ")")
    if not project:
        return None, "no GCP project — set GOOGLE_CLOUD_PROJECT"
    url = (f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/"
           f"locations/{location}/publishers/google/models/{model}:predict")
    inst = {"prompt": prompt}
    if negative_prompt:
        inst["negative_prompt"] = negative_prompt
    body = {"instances": [inst], "parameters": {"sample_count": 1}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    except urllib.error.HTTPError as e:                       # noqa: F821
        return None, f"Lyria HTTP {e.code}: {e.read().decode()[:160]}"
    except Exception as e:
        return None, f"Lyria error: {e}"
    preds = r.get("predictions", [])
    if not preds:
        return None, "Lyria returned no audio"
    p = preds[0]
    b64 = (p.get("bytesBase64Encoded") or p.get("audioContent")
           or next((v for v in p.values() if isinstance(v, str) and len(v) > 1000), None))
    if not b64:
        return None, f"no audio field (keys: {list(p.keys())})"
    try:
        return base64.b64decode(b64), ""
    except Exception as e:
        return None, str(e)


def image(prompt: str, model: str = IMAGE_MODEL):
    """Text -> image (no input) via the image model. Returns (png_bytes|None, text, r)."""
    return edit_image([], prompt, model=model)


def plan_story(brief: str, n: int = 3, model: str = VISION_MODEL):
    """Co-Director scene planning: return ([{prompt, narration}], response). Gemini
    writes a short cinematic storyline as strict JSON."""
    import json
    from google.genai import types
    instr = (f"You are a creative director (in the spirit of Google's Co-Director). "
             f"For this brief: '{brief}', write EXACTLY {n} short video scenes for a "
             f"cinematic reel. Return ONLY a JSON array of objects with keys 'prompt' "
             f"(a vivid image/video-generation prompt) and 'narration' (one short spoken "
             f"sentence). No prose, no markdown.")
    try:
        r = _client().models.generate_content(
            model=model, contents=instr,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        data = json.loads(r.text)
        scenes = [{"prompt": str(s.get("prompt", "")),
                   "narration": str(s.get("narration", ""))}
                  for s in data if isinstance(s, dict)][:n]
        return scenes, r
    except Exception:
        return [], None


def tts(text: str, voice: str = "Puck", model: str | None = None):
    """Text-to-speech → (wav_bytes | None, error). Routes by model id: gpt-* →
    OpenAI gpt-4o-mini-tts (best non-English accents, incl. Haitian Creole), else
    Gemini TTS. Both emit/are normalized to WAV. `model` defaults to TTS_MODEL."""
    model = model or TTS_MODEL
    if tts_provider(model) == "openai":
        return _tts_openai(text, voice, model)
    import io
    import wave
    from google.genai import types
    try:
        r = _client().models.generate_content(
            model=model, contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
        pcm = None
        for cand in (r.candidates or []):
            for p in (getattr(cand.content, "parts", None) or []):
                inline = getattr(p, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    pcm = inline.data
        if not pcm:
            return None, "no audio returned"
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(pcm)
        return buf.getvalue(), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _tts_openai(text: str, voice: str, model: str = "gpt-4o-mini-tts"):
    """OpenAI multilingual TTS → (wav_bytes | None, error). Raw HTTPS POST so we
    add no SDK dependency. We request raw `pcm` (24kHz 16-bit mono) and wrap it in
    a clean WAV — OpenAI's `wav` format uses a streaming header (unknown length)
    that strict parsers (ffmpeg in the NLE) mishandle. `voice` falls back to a
    valid OpenAI voice."""
    import io
    import json
    import urllib.request
    import wave
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None, "OpenAI TTS needs OPENAI_API_KEY"
    v = voice if voice in _OPENAI_VOICES else "alloy"
    body = json.dumps({"model": model, "voice": v, "input": text,
                       "response_format": "pcm"}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            pcm = r.read()
        if not pcm:
            return None, "no audio returned"
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(pcm)
        return buf.getvalue(), ""
    except Exception as e:
        detail = ""
        if hasattr(e, "read"):
            try:
                detail = " " + e.read().decode()[:200]
            except Exception:
                pass
        return None, f"{type(e).__name__}: {e}{detail}"


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
