"""Local LLM backend — Gemma 4 (Google AI Edge · LiteRT-LM) to cut cost.

When the agent or vision model is selected as a local model (a `gemma-*` id in the
settings dropdowns), the studio talks to the LiteRT-LM **OpenAI-compatible** server
(`litert-lm serve`, default :9379) instead of the cloud. This makes the reasoning +
vision-understanding layer **$0, private, and offline** on Apple Silicon.

The generative media (Nano Banana / Imagen / Veo) stays on cloud — there's no open
local equivalent. If the local server isn't reachable, callers fall back gracefully to
the deterministic SimRunner (agent) or cloud/detection (vision).

Bring it up once:
    litert-lm import --from-huggingface-repo=litert-community/gemma-4-12B-it-litert-lm
    litert-lm serve                      # OpenAI endpoint on :9379

Env: STUDIO_LOCAL_ENDPOINT (default http://localhost:9379/v1),
     STUDIO_LOCAL_MODEL    (exact registry id sent to the server; default = the
                            selected dropdown id).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import urllib.request

# model ids (dropdown values) that route to the LOCAL LiteRT-LM endpoint
LOCAL_PREFIXES = ("gemma",)

# LiteRT-LM's Metal engine is NOT concurrency-safe — two overlapping inference requests
# collide on a GPU buffer ("already has an outstanding map pending") and 500, which can
# wedge the engine. Serialise all inference from the studio so voice + text + vision never
# overlap on the local brain.
_INFER_LOCK = threading.Lock()


def is_local_model(model: str | None) -> bool:
    return bool(model) and str(model).lower().startswith(LOCAL_PREFIXES)


def endpoint() -> str:
    return os.environ.get("STUDIO_LOCAL_ENDPOINT", "http://localhost:9379/v1").rstrip("/")


_RESOLVED: dict = {}


def _served_model(model: str | None) -> str:
    """The exact model id to send. LiteRT-LM registers a base id plus a `<id>,gpu` variant;
    the Gemma builds are GPU(Metal)-only on Apple Silicon, so we PREFER the `,gpu` variant
    when the server advertises it. STUDIO_LOCAL_MODEL overrides this entirely. Cached per
    (endpoint, base)."""
    override = os.environ.get("STUDIO_LOCAL_MODEL")
    if override:
        return override
    base = model or "gemma-4-12b-it"
    key = (endpoint(), base)
    if key in _RESOLVED:
        return _RESOLVED[key]
    pick = base
    try:
        ids = [d.get("id", "") for d in (_req("/models", timeout=3).get("data") or [])]
        pick = (next((i for i in ids if i == base + ",gpu"), None)   # prefer GPU/Metal
                or (base if base in ids else None)
                or next((i for i in ids if i.startswith(base)), base))
    except Exception:
        pass
    _RESOLVED[key] = pick
    return pick


def _req(path: str, payload=None, timeout=120):
    url = endpoint() + path
    headers = {"Authorization": "Bearer local"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(r, timeout=timeout) as resp:        # noqa: S310 (localhost)
        return json.loads(resp.read().decode())


def health(timeout: float = 2.0) -> bool:
    """Is `litert-lm serve` up and serving? Cheap GET /models."""
    try:
        _req("/models", timeout=timeout)
        return True
    except Exception:
        return False


def chat(messages, tools=None, model=None, temperature: float = 0.4, timeout: int = 120) -> dict:
    """One OpenAI chat-completions call against the local server. Returns the raw JSON
    (choices[].message may carry .content and/or .tool_calls)."""
    payload = {"model": _served_model(model), "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    with _INFER_LOCK:                         # one inference at a time (Metal isn't concurrency-safe)
        return _req("/chat/completions", payload, timeout=timeout)


_VISION_SUPPORTED = None        # None=untested, True/False after the first image request


def vision_supported() -> bool | None:
    """Whether the local model accepts IMAGES — many litert-lm Gemma builds are TEXT-ONLY
    (image → 500 'Failed to create conversation'). None until first tried."""
    return _VISION_SUPPORTED


def vision_describe(image_jpg: bytes, prompt: str, model=None, timeout: int = 120) -> str:
    """Multimodal describe via the local model (image + prompt → text). Returns '' if the
    local build is text-only (cached after the first failure, so callers fall back to cloud
    without re-hitting a 500)."""
    global _VISION_SUPPORTED
    if _VISION_SUPPORTED is False:
        return ""
    uri = "data:image/jpeg;base64," + base64.b64encode(image_jpg).decode()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": uri}}]}]
    try:
        r = chat(msgs, model=model, timeout=timeout)
        txt = ((r.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
        if txt:
            _VISION_SUPPORTED = True
        return txt
    except Exception:
        _VISION_SUPPORTED = False    # text-only build → callers fall back to cloud vision
        return ""


# A curated CORE for the local model: sending all ~64 tool schemas to a 12B on every round
# bloats the prompt and tanks latency. These cover the common voice/chat intents (browse,
# try-on, filters, scene, overlays, a little media). Override/expand via STUDIO_LOCAL_TOOLS
# (comma list, or "all"). The cloud backends still get the full set.
LOCAL_CORE_TOOLS = {
    "live_control", "gesture_browse", "shop_search", "try_eyewear", "try_product",
    "virtual_try_on", "apply_face_filter", "analyze_scene", "describe_image", "medical_image",
    "create_overlay", "clear_overlays", "set_render", "set_detection", "nano_banana",
    "generate_video", "set_lens_tint", "snapshot", "go_live",
    # NB: 'plan'/'narrate' deliberately excluded — they add a reasoning round (slow on a
    # local 12B) without acting. The local model acts directly.
}


def _local_tool_names() -> set | None:
    env = os.environ.get("STUDIO_LOCAL_TOOLS")
    if env:
        return None if env.strip().lower() == "all" else {t.strip() for t in env.split(",") if t.strip()}
    return set(LOCAL_CORE_TOOLS)


def chat_tools_schema(only=None) -> list[dict]:
    """Studio tools as OpenAI chat-completions function schemas (reuses the realtime schema
    builder, reshaped to `{type:function, function:{…}}`). `only` = a set of tool names to
    include; None = all of them."""
    from . import tools as T
    from .live_openai import _tool_schema
    out = []
    for fn in T.ALL_TOOLS:
        if only is not None and fn.__name__ not in only:
            continue
        s = _tool_schema(fn)
        out.append({"type": "function", "function": {
            "name": s["name"], "description": s["description"], "parameters": s["parameters"]}})
    return out
