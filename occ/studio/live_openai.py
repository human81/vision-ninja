"""OpenAI Realtime BIDI bridge — the *second* Live Voice backend.

Why: OpenAI's realtime voice is the strongest at non-English accents — notably
**Haitian Creole** — which Gemini Live handles less well. Momentum uses the same
split (Gemini for the canvas, OpenAI `gpt-realtime` for the voice neuron). This
bridge mirrors `live.py` exactly at the *browser* edge — it emits the SAME frame
vocabulary (audio / transcript / tool / fitting / turn_complete / the studio
NDJSON UI frames) — so the studio UI and `/ws/live` route are unchanged; only the
upstream realtime protocol differs.

  browser  <->  our /ws/live (?backend=openai)  <->  wss://api.openai.com/v1/realtime

Differences vs Gemini Live (by design, documented for the demo):
  • OpenAI realtime is audio+text; inbound `video` frames are ignored (the model
    doesn't *see* you). Tools still run on the live pipeline, so overlays/try-on
    land on you exactly as before — only the model's visual awareness is absent.
  • Browser mic is PCM16 @16 kHz; OpenAI wants @24 kHz, so we resample server-side.
  • Server VAD drives turn-taking (no manual commit needed for spoken turns).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
import typing

from .agent import SYSTEM_PROMPT, frames_for, fitting_on
from . import tools as T
from .runtime import ctx

# GA realtime model — best Creole / non-English accents. Override via env.
OPENAI_REALTIME_MODEL = os.environ.get("STUDIO_OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_VOICE = os.environ.get("STUDIO_OPENAI_VOICE", "marin")
OPENAI_TRANSCRIBE = os.environ.get("STUDIO_OPENAI_TRANSCRIBE", "gpt-4o-mini-transcribe")
INPUT_RATE = int(os.environ.get("STUDIO_OPENAI_INPUT_RATE", "24000"))  # OpenAI pcm16 = 24k
_URL = "wss://api.openai.com/v1/realtime?model=" + OPENAI_REALTIME_MODEL

_VOICE_HINT = (
    "\n\nYOU ARE NOW IN LIVE VOICE MODE. Keep spoken replies short and natural — "
    "a sentence or two. Don't read out URLs, code, or coordinates. When the user "
    "asks for a visual on themselves or the scene, CALL THE TOOL (create_overlay, "
    "apply_face_filter, try_eyewear, try_product, set_render, go_live) — your "
    "overlays run on the live camera and appear on screen. Briefly say what you did.\n"
    "LANGUAGE: mirror the user's language. If they speak Haitian Creole (Kreyòl) or "
    "French, reply naturally in that language with an authentic accent.")


# ----------------------------- tool schemas -----------------------------
_PYT = {str: "string", bool: "boolean", int: "integer", float: "number"}


def _json_type(ann):
    """Map a Python annotation to a JSON-schema type (best effort)."""
    if ann in _PYT:
        return _PYT[ann], None
    if ann in (list, tuple):                          # bare `list` annotation
        return "array", "string"
    if ann is dict:
        return "object", None
    origin = typing.get_origin(ann)
    if origin in (list, tuple):
        args = typing.get_args(ann)
        item, _ = _json_type(args[0]) if args else ("string", None)
        return "array", item
    if origin is typing.Union:                       # e.g. str | None
        for a in typing.get_args(ann):
            if a is not type(None):
                return _json_type(a)
    return "string", None


def _tool_schema(fn) -> dict:
    """Build an OpenAI realtime function-tool schema from a Python tool fn."""
    props, required = {}, []
    for nm, p in inspect.signature(fn).parameters.items():
        if nm == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        t, item = _json_type(p.annotation)
        prop = {"type": t}
        if t == "array":
            prop["items"] = {"type": item or "string"}
        props[nm] = prop
        if p.default is inspect._empty:
            required.append(nm)
    doc = (inspect.getdoc(fn) or fn.__name__).strip().split("\n")[0][:300]
    return {"type": "function", "name": fn.__name__, "description": doc,
            "parameters": {"type": "object", "properties": props,
                           "required": required, "additionalProperties": False}}


def tools_payload() -> list[dict]:
    return [_tool_schema(fn) for fn in T.ALL_TOOLS]


# ----------------------------- audio resample -----------------------------
def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-resample mono PCM16 (good enough for speech). Stateless per chunk."""
    if src_rate == dst_rate or not pcm:
        return pcm
    import numpy as np
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if a.size == 0:
        return b""
    n = max(1, int(round(a.size * dst_rate / src_rate)))
    x = np.linspace(0, a.size - 1, n)
    y = np.interp(x, np.arange(a.size), a)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


# ----------------------------- event translation -----------------------------
def event_to_frames(ev: dict, accum: dict) -> list[dict]:
    """PURE translation of one OpenAI realtime server event → browser frames.

    Handles both the beta and GA event names. `accum` carries per-turn transcript
    text per side so the final aggregate doesn't double the streamed deltas
    (same de-dup contract as the Gemini bridge). Tool calls are NOT handled here
    (they need async execution) — see OpenAIRealtimeBridge._handle_event.
    """
    t = ev.get("type", "")
    out: list[dict] = []

    # model audio (b64 pcm16 24k) — forward as-is, browser already plays 24k
    if t in ("response.audio.delta", "response.output_audio.delta"):
        d = ev.get("delta")
        if d:
            out.append({"type": "audio", "data": d})
        return out

    # model transcript — stream deltas only; drop the .done aggregate (dedup)
    if t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
        tx = ev.get("delta") or ""
        if tx:
            accum["model"] = accum.get("model", "") + tx
            out.append({"type": "transcript", "role": "model", "text": tx,
                        "partial": False})
        return out
    if t in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
        return out  # full aggregate — already streamed via deltas

    # user transcript (whisper / gpt-4o-transcribe of the mic)
    if t == "conversation.item.input_audio_transcription.delta":
        tx = ev.get("delta") or ""
        if tx:
            accum["user"] = accum.get("user", "") + tx
            out.append({"type": "transcript", "role": "user", "text": tx,
                        "partial": False})
        return out
    if t == "conversation.item.input_audio_transcription.completed":
        full = (ev.get("transcript") or "").strip()
        streamed = accum.get("user", "")
        if full and full != streamed:          # only the tail we didn't stream
            tail = full[len(streamed):] if full.startswith(streamed) else full
            if tail:
                out.append({"type": "transcript", "role": "user", "text": tail,
                            "partial": False})
        accum["user"] = ""
        return out

    # barge-in: user started talking over the model
    if t == "input_audio_buffer.speech_started":
        out.append({"type": "interrupted"})
        return out

    # turn boundary — but a tool-only response.done is INTERMEDIATE (the follow-up
    # response, created after we return the function output, carries the spoken
    # reply). Suppress turn_complete there so the UI doesn't close the turn early.
    if t == "response.done":
        outputs = (ev.get("response") or {}).get("output") or []
        kinds = {o.get("type") for o in outputs}
        if outputs and kinds <= {"function_call"}:
            return out                      # tool-only; keep the turn open
        accum["user"] = ""
        accum["model"] = ""
        out.append({"type": "turn_complete"})
        return out

    if t == "error":
        err = ev.get("error") or {}
        out.append({"type": "error",
                    "message": f"openai-realtime: {err.get('message', ev)}"})
    return out


class OpenAIRealtimeBridge:
    """One OpenAI Realtime session bridged to one browser websocket."""

    def __init__(self, settings):
        self.settings = settings
        self._oa = None
        self._accum = {"user": "", "model": ""}
        # function-call args stream in deltas keyed by call_id → assemble then run
        self._fc = {}

    async def run(self, ws):
        import websockets
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            await ws.send_text(json.dumps({"type": "error",
                "message": "OpenAI Realtime needs OPENAI_API_KEY in the environment"}))
            return
        headers = {"Authorization": f"Bearer {key}"}   # GA realtime: no beta header
        brain = ctx().brain.prompt_context() if ctx().brain else ""
        instructions = SYSTEM_PROMPT.format(brain=brain) + _VOICE_HINT

        try:
            self._oa = await websockets.connect(
                _URL, additional_headers=headers, max_size=None)
        except TypeError:   # older websockets uses extra_headers=
            self._oa = await websockets.connect(
                _URL, extra_headers=headers, max_size=None)

        async with self._oa as oa:
            # GA realtime session shape (nested audio in/out). The browser mic is
            # 16k; we resample to INPUT_RATE before append, so declare that rate.
            await oa.send(json.dumps({"type": "session.update", "session": {
                "type": "realtime",
                "model": OPENAI_REALTIME_MODEL,
                "instructions": instructions,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": INPUT_RATE},
                        "transcription": {"model": OPENAI_TRANSCRIBE},
                        "turn_detection": {"type": "server_vad", "create_response": True},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": OPENAI_VOICE,
                    },
                },
                "tools": tools_payload(),
                "tool_choice": "auto",
            }}))
            await ws.send_text(json.dumps({"type": "ready",
                                           "model": OPENAI_REALTIME_MODEL,
                                           "backend": "openai"}))

            async def pump_in():
                """browser -> OpenAI."""
                try:
                    while True:
                        raw = await ws.receive_text()
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        mt = msg.get("type")
                        if mt == "audio":
                            b64 = msg.get("data", "")
                            if not b64:
                                continue
                            if INPUT_RATE != 16000:   # browser mic is 16k
                                import base64
                                pcm = resample_pcm16(base64.b64decode(b64), 16000, INPUT_RATE)
                                b64 = base64.b64encode(pcm).decode("ascii")
                            await oa.send(json.dumps({
                                "type": "input_audio_buffer.append", "audio": b64}))
                        elif mt == "text":
                            text = (msg.get("text") or "").strip()
                            if text:
                                await oa.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {"type": "message", "role": "user",
                                             "content": [{"type": "input_text",
                                                          "text": text}]}}))
                                await oa.send(json.dumps({"type": "response.create"}))
                        elif mt == "video":
                            continue   # OpenAI realtime: audio+text only — ignore
                        elif mt in ("end", "close"):
                            break
                except Exception:
                    pass

            async def pump_out():
                """OpenAI -> browser."""
                try:
                    async for raw in oa:
                        try:
                            ev = json.loads(raw)
                        except Exception:
                            continue
                        await self._handle_event(ws, oa, ev)
                except Exception as e:
                    await self._safe_send(ws, {"type": "error",
                        "message": f"{type(e).__name__}: {e}"})

            task_in = asyncio.create_task(pump_in())
            task_out = asyncio.create_task(pump_out())
            try:
                await asyncio.wait({task_in, task_out},
                                   return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in (task_in, task_out):
                    t.cancel()

    async def _handle_event(self, ws, oa, ev):
        # pure transcript/audio/turn frames
        for f in event_to_frames(ev, self._accum):
            await self._safe_send(ws, f)

        t = ev.get("type", "")
        # assemble streamed function-call arguments
        if t == "response.function_call_arguments.delta":
            cid = ev.get("call_id") or ev.get("item_id") or ""
            self._fc.setdefault(cid, {"name": ev.get("name", ""), "args": ""})
            self._fc[cid]["args"] += ev.get("delta", "") or ""
            if ev.get("name"):
                self._fc[cid]["name"] = ev["name"]
        elif t == "response.function_call_arguments.done":
            cid = ev.get("call_id") or ev.get("item_id") or ""
            name = ev.get("name") or self._fc.get(cid, {}).get("name", "")
            raw = ev.get("arguments") or self._fc.get(cid, {}).get("args", "") or "{}"
            self._fc.pop(cid, None)
            await self._run_tool(ws, oa, cid, name, raw)

    async def _run_tool(self, ws, oa, call_id, name, raw_args):
        """Execute a studio tool off the event loop, stream UI frames, return the
        result to OpenAI, and continue the response — mirrors the Gemini path."""
        if not name:
            return
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except Exception:
            args = {}
        f = fitting_on(name)
        if f:
            await self._safe_send(ws, f)
        await self._safe_send(ws, {"type": "log", "data": {"text": f"{name} …"}})

        fn = T.TOOLS_BY_NAME.get(name)
        if fn is None:
            res = {"status": "error", "error": f"unknown tool {name}"}
        else:
            loop = asyncio.get_event_loop()
            try:
                res = await loop.run_in_executor(
                    None, functools.partial(fn, **args))
            except Exception as e:
                res = {"status": "error", "error": f"{type(e).__name__}: {e}"}

        for fr in frames_for(name, res):
            await self._safe_send(ws, fr)

        # hand the result back to the model and let it keep talking
        try:
            await oa.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call_id,
                         "output": json.dumps(res)[:6000]}}))
            await oa.send(json.dumps({"type": "response.create"}))
        except Exception:
            pass

    @staticmethod
    async def _safe_send(ws, obj):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            pass
