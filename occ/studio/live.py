"""Gemini Live BIDI bridge — voice conversation + tool-calling over a websocket.

Mirrors Momentum's run_live pattern (python_service/routers/live.py): an ADK
`Runner.run_live` session in StreamingMode.BIDI with response_modalities=["AUDIO"]
and the SAME studio tool registry the text agent uses. So while you talk, the
ninja can call create_overlay / set_render / use_source / analyze_scene … and the
result is spoken back AND applied to the live pipeline (overlays land on you).

Wire protocol (JSON text frames over the websocket):
  browser -> server: {"type":"audio","data":<b64 pcm16 16k mono>}
                     {"type":"video","data":<b64 jpeg>}     # so the model sees you
                     {"type":"text","text":"..."}            # typed turn (also for tests)
                     {"type":"end"} / {"type":"ping"}
  server -> browser: {"type":"ready"}
                     {"type":"audio","data":<b64 pcm16 24k>} # model voice
                     {"type":"transcript","role":"user|model","text":"...","partial":bool}
                     {"type":"tool","phase":"start|end","name":"..."}
                     plus the studio's NDJSON UI frames (refresh/canvas/ui_action/...)
                     {"type":"interrupted"} / {"type":"turn_complete"} / {"type":"error",...}
"""

from __future__ import annotations

import asyncio
import base64
import json
import os

from .agent import SYSTEM_PROMPT, frames_for
from . import tools as T
from .runtime import ctx

# Tool-capable Gemini Live model (Gemini API). gemini-3.1-flash-live-preview is
# the cascade model that reliably function-calls mid-session (Momentum's default).
# Override via env; native-audio models sound better but tool-call less reliably.
LIVE_MODEL = os.environ.get("STUDIO_LIVE_MODEL", "gemini-3.1-flash-live-preview")

_VOICE_HINT = (
    "\n\nYOU ARE NOW IN LIVE VOICE MODE. Keep spoken replies short and natural — "
    "a sentence or two. Don't read out URLs, code, or coordinates. When the user "
    "asks for a visual on themselves or the scene, CALL THE TOOL (create_overlay, "
    "set_render, set_task, use_source, analyze_scene) — your overlays run on the "
    "live camera and appear on screen. Briefly say what you did.")


class LiveBridge:
    """One Gemini Live session bridged to one browser websocket."""

    def __init__(self, settings):
        self.settings = settings
        self._queue = None

    async def run(self, ws):
        """Drive a full BIDI session until the socket closes."""
        from google.adk.agents import Agent
        from google.adk.runners import InMemoryRunner
        from google.adk.agents.run_config import RunConfig, StreamingMode
        from google.adk.agents.live_request_queue import LiveRequestQueue
        from google.genai import types

        brain = ctx().brain.prompt_context() if ctx().brain else ""
        agent = Agent(name="vision_ninja_live", model=LIVE_MODEL,
                      instruction=SYSTEM_PROMPT.format(brain=brain) + _VOICE_HINT,
                      tools=list(T.ALL_TOOLS))
        runner = InMemoryRunner(agent=agent, app_name="studio-live")
        user_id, session_id = "local", "studio-live-session"
        await runner.session_service.create_session(
            app_name="studio-live", user_id=user_id, session_id=session_id)

        queue = LiveRequestQueue()
        self._queue = queue
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
        )

        await ws.send_text(json.dumps({"type": "ready", "model": LIVE_MODEL}))

        async def pump_in():
            """browser -> Gemini Live queue."""
            try:
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    mt = msg.get("type")
                    if mt == "audio":
                        data = base64.b64decode(msg.get("data", ""))
                        if data:
                            queue.send_realtime(types.Blob(
                                mime_type="audio/pcm;rate=16000", data=data))
                    elif mt == "video":
                        data = base64.b64decode(msg.get("data", ""))
                        if data:
                            queue.send_realtime(types.Blob(
                                mime_type="image/jpeg", data=data))
                    elif mt == "text":
                        text = (msg.get("text") or "").strip()
                        if text:
                            queue.send_content(types.Content(
                                role="user", parts=[types.Part.from_text(text=text)]))
                    elif mt in ("end", "close"):
                        break
                    # ping → ignore (keepalive)
            except Exception:
                pass
            finally:
                queue.close()

        async def pump_out():
            """Gemini Live events -> browser."""
            try:
                async for event in runner.run_live(
                        user_id=user_id, session_id=session_id,
                        live_request_queue=queue, run_config=run_config):
                    await self._emit_event(ws, event)
            except Exception as e:
                await self._safe_send(ws, {"type": "error",
                                           "message": f"{type(e).__name__}: {e}"})

        task_in = asyncio.create_task(pump_in())
        task_out = asyncio.create_task(pump_out())
        try:
            await asyncio.wait({task_in, task_out},
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            queue.close()
            for t in (task_in, task_out):
                t.cancel()

    async def _emit_event(self, ws, event):
        """Translate one ADK live Event into browser frames."""
        # 1) model audio (inline pcm 24k) + any text parts (transcript)
        content = getattr(event, "content", None)
        if content and getattr(content, "parts", None):
            role = getattr(content, "role", "model") or "model"
            for p in content.parts:
                inline = getattr(p, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    mime = getattr(inline, "mime_type", "") or ""
                    if mime.startswith("audio"):
                        await self._safe_send(ws, {
                            "type": "audio",
                            "data": base64.b64encode(inline.data).decode("ascii")})
                txt = getattr(p, "text", None)
                if txt:
                    await self._safe_send(ws, {
                        "type": "transcript", "role": role, "text": txt,
                        "partial": bool(getattr(event, "partial", False))})

        # 2) explicit input/output transcriptions (what each side said)
        for attr, role in (("input_transcription", "user"),
                           ("output_transcription", "model")):
            tr = getattr(event, attr, None)
            tx = getattr(tr, "text", None) if tr is not None else None
            if tx:
                await self._safe_send(ws, {"type": "transcript", "role": role,
                                           "text": tx, "partial": False})

        # 3) tool calls (spoken request -> studio tool). Reuse the SAME NDJSON
        # vocabulary the text agent emits (log / tool / refresh / canvas / …) so
        # the browser routes them through the existing handleFrame().
        for fc in event.get_function_calls() or []:
            if fc.name:
                from .agent import fitting_on
                f = fitting_on(fc.name)
                if f:
                    await self._safe_send(ws, f)
                await self._safe_send(ws, {"type": "log",
                                           "data": {"text": f"{fc.name} …"}})
        for fr in event.get_function_responses() or []:
            if not fr.name:
                continue
            for f in frames_for(fr.name, fr.response):
                await self._safe_send(ws, f)

        # 4) turn boundaries / barge-in
        if getattr(event, "interrupted", False):
            await self._safe_send(ws, {"type": "interrupted"})
        if getattr(event, "turn_complete", False):
            await self._safe_send(ws, {"type": "turn_complete"})

    @staticmethod
    async def _safe_send(ws, obj):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            pass
