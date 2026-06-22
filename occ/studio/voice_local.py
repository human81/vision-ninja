"""FREE local voice — a 3rd Live backend that runs entirely on this Mac, $0.

A turn-based loop (not native duplex): server-side VAD buffers your mic, Whisper
transcribes it locally (faster-whisper), the LOCAL Gemma brain reasons + calls tools
(via local_llm → litert-lm), and the browser speaks the reply with its own on-device
speech synthesis. No cloud, no per-minute audio billing.

Trade-off vs. Gemini Live / OpenAI Realtime: higher latency and no native barge-in —
it's turn by turn (speak, pause, it answers). Same WS protocol as the other bridges
(see live.py) so the frontend barely changes; the only addition is a `{"type":"speak"}`
frame telling the browser to TTS the reply.

Bring up speech: `uv pip install -e ".[voice]"` (faster-whisper) + the local Gemma
endpoint (`litert-lm serve`). Without faster-whisper, typed turns still work.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os

import numpy as np

from .neurons import NEURONS  # noqa: F401  (keep import graph warm / parity with live.py)

_SR = 16000                       # mic PCM16 sample rate (matches the frontend worklet)
_SPEECH_RMS = float(os.environ.get("STUDIO_VOICE_RMS", "480"))   # int16 RMS = "speech"
_SILENCE_MS = int(os.environ.get("STUDIO_VOICE_SILENCE_MS", "750"))   # trailing silence → end of turn
_MIN_SPEECH_MS = 250              # ignore blips shorter than this
_MAX_TURN_MS = 15000              # hard cap on one utterance

_WHISPER = None
_WHISPER_OFF = False


def _whisper():
    """Lazy faster-whisper model (CPU int8 — fast + light on Apple Silicon)."""
    global _WHISPER, _WHISPER_OFF
    if _WHISPER is not None or _WHISPER_OFF:
        return _WHISPER
    try:
        from faster_whisper import WhisperModel
        name = os.environ.get("STUDIO_WHISPER_MODEL", "base.en")
        _WHISPER = WhisperModel(name, device="cpu", compute_type="int8")
    except Exception:
        _WHISPER_OFF = True
        _WHISPER = None
    return _WHISPER


def transcribe(pcm16: bytes) -> str:
    """Local STT: PCM16 mono 16k bytes → text. '' if unavailable or empty."""
    m = _whisper()
    if m is None or not pcm16:
        return ""
    audio = np.frombuffer(pcm16, np.int16).astype(np.float32) / 32768.0
    if audio.size < _SR * _MIN_SPEECH_MS // 1000:
        return ""
    try:
        segs, _ = m.transcribe(audio, language="en", vad_filter=True, beam_size=1)
        return " ".join(s.text.strip() for s in segs).strip()
    except Exception:
        return ""


def _ctx():
    """The studio context, or None if it isn't initialised (tests / headless)."""
    try:
        from .runtime import ctx
        return ctx()
    except Exception:
        return None


def _rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    a = np.frombuffer(pcm16, np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


class LocalVoiceBridge:
    """Turn-based local voice. Same WS contract as LiveBridge; reasoning via local Gemma."""

    def __init__(self, settings):
        self.settings = settings
        self.history: list[dict] = []
        self.buf = bytearray()
        self.had_speech = False
        self.silence_bytes = 0
        self.busy = False
        self._warned_stt = False

    async def _send(self, ws, obj):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            pass

    async def run(self, ws):
        await self._send(ws, {"type": "ready", "model": "local · whisper+gemma", "local": True})
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                mt = msg.get("type")
                if mt in ("end", "close"):
                    break
                if mt == "ping":
                    continue
                if mt == "video":
                    continue                          # vision is available via tools, not the live image
                if mt == "text" and (msg.get("text") or "").strip():
                    await self._turn(ws, msg["text"].strip())
                    continue
                if mt == "commit":
                    await self._process(ws)
                    continue
                if mt == "audio" and not self.busy:
                    pcm = base64.b64decode(msg.get("data", "") or "")
                    if not pcm:
                        continue
                    if self._whisper_missing(ws):
                        continue
                    self.buf += pcm
                    if _rms(pcm) >= _SPEECH_RMS:
                        self.had_speech = True
                        self.silence_bytes = 0
                    elif self.had_speech:
                        self.silence_bytes += len(pcm)
                    sil_ms = (self.silence_bytes / 2) / _SR * 1000
                    turn_ms = (len(self.buf) / 2) / _SR * 1000
                    if self.had_speech and (sil_ms >= _SILENCE_MS or turn_ms >= _MAX_TURN_MS):
                        await self._process(ws)
        except Exception as e:
            await self._send(ws, {"type": "error", "message": f"local voice: {e}"})

    def _whisper_missing(self, ws) -> bool:
        if _whisper() is not None:
            return False
        if not self._warned_stt:
            self._warned_stt = True
            asyncio.ensure_future(self._send(ws, {"type": "error", "message":
                "local speech needs faster-whisper — run: uv pip install -e \".[voice]\" "
                "(you can still TYPE to the ninja)"}))
        return True

    async def _process(self, ws):
        """End of a spoken turn → transcribe → reason."""
        if not self.buf:
            return
        pcm = bytes(self.buf)
        self.buf = bytearray(); self.had_speech = False; self.silence_bytes = 0
        self.busy = True
        try:
            text = await asyncio.get_event_loop().run_in_executor(None, lambda: transcribe(pcm))
            if text:
                await self._turn(ws, text)
            else:
                await self._send(ws, {"type": "turn_complete"})
        finally:
            self.busy = False

    async def _turn(self, ws, user_text: str):
        """One reasoning turn on the LOCAL Gemma brain: tools + a spoken reply."""
        from . import local_llm as L
        from .agent import _local_system, frames_for, fitting_on
        from . import tools as T
        c = _ctx()
        await self._send(ws, {"type": "transcript", "role": "user", "text": user_text})
        if not L.health():
            msg = ("The local Gemma brain isn't running — start it with `litert-lm serve` "
                   "(or switch to a Gemini voice).")
            await self._send(ws, {"type": "transcript", "role": "model", "text": msg})
            await self._send(ws, {"type": "speak", "text": msg})
            await self._send(ws, {"type": "turn_complete"})
            return
        loop = asyncio.get_event_loop()
        model = self.settings.model_for("agent")
        if not L.is_local_model(model):
            model = "gemma-4-12b-it"
        brain = c.brain.prompt_context() if (c and c.brain) else ""
        msgs = ([{"role": "system", "content": _local_system(brain)}]   # compact → fits litert context
                + self.history[-4:] + [{"role": "user", "content": user_text}])
        tools = L.chat_tools_schema(L._local_tool_names())   # curated core → fast on a local 12B
        reply = ""
        try:
            for _round in range(5):
                resp = await loop.run_in_executor(None, lambda: L.chat(msgs, tools=tools, model=model))
                m = ((resp.get("choices") or [{}])[0]).get("message", {}) or {}
                if c and c.ledger:
                    u = resp.get("usage") or {}
                    c.ledger.record("agent_brain", model=model,
                                        input_tokens=int(u.get("prompt_tokens", 0) or 0),
                                        output_tokens=int(u.get("completion_tokens", 0) or 0),
                                        label="local voice")
                tcs = m.get("tool_calls") or []
                if tcs:
                    msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
                    for tc in tcs:
                        fnc = tc.get("function", {}) or {}
                        name = fnc.get("name", "")
                        try:
                            args = json.loads(fnc.get("arguments") or "{}")
                        except Exception:
                            args = {}
                        await self._send(ws, {"type": "tool", "data": {"text": name}, "name": name})
                        f = fitting_on(name)
                        if f:
                            await self._send(ws, f)
                        if name in T.TOOLS_BY_NAME:
                            try:
                                res = await loop.run_in_executor(None, lambda: T.TOOLS_BY_NAME[name](**args))
                            except Exception as e:
                                res = {"status": "error", "error": f"{type(e).__name__}: {e}"}
                        else:
                            res = {"status": "error", "error": "unknown tool"}
                        for fr in frames_for(name, res):
                            await self._send(ws, fr)
                        msgs.append({"role": "tool", "tool_call_id": tc.get("id", name),
                                     "content": json.dumps(res, default=str)[:4000]})
                    continue
                reply = m.get("content") or ""
                break
        except Exception as e:
            reply = f"(local brain error: {e})"
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-10:]
        if reply:
            await self._send(ws, {"type": "transcript", "role": "model", "text": reply})
            await self._send(ws, {"type": "speak", "text": reply})
        await self._send(ws, {"type": "turn_complete"})
