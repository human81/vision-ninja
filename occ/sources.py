"""Frame sources: file, RTSP, and webcam, behind one interface.

The real-time win is the **newest-frame-wins** capture for live sources: a
background thread drains the decoder as fast as it arrives and keeps only the
latest frame, so network/CPU jitter never accumulates as latency. File sources
read sequentially (optionally looping).

Optional downscale (`max_long_side`) is the cheapest accuracy-preserving speedup
for 4K feeds.
"""

from __future__ import annotations

import collections
import os
import struct
import subprocess
import sys
import threading
import time
from typing import Iterator

# Quiet FFMPEG's logger before cv2 loads it — a verbose av_log path has
# segfaulted while reading YouTube live HLS on CDN host-rotation. (AV_LOG_FATAL)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

import cv2
import numpy as np


def _maybe_downscale(frame: np.ndarray, max_long_side: int) -> np.ndarray:
    if not max_long_side:
        return frame
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return frame
    s = max_long_side / long_side
    return cv2.resize(frame, (round(w * s), round(h * s)),
                      interpolation=cv2.INTER_AREA)


class FileSource:
    """Sequential video reader, optionally looping. Reads EVERY frame in order —
    used for local files, finite http VODs, AND live HLS (whose frames arrive in
    per-segment bursts that newest-frame-wins would discard down to ~1 fps)."""

    def __init__(self, uri: str, loop: bool = True, max_long_side: int = 0):
        self.uri = uri
        self.loop = loop
        self.max_long_side = max_long_side
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {uri}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.is_live = False

    def _open(self) -> cv2.VideoCapture:
        # remote URLs (http HLS/VOD) need the FFMPEG backend; local paths/webcams
        # use the platform default.
        if str(self.uri).startswith(("http", "rtsp")):
            return cv2.VideoCapture(str(self.uri), cv2.CAP_FFMPEG)
        return cv2.VideoCapture(self.uri)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if ok:
                yield _maybe_downscale(frame, self.max_long_side)
                continue
            if not self.loop:
                break
            # EOF — rewind. Local files seek to 0; a non-seekable http VOD
            # (e.g. a resolved YouTube progressive URL) needs a fresh open.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                # non-seekable (http VOD) or a live-stream hiccup → reopen.
                self.cap.release()
                self.cap = self._open()
                if not self.cap.isOpened():
                    break
                ok, frame = self.cap.read()
                if not ok:
                    break
            yield _maybe_downscale(frame, self.max_long_side)

    def release(self):
        self.cap.release()


class StreamSource:
    """Newest-frame-wins reader for RTSP / webcam. A grabber thread always holds
    the latest frame; `frames()` yields whatever is current, dropping stale ones."""

    def __init__(self, uri: str, rtsp_transport: str = "tcp",
                 reconnect: bool = True, max_long_side: int = 0):
        self.uri = uri
        self.rtsp_transport = rtsp_transport
        self.reconnect = reconnect
        self.max_long_side = max_long_side
        self.is_live = True

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._stop = threading.Event()

        self.cap = self._open()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        """Open the stream FAIL-FAST. An unreachable/bad RTSP or http URL must never
        block the caller indefinitely (that freezes the whole pipeline), so we bound
        the FFMPEG open + read with timeouts before constructing the capture."""
        uri = str(self.uri)
        if uri.startswith(("rtsp", "http")):
            # microsecond FFMPEG timeouts — set BEFORE VideoCapture is created.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.rtsp_transport}|stimeout;5000000|timeout;8000000")
            cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        else:
            src: int | str = int(uri) if uri.isdigit() else uri
            cap = cv2.VideoCapture(src)
        for prop, val in ((cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000),
                          (cv2.CAP_PROP_READ_TIMEOUT_MSEC, 6000),
                          (cv2.CAP_PROP_BUFFERSIZE, 1)):
            try:
                cap.set(prop, val)
            except Exception:
                pass
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open stream: {self.uri}")
        return cap

    def _grab_loop(self):
        backoff = 0.5
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if not self.reconnect:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                try:
                    self.cap.release()
                    self.cap = self._open()
                    backoff = 0.5
                except Exception:
                    continue
                continue
            backoff = 0.5
            frame = _maybe_downscale(frame, self.max_long_side)
            with self._lock:
                self._latest = frame
                self._seq += 1

    def frames(self) -> Iterator[np.ndarray]:
        last_seq = -1
        # wait for first frame — but BOUNDED, so a stream that opens yet never
        # delivers (dead URL) can't hang the pipeline thread forever.
        t0 = time.time()
        while self._latest is None and not self._stop.is_set():
            if time.time() - t0 > 10.0:
                return
            time.sleep(0.01)
        while not self._stop.is_set():
            with self._lock:
                seq, frame = self._seq, self._latest
            if frame is None:
                break
            if seq != last_seq:          # only yield fresh frames
                last_seq = seq
                yield frame
            else:
                time.sleep(0.002)        # nothing new yet; yield CPU

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class SubprocessStreamSource:
    """Jitter-buffered, PROCESS-ISOLATED reader for HLS live (YouTube live, .m3u8).

    Two problems with reading live HLS via OpenCV:
      1. HLS delivers a whole *segment* of frames in a sub-second burst, then the
         decoder BLOCKS for the segment duration (~2-6s). Naive/newest-frame-wins
         readers freeze for seconds between bursts.
      2. The FFMPEG decoder can SEGFAULT natively (av_log on CDN host-rotation),
         which would take down the whole studio — Python can't catch a SIGSEGV.

    So the capture runs in a child process (`occ.capture_worker`) that streams
    JPEG frames back over a pipe. A reader thread here decodes them into a bounded
    FIFO; `frames()` releases them PACED at the stream fps after a short prebuffer
    (the burst fills the buffer, pacing drains it across the gap → smooth). If the
    worker dies (crash or EOF), the reader RESPAWNS it with backoff — the studio
    never goes down. Old frames drop (bounded deque) so latency stays bounded.
    """

    def __init__(self, uri: str, max_long_side: int = 0,
                 buffer_seconds: float = 14.0, prebuffer_seconds: float = 3.0,
                 reconnect: bool = True):
        self.uri = uri
        self.max_long_side = max_long_side
        self.reconnect = reconnect
        self.fps = 30.0
        self.is_live = True
        self._buf = collections.deque(maxlen=max(30, int(30.0 * buffer_seconds)))
        self._prebuffer_secs = prebuffer_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-m", "occ.capture_worker",
             str(self.uri), str(self.max_long_side)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    @staticmethod
    def _readn(pipe, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = pipe.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        return bytes(buf)

    def _reader_loop(self):
        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._proc = self._spawn()
            except Exception:
                time.sleep(backoff); backoff = min(backoff * 2, 5.0); continue
            try:
                self._consume(self._proc.stdout)
            except Exception:
                pass
            # worker exited — crash or EOF. Tear it down and (maybe) respawn.
            try:
                self._proc.kill()
            except Exception:
                pass
            if self._stop.is_set() or not self.reconnect:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)

    def _consume(self, pipe):
        hdr = self._readn(pipe, 4)
        if len(hdr) == 4:
            fps = struct.unpack("<f", hdr)[0]
            if 1.0 < fps < 121.0:
                self.fps = fps
        while not self._stop.is_set():
            lb = self._readn(pipe, 4)
            if len(lb) < 4:
                break                                   # EOF / worker died
            (n,) = struct.unpack("<I", lb)
            if n <= 0 or n > 64_000_000:
                break
            data = self._readn(pipe, n)
            if len(data) < n:
                break
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            with self._lock:
                self._buf.append(frame)

    def frames(self) -> Iterator[np.ndarray]:
        # build an initial cushion so the first segment-gap doesn't underrun.
        prebuffer = max(1, int(self.fps * self._prebuffer_secs))
        t0 = time.time()
        while not self._stop.is_set():
            with self._lock:
                have = len(self._buf)
            if have >= prebuffer or time.time() - t0 > 15.0:
                break
            time.sleep(0.03)
        dt = 1.0 / (self.fps or 30.0)
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                frame = self._buf.popleft() if self._buf else None
            if frame is None:                 # underrun — wait for the next burst
                time.sleep(0.01)
                next_t = time.perf_counter()
                continue
            yield frame
            # pace to real time so the buffer drains across the segment gap
            # instead of racing to the live edge and re-freezing.
            next_t += dt
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(min(slack, 0.5))
            else:
                next_t = time.perf_counter()

    def release(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._thread.join(timeout=1.0)


class _PushBuffer:
    """Single newest-frame-wins slot for browser-pushed frames (the BIDI live
    mode sends webcam/screen frames here over a websocket). Decoupled from the
    pipeline so source rebuilds don't lose the feed."""

    def __init__(self):
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray):
        with self._lock:
            self._latest = frame
            self._seq += 1

    def get(self):
        with self._lock:
            return self._seq, self._latest

    def clear(self):
        with self._lock:
            self._latest = None
            self._seq += 1


# one live push session at a time
PUSH = _PushBuffer()


def push_frame(frame: np.ndarray):
    """Hand a BGR frame from the browser-capture websocket to the pipeline."""
    PUSH.put(frame)


class PushSource:
    """Newest-frame-wins reader over frames pushed from the browser (webcam or
    screen share) via the live websocket. The pipeline then detects/tracks/runs
    the agent's overlays and republishes to /stream.mjpg — so overlays appear on
    *you*. If no frame has arrived yet, it idles (pipeline shows its placeholder)
    rather than blocking."""

    def __init__(self, max_long_side: int = 0):
        self.max_long_side = max_long_side
        self.fps = 30.0
        self.is_live = True
        self._stop = threading.Event()

    def frames(self) -> Iterator[np.ndarray]:
        last_seq = -1
        while not self._stop.is_set():
            seq, frame = PUSH.get()
            if frame is not None and seq != last_seq:
                last_seq = seq
                yield _maybe_downscale(frame, self.max_long_side)
            else:
                time.sleep(0.005)        # nothing new yet; yield CPU

    def release(self):
        self._stop.set()


def open_source(cfg):
    """Pick the right source from the config URI."""
    s = cfg.section("source")
    uri = str(s.get("uri"))
    max_long = int(s.get("max_long_side", 0) or 0)
    low = uri.lower()
    # Browser-pushed frames (BIDI live: webcam / screen share over the live WS).
    if low.startswith("push:"):
        return PushSource(max_long_side=max_long)
    # RTSP cameras & webcams deliver ONE real-time frame at a time, so
    # newest-frame-wins keeps latency low without dropping anything meaningful.
    rtsp_or_cam = low.startswith("rtsp") or uri.isdigit()
    if rtsp_or_cam or bool(s.get("rtsp", False)):
        return StreamSource(
            uri,
            rtsp_transport=s.get("rtsp_transport", "tcp"),
            reconnect=bool(s.get("reconnect", True)),
            max_long_side=max_long,
        )
    # Live HLS (YouTube live, .m3u8) delivers frames in per-segment BURSTS with
    # multi-second blocking gaps, AND its FFMPEG decoder can segfault natively on
    # CDN host-rotation. Run it in an ISOLATED subprocess (so a decoder crash only
    # restarts the feed, not the studio) with a jitter buffer (prebuffer + paced
    # release hides the segment cadence).
    is_hls = (".m3u8" in low or "hls_playlist" in low
              or "/manifest/" in low or "manifest.googlevideo" in low)
    if is_hls:
        return SubprocessStreamSource(uri, max_long_side=max_long,
                                      reconnect=bool(s.get("reconnect", True)))
    # Local files / finite http VODs (resolved YouTube *progressive* URLs) read
    # sequentially, looping. FileSource reopens the capture on read-failure.
    return FileSource(uri, loop=bool(s.get("loop", True)), max_long_side=max_long)
