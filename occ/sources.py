"""Frame sources: file, RTSP, and webcam, behind one interface.

The real-time win is the **newest-frame-wins** capture for live sources: a
background thread drains the decoder as fast as it arrives and keeps only the
latest frame, so network/CPU jitter never accumulates as latency. File sources
read sequentially (optionally looping).

Optional downscale (`max_long_side`) is the cheapest accuracy-preserving speedup
for 4K feeds.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Iterator

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
    """Sequential video-file reader, optionally looping."""

    def __init__(self, uri: str, loop: bool = True, max_long_side: int = 0):
        self.uri = uri
        self.loop = loop
        self.max_long_side = max_long_side
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {uri}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.is_live = False

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
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

        if str(uri).startswith("rtsp"):
            # must be set before VideoCapture is created
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;{rtsp_transport}")
        self.cap = self._open()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        src: int | str = int(self.uri) if str(self.uri).isdigit() else self.uri
        cap = cv2.VideoCapture(src)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize internal buffering
        except Exception:
            pass
        if not cap.isOpened():
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
        # wait for first frame
        while self._latest is None and not self._stop.is_set():
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


def open_source(cfg):
    """Pick the right source from the config URI."""
    s = cfg.section("source")
    uri = str(s.get("uri"))
    max_long = int(s.get("max_long_side", 0) or 0)
    is_stream = uri.startswith("rtsp") or uri.startswith("http") or uri.isdigit()
    if is_stream:
        return StreamSource(
            uri,
            rtsp_transport=s.get("rtsp_transport", "tcp"),
            reconnect=bool(s.get("reconnect", True)),
            max_long_side=max_long,
        )
    return FileSource(uri, loop=bool(s.get("loop", True)), max_long_side=max_long)
