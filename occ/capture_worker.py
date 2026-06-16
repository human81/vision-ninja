"""Isolated capture worker — runs a video capture in a SEPARATE PROCESS.

Reading network streams (esp. YouTube live HLS) through OpenCV/FFMPEG can crash
NATIVELY (SIGSEGV inside libavutil av_log on CDN host-rotation). A native crash
in-process takes down the whole studio server and Python can't catch it. So we
run the capture here, in a child process, and stream frames back to the parent
over stdout. If FFMPEG segfaults, only THIS process dies; the parent respawns it.

Wire protocol on stdout (binary):
    [4 bytes  float32 LE]            fps header (once, first)
    repeated:
      [4 bytes uint32 LE]  length    JPEG byte length
      [length bytes]                 JPEG-encoded BGR frame

Run:  python -m occ.capture_worker <uri> [max_long_side] [jpeg_quality]
"""

from __future__ import annotations

import os
import struct
import sys
import time

# Quiet FFMPEG before cv2 loads it (the native fault was in the verbose av_log
# path). AV_LOG_FATAL.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _downscale(frame: np.ndarray, max_long_side: int) -> np.ndarray:
    if not max_long_side:
        return frame
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return frame
    s = max_long_side / long_side
    return cv2.resize(frame, (round(w * s), round(h * s)),
                      interpolation=cv2.INTER_AREA)


def _open(uri: str) -> cv2.VideoCapture:
    if uri.startswith(("http", "rtsp")):
        # No HTTP keep-alive (YouTube live rotates CDN hosts between segments —
        # the reuse path is what segfaulted); let FFMPEG reconnect itself.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "http_persistent;0|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5")
        return cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(uri)


def main() -> int:
    uri = sys.argv[1]
    max_long = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else 85
    out = sys.stdout.buffer

    cap = _open(uri)
    if not cap.isOpened():
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps < 121.0):
        fps = 30.0
    try:
        out.write(struct.pack("<f", float(fps)))
        out.flush()
    except (BrokenPipeError, OSError):
        return 0

    enc = [cv2.IMWRITE_JPEG_QUALITY, quality]
    backoff = 0.5
    while True:
        ok, frame = cap.read()
        if not ok:
            # EOF / stream hiccup → reconnect (live) or stop (finite + no loop).
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
            try:
                cap.release()
            except Exception:
                pass
            cap = _open(uri)
            if not cap.isOpened():
                continue
            continue
        backoff = 0.5
        if max_long:
            frame = _downscale(frame, max_long)
        ok, buf = cv2.imencode(".jpg", frame, enc)
        if not ok:
            continue
        data = buf.tobytes()
        try:
            out.write(struct.pack("<I", len(data)))
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError):
            break  # parent went away
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
