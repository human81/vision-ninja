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
      python -m occ.capture_worker --ytdlp <youtube page url> [max_long_side]

`--ytdlp` (YouTube): OpenCV's FFMPEG HLS reader stalls on YouTube live after ~30-40s — it
reads the segments in the first playlist and then stops picking up new ones (measured: 0
frames/5s with the worker alive). yt-dlp's own live downloader keeps refreshing the playlist
(measured: a steady 30 fps for 90s), so this mode pipes `yt-dlp -o -` into the `ffmpeg` CLI,
which emits JPEG frames at a fixed 30 fps that are passed straight through.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
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


YTDLP_FPS = 30.0


def _exit_with_parent(children: list):
    """If the studio dies, don't linger as an orphan still downloading video: exit (and take
    yt-dlp/ffmpeg along). A blocked read never notices the closed pipe, so poll the parent."""
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                for c in children:
                    try:
                        c.kill()
                    except Exception:
                        pass
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def _ytdlp_cmd(page_url: str) -> list[str]:
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-part",
           "-f", "best[height<=720]/best", "--remote-components", "ejs:github"]
    # Browser cookies get past YouTube's "confirm you're not a bot" (as resolve_youtube does).
    # macOS dev box: Chrome by default. Servers have no browser profile → none.
    browser = os.environ.get("STUDIO_YTDLP_COOKIES_FROM",
                             "chrome" if sys.platform == "darwin" else "")
    if browser:
        cmd += ["--cookies-from-browser", browser]
    return cmd + ["-o", "-", "--", page_url]


def main_ytdlp(page_url: str, max_long: int) -> int:
    out = sys.stdout.buffer
    vf = f"fps={YTDLP_FPS:g}"
    if max_long:
        vf += (f",scale='if(gte(iw,ih),min(iw,{max_long}),-2)'"
               f":'if(gte(iw,ih),-2,min(ih,{max_long}))'")
    ytdlp = subprocess.Popen(_ytdlp_cmd(page_url), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                           "-an", "-vf", vf, "-f", "image2pipe", "-c:v", "mjpeg",
                           "-q:v", "5", "pipe:1"],
                          stdin=ytdlp.stdout, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    ytdlp.stdout.close()                  # ffmpeg owns the pipe now
    _exit_with_parent([ytdlp, ff])
    try:
        out.write(struct.pack("<f", YTDLP_FPS))
        out.flush()
        buf = b""
        while True:
            chunk = ff.stdout.read(1 << 16)
            if not chunk:
                return 3                   # yt-dlp/ffmpeg ended → parent respawns
            buf += chunk
            while True:                    # ffmpeg's MJPEG stream = back-to-back JPEGs
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if end < 0:
                    if start > 0:
                        buf = buf[start:]
                    break
                jpg, buf = buf[start:end + 2], buf[end + 2:]
                out.write(struct.pack("<I", len(jpg)))
                out.write(jpg)
                out.flush()
    except (BrokenPipeError, OSError):
        return 0                           # parent went away
    finally:
        for c in (ff, ytdlp):
            try:
                c.kill()
            except Exception:
                pass


def main() -> int:
    if sys.argv[1] == "--ytdlp":
        return main_ytdlp(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    uri = sys.argv[1]
    max_long = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else 85
    out = sys.stdout.buffer
    _exit_with_parent([])

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
