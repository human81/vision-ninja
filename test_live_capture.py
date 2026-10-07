"""Live-stream capture resilience (occ/sources.py SubprocessStreamSource) — offline.

YouTube Live froze for good in two ways: the capture worker could hang inside FFMPEG
without exiting (the parent's blocking pipe read then waited forever), and restarts
reopened the same short-lived HLS URL after it had expired. This drives the real
SubprocessStreamSource against FAKE workers (tiny Python processes speaking the worker's
wire protocol) and a fake resolver — no network, no YouTube.

    .venv/bin/python test_live_capture.py
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from occ import sources

# A fake capture worker: fps header, N frames, then MODE = "hang" (alive, silent — the
# FFMPEG stall) or "exit" (dies, like a stream that won't open).
_WORKER = r"""
import struct, sys, time
import cv2, numpy as np
n, mode = int(sys.argv[1]), sys.argv[2]
out = sys.stdout.buffer
out.write(struct.pack("<f", 30.0)); out.flush()
jpg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()
for _ in range(n):
    out.write(struct.pack("<I", len(jpg))); out.write(jpg); out.flush(); time.sleep(0.01)
if mode == "hang":
    time.sleep(3600)
"""


def _source(frames_per_worker: int, mode: str, page_url=None):
    spawned, resolved = [], []

    class Probe(sources.SubprocessStreamSource):
        STALL_SECONDS = 1.5
        RESOLVE_EVERY = 0.0

        def _spawn(self):
            import subprocess
            spawned.append(self.uri)
            return subprocess.Popen([sys.executable, "-c", _WORKER, str(frames_per_worker), mode],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def fake_resolve(url):
        resolved.append(url)
        return f"https://cdn.example/stream-{len(resolved)}.m3u8", ""

    real = sources.resolve_youtube
    sources.resolve_youtube = fake_resolve
    src = Probe("https://cdn.example/stream-0.m3u8", page_url=page_url)
    return src, spawned, resolved, real


def _wait(cond, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if cond():
            return True
        time.sleep(0.1)
    return False


def collect() -> list[tuple[str, bool, str]]:
    R: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        R.append((name, bool(cond), detail))

    real = sources.resolve_youtube
    try:
        # 1) worker alive but silent → the watchdog restarts it (used to freeze forever)
        src, spawned, resolved, _ = _source(5, "hang")
        ok = _wait(lambda: len(spawned) >= 2, 8)
        check("stalled worker (alive, silent) is restarted", ok, f"spawns={len(spawned)}")
        check("…and frames flow again after the restart",
              _wait(lambda: len(src._buf) >= 6, 5), f"buffered={len(src._buf)}")
        src.release()

        # 2) a YouTube stream re-resolves from its page instead of reopening a dead URL
        src, spawned, resolved, _ = _source(5, "hang", page_url="https://youtube.com/watch?v=X")
        _wait(lambda: len(spawned) >= 3, 10)
        src.release()
        check("restart re-resolves the page (fresh stream URL)",
              resolved and resolved[0] == "https://youtube.com/watch?v=X", f"resolved={resolved}")
        check("…and respawns on the NEW URL, not the expired one",
              len(spawned) >= 2 and spawned[0].endswith("stream-0.m3u8")
              and spawned[1].endswith("stream-1.m3u8"), f"spawned={spawned}")

        # 3) without a page (plain HLS/RTSP-style URL) nothing is re-resolved
        src, spawned, resolved, _ = _source(5, "hang", page_url=None)
        _wait(lambda: len(spawned) >= 2, 8)
        src.release()
        check("no page → restarts on the same URL, no resolver call",
              len(spawned) >= 2 and not resolved and len(set(spawned)) == 1, f"{spawned} {resolved}")

        # 4) a healthy stream is never killed by the watchdog
        src, spawned, resolved, _ = _source(400, "hang")       # ~4s of steady frames
        time.sleep(3.0)
        check("steady stream: watchdog leaves it alone", len(spawned) == 1, f"spawns={len(spawned)}")
        src.release()

        # 5) the page memory used by open_source
        sources.remember_page("https://cdn.example/a.m3u8", "https://youtube.com/watch?v=A")
        check("open_source can find a stream's page", sources._PAGE_FOR.get(
            "https://cdn.example/a.m3u8") == "https://youtube.com/watch?v=A")
    finally:
        sources.resolve_youtube = real
    return R


def main():
    results = collect()
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({detail})" if detail and not ok else ""))
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    assert not bad, bad


if __name__ == "__main__":
    main()
