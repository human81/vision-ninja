"""NLE timeline renderer — ffmpeg filter_complex, modeled on Momentum's Creative
Studio Workbench (clips with track/type/start/duration/offset; multi-track audio).

Timeline shape (from the browser editor):
  { "video": [ {src, t0, offset, duration} ... ],   # video track, in t0 order
    "audio": [ {src, t0, offset, duration, gain} ... ] }   # narration / music
Renders a 1280x720 / 30fps MP4: trimmed video clips concatenated, audio-track
clips delayed to their t0 and mixed. The OpenCV-annotated clips are the stars;
this is the surround that assembles them with narration/music.
"""

from __future__ import annotations

import glob
import os
import subprocess
import time
from pathlib import Path

W, H, FPS = 1280, 720, 30
_SEARCH = ["out/studio/exports", "out/recordings", "out/studio/cache"]


def _resolve(name: str) -> str | None:
    if os.path.exists(name):
        return name
    base = os.path.basename(name)
    for d in _SEARCH:
        p = os.path.join(d, base)
        if os.path.exists(p):
            return p
    hits = [h for d in _SEARCH for h in glob.glob(os.path.join(d, base))]
    return hits[0] if hits else None


def render(timeline: dict) -> dict:
    vclips = [c for c in (timeline.get("video") or []) if _resolve(c.get("src", ""))]
    aclips = [c for c in (timeline.get("audio") or []) if _resolve(c.get("src", ""))]
    if not vclips and not aclips:
        return {"status": "error", "error": "timeline is empty"}

    Path("out/studio/exports").mkdir(parents=True, exist_ok=True)
    out = f"out/studio/exports/edit_{time.strftime('%H%M%S')}.mp4"
    args = ["ffmpeg", "-y"]
    fc = []                          # filter_complex parts
    idx = 0                          # ffmpeg input index

    # video track: trim -> scale/pad -> reset PTS -> concat
    vlabels = []
    for c in sorted(vclips, key=lambda c: c.get("t0", 0)):
        src = _resolve(c["src"])
        off = float(c.get("offset", 0)); dur = float(c.get("duration", 0)) or None
        args += ["-i", src]
        trim = f"trim=start={off}" + (f":duration={dur}" if dur else "")
        fc.append(f"[{idx}:v]{trim},setpts=PTS-STARTPTS,scale={W}:{H}:"
                  f"force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
                  f"fps={FPS},format=yuv420p[v{idx}]")
        vlabels.append(f"[v{idx}]")
        idx += 1

    alabels = []
    for c in aclips:
        src = _resolve(c["src"])
        off = float(c.get("offset", 0)); dur = float(c.get("duration", 0)) or None
        t0 = float(c.get("t0", 0)); gain = float(c.get("gain", 1.0))
        args += ["-i", src]
        atrim = f"atrim=start={off}" + (f":duration={dur}" if dur else "")
        fc.append(f"[{idx}:a]{atrim},asetpts=PTS-STARTPTS,"
                  f"adelay={int(t0 * 1000)}|{int(t0 * 1000)},volume={gain}[a{idx}]")
        alabels.append(f"[a{idx}]")
        idx += 1

    maps = []
    if vlabels:
        if len(vlabels) == 1:
            fc.append(f"{vlabels[0]}copy[vout]")
        else:
            fc.append("".join(vlabels) + f"concat=n={len(vlabels)}:v=1:a=0[vout]")
        maps += ["-map", "[vout]"]
    if alabels:
        if len(alabels) == 1:
            fc.append(f"{alabels[0]}anull[aout]")
        else:
            fc.append("".join(alabels) + f"amix=inputs={len(alabels)}:normalize=0[aout]")
        maps += ["-map", "[aout]", "-shortest"]

    args += ["-filter_complex", ";".join(fc)] + maps
    args += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast"]
    if alabels:
        args += ["-c:a", "aac"]
    args.append(out)

    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=300)
    except Exception as e:
        return {"status": "error", "error": f"ffmpeg: {e}"}
    if p.returncode != 0 or not os.path.exists(out):
        return {"status": "error", "error": (p.stderr or "")[-400:]}
    return {"status": "success", "output": out, "kind": "edit",
            "clips": len(vclips) + len(aclips)}
