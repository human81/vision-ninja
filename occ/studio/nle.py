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


def _scaled(idx: int, c: dict) -> str:
    """trim -> reset PTS -> scale/pad to canonical size -> fps."""
    off = float(c.get("offset", 0)); dur = float(c.get("duration", 0)) or None
    trim = f"trim=start={off}" + (f":duration={dur}" if dur else "")
    return (f"[{idx}:v]{trim},setpts=PTS-STARTPTS,scale={W}:{H}:"
            f"force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={FPS},format=yuv420p")


def _input_for(c: dict) -> str | None:
    """Resolve a clip to a video input path — a still IMAGE becomes a Ken-Burns clip;
    gifs/videos pass through (ffmpeg reads them directly)."""
    from .ffmpeg_ops import still_to_clip
    src = _resolve(c.get("src", ""))
    if not src:
        return None
    if Path(src).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        r = still_to_clip(src, float(c.get("duration", 3)) or 3.0)
        return r.get("output") if r.get("status") == "success" else None
    return src


def render(timeline: dict) -> dict:
    vclips = []
    for c in (timeline.get("video") or []):
        inp = _input_for(c)
        if inp:
            c = dict(c); c["_in"] = inp; vclips.append(c)
    aclips = [c for c in (timeline.get("audio") or []) if _resolve(c.get("src", ""))]
    xfade = float(timeline.get("transitions", 0) or 0)        # crossfade seconds
    if not vclips and not aclips:
        return {"status": "error", "error": "timeline is empty"}

    Path("out/studio/exports").mkdir(parents=True, exist_ok=True)
    out = f"out/studio/exports/edit_{time.strftime('%H%M%S')}.mp4"
    args, fc, idx = ["ffmpeg", "-y"], [], 0

    # ---- base video track (track 0): xfade-chain or concat ----
    base = sorted([c for c in vclips if int(c.get("track", 0)) == 0],
                  key=lambda c: c.get("t0", 0))
    base_lbls = []
    for c in base:
        args += ["-i", c["_in"]]
        fc.append(_scaled(idx, c) + f"[v{idx}]")
        base_lbls.append((f"[v{idx}]", float(c.get("duration", 0)) or 4.0))
        idx += 1
    base_out = None
    if len(base_lbls) == 1:
        fc.append(f"{base_lbls[0][0]}null[base]"); base_out = "[base]"
    elif len(base_lbls) > 1 and xfade > 0:
        acc, cum = base_lbls[0][0], base_lbls[0][1]
        for k in range(1, len(base_lbls)):
            lbl, dur = base_lbls[k]
            off = max(0.0, cum - xfade)
            nxt = "[base]" if k == len(base_lbls) - 1 else f"[bx{k}]"
            fc.append(f"{acc}{lbl}xfade=transition=fade:duration={xfade}:"
                      f"offset={off:.3f}{nxt}")
            acc, cum = nxt, cum - xfade + dur
        base_out = "[base]"
    elif len(base_lbls) > 1:
        fc.append("".join(l for l, _ in base_lbls)
                  + f"concat=n={len(base_lbls)}:v=1:a=0[base]"); base_out = "[base]"

    # ---- overlay tracks (track >= 1): picture-in-picture, gated to their window ----
    cur = base_out
    for c in [c for c in vclips if int(c.get("track", 0)) >= 1]:
        if cur is None:
            break
        args += ["-i", c["_in"]]
        t0, dur = float(c.get("t0", 0)), float(c.get("duration", 0)) or 4.0
        off = float(c.get("offset", 0))
        fc.append(f"[{idx}:v]trim=start={off}:duration={dur},"
                  f"setpts=PTS-STARTPTS+{t0}/TB,scale={W // 3}:-2[ov{idx}]")
        nxt = f"[ovo{idx}]"
        fc.append(f"{cur}[ov{idx}]overlay=W-w-24:24:"
                  f"enable='between(t,{t0:.3f},{t0 + dur:.3f})':eof_action=pass{nxt}")
        cur, idx = nxt, idx + 1
    vout = cur

    # ---- audio tracks: delay each to its t0 and mix ----
    alabels = []
    for c in aclips:
        args += ["-i", _resolve(c["src"])]
        off = float(c.get("offset", 0)); dur = float(c.get("duration", 0)) or None
        t0 = float(c.get("t0", 0)); gain = float(c.get("gain", 1.0))
        atrim = f"atrim=start={off}" + (f":duration={dur}" if dur else "")
        fc.append(f"[{idx}:a]{atrim},asetpts=PTS-STARTPTS,"
                  f"adelay={int(t0 * 1000)}|{int(t0 * 1000)},volume={gain}[a{idx}]")
        alabels.append(f"[a{idx}]"); idx += 1

    maps = []
    if vout:
        maps += ["-map", vout]
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
            "clips": len(vclips) + len(aclips),
            "transitions": xfade, "tracks": 1 + sum(1 for c in vclips if int(c.get("track", 0)) >= 1)}
