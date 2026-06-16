"""ffmpeg toolkit — the studio's export muscle.

"A friend once told me ffmpeg can save the world." These are thin, well-shaped
wrappers the agent calls as tools: probe, clip, gif, transcode, speed-ramp,
contact-sheet, concat. Each returns a dict {status, output, seconds, ...} so the
ledger can meter ffmpeg work in compute points by output duration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

EXPORT_DIR = Path("out/studio/exports")


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _out(name: str, ext: str) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    return EXPORT_DIR / f"{name}_{stamp}.{ext}"


def _run(args: list[str], timeout: int = 120) -> tuple[bool, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stderr or p.stdout)[-800:]
    except Exception as e:
        return False, str(e)


def probe(path: str) -> dict:
    if not Path(path).exists():
        return {"status": "error", "error": f"not found: {path}"}
    ok, out = _run(["ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", path], timeout=30)
    if not ok:
        return {"status": "error", "error": out}
    d = json.loads(out)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    fps = 0.0
    if v.get("r_frame_rate", "0/1") != "0/0":
        try:
            num, den = v["r_frame_rate"].split("/")
            fps = round(float(num) / float(den), 2)
        except Exception:
            pass
    return {"status": "success", "path": path,
            "duration": round(float(d.get("format", {}).get("duration", 0)), 2),
            "width": v.get("width"), "height": v.get("height"), "fps": fps,
            "codec": v.get("codec_name"),
            "size_mb": round(int(d.get("format", {}).get("size", 0)) / 1e6, 2)}


def make_gif(src: str, start: float = 0, duration: float = 4,
             fps: int = 12, width: int = 480) -> dict:
    out = _out("clip", "gif")
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,"
          "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse")
    ok, err = _run(["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
                    "-i", src, "-vf", vf, "-loop", "0", str(out)])
    if not ok:
        return {"status": "error", "error": err}
    return {"status": "success", "output": str(out), "seconds": duration,
            "kind": "gif"}


def clip(src: str, start: float = 0, duration: float = 8, reencode: bool = True) -> dict:
    out = _out("cut", "mp4")
    args = ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", src]
    args += (["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-an"]
             if reencode else ["-c", "copy"])
    args.append(str(out))
    ok, err = _run(args)
    if not ok:
        return {"status": "error", "error": err}
    return {"status": "success", "output": str(out), "seconds": duration, "kind": "clip"}


def transcode(src: str, codec: str = "libx264", crf: int = 23,
              scale_w: int = 0) -> dict:
    out = _out("transcode", "mp4")
    args = ["ffmpeg", "-y", "-i", src, "-c:v", codec, "-crf", str(crf),
            "-preset", "veryfast"]
    if scale_w:
        args += ["-vf", f"scale={scale_w}:-2"]
    args.append(str(out))
    ok, err = _run(args)
    if not ok:
        return {"status": "error", "error": err}
    info = probe(str(out))
    return {"status": "success", "output": str(out),
            "seconds": info.get("duration", 0), "kind": "transcode"}


def speed_ramp(src: str, factor: float = 2.0) -> dict:
    """factor>1 = faster (timelapse), <1 = slow-mo."""
    out = _out("speed", "mp4")
    ok, err = _run(["ffmpeg", "-y", "-i", src, "-filter:v",
                    f"setpts={1.0/factor}*PTS", "-an", str(out)])
    if not ok:
        return {"status": "error", "error": err}
    info = probe(str(out))
    return {"status": "success", "output": str(out),
            "seconds": info.get("duration", 0), "factor": factor, "kind": "speed"}


def contact_sheet(src: str, cols: int = 4, rows: int = 3) -> dict:
    """Tile evenly-sampled frames into one PNG — a visual summary of the clip."""
    info = probe(src)
    if info.get("status") != "success" or not info.get("duration"):
        return {"status": "error", "error": "could not probe source"}
    n = cols * rows
    dur = info["duration"]
    rate = max(n / dur, 0.01)
    out = _out("sheet", "png")
    ok, err = _run(["ffmpeg", "-y", "-i", src, "-frames:v", "1", "-vf",
                    f"fps={rate},scale=320:-1,tile={cols}x{rows}", str(out)])
    if not ok:
        return {"status": "error", "error": err}
    return {"status": "success", "output": str(out), "tiles": n, "kind": "sheet"}


def _vf(src: str, vf: str, name: str, audio: bool = True) -> dict:
    out = _out(name, "mp4")
    args = ["ffmpeg", "-y", "-i", src, "-vf", vf, "-c:v", "libx264",
            "-crf", "20", "-preset", "veryfast"] + ([] if audio else ["-an"])
    ok, err = _run(args + [str(out)])
    if not ok:
        return {"status": "error", "error": err}
    return {"status": "success", "output": str(out),
            "seconds": probe(str(out)).get("duration", 0), "kind": name}


def crop(src: str, x: int, y: int, w: int, h: int) -> dict:
    """Crop to a w×h box at (x,y)."""
    return _vf(src, f"crop={w}:{h}:{x}:{y}", "crop")


def rotate(src: str, degrees: int = 90) -> dict:
    vf = {90: "transpose=1", 180: "transpose=1,transpose=1",
          270: "transpose=2"}.get(int(degrees) % 360, f"rotate={degrees}*PI/180")
    return _vf(src, vf, "rotate")


def flip(src: str, direction: str = "h") -> dict:
    return _vf(src, "hflip" if direction.startswith("h") else "vflip", "flip")


def vfilter(src: str, name: str = "grayscale") -> dict:
    """Named look: grayscale | sepia | invert | sharpen | blur | vintage | edges."""
    vf = {"grayscale": "format=gray", "invert": "negate",
          "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
          "sharpen": "unsharp=5:5:1.0", "blur": "boxblur=2:1",
          "vintage": "curves=vintage", "edges": "edgedetect=mode=colormix"}.get(
              name, "format=gray")
    return _vf(src, vf, "filter")


def reverse(src: str) -> dict:
    return _vf(src, "reverse", "reverse", audio=False)


def boomerang(src: str) -> dict:
    """Play forward then backward (a seamless loop)."""
    rev = reverse(src)
    if rev.get("status") != "success":
        return rev
    return concat([src, rev["output"]])


def fade(src: str, fade_in: float = 1.0, fade_out: float = 1.0) -> dict:
    dur = probe(src).get("duration", 0)
    if not dur:
        return {"status": "error", "error": "could not probe source"}
    vf = f"fade=t=in:st=0:d={fade_in},fade=t=out:st={max(0, dur - fade_out)}:d={fade_out}"
    return _vf(src, vf, "fade")


def loop(src: str, count: int = 3) -> dict:
    out = _out("loop", "mp4")
    ok, err = _run(["ffmpeg", "-y", "-stream_loop", str(int(count)), "-i", src,
                    "-c", "copy", str(out)])
    if not ok:
        return {"status": "error", "error": err}
    return {"status": "success", "output": str(out),
            "seconds": probe(str(out)).get("duration", 0), "kind": "loop"}


def overlay_text(src: str, text: str, size: int = 36) -> dict:
    safe = text.replace(":", r"\:").replace("'", "")
    vf = (f"drawtext=text='{safe}':fontcolor=white:fontsize={size}:box=1:"
          "boxcolor=black@0.5:boxborderw=8:x=(w-text_w)/2:y=h-text_h-30")
    return _vf(src, vf, "captioned")


def stack(a: str, b: str, direction: str = "h") -> dict:
    """Side-by-side (h) or stacked (v) — great for before/after."""
    out = _out("stack", "mp4")
    fc = "hstack=inputs=2" if direction.startswith("h") else "vstack=inputs=2"
    ok, err = _run(["ffmpeg", "-y", "-i", a, "-i", b, "-filter_complex", fc,
                    "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", str(out)])
    if not ok:
        return {"status": "error", "error": err}
    return {"status": "success", "output": str(out),
            "seconds": probe(str(out)).get("duration", 0), "kind": "stack"}


def extract_audio(src: str) -> dict:
    out = _out("audio", "mp3")
    ok, err = _run(["ffmpeg", "-y", "-i", src, "-vn", "-q:a", "2", str(out)])
    if not ok:
        return {"status": "error", "error": err}
    return {"status": "success", "output": str(out), "kind": "audio"}


def frame_at(src: str, t: float = 0.0) -> dict:
    out = _out("frame", "png")
    ok, err = _run(["ffmpeg", "-y", "-ss", str(t), "-i", src, "-frames:v", "1", str(out)])
    if not ok:
        return {"status": "error", "error": err}
    return {"status": "success", "output": str(out), "kind": "image"}


def concat(paths: list[str]) -> dict:
    paths = [p for p in paths if Path(p).exists()]
    if len(paths) < 2:
        return {"status": "error", "error": "need >=2 existing clips"}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    listf = EXPORT_DIR / "concat_list.txt"
    listf.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    out = _out("joined", "mp4")
    ok, err = _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
                    "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", str(out)])
    if not ok:
        return {"status": "error", "error": err}
    info = probe(str(out))
    return {"status": "success", "output": str(out),
            "seconds": info.get("duration", 0), "parts": len(paths), "kind": "concat"}
