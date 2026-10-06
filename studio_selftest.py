"""Studio self-test — the 'loss function' for the Vision Ninja studio.

Exercises every agent tool + endpoint + the NLE renderer against a live pipeline,
counts failures (= loss), and prints a per-check report. Run it each epoch; the
loss should monotonically decrease.

    .venv/bin/python studio_selftest.py

Gemini tools (describe/nano/narrate) run for real when a key is in .env (cheap);
Veo is wired but SKIPPED (slow + costly). Stop the studio server first (MPS).
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
os.environ.setdefault("STUDIO_AUTH", "off")      # in-process client = loopback; no sign-in
os.environ.setdefault("STUDIO_AGENT_CODE", "on")  # exercises create_overlay / run_cv_code

from occ.studio.server import create_studio_app           # noqa: E402
from occ.studio import tools as T, nle                     # noqa: E402

RESULTS = []


def check(name, fn, *, expect_error=False, optional=False):
    try:
        res = fn()
        ok = isinstance(res, dict) and res.get("status") == "success"
        if expect_error:
            ok = isinstance(res, dict) and res.get("status") == "error"
        detail = "" if ok else str(res)[:160]
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    RESULTS.append((name, ok, detail, optional))
    print(f"  {'✓' if ok else '✗'} {name}" + ("" if ok else f"   → {detail}"))
    return RESULTS[-1]


def main():
    app = create_studio_app()
    pipe = app.state.pipe
    lib = app.state.ledger
    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    print(f"agent mode: {app.state.agent.mode()}  |  gemini key: {has_key}")
    # warm up: source + first frames
    T.set_source(source="assets/videos/vehicles-2.mp4")
    for _ in range(80):
        if pipe.snapshot_clean() is not None and pipe.stats().get("tracks", 0):
            break
        time.sleep(0.1)
    time.sleep(1.0)

    print("\n— pipeline control (the agentic CV core) —")
    check("set_detector(yolo, car)", lambda: T.set_detector(backend="yolo", classes="car"))
    for trk in ("bytetrack", "botsort", "ocsort", "sort"):
        check(f"set_tracker({trk})", lambda t=trk: T.set_tracker(algorithm=t))
    check("set_detect_every(2)", lambda: T.set_detect_every(n=2))
    # rfdetr/owlv2 are verified manually (heavy model load) — see commit notes.

    print("\n— overlays (the OpenCV ninja) —")
    check("toggle density_heatmap", lambda: T.toggle_overlay(name="density_heatmap", on=True))
    check("create_overlay (valid)", lambda: T.create_overlay(
        name="t_ring", intent="ring cars",
        code="def draw(ctx):\n    for c in ctx.anchors: ctx.ring((c[0],c[1]-20),20,'cyan')"))
    check("create_overlay (bad code -> error)", lambda: T.create_overlay(
        name="t_bad", intent="bad", code="def draw(ctx)\n  syntax error"),
        expect_error=True)
    check("list_overlays", lambda: T.list_overlays())
    check("remove_overlay", lambda: T.remove_overlay(name="t_ring"))

    print("\n— annotations —")
    check("draw_zone (3 pts)", lambda: T.draw_zone(points=[[0.2, 0.6], [0.8, 0.6], [0.5, 0.95]]))
    check("draw_line (2 pts)", lambda: T.draw_line(points=[[0.1, 0.7], [0.9, 0.7]]))
    check("draw_zone (bad <3 -> error)", lambda: T.draw_zone(points=[[0.1, 0.1]]), expect_error=True)
    check("clear_annotations", lambda: T.clear_annotations())

    print("\n— perception —")
    sc = check("analyze_scene", lambda: T.analyze_scene())
    check("analyze_image (test card)", lambda: T.analyze_image(
        image=T.test_image()["output"]))

    print("\n— media library + sources —")
    snap = check("snapshot -> library", lambda: T.snapshot())
    check("test_image", lambda: T.test_image(label="selftest"))
    check("save_to_library", lambda: T.save_to_library(caption="selftest frame", tags="test"))
    check("search_library('snapshot')", lambda: T.search_library(query="snapshot"))
    check("list_library", lambda: T.list_library())
    check("save_source (rtsp)", lambda: T.save_source(url="rtsp://demo/cam", name="cam1"))
    check("list_sources", lambda: T.list_sources())

    print("\n— ffmpeg —")
    check("probe_media(source)", lambda: T.probe_media(which="source"))
    check("export_gif(source)", lambda: T.export_gif(which="source", duration=2))
    check("export_contact_sheet", lambda: T.export_contact_sheet(which="source"))
    check("edit_video filter grayscale", lambda: T.edit_video(op="filter", look="grayscale", which="source"))
    clip1 = check("edit_video frame grab", lambda: T.edit_video(op="frame", which="source", seconds=1.0))

    print("\n— gemini media —")
    if has_key:
        check("describe_image (vision)", lambda: T.describe_image(question="one word: scene?"))
        check("nano_banana edit", lambda: T.nano_banana(prompt="make it look like night"))
        check("narrate (TTS)", lambda: T.narrate(text="Vision Ninja self test.", voice="Puck"))
    else:
        print("  (no key — skipping vision/nano/tts)")
    check("generate_music (gated -> error)", lambda: T.generate_music(prompt="lofi"), expect_error=True)

    print("\n— NLE timeline render —")
    c1 = T.export_clip(which="source", start=0, duration=2)
    c2 = T.export_clip(which="source", start=2, duration=2)
    tl = {"video": [{"src": c1.get("output", ""), "t0": 0, "offset": 0, "duration": 2},
                    {"src": c2.get("output", ""), "t0": 2, "offset": 0, "duration": 2}],
          "audio": []}
    if has_key:
        nar = T.narrate(text="A short edit.", voice="Puck")
        if nar.get("status") == "success":
            tl["audio"].append({"src": nar["output"], "t0": 0, "offset": 0, "duration": 2, "gain": 1.0})
    check("nle.render (2 video + audio)", lambda: nle.render(tl))
    check("co_direct (assemble timeline)", lambda: T.co_direct(brief="highlight reel", max_scenes=2))
    if has_key:
        check("direct_story (fast, 2 scenes)",
              lambda: T.direct_story(brief="a calm city morning", scenes=2, mode="fast"))

    # ---- loss report ----
    fails = [r for r in RESULTS if not r[1] and not r[3]]
    opt_fails = [r for r in RESULTS if not r[1] and r[3]]
    total = len(RESULTS)
    print("\n" + "=" * 60)
    print(f"  LOSS = {len(fails)} hard failures / {total} checks"
          + (f"  (+{len(opt_fails)} optional)" if opt_fails else ""))
    if fails:
        print("  FAILURES:")
        for n, _, d, _ in fails:
            print(f"    ✗ {n}: {d}")
    print(f"  ledger points spent: {lib.summary()['wallet']['points_spent']:.0f}"
          f"  | brain $ {lib.summary()['wallet']['usd_spent']}")
    print("=" * 60)
    pipe.shutdown()
    return len(fails)


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
