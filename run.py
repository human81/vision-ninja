#!/usr/bin/env python
"""CLI entrypoint.

    python run.py run                                   # default config
    python run.py run --source rtsp://localhost:8554/cam
    python run.py run --set detector.model=yolo11s --set runtime.detect_every=3
    python run.py run --no-show --out out/annotated.mp4 --max-frames 300

Any config field is overridable with repeated --set key.path=value.
"""

from __future__ import annotations

import argparse
import sys

from occ.config import Config
from occ.pipeline import Pipeline


def _common_overrides(args) -> list[str]:
    overrides = list(args.set or [])
    if getattr(args, "source", None):
        overrides.append(f"source.uri={args.source}")
    if getattr(args, "annotations", None):
        overrides.append(f"annotations={args.annotations}")
    return overrides


def cmd_run(args):
    overrides = _common_overrides(args)
    if args.detector:
        overrides.append(f"detector.backend={args.detector}")
    if args.model:
        overrides.append(f"detector.model={args.model}")
    if args.tracker:
        overrides.append(f"tracker.algorithm={args.tracker}")
    if args.detect_every is not None:
        overrides.append(f"runtime.detect_every={args.detect_every}")

    cfg = Config.load(args.config, overrides=overrides)
    pipe = Pipeline(cfg)
    show = False if args.no_show else None
    stats = pipe.run(show=show, out_path=args.out, emit_proto=args.emit_proto,
                     analytics_csv=args.analytics_csv,
                     analytics_json=args.analytics_json,
                     max_frames=args.max_frames)
    print(f"\ndone: {stats['frames']} frames, "
          f"{stats['fps']:.1f} fps avg" if stats.get("fps") else stats)


def cmd_edit(args):
    from occ.editor import Editor
    overrides = _common_overrides(args)
    cfg = Config.load(args.config, overrides=overrides)
    ann_path = args.annotations or cfg.get("annotations") or "configs/annotations.json"
    s = Editor(cfg, ann_path).run()
    s.save(ann_path)
    print(f"saved {len(s.annotations)} annotations → {ann_path}")


def cmd_ground(args):
    """Open-vocab VLM grounding: locate `--prompt` on a frame, optionally save a zone."""
    import cv2
    from occ.grounding import build_grounder, suggest_zone_from_boxes
    from occ.annotations import AnnotationSet
    from occ.sources import open_source

    overrides = _common_overrides(args)
    cfg = Config.load(args.config, overrides=overrides)
    src = open_source(cfg)
    try:
        frame = next(src.frames())
    finally:
        src.release()

    grounder = build_grounder(cfg)
    print(f"grounding '{args.prompt}' with {grounder.name} (first run downloads weights)…")
    boxes = grounder.ground(frame, args.prompt)
    print(f"found {len(boxes)} boxes")
    for b in boxes:
        print(f"  {b.label or args.prompt}: ({b.x1:.3f},{b.y1:.3f})-({b.x2:.3f},{b.y2:.3f})")

    if args.suggest_zone and boxes:
        zone = suggest_zone_from_boxes(boxes, name=args.suggest_zone)
        ann_path = args.annotations or cfg.get("annotations") or "configs/annotations.json"
        s = AnnotationSet.load(ann_path)
        s.annotations.append(zone)
        s.save(ann_path)
        print(f"saved suggested zone '{zone.id}' → {ann_path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="run.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the real-time pipeline")
    r.add_argument("--config", default=None, help="path to a config YAML")
    r.add_argument("--source", help="override source.uri (file | rtsp:// | cam index)")
    r.add_argument("--detector", choices=["yolo", "rfdetr", "owlv2"], help="detector backend")
    r.add_argument("--model", help="override detector.model (e.g. yolo11s.pt)")
    r.add_argument("--tracker", choices=["bytetrack", "botsort", "ocsort", "sort"])
    r.add_argument("--detect-every", type=int, dest="detect_every",
                   help="run detector every N frames (cost lever)")
    r.add_argument("--set", action="append", metavar="k.path=value",
                   help="override any config field (repeatable)")
    r.add_argument("--annotations", help="zones/lines JSON to load")
    r.add_argument("--no-show", action="store_true", help="headless (no window)")
    r.add_argument("--out", help="write annotated mp4 to this path")
    r.add_argument("--emit-proto", dest="emit_proto",
                   help="write length-delimited OccupancyCountingPredictionResult stream")
    r.add_argument("--analytics-csv", dest="analytics_csv",
                   help="write per-interval traffic counts CSV")
    r.add_argument("--analytics-json", dest="analytics_json",
                   help="write analytics JSON summary")
    r.add_argument("--max-frames", type=int, dest="max_frames",
                   help="stop after N frames")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("edit", help="interactively draw zones/lines")
    e.add_argument("--config", default=None, help="path to a config YAML")
    e.add_argument("--source", help="override source.uri (frame to draw on)")
    e.add_argument("--annotations", help="annotations JSON to load/save")
    e.add_argument("--set", action="append", metavar="k.path=value")
    e.set_defaults(func=cmd_edit)

    g = sub.add_parser("ground", help="open-vocab VLM grounding (zone setup)")
    g.add_argument("--config", default=None)
    g.add_argument("--source", help="frame source to ground on")
    g.add_argument("--prompt", required=True, help="what to locate, e.g. 'forklifts'")
    g.add_argument("--suggest-zone", dest="suggest_zone", metavar="NAME",
                   help="save a suggested zone with this name into the annotations")
    g.add_argument("--annotations", help="annotations JSON to append the zone to")
    g.add_argument("--set", action="append", metavar="k.path=value")
    g.set_defaults(func=cmd_ground)

    st = sub.add_parser("studio", help="agentic Vision Ninja studio (ADK + browser)")
    st.add_argument("--source", help="initial source.uri")
    st.add_argument("--port", type=int, default=8011)
    st.add_argument("--host", default="127.0.0.1")
    st.set_defaults(func=cmd_studio)

    args = p.parse_args(argv)
    return args.func(args)


def cmd_studio(args):
    import os
    import uvicorn
    if args.source:
        os.environ["OCC_SOURCE"] = args.source
    print(f"🥷 Vision Ninja Studio → http://{args.host}:{args.port}")
    uvicorn.run("occ.studio.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
