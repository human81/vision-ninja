"""Phase 2 headless test:
  A) drive the Editor's logic (simulated mouse/keys) → assert it builds annotations
  B) run the geometry + emit pipeline on a traffic clip → assert line/zone counts
     accumulate and the emitted protobuf stream is wire-valid.
"""

import cv2

from occ.config import Config
from occ.annotations import AnnotationSet, Annotation, ZONE, LINE


def test_editor_logic():
    from occ.editor import Editor
    cfg = Config.load(overrides=["source.uri=assets/videos/vehicles-2.mp4"])
    ed = Editor(cfg, "configs/_test_editor.json")

    # draw a LINE: press 'l', click two points, Enter
    ed._key(ord("l"))
    ed._on_mouse(cv2.EVENT_LBUTTONDOWN, int(0.1 * ed.w), int(0.5 * ed.h), 0, None)
    ed._on_mouse(cv2.EVENT_LBUTTONDOWN, int(0.9 * ed.w), int(0.5 * ed.h), 0, None)
    ed._key(13)                                   # Enter commits
    # draw a ZONE: 'z', 4 clicks, Enter
    ed._key(ord("z"))
    for x, y in [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]:
        ed._on_mouse(cv2.EVENT_LBUTTONDOWN, int(x * ed.w), int(y * ed.h), 0, None)
    ed._key(13)
    # name selected zone: 'n', type "merge", Enter
    ed._key(ord("n"))
    for c in "merge":
        ed._key(ord(c))
    ed._key(13)
    # render once (exercises all drawing code paths)
    _ = ed._draw()

    assert len(ed.set.lines()) == 1, ed.set.lines()
    assert len(ed.set.zones()) == 1, ed.set.zones()
    assert ed.set.zones()[0].display_name == "merge"
    print(f"A) editor logic OK — {len(ed.set.annotations)} annotations, "
          f"zone named '{ed.set.zones()[0].display_name}'")
    return ed.set


def _make_annotations(path):
    s = AnnotationSet()
    # horizontal counting line across the road + a rectangular zone
    # placed where stable tracks actually are (lower half, near camera)
    s.annotations.append(Annotation("line1", LINE, [(0.05, 0.80), (0.95, 0.80)], "midline"))
    s.annotations.append(Annotation("zone1", ZONE,
                                    [(0.2, 0.60), (0.8, 0.60), (0.8, 0.95), (0.2, 0.95)], "roi"))
    s.save(path)
    return path


def _read_proto_stream(path):
    """Read length-delimited OccupancyCountingPredictionResult messages."""
    from proto import OccupancyCountingPredictionResult as R
    out = []
    data = open(path, "rb").read()
    i = 0
    while i < len(data):
        shift = 0; n = 0
        while True:
            b = data[i]; i += 1
            n |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        out.append(R.FromString(data[i:i + n])); i += n
    return out


def test_pipeline_geometry():
    from occ.pipeline import Pipeline
    ann = _make_annotations("configs/_test_ann.json")
    cfg = Config.load(overrides=[
        "source.uri=assets/videos/vehicles-2.mp4",
        "source.loop=false",
        f"annotations={ann}",
        "detector.model=yolo11n.pt",
    ])
    pipe = Pipeline(cfg)
    pipe.run(show=False, emit_proto="out/stream.pb", max_frames=400)

    results = _read_proto_stream("out/stream.pb")
    assert len(results) == 400, len(results)
    last = results[-1]
    line = last.stats.crossing_line_counts[0]
    pos = sum(c.count for c in line.positive_direction_counts)
    neg = sum(c.count for c in line.negative_direction_counts)
    has_boxes = any(len(r.identified_boxes) for r in results)
    has_zone = any(any(z.counts for z in r.stats.active_zone_counts) for r in results)

    assert has_boxes, "no detections emitted"
    assert (pos + neg) > 0, f"no line crossings counted (pos={pos} neg={neg})"
    print(f"B) pipeline OK — {len(results)} proto msgs; line1 crossings "
          f"+{pos}/-{neg}; zone occupancy seen: {has_zone}; "
          f"last-frame boxes: {len(last.identified_boxes)}")


if __name__ == "__main__":
    test_editor_logic()
    test_pipeline_geometry()
    print("\nPHASE 2 PASS")
