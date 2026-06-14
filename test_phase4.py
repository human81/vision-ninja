"""Phase 4 headless test:
  A) speed — homography speed estimation over a clip with a sample calibration;
     assert moving vehicles get plausible (0 < v < 300 km/h) speeds.
  B) VLM grounding (model-free) — box parsing, zone suggestion, and that the
     grounder constructs lazily WITHOUT downloading any weights.
"""

import numpy as np

from occ.config import Config


def test_speed():
    from occ.detectors import build_detector
    from occ.tracking import Tracker
    from occ.sources import open_source
    from occ.speed import SpeedEstimator

    cfg = Config.load(overrides=["source.uri=assets/videos/vehicles-2.mp4",
                                 "source.loop=false"])
    src = open_source(cfg); det = build_detector(cfg)
    trk = Tracker("bytetrack", {"frame_rate": 30})
    speed = SpeedEstimator(
        image_points=[[0.35, 0.62], [0.65, 0.62], [0.95, 0.95], [0.05, 0.95]],
        world_points=[[0, 40], [12, 40], [12, 0], [0, 0]], units="kmh")

    fps = src.fps or 30.0
    peak: dict[int, float] = {}
    n = 0
    for frame in src.frames():
        h, w = frame.shape[:2]
        d = trk.update(det.detect(frame), frame)
        for tid, v in speed.update(d, w, h, n / fps).items():
            peak[tid] = max(peak.get(tid, 0), v)
        n += 1
        if n >= 150:
            break
    src.release()

    moving = [v for v in peak.values() if v > 1]
    assert moving, "no speeds computed"
    assert all(0 < v < 300 for v in peak.values()), f"implausible speeds: {sorted(peak.values())[-5:]}"
    med = float(np.median(moving))
    print(f"A) speed OK — {len(peak)} tracks measured, median peak {med:.0f} km/h, "
          f"max {max(peak.values()):.0f} km/h")


def test_grounding_logic():
    from occ.grounding import (build_grounder, parse_locate_anything_boxes,
                               suggest_zone_from_boxes)
    # parse <box> markup (coords 0..1000 → normalized)
    answer = "Here: <box><100><200><300><400></box> and <box><500><600><700><800></box>"
    boxes = parse_locate_anything_boxes(answer, label="forklift")
    assert len(boxes) == 2
    assert abs(boxes[0].x1 - 0.1) < 1e-6 and abs(boxes[0].y2 - 0.4) < 1e-6
    # point markup → tiny box
    pts = parse_locate_anything_boxes("<point><500><500></point>")
    assert len(pts) == 1

    zone = suggest_zone_from_boxes(boxes, name="forklift_area")
    assert zone is not None and len(zone.vertices) == 4
    xs = [v[0] for v in zone.vertices]; ys = [v[1] for v in zone.vertices]
    assert min(xs) <= 0.1 and max(xs) >= 0.7 and min(ys) <= 0.2 and max(ys) >= 0.8

    # grounder constructs lazily — NO weight download on build
    g = build_grounder(Config.load(overrides=["grounding.backend=locate_anything"]))
    assert g.name == "locate_anything"
    assert g._model is None, "model should not load until ground() is called"
    print(f"B) grounding logic OK — parsed 2 boxes + 1 point, suggested zone "
          f"'{zone.id}' ({len(zone.vertices)} pts), grounder lazy (no download)")


if __name__ == "__main__":
    test_speed()
    test_grounding_logic()
    print("\nPHASE 4 PASS")
