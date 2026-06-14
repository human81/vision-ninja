"""Phase 3 headless test:
  A) analytics — run the pipeline with small interval bins, assert the CSV has
     line_crossing / zone_occupancy / full_frame rows and the JSON summary totals.
  B) RF-DETR — build the rfdetr backend, detect on one frame, assert it finds the
     target classes (people/vehicles) and outputs supervision.Detections.
"""

import csv
import json

from occ.config import Config
from occ.annotations import AnnotationSet, Annotation, ZONE, LINE


def _ann(path):
    s = AnnotationSet()
    s.annotations.append(Annotation("line1", LINE, [(0.05, 0.80), (0.95, 0.80)], "midline"))
    s.annotations.append(Annotation("zone1", ZONE,
                                    [(0.2, 0.6), (0.8, 0.6), (0.8, 0.95), (0.2, 0.95)], "roi"))
    s.save(path)
    return path


def test_analytics():
    from occ.pipeline import Pipeline
    ann = _ann("configs/_t3_ann.json")
    cfg = Config.load(overrides=[
        "source.uri=assets/videos/vehicles-2.mp4", "source.loop=false",
        f"annotations={ann}",
        "analytics.enabled=true", "analytics.interval_seconds=2",
    ])
    Pipeline(cfg).run(show=False, analytics_csv="out/counts.csv",
                      analytics_json="out/summary.json", max_frames=300)

    rows = list(csv.DictReader(open("out/counts.csv")))
    metrics = {r["metric"] for r in rows}
    intervals = {r["interval_start"] for r in rows}
    summary = json.load(open("out/summary.json"))

    assert {"line_crossing", "zone_occupancy", "full_frame"} <= metrics, metrics
    assert len(intervals) >= 3, f"expected several interval bins, got {intervals}"
    assert summary["line_crossings"], summary
    crossings = sum(int(r["value"]) for r in rows if r["metric"] == "line_crossing")
    print(f"A) analytics OK — {len(rows)} rows, {len(intervals)} interval bins, "
          f"metrics={sorted(metrics)}, total line crossings={crossings}")
    print(f"   summary.line_crossings={summary['line_crossings']}")


def test_rfdetr():
    import cv2
    from occ.detectors import build_detector
    cfg = Config.load(overrides=[
        "detector.backend=rfdetr",
        "detector.rfdetr_checkpoint=Roboflow/rf-detr-nano",
        "detector.conf=0.3",
    ])
    det = build_detector(cfg)
    cap = cv2.VideoCapture("assets/videos/vehicles-2.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
    ok, frame = cap.read(); cap.release()
    assert ok
    d = det.detect(frame)
    names = list(d.data.get("class_name", [])) if len(d) else []
    targets = cfg.resolve_class_names()
    assert len(d) > 0, "RF-DETR found nothing"
    assert all(n in targets for n in names), f"unfiltered classes: {set(names)-targets}"
    print(f"B) RF-DETR OK — device={det.device}, {len(d)} detections on frame 120, "
          f"classes={dict((n, names.count(n)) for n in set(names))}")


if __name__ == "__main__":
    test_analytics()
    test_rfdetr()
    print("\nPHASE 3 PASS")
