"""Phase 0 contract test: build a representative OccupancyCountingPredictionResult,
serialize it, parse it back through parse_occupancy.py, and assert every field
group survives the round trip. Locks the protobuf data contract before any CV code.

Run:  .venv/bin/python test_proto_roundtrip.py
"""

from proto import (
    OccupancyCountingPredictionResult as R,
    StreamAnnotation,
    StreamAnnotationType,
    NormalizedVertex,
)
from parse_occupancy import parse_occupancy_result


def _ts(msg, seconds):
    msg.seconds = seconds
    return msg


def build_sample() -> bytes:
    r = R()
    _ts(r.current_time, 1_700_000_000)
    r.pts = 42

    # two detections: a person and a vehicle, each tracked
    for box_id, track_id, label_id, label, score, (x, y, w, h) in [
        (1, 101, 1, "Person", 0.91, (0.10, 0.20, 0.05, 0.12)),
        (2, 102, 2, "Vehicle", 0.88, (0.40, 0.55, 0.18, 0.14)),
    ]:
        b = r.identified_boxes.add()
        b.box_id = box_id
        b.track_id = track_id
        b.entity.label_id = label_id
        b.entity.label_string = label
        b.score = score
        b.normalized_bounding_box.xmin = x
        b.normalized_bounding_box.ymin = y
        b.normalized_bounding_box.width = w
        b.normalized_bounding_box.height = h

    # full-frame counts
    for label_id, label, count in [(1, "Person", 1), (2, "Vehicle", 1)]:
        oc = r.stats.full_frame_count.add()
        oc.entity.label_id = label_id
        oc.entity.label_string = label
        oc.count = count

    # one crossing line with +/- direction counts (right-hand rule)
    cl = r.stats.crossing_line_counts.add()
    cl.annotation.id = "line-north"
    cl.annotation.type = StreamAnnotationType.STREAM_ANNOTATION_TYPE_CROSSING_LINE
    cl.annotation.crossing_line.normalized_vertices.extend(
        [NormalizedVertex(x=0.0, y=0.5), NormalizedVertex(x=1.0, y=0.5)]
    )
    pos = cl.positive_direction_counts.add()
    pos.entity.label_string = "Vehicle"
    pos.count = 7
    neg = cl.negative_direction_counts.add()
    neg.entity.label_string = "Vehicle"
    neg.count = 3

    # one active zone
    az = r.stats.active_zone_counts.add()
    az.annotation.id = "zone-crosswalk"
    az.annotation.type = StreamAnnotationType.STREAM_ANNOTATION_TYPE_ACTIVE_ZONE
    az.annotation.active_zone.normalized_vertices.extend(
        [NormalizedVertex(x=0.3, y=0.3), NormalizedVertex(x=0.6, y=0.3),
         NormalizedVertex(x=0.6, y=0.7), NormalizedVertex(x=0.3, y=0.7)]
    )
    zc = az.counts.add()
    zc.entity.label_string = "Person"
    zc.count = 1

    # track info + dwell
    ti = r.track_info.add()
    ti.track_id = "102"
    _ts(ti.start_time, 1_699_999_990)

    dw = r.dwell_time_info.add()
    dw.track_id = "101"
    dw.zone_id = "zone-crosswalk"
    _ts(dw.dwell_start_time, 1_699_999_995)
    _ts(dw.dwell_end_time, 1_700_000_010)

    return r.SerializeToString()


def main():
    blob = build_sample()
    parsed = parse_occupancy_result(blob)

    assert len(parsed["detections"]) == 2, parsed["detections"]
    assert parsed["full_frame_counts"] == {"Person": 1, "Vehicle": 1}
    assert parsed["line_counts"][0]["line"] == "line-north"
    assert parsed["line_counts"][0]["positive"] == {"Vehicle": 7}
    assert parsed["line_counts"][0]["negative"] == {"Vehicle": 3}
    assert parsed["zone_counts"][0]["zone"] == "zone-crosswalk"
    assert parsed["zone_counts"][0]["counts"] == {"Person": 1}
    assert parsed["dwell"][0]["track_id"] == "101"
    assert parsed["dwell"][0]["seconds"] == 15.0

    print(f"round-trip OK — {len(blob)} bytes")
    print(f"  detections={len(parsed['detections'])} "
          f"full_frame={parsed['full_frame_counts']}")
    print(f"  line {parsed['line_counts'][0]['line']}: "
          f"+{parsed['line_counts'][0]['positive']} / "
          f"-{parsed['line_counts'][0]['negative']}")
    print(f"  zone {parsed['zone_counts'][0]['zone']}: "
          f"{parsed['zone_counts'][0]['counts']}")
    print(f"  dwell {parsed['dwell'][0]['track_id']} in "
          f"{parsed['dwell'][0]['zone_id']}: {parsed['dwell'][0]['seconds']}s")
    print("\nPASS — protobuf contract locked.")


if __name__ == "__main__":
    main()
