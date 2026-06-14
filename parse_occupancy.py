"""Parse Vertex AI Vision occupancy-analytics output.

The occupancy-analytics model emits an ``OccupancyCountingPredictionResult``
protobuf per frame (constant 3 FPS). This module decodes one serialized result
into plain Python dicts for downstream use.

Uses the vendored, locally-compiled protobuf (proto/visionai_annotations.proto) —
no cloud dependency. The schema is wire-identical to the official model output.
"""

from proto import OccupancyCountingPredictionResult


def parse_occupancy_result(serialized: bytes) -> dict:
    """Decode one frame of occupancy-analytics output."""
    result = OccupancyCountingPredictionResult.FromString(serialized)

    frame = {
        "time": result.current_time.ToDatetime(),  # google.protobuf.Timestamp
        "detections": [],
        "full_frame_counts": {},
        "line_counts": [],
        "zone_counts": [],
        "dwell": [],
    }

    # --- per-object detections ---------------------------------------
    for box in result.identified_boxes:
        bb = box.normalized_bounding_box
        frame["detections"].append({
            "track_id": box.track_id,
            "label": box.entity.label_string,
            "score": round(box.score, 3),
            # normalized 0..1 -> keep normalized, scale later by frame W/H
            "bbox": (bb.xmin, bb.ymin, bb.width, bb.height),
        })

    # --- whole-frame totals ------------------------------------------
    for oc in result.stats.full_frame_count:
        frame["full_frame_counts"][oc.entity.label_string] = oc.count

    # --- line crossings (right-hand rule => +/- direction) -----------
    for line in result.stats.crossing_line_counts:
        frame["line_counts"].append({
            "line": line.annotation.id,
            "positive": {c.entity.label_string: c.count
                         for c in line.positive_direction_counts},
            "negative": {c.entity.label_string: c.count
                         for c in line.negative_direction_counts},
        })

    # --- active zone occupancy ---------------------------------------
    for zone in result.stats.active_zone_counts:
        frame["zone_counts"].append({
            "zone": zone.annotation.id,
            "counts": {c.entity.label_string: c.count for c in zone.counts},
        })

    # --- dwell time per track/zone -----------------------------------
    for d in result.dwell_time_info:
        start = d.dwell_start_time.ToDatetime()
        end = d.dwell_end_time.ToDatetime()
        frame["dwell"].append({
            "track_id": d.track_id,
            "zone_id": d.zone_id,
            "seconds": (end - start).total_seconds(),
        })

    return frame


def to_pixels(bbox, frame_w: int, frame_h: int):
    """Convert a normalized (xmin, ymin, width, height) box to pixel coords."""
    xmin, ymin, w, h = bbox
    return (int(xmin * frame_w), int(ymin * frame_h),
            int(w * frame_w), int(h * frame_h))


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "frame.pb"
    with open(path, "rb") as f:
        parsed = parse_occupancy_result(f.read())

    print(f"{parsed['time']}  total={parsed['full_frame_counts']}")
    for det in parsed["detections"]:
        print(f"  track {det['track_id']:>4}  {det['label']:8} "
              f"{det['score']:.2f}  bbox={det['bbox']}")
    for ln in parsed["line_counts"]:
        print(f"  line {ln['line']}: +{ln['positive']} / -{ln['negative']}")
    for z in parsed["zone_counts"]:
        print(f"  zone {z['zone']}: {z['counts']}")
    for d in parsed["dwell"]:
        print(f"  dwell track {d['track_id']} in zone {d['zone_id']}: "
              f"{d['seconds']:.1f}s")
