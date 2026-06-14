"""Build OccupancyCountingPredictionResult protobufs from tracked detections +
geometry stats — wire-identical to the real occupancy-analytics model output.

Optionally stream length-delimited results to a file (varint length prefix per
message) so a whole run is one replayable .pb stream.
"""

from __future__ import annotations

import struct
from collections import Counter

import numpy as np
import supervision as sv

from proto import OccupancyCountingPredictionResult as Result
from .annotations import AnnotationSet
from .geometry import GeoStats, _class_names


def _set_ts(ts, seconds: float) -> None:
    ts.seconds = int(seconds)
    ts.nanos = int((seconds - int(seconds)) * 1e9)


def _object_counts(repeated, counter: Counter, label_ids: dict[str, int]) -> None:
    for name, count in counter.items():
        oc = repeated.add()
        oc.entity.label_string = name
        oc.entity.label_id = label_ids.get(name, 0)
        oc.count = int(count)


def build_result(det: sv.Detections, geo: GeoStats, ann: AnnotationSet,
                 w: int, h: int, t: float, pts: int | None = None) -> Result:
    r = Result()
    _set_ts(r.current_time, t)
    if pts is not None:
        r.pts = int(pts)

    names = _class_names(det)
    cls_ids = det.class_id if det.class_id is not None else [0] * len(det)
    tids = det.tracker_id if det.tracker_id is not None else [None] * len(det)
    label_ids = {names[i]: int(cls_ids[i]) for i in range(len(det))}

    # identified boxes (normalized)
    for i in range(len(det)):
        x1, y1, x2, y2 = det.xyxy[i]
        tid = tids[i]
        has_tid = tid is not None and int(tid) >= 0   # -1 = unconfirmed track
        b = r.identified_boxes.add()
        b.box_id = int(tid) if has_tid else i
        if has_tid:
            b.track_id = int(tid)
        nb = b.normalized_bounding_box
        nb.xmin = float(x1 / w)
        nb.ymin = float(y1 / h)
        nb.width = float((x2 - x1) / w)
        nb.height = float((y2 - y1) / h)
        if det.confidence is not None:
            b.score = float(det.confidence[i])
        b.entity.label_string = names[i]
        b.entity.label_id = int(cls_ids[i])

    # stats: full frame
    _object_counts(r.stats.full_frame_count, geo.full_frame, label_ids)

    # stats: crossing lines (+ right-hand-rule direction)
    by_id = {a.id: a for a in ann.annotations}
    for lid, dirs in geo.line_counts.items():
        cl = r.stats.crossing_line_counts.add()
        cl.annotation.CopyFrom(by_id[lid].to_proto())
        _object_counts(cl.positive_direction_counts, dirs["positive"], label_ids)
        _object_counts(cl.negative_direction_counts, dirs["negative"], label_ids)

    # stats: active zones
    for zid, counter in geo.zone_counts.items():
        az = r.stats.active_zone_counts.add()
        az.annotation.CopyFrom(by_id[zid].to_proto())
        _object_counts(az.counts, counter, label_ids)

    # track info (live tracks)
    for tid, start in geo.track_start.items():
        ti = r.track_info.add()
        ti.track_id = str(tid)
        _set_ts(ti.start_time, start)

    # dwell info
    for track_id, zone_id, start, end in geo.dwell:
        dw = r.dwell_time_info.add()
        dw.track_id = track_id
        dw.zone_id = zone_id
        _set_ts(dw.dwell_start_time, start)
        _set_ts(dw.dwell_end_time, end)

    return r


class ResultWriter:
    """Length-delimited protobuf stream writer (varint length + payload)."""

    def __init__(self, path: str):
        self.f = open(path, "wb")

    def write(self, result: Result) -> None:
        payload = result.SerializeToString()
        # varint length prefix
        n = len(payload)
        while True:
            b = n & 0x7F
            n >>= 7
            self.f.write(struct.pack("B", b | (0x80 if n else 0)))
            if not n:
                break
        self.f.write(payload)

    def close(self) -> None:
        self.f.close()
