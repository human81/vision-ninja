"""Traffic analytics: roll the per-frame geometry into fixed time intervals and
write a tidy CSV (one row per metric per interval) plus a JSON summary.

Metrics per interval:
  * line_crossing  — directional crossings counted *in this interval* (delta of the
                     cumulative line counters), per line / direction / class
  * zone_occupancy — average & peak instantaneous occupancy, per zone / class
  * full_frame     — average & peak instantaneous count, per class

Line counters in GeometryEngine are cumulative; we snapshot them at each interval
boundary to get per-interval deltas. Zone/full-frame counts are instantaneous, so
we accumulate sum/count (for the mean) and a running max (for the peak).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict


class AnalyticsSink:
    def __init__(self, interval_seconds: float = 900.0,
                 csv_path: str | None = None, json_path: str | None = None):
        self.interval = float(interval_seconds)
        self.csv_path = csv_path
        self.json_path = json_path
        self.bin_start: float | None = None

        self._line_base: dict[tuple, int] = defaultdict(int)   # value at interval start
        self._line_latest: dict[tuple, int] = defaultdict(int)
        self._zone_sum: dict[tuple, float] = defaultdict(float)
        self._zone_cnt: dict[tuple, int] = defaultdict(int)
        self._zone_max: dict[tuple, int] = defaultdict(int)
        self._full_sum: dict[str, float] = defaultdict(float)
        self._full_cnt: dict[str, int] = defaultdict(int)
        self._full_max: dict[str, int] = defaultdict(int)

        self.rows: list[dict] = []
        self._csv_file = None
        self._writer = None
        if csv_path:
            self._csv_file = open(csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=[
                "interval_start", "interval_end", "metric", "id",
                "direction", "entity", "value", "value_max"])
            self._writer.writeheader()

    def update(self, t: float, geo) -> None:
        if self.bin_start is None:
            self.bin_start = t
        while t >= self.bin_start + self.interval:
            self._flush(self.bin_start, self.bin_start + self.interval)
            self.bin_start += self.interval

        for cls, c in geo.full_frame.items():
            self._full_sum[cls] += c
            self._full_cnt[cls] += 1
            self._full_max[cls] = max(self._full_max[cls], c)
        for zid, counter in geo.zone_counts.items():
            for cls, c in counter.items():
                self._zone_sum[(zid, cls)] += c
                self._zone_cnt[(zid, cls)] += 1
                self._zone_max[(zid, cls)] = max(self._zone_max[(zid, cls)], c)
        for lid, dirs in geo.line_counts.items():
            for direction, counter in dirs.items():
                for cls, c in counter.items():
                    self._line_latest[(lid, direction, cls)] = c

    def _emit(self, row: dict) -> None:
        self.rows.append(row)
        if self._writer:
            self._writer.writerow(row)

    def _flush(self, start: float, end: float) -> None:
        r = lambda **k: {"interval_start": round(start, 3),
                         "interval_end": round(end, 3), "direction": "",
                         "id": "", "value_max": "", **k}
        # line crossings (deltas)
        for key, latest in sorted(self._line_latest.items()):
            base = self._line_base.get(key, 0)
            delta = latest - base
            if delta:
                lid, direction, cls = key
                self._emit(r(metric="line_crossing", id=lid, direction=direction,
                             entity=cls, value=delta))
            self._line_base[key] = latest
        # zone occupancy (avg + peak)
        for (zid, cls), s in sorted(self._zone_sum.items()):
            cnt = self._zone_cnt[(zid, cls)]
            self._emit(r(metric="zone_occupancy", id=zid, entity=cls,
                         value=round(s / cnt, 3) if cnt else 0,
                         value_max=self._zone_max[(zid, cls)]))
        # full-frame (avg + peak)
        for cls, s in sorted(self._full_sum.items()):
            cnt = self._full_cnt[cls]
            self._emit(r(metric="full_frame", entity=cls,
                         value=round(s / cnt, 3) if cnt else 0,
                         value_max=self._full_max[cls]))
        # reset per-interval accumulators (line_base persists)
        self._zone_sum.clear(); self._zone_cnt.clear(); self._zone_max.clear()
        self._full_sum.clear(); self._full_cnt.clear(); self._full_max.clear()

    def close(self) -> None:
        if self.bin_start is not None:
            self._flush(self.bin_start, self.bin_start + self.interval)
        if self._csv_file:
            self._csv_file.close()
        if self.json_path:
            summary = self._summary()
            with open(self.json_path, "w") as f:
                json.dump(summary, f, indent=2)

    def _summary(self) -> dict:
        totals: dict = {"line_crossings": defaultdict(int),
                        "intervals": len({(r_["interval_start"]) for r_ in self.rows})}
        for row in self.rows:
            if row["metric"] == "line_crossing":
                k = f"{row['id']}/{row['direction']}/{row['entity']}"
                totals["line_crossings"][k] += int(row["value"])
        totals["line_crossings"] = dict(totals["line_crossings"])
        totals["rows"] = len(self.rows)
        return totals
