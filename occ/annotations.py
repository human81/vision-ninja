"""Zone / line annotations: the user-drawn geometry, stored resolution-independent
(normalized 0..1) and convertible to the protobuf `StreamAnnotation`.

A zone is a polygon (NormalizedPolygon). A line is a 2-point polyline
(NormalizedPolyline); its positive direction follows the right-hand rule of the
directed segment v0 -> v1 (the editor draws an arrow so the user controls it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from proto import (
    StreamAnnotation,
    StreamAnnotationType,
    NormalizedVertex,
)

ZONE = "active_zone"
LINE = "crossing_line"


@dataclass
class Annotation:
    id: str
    type: str                                   # ZONE | LINE
    vertices: list[tuple[float, float]]         # normalized 0..1
    display_name: str = ""

    def to_proto(self) -> StreamAnnotation:
        sa = StreamAnnotation()
        sa.id = self.id
        sa.display_name = self.display_name or self.id
        verts = [NormalizedVertex(x=float(x), y=float(y)) for x, y in self.vertices]
        if self.type == ZONE:
            sa.type = StreamAnnotationType.STREAM_ANNOTATION_TYPE_ACTIVE_ZONE
            sa.active_zone.normalized_vertices.extend(verts)
        else:
            sa.type = StreamAnnotationType.STREAM_ANNOTATION_TYPE_CROSSING_LINE
            sa.crossing_line.normalized_vertices.extend(verts)
        return sa

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "display_name": self.display_name,
            "vertices": [[float(x), float(y)] for x, y in self.vertices],
        }


@dataclass
class AnnotationSet:
    annotations: list[Annotation] = field(default_factory=list)
    source: str = ""

    def zones(self) -> list[Annotation]:
        return [a for a in self.annotations if a.type == ZONE]

    def lines(self) -> list[Annotation]:
        return [a for a in self.annotations if a.type == LINE]

    def next_id(self, kind: str) -> str:
        prefix = "zone" if kind == ZONE else "line"
        n = sum(1 for a in self.annotations if a.type == kind) + 1
        existing = {a.id for a in self.annotations}
        while f"{prefix}{n}" in existing:
            n += 1
        return f"{prefix}{n}"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(
            {"source": self.source,
             "annotations": [a.to_dict() for a in self.annotations]}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "AnnotationSet":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        anns = [Annotation(
            id=a["id"], type=a["type"],
            vertices=[tuple(v) for v in a["vertices"]],
            display_name=a.get("display_name", "")) for a in raw.get("annotations", [])]
        return cls(annotations=anns, source=raw.get("source", ""))
