"""Grounding interface + the deterministic, model-free helpers (output parsing,
zone suggestion). These are unit-tested without downloading any weights."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..annotations import Annotation, ZONE

# LocateAnything emits boxes as <box><x1><y1><x2><y2></box>, coords 0..1000.
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# Some grounding VLMs emit points as <point><x><y></point>.
_POINT_RE = re.compile(r"<point><(\d+)><(\d+)></point>")


@dataclass
class GroundBox:
    x1: float
    y1: float
    x2: float
    y2: float          # all normalized 0..1
    label: str = ""
    score: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Grounder(Protocol):
    name: str

    def ground(self, frame, prompt: str) -> list[GroundBox]:
        """Run the VLM on one BGR frame; return normalized boxes for `prompt`."""
        ...


def parse_locate_anything_boxes(answer: str, label: str = "") -> list[GroundBox]:
    """Parse <box>..</box> (and <point>..</point>) markup into normalized boxes."""
    out: list[GroundBox] = []
    for m in _BOX_RE.finditer(answer):
        x1, y1, x2, y2 = (int(g) / 1000.0 for g in m.groups())
        out.append(GroundBox(min(x1, x2), min(y1, y2),
                             max(x1, x2), max(y1, y2), label=label))
    for m in _POINT_RE.finditer(answer):          # points → tiny boxes
        x, y = int(m.group(1)) / 1000.0, int(m.group(2)) / 1000.0
        out.append(GroundBox(x - 0.01, y - 0.01, x + 0.01, y + 0.01, label=label))
    return out


def suggest_zone_from_boxes(boxes: list[GroundBox], name: str = "zone",
                            pad: float = 0.02) -> Annotation | None:
    """Propose a rectangular active-zone covering all grounded boxes (padded)."""
    if not boxes:
        return None
    x1 = max(0.0, min(b.x1 for b in boxes) - pad)
    y1 = max(0.0, min(b.y1 for b in boxes) - pad)
    x2 = min(1.0, max(b.x2 for b in boxes) + pad)
    y2 = min(1.0, max(b.y2 for b in boxes) + pad)
    verts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return Annotation(id=name, type=ZONE, vertices=verts, display_name=name)
