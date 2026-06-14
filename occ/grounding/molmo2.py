"""Molmo2 grounder (Ai2) — points/tracks/counts in images and video.

Skeleton matching the Grounder interface. Molmo2 emits points (and can count by
pointing); base.parse_locate_anything_boxes already understands <point> markup, so
wiring is: load allenai/Molmo2-* per its model card in _ensure_loaded(), then parse
the pointed output here. Left as an explicit follow-up so we don't pull a second
multi-GB VLM until needed — LocateAnything covers the box-grounding use today.
"""

from __future__ import annotations

import numpy as np

from .base import GroundBox


class Molmo2Grounder:
    name = "molmo2"

    def __init__(self, cfg):
        g = cfg.section("grounding")
        self.model_path = g.get("checkpoint", "allenai/Molmo2-7B-O")
        self.device_pref = g.get("device", "mps")

    def ground(self, frame: np.ndarray, prompt: str) -> list[GroundBox]:
        raise NotImplementedError(
            "Molmo2 grounder is a stub. Implement _ensure_loaded()/generate() per "
            "the allenai/Molmo2 model card; parse the pointed output with "
            "base.parse_locate_anything_boxes. Use backend 'locate_anything' for now.")
