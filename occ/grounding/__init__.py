"""Open-vocabulary VLM grounding — OFF the real-time hot path.

Used for (a) open-vocab zone setup ("count forklifts") and (b) sampled validation
of the fast detector. Models are 3-8B VLMs; weights load lazily on first ground()
so importing/constructing costs nothing. The interface is host-agnostic: a local
MPS model today, a GCP endpoint later, same `ground()` call.
"""

from .base import GroundBox, Grounder, parse_locate_anything_boxes, suggest_zone_from_boxes


def build_grounder(cfg):
    backend = cfg.get("grounding.backend", "locate_anything").lower()
    if backend == "locate_anything":
        from .locate_anything import LocateAnythingGrounder
        return LocateAnythingGrounder(cfg)
    if backend == "molmo2":
        from .molmo2 import Molmo2Grounder
        return Molmo2Grounder(cfg)
    raise ValueError(f"unknown grounding backend: {backend!r}")


__all__ = ["GroundBox", "Grounder", "build_grounder",
           "parse_locate_anything_boxes", "suggest_zone_from_boxes"]
