"""Torch device resolution shared by the detectors.

Configs say `device: "mps"` (the M5 default). On a host without that backend — e.g.
Linux/Cloud Run — fall back to the best available one instead of crashing:
mps → cuda → cpu. "auto" means "best available".
"""

from __future__ import annotations

import logging

log = logging.getLogger("occ.device")


def pick_device(pref: str | None = "auto") -> str:
    import torch
    avail = [d for d, ok in (("mps", torch.backends.mps.is_available()),
                             ("cuda", torch.cuda.is_available())) if ok]
    pref = (pref or "auto").lower()
    if pref == "cpu" or (pref in avail):
        return pref
    best = avail[0] if avail else "cpu"
    if pref != "auto":
        log.warning("device %r unavailable here — using %r", pref, best)
    return best
