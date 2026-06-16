"""Shared studio context for tools (Momentum's module-level service pattern).

The agent's tools are plain functions ADK introspects; they reach the live
pipeline / overlays / ledger / brain / settings through this single context set
once at agent-build time. Keeps tool signatures clean and ADK-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StudioContext:
    pipe: object
    overlays: object
    ledger: object
    brain: object
    settings: object
    library: object = None
    sources: object = None


CTX: "StudioContext | None" = None


def set_context(c: "StudioContext"):
    global CTX
    CTX = c


def ctx() -> "StudioContext":
    if CTX is None:
        raise RuntimeError("studio context not initialised")
    return CTX
