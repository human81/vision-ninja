"""Studio settings — typed, with a simulation axis and per-node modes.

Mirrors Momentum's settings model: a unified `simulation` level
(live / simulated / zero) plus per-node ON/OFF overrides. This is what lets the
studio run with **no API key** — the agent brain falls back to a deterministic
SimRunner and heavy ops meter as simulated. Persisted to out/studio/settings.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import STUDIO_DIR
from .neurons import NEURONS

# Simulation presets -> which neurons are REAL (everything else is faked/cheap).
# 'real_under_sim' neurons (agent_brain, vision_brain) stay on under 'simulated'.
_PRESET_REAL = {
    "live": {n["id"] for n in NEURONS},
    "simulated": {n["id"] for n in NEURONS if n.get("real_under_sim")}
                 | {"source", "detect", "track", "geometry", "render", "overlay"},
    "zero": set(),
}

# Per-task model wiring (Momentum's "AI Model Configuration"). The agent + media
# tools resolve their model here, so you can pick the model per capability.
DEFAULT_MODELS = {
    "agent": "gemini-2.5-flash",          # the reasoning brain (ADK root)
    "vision": "gemini-2.5-flash",         # image understanding / describe
    "image_edit": "gemini-2.5-flash-image",   # nano-banana edit + apparel try-on (fast/cheap)
    # Nano Banana PRO (GA) — the highest-fidelity image model. Used for the ONE-TIME,
    # cached eyewear canonical render where registration/cleanliness matters most.
    "image_pro": os.environ.get("STUDIO_NANO_MODEL", "gemini-3-pro-image"),
    "image_gen": "imagen-4.0-generate-001",   # new-image generation
    "video": "veo-3.0-fast-generate-001",     # Veo video generation
}
MODEL_OPTIONS = {
    "agent": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash",
              "gemini-2.5-flash-lite"],
    "vision": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "image_edit": ["gemini-2.5-flash-image", "gemini-3.1-flash-image"],
    # Nano Banana Pro (gemini-3-pro-image) + Nano Banana 2 (gemini-3.1-flash-image), both GA.
    "image_pro": ["gemini-3-pro-image", "gemini-3-pro-image-preview",
                  "gemini-3.1-flash-image", "gemini-2.5-flash-image"],
    "image_gen": ["imagen-4.0-generate-001", "imagen-4.0-fast-generate-001",
                  "imagen-3.0-generate-002"],
    "video": ["veo-3.0-fast-generate-001", "veo-3.0-generate-001",
              "veo-2.0-generate-001"],
}


@dataclass
class StudioSettings:
    agent_model: str = "gemini-2.5-flash"  # legacy alias for models['agent']
    vision_model: str = "gemini-2.5-flash"
    simulation: str = "simulated"          # live | simulated | zero
    autonomy: str = "autonomous"           # guided | autonomous
    narrate: bool = True
    points_budget: float = 100_000.0
    node_modes: dict = field(default_factory=dict)   # node_id -> 'on'|'off'
    models: dict = field(default_factory=lambda: dict(DEFAULT_MODELS))  # per-task

    def model_for(self, task: str) -> str:
        return (self.models or {}).get(task) or DEFAULT_MODELS.get(task, "gemini-2.5-flash")

    # ---- persistence ----
    @classmethod
    def load(cls, path: str | None = None) -> "StudioSettings":
        p = Path(path or f"{STUDIO_DIR}/settings.json")
        if p.exists():
            try:
                d = json.loads(p.read_text())
                known = {f for f in cls().__dict__}
                return cls(**{k: v for k, v in d.items() if k in known})
            except Exception:
                pass
        return cls()

    def save(self, path: str | None = None):
        p = Path(path or f"{STUDIO_DIR}/settings.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    # ---- resolution ----
    def node_mode(self, node: str) -> str:
        """ON if the explicit override says so, else by the simulation preset."""
        if node in self.node_modes:
            return self.node_modes[node]
        return "on" if node in _PRESET_REAL.get(self.simulation, set()) else "off"

    def is_node_fake(self, node: str) -> bool:
        return self.node_mode(node) == "off"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolved_modes"] = {n["id"]: self.node_mode(n["id"]) for n in NEURONS}
        d["model_options"] = MODEL_OPTIONS
        return d

    def update(self, patch: dict):
        for k, v in patch.items():
            if k in ("node_modes", "models") and isinstance(v, dict):
                getattr(self, k).update(v)
            elif hasattr(self, k):
                setattr(self, k, v)
        self.agent_model = self.model_for("agent")        # keep alias in sync
        self.save()
