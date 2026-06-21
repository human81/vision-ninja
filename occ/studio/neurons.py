"""The neuron graph — every metered charge point in the studio.

Mirrors Momentum's `billing/neurons.ts`: a TRUE data-flow graph (senses ->
agent_brain -> outputs) where `feeds` are edges. Each neuron is a place where
work happens and cost accrues. Local CV work has no dollar cost, so we meter it
in **compute points** (a synthetic unit that makes the cost of autonomy
*visible*); the agent brain (Gemini) additionally carries real USD priced from
token tables.

Add a neuron -> give it an honest `feeds` edge so the brain graph stays whole.
"""

from __future__ import annotations

# ---- regions (columns in the brain graph, left -> right) ----
SENSES, PERCEPTION, ANALYTICS, RENDER, EXPORT, BRAIN, AGENT = (
    "Senses", "Perception", "Analytics", "Render", "Export", "Brain", "Agent")

# Each neuron: id, label, region, modality, unit, feeds (downstream ids),
# real_under_sim (stays REAL even in 'simulated' mode — the brain must think).
NEURONS: list[dict] = [
    {"id": "source", "label": "Source / Capture", "region": SENSES,
     "modality": "frames", "unit": "frame", "feeds": ["detect", "track", "render"]},
    {"id": "detect", "label": "Object Detection", "region": PERCEPTION,
     "modality": "vision", "unit": "inference", "feeds": ["track", "geometry"]},
    {"id": "grounding", "label": "Open-Vocab Grounding", "region": PERCEPTION,
     "modality": "vision", "unit": "inference", "feeds": ["detect", "brain"]},
    {"id": "track", "label": "Multi-Object Tracking", "region": PERCEPTION,
     "modality": "vision", "unit": "frame", "feeds": ["geometry", "render"]},
    {"id": "geometry", "label": "Spatial Analytics", "region": ANALYTICS,
     "modality": "math", "unit": "frame", "feeds": ["render", "brain"]},
    {"id": "overlay", "label": "Dynamic Overlays", "region": RENDER,
     "modality": "render", "unit": "overlay-frame", "feeds": ["render"]},
    {"id": "render", "label": "Frame Compositor", "region": RENDER,
     "modality": "render", "unit": "frame", "feeds": ["export"]},
    {"id": "ffmpeg", "label": "ffmpeg Export", "region": EXPORT,
     "modality": "encode", "unit": "out-second", "feeds": []},
    {"id": "snapshot", "label": "Snapshot / Frame Grab", "region": EXPORT,
     "modality": "encode", "unit": "op", "feeds": []},
    {"id": "vision_brain", "label": "Vision Brain", "region": BRAIN,
     "modality": "memory", "unit": "synthesis", "feeds": ["agent_brain"],
     "real_under_sim": True},
    {"id": "agent_brain", "label": "Agent Brain (Gemini)", "region": AGENT,
     "modality": "reasoning", "unit": "1k-tokens", "real_under_sim": True,
     "feeds": ["detect", "grounding", "track", "geometry", "overlay",
               "render", "ffmpeg", "snapshot", "vision_brain"]},
]

NEURON_BY_ID = {n["id"]: n for n in NEURONS}

# A checkpoint is the concrete event name recorded by a tool; it maps to a
# neuron. Prefix families (e.g. "ffmpeg:gif") roll up to their neuron.
_CHECKPOINT_ALIAS = {
    "detect": "detect", "grounding": "grounding", "ground": "grounding",
    "track": "track", "geometry": "geometry", "overlay": "overlay",
    "overlay_exec": "overlay", "render": "render", "snapshot": "snapshot",
    "agent_brain": "agent_brain", "brain": "vision_brain",
    "vision_brain": "vision_brain", "source": "source",
}


def node_for_checkpoint(cp: str) -> str:
    """Resolve a checkpoint (possibly 'family:concrete') to a neuron id."""
    if cp in _CHECKPOINT_ALIAS:
        return _CHECKPOINT_ALIAS[cp]
    fam = cp.split(":", 1)[0]
    if fam in NEURON_BY_ID:
        return fam
    if fam in _CHECKPOINT_ALIAS:
        return _CHECKPOINT_ALIAS[fam]
    return fam if fam in NEURON_BY_ID else "agent_brain"


# ---- POINTS pricing (synthetic compute cost) ----------------------------
# Detection points scale with model tier — the lever the agent trades for speed.
_DETECT_POINTS = {"yolo11n.pt": 1.0, "yolo11s.pt": 2.0, "yolo11m.pt": 4.0,
                  "yolo11l.pt": 7.0, "yolo11x.pt": 12.0,
                  "rfdetr": 6.0, "owlv2": 20.0}


def _detect_points(model: str) -> float:
    if not model:
        return 3.0
    if model in _DETECT_POINTS:
        return _DETECT_POINTS[model]
    for k, v in _DETECT_POINTS.items():
        if k.rstrip(".pt") in model or model in k:
            return v
    return 3.0


# Per-unit point cost by neuron. Callables get the (model, units) context.
_POINTS = {
    "source": lambda model, u: 0.05 * u.get("frames", 1),
    "detect": lambda model, u: _detect_points(model) * u.get("inferences", 1),
    "grounding": lambda model, u: 20.0 * u.get("inferences", 1),
    "track": lambda model, u: 0.3 * u.get("frames", 1),
    "geometry": lambda model, u: 0.15 * u.get("frames", 1),
    "overlay": lambda model, u: 0.2 * u.get("overlay_frames", 1),
    "render": lambda model, u: 0.1 * u.get("frames", 1),
    "ffmpeg": lambda model, u: 2.0 * u.get("out_seconds", 1) + 1.0 * u.get("ops", 0),
    "snapshot": lambda model, u: 0.5 * u.get("ops", 1),
    "vision_brain": lambda model, u: 3.0 * u.get("synthesis", 1),
    "agent_brain": lambda model, u: 0.0,   # brain is priced in USD, not points
}


def points_for(checkpoint: str, model: str, units: dict) -> float:
    node = node_for_checkpoint(checkpoint)
    fn = _POINTS.get(node)
    return round(float(fn(model, units or {})), 3) if fn else 0.0


# ---- USD pricing for the agent brain (token tables, per 1M tokens) -------
# Single source of truth, like Momentum's pricing.ts. (input, output) USD/1M.
TOKEN_PRICES_USD = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.0-flash": (0.15, 0.60),
    # Gemma 4 runs LOCALLY (LiteRT-LM on Apple Silicon) → $0 marginal cost.
    "gemma-4-12b-it": (0.0, 0.0),
    "gemma-4-e2b-it": (0.0, 0.0),
    "medgemma-4b-it": (0.0, 0.0),       # local medical vision (transformers/MPS) → $0
}


def usd_micros_for(model: str, input_tokens: int, output_tokens: int) -> int:
    """USD micro-dollars for a brain call (1 micro = $1e-6)."""
    inp, out = TOKEN_PRICES_USD.get(model, TOKEN_PRICES_USD["gemini-2.5-flash"])
    usd = (input_tokens * inp + output_tokens * out) / 1_000_000.0
    return int(round(usd * 1_000_000))


def graph_for_ui() -> dict:
    """Neuron graph for the brain panel (nodes + edges)."""
    nodes = [{"id": n["id"], "label": n["label"], "region": n["region"],
              "modality": n["modality"], "unit": n["unit"]} for n in NEURONS]
    edges = [{"from": n["id"], "to": t} for n in NEURONS for t in n.get("feeds", [])]
    regions = [SENSES, PERCEPTION, ANALYTICS, RENDER, EXPORT, BRAIN, AGENT]
    return {"nodes": nodes, "edges": edges, "regions": regions}
