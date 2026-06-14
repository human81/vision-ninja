"""Configuration: one YAML drives everything, with dotted CLI overrides.

    cfg = Config.load("configs/default.yaml", overrides=["detector.model=yolo11s",
                                                          "runtime.detect_every=3"])

Everything is a plain dict under the hood (easy to merge/override); typed accessors
live on small dataclasses for ergonomics. Class-group names ("vehicle") expand to
COCO ids via the `class_groups` table.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

# Standard COCO-80 class names (Ultralytics ordering) — used to translate the
# id-based class filter into names so name-keyed detectors (RF-DETR) can filter too.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(s: str) -> Any:
    """Turn a CLI string value into bool/int/float/list/str."""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if "," in s:
        return [_coerce(p.strip()) for p in s.split(",")]
    return s


def _set_dotted(d: dict, dotted: str, value) -> None:
    """Set an already-typed value at a nested dotted path, in place."""
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _apply_dotted(d: dict, dotted: str) -> None:
    """Apply 'a.b.c=value' into nested dict d, in place."""
    key, _, raw = dotted.partition("=")
    if not _:
        raise ValueError(f"override must be key=value, got: {dotted!r}")
    _set_dotted(d, key, _coerce(raw))


@dataclass
class Config:
    data: dict

    @classmethod
    def load(cls, path: str | Path | None = None,
             overrides: list[str] | None = None) -> "Config":
        base = yaml.safe_load(DEFAULT_CONFIG.read_text())
        if path and Path(path) != DEFAULT_CONFIG:
            user = yaml.safe_load(Path(path).read_text()) or {}
            base = _deep_merge(base, user)
        explicit: set[str] = set()
        for ov in overrides or []:
            _apply_dotted(base, ov)
            explicit.add(ov.split("=", 1)[0])
        cfg = cls(base)
        cfg._apply_source_overrides(explicit)
        return cfg

    def _apply_source_overrides(self, explicit_keys: set[str] = frozenset()) -> None:
        """Apply `source_overrides[<substr>]` blocks whose key appears in source.uri.
        Explicit CLI/user keys win, so an override never clobbers what you set."""
        uri = str(self.get("source.uri", ""))
        for substr, block in (self.data.get("source_overrides") or {}).items():
            if substr in uri:
                for dotted, val in (block or {}).items():
                    if dotted not in explicit_keys:
                        _set_dotted(self.data, dotted, val)

    # dotted getter: cfg.get("detector.model")
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for p in dotted.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def section(self, name: str) -> dict:
        return self.data.get(name, {})

    def resolve_class_ids(self) -> list[int] | None:
        """Expand detector.classes (names/groups/ids) to a flat sorted id list.
        Returns None to mean 'all classes'."""
        classes = self.get("detector.classes")
        if not classes:
            return None
        groups = self.get("class_groups", {})
        ids: set[int] = set()
        for c in classes:
            if isinstance(c, int):
                ids.add(c)
            elif isinstance(c, str) and c in groups:
                ids.update(groups[c])
            else:
                raise ValueError(f"unknown class/group: {c!r}")
        return sorted(ids)

    def resolve_class_names(self) -> set[str] | None:
        """The same class filter as a set of COCO names (for name-keyed detectors).
        Returns None to mean 'all classes'."""
        ids = self.resolve_class_ids()
        if ids is None:
            return None
        return {COCO80[i] for i in ids if 0 <= i < len(COCO80)}
