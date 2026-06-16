"""The Vision Brain — persistent scene & task understanding.

Momentum's Brand Brain centralizes "what does this brand sound like?" so every
tool is brand-aware without re-explaining. The Vision Brain is the visual twin:
"what is in this scene, what has the user asked for, what overlays exist" — a
versioned memory the agent folds live observations into and injects into its own
prompt, so autonomy compounds across turns instead of starting cold each time.

Persisted to out/studio/brain.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import STUDIO_DIR


class VisionBrain:
    def __init__(self, path: str | None = None):
        self.path = Path(path or f"{STUDIO_DIR}/brain.json")
        self.d: dict = {
            "version": 0, "updated_at": 0,
            "source": "", "resolution": [0, 0], "fps": 0.0,
            "task": "", "task_history": [],
            "observations": {"classes": {}, "tracks_peak": 0, "last": {}},
            "zones": {}, "lines": {},
            "overlays": [],            # [{name, intent}]
            "notes": [],               # [{key, text, ts}]
        }
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.d.update(json.loads(self.path.read_text()))
            except Exception:
                pass

    def _bump(self):
        self.d["version"] += 1
        self.d["updated_at"] = int(time.time() * 1000)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, indent=2))

    # ---- folding live observations ----
    def observe(self, stats: dict, source: str = "", resolution=None, fps=None):
        if source:
            self.d["source"] = source
        if resolution:
            self.d["resolution"] = list(resolution)
        if fps:
            self.d["fps"] = round(float(fps), 1)
        obs = self.d["observations"]
        for cls, cnt in (stats.get("full_frame") or {}).items():
            slot = obs["classes"].setdefault(cls, {"peak": 0, "last": 0})
            slot["last"] = cnt
            slot["peak"] = max(slot["peak"], cnt)
        obs["tracks_peak"] = max(obs.get("tracks_peak", 0), stats.get("tracks", 0))
        obs["last"] = {"full_frame": stats.get("full_frame", {}),
                       "tracks": stats.get("tracks", 0), "fps": stats.get("fps")}
        for zid, c in (stats.get("zones") or {}).items():
            self.d["zones"][zid] = c
        for lid, d in (stats.get("lines") or {}).items():
            self.d["lines"][lid] = d
        self._bump()

    # ---- task + notes ----
    def set_task(self, goal: str):
        if goal and goal != self.d.get("task"):
            self.d["task"] = goal
            self.d["task_history"] = (self.d.get("task_history", []) + [
                {"goal": goal, "ts": int(time.time() * 1000)}])[-20:]
            self._bump()

    def remember(self, key: str, text: str):
        notes = [n for n in self.d["notes"] if n.get("key") != key]
        notes.append({"key": key, "text": text, "ts": int(time.time() * 1000)})
        self.d["notes"] = notes[-50:]
        self._bump()

    def register_overlay(self, name: str, intent: str):
        ov = [o for o in self.d["overlays"] if o.get("name") != name]
        ov.append({"name": name, "intent": intent})
        self.d["overlays"] = ov
        self._bump()

    def forget_overlay(self, name: str):
        self.d["overlays"] = [o for o in self.d["overlays"] if o.get("name") != name]
        self._bump()

    # ---- read ----
    def to_dict(self) -> dict:
        return dict(self.d)

    def prompt_context(self) -> str:
        """Compact brief the agent injects into its system prompt each turn."""
        d = self.d
        classes = ", ".join(f"{k}(peak {v['peak']})"
                            for k, v in d["observations"]["classes"].items()) or "none yet"
        res = d["resolution"]
        ov = ", ".join(o["name"] for o in d["overlays"]) or "none"
        notes = "; ".join(f"{n['key']}: {n['text']}" for n in d["notes"][-6:]) or "none"
        lines = [
            f"SOURCE: {d['source'] or 'unset'}  {res[0]}x{res[1]} @ {d['fps']}fps",
            f"CURRENT TASK: {d['task'] or '(none set)'}",
            f"OBSERVED CLASSES: {classes}",
            f"PEAK TRACKS: {d['observations'].get('tracks_peak', 0)}",
            f"ZONES: {list(d['zones'].keys()) or 'none'}  LINES: {list(d['lines'].keys()) or 'none'}",
            f"ACTIVE OVERLAYS: {ov}",
            f"BRAIN NOTES: {notes}",
        ]
        return "\n".join(lines)
