"""Media Library — every artifact the studio makes, searchable.

Momentum's Media Library centralizes generated/uploaded media so the agent can
recall and reuse it. Here it's the local twin: snapshots, recordings, gifs,
clips, contact sheets, analyzed frames and ingested images all auto-register as
they're produced, each with a caption (detection summary or a Gemini description)
and tags. The agent searches it in natural language. Persisted to
out/studio/library.json (files live under out/studio/exports + out/recordings).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from . import STUDIO_DIR

_MAX = 300


def _tok(s: str):
    return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(t) > 1]


class MediaLibrary:
    def __init__(self, path: str | None = None):
        self.path = Path(path or f"{STUDIO_DIR}/library.json")
        self._lock = threading.Lock()
        self._seq = 0
        self.items: list[dict] = []
        self._load()
        self.prune()                       # drop dead media on startup

    def _load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                self.items = d.get("items", [])
                self._seq = d.get("seq", len(self.items))
            except Exception:
                pass

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.parent / f".library.{os.getpid()}.tmp"
            tmp.write_text(json.dumps({"seq": self._seq,
                                       "items": self.items[-_MAX:]}, indent=2))
            os.replace(tmp, self.path)
        except Exception:
            pass

    def add(self, kind: str, path: str, caption: str = "", tags=None,
            source: str = "", meta: dict | None = None) -> dict:
        with self._lock:
            for it in self.items:                       # dedupe by path
                if it["path"] == path:
                    if caption:
                        it["caption"] = caption
                    if tags:
                        it["tags"] = sorted(set(it.get("tags", []) + list(tags)))
                    self._save()
                    return it
            self._seq += 1
            item = {"id": self._seq, "kind": kind, "path": path,
                    "name": os.path.basename(path), "caption": caption or kind,
                    "tags": sorted(set(tags or [])), "source": os.path.basename(source),
                    "ts": int(time.time() * 1000), "meta": meta or {}}
            self.items.append(item)
            if len(self.items) > _MAX:
                self.items = self.items[-_MAX:]
            self._save()
            return item

    def get(self, item_id: int) -> dict | None:
        with self._lock:
            return next((i for i in self.items if i["id"] == int(item_id)), None)

    def remove(self, item_id: int) -> bool:
        with self._lock:
            n = len(self.items)
            self.items = [i for i in self.items if i["id"] != int(item_id)]
            self._save()
            return len(self.items) < n

    def prune(self) -> int:
        """Drop dead entries: media whose file is gone (URLs / live sources kept)."""
        with self._lock:
            keep = [it for it in self.items
                    if str(it.get("path", "")).startswith(("rtsp", "http"))
                    or os.path.exists(it.get("path", ""))]
            removed = len(self.items) - len(keep)
            if removed:
                self.items = keep
                self._save()
            return removed

    def list(self, limit: int = 60) -> list[dict]:
        with self._lock:
            return list(reversed(self.items[-limit:]))

    def search(self, query: str, limit: int = 40) -> list[dict]:
        toks = _tok(query)
        with self._lock:
            if not toks:
                return list(reversed(self.items[-limit:]))
            scored = []
            for it in self.items:
                hay = " ".join([it["caption"], it["kind"], it["name"],
                                it.get("source", ""), " ".join(it.get("tags", []))]).lower()
                score = sum(3 if t in it["caption"].lower() else
                            (2 if t in " ".join(it.get("tags", [])).lower() else
                             (1 if t in hay else 0)) for t in toks)
                if score:
                    scored.append((score, it))
            scored.sort(key=lambda s: (s[0], s[1]["ts"]), reverse=True)
            return [it for _, it in scored[:limit]]
