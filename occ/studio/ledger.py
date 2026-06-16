"""Usage ledger — meters every studio op in compute points (+ USD for the brain).

Mirrors Momentum's three-layer billing (events -> daily rollup -> wallet) but
local + single-tenant: one append-only JSON file. Every tool calls `record()`;
the AI Meter panel reads `summary()`. Costs are attributed to a **neuron**, so
the brain graph can pulse with live spend exactly like Momentum's.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import STUDIO_DIR
from .neurons import node_for_checkpoint, points_for, usd_micros_for

_MAX_EVENTS = 500


@dataclass
class UsageEvent:
    id: int
    ts: int                       # ms epoch
    checkpoint: str
    node: str
    model: str
    points: float
    usd_micros: int
    simulated: bool
    label: str
    units: dict = field(default_factory=dict)


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


class Ledger:
    def __init__(self, path: str | None = None):
        self.path = Path(path or f"{STUDIO_DIR}/ledger.json")
        self._lock = threading.Lock()
        self._seq = 0
        self.events: list[dict] = []
        self.daily: dict[str, dict] = {}
        self.wallet = {"points_spent": 0.0, "usd_micros_spent": 0,
                       "sim_points_spent": 0.0, "sim_usd_micros_spent": 0,
                       "points_budget": 100_000.0}
        self._load()

    # ---- persistence ----
    def _load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                self.events = d.get("events", [])[-_MAX_EVENTS:]
                self.daily = d.get("daily", {})
                self.wallet.update(d.get("wallet", {}))
                self._seq = d.get("seq", len(self.events))
            except Exception:
                pass

    def _save(self):
        # Called under self._lock. Unique temp per write + atomic replace, and
        # never raise into the caller (the pipeline thread meters via this).
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.parent / f".ledger.{os.getpid()}.{self._seq}.tmp"
            tmp.write_text(json.dumps({
                "seq": self._seq, "events": self.events[-_MAX_EVENTS:],
                "daily": self.daily, "wallet": self.wallet}, indent=2))
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ---- record ----
    def record(self, checkpoint: str, *, model: str = "", units: dict | None = None,
               simulated: bool = False, input_tokens: int = 0, output_tokens: int = 0,
               label: str = "", persist: bool = True) -> dict:
        node = node_for_checkpoint(checkpoint)
        pts = points_for(checkpoint, model, units or {})
        usd = (usd_micros_for(model or "gemini-2.5-flash", input_tokens, output_tokens)
               if node == "agent_brain" else 0)
        with self._lock:
            self._seq += 1
            ev = UsageEvent(id=self._seq, ts=int(time.time() * 1000),
                            checkpoint=checkpoint, node=node, model=model,
                            points=pts, usd_micros=usd, simulated=simulated,
                            label=label or checkpoint, units=units or {})
            self.events.append(asdict(ev))
            if len(self.events) > _MAX_EVENTS:
                self.events = self.events[-_MAX_EVENTS:]
            # daily rollup
            day = self.daily.setdefault(_today(), {"by_node": {}, "events": 0,
                                                   "points": 0.0, "usd_micros": 0})
            bn = day["by_node"].setdefault(node, {"points": 0.0, "usd_micros": 0,
                                                  "events": 0})
            bn["points"] += pts; bn["usd_micros"] += usd; bn["events"] += 1
            day["events"] += 1; day["points"] += pts; day["usd_micros"] += usd
            # wallet
            if simulated:
                self.wallet["sim_points_spent"] += pts
                self.wallet["sim_usd_micros_spent"] += usd
            else:
                self.wallet["points_spent"] += pts
                self.wallet["usd_micros_spent"] += usd
            if persist:
                self._save()
            return asdict(ev)

    # ---- read ----
    def summary(self, recent: int = 40) -> dict:
        with self._lock:
            day = self.daily.get(_today(), {"by_node": {}, "events": 0,
                                            "points": 0.0, "usd_micros": 0})
            by_node = {k: {"points": round(v["points"], 2),
                           "usd": round(v["usd_micros"] / 1e6, 4),
                           "events": v["events"]}
                       for k, v in day["by_node"].items()}
            w = self.wallet
            return {
                "today": {"points": round(day["points"], 2),
                          "usd": round(day["usd_micros"] / 1e6, 4),
                          "events": day["events"]},
                "by_node": by_node,
                "wallet": {
                    "points_spent": round(w["points_spent"], 2),
                    "points_budget": w["points_budget"],
                    "points_remaining": round(w["points_budget"] - w["points_spent"], 2),
                    "usd_spent": round(w["usd_micros_spent"] / 1e6, 4),
                    "sim_points_spent": round(w["sim_points_spent"], 2),
                    "sim_usd_spent": round(w["sim_usd_micros_spent"] / 1e6, 4)},
                "recent": list(reversed(self.events[-recent:])),
            }

    def set_budget(self, points: float):
        with self._lock:
            self.wallet["points_budget"] = float(points)
            self._save()
