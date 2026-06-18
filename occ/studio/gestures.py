"""Touchless on-screen BUTTONS for the sponsored stores — OpenCV + MediaPipe.

Gesture *classification* (fist/peace/…) is noisy; the hand *position* is not. So we
drop gestures entirely and draw a column of virtual buttons on the live video —
◀ PREV, ▶ NEXT, ✓ TRY ON, ⇄ STORE, ✕ CLEAR — that you press by moving your hand so the
finger cursor hovers a button and HOLDS it briefly (dwell-to-click, with a fill ring).
Only the index fingertip (one landmark) drives it, gated by the palm detector so the
face never controls anything. The frame is mirrored so hand-right = screen-right.

Self-contained in the CV loop (occ.studio.pipeline) — works on /stream.mjpg and over
Live Voice. The agent can do all the same things by voice over BIDI (its own tools);
the two coexist.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request

import cv2
import numpy as np

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
              "gesture_recognizer/float16/1/gesture_recognizer.task")
_MODEL_DIR = "out/studio/models"
_MODEL_PATH = os.path.join(_MODEL_DIR, "gesture_recognizer.task")

# A hand (never the face) drives it — the palm DETECTOR confidence is the
# discriminator; the handedness score is a light secondary gate.
_HAND_MIN_SCORE = 0.5

# Virtual buttons rendered by the browser ON THE SELF-VIEW PiP (compact icons). We
# only own the geometry + hit-test + dwell here (in mirrored-display coords, since the
# PiP is mirrored). (id, centre-x, centre-y) normalised. A vertical column on the right.
_BTN_W, _BTN_H = 0.25, 0.135
_BTN = [
    ("prev",  0.80, 0.11),
    ("next",  0.80, 0.29),
    ("try",   0.80, 0.50),
    ("store", 0.80, 0.71),
    ("clear", 0.80, 0.89),
]
_ACTIONS = ("prev", "next", "try", "store", "clear")
_DWELL = 11          # frames to hover before a press fires (~0.7s @15fps)
_REPEAT = 16         # PREV/NEXT auto-repeat while HELD — SLOW (~1.1s/item) so you see each
_COUNTDOWN = 3.0     # seconds to "strike a pose" after TRY ON, before the capture


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class GestureBrowser:
    """One gesture-driven store browser bound to a pipeline. process() runs per
    frame; everything else (the carousel, the try-on) is driven from gestures."""

    def __init__(self):
        self.active = False
        self.store = "eyewear"
        self.idx = 0
        self.items: list[dict] = []
        self.busy = False
        self.banner = ""
        self.err = ""
        self._rec = None
        self._mp = None
        self._ts = 0
        self._hand_score = 0.0
        self._present = False
        self.result = None        # last try-on result to surface (apparel image)
        self.hover = None         # button id the cursor is over
        self.finger = None        # (x,y) mirrored display coords of the fingertip, or None
        self._dwell = 0           # frames the cursor has hovered the current button
        self._fired = False       # latched after a press until the cursor leaves
        self._rep = 0             # countdown to the next PREV/NEXT repeat (slow)
        self.counting = False     # 3-2-1 "strike a pose" countdown before a try-on
        self._count_until = 0.0
        self._pending = None      # the item to try on when the countdown ends
        self._lock = threading.Lock()

    # ---------------- lifecycle ----------------
    def toggle(self, on=None, store=None) -> dict:
        if store in ("eyewear", "apparel"):
            self.store = store
        self.active = (not self.active) if on is None else bool(on)
        if self.active:
            self._load()
            self.idx = min(self.idx, max(0, len(self.items) - 1))
            self.hover = None; self._dwell = 0; self._fired = False
            self.counting = False; self._pending = None
            self._ensure_recognizer()
        return self.state()

    def _slim(self, j):
        it = self.items[j]
        return {"idx": j, "title": it.get("title", ""), "price": it.get("price", ""),
                "brand": it.get("brand", ""), "img": it.get("img", ""),
                "current": j == self.idx}

    def state(self, radius: int = 4) -> dict:
        """Full browse state for the browser side-rails. `window` is the slice of
        items around the cursor so the UI renders a filmstrip without re-fetching."""
        n = len(self.items)
        cur = self._slim(self.idx) if (n and 0 <= self.idx < n) else {}
        window = [self._slim(j) for j in range(max(0, self.idx - radius),
                                               min(n, self.idx + radius + 1))]
        import math
        countdown = (int(math.ceil(max(0.0, self._count_until - time.time())))
                     if self.counting else 0)
        return {"active": self.active, "store": self.store, "idx": self.idx,
                "total": n, "busy": self.busy, "banner": self.banner,
                "hover": self.hover, "hand": self._present, "countdown": countdown,
                "dwell": (1.0 if self._fired else round(min(self._dwell / _DWELL, 1.0), 2)),
                "finger": ({"x": round(self.finger[0], 3), "y": round(self.finger[1], 3)}
                           if self.finger else None),
                "buttons": [{"id": b[0], "x": b[1], "y": b[2], "w": _BTN_W, "h": _BTN_H}
                            for b in _BTN],
                "result": self.result, "current": cur, "window": window}

    def _load(self):
        from . import tools as T
        cats = T._load_catalogs()
        self.items = list(cats.get(self.store, []))

    def _ensure_recognizer(self):
        if self._rec is not None:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._mp = mp
            opts = vision.GestureRecognizerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.7,   # rejects the face (palm detector)
                min_hand_presence_confidence=0.4,    # keep the fist tracked
                min_tracking_confidence=0.4)
            self._rec = vision.GestureRecognizer.create_from_options(opts)
        except Exception as e:
            self.err = f"gesture recogniser unavailable: {type(e).__name__}: {e}"
            self._rec = None

    # ---------------- per-frame ----------------
    def process(self, vis: np.ndarray, clean: np.ndarray):
        """Detect the index fingertip and run the touchless buttons (hover → dwell →
        press). NOTHING is drawn on the main canvas — the buttons, finger cursor and
        the 3-2-1 pose countdown are rendered by the browser ON THE SELF-VIEW PiP /
        canvas from state(). The main view stays a clean, un-mirrored try-on."""
        if not self.active:
            return
        if self._rec is None:
            self._ensure_recognizer()
        tip = self._fingertip(clean)
        self.finger = (1.0 - tip[0], tip[1]) if tip is not None else None   # mirror x for the PiP
        if self.counting:                          # "strike a pose" — freeze input, then fire
            if time.time() >= self._count_until:
                self.counting = False
                pending, self._pending = self._pending, None
                if pending is not None:
                    self._begin_tryon(pending)
            return
        self._update_buttons(self.finger)

    def _fingertip(self, clean):
        """Index fingertip (landmark 8) of a CONFIDENT hand, else None. The palm
        detector + handedness gate keep the face from ever being a cursor."""
        if self._rec is None:
            self._present = False
            return None
        try:
            rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            self._ts += 33
            res = self._rec.recognize_for_video(image, self._ts)
        except Exception:
            return None
        if not res.hand_landmarks or not res.handedness:
            self._present = False
            return None
        try:
            score = res.handedness[0][0].score
        except Exception:
            score = 0.0
        self._hand_score = score
        if score < _HAND_MIN_SCORE:
            self._present = False
            return None
        self._present = True
        tip = res.hand_landmarks[0][8]
        return (tip.x, tip.y)

    def _hit(self, cur):
        if cur is None:
            return None
        for bid, cx, cy in _BTN:
            if abs(cur[0] - cx) <= _BTN_W / 2 and abs(cur[1] - cy) <= _BTN_H / 2:
                return bid
        return None

    def _update_buttons(self, cur):
        hov = self._hit(cur)
        if hov != self.hover:                      # moved off / onto a (different) button
            self.hover = hov
            self._dwell = 0
            self._fired = False
            self._rep = 0
        if hov is None or self.busy:
            return
        self._dwell += 1
        if not self._fired:
            if self._dwell >= _DWELL:               # dwell complete → one press
                self.act(hov)
                self._fired = True
                self._rep = _REPEAT
        elif hov in ("prev", "next"):              # hold PREV/NEXT → SLOW repeat (see each item)
            self._rep -= 1
            if self._rep <= 0:
                self.act(hov)
                self._rep = _REPEAT

    # ---------------- the ONE engine: same actions for buttons AND the agent ----------------
    def act(self, action: str) -> dict:
        """Drive the live shopping engine. Called by the touchless buttons (dwell) AND
        by the agent's live_control tool over voice — there is no parallel path."""
        a = (action or "").lower().strip()
        if a in ("prev", "previous", "back", "<"):
            self.step(-1)
        elif a in ("next", "forward", ">"):
            self.step(1)
        elif a in ("try", "try_on", "tryon", "fit", "wear"):
            self._select()
        elif a in ("store", "switch", "switch_store", "toggle"):
            self._switch_store()
        elif a in ("clear", "reset", "live", "go_live", "off"):
            self._clear()
        return self.state()

    def goto(self, query: str) -> dict:
        """Jump the selection to the best-matching item by name (for the agent)."""
        from . import tools as T
        hits = T._search_catalog(query, store=self.store)
        if hits:
            title = (hits[0].get("title") or "").lower()
            for i, it in enumerate(self.items):
                if (it.get("title") or "").lower() == title:
                    self.idx = i
                    break
        return self.state()

    def step(self, d: int):
        if not self.items:
            return
        self.idx = min(max(self.idx + d, 0), len(self.items) - 1)

    # ---------------- actions ----------------
    def _select(self):
        """TRY ON → start a 3-2-1 'strike a pose' countdown, then capture/fit."""
        if not self.items or self.busy or self.counting:
            return
        self._pending = dict(self.items[self.idx])
        self._count_until = time.time() + _COUNTDOWN
        self.counting = True
        self.banner = "Strike a pose…"

    def _begin_tryon(self, it):
        title = it.get("title", "item")
        self.busy = True                      # → the browser shows the canvas loader
        self.result = None
        self.banner = ("Fitting " if self.store == "eyewear" else "Styling ") + title + "…"
        threading.Thread(target=self._try_on, args=(it,), daemon=True).start()

    def _try_on(self, it):
        from . import tools as T
        try:
            if self.store == "eyewear":
                T.try_eyewear(image=it.get("img", ""), label=it.get("title", ""))
                self.result = {"kind": "eyewear"}          # live AR — no canvas swap
            else:
                res = T.try_product(query=it.get("title", ""))
                out = (res or {}).get("output", "")
                self.result = ({"kind": "image",
                                "src": "/download/" + os.path.basename(out)} if out
                               else {"kind": "apparel"})
            self.banner = ("Wearing " if self.store == "eyewear" else "Look ready: ") \
                + it.get("title", "")
        except Exception as e:
            self.banner = f"couldn't try that on ({type(e).__name__})"
        finally:
            self.busy = False
            self._fired = True          # require leaving TRY before it can fire again
            threading.Timer(2.2, self._clear_banner).start()

    def _clear_banner(self):
        if not self.busy:
            self.banner = ""

    def _switch_store(self):
        self.store = "apparel" if self.store == "eyewear" else "eyewear"
        self.idx = 0
        self._load()
        self.banner = "Browsing " + ("eyewear · Ralba Optical" if self.store == "eyewear"
                                     else "apparel · Mode Marco")
        threading.Timer(1.6, self._clear_banner).start()

    def _clear(self):
        from . import tools as T
        try:
            T.go_live()
        except Exception:
            pass
        self.result = None             # drop the VTO result so the canvas returns to live
        self.busy = False
        self.counting = False; self._pending = None   # cancel any pending countdown
        self.banner = "Cleared — back to live"
        threading.Timer(1.4, self._clear_banner).start()
