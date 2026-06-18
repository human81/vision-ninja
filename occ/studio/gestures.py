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

# Virtual buttons: (id, label, centre-x, centre-y) in NORMALISED mirrored-display
# coords. A vertical column down the right edge so it never covers the face.
_BTN_W, _BTN_H = 0.165, 0.12
_BTN = [
    ("prev",  "< PREV",  0.885, 0.15),
    ("next",  "NEXT >",  0.885, 0.30),
    ("try",   "TRY ON",  0.885, 0.50),
    ("store", "STORE",   0.885, 0.70),
    ("clear", "CLEAR",   0.885, 0.85),
]
_DWELL = 11          # frames to hover before a press fires (~0.7s)
_REPEAT = 5          # PREV/NEXT re-fire interval while held (frames; accelerates)
_REPEAT_MIN = 2


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
        self.hover = None         # button id the cursor is over (for the rail status)
        self._dwell = 0           # frames the cursor has hovered the current button
        self._fired = False       # latched after a press until the cursor leaves
        self._reps = 0            # consecutive PREV/NEXT repeats (for acceleration)
        self._rep = 0             # countdown to the next repeat
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
        return {"active": self.active, "store": self.store, "idx": self.idx,
                "total": n, "busy": self.busy, "banner": self.banner,
                "hover": self.hover, "hand": self._present,
                "dwell": (1.0 if self._fired else round(min(self._dwell / _DWELL, 1.0), 2)),
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
        """Detect the index fingertip, mirror the frame, then run the touchless
        buttons: hover-to-highlight, dwell-to-press. Drawn right on `vis`."""
        if not self.active:
            return
        if self._rec is None:
            self._ensure_recognizer()
        tip = self._fingertip(clean)               # (x, y) in clean coords, or None
        cv2.flip(vis, 1, vis)                       # mirror: hand-right = screen-right
        cur = (1.0 - tip[0], tip[1]) if tip is not None else None   # cursor in display coords
        self._update_buttons(cur)
        self._draw_buttons(vis, cur)

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
        for bid, _label, cx, cy in _BTN:
            if abs(cur[0] - cx) <= _BTN_W / 2 and abs(cur[1] - cy) <= _BTN_H / 2:
                return bid
        return None

    def _update_buttons(self, cur):
        hov = self._hit(cur)
        if hov != self.hover:                      # moved off / onto a (different) button
            self.hover = hov
            self._dwell = 0
            self._fired = False
            self._reps = 0
            self._rep = 0
        if hov is None or self.busy:
            return
        self._dwell += 1
        nav = hov in ("prev", "next")
        if not self._fired:
            if self._dwell >= _DWELL:               # dwell complete → press
                self._fire(hov)
                self._fired = True
                self._reps = 0
                self._rep = _REPEAT
        elif nav:                                   # hold PREV/NEXT to repeat (accelerating)
            self._rep -= 1
            if self._rep <= 0:
                self._fire(hov)
                self._reps += 1
                self._rep = max(_REPEAT_MIN, _REPEAT - self._reps // 3)

    def _fire(self, bid):
        if bid == "prev":
            self.step(-1)
        elif bid == "next":
            self.step(1)
        elif bid == "try":
            self._select()
        elif bid == "store":
            self._switch_store()
        elif bid == "clear":
            self._clear()

    def step(self, d: int):
        if not self.items:
            return
        self.idx = min(max(self.idx + d, 0), len(self.items) - 1)

    # ---------------- drawing ----------------
    def _draw_buttons(self, vis, cur):
        H, W = vis.shape[:2]
        for bid, label, cx, cy in _BTN:
            x0 = int((cx - _BTN_W / 2) * W); x1 = int((cx + _BTN_W / 2) * W)
            y0 = int((cy - _BTN_H / 2) * H); y1 = int((cy + _BTN_H / 2) * H)
            hov = (bid == self.hover)
            self._panel(vis, x0, y0, x1, y1,
                        (60, 120, 90) if hov else (22, 26, 34), 0.6 if hov else 0.42)
            col = (130, 245, 200) if hov else (190, 205, 222)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3 if hov else 1)
            self._ctext(vis, label, (x0 + x1) // 2, (y0 + y1) // 2, 0.62, col, 2)
            if hov and not self._fired and self._dwell > 0:     # dwell fill (bottom edge)
                frac = min(self._dwell / _DWELL, 1.0)
                cv2.rectangle(vis, (x0, y1 - 6), (x0 + int((x1 - x0) * frac), y1),
                              (120, 240, 190), -1)
            elif hov and self._fired:
                cv2.rectangle(vis, (x0, y1 - 6), (x1, y1), (120, 240, 190), -1)
        if cur is not None:                          # finger cursor
            px, py = int(cur[0] * W), int(cur[1] * H)
            cv2.circle(vis, (px, py), 15, (90, 240, 200), 2)
            cv2.circle(vis, (px, py), 4, (255, 255, 255), -1)

    @staticmethod
    def _panel(vis, x0, y0, x1, y1, color, alpha):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(vis.shape[1], x1), min(vis.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return
        sub = vis[y0:y1, x0:x1]
        cv2.addWeighted(np.full_like(sub, color, np.uint8), alpha, sub, 1 - alpha, 0, sub)

    @staticmethod
    def _ctext(vis, text, cx, cy, scale, color, th):
        (tw, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        cv2.putText(vis, text, (int(cx - tw / 2), int(cy + t_h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)

    # ---------------- actions ----------------
    def _select(self):
        if not self.items:
            return
        it = self.items[self.idx]
        title = it.get("title", "item")
        self.busy = True                      # → the browser shows the canvas loader
        self.result = None
        self.banner = ("Fitting " if self.store == "eyewear" else "Styling ") + title + "…"
        threading.Thread(target=self._try_on, args=(dict(it),), daemon=True).start()

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
        self.banner = "Cleared — back to live"
        threading.Timer(1.4, self._clear_banner).start()
