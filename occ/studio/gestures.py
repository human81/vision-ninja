"""Hands-free gesture browsing of the sponsored stores — OpenCV + MediaPipe magic.

Wave your hand to scroll a live product carousel drawn ON the video; make a FIST to
grab + try the centred item on (real-time AR for eyewear, generative for apparel);
a ✌ V switches eyewear↔apparel; 👎 thumbs-down clears. It's self-contained in the CV
loop (occ.studio.pipeline) — no JS round-trip — so it works on /stream.mjpg and over
Live Voice alike.

UX principle: it must be OBVIOUS. Every control is captioned on screen, the tracked
hand has a glowing cursor, the centred product shows name+price, and dangerous
actions (try-on) require a short HOLD with a visible progress ring so nothing fires
by accident. Four gestures, each with a permanent on-screen hint. That's the whole
vocabulary.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request

import cv2

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
              "gesture_recognizer/float16/1/gesture_recognizer.task")
_MODEL_DIR = "out/studio/models"
_MODEL_PATH = os.path.join(_MODEL_DIR, "gesture_recognizer.task")

# how long (frames) a FIST must be held to commit a try-on, and the cool-down
# (frames) between discrete actions so one gesture fires exactly once.
_HOLD_FRAMES = 8
_COOLDOWN = 18
# Browsing must be driven by a REAL HAND, never the face. The discriminator is the
# PALM DETECTOR confidence (min_hand_detection_confidence) — a face doesn't pass it.
# Presence/tracking are kept LOW so a fist (a very different shape than an open palm)
# stays tracked, and the handedness gate is a light secondary filter only.
_HAND_MIN_SCORE = 0.55

# --- browse feel: a JOG-WHEEL, not absolute pointing. The hand's horizontal offset
# from centre sets the SCROLL RATE (eased), so small motions barely move and you stop
# on an item by re-centring your hand. Smoothing kills jitter/snap. Tune here. ---
_BROWSE_SMOOTH = 0.22       # EMA on hand x (lower = smoother / more lag)
_BROWSE_DEADZONE = 0.11     # |x-0.5| within this → HOLD (no scroll) — generous, calm
_BROWSE_MAX_RATE = 3.2      # items / frame at full deflection (eased toward centre)
_BROWSE_EASE = 2.8          # >1 ⇒ very fine control near centre, fast only at the edges
# ONLY an open hand browses. Fist=try-on, Peace=switch, Thumbs-down=clear are
# independent — moving the hand during them must NEVER scroll the catalogue.
_BROWSE_GESTURES = ("Open_Palm", "Pointing_Up")


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
        self._hold = 0            # consecutive FIST frames
        self._fired_hold = False  # latched until the fist releases
        self._cool = 0           # frames until the next discrete action may fire
        self._last_gesture = "—"
        self._hand_score = 0.0
        self._hx = None           # smoothed hand x (EMA) for the jog-wheel
        self._pos = 0.0           # fractional catalog position; idx = round(_pos)
        self.result = None        # last try-on result to surface (apparel image)
        self._lock = threading.Lock()

    # ---------------- lifecycle ----------------
    def toggle(self, on=None, store=None) -> dict:
        if store in ("eyewear", "apparel"):
            self.store = store
        self.active = (not self.active) if on is None else bool(on)
        if self.active:
            self._load()
            self.idx = min(self.idx, max(0, len(self.items) - 1))
            self._pos = float(self.idx)
            self._hx = None
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
                "gesture": self._last_gesture, "hold": round(self._hold / _HOLD_FRAMES, 2),
                "hand": self._last_gesture != "—", "hand_score": round(self._hand_score, 2),
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
        """Recognise the HAND on `clean` and act. Nothing is drawn on the video — all
        browse feedback lives in the browser side-rails, so the user's face is never
        touched and there is no pointer on the face."""
        if not self.active:
            return
        if self._rec is None:
            self._ensure_recognizer()
        gesture, hand = self._recognize(clean)
        self._last_gesture = gesture if hand is not None else "—"
        if self._cool > 0:
            self._cool -= 1
        self._drive(gesture, hand, vis.shape[1], vis.shape[0])

    def _recognize(self, clean):
        """Return (gesture, hand) ONLY for a confident, real hand — else (None, None).
        A face (or anything that isn't clearly a hand) never drives browsing."""
        if self._rec is None:
            return None, None
        try:
            rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            self._ts += 33
            res = self._rec.recognize_for_video(image, self._ts)
        except Exception:
            return None, None
        if not res.hand_landmarks or not res.handedness:
            return None, None
        try:
            score = res.handedness[0][0].score
        except Exception:
            score = 0.0
        self._hand_score = score
        if score < _HAND_MIN_SCORE:          # not confidently a hand → ignore (e.g. a face)
            return None, None
        lm = res.hand_landmarks[0]
        name = ""
        if res.gestures and res.gestures[0]:
            name = res.gestures[0][0].category_name or ""
        return name, lm

    def _drive(self, gesture, hand, W, H):
        # Hold accumulates while a FIST is seen and DECAYS otherwise (so a single
        # dropped/misread frame doesn't reset the try-on) — robust to recogniser jitter.
        if gesture == "Closed_Fist" and hand is not None:
            self._hold = min(self._hold + 1, _HOLD_FRAMES)
        else:
            self._hold = max(0, self._hold - 2)
        if self._hold == 0:
            self._fired_hold = False
        if self._hold >= _HOLD_FRAMES and not self._fired_hold and not self.busy:
            self._fired_hold = True
            self._select()

        if hand is None:
            self._hx = None
            return
        # JOG-WHEEL browse, ONLY on an open-hand gesture (so Fist/Peace/Thumbs-down
        # never scroll). The hand's offset from centre sets the SCROLL RATE (eased), so
        # barely moving / a centred hand HOLDS the catalog and you settle on an item;
        # only a deliberate left/right offset scrolls. Smoothed to kill jitter.
        if self.items and not self.busy and gesture in _BROWSE_GESTURES:
            x = hand[8].x
            self._hx = x if self._hx is None else \
                (1 - _BROWSE_SMOOTH) * self._hx + _BROWSE_SMOOTH * x
            off = self._hx - 0.5
            if abs(off) > _BROWSE_DEADZONE:
                mag = min((abs(off) - _BROWSE_DEADZONE) / (0.42 - _BROWSE_DEADZONE), 1.0)
                rate = (1.0 if off > 0 else -1.0) * (mag ** _BROWSE_EASE) * _BROWSE_MAX_RATE
                self._pos = min(max(self._pos + rate, 0.0), len(self.items) - 1)
                self.idx = int(round(self._pos))
        else:
            self._hx = None          # not browsing → re-anchor on the next open hand

        # discrete one-shot actions (cool-down gated)
        if self._cool == 0 and not self.busy:
            if gesture == "Victory":
                self._switch_store(); self._cool = _COOLDOWN
            elif gesture == "Thumb_Down":
                self._clear(); self._cool = _COOLDOWN

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
            self._cool = _COOLDOWN
            threading.Timer(2.2, self._clear_banner).start()

    def _clear_banner(self):
        if not self.busy:
            self.banner = ""

    def _switch_store(self):
        self.store = "apparel" if self.store == "eyewear" else "eyewear"
        self.idx = 0
        self._pos = 0.0
        self._hx = None
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

    # NOTE: deliberately NOTHING is drawn on the video frame. The face stays
    # untouched; every browse cue (current frame, filmstrip, hold progress, status)
    # lives in the browser side-rails, driven by state().
