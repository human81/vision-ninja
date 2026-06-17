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
import numpy as np

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
              "gesture_recognizer/float16/1/gesture_recognizer.task")
_MODEL_DIR = "out/studio/models"
_MODEL_PATH = os.path.join(_MODEL_DIR, "gesture_recognizer.task")

# how long (frames) a FIST must be held to commit a try-on, and the cool-down
# (frames) between discrete actions so one gesture fires exactly once.
_HOLD_FRAMES = 10
_COOLDOWN = 18
_VISIBLE = 5            # carousel items shown (centred on current index)


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class _ThumbCache:
    """Lazily fetch + decode product thumbnails (background thread; never blocks the
    render loop). Keyed by image URL."""

    def __init__(self, height: int = 150):
        self.h = height
        self._cache: dict[str, np.ndarray] = {}
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def get(self, url: str):
        if not url:
            return None
        with self._lock:
            if url in self._cache:
                return self._cache[url]
            if url in self._pending:
                return None
            self._pending.add(url)
        threading.Thread(target=self._fetch, args=(url,), daemon=True).start()
        return None

    def _fetch(self, url: str):
        img = None
        try:
            if url.startswith(("http://", "https://")):
                req = urllib.request.Request(url, headers={"User-Agent": "occ-studio"})
                data = urllib.request.urlopen(req, timeout=8).read()
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            elif os.path.exists(url):
                img = cv2.imread(url)
        except Exception:
            img = None
        if img is not None:
            scale = self.h / img.shape[0]
            img = cv2.resize(img, (max(1, int(img.shape[1] * scale)), self.h))
        with self._lock:
            self._cache[url] = img        # cache None too → don't refetch a 404
            self._pending.discard(url)


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
        self._cursor = None      # (x,y) px of the index fingertip
        self._thumbs = _ThumbCache()
        self._lock = threading.Lock()

    # ---------------- lifecycle ----------------
    def toggle(self, on=None, store=None) -> dict:
        if store in ("eyewear", "apparel"):
            self.store = store
        self.active = (not self.active) if on is None else bool(on)
        if self.active:
            self._load()
            self.idx = min(self.idx, max(0, len(self.items) - 1))
            self._ensure_recognizer()
        return self.state()

    def state(self) -> dict:
        it = self.items[self.idx] if (self.items and 0 <= self.idx < len(self.items)) else {}
        return {"active": self.active, "store": self.store, "idx": self.idx,
                "total": len(self.items), "busy": self.busy,
                "current": {"title": it.get("title", ""), "price": it.get("price", ""),
                            "brand": it.get("brand", "")}}

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
                running_mode=vision.RunningMode.VIDEO, num_hands=1)
            self._rec = vision.GestureRecognizer.create_from_options(opts)
        except Exception as e:
            self.err = f"gesture recogniser unavailable: {type(e).__name__}: {e}"
            self._rec = None

    # ---------------- per-frame ----------------
    def process(self, vis: np.ndarray, clean: np.ndarray):
        """Recognise the hand on `clean`, act, and draw the UI onto `vis`."""
        if not self.active:
            return
        if self._rec is None:
            self._ensure_recognizer()
        gesture, hand = self._recognize(clean)
        self._last_gesture = gesture or "—"
        if self._cool > 0:
            self._cool -= 1
        self._drive(gesture, hand, vis.shape[1], vis.shape[0])
        self._draw(vis)

    def _recognize(self, clean):
        if self._rec is None:
            return None, None
        try:
            rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            self._ts += 33
            res = self._rec.recognize_for_video(image, self._ts)
        except Exception:
            return None, None
        if not res.hand_landmarks:
            self._cursor = None
            return None, None
        lm = res.hand_landmarks[0]
        name = ""
        if res.gestures and res.gestures[0]:
            name = res.gestures[0][0].category_name or ""
        return name, lm

    def _drive(self, gesture, hand, W, H):
        if hand is None:
            self._hold = 0
            self._fired_hold = False
            return
        # cursor = index fingertip (landmark 8); browse by its x position
        tip = hand[8]
        self._cursor = (int(tip.x * W), int(tip.y * H))
        if self.items and not self.busy:
            x = min(max((tip.x - 0.18) / 0.64, 0.0), 1.0)    # deadzones at the edges
            self.idx = int(round(x * (len(self.items) - 1)))

        # FIST → hold to try on (latched until released)
        if gesture == "Closed_Fist":
            self._hold = min(self._hold + 1, _HOLD_FRAMES)
            if self._hold >= _HOLD_FRAMES and not self._fired_hold and not self.busy:
                self._fired_hold = True
                self._select()
        else:
            self._hold = 0
            self._fired_hold = False

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
        self.busy = True
        self.banner = ("Fitting " if self.store == "eyewear" else "Styling ") + title + "…"
        threading.Thread(target=self._try_on, args=(dict(it),), daemon=True).start()

    def _try_on(self, it):
        from . import tools as T
        try:
            if self.store == "eyewear":
                T.try_eyewear(image=it.get("img", ""), label=it.get("title", ""))
            else:
                T.try_product(query=it.get("title", ""))
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
        self.banner = "Cleared — back to live"
        threading.Timer(1.4, self._clear_banner).start()

    # ---------------- drawing ----------------
    def _draw(self, vis):
        H, W = vis.shape[:2]
        self._draw_carousel(vis, W, H)
        self._draw_cursor(vis)
        self._draw_hud(vis, W, H)
        self._draw_hints(vis, W, H)
        if self.banner:
            self._draw_banner(vis, W, H)

    def _draw_carousel(self, vis, W, H):
        if not self.items:
            return
        cy = int(H * 0.40)
        cx = W // 2
        gap = int(W * 0.165)
        for off in range(-(_VISIBLE // 2), _VISIBLE // 2 + 1):
            j = self.idx + off
            if not (0 <= j < len(self.items)):
                continue
            it = self.items[j]
            center = (off == 0)
            x = cx + off * gap
            tw = int(W * (0.135 if center else 0.092))
            thumb = self._thumbs.get(it.get("img", ""))
            self._draw_tile(vis, x, cy, tw, thumb, it, center)
        # count + store, just under the centre tile
        it = self.items[self.idx]
        label = f"{it.get('title','')[:34]}"
        price = (f"${it.get('price','')}" if it.get("price") else "")
        self._center_text(vis, label, cx, cy + int(W * 0.105), 0.66, (255, 255, 255), 2)
        if price:
            self._center_text(vis, price, cx, cy + int(W * 0.105) + 26, 0.62,
                              (120, 240, 170), 2)

    def _draw_tile(self, vis, x, cy, tw, thumb, it, center):
        if thumb is not None:
            th = int(thumb.shape[0] * (tw * 2) / thumb.shape[1])
            th = min(th, int(tw * 2.4))
            patch = cv2.resize(thumb, (tw * 2, th))
            y0 = cy - th // 2
            x0 = x - tw
            y1, x1 = y0 + th, x0 + tw * 2
            if 0 <= y0 and y1 <= vis.shape[0] and 0 <= x0 and x1 <= vis.shape[1]:
                roi = vis[y0:y1, x0:x1]
                a = 1.0 if center else 0.62
                cv2.addWeighted(patch, a, roi, 1 - a, 0, roi)
        else:                      # placeholder card with the title
            h2 = int(tw * 1.2)
            self._panel(vis, x - tw, cy - h2, x + tw, cy + h2,
                        (40, 46, 60), 0.85 if center else 0.5)
            self._center_text(vis, it.get("title", "")[:16], x, cy, 0.42,
                              (210, 220, 235), 1)
        if center:                 # glowing selection frame
            h2 = int(tw * 1.25)
            cv2.rectangle(vis, (x - tw - 4, cy - h2 - 4), (x + tw + 4, cy + h2 + 4),
                          (80, 230, 180), 3)
            # hold-to-try progress ring
            if self._hold > 0:
                frac = self._hold / _HOLD_FRAMES
                cv2.ellipse(vis, (x, cy - h2 - 22), (16, 16), -90, 0, int(360 * frac),
                            (90, 240, 190), 4)

    def _draw_cursor(self, vis):
        if not self._cursor:
            return
        cv2.circle(vis, self._cursor, 16, (90, 240, 200), 2)
        cv2.circle(vis, self._cursor, 4, (255, 255, 255), -1)
        g = {"Closed_Fist": "GRAB", "Victory": "SWITCH", "Thumb_Down": "CLEAR",
             "Open_Palm": "BROWSE", "Pointing_Up": "BROWSE"}.get(self._last_gesture, "")
        if g:
            cv2.putText(vis, g, (self._cursor[0] + 20, self._cursor[1] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 240, 200), 2, cv2.LINE_AA)

    def _draw_hud(self, vis, W, H):
        store = "Ralba Optical · eyewear" if self.store == "eyewear" else "Mode Marco · apparel"
        txt = f"{store}   {self.idx + 1}/{len(self.items) or 0}"
        self._panel(vis, 14, 14, 14 + 11 * len(txt) + 20, 50, (16, 20, 30), 0.6)
        cv2.putText(vis, txt, (26, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (180, 230, 255), 2, cv2.LINE_AA)

    def _draw_hints(self, vis, W, H):
        hints = "MOVE HAND = browse      FIST = try on      V = switch store      THUMBS-DOWN = clear"
        y = H - 22
        self._panel(vis, 0, H - 44, W, H, (10, 12, 18), 0.55)
        self._center_text(vis, hints, W // 2, y, 0.6, (210, 225, 240), 2)

    def _draw_banner(self, vis, W, H):
        self._panel(vis, 0, int(H * 0.5) - 30, W, int(H * 0.5) + 30, (12, 16, 24), 0.6)
        col = (120, 240, 170) if not self.busy else (250, 210, 120)
        self._center_text(vis, self.banner, W // 2, int(H * 0.5) + 8, 0.95, col, 2)

    # ---- tiny draw helpers ----
    @staticmethod
    def _panel(vis, x0, y0, x1, y1, color, alpha):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(vis.shape[1], x1), min(vis.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return
        roi = vis[y0:y1, x0:x1]
        block = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(block, alpha, roi, 1 - alpha, 0, roi)

    @staticmethod
    def _center_text(vis, text, cx, cy, scale, color, thick):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(vis, text, (int(cx - tw / 2), int(cy + th / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
