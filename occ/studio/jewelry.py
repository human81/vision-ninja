"""Face/body jewelry try-on — presets + procedural assets. Pure cv2/np (no imports
back into face_filters, to avoid a cycle: face_filters imports THIS).

A piece type maps to one or more ANCHOR specs; `filter_jewelry` (in face_filters)
stickers the asset at each anchor, scaled to the face and following head tilt. Adding
a new kind = one entry here. Procedural gold assets render the try-on instantly, $0 and
offline; a real product image (e.g. an Amazon hero shot) overrides them via rembg.

Anchor spec fields:
  at    : Face landmark name (facemesh.IDX) the piece hangs from
  dx,dy : offset from that point, in EYE-DISTANCE units (face-local: +x = the face's
          right-ear direction, +y = down). Lets a hoop sit just below/outside a nostril.
  size  : asset width, in units of `unit`
  unit  : 'eye' (interocular dist) | 'facew' (face width) — necklaces scale to face width
  roll  : follow head roll (rings) or hang ~vertical (dangly earrings/necklaces)
  side  : 'l'/'r'/None — which side it lives on, so it hides when that side turns away
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:                                    # pragma: no cover
    cv2 = None

# Gold + silver in BGR (the studio works in BGR frames).
GOLD = (40, 175, 222)
GOLD_HI = (170, 230, 255)
SILVER = (190, 190, 195)
SILVER_HI = (240, 240, 245)


def _spec(at, dx, dy, size, unit="eye", roll=True, side=None):
    return {"at": at, "dx": dx, "dy": dy, "size": size, "unit": unit, "roll": roll, "side": side}


# Each preset = (default asset kind, [anchor specs]). Tunable; chosen to read well front-on.
PRESETS: dict[str, dict] = {
    "nose_ring": {"asset": "hoop", "label": "Nose ring",          # Tupac — left nostril hoop
                  "anchors": [_spec("nostril_l", 0.05, 0.10, 0.30, roll=True, side="l")]},
    "septum":    {"asset": "hoop", "label": "Septum ring",
                  "anchors": [_spec("nose_bottom", 0.0, 0.07, 0.28, roll=True)]},
    "earrings":  {"asset": "dangle", "label": "Earrings",         # a PAIR, dangle from the lobes
                  "anchors": [_spec("ear_l", -0.01, 0.60, 0.40, roll=False, side="l"),
                              _spec("ear_r", 0.01, 0.60, 0.40, roll=False, side="r")]},
    "studs":     {"asset": "stud", "label": "Ear studs",          # sit ON the lobe
                  "anchors": [_spec("ear_l", -0.01, 0.48, 0.16, roll=False, side="l"),
                              _spec("ear_r", 0.01, 0.48, 0.16, roll=False, side="r")]},
    "lip_ring":  {"asset": "hoop", "label": "Lip ring",
                  "anchors": [_spec("lowerlip_bot", 0.0, 0.08, 0.22, roll=True)]},
    "necklace":  {"asset": "necklace", "label": "Necklace",       # drapes on the chest below the chin
                  "anchors": [_spec("chin", 0.0, 1.15, 1.60, unit="facew", roll=False)]},
}

KINDS = list(PRESETS)


def resolve_kind(kind: str) -> str:
    k = (kind or "").strip().lower().replace(" ", "_").replace("-", "_")
    if k in PRESETS:
        return k
    aliases = {"noserring": "nose_ring", "nosering": "nose_ring", "nose": "nose_ring",
               "earring": "earrings", "ear": "earrings", "stud": "studs",
               "chain": "necklace", "pendant": "necklace", "lip": "lip_ring"}
    return aliases.get(k, "nose_ring")


# ----------------------------- procedural assets -----------------------------
def _canvas(d):
    return np.zeros((d, d, 4), np.uint8)


def _shine(img, c, r, hi):
    """A small specular highlight arc/dot so the metal reads as metal, not a flat ring."""
    cv2.circle(img, (int(c[0] - r * 0.35), int(c[1] - r * 0.4)), max(1, int(r * 0.18)),
               (*hi, 255), -1, cv2.LINE_AA)


def gold_hoop(d=200, gold=GOLD, hi=GOLD_HI) -> np.ndarray:
    """A see-through metal hoop (nose ring / hoop earring / lip ring)."""
    img = _canvas(d)
    c = (d // 2, int(d * 0.46)); R = int(d * 0.36); t = max(3, int(d * 0.11))
    cv2.circle(img, c, R, (*gold, 255), t, cv2.LINE_AA)
    cv2.ellipse(img, c, (R, R), 0, 200, 320, (*hi, 255), max(1, t // 2), cv2.LINE_AA)  # top-left shine
    _shine(img, (c[0] - R, c[1]), t, hi)
    return img


def gem_stud(d=120, gold=GOLD, hi=SILVER_HI) -> np.ndarray:
    """A small stud / ball earring."""
    img = _canvas(d)
    c = (d // 2, d // 2); r = int(d * 0.30)
    cv2.circle(img, c, r, (*gold, 255), -1, cv2.LINE_AA)
    cv2.circle(img, c, r, (*hi, 200), max(1, int(d * 0.02)), cv2.LINE_AA)
    _shine(img, c, r, hi)
    return img


def dangle_earring(d=220, gold=GOLD, hi=GOLD_HI) -> np.ndarray:
    """A post + drop hoop that hangs from the lobe."""
    img = _canvas(d)
    cx = d // 2
    cv2.circle(img, (cx, int(d * 0.16)), max(2, int(d * 0.045)), (*gold, 255), -1, cv2.LINE_AA)  # post
    cv2.line(img, (cx, int(d * 0.18)), (cx, int(d * 0.42)), (*gold, 255), max(2, int(d * 0.03)), cv2.LINE_AA)
    c = (cx, int(d * 0.66)); R = int(d * 0.27); t = max(3, int(d * 0.085))
    cv2.circle(img, c, R, (*gold, 255), t, cv2.LINE_AA)                                          # drop hoop
    cv2.ellipse(img, c, (R, R), 0, 200, 320, (*hi, 255), max(1, t // 2), cv2.LINE_AA)
    return img


def necklace(d=320, gold=GOLD, hi=GOLD_HI) -> np.ndarray:
    """A chain that drapes in a catenary with a small pendant — wide canvas."""
    w, h = d, int(d * 0.55)
    img = np.zeros((h, w, 4), np.uint8)
    pts = []
    for i in range(0, w, max(4, w // 60)):
        x = i / w                                    # 0..1
        y = 0.12 + 0.62 * (4 * (x - 0.5) ** 2)       # parabola (catenary-ish), lowest at center
        pts.append((int(i), int(y * h)))
    for i, p in enumerate(pts):                      # chain links
        cv2.circle(img, p, max(2, int(d * 0.012)), (*gold, 255), -1, cv2.LINE_AA)
        if i % 2 == 0:
            cv2.circle(img, p, max(1, int(d * 0.006)), (*hi, 230), -1, cv2.LINE_AA)
    cen = pts[len(pts) // 2]                          # pendant at the lowest point
    cv2.circle(img, (cen[0], cen[1] + int(d * 0.05)), int(d * 0.05), (*gold, 255), -1, cv2.LINE_AA)
    _shine(img, (cen[0], cen[1] + int(d * 0.05)), int(d * 0.05), hi)
    return img


_ASSET_FNS = {"hoop": gold_hoop, "stud": gem_stud, "dangle": dangle_earring, "necklace": necklace}


def default_asset(kind: str) -> np.ndarray:
    """The procedural gold asset for a piece type (used when no product image is given)."""
    if cv2 is None:
        return np.zeros((8, 8, 4), np.uint8)
    return _ASSET_FNS.get(PRESETS.get(resolve_kind(kind), {}).get("asset", "hoop"), gold_hoop)()
