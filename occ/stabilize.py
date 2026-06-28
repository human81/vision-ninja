"""Feature-anchored annotation stabilization.

Zones and lines are drawn in NORMALIZED (0..1) frame coordinates, so when the camera
pans / tilts / zooms / rolls they end up over the wrong physical spot. This module pins
each annotation to the SCENE instead of the frame:

  1. anchor(frame)   — snapshot the scene's trackable keypoints + descriptors (the
                       "reference"). The annotations' normalized vertices are interpreted
                       in THIS reference frame.
  2. register(frame) — each live frame, detect keypoints, match them to the reference,
                       and estimate the homography H that maps reference -> current view
                       (RANSAC, ratio-test, sanity-checked, temporally smoothed).
  3. warp(annset)    — push the reference vertices through H, so the zone/line is redrawn
                       at the right place AND the right size in the moved view.

Keypoint detector (why these):
  • ORB    (default) — FAST, license-free, binary descriptors (Hamming match), rotation +
                       multi-scale (pyramid). Best for continuous real-time tracking on CPU.
  • SIFT             — scale + rotation robust, now patent-free; slower. Best for LARGE
                       viewpoint changes / re-localization after a big move.
  • AKAZE            — nonlinear scale space, license-free; a solid middle ground.
  (SURF is patented / opencv-nonfree, so it's deliberately not offered.)

Assumes a roughly planar or far-field scene (the usual surveillance/overhead case); a
homography can't model strong parallax, and that limit is reported honestly via the
confidence in status().
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from .annotations import Annotation, AnnotationSet

_BINARY = {"orb", "akaze", "brisk"}            # binary descriptors -> Hamming matching


def warp_annset(annset: AnnotationSet, H, rw: int, rh: int, cw: int, ch: int) -> AnnotationSet:
    """Push every annotation's NORMALIZED vertices (defined in a reference frame rw×rh)
    through homography H into a current frame cw×ch, returning a new AnnotationSet with the
    SAME ids/types (so a GeometryEngine's per-track state survives when it's fed back in)."""
    out = []
    for a in annset.annotations:
        if not a.vertices:
            out.append(a)
            continue
        ref_px = np.float32([[x * rw, y * rh] for x, y in a.vertices]).reshape(-1, 1, 2)
        cur_px = cv2.perspectiveTransform(ref_px, np.asarray(H, np.float64)).reshape(-1, 2)
        verts = [(float(px / cw), float(py / ch)) for px, py in cur_px]
        out.append(Annotation(id=a.id, type=a.type, vertices=verts,
                              display_name=a.display_name))
    return AnnotationSet(annotations=out, source=annset.source)


def _build_detector(name: str):
    name = (name or "orb").lower()
    try:
        if name == "sift":
            return cv2.SIFT_create(nfeatures=1500), "sift"
        if name == "akaze":
            return cv2.AKAZE_create(), "akaze"
        if name == "brisk":
            return cv2.BRISK_create(), "brisk"
    except Exception:
        pass
    return cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8), "orb"


def sim_transform(mode: str, w: int, h: int, degrees: float = 0.0):
    """The exact 3×3 homography for a SIMULATED camera move on a w×h frame (same canvas).
    Used to flip/rotate the stream in the UI and redraw zones/lines exactly — a ground-truth
    demo of re-localization. Returns None for 'none'. Maps original px -> transformed px."""
    m = (mode or "none").lower()
    if m in ("none", "", "off"):
        return None
    if m in ("flipv", "vertical", "upsidedown_mirror"):
        return np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], np.float64)
    if m in ("fliph", "horizontal", "mirror"):
        return np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], np.float64)
    if m in ("rotate180", "180", "upsidedown"):
        return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float64)
    # arbitrary rotation about the frame centre
    R = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(degrees), 1.0)
    return np.vstack([R, [0, 0, 1]]).astype(np.float64)


class SceneStabilizer:
    """Estimate + apply the reference->current homography for annotation re-localization."""

    def __init__(self, detector: str = "orb", min_inliers: int = 18,
                 ratio: float = 0.75, smooth: float = 0.5):
        self.min_inliers = int(min_inliers)
        self.ratio = float(ratio)
        self.smooth = float(smooth)             # EMA on H -> a steady, non-jittery overlay
        self._lock = threading.Lock()
        self.set_detector(detector)
        self.ref: dict | None = None            # {kp, desc, size:(w,h)}
        self.H: np.ndarray | None = None        # last good homography ref->cur
        self.conf: float = 0.0
        self.inliers: int = 0
        self.lost: bool = True
        self._have_prev: bool = False           # a real prior registration to smooth against

    # ---- configuration ----
    def set_detector(self, detector: str):
        det, name = _build_detector(detector)
        norm = cv2.NORM_HAMMING if name in _BINARY else cv2.NORM_L2
        with self._lock:
            self._det = det
            self.detector = name
            self._matcher = cv2.BFMatcher(norm)

    def _features(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self._det.detectAndCompute(gray, None)
        return kp, desc

    # ---- anchoring ----
    def anchor(self, frame_bgr) -> bool:
        """Snapshot the current view as the reference the annotations are pinned to."""
        h, w = frame_bgr.shape[:2]
        kp, desc = self._features(frame_bgr)
        if desc is None or len(kp) < self.min_inliers:
            return False
        with self._lock:
            self.ref = {"kp": kp, "desc": desc, "size": (w, h)}
            self.H = np.eye(3, dtype=np.float64)
            self.conf, self.inliers, self.lost = 1.0, len(kp), False
            self._have_prev = False             # don't smooth the first fix against identity
        return True

    def clear(self):
        with self._lock:
            self.ref = None
            self.H = None
            self.conf, self.inliers, self.lost = 0.0, 0, True

    def anchored(self) -> bool:
        return self.ref is not None

    # ---- per-frame registration ----
    @staticmethod
    def _sane(H) -> bool:
        """Reject degenerate homographies (flips, blow-ups, wild perspective)."""
        if H is None or not np.all(np.isfinite(H)):
            return False
        a = H[:2, :2]
        det = float(np.linalg.det(a))
        if det <= 1e-3 or det >= 1e3:            # mirror / extreme zoom -> not a real camera move
            return False
        scale = float(np.sqrt(abs(det)))
        if scale < 0.2 or scale > 5.0:
            return False
        if abs(H[2, 0]) > 1e-2 or abs(H[2, 1]) > 1e-2:   # implausibly strong perspective
            return False
        return True

    def register(self, frame_bgr) -> bool:
        """Update H from reference->current view. Returns True on a confident fix; on a
        weak/failed match the LAST good H is kept (annotations hold their place) and the
        confidence decays so callers can detect tracking loss."""
        with self._lock:
            ref = self.ref
        if ref is None:
            return False
        kp, desc = self._features(frame_bgr)
        if desc is None or len(kp) < 4 or ref["desc"] is None:
            return self._degrade()
        try:
            knn = self._matcher.knnMatch(ref["desc"], desc, k=2)
        except cv2.error:
            return self._degrade()
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio * n.distance]
        if len(good) < self.min_inliers:
            return self._degrade()
        src = np.float32([ref["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=2000, confidence=0.995)
        inl = int(mask.sum()) if mask is not None else 0
        if H is None or inl < self.min_inliers or not self._sane(H):
            return self._degrade()
        with self._lock:
            # EMA-smooth H so the overlay is steady, not jittery (renormalize H[2,2]=1).
            # Only smooth against a REAL prior fix — never against the anchor's identity.
            if self._have_prev and not self.lost and self.H is not None:
                H = self.smooth * H + (1.0 - self.smooth) * self.H
                if abs(H[2, 2]) > 1e-9:
                    H = H / H[2, 2]
            self.H = H
            self.inliers = inl
            self.conf = min(1.0, inl / max(len(good), 1))
            self.lost = False
            self._have_prev = True
        return True

    def _degrade(self) -> bool:
        with self._lock:
            self.conf *= 0.7
            if self.conf < 0.2:
                self.lost = True
        return False

    # ---- apply ----
    def warp(self, annset: AnnotationSet, cur_w: int, cur_h: int) -> AnnotationSet:
        """Return a copy of `annset` with vertices moved from the reference view into the
        current view via H. Same ids/types, so the GeometryEngine's per-track state is kept
        when this is fed back in. If not anchored, returns the input unchanged."""
        with self._lock:
            ref, H = self.ref, self.H
        if ref is None or H is None:
            return annset
        rw, rh = ref["size"]
        return warp_annset(annset, H, rw, rh, cur_w, cur_h)

    def status(self) -> dict:
        with self._lock:
            return {"anchored": self.ref is not None,
                    "detector": self.detector,
                    "tracking": (not self.lost) and self.ref is not None,
                    "confidence": round(self.conf, 3),
                    "inliers": self.inliers,
                    "ref_size": list(self.ref["size"]) if self.ref else None}
