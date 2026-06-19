"""Dense facial landmarks for AR filters / virtual try-on (MediaPipe FaceLandmarker).

478 landmarks + per-face blendshapes (smile, blink, jaw-open, brow…) + a metric
4x4 facial-transformation matrix (head pose). Runs on CPU in real time. We only
run it when a *face* overlay asks for `ctx.faces`, so non-face streams pay nothing.

The Tasks API model (`face_landmarker.task`) auto-downloads once to
`out/studio/models/`. A process-wide singleton keeps one LIVE/VIDEO landmarker
(its detect_for_video needs monotonically increasing timestamps).
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass, field

import numpy as np

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
_MODEL_DIR = "out/studio/models"
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")

# ---- canonical MediaPipe-468/478 landmark indices we anchor filters to --------
IDX = {
    "nose_tip": 1, "nose_bottom": 2, "nose_bridge": 168, "nose_mid": 6,
    "nostril_l": 64, "nostril_r": 294,
    "eye_l_outer": 33, "eye_l_inner": 133, "eye_l_top": 159, "eye_l_bot": 145,
    "eye_r_outer": 263, "eye_r_inner": 362, "eye_r_top": 386, "eye_r_bot": 374,
    "iris_l": 468, "iris_r": 473,                 # centers (refine_landmarks)
    "brow_l_outer": 70, "brow_l_inner": 107, "brow_r_outer": 300, "brow_r_inner": 336,
    "temple_l": 234, "temple_r": 454,             # face-oval sides at eye level
    "ear_l": 127, "ear_r": 356,
    "cheek_l": 50, "cheek_r": 280,
    "mouth_l": 61, "mouth_r": 291, "lip_top": 13, "lip_bot": 14,
    "upperlip_top": 0, "lowerlip_bot": 17,
    "chin": 152, "forehead": 10, "forehead_l": 67, "forehead_r": 297,
    "glab": 9,                                     # glabella (between brows)
}

# face-oval contour (clockwise) for masks / silhouettes
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@dataclass
class Face:
    """One detected face, in PIXEL coordinates of the frame it came from."""
    lm: np.ndarray                       # (N,2) float32 pixel landmarks
    lmz: np.ndarray                      # (N,) relative depth (smaller = closer)
    blend: dict = field(default_factory=dict)   # blendshape name -> 0..1
    matrix: np.ndarray | None = None     # 4x4 facial transform (head pose), or None
    w: int = 0
    h: int = 0

    # ---- convenience anchors (pixel xy) ----
    def p(self, name: str) -> np.ndarray:
        return self.lm[IDX[name]]

    @property
    def eye_l(self) -> np.ndarray: return self.lm[IDX["iris_l"]] if len(self.lm) > 468 else (self.p("eye_l_outer") + self.p("eye_l_inner")) / 2

    @property
    def eye_r(self) -> np.ndarray: return self.lm[IDX["iris_r"]] if len(self.lm) > 473 else (self.p("eye_r_outer") + self.p("eye_r_inner")) / 2

    @property
    def eyes_center(self) -> np.ndarray: return (self.eye_l + self.eye_r) / 2

    @property
    def eye_dist(self) -> float: return float(np.linalg.norm(self.eye_r - self.eye_l))

    @property
    def roll(self) -> float:
        """In-plane head tilt (degrees), +ve = right ear down."""
        d = self.p("temple_r") - self.p("temple_l")
        return float(np.degrees(np.arctan2(d[1], d[0])))

    def _pose_rad(self):
        """(yaw, pitch, roll) in RADIANS from MediaPipe's 4×4 facial transform MATRIX —
        true 3D head pose. Returns None if the matrix is absent. Signs are calibrated to
        match the landmark conventions (verified against the geometric roll/yaw):
        +yaw = turned so the RIGHT side faces the camera; +pitch = chin up; +roll = right
        ear down."""
        M = self.matrix
        if M is None:
            return None
        R = np.array(M, dtype=float)[:3, :3]
        for i in range(3):                               # strip scale → pure rotation
            n = np.linalg.norm(R[:, i]) or 1.0
            R[:, i] /= n
        sy = float(np.hypot(R[0, 0], R[1, 0]))
        if sy > 1e-6:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1]); yaw = np.arctan2(-R[2, 0], sy); roll = 0.0
        # calibrate to the landmark conventions (image y is DOWN)
        return (float(yaw), float(-pitch), float(-roll))

    def _yaw_geom(self) -> float:
        ecx = float(self.eyes_center[0])
        lw = abs(ecx - float(self.p("ear_l")[0]))
        rw = abs(float(self.p("ear_r")[0]) - ecx)
        s = lw + rw
        return float((rw - lw) / s) if s > 1e-6 else 0.0

    @property
    def yaw(self) -> float:
        """Head turn, signed ≈[-1, 1]: 0 = forward, +ve = LEFT side receding (left temple
        arm should hide). The 3D MATRIX gives the accurate magnitude; the landmark
        geometry gives the TRUSTED sign convention (so the temple-arm occlusion can never
        flip) — they agree on the front face, verified."""
        g = self._yaw_geom()
        pr = self._pose_rad()
        if pr is None:
            return g
        m = float(np.clip(np.sin(pr[0]), -1.0, 1.0))
        if abs(g) < 0.03:                                # near-forward → trust the matrix
            return m
        return max(abs(m), abs(g)) * (1.0 if g >= 0 else -1.0)

    @property
    def pitch(self) -> float:
        """Head nod, signed ≈[-1, 1] (sin of true pitch): +ve = chin up / looking up.
        From the 3D matrix; 0 if unavailable."""
        pr = self._pose_rad()
        return float(np.clip(np.sin(pr[1]), -1.0, 1.0)) if pr else 0.0

    @property
    def face_w(self) -> float:
        return float(np.linalg.norm(self.p("temple_r") - self.p("temple_l")))

    @property
    def face_h(self) -> float:
        return float(np.linalg.norm(self.p("chin") - self.p("forehead")))

    def oval(self) -> np.ndarray:
        return self.lm[FACE_OVAL].astype(np.int32)

    def bbox(self):
        x0, y0 = self.lm.min(0); x1, y1 = self.lm.max(0)
        return int(x0), int(y0), int(x1), int(y1)

    # ---- expressions (from blendshapes) ----
    def blendv(self, *names) -> float:
        return max((self.blend.get(n, 0.0) for n in names), default=0.0)

    @property
    def smile(self) -> float:
        return self.blendv("mouthSmileLeft", "mouthSmileRight")

    @property
    def mouth_open(self) -> float:
        return self.blendv("jawOpen")

    @property
    def blink_l(self) -> float:
        return self.blendv("eyeBlinkLeft")

    @property
    def blink_r(self) -> float:
        return self.blendv("eyeBlinkRight")

    @property
    def brow_up(self) -> float:
        return self.blendv("browInnerUp", "browOuterUpLeft", "browOuterUpRight")


class FaceMeshEngine:
    """Process-wide singleton landmarker (VIDEO mode, monotonic timestamps)."""

    _instance: "FaceMeshEngine | None" = None
    _instlock = threading.Lock()

    @classmethod
    def get(cls) -> "FaceMeshEngine | None":
        with cls._instlock:
            if cls._instance is None:
                try:
                    cls._instance = cls()
                except Exception as e:        # mediapipe missing / model fetch failed
                    cls._instance = _Disabled(str(e))  # type: ignore
            inst = cls._instance
        return None if isinstance(inst, _Disabled) else inst

    def __init__(self, num_faces: int = 2):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        _ensure_model()
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._lock = threading.Lock()
        self._ts = 0

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        """Return faces (pixel coords) for a BGR frame. Thread-safe; serialized."""
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            self._ts += 33                       # ms; must be monotonic increasing
            res = self._lm.detect_for_video(image, self._ts)
        faces: list[Face] = []
        if not res.face_landmarks:
            return faces
        mats = getattr(res, "facial_transformation_matrixes", None) or []
        blends = getattr(res, "face_blendshapes", None) or []
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
            z = np.array([p.z for p in lms], np.float32)
            bd = {}
            if i < len(blends):
                bd = {c.category_name: float(c.score) for c in blends[i]}
            mat = np.array(mats[i]).reshape(4, 4) if i < len(mats) else None
            faces.append(Face(lm=pts, lmz=z, blend=bd, matrix=mat, w=w, h=h))
        # nearest (largest) face first — filters usually target the main subject
        faces.sort(key=lambda f: f.face_w, reverse=True)
        return faces


class _Disabled:
    def __init__(self, reason: str): self.reason = reason


def _ensure_model():
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 100000:
        return
    os.makedirs(_MODEL_DIR, exist_ok=True)
    tmp = _MODEL_PATH + ".tmp"
    urllib.request.urlretrieve(_MODEL_URL, tmp)
    os.replace(tmp, _MODEL_PATH)


def detect_faces(frame_bgr: np.ndarray) -> list[Face]:
    eng = FaceMeshEngine.get()
    return eng.detect(frame_bgr) if eng is not None else []
