"""Interactive zone/line editor — mouse + keyboard, with on-screen guidance.

Runs on a grabbed frame in an OpenCV window. Vertices are stored normalized so
annotations are resolution-independent and reusable across cameras.

Controls (always shown on screen; press H to toggle the full help panel):
  z            start a new ZONE (polygon)        l   start a new LINE (2 points)
  left-click   add a vertex / (idle) select      Enter/C  commit current shape
  Backspace/U  undo last vertex                   N   rename selected item
  F            flip selected LINE direction       Tab cycle selection
  D            delete selected                     S   save        R  reset all
  H            toggle help                         Q/Esc  quit
A green arrow on each line shows its POSITIVE (right-hand-rule) crossing direction.
"""

from __future__ import annotations

import cv2
import numpy as np

from .annotations import Annotation, AnnotationSet, ZONE, LINE
from .config import Config
from .sources import open_source

IDLE, DRAW_ZONE, DRAW_LINE, NAMING = "idle", "zone", "line", "naming"


def _grab_frame(cfg: Config) -> np.ndarray:
    src = open_source(cfg)
    try:
        for frame in src.frames():
            return frame.copy()
    finally:
        src.release()
    raise RuntimeError("could not grab a frame from source")


class Editor:
    def __init__(self, cfg: Config, ann_path: str):
        self.cfg = cfg
        self.path = ann_path
        self.frame = _grab_frame(cfg)
        self.h, self.w = self.frame.shape[:2]
        self.set = AnnotationSet.load(ann_path)
        self.set.source = str(cfg.get("source.uri", ""))
        self.mode = IDLE
        self.draft: list[tuple[float, float]] = []     # normalized in-progress verts
        self.cursor = (0, 0)
        self.selected = len(self.set.annotations) - 1
        self.show_help = True
        self.name_buf = ""
        self.status = "ready"
        self.win = "editor"

    # ---------- mouse ----------
    def _on_mouse(self, event, x, y, flags, _):
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode in (DRAW_ZONE, DRAW_LINE):
                self.draft.append((x / self.w, y / self.h))
                if self.mode == DRAW_LINE and len(self.draft) == 2:
                    self.status = "press Enter to commit line (F flips direction)"
            elif self.mode == IDLE:
                self._select_nearest(x, y)

    def _select_nearest(self, x, y):
        best, bestd = -1, 1e18
        for i, a in enumerate(self.set.annotations):
            for (nx, ny) in a.vertices:
                d = (nx * self.w - x) ** 2 + (ny * self.h - y) ** 2
                if d < bestd:
                    bestd, best = d, i
        if best >= 0 and bestd < (40 ** 2):
            self.selected = best
            self.status = f"selected {self.set.annotations[best].id}"

    # ---------- commit / edit ----------
    def _commit(self):
        if self.mode == DRAW_ZONE and len(self.draft) >= 3:
            aid = self.set.next_id(ZONE)
            self.set.annotations.append(Annotation(aid, ZONE, list(self.draft)))
            self.selected = len(self.set.annotations) - 1
            self.status = f"added {aid}"
        elif self.mode == DRAW_LINE and len(self.draft) == 2:
            aid = self.set.next_id(LINE)
            self.set.annotations.append(Annotation(aid, LINE, list(self.draft)))
            self.selected = len(self.set.annotations) - 1
            self.status = f"added {aid}"
        else:
            self.status = "need >=3 points for a zone, 2 for a line"
            return
        self.draft = []
        self.mode = IDLE

    def _flip_selected(self):
        a = self._sel()
        if a and a.type == LINE:
            a.vertices = [a.vertices[1], a.vertices[0]]
            self.status = f"flipped {a.id} direction"
        elif len(self.draft) == 2:
            self.draft = [self.draft[1], self.draft[0]]
            self.status = "flipped draft line"

    def _sel(self) -> Annotation | None:
        if 0 <= self.selected < len(self.set.annotations):
            return self.set.annotations[self.selected]
        return None

    # ---------- key handling ----------
    def _key(self, k: int) -> bool:
        """Return False to quit."""
        if self.mode == NAMING:
            return self._key_naming(k)
        ch = chr(k & 0xFF).lower() if 0 <= (k & 0xFF) < 128 else ""
        if k in (ord("q"), 27):
            return False
        elif ch == "z":
            self.mode, self.draft = DRAW_ZONE, []
            self.status = "ZONE: click vertices, Enter to close"
        elif ch == "l":
            self.mode, self.draft = DRAW_LINE, []
            self.status = "LINE: click 2 points (1st = start of arrow)"
        elif k in (13, 10) or ch == "c":          # Enter / C
            self._commit()
        elif k in (8, 127) or ch == "u":          # Backspace / U
            if self.draft:
                self.draft.pop()
            elif self._sel():
                self.status = f"({self._sel().id} selected — press D to delete)"
        elif ch == "n" and self._sel():
            self.mode, self.name_buf = NAMING, self._sel().display_name
            self.status = "type a name, Enter to confirm"
        elif ch == "f":
            self._flip_selected()
        elif k == 9:                               # Tab
            if self.set.annotations:
                self.selected = (self.selected + 1) % len(self.set.annotations)
                self.status = f"selected {self._sel().id}"
        elif ch == "d" and self._sel():
            removed = self.set.annotations.pop(self.selected)
            self.selected = min(self.selected, len(self.set.annotations) - 1)
            self.status = f"deleted {removed.id}"
        elif ch == "s":
            self.set.save(self.path)
            self.status = f"saved → {self.path}"
        elif ch == "r":
            self.set.annotations.clear(); self.draft = []; self.selected = -1
            self.status = "reset"
        elif ch == "h":
            self.show_help = not self.show_help
        return True

    def _key_naming(self, k: int) -> bool:
        if k in (13, 10):                          # Enter confirms
            a = self._sel()
            if a:
                a.display_name = self.name_buf
            self.mode, self.status = IDLE, f"named → {self.name_buf}"
        elif k == 27:                              # Esc cancels
            self.mode, self.status = IDLE, "rename cancelled"
        elif k in (8, 127):
            self.name_buf = self.name_buf[:-1]
        elif 32 <= (k & 0xFF) < 127:
            self.name_buf += chr(k & 0xFF)
        return True

    # ---------- drawing ----------
    def _draw(self) -> np.ndarray:
        img = self.frame.copy()
        # committed annotations
        for i, a in enumerate(self.set.annotations):
            sel = (i == self.selected)
            self._draw_annotation(img, a, selected=sel)
        # draft preview
        if self.draft:
            self._draw_draft(img)
        self._draw_hud(img)
        return img

    def _draw_annotation(self, img, a: Annotation, selected: bool):
        col = (0, 255, 255) if selected else (
            (0, 180, 255) if a.type == ZONE else (0, 255, 0))
        pts = [(int(x * self.w), int(y * self.h)) for x, y in a.vertices]
        if a.type == ZONE:
            arr = np.array(pts, np.int32)
            ov = img.copy(); cv2.fillPoly(ov, [arr], col)
            cv2.addWeighted(ov, 0.20, img, 0.80, 0, img)
            cv2.polylines(img, [arr], True, col, 2 + 2 * selected, cv2.LINE_AA)
        else:
            cv2.line(img, pts[0], pts[1], col, 2 + 2 * selected, cv2.LINE_AA)
            self._draw_arrow(img, pts[0], pts[1], col)
        for p in pts:
            cv2.circle(img, p, 4, col, -1)
        cv2.putText(img, a.display_name or a.id, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_arrow(img, a, b, col):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        nx, ny = dy / L, -dx / L
        cv2.arrowedLine(img, (int(mx), int(my)),
                        (int(mx + nx * 30), int(my + ny * 30)), col, 2,
                        cv2.LINE_AA, tipLength=0.4)

    def _draw_draft(self, img):
        pts = [(int(x * self.w), int(y * self.h)) for x, y in self.draft]
        col = (0, 180, 255) if self.mode == DRAW_ZONE else (0, 255, 0)
        for p in pts:
            cv2.circle(img, p, 4, col, -1)
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)],
                          self.mode == DRAW_ZONE, col, 1, cv2.LINE_AA)
        if pts:                                   # rubber-band to cursor
            cv2.line(img, pts[-1], self.cursor, col, 1, cv2.LINE_AA)
        if self.mode == DRAW_LINE and len(pts) == 2:
            self._draw_arrow(img, pts[0], pts[1], col)

    def _draw_hud(self, img):
        bar = f"[{self.mode.upper()}]  items:{len(self.set.annotations)}  " \
              f"sel:{self._sel().id if self._sel() else '-'}  |  {self.status}"
        if self.mode == NAMING:
            bar = f"NAME> {self.name_buf}_"
        cv2.rectangle(img, (0, 0), (self.w, 28), (0, 0, 0), -1)
        cv2.putText(img, bar, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)
        if self.show_help:
            help_lines = [
                "z new zone   l new line   click add vertex   Enter commit",
                "u undo   n name   f flip line   Tab select   d delete",
                "s save   r reset   h hide help   q quit",
            ]
            y = self.h - 12 * len(help_lines) - 14
            cv2.rectangle(img, (0, y - 6), (560, self.h), (0, 0, 0), -1)
            for ln in help_lines:
                cv2.putText(img, ln, (8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (220, 220, 220), 1, cv2.LINE_AA)
                y += 14

    # ---------- main loop ----------
    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)
        while True:
            cv2.imshow(self.win, self._draw())
            k = cv2.waitKeyEx(20)
            if k == -1:
                continue
            if not self._key(k):
                break
        cv2.destroyAllWindows()
        return self.set
