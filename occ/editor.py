"""Interactive zone/line editor — a rich, fluid OpenCV drawing surface.

Runs on a grabbed frame in an OpenCV window. Geometry is stored normalized
(0..1) so annotations are resolution-independent and reusable across cameras.
Lines are **polylines** (2+ points); their positive crossing direction follows
the right-hand rule of each directed segment and is shaded + arrowed on screen.

Controls (a contextual hint shows the relevant ones; press H for the full panel):
  Z            new ZONE (polygon)          L   new LINE (polyline, 2+ pts)
  left-click   add a point / select        drag a vertex to move it
  Enter / right-click / double-click  commit the current shape
  click first vertex   close the zone      Bksp / U   undo last point
  N  rename selected    F  flip line dir    Tab  cycle selection
  D  delete selected    S  save             R  reset all
  H  toggle help        Esc  cancel draft / quit        Q  quit
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .annotations import Annotation, AnnotationSet, ZONE, LINE
from .config import Config
from .sources import open_source

IDLE, DRAW_ZONE, DRAW_LINE, NAMING = "idle", "zone", "line", "naming"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---- palette (BGR) ----
C_ZONE   = (60, 180, 255)     # amber
C_LINE   = (90, 255, 90)      # green
C_SEL    = (255, 230, 60)     # cyan
C_POS    = (90, 255, 90)      # positive side / direction
C_NEG    = (70, 70, 255)      # negative side
C_CURSOR = (240, 240, 240)
C_SNAP   = (255, 90, 255)     # magenta snap highlight
C_GUIDE  = (120, 120, 120)
SNAP_PX  = 14


def _dist2(ax, ay, bx, by):
    return (ax - bx) ** 2 + (ay - by) ** 2


def _seg_dist2(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return _dist2(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return _dist2(px, py, ax + t * dx, ay + t * dy)


def _unit_normal(a, b):
    """Right-hand-rule unit normal of segment a->b (points to the positive side)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1.0
    return dy / L, -dx / L


def _dashed(img, p1, p2, color, thick=1, dash=14, gap=9, offset=0):
    x1, y1 = p1; x2, y2 = p2
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 1:
        return
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    s = -(offset % (dash + gap))
    while s < L:
        a = max(s, 0.0); b = min(s + dash, L)
        if b > a:
            cv2.line(img, (int(x1 + ux * a), int(y1 + uy * a)),
                     (int(x1 + ux * b), int(y1 + uy * b)), color, thick, cv2.LINE_AA)
        s += dash + gap


def _tag(img, text, org, color=(240, 240, 240), scale=0.5, thick=1, pad=5, alpha=0.82):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    # keep the tag on-screen
    H, W = img.shape[:2]
    x = min(max(x, pad), W - tw - pad - 1)
    y = min(max(y, th + pad + 1), H - bl - pad - 1)
    ov = img.copy()
    cv2.rectangle(ov, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), (18, 18, 18), -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return (x, y)


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
        self.s = max(0.8, self.h / 1080.0)         # resolution scale
        self.t = max(2, round(2 * self.s))         # stroke thickness
        self.set = AnnotationSet.load(ann_path)
        self.set.source = str(cfg.get("source.uri", ""))
        self.mode = IDLE
        self.draft: list[tuple[float, float]] = []
        self.cursor = (self.w // 2, self.h // 2)
        self.snap = None                            # (px, py, kind)
        self.drag = None                            # (ann_index, vert_index)
        self.hover = None                           # ann_index under cursor (idle)
        self.selected = len(self.set.annotations) - 1
        self.show_help = True
        self.show_loupe = True
        self.name_buf = ""
        self.status = "ready"
        self.tick = 0
        self.win = "editor"

    # ---------- helpers ----------
    def _px(self, verts) -> np.ndarray:
        return np.array([[int(x * self.w), int(y * self.h)] for x, y in verts], np.int32)

    def _col(self, a: Annotation, sel: bool):
        return C_SEL if sel else (C_ZONE if a.type == ZONE else C_LINE)

    def _sel(self) -> Annotation | None:
        if 0 <= self.selected < len(self.set.annotations):
            return self.set.annotations[self.selected]
        return None

    def _hit_vertex(self, x, y):
        best, bd = None, (SNAP_PX * self.s) ** 2
        for ai, a in enumerate(self.set.annotations):
            for vi, (nx, ny) in enumerate(a.vertices):
                d = _dist2(nx * self.w, ny * self.h, x, y)
                if d < bd:
                    bd, best = d, (ai, vi)
        return best

    def _snap_target(self, x, y):
        # closing a zone: snap to the first draft vertex
        if self.mode == DRAW_ZONE and len(self.draft) >= 3:
            fx, fy = self.draft[0][0] * self.w, self.draft[0][1] * self.h
            if _dist2(fx, fy, x, y) < (SNAP_PX * self.s) ** 2:
                return (fx, fy, "close")
        # alignment: snap to any committed vertex
        hit = self._hit_vertex(x, y)
        if hit and self.mode in (DRAW_ZONE, DRAW_LINE):
            nx, ny = self.set.annotations[hit[0]].vertices[hit[1]]
            return (nx * self.w, ny * self.h, "vertex")
        return None

    def _hover_shape(self, x, y):
        for i, a in enumerate(self.set.annotations):
            if a.type == ZONE and len(a.vertices) >= 3:
                if cv2.pointPolygonTest(self._px(a.vertices), (float(x), float(y)), False) >= 0:
                    return i
            else:
                pts = self._px(a.vertices)
                for j in range(len(pts) - 1):
                    if _seg_dist2(pts[j], pts[j + 1], (x, y)) < (10 * self.s) ** 2:
                        return i
        return None

    # ---------- mouse ----------
    def _on_mouse(self, event, x, y, flags, _):
        self.cursor = (x, y)
        self.snap = self._snap_target(x, y)
        if self.mode == IDLE and not (flags & cv2.EVENT_FLAG_LBUTTON):
            self.hover = self._hover_shape(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode in (DRAW_ZONE, DRAW_LINE):
                if self.snap and self.snap[2] == "close":
                    self._commit()
                    return
                px, py = (self.snap[0], self.snap[1]) if self.snap else (x, y)
                if self.draft:                       # dedupe near-duplicate clicks
                    lx, ly = self.draft[-1][0] * self.w, self.draft[-1][1] * self.h
                    if _dist2(lx, ly, px, py) < (SNAP_PX * self.s) ** 2:
                        return
                self.draft.append((px / self.w, py / self.h))
                self._draft_status()
            elif self.mode == IDLE:
                hit = self._hit_vertex(x, y)
                if hit:
                    self.drag, self.selected = hit, hit[0]
                    self.status = f"dragging {self.set.annotations[hit[0]].id}"
                else:
                    self._select_nearest(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag and (flags & cv2.EVENT_FLAG_LBUTTON):
                ai, vi = self.drag
                self.set.annotations[ai].vertices[vi] = (x / self.w, y / self.h)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag:
                self.status = f"moved vertex of {self.set.annotations[self.drag[0]].id}"
                self.drag = None

        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            if self.mode in (DRAW_ZONE, DRAW_LINE):
                self._commit()

    def _select_nearest(self, x, y):
        inside = self._hover_shape(x, y)
        if inside is not None:
            self.selected = inside
            self.status = f"selected {self.set.annotations[inside].id}"
            return
        best, bd = -1, (40 * self.s) ** 2
        for i, a in enumerate(self.set.annotations):
            for (nx, ny) in a.vertices:
                d = _dist2(nx * self.w, ny * self.h, x, y)
                if d < bd:
                    bd, best = d, i
        if best >= 0:
            self.selected = best
            self.status = f"selected {self.set.annotations[best].id}"

    def _draft_status(self):
        if self.mode == DRAW_ZONE:
            self.status = (f"zone: {len(self.draft)} pts - click first point or Enter to close"
                           if len(self.draft) >= 3 else f"zone: {len(self.draft)} pts (need 3+)")
        else:
            self.status = (f"line: {len(self.draft)} pts - Enter/right-click to finish"
                           if len(self.draft) >= 2 else "line: 1 pt - add more (polyline)")

    # ---------- commit / edit ----------
    def _commit(self):
        if self.mode == DRAW_ZONE and len(self.draft) >= 3:
            aid = self.set.next_id(ZONE)
            self.set.annotations.append(Annotation(aid, ZONE, list(self.draft)))
        elif self.mode == DRAW_LINE and len(self.draft) >= 2:
            aid = self.set.next_id(LINE)
            self.set.annotations.append(Annotation(aid, LINE, list(self.draft)))
        else:
            self.status = "need 3+ points for a zone, 2+ for a line"
            return
        self.selected = len(self.set.annotations) - 1
        self.status = f"added {aid}"
        self.draft, self.mode = [], IDLE

    def _flip_selected(self):
        a = self._sel()
        if a and a.type == LINE:
            a.vertices = list(reversed(a.vertices))
            self.status = f"flipped {a.id} direction"
        elif self.mode == DRAW_LINE and len(self.draft) >= 2:
            self.draft = list(reversed(self.draft))
            self.status = "flipped draft direction"

    # ---------- key handling ----------
    def _key(self, k: int) -> bool:
        if self.mode == NAMING:
            return self._key_naming(k)
        ch = chr(k & 0xFF).lower() if 0 <= (k & 0xFF) < 128 else ""
        if ch == "q":
            return False
        elif k == 27:                                  # Esc: cancel draft, else quit
            if self.mode in (DRAW_ZONE, DRAW_LINE):
                self.mode, self.draft, self.status = IDLE, [], "draft cancelled"
            else:
                return False
        elif ch == "z":
            self.mode, self.draft = DRAW_ZONE, []
            self.status = "ZONE: click vertices; click first point to close"
        elif ch == "l":
            self.mode, self.draft = DRAW_LINE, []
            self.status = "LINE: click points; Enter/right-click to finish (1st pt = arrow base)"
        elif k in (13, 10) or ch == "c":               # Enter / C
            self._commit()
        elif k in (8, 127) or ch == "u":               # Backspace / U
            if self.draft:
                self.draft.pop()
                self._draft_status()
            elif self._sel():
                self.status = f"{self._sel().id} selected - D deletes it"
        elif ch == "n" and self._sel():
            self.mode, self.name_buf = NAMING, self._sel().display_name
            self.status = "type a name, Enter confirm, Esc cancel"
        elif ch == "f":
            self._flip_selected()
        elif k == 9:                                   # Tab
            if self.set.annotations:
                self.selected = (self.selected + 1) % len(self.set.annotations)
                self.status = f"selected {self._sel().id}"
        elif ch == "d" and self._sel():
            removed = self.set.annotations.pop(self.selected)
            self.selected = min(self.selected, len(self.set.annotations) - 1)
            self.status = f"deleted {removed.id}"
        elif ch == "s":
            self.set.save(self.path)
            self.status = f"saved -> {self.path}"
        elif ch == "r":
            self.set.annotations.clear(); self.draft = []; self.selected = -1
            self.status = "reset all"
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "m":
            self.show_loupe = not self.show_loupe
        return True

    def _key_naming(self, k: int) -> bool:
        if k in (13, 10):
            a = self._sel()
            if a:
                a.display_name = self.name_buf
            self.mode, self.status = IDLE, f"named -> {self.name_buf}"
        elif k == 27:
            self.mode, self.status = IDLE, "rename cancelled"
        elif k in (8, 127):
            self.name_buf = self.name_buf[:-1]
        elif 32 <= (k & 0xFF) < 127:
            self.name_buf += chr(k & 0xFF)
        return True

    # ---------- drawing ----------
    def _draw(self) -> np.ndarray:
        self.tick += 1
        img = self.frame.copy()
        self._draw_glow(img)
        for i, a in enumerate(self.set.annotations):
            self._draw_annotation(img, a, i == self.selected)
        if self.draft:
            self._draw_draft(img)
        self._draw_cursor(img)
        if self.show_loupe and (self.mode in (DRAW_ZONE, DRAW_LINE) or self.drag):
            self._draw_loupe(img)
        self._draw_hud(img)
        return img

    def _draw_glow(self, img):
        """One blurred pass: soft halo behind the selected shape and the live draft."""
        layer = np.zeros_like(img)
        drew = False
        a = self._sel()
        if a:
            cv2.polylines(layer, [self._px(a.vertices)], a.type == ZONE,
                          self._col(a, True), self.t + 6, cv2.LINE_AA)
            drew = True
        if self.draft:
            cv2.polylines(layer, [self._px(self.draft)], False,
                          C_ZONE if self.mode == DRAW_ZONE else C_LINE,
                          self.t + 4, cv2.LINE_AA)
            drew = True
        if drew:
            cv2.GaussianBlur(layer, (0, 0), 7, dst=layer)
            cv2.addWeighted(img, 1.0, layer, 0.65, 0, img)

    def _direction(self, img, a, b, alpha=0.16, band=26, arrows=True):
        """Shade the positive (right-hand) side of segment a->b, with arrows."""
        nx, ny = _unit_normal(a, b)
        bw = band * self.s
        quad = np.array([a, b, [b[0] + nx * bw, b[1] + ny * bw],
                         [a[0] + nx * bw, a[1] + ny * bw]], np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [quad], C_POS)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
        if arrows:
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(L // (80 * self.s)))
            for i in range(1, n + 1):
                t = i / (n + 1)
                mx, my = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                cv2.arrowedLine(img, (int(mx), int(my)),
                                (int(mx + nx * 20 * self.s), int(my + ny * 20 * self.s)),
                                C_POS, max(1, int(2 * self.s)), cv2.LINE_AA, tipLength=0.5)

    def _plus_minus(self, img, pts):
        j = len(pts) // 2 - 1 if len(pts) > 2 else 0
        a, b = pts[j], pts[j + 1]
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        nx, ny = _unit_normal(a, b)
        d = 22 * self.s
        _tag(img, "+", (int(mx + nx * d), int(my + ny * d)), C_POS, scale=0.7 * self.s, thick=2)
        _tag(img, "-", (int(mx - nx * d), int(my - ny * d)), C_NEG, scale=0.7 * self.s, thick=2)

    def _ants(self, img, pts, col, closed):
        seq = list(pts) + ([pts[0]] if closed else [])
        off = (self.tick * 2) % 23
        for i in range(len(seq) - 1):
            _dashed(img, tuple(seq[i]), tuple(seq[i + 1]), col, self.t, 14, 9, off)

    def _vertices(self, img, pts, col, hot=-1):
        for i, p in enumerate(pts):
            r = int((8 if i == hot else 6) * self.s)
            cv2.circle(img, tuple(p), r, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), r, col, 2, cv2.LINE_AA)

    def _draw_annotation(self, img, a: Annotation, sel: bool):
        pts = self._px(a.vertices)
        col = self._col(a, sel)
        if a.type == ZONE:
            ov = img.copy()
            cv2.fillPoly(ov, [pts], col)
            cv2.addWeighted(ov, 0.16 + 0.08 * sel, img, 0.84 - 0.08 * sel, 0, img)
            if sel:
                self._ants(img, pts, col, closed=True)
            else:
                cv2.polylines(img, [pts], True, col, self.t, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], False, col, self.t, cv2.LINE_AA)
            for j in range(len(pts) - 1):
                self._direction(img, pts[j], pts[j + 1])
            self._plus_minus(img, pts)
            if sel:
                self._ants(img, pts, col, closed=False)
        self._vertices(img, pts, col)
        _tag(img, a.display_name or a.id, (pts[0][0] + 10, pts[0][1] - 8),
             col, scale=0.5 * self.s)

    def _draw_draft(self, img):
        pts = self._px(self.draft)
        cur = self.cursor
        if self.mode == DRAW_ZONE:
            col = C_ZONE
            preview = np.vstack([pts, [cur]]).astype(np.int32)
            if len(preview) >= 3:
                ov = img.copy()
                cv2.fillPoly(ov, [preview], col)
                cv2.addWeighted(ov, 0.14, img, 0.86, 0, img)
            cv2.polylines(img, [pts], False, col, self.t, cv2.LINE_AA)
            _dashed(img, tuple(pts[-1]), cur, col, self.t)               # rubber-band edge
            if len(pts) >= 2:
                _dashed(img, cur, tuple(pts[0]), col, 1, 10, 8)          # closing edge
                cv2.circle(img, tuple(pts[0]), int(10 * self.s), C_SNAP, 2, cv2.LINE_AA)
        else:
            col = C_LINE
            cv2.polylines(img, [pts], False, col, self.t, cv2.LINE_AA)
            for j in range(len(pts) - 1):
                self._direction(img, pts[j], pts[j + 1])
            _dashed(img, tuple(pts[-1]), cur, col, self.t)               # rubber-band segment
            self._direction(img, tuple(pts[-1]), cur)                    # preview direction
            dx, dy = cur[0] - pts[-1][0], cur[1] - pts[-1][1]
            _tag(img, f"{math.hypot(dx, dy):.0f}px  {math.degrees(math.atan2(-dy, dx)):+.0f}deg",
                 (cur[0] + 18, cur[1] - 18), col, scale=0.45 * self.s)
        for p in pts:
            cv2.drawMarker(img, tuple(p), col, cv2.MARKER_SQUARE, int(11 * self.s), 2, cv2.LINE_AA)

    def _draw_cursor(self, img):
        x, y = self.cursor
        _dashed(img, (0, y), (self.w, y), C_GUIDE, 1, 10, 8)
        _dashed(img, (x, 0), (x, self.h), C_GUIDE, 1, 10, 8)
        cv2.drawMarker(img, (x, y), C_CURSOR, cv2.MARKER_CROSS, int(22 * self.s), 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), int(7 * self.s), C_CURSOR, 1, cv2.LINE_AA)
        if self.snap:
            sx, sy, kind = int(self.snap[0]), int(self.snap[1]), self.snap[2]
            cv2.circle(img, (sx, sy), int(11 * self.s), C_SNAP, 2, cv2.LINE_AA)
            if kind == "close":
                _tag(img, "click to close", (sx + 16, sy - 12), C_SNAP, scale=0.45 * self.s)
        if self.mode in (DRAW_ZONE, DRAW_LINE):
            _tag(img, f"{x},{y}  ({x / self.w:.3f}, {y / self.h:.3f})",
                 (x + 18, y + 24), C_CURSOR, scale=0.42 * self.s)

    def _draw_loupe(self, img):
        x, y = self.cursor
        z, R = 5, int(70 * self.s)
        r = R // z
        x0 = int(np.clip(x - r, 0, self.w - 2 * r - 1))
        y0 = int(np.clip(y - r, 0, self.h - 2 * r - 1))
        patch = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
        if patch.shape[0] < 2 * r or patch.shape[1] < 2 * r:
            return
        mag = cv2.resize(patch, (2 * R, 2 * R), interpolation=cv2.INTER_NEAREST)
        px = self.w - 2 * R - int(16 * self.s)
        py = int(38 * self.s)
        mask = np.zeros((2 * R, 2 * R), np.uint8)
        cv2.circle(mask, (R, R), R, 255, -1)
        roi = img[py:py + 2 * R, px:px + 2 * R]
        if roi.shape[:2] == mag.shape[:2]:
            np.copyto(roi, mag, where=mask[..., None].astype(bool))
            cv2.circle(img, (px + R, py + R), R, C_CURSOR, 2, cv2.LINE_AA)
            crx, cry = int((x - x0) * z), int((y - y0) * z)
            cv2.drawMarker(img, (px + crx, py + cry), C_SNAP, cv2.MARKER_CROSS,
                           int(18 * self.s), 1, cv2.LINE_AA)

    def _hint(self) -> str:
        if self.mode == DRAW_ZONE:
            return "ZONE  |  click to add points  |  click first point / Enter to close  |  Bksp undo  |  Esc cancel"
        if self.mode == DRAW_LINE:
            return "LINE  |  click to add points (polyline)  |  Enter / right-click to finish  |  F flip  |  Bksp undo"
        if self.mode == NAMING:
            return "NAMING  |  type a name  |  Enter confirm  |  Esc cancel"
        return "IDLE  |  Z zone  |  L line  |  drag a vertex  |  click a shape to select  |  Tab cycle  |  H help"

    def _draw_hud(self, img):
        s = self.s
        bar = int(30 * s)
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (self.w, bar), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
        mode_col = {IDLE: C_CURSOR, DRAW_ZONE: C_ZONE,
                    DRAW_LINE: C_LINE, NAMING: C_SNAP}[self.mode]
        left = (f"[{self.mode.upper()}]  items:{len(self.set.annotations)}  "
                f"sel:{self._sel().id if self._sel() else '-'}")
        cv2.putText(img, left, (int(10 * s), int(20 * s)), FONT, 0.55 * s,
                    mode_col, max(1, int(1.3 * s)), cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(self.status, FONT, 0.5 * s, 1)
        cv2.putText(img, self.status, (self.w - tw - int(12 * s), int(20 * s)),
                    FONT, 0.5 * s, (205, 205, 205), 1, cv2.LINE_AA)
        if self.mode == NAMING:
            _tag(img, f"name > {self.name_buf}_", (int(10 * s), bar + int(24 * s)),
                 C_SNAP, scale=0.6 * s)
        _tag(img, self._hint(), (int(10 * s), self.h - int(14 * s)),
             (235, 235, 235), scale=0.5 * s)
        if self.show_help:
            self._draw_help(img)

    def _draw_help(self, img):
        s = self.s
        lines = [
            "Z  new zone        L  new line (polyline)",
            "click  add point   drag vertex  move",
            "Enter / right / dbl-click  commit shape",
            "click 1st pt  close zone   Bksp/U  undo",
            "F  flip dir   N  rename   Tab  cycle",
            "D  delete   S  save   R  reset",
            "M  loupe   H  help   Esc  cancel/quit",
        ]
        lh = int(20 * s)
        w_box = int(330 * s)
        x = self.w - w_box - int(12 * s)
        y0 = int(46 * s) + (2 * int(70 * s) + int(38 * s) if self.show_loupe and
                            self.mode in (DRAW_ZONE, DRAW_LINE) else 0)
        h_box = lh * (len(lines) + 1) + int(12 * s)
        ov = img.copy()
        cv2.rectangle(ov, (x, y0), (x + w_box, y0 + h_box), (18, 18, 18), -1)
        cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
        cv2.rectangle(img, (x, y0), (x + w_box, y0 + h_box), (90, 90, 90), 1, cv2.LINE_AA)
        # legend swatches
        ly = y0 + lh
        for label, col in (("zone", C_ZONE), ("line", C_LINE), ("selected", C_SEL)):
            cv2.rectangle(img, (x + int(10 * s), ly - int(10 * s)),
                          (x + int(24 * s), ly), col, -1)
            cv2.putText(img, label, (x + int(30 * s), ly), FONT, 0.42 * s,
                        (220, 220, 220), 1, cv2.LINE_AA)
            x += int(108 * s)
        x = self.w - w_box - int(12 * s)
        y = y0 + 2 * lh
        for ln in lines:
            cv2.putText(img, ln, (x + int(10 * s), y), FONT, 0.45 * s,
                        (225, 225, 225), 1, cv2.LINE_AA)
            y += lh

    # ---------- main loop ----------
    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)
        while True:
            cv2.imshow(self.win, self._draw())
            k = cv2.waitKeyEx(15)
            if k != -1 and not self._key(k):
                break
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        return self.set
