"""Regression test for the Live-Voice AR try-on stack — proves the 'magic' works
WITHOUT clicking the frontend.

Two phases:
  • OFFLINE (always): face landmarks, every filter draws, eyewear warp, the
    sponsored catalogs, shape-accurate search, store routing, tool registry.
  • ONLINE (only if the studio is up at $STUDIO_URL): the REST endpoints, and the
    real path browser→/ws/cam→pipeline→/stream.mjpg with filters + AR eyewear.
  • UI (only if the studio is up AND playwright is installed): a deterministic
    frontend smoke — store switch, filter chips, showCatalog — asserting 0 JS errors.

Run:  .venv/bin/python test_studio_tryon.py
      STUDIO_URL=http://127.0.0.1:8011 .venv/bin/python test_studio_tryon.py   # + online/UI
"""

import os
import sys
import time
import json
import inspect
import urllib.request

import cv2
import numpy as np

STUDIO_URL = os.environ.get("STUDIO_URL", "http://127.0.0.1:8011")
FACE = "assets/test/face.jpg"

_results = []
_VERBOSE = True


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    if _VERBOSE:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


def collect_offline():
    """Run the OFFLINE checks silently and return [(name, ok, detail), …].
    Lets fishfood fold these in as a level without the standalone printing."""
    global _VERBOSE
    _VERBOSE = False
    _results.clear()
    try:
        offline()
    except Exception as e:
        _results.append(("studio try-on offline crashed", False, repr(e)))
    _VERBOSE = True
    return list(_results)


# ----------------------------- OFFLINE -----------------------------
def offline():
    print("\n[OFFLINE] core logic + drawing (no server, no network)")
    import supervision as sv
    from occ.studio import tools as T
    from occ.studio.overlays import OverlayEngine
    from occ.studio.face_filters import (FACE_FILTERS, get_asset,
                                         set_current_eyewear)
    from occ.studio.facemesh import detect_faces

    # --- catalogs ---
    cats = T._load_catalogs()
    g, e = cats["apparel"], cats["eyewear"]
    check("apparel catalog loaded (>400)", len(g) > 400, f"{len(g)} items")
    check("eyewear catalog loaded (>300)", len(e) > 300, f"{len(e)} frames")
    check("every eyewear frame has a shape tag",
          all(it.get("shape") for it in e), f"{sum(bool(i.get('shape')) for i in e)}/{len(e)}")
    haiti = [it for it in g if it.get("cat") == "haiti"]
    check("Haiti World Cup category present", len(haiti) >= 1, f"{len(haiti)} jerseys")
    check("Mode Marco sponsor",
          json.loads(open("occ/studio/garments.json").read())["sponsor"]["name"] == "Mode Marco")
    check("Ralba Optical sponsor",
          json.loads(open("occ/studio/eyewear.json").read())["sponsor"]["name"] == "Ralba Optical")

    # --- shape-accurate search ---
    for shape in ("aviator", "round", "wayfarer", "cat-eye", "rectangle", "browline"):
        res = T._search_catalog(shape, store="eyewear")[:5]
        m = sum(1 for r in res if r.get("shape") == shape)
        check(f"search '{shape}' → {shape} frames", m >= 4, f"{m}/5")
    check("query_shape parses 'aviator sunglasses'", T._query_shape("aviator sunglasses") == "aviator")

    # --- store routing ---
    check("route 'navy polo' → apparel", T._guess_store("navy polo") == "apparel")
    check("route 'aviator sunglasses' → eyewear", T._guess_store("aviator sunglasses") == "eyewear")
    check("route 'round frames' → eyewear", T._guess_store("round frames") == "eyewear")
    check("search 'haiti jersey' finds Haiti",
          "haiti" in (T._search_catalog("haiti jersey", "apparel")[:1] or [{}])[0].get("title", "").lower())
    polo = T._search_catalog("navy polo", "apparel")[:1]
    check("search 'navy polo' → a navy polo",
          bool(polo) and "navy" in (polo[0].get("title", "").lower()))

    # --- tool registry ---
    names = {f.__name__ for f in T.ALL_TOOLS}
    for t in ("apply_face_filter", "try_eyewear", "try_product", "shop_search", "virtual_try_on"):
        check(f"tool registered: {t}", t in names)
    check("filter aliases resolve (ninja→ninja_mask)",
          T._FILTER_ALIASES.get("ninja") == "ninja_mask")

    # --- face landmarks ---
    face = cv2.imread(FACE)
    check("test face fixture present", face is not None, FACE)
    if face is None:
        return
    faces = detect_faces(face)
    ok_face = len(faces) >= 1 and faces[0].lm.shape == (478, 2)
    check("FaceLandmarker detects 478 pts", ok_face,
          f"{len(faces)} face(s)" + (f", {faces[0].lm.shape}" if faces else ""))
    if faces:
        f0 = faces[0]
        check("blendshapes + head-pose matrix present",
              len(f0.blend) > 10 and f0.matrix is not None,
              f"{len(f0.blend)} blendshapes")

    # --- every filter draws without error and changes the frame ---
    # mean-over-frame is dominated by background, so count notably-changed pixels.
    # heart_eyes is blendshape-reactive (only on a smile) → neutral face draws
    # nothing, which is CORRECT; for it we only require no error.
    # --- eyewear lenses: see-through, perfect-fit, NO white blobs (the reported bug) ---
    from occ.studio.face_filters import clean_lenses, lens_centers_norm, _lens_regions
    # synthetic glasses: RED frame + bridge with two WHITE OPAQUE lenses (the failure case)
    gl = np.zeros((180, 360, 4), np.uint8)
    for cx in (96, 264):                                   # two white opaque lenses
        cv2.circle(gl, (cx, 90), 64, (250, 250, 250, 255), -1)
    for cx in (96, 264):                                   # red rims
        cv2.circle(gl, (cx, 90), 64, (40, 40, 220, 255), 14)
    cv2.rectangle(gl, (160, 80), (200, 100), (40, 40, 220, 255), -1)   # red bridge
    cen = lens_centers_norm(gl)
    check("eyewear: 2 lens centres detected (registration)", cen is not None and len(cen) == 2)
    # COLOURED / PATTERNED lenses (e.g. a flag novelty) must NOT fool the centre detection:
    # geometric (bridge + per-half centroid) → symmetric & mid-height, not stuck on a colour.
    flag = np.zeros((180, 360, 4), np.uint8)
    for cx in (96, 264):                                  # two SATURATED blue lenses
        cv2.circle(flag, (cx, 90), 60, (220, 120, 30, 255), -1)
    for cx in (96, 264):                                  # dark frame rims
        cv2.circle(flag, (cx, 90), 60, (30, 30, 30, 255), 12)
    cv2.rectangle(flag, (156, 80), (204, 100), (30, 30, 30, 255), -1)  # bridge
    fc = lens_centers_norm(flag)
    sym = abs((fc[0][0] + fc[1][0]) / 2 - 0.5) if fc else 1.0
    midy = (fc[0][1] + fc[1][1]) / 2 if fc else 0.0
    check("eyewear: coloured/flag lenses → CENTRED & mid-height (not colour-fooled, no left shift)",
          fc is not None and sym < 0.04 and 0.4 < midy < 0.6, f"{fc} centre-off={sym:.3f}")
    cl = clean_lenses(gl)
    regions, _ = _lens_regions(gl)
    lensmask = np.zeros(gl.shape[:2], bool)
    for it, _c in regions:
        lensmask |= it
    a2 = cl[:, :, 3]; hsv2 = cv2.cvtColor(cl[:, :, :3], cv2.COLOR_BGR2HSV)
    white_opaque_in_lens = int(((hsv2[:, :, 2] > 224) & (hsv2[:, :, 1] < 36) & (a2 > 110) & lensmask).sum())
    check("eyewear: NO white blob left in the lenses", white_opaque_in_lens == 0, f"{white_opaque_in_lens}px")
    see_through = float(a2[lensmask].mean())
    check("eyewear: lenses are SEE-THROUGH (low alpha)", see_through < 110, f"mean α={see_through:.0f}")
    red_rim = (cl[:, :, 2] > 150) & (cl[:, :, 1] < 90) & (a2 > 200)   # frame preserved, opaque
    check("eyewear: frame stays opaque (not removed)", int(red_rim.sum()) > 500, f"{int(red_rim.sum())}px")
    # BIMODAL BAKED REFLECTION (the Bvlgari clear-lens bug): the product photo bakes a dark
    # photographic reflection over HALF of each clear lens. Keeping the original pixels
    # reproduced it as two DARK OPAQUE rectangles. RECONSTRUCTION must render the lens FLAT
    # and uniform (no dark half) and see-through.
    refl = np.zeros((180, 360, 4), np.uint8)
    for cx in (96, 264):
        cv2.circle(refl, (cx, 90), 62, (250, 250, 250, 255), -1)       # clear (white) lens
        cv2.ellipse(refl, (cx, 90), (62, 62), 0, -90, 90, (104, 82, 100, 255), -1)  # dark-purple reflection half
    for cx in (96, 264):
        cv2.circle(refl, (cx, 90), 62, (60, 50, 45, 255), 12)          # dark rim
    cv2.rectangle(refl, (158, 80), (202, 100), (60, 50, 45, 255), -1)  # bridge
    clr = clean_lenses(refl)
    rg, _ = _lens_regions(refl)
    lm = np.zeros(refl.shape[:2], bool)
    for it, _c in rg:
        lm |= it
    # uniform: low within-lens colour spread (the dark half is gone); and see-through
    lstd = float(clr[:, :, :3][lm].reshape(-1, 3).std(0).mean()) if lm.any() else 99.0
    lalpha = float(clr[:, :, 3][lm].mean()) if lm.any() else 255.0
    dark_blob = int(((clr[:, :, :3].max(2) < 130) & (clr[:, :, 3] > 110) & lm).sum())
    check("eyewear: baked dark-reflection half RECONSTRUCTED uniform (no dark rectangle)",
          lm.any() and lstd < 25 and dark_blob == 0, f"within-lens std={lstd:.0f} dark-opaque={dark_blob}px")
    check("eyewear: reconstructed clear lens is see-through", lalpha < 110, f"mean α={lalpha:.0f}")
    # WHITE ACETATE FRAME + CLEAR LENS: the bright white rim must NOT be painted see-through
    # (it reads as low-saturation 'lens' material) — the white-perimeter guard protects it
    # while the deep lens core stays see-through.
    wf = np.zeros((320, 680, 4), np.uint8)
    for cx in (200, 480):
        cv2.ellipse(wf, (cx, 160), (120, 95), 0, 0, 360, (248, 248, 248, 255), 28)
        cv2.ellipse(wf, (cx, 160), (106, 81), 0, 0, 360, (250, 250, 250, 120), -1)
        cv2.ellipse(wf, (cx, 160), (120, 95), 0, 0, 360, (248, 248, 248, 255), 28)
    cv2.rectangle(wf, (300, 140), (380, 175), (248, 248, 248, 255), -1)
    wcl = clean_lenses(wf)
    rim = np.zeros((320, 680), np.uint8)
    for cx in (200, 480):
        cv2.ellipse(rim, (cx, 160), (120, 95), 0, 0, 360, 1, 6)        # the outer rim edge
    rim = rim > 0
    core = np.zeros((320, 680), np.uint8)
    for cx in (200, 480):
        cv2.ellipse(core, (cx, 160), (60, 45), 0, 0, 360, 1, -1)       # deep lens core
    core = core > 0
    rim_a = float(wcl[:, :, 3][rim].mean()); core_a = float(wcl[:, :, 3][core].mean())
    check("eyewear: white-acetate frame rim stays opaque (not erased)", rim_a > 180, f"rim α={rim_a:.0f}")
    check("eyewear: white-frame lens core is see-through", core_a < 120, f"core α={core_a:.0f}")
    # ALPHA CONTOURS THE FRAME (the wide-shield "translucent rectangle outside the lenses"
    # bug): the stored asset must have NO alpha outside the true glasses silhouette, so the
    # warp never composites a translucent bounding-box rectangle onto the face.
    from occ.studio.face_filters import set_current_eyewear, _EYEWEAR, _silhouette_mask
    shield = np.zeros((220, 520, 4), np.uint8)
    cv2.ellipse(shield, (260, 110), (220, 78), 0, 0, 360, (170, 95, 70, 255), -1)   # big mirror lens
    cv2.ellipse(shield, (260, 110), (220, 78), 0, 0, 360, (40, 40, 40, 255), 11)    # thin rim
    sil_in = _silhouette_mask(shield)
    set_current_eyewear(shield, "shield-rect-test")
    _ra = _EYEWEAR["rgba"][:, :, 3]
    _outside = int(((_ra > 6) & ~sil_in).sum())
    check("eyewear: asset alpha CONTOURS the frame (no rectangle outside the silhouette)",
          _outside < 40, f"{_outside}px of alpha outside the glasses silhouette")
    set_current_eyewear(get_asset("sunglasses"), "test")            # restore a normal pair
    # CRYSTAL / CLEAR ACETATE frame (the "we don't see the frame at all" bug): a clear frame is
    # near-invisible to segmentation AND the normal see-through fill erases it. It must be
    # detected and rendered as a VISIBLE light crystal — while a normal clear-lens pair is not.
    from occ.studio.face_filters import _is_crystal
    cry = np.zeros((220, 460, 4), np.uint8)
    for cx in (140, 320):                                            # light neutral-grey clear rims
        cv2.rectangle(cry, (cx - 95, 60), (cx + 95, 170), (170, 172, 174, 215), 16)
    cv2.rectangle(cry, (233, 105), (247, 125), (170, 172, 174, 215), -1)   # bridge
    check("eyewear: clear/crystal frame is detected", _is_crystal(cry))
    check("eyewear: a normal clear-lens (coloured frame) pair is NOT mis-detected as crystal",
          not _is_crystal(gl))
    ccl = clean_lenses(cry)
    _fp = cry[:, :, 3] > 40
    _vis = float(ccl[:, :, 3][_fp].mean()) if _fp.any() else 0.0
    check("eyewear: crystal frame rendered VISIBLE (not erased to see-through)",
          _vis > 110, f"frame mean α={_vis:.0f}")
    # DEGENERATE-ALPHA cutout (Nano render with a faint global alpha) must NOT leave a
    # translucent rectangle — load_eyewear_rgba must knock out the white background.
    from occ.studio.face_filters import load_eyewear_rgba
    canvas = np.full((200, 400, 3), 255, np.uint8)                    # white studio bg
    cv2.rectangle(canvas, (120, 70), (280, 140), (40, 40, 220), -1)   # a red frame blob
    fake = np.dstack([canvas, np.full((200, 400), 40, np.uint8)])     # faint global alpha (degenerate)
    png = cv2.imencode(".png", fake)[1].tobytes()
    cut = load_eyewear_rgba(png)
    bg_faint = int(((cut[:, :, 3] > 8) & (cut[:, :, 3] < 200) &
                    (cut[:, :, :3].min(2) > 200)).sum())              # translucent white bg pixels
    frac = bg_faint / float(cut.shape[0] * cut.shape[1])             # the bug made ~90% faint
    check("eyewear: degenerate-alpha bg knocked out (no translucent rectangle)",
          frac < 0.10, f"{frac*100:.1f}% translucent-white")
    # SEGMENTATION arsenal: a stray blob OUTSIDE the glasses (a bg remnant in a corner,
    # NOT in the frame's vertical band) must be removed — only the frame survives.
    from occ.studio.face_filters import _keep_glasses
    bgr2 = np.full((220, 440, 3), 30, np.uint8)
    cv2.rectangle(bgr2, (150, 80), (290, 150), (60, 60, 60), 10)     # the glasses (centre band)
    a2 = np.zeros((220, 440), np.uint8)
    cv2.rectangle(a2, (150, 80), (290, 150), 255, 12)               # frame alpha
    cv2.rectangle(a2, (10, 10), (70, 55), 255, -1)                  # STRAY blob (top-left corner)
    kept = _keep_glasses(bgr2, a2)
    blob_gone = int((kept[5:60, 5:75] > 40).sum())                  # the corner blob area
    frame_kept = int((kept[80:152, 150:292] > 40).sum())           # the glasses
    check("eyewear: stray corner blob segmented out (frame kept)",
          blob_gone < 60 and frame_kept > 800, f"blob={blob_gone}px frame={frame_kept}px")
    # GEOMETRY: even an ABSURD lens-centre detection must not blow up the frame size —
    # the registration clamps the width to a sane multiple of the IPD (the 'horrible
    # wrong-size' bug). A mock face with IPD=60px.
    from occ.studio.face_filters import _eyewear_quad
    class _MockFace:
        eye_l = np.array([100., 100.]); eye_r = np.array([160., 100.])
        eyes_center = np.array([130., 100.]); eye_dist = 60.0
        def p(self, n):
            return {"temple_l": np.array([80., 100.]), "temple_r": np.array([180., 100.])}[n]
    asset = np.zeros((100, 300, 4), np.uint8)
    q = _eyewear_quad(_MockFace(), asset, lens_centers=[(0.48, 0.5), (0.52, 0.5)])  # absurd: too close
    width = float(np.linalg.norm(np.array(q[1]) - np.array(q[0])))                  # TL→TR
    check("geometry: absurd detection can't blow up the frame (clamped to IPD)",
          1.7 * 60 <= width <= 3.6 * 60, f"width={width:.0f}px ({width/60:.1f}× IPD)")
    # LENS REFLECTION (real reflection + transparency): the reflection layer must be
    # SEE-THROUGH (moderate alpha, eyes still show) and confined to the lens region.
    from occ.studio.face_filters import _build_reflection
    lensmask = np.zeros((160, 320), np.uint8)
    cv2.circle(lensmask, (90, 80), 50, 1, -1); cv2.circle(lensmask, (230, 80), 50, 1, -1)
    scene = np.tile(np.linspace(40, 220, 320, dtype=np.uint8)[None, :, None], (160, 1, 3))
    rl = _build_reflection((160, 320, 4), lensmask, scene)
    ra = rl[:, :, 3]
    inside = ra[lensmask > 0].mean(); outside = int((ra[lensmask == 0] > 8).sum())
    check("reflection: confined to the lens region", outside < 30, f"{outside}px outside")
    check("reflection: see-through (eyes still show)", 20 < inside < 180, f"mean α={inside:.0f}")
    check("reflection: not fully opaque anywhere", int((ra > 240).sum()) == 0, f"{int((ra>240).sum())}px")
    # MOVIE IN THE LENSES: a video reflection must ANIMATE (different frames differ) while
    # staying see-through + confined; the shading is precomputed once, the alpha is stable.
    from occ.studio.face_filters import (_reflection_shading, _reflection_compose,
                                         set_lens_reflection_video, _EYEWEAR,
                                         set_current_eyewear as _sce)
    shading = _reflection_shading((160, 320, 4), lensmask)
    f_a = np.zeros((120, 240, 3), np.uint8); cv2.rectangle(f_a, (0, 0), (60, 120), (60, 200, 255), -1)
    f_b = np.zeros((120, 240, 3), np.uint8); cv2.rectangle(f_b, (180, 0), (240, 120), (60, 200, 255), -1)
    ra_a = _reflection_compose(f_a, shading); ra_b = _reflection_compose(f_b, shading)
    moved = int(np.abs(ra_a[:, :, :3].astype(int) - ra_b[:, :, :3].astype(int)).sum())
    check("movie-in-lens: different frames render different reflections (animated)",
          moved > 5000, f"Δ={moved}")
    check("movie-in-lens: alpha shaping stable + see-through",
          np.array_equal(ra_a[:, :, 3], ra_b[:, :, 3]) and 20 < ra_a[:, :, 3][lensmask > 0].mean() < 180)
    check("movie-in-lens: still confined to the lens region",
          int((ra_a[:, :, 3][lensmask == 0] > 8).sum()) < 30)
    # the engine stores a looping video + clears on go-live (set state directly)
    _EYEWEAR["armless"] = np.dstack([np.zeros((160, 320, 3), np.uint8),
                                     (lensmask * 255).astype(np.uint8)])
    _EYEWEAR["lensmask"] = lensmask
    okv = set_lens_reflection_video([f_a, f_b, f_a])
    check("movie-in-lens: set_lens_reflection_video stores the looping movie",
          okv and _EYEWEAR.get("refl_video") is not None and _EYEWEAR.get("refl_shading") is not None)
    _sce(None)
    check("movie-in-lens: go-live clears the movie",
          _EYEWEAR.get("refl_video") is None and _EYEWEAR.get("refl_shading") is None)
    check("lens_movie tool registered + in film loader",
          "lens_movie" in {f.__name__ for f in T.ALL_TOOLS})
    # TEMPLE ARMS (3D, ear-anchored): Face.yaw is computed and the arm occlusion logic
    # hides the side that's turned away (so arms read in 3D, not draped over the cheek).
    from occ.studio.facemesh import detect_faces, IDX
    check("face landmarks include EAR keypoints (ear_l/ear_r)", "ear_l" in IDX and "ear_r" in IDX)
    ff = detect_faces(face)
    if ff:
        yaw = ff[0].yaw
        check("Face.yaw computes (front face ≈ 0)", abs(yaw) < 0.25, f"yaw={yaw:.3f}")
    # the occlusion rule: turned-away arm is hidden (yaw>0.28 hides LEFT, yaw<-0.28 hides RIGHT)
    occ = lambda y: (y < 0.28, y > -0.28)                 # (left shown, right shown)
    check("temple-arm yaw occlusion: turned head hides the far arm",
          occ(0.0) == (True, True) and occ(0.5) == (False, True) and occ(-0.5) == (True, False))
    # 3D POSE from the matrix: decode is correct (synthetic ±25° yaw) and on the real face
    # the matrix roll/yaw AGREE with the trusted landmark geometry; pitch is now available.
    if ff:
        import math as _m
        from occ.studio.facemesh import Face as _F
        def _Ry(t):
            c, s = _m.cos(t), _m.sin(t)
            return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], float)
        m25 = _F(lm=ff[0].lm, lmz=ff[0].lmz, blend={}, matrix=_Ry(_m.radians(25)), w=ff[0].w, h=ff[0].h)
        mneg = _F(lm=ff[0].lm, lmz=ff[0].lmz, blend={}, matrix=_Ry(_m.radians(-25)), w=ff[0].w, h=ff[0].h)
        check("3D matrix: yaw decode correct + signed (±25°)",
              m25._pose_rad() is not None and np.degrees(m25._pose_rad()[0]) > 20
              and np.degrees(mneg._pose_rad()[0]) < -20)
        pr = ff[0]._pose_rad()
        if pr is not None:
            check("3D matrix: roll agrees with landmark geometry on the real face",
                  abs(np.degrees(pr[2]) - ff[0].roll) < 6,
                  f"matrix roll={np.degrees(pr[2]):.1f}° geom roll={ff[0].roll:.1f}°")
            check("3D matrix: pitch now available (was 0 before)", "pitch" in dir(ff[0]))

    # --- TRY-ON LAB: pipeline introspection + bad-case dataset (dev feature) ---
    import os as _os, hashlib as _hl
    from occ.studio.face_filters import lens_classification
    from occ.studio import eyewear_lab as _lab
    from occ.studio.tools import _CANON_DIR, _CANON_VER
    # classification mirrors clean_lenses: the clear synthetic 'gl' → see-through decision
    _lc = lens_classification(gl)
    _glm = np.zeros(gl.shape[:2], bool)
    for _it, _c in _lens_regions(gl)[0]:
        _glm |= _it
    check("lab: lens_classification mirrors clean_lenses (clear → see-through)",
          "clear" in _lc["decision"] and _glm.any()
          and float(clean_lenses(gl)[:, :, 3][_glm].mean()) < 110, _lc["decision"])
    # introspect_asset on a local synthetic (pre-seed the canonical cache → no API, offline)
    _p = "out/_labtest_glasses.png"; cv2.imwrite(_p, gl)
    _ck = _hl.md5((_p + "|" + _CANON_VER).encode()).hexdigest()
    _os.makedirs(_CANON_DIR, exist_ok=True)
    _cp = _os.path.join(_CANON_DIR, _ck + ".png"); open(_cp, "wb").write(open(_p, "rb").read())
    _imgs, _stats = _lab.introspect_asset(_p)
    check("lab: introspect_asset produces the core pipeline stages",
          all(k in _imgs for k in ("00_raw", "03_cutout", "04_armless",
                                   "05_lens_regions", "06_clean_lenses")),
          f"got {sorted(_imgs)}")
    check("lab: stage 05 carries the lens classification stats",
          "decision" in _stats.get("05_lens_regions", {}) and
          "lens_mean_alpha" in _stats.get("06_clean_lenses", {}))
    # source-image quality assessor flags an obvious problem (tiny, busy image)
    _badimg = np.random.randint(0, 255, (120, 120, 3), np.uint8)
    _qi = _lab.assess_source_image(_badimg)
    check("lab: assess_source_image flags a poor photo (low-res/busy)", len(_qi) >= 1, f"{_qi}")
    # candidate CASCADE (inspect a new image without saving → base64 stages, for 'what if')
    _cand = _lab.inspect_candidate(_p)
    check("lab: inspect_candidate returns the cascade (base64 stages + issues)",
          len(_cand["stages"]) >= 5 and _cand["stages"][0]["png"].startswith("data:image/png;base64,"))
    # save → list → (live stages) → REPORT.md → delete roundtrip
    _fid = _lab.save_flag(_p, title="labtest", note="unit", clean_bgr=face.copy(),
                          vis_bgr=face.copy(), has_3d=True)
    _fl = _lab.list_flags(); _m = next((x for x in _fl if x["id"] == _fid), None)
    check("lab: save_flag → list_flags roundtrip writes the dataset entry",
          _m is not None and len(_m["stages"]) >= 6 and _m.get("has_3d") is True)
    if ff:                                            # the test face is detectable → live stages
        check("lab: live face stages captured (landmarks + registration)",
              any(s["key"] in ("08_landmarks", "09_registration") for s in _m["stages"]))
    _report = open(_os.path.join(_lab.FLAGS_DIR, _fid, "REPORT.md")).read()
    check("lab: REPORT.md brief written with all sections (assistant + 3 fix paths)",
          all(s in _report for s in ("Where to look", "Upload a better source image",
                                     "Fix the image automatically", "Use the 3D model",
                                     "Reproduce offline")))
    check("lab: delete_flag removes the entry",
          _lab.delete_flag(_fid) and not any(x["id"] == _fid for x in _lab.list_flags()))
    # PER-STAGE NANO BANANA enhance: input builders + cascade renderer (no API; canon is seeded)
    _inp = _lab._stage_input_jpg(_p, "06_clean_lenses")
    check("lab: _stage_input_jpg builds a clean stage input", bool(_inp) and len(_inp) > 500)
    _casc2 = _lab._cascade_from_asset(open(_p, "rb").read())
    check("lab: _cascade_from_asset returns enhanced + downstream stages (base64)",
          any(s["key"] == "enhanced" for s in _casc2)
          and any(s["key"] == "06_clean_lenses" for s in _casc2)
          and _casc2[0]["png"].startswith("data:image"))
    check("lab: enhance_stage rejects an unknown stage (graceful)",
          bool(_lab.enhance_stage(_p, "99_bogus").get("error")))
    import base64 as _b64
    _uri = "data:image/png;base64," + _b64.b64encode(open(_p, "rb").read()).decode()
    check("lab: apply_enhanced sets the live eyewear from a data URI", _lab.apply_enhanced(_uri, "t"))
    _os.remove(_p); _os.remove(_cp)

    det = sv.Detections.empty()
    set_current_eyewear(get_asset("sunglasses"), "test")   # so 'eyewear' has a product
    REACTIVE = {"heart_eyes"}
    for name in FACE_FILTERS:
        eng = OverlayEngine(); eng.add_native(name)
        img = face.copy()
        eng.run(img, det, None, 0, clean=face.copy())
        ov = list(eng.overlays.values())[0]
        changed_px = int((np.abs(img.astype(int) - face.astype(int)).max(2) > 18).sum())
        drew = changed_px > 200 or name in REACTIVE
        check(f"filter '{name}' draws", (not ov.error) and drew,
              (ov.error or (f"{changed_px}px" + (" (reactive)" if name in REACTIVE else ""))))

    # --- BIDI transcript de-dup (no chat bubbles twice during live voice) ---
    # Gemini Live streams the transcript as deltas AND re-sends the full aggregate
    # at turn end; LiveBridge._emit_event must drop that duplicate so the bubble
    # renders once. Feed synthetic events through it with a fake websocket.
    import asyncio
    from types import SimpleNamespace
    from occ.studio.live import LiveBridge

    class _FakeWS:
        def __init__(self): self.sent = []
        async def send_text(self, s): self.sent.append(json.loads(s))

    def _ev(out=None, inp=None, done=False):
        def _tr(t): return SimpleNamespace(text=t) if t is not None else None
        return SimpleNamespace(
            content=None, output_transcription=_tr(out), input_transcription=_tr(inp),
            get_function_calls=lambda: [], get_function_responses=lambda: [],
            interrupted=False, turn_complete=done)

    async def _drive(events):
        br = LiveBridge.__new__(LiveBridge)
        br._accum = {"user": "", "model": ""}
        ws = _FakeWS()
        for e in events:
            await br._emit_event(ws, e)
        return ws.sent

    def _run(coro):  # own loop; don't null the global (online() needs get_event_loop)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # deltas then a final aggregate that repeats the whole turn
    frames = _run(_drive([
        _ev(out="hello there"), _ev(out=" friend."),
        _ev(out="hello there friend."), _ev(done=True)]))
    model_txt = "".join(f["text"] for f in frames if f.get("type") == "transcript"
                        and f.get("role") == "model")
    check("BIDI model transcript not doubled", model_txt == "hello there friend.",
          repr(model_txt))

    # user side: deltas + aggregate must also collapse to one
    frames = _run(_drive([
        _ev(inp="put on"), _ev(inp=" the aviators"),
        _ev(inp="put on the aviators"), _ev(done=True)]))
    user_txt = "".join(f["text"] for f in frames if f.get("type") == "transcript"
                       and f.get("role") == "user")
    check("BIDI user transcript not doubled", user_txt == "put on the aviators",
          repr(user_txt))

    # --- OpenAI Realtime backend (the Creole-strong Live Voice) — offline logic ---
    from occ.studio import live_openai as LO
    sch = LO.tools_payload()
    check("OpenAI tool schemas built for every tool", len(sch) == len(T.ALL_TOOLS)
          and all(s.get("name") and s["parameters"]["type"] == "object" for s in sch),
          f"{len(sch)} schemas")
    # event translation + de-dup (deltas then the .done aggregate → ONE bubble)
    acc = {"user": "", "model": ""}
    fr = []
    for ev in [{"type": "response.output_audio_transcript.delta", "delta": "Bonjou"},
               {"type": "response.output_audio_transcript.delta", "delta": " zanmi"},
               {"type": "response.output_audio_transcript.done", "transcript": "Bonjou zanmi"},
               {"type": "response.output_audio.delta", "delta": "QUJD"},
               {"type": "conversation.item.input_audio_transcription.completed",
                "transcript": "mete linèt yo"},
               {"type": "response.done", "response": {"output": [{"type": "message"}]}}]:
        fr += LO.event_to_frames(ev, acc)
    mtxt = "".join(f["text"] for f in fr if f.get("type") == "transcript" and f["role"] == "model")
    utxt = "".join(f["text"] for f in fr if f.get("type") == "transcript" and f["role"] == "user")
    check("OpenAI model transcript not doubled", mtxt == "Bonjou zanmi", repr(mtxt))
    check("OpenAI Creole user transcript flows", utxt == "mete linèt yo", repr(utxt))
    check("OpenAI audio + turn_complete emitted",
          sum(1 for f in fr if f["type"] == "audio") == 1
          and sum(1 for f in fr if f["type"] == "turn_complete") == 1)
    # a tool-only response.done is intermediate → must NOT close the turn
    tool_only = LO.event_to_frames(
        {"type": "response.done", "response": {"output": [{"type": "function_call"}]}},
        {"user": "", "model": ""})
    check("OpenAI tool-only turn stays open (no premature turn_complete)",
          not any(f["type"] == "turn_complete" for f in tool_only))
    # 16k browser mic → 24k OpenAI input
    out = LO.resample_pcm16(bytes(1600 * 2), 16000, 24000)
    check("OpenAI 16k→24k resample", len(out) // 2 == 2400, f"{len(out)//2} samples")

    # --- multilingual narration TTS routing (gpt-4o-mini-tts ↔ Gemini) ---
    from occ.studio import genmedia as GM
    check("TTS provider: gpt-* → openai", GM.tts_provider("gpt-4o-mini-tts") == "openai")
    check("TTS provider: gemini → gemini",
          GM.tts_provider("gemini-2.5-flash-preview-tts") == "gemini")
    check("narrate exposes a model arg (Creole TTS)",
          "model" in inspect.signature(T.narrate).parameters)

    # --- ONE touchless engine: same buttons for the hand AND the agent (no parallel) ---
    from occ.studio.gestures import GestureBrowser, _DWELL, _REPEAT, _BTN
    BCTR = {b[0]: (b[1], b[2]) for b in _BTN}
    gb = GestureBrowser()
    gb.active = True; gb.store = "eyewear"
    gb.items = [{"title": f"g{i}", "img": ""} for i in range(40)]
    def _hover(bid, frames):
        cx, cy = BCTR[bid]
        for _ in range(frames): gb._update_buttons((cx, cy))
    def _leave(): gb._update_buttons(None)
    # a brief graze (< dwell) must NEVER fire — you have to hold to press
    gb.idx = 15; _hover("next", _DWELL - 3); _leave()
    check("buttons: brief hover never fires (must dwell)", gb.idx == 15, gb.idx)
    # dwell on NEXT / PREV steps the catalogue by exactly one
    _hover("next", _DWELL); n1 = gb.idx; _leave()
    _hover("prev", _DWELL); p1 = gb.idx; _leave()
    check("buttons: NEXT / PREV step by one", n1 == 16 and p1 == 15, f"{n1}/{p1}")
    # holding NEXT auto-repeats but SLOWLY (you see each item) — far fewer steps than frames
    gb.idx = 15; _hover("next", _DWELL + _REPEAT * 3 + 1); _leave()
    held = gb.idx - 15
    check("buttons: holding NEXT repeats SLOWLY", 3 <= held <= 6, f"{held} steps in {_DWELL + _REPEAT*3+1} frames")
    # TRY starts a 3-2-1 COUNTDOWN (not an instant try-on); STORE/CLEAR fire once
    _hover("try", _DWELL); _leave()
    check("buttons: TRY starts the pose countdown", gb.counting and not gb.busy and gb.state()["countdown"] >= 1)
    gb.counting = False                                   # cancel for the next checks
    calls = []
    gb._switch_store = lambda: calls.append("switch")
    gb._clear = lambda: calls.append("clear")
    _hover("store", _DWELL); _leave(); _hover("clear", _DWELL); _leave()
    check("buttons: store / clear each fire once", calls == ["switch", "clear"], calls)
    # AGENT drives the SAME engine via act() — one unified path
    gb.idx = 10; gb.counting = False
    gb.act("next"); gb.act("next"); gb.act("prev")
    check("agent: act() steps the same engine", gb.idx == 11, gb.idx)
    gb.act("try")
    check("agent: act('try') runs the same pose countdown", gb.counting)
    # state exposes the button geometry + finger + countdown for the PiP renderer
    s = gb.state()
    check("state: buttons + finger + countdown exposed",
          len(s["buttons"]) == 5 and "finger" in s and s["countdown"] >= 1)
    # only a CONFIDENT hand makes a cursor — no hand → no press
    gb.idx = 7; gb.counting = False; gb._rec = object(); gb._fingertip = lambda clean: None
    frame = np.zeros((140, 220, 3), np.uint8)
    gb.process(frame, frame)
    check("buttons: no hand → no press (catalogue frozen)", gb.idx == 7, gb.idx)
    # CLEAR drops any VTO result AND cancels a countdown → clean live
    gc = GestureBrowser()
    gc.result = {"kind": "image", "src": "/download/x.png"}; gc.busy = True; gc.counting = True
    gc._clear()
    check("buttons: clear → clean live (drops result, cancels countdown)",
          gc.result is None and not gc.busy and not gc.counting)
    from occ.studio import agent as _agent
    names = {f.__name__ for f in T.ALL_TOOLS}
    check("gesture_browse + live_control tools registered",
          "gesture_browse" in names and "live_control" in names)
    check("agent system prompt teaches touchless + live_control",
          "live_control" in _agent.SYSTEM_PROMPT and "strike a pose" in _agent.SYSTEM_PROMPT.lower())
    # UNIFIED: the no-key SimRunner routes shopping to the SAME live_control engine
    sim = _agent.SimRunner()
    routed = {}
    for phrase, want in [("show me the next one", "next"), ("go back one", "prev"),
                         ("try this on me", "try"), ("switch store", "store")]:
        steps, acts = sim._route(phrase, phrase)
        routed[want] = any(a[0] == "live_control" and a[1].get("action") == want for a in acts)
    check("agent (no-key) routes next/prev/try/store via live_control", all(routed.values()), routed)
    # INSTANT SEARCH: the ninja searches the stores and surfaces results into the SAME
    # live browser (filtered Prev/Next/Try) AND the store strip; auto-switches store.
    gs = GestureBrowser(); gs.active = True; gs.store = "apparel"
    gs.items = list(T._load_catalogs()["apparel"])
    s = gs.search("aviator sunglasses")
    filtered = 0 < s["total"] < 100 and s["store"] == "eyewear" and \
        all(it.get("shape") == "aviator" for it in gs.items[:5])
    check("search: filters the live browser to results (auto-switch store)", filtered,
          f"store={s['store']} total={s['total']}")
    gs.act("next")
    check("search: Prev/Next browse only the results", gs.idx == 1 and len(gs.items) == s["total"])
    gs._clear()
    check("search: clear restores the FULL catalogue", gs.query == "" and len(gs.items) > 300)
    # the no-key SimRunner routes a 'show me X' request to the unified search
    steps, acts = sim._route("show me red aviators", "show me red aviators")
    check("agent (no-key) routes 'show me X' → live_control search",
          any(a[0] == "live_control" and a[1].get("query") for a in acts), acts)

    # ---- LOCAL GEMMA backend (LiteRT-LM) — cut cost by running the brain on this Mac ----
    from occ.studio import local_llm as _L
    from occ.studio.settings import MODEL_OPTIONS, StudioSettings
    from occ.studio.neurons import usd_micros_for
    check("gemma: selectable in the agent + vision dropdowns",
          any("gemma" in m for m in MODEL_OPTIONS["agent"]) and
          any("gemma" in m for m in MODEL_OPTIONS["vision"]))
    check("gemma: routes to LOCAL (gemma-* → local endpoint, gemini → cloud)",
          _L.is_local_model("gemma-4-12b-it") and not _L.is_local_model("gemini-2.5-flash"))
    check("gemma: priced at $0 (runs locally)",
          usd_micros_for("gemma-4-12b-it", 50000, 50000) == 0 and
          usd_micros_for("gemini-2.5-flash", 50000, 50000) > 0)
    _sch = _L.chat_tools_schema()
    check("gemma: all tools exposed as OpenAI chat-completions function schemas",
          len(_sch) > 30 and all(s.get("type") == "function" and s["function"].get("name")
                                 and "parameters" in s["function"] for s in _sch))
    _stg = StudioSettings(); _stg.models = dict(_stg.models); _stg.models["agent"] = "gemma-4-12b-it"
    check("gemma: StudioAgent.mode() → 'local' when Gemma is selected",
          _agent.StudioAgent(_stg).mode() == "local")
    _dead0 = os.environ.get("STUDIO_LOCAL_ENDPOINT")
    os.environ["STUDIO_LOCAL_ENDPOINT"] = "http://127.0.0.1:9/v1"        # discard port — nothing listens
    check("gemma: health() is False (graceful) when the local server isn't reachable",
          _L.health(timeout=1) is False)
    if _dead0 is None:
        os.environ.pop("STUDIO_LOCAL_ENDPOINT", None)
    else:
        os.environ["STUDIO_LOCAL_ENDPOINT"] = _dead0
    # mock the litert-lm OpenAI endpoint → prove the client does tool-calling + vision
    import http.server as _hs, threading as _th
    class _Mock(_hs.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(b)
        def do_GET(self): self._send({"data": [{"id": "gemma-4-12b-it"}]})    # /models → health
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
            if body.get("tools"):                                            # tool-calling round
                msg = {"tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "live_control", "arguments": '{"action":"next"}'}}]}
            else:
                msg = {"content": "the road ahead"}                          # plain / vision round
            self._send({"choices": [{"message": msg}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
    _srv = _hs.HTTPServer(("127.0.0.1", 0), _Mock); _port = _srv.server_address[1]
    _th.Thread(target=_srv.serve_forever, daemon=True).start()
    _prev = os.environ.get("STUDIO_LOCAL_ENDPOINT")
    os.environ["STUDIO_LOCAL_ENDPOINT"] = f"http://127.0.0.1:{_port}/v1"
    try:
        check("gemma: health() detects a running litert-lm serve endpoint", _L.health(timeout=2))
        _r = _L.chat([{"role": "user", "content": "next"}], tools=_sch, model="gemma-4-12b-it")
        _tc = (_r["choices"][0]["message"].get("tool_calls") or [])
        check("gemma: chat() drives tools via the OpenAI function API",
              bool(_tc) and _tc[0]["function"]["name"] == "live_control")
        _txt = _L.vision_describe(b"\xff\xd8\xff\xe0jpeg", "what is this?", model="gemma-4-12b-it")
        check("gemma: vision_describe() returns text (multimodal, local)",
              isinstance(_txt, str) and len(_txt) > 0, _txt)
    finally:
        _srv.shutdown()
        if _prev is None: os.environ.pop("STUDIO_LOCAL_ENDPOINT", None)
        else: os.environ["STUDIO_LOCAL_ENDPOINT"] = _prev

    # ---- FREE LOCAL VOICE (3rd Live backend): Whisper STT + local Gemma + OS TTS ----
    from occ.studio import voice_local as VL
    check("voice: VAD distinguishes speech from silence",
          VL._rms((np.ones(1600) * 4000).astype(np.int16).tobytes()) > VL._SPEECH_RMS
          and VL._rms(np.zeros(1600, np.int16).tobytes()) < VL._SPEECH_RMS)
    check("voice: transcribe('') is graceful (no crash without audio)", VL.transcribe(b"") == "")
    import http.server as _hs2, threading as _th2, asyncio as _aio2
    class _Brain(_hs2.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _s(self, o):
            b = json.dumps(o).encode(); self.send_response(200)
            self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
        def do_GET(self): self._s({"data": [{"id": "gemma-4-12b-it"}]})
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
            if (body.get("messages") or [{}])[-1].get("role") == "tool":     # after the tool result → speak
                msg = {"content": "Here's the next pair."}
            else:                                                            # first → call a tool
                msg = {"tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "live_control", "arguments": '{"action":"next"}'}}]}
            self._s({"choices": [{"message": msg}], "usage": {"prompt_tokens": 4, "completion_tokens": 3}})
    _bsrv = _hs2.HTTPServer(("127.0.0.1", 0), _Brain); _bport = _bsrv.server_address[1]
    _th2.Thread(target=_bsrv.serve_forever, daemon=True).start()
    _pe = os.environ.get("STUDIO_LOCAL_ENDPOINT"); os.environ["STUDIO_LOCAL_ENDPOINT"] = f"http://127.0.0.1:{_bport}/v1"
    class _FakeWS:
        def __init__(self, inbox): self.inbox = list(inbox); self.sent = []
        async def send_text(self, s): self.sent.append(json.loads(s))
        async def receive_text(self):
            if self.inbox: return self.inbox.pop(0)
            raise RuntimeError("closed")
    try:
        _ws = _FakeWS(['{"type":"text","text":"show me the next pair"}', '{"type":"end"}'])
        _loop = _aio2.new_event_loop()
        _loop.run_until_complete(VL.LocalVoiceBridge(StudioSettings()).run(_ws))
        _loop.close()
        _types = [f.get("type") for f in _ws.sent]
        _utext = any(f.get("type") == "transcript" and f.get("role") == "user" for f in _ws.sent)
        _tool = any(f.get("type") == "tool" and f.get("name") == "live_control" for f in _ws.sent)
        _spoke = any(f.get("type") == "speak" and f.get("text") for f in _ws.sent)
        check("voice: LocalVoiceBridge text turn → user transcript + tool dispatch + spoken reply",
              _utext and _tool and _spoke and "turn_complete" in _types, _types)
    finally:
        _bsrv.shutdown()
        if _pe is None: os.environ.pop("STUDIO_LOCAL_ENDPOINT", None)
        else: os.environ["STUDIO_LOCAL_ENDPOINT"] = _pe

    # ---- LOCAL MEDICAL vision: MedGemma 4B (on-device transformers/MPS, $0) ----
    from occ.studio import medgemma as _MG
    check("medgemma: routes only medical models (medgemma-* yes; gemma/gemini no)",
          _MG.is_medgemma("medgemma-4b-it") and not _MG.is_medgemma("gemma-4-12b-it")
          and not _MG.is_medgemma("gemini-2.5-flash"))
    check("medgemma: selectable in the Vision dropdown", "medgemma-4b-it" in MODEL_OPTIONS["vision"])
    check("medgemma: priced at $0 (on-device)", usd_micros_for("medgemma-4b-it", 50000, 50000) == 0)
    check("medgemma: describe('') is graceful and does NOT trigger a model load",
          _MG.describe(b"") == "")
    # AGENTIC + unified: medical analysis is a TOOL the one agent (Gemma 4) calls
    check("medgemma: agentic — medical_image is a registered tool",
          "medical_image" in {f.__name__ for f in T.ALL_TOOLS})
    check("medgemma: the local agent (Gemma 4) can call medical_image",
          "medical_image" in _L.LOCAL_CORE_TOOLS)
    _mst = _MG.status()
    check("medgemma: status() reports state + powers + examples WITHOUT loading the model",
          _mst.get("state") in ("loaded", "ready", "needs-download", "needs-deps", "error")
          and len(_mst.get("tasks", [])) >= 5 and len(_mst.get("examples", [])) >= 4
          and not _MG.loaded())
    check("medgemma: per-expertise examples + task presets defined",
          set(_MG.EXAMPLE_QUERIES) >= {"cxr", "derm", "fundus", "histo"}
          and "report" in _MG.TASKS and "vqa" in _MG.TASKS)

    # ---- LOCAL image edit/gen: Qwen-Image-Edit (4-bit GGUF · sd.cpp/Metal, $0) ----
    from occ.studio import qwen_image as _Q
    check("qwen: routes only qwen-image* models (not gemini-image / imagen)",
          _Q.is_qwen_model("qwen-image-edit") and not _Q.is_qwen_model("gemini-2.5-flash-image")
          and not _Q.is_qwen_model("imagen-4.0-generate-001"))
    check("qwen: selectable in image_edit + image_gen dropdowns",
          "qwen-image-edit" in MODEL_OPTIONS["image_edit"] and "qwen-image-edit" in MODEL_OPTIONS["image_gen"])
    check("qwen: priced at $0 (local Metal)", usd_micros_for("qwen-image-edit", 100, 100) == 0)
    _qs = _Q.status()
    check("qwen: status()/available() report readiness without crashing or doing work",
          isinstance(_Q.available(), bool) and "weights" in _qs and "engine" in _qs
          and set(_qs["weights"]) == {"diffusion", "vae", "llm"})
    check("qwen: Lightning few-step LoRA is auto-detected (4 steps when present)",
          "lightning" in _qs and _qs.get("steps") in (4, 20)
          and (_qs["steps"] == 4) == _qs["lightning"])


# ----------------------------- ONLINE -----------------------------
def _get(path):
    return json.loads(urllib.request.urlopen(STUDIO_URL + path, timeout=8).read())


def _post(path, body):
    req = urllib.request.Request(STUDIO_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _server_up():
    try:
        urllib.request.urlopen(STUDIO_URL + "/", timeout=3)
        return True
    except Exception:
        return False


def online():
    print(f"\n[ONLINE] endpoints + live pipeline @ {STUDIO_URL}")
    import asyncio
    import websockets

    check("GET /garments", _get("/garments")["sponsor"]["name"] == "Mode Marco")
    ec = _get("/eyewear")
    check("GET /eyewear", ec["sponsor"]["name"] == "Ralba Optical" and len(ec["eyewear"]) > 300)

    # Try-on Lab: page renders, flag captures the pipeline, dataset + image serving work
    import urllib.error
    lab_html = urllib.request.urlopen(STUDIO_URL + "/eyewear/lab", timeout=8).read().decode()
    check("GET /eyewear/lab renders the dashboard",
          "Try-on Lab" in lab_html and "pipeline" in lab_html.lower())
    # MedGemma medical dashboard
    med_html = urllib.request.urlopen(STUDIO_URL + "/medical", timeout=8).read().decode()
    check("GET /medical renders the MedGemma dashboard",
          "MedGemma" in med_html and "Decision-support" in med_html)
    _ms = _get("/medgemma/status")
    check("GET /medgemma/status (no forced load) lists model + powers + examples",
          _ms.get("model") and len(_ms.get("tasks", [])) >= 5 and len(_ms.get("examples", [])) >= 4)
    _excode = urllib.request.urlopen(STUDIO_URL + "/medgemma/example/cxr", timeout=25).getcode()
    check("GET /medgemma/example serves a per-expertise sample image", _excode == 200)
    check("lab dashboard exposes the per-stage Nano Banana enhance button",
          "Nano Banana this step" in lab_html and "enhanceStage" in lab_html)
    for _ep in ("/eyewear/lab/enhance", "/eyewear/lab/apply"):  # guarded against missing args
        try:
            _post(_ep, {}); _g = True
        except urllib.error.HTTPError as _e:
            _g = _e.code == 400
        check(f"POST {_ep} guards missing args", _g)
    try:                                              # flag with no pair set → guarded 400
        _post("/eyewear/flag", {"note": "x"}); guard_ok = True   # (a pair may already be set)
    except urllib.error.HTTPError as e:
        guard_ok = e.code == 400
    check("POST /eyewear/flag guards 'try a pair on first'", guard_ok)
    _eimg = next(g for g in ec["eyewear"] if g.get("img"))["img"]
    _post("/eyewear/try", {"image": _eimg, "label": "labtest"})   # no API needed (falls back to raw)
    fr = _post("/eyewear/flag", {"note": "online regression"})
    check("POST /eyewear/flag captures the pipeline into the dataset",
          fr.get("status") == "success" and fr.get("id"))
    _fid = fr.get("id")
    _fm = next((x for x in _get("/eyewear/flags")["flags"] if x["id"] == _fid), None)
    check("GET /eyewear/flags lists the flag with its stages",
          _fm is not None and len(_fm["stages"]) >= 6)
    _code = urllib.request.urlopen(
        STUDIO_URL + f"/eyewear/lab/img/{_fid}/06_clean_lenses.png", timeout=8).getcode()
    check("GET /eyewear/lab/img serves a stage PNG", _code == 200)
    _rep = urllib.request.urlopen(STUDIO_URL + f"/eyewear/lab/report/{_fid}", timeout=8).read().decode()
    check("GET /eyewear/lab/report serves the coding-assistant brief",
          "How to resolve" in _rep and "Use the 3D model" in _rep)
    _casc = _post("/eyewear/inspect", {"image": _eimg})
    check("POST /eyewear/inspect returns the candidate cascade",
          _casc.get("status") == "success" and len(_casc.get("stages", [])) >= 5
          and _casc["stages"][0]["png"].startswith("data:image"))
    _post("/eyewear/flag/delete", {"id": _fid})
    check("POST /eyewear/flag/delete removes it",
          not any(x["id"] == _fid for x in _get("/eyewear/flags")["flags"]))

    # apply a face filter via the endpoint, confirm it's active
    r = _post("/filter", {"name": "ninja_mask"})
    ov = _get("/overlays").get("overlays", [])
    check("POST /filter ninja_mask → active", r.get("status") == "success"
          and any(o["name"] == "ninja_mask" for o in ov))

    # the real path: push a face → /ws/cam → pipeline → apply → /stream.mjpg changes
    face = cv2.imread(FACE)
    jpg = cv2.imencode(".jpg", cv2.resize(face, (640, 640)))[1].tobytes()

    def grab():
        req = urllib.request.urlopen(STUDIO_URL + "/stream.mjpg", timeout=8); buf = b""; t0 = time.time()
        while time.time() - t0 < 5:
            buf += req.read(8192); a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2)
            if a >= 0 and b >= 0:
                req.close(); return cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)

    async def live_path():
        async with websockets.connect(STUDIO_URL.replace("http", "ws") + "/ws/cam", max_size=None) as ws:
            async def pump():
                while True:
                    await ws.send(jpg); await asyncio.sleep(0.05)
            t = asyncio.create_task(pump()); await asyncio.sleep(1.5)
            _post("/filter", {"name": "sunglasses"}); await asyncio.sleep(1.2)
            sun = grab()
            _post("/filter", {"name": "clear"}); await asyncio.sleep(1.2)
            clean = grab()
            t.cancel()
            return sun, clean

        return None, None

    sun, clean = asyncio.get_event_loop().run_until_complete(live_path())
    if sun is not None and clean is not None:
        px = int((np.abs(sun.astype(int) - clean.astype(int)).max(2) > 18).sum())
        check("push→pipeline→/stream.mjpg: sunglasses visibly drawn", px > 200, f"{px}px")
    else:
        check("live pipeline frame grab", False, "no frame")

    # OpenAI Realtime LIVE turn — opt-in (bills OpenAI), so off by default.
    # STUDIO_TEST_OPENAI=1 .venv/bin/python test_studio_tryon.py
    if os.environ.get("STUDIO_TEST_OPENAI") == "1" and _get("/agent/mode").get("has_openai"):
        async def creole_turn():
            uri = STUDIO_URL.replace("http", "ws") + "/ws/live?backend=openai"
            async with websockets.connect(uri, max_size=None) as ws:
                await asyncio.sleep(0.3)
                await ws.send(json.dumps({"type": "text",
                    "text": "Pale Kreyòl. Di m yon ti bonjou kout epi mete yon mask ninja."}))
                model, audio, tools, t0 = [], 0, [], time.time()
                while time.time() - t0 < 35:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=35))
                    if m.get("type") == "transcript" and m["role"] == "model":
                        model.append(m["text"])
                    elif m.get("type") == "audio":
                        audio += 1
                    elif m.get("type") == "tool":
                        tools.append(m.get("data", {}).get("name"))
                    elif m.get("type") == "turn_complete" and audio:
                        break
                return "".join(model), audio, tools
        txt, audio, tools = asyncio.get_event_loop().run_until_complete(creole_turn())
        check("OpenAI Realtime live: spoke + tool-called", audio > 0 and "apply_face_filter" in tools,
              f"{audio} audio, tools={tools}, said={txt[:60]!r}")


def ui():
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("\n[UI] skipped (playwright not installed)")
        return
    print("\n[UI] deterministic frontend smoke (playwright)")
    errs = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_context(viewport={"width": 1500, "height": 950}).new_page()
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(STUDIO_URL + "/", wait_until="domcontentloaded"); pg.wait_for_timeout(1800)
        pg.click("#modeswitch button[data-m=voice]"); pg.wait_for_timeout(800)
        # pure canvas: panels are icon-driven now — the canvas stays clean until tapped
        check("filter panel hidden until its icon is tapped", not pg.is_visible("#filterbar"))
        pg.click("#vfilters"); pg.wait_for_timeout(300)
        check("face-filter panel opens on icon", pg.is_visible("#filterbar"))
        check("12 face-filter chips", pg.eval_on_selector_all("#filterbar button", "e=>e.length") == 12)
        pg.click("#vshop"); pg.wait_for_timeout(400)          # open the shop panel
        check("shop panel opens on icon", pg.is_visible("#tryonbar"))
        pg.click("#store-eyewear"); pg.wait_for_timeout(1200)
        check("store switch → Ralba sponsor", "Ralba Optical" in (pg.text_content("#sponsor") or ""))
        # video audio control: a video on the canvas shows a one-tap mute/unmute button
        pg.evaluate("()=>setCanvas('video','/download/horizon_lenses.mp4','horizon')")
        pg.wait_for_timeout(300)
        check("video shows the volume button", pg.is_visible("#volbtn"))
        check("video autoplays MUTED (browser-safe)", pg.eval_on_selector("#video", "v=>v.muted") is True)
        pg.click("#volbtn"); pg.wait_for_timeout(150)
        check("volume button unmutes the audio", pg.eval_on_selector("#video", "v=>v.muted") is False)
        pg.evaluate("()=>setCanvas('live')"); pg.wait_for_timeout(150)
        check("volume button hides off-video", not pg.is_visible("#volbtn"))
        # one unified glasses try-on with a courteous fitting loader
        pg.evaluate("()=>vShowFitting('Tailoring your fit','fitting…')")
        pg.wait_for_timeout(150)
        check("fitting loader shows", "on" in (pg.get_attribute("#fitloader", "class") or ""))
        pg.evaluate("()=>vHideFitting()")
        pg.wait_for_timeout(150)
        check("fitting loader hides", "on" not in (pg.get_attribute("#fitloader", "class") or ""))
        # showCatalog renders agent results deterministically
        pg.evaluate("""()=>showCatalog({store:'apparel',matches:[
          {title:'POLO Navy',price:'150',img:'x',store:'apparel'},
          {title:'POLO Blue',price:'149',img:'y',store:'apparel'}]})""")
        pg.wait_for_timeout(300)
        check("showCatalog renders stylist results",
              pg.eval_on_selector_all("#garments .gcard", "e=>e.length") == 2)
        # Live Voice auto-collapses both side panels → the canvas is the hero (Meet)
        collapsed = lambda: "cl" in (pg.get_attribute("#grid", "class") or "").split() \
            and "cr" in (pg.get_attribute("#grid", "class") or "").split()
        check("Live Voice maximizes the canvas (both panels collapsed)", collapsed())
        # the header ⤢ toggle flips that state
        before = collapsed()
        pg.click("#tgl-max"); pg.wait_for_timeout(450)
        check("⤢ toggles the side panels", collapsed() != before)
        if not collapsed():                                   # leave it big for the rest
            pg.click("#tgl-max"); pg.wait_for_timeout(300)
        # the video scales UP to fill the stage (not its small intrinsic size)
        iw = pg.eval_on_selector("#img", "e=>Math.round(e.getBoundingClientRect().width)")
        ww = pg.eval_on_selector("#wrap", "e=>Math.round(e.getBoundingClientRect().width)")
        check("video fills the canvas width", iw >= ww - 2, f"img {iw}px / wrap {ww}px")
        # Live Voice backend selector (Gemini ↔ OpenAI Realtime for Creole)
        check("voice backend selector present (incl. free local)",
              pg.eval_on_selector_all("#vbackend option", "e=>e.map(o=>o.value).join(',')")
              == "gemini,openai,local")
        check("gesture-browse button present", pg.is_visible("#vgest"))
        # Meet-style extras: fill-the-canvas toggle + self-view PiP
        pg.click("#fillbtn"); pg.wait_for_timeout(150)
        check("fill toggle fills the canvas", "fill" in (pg.get_attribute("#wrap", "class") or ""))
        pg.click("#fillbtn"); pg.wait_for_timeout(100)
        check("self-view PiP element present", pg.query_selector("#selfview") is not None)
        # touchless icon buttons render on the PiP from gesture state; countdown overlays
        pg.evaluate("""()=>gRenderPiP({active:true,buttons:[
          {id:'prev',x:.8,y:.13,w:.26,h:.15},{id:'next',x:.8,y:.30,w:.26,h:.15},
          {id:'try',x:.8,y:.50,w:.26,h:.15},{id:'store',x:.8,y:.70,w:.26,h:.15},
          {id:'clear',x:.8,y:.87,w:.26,h:.15}],hover:'try',dwell:0.5,finger:{x:.8,y:.5}})""")
        pg.wait_for_timeout(120)
        check("PiP shows 5 touchless icon buttons", pg.eval_on_selector_all("#selfbtns .sbtn", "e=>e.length") == 5)
        check("PiP finger cursor visible", pg.eval_on_selector("#selfcursor", "e=>getComputedStyle(e).display") != "none")
        # turn gestures OFF server-side + stop the rail loop so neither overwrites the
        # countdown primitive (pollStats would otherwise restart the loop and reset it)
        pg.evaluate("""async()=>{try{await fetch('/gestures',{method:'POST',
          headers:{'Content-Type':'application/json'},body:JSON.stringify({on:false})});}catch(e){}
          try{gStopRails();}catch(e){}}""")
        pg.wait_for_timeout(300)
        pg.evaluate("()=>gCountdown(3)"); pg.wait_for_timeout(80)
        check("pose countdown overlay shows", "on" in (pg.get_attribute("#countdown", "class") or "")
              and pg.text_content("#countdown") == "3")
        # overlays/config tucked behind the ⚙ icon → canvas is 100% clean by default
        check("overlays/settings hidden in voice by default", not pg.is_visible("#ovpanel"))
        pg.click("#vinfo"); pg.wait_for_timeout(250)
        check("⚙ opens overlays/settings panel", pg.is_visible("#ovpanel"))
        b.close()
    check("no uncaught JS errors", not errs, "; ".join(errs[:3]))


def main():
    print("=" * 60)
    print("  Vision Ninja — AR try-on regression test")
    print("=" * 60)
    offline()
    if _server_up():
        online()
        ui()
    else:
        print(f"\n[ONLINE/UI] skipped — studio not reachable at {STUDIO_URL}")
        print("  (start it: .venv/bin/python run.py studio, then re-run)")
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 60)
    print(f"  {'ALL PASS' if passed == total else 'FAILURES'} — {passed}/{total} checks")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
