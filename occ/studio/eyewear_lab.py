"""Try-on Lab: a dev dashboard + bad-case dataset for the eyewear virtual try-on.

A "flag" captures the live try-on (the glasses on your face) together with EVERY
intermediate artifact of the full pipeline — raw product photo → Nano Banana Pro
canonical render → rembg/U²-Net segmentation → background knockout → remove_arms →
lens isolation+classification → clean_lenses reconstruction → registration centres →
face landmarks → 2-point registration → final composite. Each stage is produced by the
SAME functions the live pipeline uses (no re-implementation), so the dashboard can't lie.

Flags live under out/studio/tryon_flags/<id>/ (gitignored) as PNG stages + meta.json —
a growing corpus of bad try-ons to study and fix. Served by /eyewear/lab.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

FLAGS_DIR = "out/studio/tryon_flags"

# The pipeline, frame by frame. (key, title, one-line description.) Stats are attached
# per-flag at capture time. 00–07 are the ASSET pipeline (run once per product, cached);
# 08–10 are the LIVE per-frame registration onto your face.
STAGES = [
    ("00_raw", "Raw product photo",
     "The original Ralba catalog image, exactly as shot on the store's background."),
    ("01_canonical", "Nano Banana Pro — canonical front-on",
     "gemini-3-pro-image renders a clean, symmetric, arms-removed front-on view on pure "
     "white. Cached per product, so it's $0 after the first try-on."),
    ("02_segmentation", "Segmentation matte (rembg · U²-Net / ISNet)",
     "The matting model's alpha — glasses vs. background. Classical flood-fill is the "
     "fallback when rembg is unavailable."),
    ("03_cutout", "Background knockout (RGBA)",
     "load_eyewear_rgba: background removed and tight-cropped to the frame."),
    ("04_armless", "remove_arms",
     "Temple arms and hinges dropped — just the two lens rims and the nose bridge."),
    ("05_lens_regions", "Lens isolation + classification",
     "_lens_regions finds each lens (colour-aware + white-frame guard + disc bound); each "
     "is classified clear vs. tinted. THIS is where most bugs originate."),
    ("06_clean_lenses", "Lens reconstruction (clean_lenses)",
     "Both lenses repainted as ONE uniform surface — clear → see-through glass, tinted → "
     "one flat colour. No baked-in photo reflections survive."),
    ("07_centers", "Registration centres (lens_centers_norm)",
     "The two geometric lens centres that get mapped onto your pupils."),
    ("08_landmarks", "Face landmarks (MediaPipe FaceLandmarker)",
     "478-point mesh + iris. The pupils and head roll drive the registration."),
    ("09_registration", "Frame registration (2-pt similarity)",
     "_eyewear_quad solves scale+roll+translation mapping the asset's lens centres → your "
     "pupils, then transforms the asset corners."),
    ("10_final", "Final composite",
     "The warped, reconstructed glasses on your face — exactly what you saw live."),
]
STAGE_TITLES = {k: t for k, t, _ in STAGES}
STAGE_DESCS = {k: d for k, _, d in STAGES}

# Which function/file owns each stage — so a coding assistant knows exactly where to look.
STAGE_FN = {
    "00_raw": ("—", "the Ralba catalog image"),
    "01_canonical": ("_canonical_eyewear() + _CANON_PROMPT/_CORRECT_PROMPT", "occ/studio/tools.py"),
    "02_segmentation": ("_rembg_alpha() / _knockout_bg() fallback", "occ/studio/face_filters.py"),
    "03_cutout": ("load_eyewear_rgba() (+ _keep_glasses)", "occ/studio/face_filters.py"),
    "04_armless": ("remove_arms()", "occ/studio/face_filters.py"),
    "05_lens_regions": ("_lens_regions() + lens_classification()", "occ/studio/face_filters.py"),
    "06_clean_lenses": ("clean_lenses()", "occ/studio/face_filters.py"),
    "07_centers": ("lens_centers_norm()", "occ/studio/face_filters.py"),
    "08_landmarks": ("detect_faces() / FaceMeshEngine", "occ/studio/facemesh.py"),
    "09_registration": ("_eyewear_quad()", "occ/studio/face_filters.py"),
    "10_final": ("filter_eyewear()", "occ/studio/face_filters.py"),
}

# Symptom → first stage(s) to suspect. The golden rule: find the EARLIEST stage that already
# looks wrong; everything downstream inherits it.
SYMPTOM_GUIDE = [
    ("Lens too dark / opaque / wrong colour / not see-through",
     ["05_lens_regions", "06_clean_lenses"],
     "Lens TYPE mis-classified or the reconstruction fill is off. Check `clear_frac`/`frac_hole` "
     "in stage 05 and the `decision`+`fill_bgr` — clear should reconstruct to a uniform low-alpha "
     "glass; a baked photo reflection must NOT survive."),
    ("Two lenses look different from each other",
     ["05_lens_regions", "06_clean_lenses"],
     "Both lenses must share ONE appearance. clean_lenses unifies the pair — check the per-lens "
     "votes; an asymmetric photo can split the vote."),
    ("A coloured / white rectangle or halo OUTSIDE the lenses",
     ["02_segmentation", "03_cutout", "05_lens_regions"],
     "Background not fully knocked out (rembg matte or _knockout_bg) or a lens region bled past "
     "the rim. Check stage 03 over the checkerboard and the white-frame disc bound in _lens_regions."),
    ("White acetate frame disappears / turns see-through",
     ["05_lens_regions", "06_clean_lenses"],
     "White frame read as lens material. The white-perimeter guard in _lens_regions and the "
     "safety-net `white_frame` mask in clean_lenses must protect it."),
    ("Frame wrong size (too big/small) or shifted left/right",
     ["07_centers", "09_registration"],
     "Lens-centre detection or the 2-point registration. Check stage 07 centres (should be "
     "symmetric ~0.25/0.75) and stage 09 `width_over_ipd` (clamped 1.9–3.4×)."),
    ("Frame tilted / not following head pose",
     ["08_landmarks", "09_registration"],
     "Roll/pupils from MediaPipe drive the rotation. Check stage 08 `roll_deg`/`ipd_px` and the "
     "angle in _eyewear_quad."),
    ("Temple arms / hinges look fake or are visible head-on",
     ["01_canonical", "04_armless"],
     "Arms should be gone by stage 04. _TEMPLE_ARMS is OFF by default; the canonical prompt also "
     "removes them. Check remove_arms output."),
    ("Glasses don't appear at all / not tracked",
     ["08_landmarks", "10_final"],
     "No face detected, or the overlay isn't active. Check stage 08 `faces` and that 'eyewear' is "
     "in the overlay stack."),
]


# ---------------------------------------------------------------- visual helpers ----
# What makes a product photo that the pipeline nails. Shown to the human (report + dashboard).
IMAGE_GUIDE = [
    "Front-on, straight from the front — not angled, tilted, or 3/4 view.",
    "Both lenses fully visible and equal; the frame roughly horizontal.",
    "Temple arms folded back or cropped — we only need the lens rims + bridge.",
    "Plain, seamless background (pure white is ideal). No hands, faces, props, or shadows.",
    "Lenses clean and unobstructed — no big studio reflection or glare baked onto the glass.",
    "Sharp and reasonably high-res (the frame at least ~500 px wide).",
    "The frame fills most of the photo and is centred.",
]


def assess_source_image(bgr):
    """Heuristic quality check on the RAW product photo → a list of {issue, advice} the
    human can act on. Catches the common reasons a try-on comes out wrong at the source."""
    issues = []
    if bgr is None:
        return [{"issue": "image could not be read", "advice": "re-upload a JPG or PNG."}]
    h, w = bgr.shape[:2]
    if min(h, w) < 400:
        issues.append({"issue": f"low resolution ({w}×{h})",
                       "advice": "upload a larger image — aim for the frame ≥ 500 px wide."})
    # background: sample the 4 borders — a clean product shot has a plain, bright border
    b = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]]).reshape(-1, 3)
    bright = float((b.min(1) > 180).mean())
    busy = float(b.std(0).mean())
    if bright < 0.6:
        issues.append({"issue": "busy / dark background",
                       "advice": "shoot on a plain white seamless background — no props, hands, or scenery."})
    elif busy > 38:
        issues.append({"issue": "textured / uneven background",
                       "advice": "use a flat, evenly-lit white backdrop so the cutout is clean."})
    # aspect: very tall/wide crops usually mean an angled shot or extra junk
    ar = w / max(h, 1)
    if ar < 1.2:
        issues.append({"issue": f"not a wide front-on crop (aspect {ar:.1f})",
                       "advice": "use a straight front-on shot where the frame is clearly wider than tall."})
    return issues


def _checker(h, w, sq=22):
    ck = ((np.indices((h, w)).sum(0) // sq) % 2)
    return (np.where(ck[..., None], 205, 120) * np.ones((1, 1, 3), np.uint8)).astype(np.uint8)


def _over_checker(rgba):
    """Composite an RGBA cutout over a checkerboard so transparency is visible."""
    if rgba is None:
        return None
    h, w = rgba.shape[:2]
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        a = rgba[:, :, 3:4].astype(np.float32) / 255.0
        return (rgba[:, :, :3] * a + _checker(h, w) * (1 - a)).astype(np.uint8)
    return rgba[:, :, :3].copy()


def _matte(alpha):
    """Show a single-channel alpha matte as a legible grayscale image."""
    return cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)


def _fit(img, long_side=520):
    """Downscale large stage images so the dashboard stays light."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


# ----------------------------------------------------------------- asset pipeline ----
def introspect_asset(src: str):
    """Re-run the FULL asset pipeline for a product image, capturing every stage as a
    BGR image. Returns (images: {stage_key: bgr}, stats: {stage_key: dict}). Uses the
    real functions; the canonical render is cache-only ($0 if the pair was tried on)."""
    from . import tools
    from .face_filters import (load_eyewear_rgba, remove_arms, clean_lenses,
                               _lens_regions, lens_centers_norm, lens_classification,
                               _rembg_alpha, _REMBG_ON)
    imgs, stats, t = {}, {}, {}

    def stamp(key, t0):
        t[key] = round((time.time() - t0) * 1000)

    raw = tools._fetch_bytes(src)
    if not raw:
        return imgs, {"error": "could not fetch product image"}
    t0 = time.time()
    raw_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    imgs["00_raw"] = raw_bgr
    stamp("00_raw", t0)

    t0 = time.time()
    canon = tools._canonical_eyewear(raw, src)        # cached per URL; None if no key + uncached
    if canon:
        imgs["01_canonical"] = cv2.imdecode(np.frombuffer(canon, np.uint8), cv2.IMREAD_COLOR)
        stats["01_canonical"] = {"source": "Nano Banana Pro (cached)"}
    else:
        stats["01_canonical"] = {"source": "not cached / no API key — using raw image"}
    stamp("01_canonical", t0)

    base = canon or raw
    base_bgr = cv2.imdecode(np.frombuffer(base, np.uint8), cv2.IMREAD_COLOR)

    t0 = time.time()
    alpha = _rembg_alpha(base_bgr) if _REMBG_ON else None
    if alpha is not None:
        imgs["02_segmentation"] = _matte(alpha)
        stats["02_segmentation"] = {"model": os.environ.get("STUDIO_REMBG_MODEL",
                                                             "isnet-general-use"),
                                    "coverage_%": round(100 * float((alpha > 40).mean()), 1)}
    else:
        stats["02_segmentation"] = {"model": "classical flood-fill (rembg off/failed)"}
    stamp("02_segmentation", t0)

    t0 = time.time()
    cutout = load_eyewear_rgba(base)
    imgs["03_cutout"] = _over_checker(cutout)
    stamp("03_cutout", t0)

    t0 = time.time()
    armless = remove_arms(cutout) if cutout is not None else None
    imgs["04_armless"] = _over_checker(armless)
    stamp("04_armless", t0)

    if armless is not None:
        t0 = time.time()
        cls = lens_classification(armless)
        regions, _ = _lens_regions(armless)
        viz = _over_checker(armless).copy()
        cols = [(80, 220, 80), (80, 160, 255)]
        for i, (interior, c) in enumerate(regions):
            cnts, _ = cv2.findContours(interior.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz, cnts, -1, cols[i % 2], 2)
            cv2.circle(viz, (int(c[0]), int(c[1])), 4, (40, 40, 240), -1)
            lab = cls["lenses"][i]
            tag = "CLEAR" if lab["is_clear"] else "TINT"
            cv2.putText(viz, tag, (int(c[0]) - 26, int(c[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cols[i % 2], 2, cv2.LINE_AA)
        imgs["05_lens_regions"] = viz
        stats["05_lens_regions"] = cls
        stamp("05_lens_regions", t0)

        t0 = time.time()
        clean = clean_lenses(armless)
        imgs["06_clean_lenses"] = _over_checker(clean)
        # measure the see-through-ness of the reconstructed lenses
        lm = np.zeros(armless.shape[:2], bool)
        for interior, _c in regions:
            lm |= interior
        if lm.any():
            stats["06_clean_lenses"] = {"lens_mean_alpha": int(clean[:, :, 3][lm].mean()),
                                        "within_lens_std": int(clean[:, :, :3][lm].reshape(-1, 3).std(0).mean())}
        stamp("06_clean_lenses", t0)

        t0 = time.time()
        centers = lens_centers_norm(armless)
        cviz = _over_checker(armless).copy()
        h, w = armless.shape[:2]
        if centers:
            for (cx, cy) in centers:
                cv2.circle(cviz, (int(cx * w), int(cy * h)), 8, (40, 220, 240), 2)
                cv2.drawMarker(cviz, (int(cx * w), int(cy * h)), (40, 220, 240),
                               cv2.MARKER_CROSS, 18, 2)
            stats["07_centers"] = {"left": [round(centers[0][0], 3), round(centers[0][1], 3)],
                                   "right": [round(centers[1][0], 3), round(centers[1][1], 3)]}
        else:
            stats["07_centers"] = {"note": "geometric detection failed → symmetric fallback"}
        imgs["07_centers"] = cviz
        stamp("07_centers", t0)

    stats["_timing_ms"] = t
    return imgs, stats


# ------------------------------------------------------------------ live pipeline ----
def introspect_live(clean_bgr, vis_bgr, src: str):
    """Capture the LIVE registration stages on the user's face: landmarks, the registration
    quad, and the final composite (the annotated frame they saw)."""
    from .facemesh import detect_faces, IDX, FACE_OVAL
    from .face_filters import _eyewear_quad, _EYEWEAR
    imgs, stats = {}, {}
    if vis_bgr is not None:
        imgs["10_final"] = vis_bgr
    if clean_bgr is None:
        stats["08_landmarks"] = {"note": "no clean frame captured"}
        return imgs, stats
    faces = detect_faces(clean_bgr)
    if not faces:
        stats["08_landmarks"] = {"note": "no face detected in the captured frame"}
        return imgs, stats
    f = faces[0]
    lmviz = clean_bgr.copy()
    for (x, y) in f.lm.astype(int):
        cv2.circle(lmviz, (int(x), int(y)), 1, (90, 240, 90), -1)
    cv2.polylines(lmviz, [f.lm[FACE_OVAL].astype(np.int32)], True, (240, 200, 90), 1, cv2.LINE_AA)
    for nm, col in (("iris_l", (40, 40, 240)), ("iris_r", (40, 40, 240))):
        if len(f.lm) > IDX[nm]:
            cv2.circle(lmviz, tuple(f.lm[IDX[nm]].astype(int)), 4, col, -1)
    imgs["08_landmarks"] = lmviz
    stats["08_landmarks"] = {"faces": len(faces), "ipd_px": round(f.eye_dist, 1),
                             "roll_deg": round(f.roll, 1),
                             "yaw": round(float(f.yaw), 2), "pitch": round(float(f.pitch), 2)}

    armless = _EYEWEAR.get("armless")
    centers = _EYEWEAR.get("lens_centers")
    if armless is not None:
        quad = _eyewear_quad(f, armless, lens_centers=centers)
        qv = clean_bgr.copy()
        pts = np.array(quad, np.int32)
        cv2.polylines(qv, [pts], True, (40, 220, 240), 2, cv2.LINE_AA)
        cv2.circle(qv, tuple(np.array(f.eye_l, int)), 4, (40, 40, 240), -1)
        cv2.circle(qv, tuple(np.array(f.eye_r, int)), 4, (40, 40, 240), -1)
        width = float(np.linalg.norm(pts[1] - pts[0]))
        imgs["09_registration"] = qv
        stats["09_registration"] = {"frame_width_px": round(width),
                                    "width_over_ipd": round(width / max(f.eye_dist, 1e-6), 2)}
    return imgs, stats


# ----------------------------------------------- candidate cascade ("what if?") ----
def inspect_candidate(src: str):
    """Run the ASSET pipeline on a CANDIDATE image (new upload / URL) WITHOUT saving, and
    return each stage as a base64 PNG + stats + source-image issues. Lets the human see the
    CASCADE of a different image through the whole pipeline before committing to it."""
    import base64
    imgs, stats = introspect_asset(src)
    out = []
    for key, title, desc in STAGES:
        im = imgs.get(key)
        if im is None:
            continue
        ok, buf = cv2.imencode(".png", _fit(im))
        if ok:
            out.append({"key": key, "title": title, "desc": desc,
                        "png": "data:image/png;base64," + base64.b64encode(buf).decode(),
                        "stats": stats.get(key) or {}})
    return {"stages": out, "source_issues": assess_source_image(imgs.get("00_raw")),
            "error": stats.get("error")}


def regenerate_canonical(src: str):
    """FIX THE IMAGE: force a fresh Nano Banana Pro canonical render for `src` (drop the cached
    one) so a messy/angled photo gets re-normalized. Returns True if a new canonical was made.
    Requires an API key (returns False otherwise)."""
    from . import tools
    import hashlib
    raw = tools._fetch_bytes(src)
    if not raw:
        return False
    key = hashlib.md5(((src or "") + "|" + tools._CANON_VER).encode()).hexdigest()
    cache = os.path.join(tools._CANON_DIR, key + ".png")
    try:
        if os.path.exists(cache):
            os.remove(cache)
    except Exception:
        pass
    return bool(tools._canonical_eyewear(raw, src))


# ---------------------------------------------------------- coding-assistant report ----
def _md_table(stats: dict) -> str:
    if not stats:
        return ""
    rows = []
    for k, v in stats.items():
        if k == "lenses":
            for i, l in enumerate(v):
                rows.append(f"| lens {i + 1} | {'CLEAR' if l['is_clear'] else 'TINT'} · "
                            f"clear_frac={l['clear_frac']} · frac_hole={l['frac_hole']} |")
            continue
        rows.append(f"| {k} | {json.dumps(v) if isinstance(v, (dict, list)) else v} |")
    return "| field | value |\n|---|---|\n" + "\n".join(rows) + "\n"


def _report_md(meta: dict, source_issues: list, has_3d: bool) -> str:
    """A self-contained brief a CODING ASSISTANT can act on, plus the human's options to
    upload a better image / fix the image / use the 3D model. All image paths are relative
    to this file's folder, so it renders anywhere."""
    L = []
    L.append(f"# 🚩 Try-on bug report — {meta.get('title') or meta['src'].rsplit('/', 1)[-1]}\n")
    L.append(f"- **Flagged:** {meta['ts']}")
    if meta.get("brand"):
        L.append(f"- **Brand:** {meta['brand']}")
    L.append(f"- **Source image:** {meta['src']}")
    L.append(f"- **3D model available:** {'yes (`ar: true`)' if has_3d else 'no'}\n")
    L.append("## What the human says is wrong\n")
    L.append(f"> {meta.get('note') or '(no note given)'}\n")
    # the result, side by side
    keys = {s["key"]: s for s in meta["stages"]}
    raw = keys.get("00_raw"); fin = keys.get("10_final") or keys.get("06_clean_lenses")
    if raw and fin:
        L.append("## Result\n")
        L.append(f"| Original product | On the face |\n|---|---|\n"
                 f"| ![raw]({raw['file']}) | ![final]({fin['file']}) |\n")
    # symptom router
    L.append("## Where to look — symptom → stage → function\n")
    L.append("> Golden rule: open the stage images below and find the **earliest** one that already "
             "looks wrong. Everything after it inherits the problem.\n")
    L.append("| If the symptom is… | suspect stage(s) | what to check |\n|---|---|---|")
    for sym, stages_k, why in SYMPTOM_GUIDE:
        labels = ", ".join(f"`{k}`" for k in stages_k)
        L.append(f"| {sym} | {labels} | {why} |")
    L.append("")
    # full pipeline
    L.append("## The pipeline, stage by stage\n")
    for s in meta["stages"]:
        fn, where = STAGE_FN.get(s["key"], ("—", "—"))
        L.append(f"### {s['key'].replace('_', ' ')} — {s['title']}")
        L.append(f"{s['desc']}\n")
        L.append(f"![{s['key']}]({s['file']})\n")
        L.append(f"**Code:** `{fn}` in `{where}`" + (f"  ·  {s['ms']} ms" if s.get("ms") else ""))
        tbl = _md_table(s.get("stats") or {})
        if tbl:
            L.append("\n" + tbl)
        L.append("")
    # resolution paths
    L.append("## How to resolve — three paths\n")
    L.append("### 1. Upload a better source image (fastest)\n")
    if source_issues:
        L.append("Auto-detected issues with THIS photo:")
        for it in source_issues:
            L.append(f"- **{it['issue']}** → {it['advice']}")
        L.append("")
    L.append("A good product photo for this pipeline:")
    for g in IMAGE_GUIDE:
        L.append(f"- {g}")
    L.append("\nIn the studio: open the 🛍 try-on bar → 📎 upload (or paste a URL) → try it on → "
             "flag again to compare.\n")
    L.append("### 2. Fix the image automatically (Nano Banana Pro)\n")
    L.append("Re-render the canonical front-on asset (stage 01) — this often cleans up an angled "
             "or messy photo without a new upload:\n")
    L.append("- In the Lab, click **🪄 Fix image** (forces a fresh `_canonical_eyewear` render and "
             "re-applies it live), **or**")
    L.append("- bump `_CANON_VER` in `occ/studio/tools.py` to invalidate the cache and re-try, **or**")
    L.append("- strengthen `_CANON_PROMPT` / `_CORRECT_PROMPT` for the failing trait.\n")
    L.append("### 3. Use the 3D model\n")
    if has_3d:
        L.append("This frame **has a 3D `.glb` model** (`ar: true`). For frames the 2D warp can't "
                 "nail (extreme wrap, thick 3D bevels), the glb is the robust path:\n")
        L.append("- Refresh the catalog with the glb URLs via `scripts/fetch_ralba.py` "
                 "(`product_tryon_3d_image[].pro_3d_image`), then render the glb posed by the head "
                 "matrix (`Face.matrix` / yaw·pitch·roll in `occ/studio/facemesh.py`).")
    else:
        L.append("No 3D model is available for this frame (`ar: false`) — use path 1 or 2.")
    L.append("\n---\n")
    L.append("### Reproduce offline\n```python\nfrom occ.studio import eyewear_lab as lab\n"
             f"imgs, stats = lab.introspect_asset({meta['src']!r})\n"
             "# imgs[stage] = BGR np.array; stats[stage] = the numbers in this report\n```\n")
    return "\n".join(L)


# ----------------------------------------------------------------------- dataset ----
def _slug(s):
    s = "".join(c if c.isalnum() else "-" for c in (s or "frame").lower())
    return "-".join(p for p in s.split("-") if p)[:40] or "frame"


def save_flag(src: str, title="", brand="", note="", clean_bgr=None, vis_bgr=None, has_3d=False):
    """Capture a flagged try-on: run both pipelines, save every stage + meta.json + a
    REPORT.md (coding-assistant brief + human remediation) under a new dataset folder.
    Returns the flag id (or raises ValueError if no src)."""
    if not src:
        raise ValueError("no eyewear source — try a pair on first")
    fid = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + _slug(title or src.rsplit("/", 1)[-1])
    d = os.path.join(FLAGS_DIR, fid)
    os.makedirs(d, exist_ok=True)
    asset_imgs, asset_stats = introspect_asset(src)
    live_imgs, live_stats = introspect_live(clean_bgr, vis_bgr, src)
    all_imgs = {**asset_imgs, **live_imgs}
    all_stats = {**{k: v for k, v in asset_stats.items() if not k.startswith("_")}, **live_stats}
    source_issues = assess_source_image(asset_imgs.get("00_raw"))

    stage_meta = []
    for key, title_s, desc in STAGES:
        img = all_imgs.get(key)
        if img is None:
            continue
        fn = key + ".png"
        cv2.imwrite(os.path.join(d, fn), _fit(img))
        stage_meta.append({"key": key, "title": title_s, "desc": desc, "file": fn,
                           "stats": all_stats.get(key, {}),
                           "ms": asset_stats.get("_timing_ms", {}).get(key)})
    if clean_bgr is not None:
        cv2.imwrite(os.path.join(d, "face_clean.png"), _fit(clean_bgr, 720))

    meta = {"id": fid, "ts": datetime.now().isoformat(timespec="seconds"),
            "src": src, "title": title, "brand": brand, "note": note, "has_3d": bool(has_3d),
            "source_issues": source_issues, "report": "REPORT.md",
            "thumb": "10_final.png" if "10_final" in all_imgs else "06_clean_lenses.png",
            "stages": stage_meta}
    with open(os.path.join(d, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    with open(os.path.join(d, "REPORT.md"), "w") as fp:
        fp.write(_report_md(meta, source_issues, bool(has_3d)))
    return fid


def list_flags():
    """All flags, newest first (metadata only)."""
    out = []
    if not os.path.isdir(FLAGS_DIR):
        return out
    for fid in sorted(os.listdir(FLAGS_DIR), reverse=True):
        mp = os.path.join(FLAGS_DIR, fid, "meta.json")
        if os.path.exists(mp):
            try:
                out.append(json.load(open(mp)))
            except Exception:
                pass
    return out


def flag_image_path(fid: str, fn: str):
    """Safe path to a stage image inside a flag folder (no traversal)."""
    if "/" in fid or ".." in fid or "/" in fn or ".." in fn:
        return None
    p = os.path.join(FLAGS_DIR, fid, fn)
    return p if os.path.exists(p) else None


def delete_flag(fid: str):
    import shutil
    if "/" in fid or ".." in fid:
        return False
    d = os.path.join(FLAGS_DIR, fid)
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False
