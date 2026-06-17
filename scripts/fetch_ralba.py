#!/usr/bin/env python
"""Refresh occ/studio/eyewear.json from Ralba Optical's live catalog.

Ralba's storefront (ralbaoptical.com) is an Angular SPA that talks to a signed API
(api.ralbatech.com). Every browser does the same handshake: AES-256-CBC encrypt a
"carrier packet" with the app's public ENC_KEY/IV (shipped in the client bundle),
POST it to /sec/v1/sign-request to get x-client-id/x-timestamp/x-nonce/x-signature,
then call the real endpoint and decrypt the response. We replicate exactly that to
pull the full 2D product list (title, price, image, brand) AND each frame's 3D .glb.

We MERGE (union) the live list into the existing eyewear.json by title — keeping the
designer frames the live store no longer carries AND adding the new ones, never
dropping anything. Existing hand-/vision-tagged shapes win; new frames get a shape
derived from their name + description. 3D .glb URLs are captured for every frame.

    .venv/bin/python scripts/fetch_ralba.py            # merge into eyewear.json
    .venv/bin/python scripts/fetch_ralba.py --replace  # replace instead of union

Keys below are the store's PUBLIC client-side keys (served to every visitor in
main.*.js); this only reads the public product catalog, same as the website.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.request

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as _pad

ENC_KEY = base64.b64decode("WmS81fl8F6v3ghnN7VzbrTKbd85H7n4uFwhcFj89j0M=")
IV = base64.b64decode("hMhzs+WOGF1xZwV+L1COeg==")
BASE = "https://api.ralbatech.com/api/v1/"
SEC = "https://api.ralbatech.com/sec/v1/"
CLIENT = "ralbatech"
STORE_SLUG = os.environ.get("RALBA_STORE_SLUG", "ralba")
EYEWEAR_JSON = os.path.join(os.path.dirname(__file__), "..", "occ", "studio", "eyewear.json")

_SHAPES = [  # (keyword regex, canonical tag) — first match wins, scanned on name+desc
    (r"aviator|pilot|teardrop", "aviator"),
    (r"cat[\s\-]?eye|cateye|butterfly", "cat-eye"),
    (r"wayfarer", "wayfarer"),
    (r"browline|clubmaster", "browline"),
    (r"round|circular|panto", "round"),
    (r"oval", "oval"),
    (r"rimless", "rimless"),
    (r"geometric|hexagon", "geometric"),
    (r"wrap|shield|sport|sporty", "sport"),
    (r"oversize", "oversized"),
    (r"square", "square"),
    (r"rectangular|rectangle", "rectangle"),
]


def _enc(obj) -> str:
    data = (json.dumps(obj, separators=(",", ":")) if not isinstance(obj, str) else obj).encode()
    p = _pad.PKCS7(128).padder(); d = p.update(data) + p.finalize()
    c = Cipher(algorithms.AES(ENC_KEY), modes.CBC(IV)).encryptor()
    return base64.b64encode(c.update(d) + c.finalize()).decode()


def _dec(b64: str) -> str:
    raw = base64.b64decode(b64)
    c = Cipher(algorithms.AES(ENC_KEY), modes.CBC(IV)).decryptor()
    d = c.update(raw) + c.finalize()
    u = _pad.PKCS7(128).unpadder()
    return (u.update(d) + u.finalize()).decode("utf-8", "ignore")


def _post(url, obj, headers=None):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0", **(headers or {})})
    r = urllib.request.urlopen(req, timeout=40)
    return json.loads(r.read().decode())


def _signed(method, url, body=None):
    Y = _enc(body) if body is not None else None
    parts = url.split("/"); idx = parts.index("v1")
    ne = {"clientId": CLIENT, "method": method,
          "path": "/api/" + "/".join(parts[idx:]), "bodyStr": Y}
    sig = json.loads(_dec(_post(SEC + "sign-request", {"carrierPacket": _enc(ne)})["payload"]))
    h = {"x-client-id": sig["clientId"], "x-timestamp": str(sig["timestamp"]),
         "x-nonce": sig["nonce"], "x-signature": sig["signature"]}
    resp = _post(url, {"carrierPacket": Y} if Y is not None else {}, h)
    return json.loads(_dec(resp["payload"])) if "payload" in resp else resp


def _shape_of(name, desc):
    hay = f"{name} {desc}".lower()
    for rx, tag in _SHAPES:
        if re.search(rx, hay):
            return tag
    return "rectangle"            # modal shape — keeps every frame searchable


def _first(lst, key):
    for it in (lst or []):
        v = (it or {}).get(key)
        if v:
            return v
    return ""


def _map(p):
    name = (p.get("product_name") or "").strip()
    return {
        "title": name,
        "price": str(p.get("product_retail_price") or p.get("product_price") or ""),
        "img": _first(p.get("product_image"), "pro_image"),
        "gender": (p.get("product_gender") or "unisex"),
        "ar": True,
        "brand": ((p.get("product_brand") or {}).get("brand_name") or "").strip(),
        "shape": _shape_of(name, p.get("product_description") or ""),
        "glb": _first(p.get("product_tryon_3d_image"), "pro_3d_image"),
    }


def fetch_live(slug=STORE_SLUG):
    out, page = [], 1
    while True:
        d = _signed("POST", BASE + "stores/all-product-2d-list-by-vendor-for-user",
                    {"store_slug": slug, "page": page, "limit": 500}).get("data", {})
        prods = d.get("products", [])
        out += prods
        if page >= (d.get("totalPages") or 1) or not prods:
            break
        page += 1
    return out


def _norm(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true",
                    help="replace the catalog with the live list instead of union")
    args = ap.parse_args()

    doc = json.loads(open(EYEWEAR_JSON).read())
    old = doc.get("eyewear", [])
    live_raw = fetch_live()
    live = [_map(p) for p in live_raw if (p.get("product_name") and _first(p.get("product_image"), "pro_image"))]
    print(f"live frames: {len(live)}  | existing: {len(old)}")

    if args.replace:
        merged = live
    else:
        by = {_norm(it["title"]): dict(it) for it in old}
        added = 0
        for it in live:
            k = _norm(it["title"])
            if k in by:
                if not by[k].get("glb") and it.get("glb"):   # enrich with 3D
                    by[k]["glb"] = it["glb"]
            else:
                by[k] = it; added += 1
        merged = list(by.values())
        print(f"added {added} new frames (union)")

    # guarantee every frame has a shape tag (the regression depends on it)
    for it in merged:
        it.setdefault("shape", "rectangle")
        if not it.get("shape"):
            it["shape"] = "rectangle"
    with_glb = sum(1 for it in merged if it.get("glb"))
    doc["eyewear"] = merged
    open(EYEWEAR_JSON, "w").write(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"wrote {len(merged)} frames → eyewear.json  ({with_glb} with 3D .glb)")


if __name__ == "__main__":
    sys.exit(main())
