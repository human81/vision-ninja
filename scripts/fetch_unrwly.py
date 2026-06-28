#!/usr/bin/env python
"""Build occ/studio/unrwly.json from the unrwly Etsy shop — the LEGIT way (Etsy Open API v3).

Etsy blocks page scraping (403) and its ToS forbids it, so we use the official API, which
serves a shop's public active listings (title, price, url, images) with just an app API key
— no seller OAuth needed for read-only public listings.

  1. Create a free Etsy app:  https://www.etsy.com/developers/your-apps  → copy the "keystring".
  2. export ETSY_API_KEY=...          (or put ETSY_API_KEY=... in .env — it's gitignored)
  3. .venv/bin/python scripts/fetch_unrwly.py            # → occ/studio/unrwly.json
     .venv/bin/python scripts/fetch_unrwly.py --shop Unrwly --max 200

Output schema matches garments.json so it plugs straight into the studio's apparel-style
store (every item is shoppable + generative virtual try-on):
  {"sponsor": {"name","url","tagline"}, "garments": [{"title","price","type","cat","img","url"}]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

API = "https://openapi.etsy.com/v3/application"
OUT = os.path.join(os.path.dirname(__file__), "..", "occ", "studio", "unrwly.json")


def _get(path: str, key: str, **params) -> dict:
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-api-key": key,
                                               "User-Agent": "occ-studio/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())


def _shop_id(name: str, key: str) -> int:
    d = _get("/shops", key, shop_name=name)
    results = d.get("results") or []
    if not results:
        sys.exit(f"shop '{name}' not found via the Etsy API")
    return int(results[0]["shop_id"])


def _price(listing: dict) -> str:
    p = listing.get("price") or {}
    amt, div = p.get("amount"), p.get("divisor") or 100
    if amt is None:
        return ""
    val = amt / div
    cur = p.get("currency_code", "")
    return f"{val:.2f}{(' ' + cur) if cur and cur != 'USD' else ''}"


def _img(listing: dict) -> str:
    imgs = listing.get("images") or []
    if not imgs:
        return ""
    im = imgs[0]
    return im.get("url_570xN") or im.get("url_fullxfull") or im.get("url_680x540") or ""


def _map(listing: dict) -> dict:
    return {"title": (listing.get("title") or "").strip()[:120],
            "price": _price(listing),
            "type": (listing.get("taxonomy_path") or ["Item"])[-1] if listing.get("taxonomy_path") else "Item",
            "cat": "unrwly",
            "img": _img(listing),
            "url": listing.get("url") or f"https://www.etsy.com/shop/{listing.get('shop_id','')}"}


def fetch(shop: str, key: str, cap: int) -> list[dict]:
    sid = _shop_id(shop, key)
    items, offset = [], 0
    while len(items) < cap:
        d = _get(f"/shops/{sid}/listings/active", key,
                 limit=min(100, cap - len(items)), offset=offset, includes="Images")
        batch = d.get("results") or []
        if not batch:
            break
        items += [_map(b) for b in batch]
        offset += len(batch)
        if len(batch) < 100:
            break
    return [it for it in items if it["img"]]      # only try-onable items (need an image)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="Unrwly")
    ap.add_argument("--max", type=int, default=200)
    a = ap.parse_args()
    key = os.environ.get("ETSY_API_KEY")
    if not key:
        sys.exit("set ETSY_API_KEY (free app keystring from etsy.com/developers/your-apps)")
    items = fetch(a.shop, key, a.max)
    doc = {"sponsor": {"name": "unrwly",
                       "url": f"https://www.etsy.com/shop/{a.shop}",
                       "tagline": "Sponsored by unrwly"},
           "garments": items}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(items)} unrwly items → {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
