"""Agentic checkout — turn a try-on into a real (test-mode) Stripe purchase.

The studio is the MERCHANT: the agent builds a cart from the selected frame/garment
(title + price + image from the sponsored catalog) and creates a **Stripe Checkout
Session**. The shopper pays — and enters their shipping address — on Stripe's own
hosted, PCI-compliant page; we never see or store a card number. On `checkout.session
.completed` (webhook) the order is recorded under out/studio/orders/.

Keys live in .env (gitignored), never in chat or the repo:
  STRIPE_API_KEY        sk_test_...  (test mode → no real money; sk_live_ for real)
  STRIPE_WEBHOOK_SECRET whsec_...    (optional; verifies webhook authenticity)
  STRIPE_CURRENCY       default currency when a price string has none (default usd)
  STUDIO_PUBLIC_URL     base URL Stripe redirects back to (default http://localhost:8011)

A shopper's name/email/address (PII, not payment data) may be cached locally in
out/studio/profile.json (gitignored) to prefill checkout — Stripe still collects/
confirms shipping on its page.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from . import STUDIO_DIR

_ORDERS_DIR = Path(STUDIO_DIR) / "orders"
_PROFILE = Path(STUDIO_DIR) / "profile.json"

# Currency symbols → ISO code (for parsing catalog price strings like "CA$189", "€120").
_SYMBOLS = {"$": "usd", "us$": "usd", "ca$": "cad", "c$": "cad", "a$": "aud",
            "£": "gbp", "€": "eur", "¥": "jpy", "₹": "inr"}
_ZERO_DECIMAL = {"jpy", "krw", "vnd", "clp", "xof"}     # Stripe: no minor unit

# The last product the user tried on / shopped — so "buy it" needs no arguments.
_LAST: dict | None = None


def public_url() -> str:
    return os.environ.get("STUDIO_PUBLIC_URL", "http://localhost:8011").rstrip("/")


def api_key() -> str:
    """The Stripe secret key — STRIPE_API_KEY or the Stripe-conventional STRIPE_SECRET_KEY
    (the name Momentum uses), whichever is set."""
    return os.environ.get("STRIPE_API_KEY") or os.environ.get("STRIPE_SECRET_KEY") or ""


def ready() -> tuple[bool, str]:
    """(ok, reason). False with a human reason if the rail isn't configured."""
    try:
        import stripe  # noqa: F401
    except Exception:
        return False, "the `stripe` SDK isn't installed — run `uv pip install -e \".[pay]\"`"
    if not api_key():
        return False, ("no Stripe key — add STRIPE_API_KEY (or STRIPE_SECRET_KEY) to .env, "
                       "a TEST secret key (sk_test_… from dashboard.stripe.com/test/apikeys)")
    return True, ""


def test_mode() -> bool:
    return api_key().startswith(("sk_test_", "rk_test_"))


# ---- price parsing ---------------------------------------------------------
def parse_money(price: str, default_ccy: str | None = None) -> tuple[int, str]:
    """'$189' / 'CA$189.00' / '120 EUR' → (unit_amount_minor, currency). (0,ccy) if unknown."""
    s = (price or "").strip().lower()
    ccy = (default_ccy or os.environ.get("STRIPE_CURRENCY", "usd")).lower()
    if not s:
        return 0, ccy
    for sym, code in sorted(_SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        if sym in s:
            ccy = code
            break
    m = re.search(r"\b([a-z]{3})\b", s)                 # explicit ISO code wins (e.g. "189 cad")
    if m and m.group(1) in {*_SYMBOLS.values(), "chf", "sek", "nok", "dkk", "nzd", "mxn", "brl"}:
        ccy = m.group(1)
    num = re.search(r"(\d+(?:[.,]\d{1,2})?)", s.replace(",", ""))
    if not num:
        return 0, ccy
    amount = float(num.group(1))
    minor = int(round(amount if ccy in _ZERO_DECIMAL else amount * 100))
    return minor, ccy


# ---- shopper profile (PII, NOT payment data) -------------------------------
def profile() -> dict:
    try:
        return json.loads(_PROFILE.read_text()) if _PROFILE.exists() else {}
    except Exception:
        return {}


def save_profile(d: dict) -> dict:
    p = profile()
    for k in ("name", "email", "phone", "address", "country"):
        if k in d and d[k] is not None:
            p[k] = d[k]
    _PROFILE.parent.mkdir(parents=True, exist_ok=True)
    _PROFILE.write_text(json.dumps(p, indent=2))
    return p


# ---- the cart's "pending" item ---------------------------------------------
def set_last(item: dict | None):
    """Remember the most-recently tried-on/shopped product so `buy` needs no args."""
    global _LAST
    if item and (item.get("title") or item.get("img")):
        _LAST = {"title": item.get("title", ""), "price": item.get("price", ""),
                 "brand": item.get("brand", ""), "img": item.get("img", ""),
                 "store": item.get("store", ""), "url": item.get("url", item.get("img", ""))}


def last() -> dict | None:
    return _LAST


# ---- Stripe Checkout -------------------------------------------------------
def _line_item(it: dict) -> dict:
    minor, ccy = parse_money(it.get("price", ""), it.get("currency"))
    if minor <= 0:
        minor, ccy = 100, (it.get("currency") or os.environ.get("STRIPE_CURRENCY", "usd")).lower()
    name = it.get("title") or it.get("brand") or "Item"
    pd = {"name": name[:250]}
    if it.get("brand"):
        pd["description"] = it["brand"][:250]
    img = it.get("img") or ""
    if img.startswith("http"):
        pd["images"] = [img]
    return {"quantity": int(it.get("qty", 1) or 1),
            "price_data": {"currency": ccy, "unit_amount": minor, "product_data": pd}}


def create_checkout(items: list[dict], ship_countries: list[str] | None = None) -> dict:
    """Create a Stripe Checkout Session for `items`. Returns {ok,url,id,amount,currency,...}."""
    ok, why = ready()
    if not ok:
        return {"ok": False, "error": why}
    if not items:
        return {"ok": False, "error": "empty cart"}
    import stripe
    stripe.api_key = api_key()
    line_items = [_line_item(it) for it in items]
    countries = ship_countries or ["US", "CA", "GB", "FR", "DE", "AU", "NL", "IE"]
    base = public_url()
    prof = profile()
    kwargs = dict(
        mode="payment",
        line_items=line_items,
        shipping_address_collection={"allowed_countries": countries},
        phone_number_collection={"enabled": True},
        success_url=f"{base}/checkout/return?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/checkout/return?status=cancel",
        metadata={"source": "vision-ninja-studio",
                  "items": "; ".join(i.get("title", "?") for i in items)[:480]},
    )
    if prof.get("email"):
        kwargs["customer_email"] = prof["email"]
    try:
        sess = stripe.checkout.Session.create(**kwargs)
    except Exception as e:                                   # surface Stripe's own message
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    amount = sum(li["price_data"]["unit_amount"] * li["quantity"] for li in line_items)
    ccy = line_items[0]["price_data"]["currency"]
    record_order(sess.id, items, status="created",
                 amount=amount, currency=ccy, url=sess.url)
    return {"ok": True, "url": sess.url, "id": sess.id, "amount": amount,
            "currency": ccy, "test_mode": test_mode(),
            "items": [li["price_data"]["product_data"]["name"] for li in line_items]}


# ---- orders ----------------------------------------------------------------
def record_order(session_id: str, items: list[dict], status: str,
                 amount: int = 0, currency: str = "usd", url: str = "",
                 shipping: dict | None = None, ts: float | None = None) -> dict:
    _ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    p = _ORDERS_DIR / f"{session_id}.json"
    rec = json.loads(p.read_text()) if p.exists() else {"session_id": session_id,
                                                         "created": ts or time.time()}
    rec.update({"status": status, "amount": amount, "currency": currency,
                "items": [{"title": i.get("title", ""), "price": i.get("price", ""),
                           "brand": i.get("brand", ""), "img": i.get("img", "")} for i in items]
                if items else rec.get("items", []),
                "url": url or rec.get("url", ""), "updated": ts or time.time()})
    if shipping:
        rec["shipping"] = shipping
    p.write_text(json.dumps(rec, indent=2))
    return rec


def mark_paid(session_id: str, shipping: dict | None = None) -> dict:
    p = _ORDERS_DIR / f"{session_id}.json"
    if not p.exists():
        return record_order(session_id, [], status="paid", shipping=shipping)
    rec = json.loads(p.read_text())
    rec["status"] = "paid"
    rec["updated"] = time.time()
    if shipping:
        rec["shipping"] = shipping
    p.write_text(json.dumps(rec, indent=2))
    return rec


def list_orders(limit: int = 25) -> list[dict]:
    if not _ORDERS_DIR.exists():
        return []
    recs = []
    for f in _ORDERS_DIR.glob("*.json"):
        try:
            recs.append(json.loads(f.read_text()))
        except Exception:
            pass
    recs.sort(key=lambda r: r.get("updated", 0), reverse=True)
    return recs[:limit]
