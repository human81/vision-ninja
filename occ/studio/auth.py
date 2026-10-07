"""Firebase Auth gate for the studio — Momentum's session-cookie pattern, ported.

  browser signs in (Firebase JS SDK: Google popup or email/password)
    → POST /auth/session {idToken}
    → server verifies the ID token, rejects unverified password accounts,
      checks the allowlist, mints a 5-day Firebase SESSION COOKIE (httpOnly `__session`)
    → every request (HTTP *and* WebSocket) is checked by `AuthMiddleware`.

A cookie (not an Authorization header) because the MJPEG <img> and the /ws/* sockets
can't send custom headers but do carry cookies.

Env:
  STUDIO_AUTH                         ON unless explicitly "off". "off" is a LOCAL-DEV mode that
                                      serves loopback clients only and refuses anything proxied
                                      (X-Forwarded-For / Forwarded) — so a deploy that forgets to
                                      turn auth on fails closed instead of open.
  STUDIO_AUTH_ALLOW=a@x.com,@corp.com who may enter: emails and/or @domains; "*" = any
                                      signed-in user. EMPTY = nobody (fail closed — the
                                      agent can exec code, so "any Google account" is opt-in).
  STUDIO_FIREBASE_API_KEY / _AUTH_DOMAIN / _PROJECT_ID / _APP_ID   web SDK config for /login
  GOOGLE_APPLICATION_CREDENTIALS      optional; else gcloud ADC locally / the service account on Cloud Run
"""

from __future__ import annotations

import logging
import os
import threading
import time
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from urllib.parse import quote

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

log = logging.getLogger("studio.auth")

COOKIE = "__session"                 # the one cookie name Firebase Hosting forwards
SESSION_TTL = 60 * 60 * 24 * 5       # 5 days, same as Momentum
_CACHE_TTL = 300                     # re-verify a cookie (revocation check) at most every 5 min
_LOGIN = Path(__file__).with_name("login.html")

# Reachable without a session. Stripe's webhook is authenticated by its signature
# (required when auth is on — see server.py), not by a user cookie.
PUBLIC = {"/login", "/auth/config", "/auth/session", "/auth/logout", "/healthz",
          "/favicon.ico", "/stripe/webhook"}


def enabled() -> bool:
    """Signed-in gate ON unless explicitly switched off (secure by default)."""
    return os.environ.get("STUDIO_AUTH", "").strip().lower() not in ("0", "off", "false", "no")


def web_config() -> dict:
    """Public Firebase web-SDK config (these values ship to every browser by design)."""
    g = lambda k: os.environ.get(f"STUDIO_FIREBASE_{k}", "")
    return {"apiKey": g("API_KEY"), "authDomain": g("AUTH_DOMAIN"),
            "projectId": g("PROJECT_ID"), "appId": g("APP_ID")}


def allowed(email: str | None) -> bool:
    rules = [r.strip().lower() for r in os.environ.get("STUDIO_AUTH_ALLOW", "").split(",")
             if r.strip()]
    if "*" in rules:
        return True
    e = (email or "").lower()
    if not e:
        return False
    return any(e == r or (r.startswith("@") and e.endswith(r)) for r in rules)


# ---------- Firebase Admin (lazy, so auth-off never imports it) ----------
_init_lock = threading.Lock()


def _admin_auth():
    import firebase_admin
    from firebase_admin import auth, credentials
    with _init_lock:
        if not firebase_admin._apps:
            # httpTimeout: these calls run on the shared threadpool that every sync route and
            # the session check use; with no timeout one stalled call can hold a thread forever.
            opts = {"httpTimeout": 10}
            if web_config()["projectId"]:
                opts["projectId"] = web_config()["projectId"]
            # ADC covers every case: GOOGLE_APPLICATION_CREDENTIALS (service-account OR
            # gcloud user file), gcloud's default login, and the Cloud Run service account.
            firebase_admin.initialize_app(credentials.ApplicationDefault(), opts)
    return auth


def _mint(id_token: str) -> tuple[str, dict]:
    """ID token → (session cookie, decoded claims). Raises PermissionError on policy."""
    auth = _admin_auth()
    decoded = auth.verify_id_token(id_token)
    # Momentum HIGH-1: enforce email verification SERVER-side for password accounts
    # (OAuth providers are provider-verified).
    provider = (decoded.get("firebase") or {}).get("sign_in_provider")
    if provider == "password" and decoded.get("email_verified") is not True:
        raise PermissionError("Please verify your email address before signing in.")
    if not allowed(decoded.get("email")):
        raise PermissionError(f"{decoded.get('email') or 'This account'} is not allowed "
                              "into this studio.")
    import datetime as _dt
    cookie = auth.create_session_cookie(id_token,
                                        expires_in=_dt.timedelta(seconds=SESSION_TTL))
    return cookie, decoded


_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def _verify(cookie: str) -> dict | None:
    """Session cookie → claims, or None. Cached so the MJPEG stream + /stats polling don't
    hit Identity Toolkit on every request (Momentum dedupes the same call)."""
    now = time.time()
    with _cache_lock:
        hit = _cache.get(cookie)
        if hit and hit[0] > now:
            return hit[1]
    try:
        claims = _admin_auth().verify_session_cookie(cookie, check_revoked=True)
    except Exception as e:                     # expired / revoked / malformed / no creds
        log.info("session rejected: %s", type(e).__name__)
        return None
    if not allowed(claims.get("email")):       # allowlist tightened since the cookie was minted
        return None
    with _cache_lock:
        if len(_cache) > 1000:
            _cache.clear()
        _cache[cookie] = (now + _CACHE_TTL, claims)
    return claims


def _forget(cookie: str | None):
    if cookie:
        with _cache_lock:
            _cache.pop(cookie, None)


def _cookie_of(headers: list[tuple[bytes, bytes]]) -> str | None:
    for k, v in headers:
        if k == b"cookie":
            jar = SimpleCookie()
            try:
                jar.load(v.decode("latin-1"))      # handles "quoted" values too
            except CookieError:
                continue
            if COOKIE in jar and jar[COOKIE].value:
                return jar[COOKIE].value
    return None


def _secure(req: Request) -> bool:
    xf = req.headers.get("x-forwarded-proto")
    return (xf.split(",")[0].strip() if xf else req.url.scheme) == "https"


# ---------- the gate: pure ASGI so it covers WebSockets too ----------
class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind not in ("http", "websocket") or scope["path"] in PUBLIC:
            return await self.app(scope, receive, send)
        headers = scope.get("headers") or []
        cookie = _cookie_of(headers)
        claims = await anyio.to_thread.run_sync(_verify, cookie) if cookie else None

        if kind == "websocket":
            # Cross-site pages can open sockets to us with the user's cookie attached,
            # so also require a same-origin handshake.
            h = dict(headers)
            origin = h.get(b"origin", b"").decode().split("://", 1)[-1]
            if claims is None or origin != h.get(b"host", b"").decode():
                await send({"type": "websocket.close", "code": 1008})   # → 403 pre-accept
                return
        elif claims is None:
            wants_html = (scope.get("method") == "GET"
                          and b"text/html" in dict(headers).get(b"accept", b""))
            if wants_html:
                nxt = scope["path"] + (("?" + scope["query_string"].decode())
                                       if scope.get("query_string") else "")
                resp = HTMLResponse("", status_code=302,
                                    headers={"location": "/login?next=" + quote(nxt)})
            else:
                resp = JSONResponse({"error": "unauthenticated"}, status_code=401)
            return await resp(scope, receive, send)

        scope.setdefault("state", {})["user"] = claims
        return await self.app(scope, receive, send)


# "testclient" = Starlette's in-process TestClient (never a real socket peer).
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


class LoopbackOnlyMiddleware:
    """STUDIO_AUTH=off: no sign-in, but only for requests made on this machine. Anything
    from another host — or relayed by a proxy/load balancer (Cloud Run always adds
    X-Forwarded-For) — is refused, so auth-off can never be reachable from outside."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        host = (scope.get("client") or ("", 0))[0]
        names = {k for k, _ in scope.get("headers") or []}
        if host in _LOOPBACK and b"x-forwarded-for" not in names and b"forwarded" not in names:
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        resp = JSONResponse({"error": "STUDIO_AUTH=off serves this machine only"},
                            status_code=403)
        return await resp(scope, receive, send)


# Every studio page loads this (it's behind the gate like everything else): an expired or
# revoked session turns any API call into a 401 → bounce to /login and come back.
GUARD_JS = """(()=>{const f=window.fetch.bind(window);
window.fetch=async(...a)=>{const r=await f(...a);
if(r.status===401)location.replace('/login?next='+encodeURIComponent(location.pathname+location.search));
return r}})();"""


def install(app: FastAPI) -> bool:
    """Add the /auth routes + the sign-in gate (or, with STUDIO_AUTH=off, the loopback-only
    guard). Returns whether sign-in is on."""

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/auth/me")
    def me(req: Request):
        u = getattr(req.state, "user", None) if enabled() else None
        return {"auth": enabled(),
                "email": (u or {}).get("email"), "name": (u or {}).get("name")}

    @app.get("/auth/guard.js")
    def guard_js():
        return Response(GUARD_JS, media_type="text/javascript")

    if not enabled():
        app.add_middleware(LoopbackOnlyMiddleware)
        log.warning("studio auth OFF (STUDIO_AUTH=off) — serving loopback clients only")
        return False

    if not os.environ.get("STUDIO_AUTH_ALLOW", "").strip():
        log.warning("STUDIO_AUTH is on but STUDIO_AUTH_ALLOW is empty — nobody can sign in")
    if not web_config()["apiKey"]:
        log.warning("STUDIO_FIREBASE_API_KEY unset — /login can't start the Firebase SDK")

    @app.get("/login", response_class=HTMLResponse)
    def login_page():
        return _LOGIN.read_text()

    @app.get("/auth/config")
    def config():
        return web_config()

    @app.post("/auth/session")
    async def session(req: Request):
        try:
            id_token = (await req.json()).get("idToken")
        except Exception:
            id_token = None
        if not id_token or not isinstance(id_token, str):
            return JSONResponse({"error": "ID token is required"}, status_code=400)
        try:
            cookie, decoded = await anyio.to_thread.run_sync(_mint, id_token)
        except PermissionError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        except Exception as e:
            log.warning("session mint failed: %s", e)
            return JSONResponse({"error": "Sign-in failed."}, status_code=401)
        resp = JSONResponse({"ok": True, "email": decoded.get("email")})
        resp.set_cookie(COOKIE, cookie, max_age=SESSION_TTL, httponly=True,
                        secure=_secure(req), samesite="lax", path="/")
        return resp

    @app.post("/auth/logout")
    async def logout(req: Request):
        cookie = req.cookies.get(COOKIE)
        _forget(cookie)
        if cookie:                       # revoke so a copied cookie stops working everywhere
            try:
                claims = await anyio.to_thread.run_sync(
                    lambda: _admin_auth().verify_session_cookie(cookie))
                await anyio.to_thread.run_sync(
                    lambda: _admin_auth().revoke_refresh_tokens(claims["sub"]))
            except Exception:
                pass
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE, path="/")
        return resp

    app.add_middleware(AuthMiddleware)
    log.info("studio auth ON (allow=%s)", os.environ.get("STUDIO_AUTH_ALLOW", ""))
    return True
