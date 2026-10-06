"""Studio auth gate (occ/studio/auth.py) — offline regression.

Mounts the gate on a tiny FastAPI app and swaps Firebase Admin for a fake, so it
needs no network, credentials, or running studio. Covers: auth-off passthrough,
401/302 for anonymous HTTP, the session mint policy (allowlist, unverified
password accounts), WebSocket cookie + same-origin check, logout revocation,
cookie-verification caching, and the public-path allowlist.

    .venv/bin/python test_studio_auth.py
"""

from __future__ import annotations

import os

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from occ.studio import auth

_ENV = ("STUDIO_AUTH", "STUDIO_AUTH_ALLOW", "STUDIO_FIREBASE_API_KEY")


class FakeAdmin:
    """ID token "id:<email>[:password[:unverified]]" → session cookie "sess:<email>"."""

    def __init__(self):
        self.revoked: set[str] = set()
        self.verifies = 0

    def verify_id_token(self, tok):
        _, email, *rest = tok.split(":")
        provider = rest[0] if rest else "google.com"
        return {"sub": email, "email": email, "firebase": {"sign_in_provider": provider},
                "email_verified": "unverified" not in rest}

    def create_session_cookie(self, tok, expires_in):
        return "sess:" + tok.split(":")[1]

    def verify_session_cookie(self, cookie, check_revoked=False):
        self.verifies += 1
        if not cookie.startswith("sess:"):
            raise ValueError("malformed")
        email = cookie[5:]
        if check_revoked and email in self.revoked:
            raise ValueError("revoked")
        return {"sub": email, "email": email}

    def revoke_refresh_tokens(self, uid):
        self.revoked.add(uid)


def _app(env: dict):
    for k in _ENV:
        os.environ.pop(k, None)
    os.environ.update(env)
    auth._cache.clear()
    fake = FakeAdmin()
    auth._admin_auth = lambda: fake
    app = FastAPI()
    on = auth.install(app)

    @app.get("/stats")
    def stats():
        return {"ok": True}

    @app.post("/stripe/webhook")
    def hook():
        return {"hook": True}

    @app.websocket("/ws/cam")
    async def ws(w: WebSocket):
        await w.accept()
        await w.send_text("hi")
        await w.close()

    return app, fake, on


def _ws_ok(c: TestClient, origin: str) -> bool:
    try:
        with c.websocket_connect("/ws/cam", headers={"origin": origin}) as w:
            return w.receive_text() == "hi"
    except Exception:
        return False


def _login(c: TestClient, tok: str):
    return c.post("/auth/session", json={"idToken": tok})


def collect() -> list[tuple[str, bool, str]]:
    R: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        R.append((name, bool(cond), detail))

    saved = {k: os.environ.get(k) for k in _ENV}
    real_admin = auth._admin_auth
    try:
        # --- secure by default: no STUDIO_AUTH at all → signed-in gate ---
        app, _, on = _app({})
        check("default (unset) → sign-in required", on is True
              and TestClient(app).get("/stats").status_code == 401)

        # --- STUDIO_AUTH=off: local dev, THIS MACHINE ONLY ---
        app, _, on = _app({"STUDIO_AUTH": "off"})
        c = TestClient(app)
        check("auth off: sign-in gate not installed", on is False)
        check("auth off: loopback routes open", c.get("/stats").status_code == 200)
        check("auth off: /auth/me reports off", c.get("/auth/me").json()["auth"] is False)
        check("auth off: loopback websocket open", _ws_ok(c, "http://testserver"))
        check("auth off: guard.js served", c.get("/auth/guard.js").status_code == 200)
        remote = TestClient(app, client=("203.0.113.5", 4321))
        check("auth off: remote client → 403", remote.get("/stats").status_code == 403)
        check("auth off: remote websocket refused", not _ws_ok(remote, "http://testserver"))
        r = c.get("/stats", headers={"x-forwarded-for": "203.0.113.5"})
        check("auth off: proxied (X-Forwarded-For) → 403", r.status_code == 403)
        r = c.get("/stats", headers={"forwarded": "for=203.0.113.5"})
        check("auth off: proxied (Forwarded) → 403", r.status_code == 403)

        # --- auth ON ---
        app, fake, on = _app({"STUDIO_AUTH": "on", "STUDIO_FIREBASE_API_KEY": "k",
                              "STUDIO_AUTH_ALLOW": "alice@x.com,@corp.com"})
        c = TestClient(app, follow_redirects=False)
        check("auth on: gate installed", on is True)
        r = c.get("/stats")
        check("anon API → 401", r.status_code == 401, str(r.status_code))
        r = c.get("/stats?x=1", headers={"accept": "text/html"})
        check("anon page → 302 /login?next=", r.status_code == 302
              and r.headers["location"] == "/login?next=/stats%3Fx%3D1", r.headers.get("location"))
        check("anon websocket refused", not _ws_ok(c, "http://testserver"))
        check("anon guard.js → 401 (gated like everything else)",
              c.get("/auth/guard.js").status_code == 401)
        check("public: /healthz /login /auth/config",
              all(c.get(p).status_code == 200 for p in ("/healthz", "/login", "/auth/config")))
        check("public: /stripe/webhook (signature-checked)", c.post("/stripe/webhook").status_code == 200)
        check("bogus cookie → 401",
              TestClient(app, cookies={auth.COOKIE: "junk"}).get("/stats").status_code == 401)

        r = _login(c, "id:bob@x.com:password:unverified")
        check("unverified password account → 403", r.status_code == 403, r.text)
        r = _login(c, "id:eve@evil.com")
        check("not on allowlist → 403", r.status_code == 403, r.text)
        check("missing idToken → 400", c.post("/auth/session", json={}).status_code == 400)
        r = _login(c, "id:dan@corp.com")
        check("@domain allowlist entry admits", r.status_code == 200, r.text)
        c.cookies.clear()

        r = _login(c, "id:alice@x.com")
        sc = r.headers.get("set-cookie", "")
        check("allowed login → httpOnly __session cookie",
              r.status_code == 200 and sc.startswith(auth.COOKIE + "=") and "HttpOnly" in sc, sc)
        check("signed in: API open", c.get("/stats").status_code == 200)
        check("signed in: /auth/me email", c.get("/auth/me").json().get("email") == "alice@x.com")
        check("signed in: same-origin websocket", _ws_ok(c, "http://testserver"))
        check("signed in: guard.js served", c.get("/auth/guard.js").status_code == 200)
        check("signed in: cross-origin websocket refused", not _ws_ok(c, "https://evil.com"))
        n = fake.verifies
        for _ in range(5):
            c.get("/stats")
        check("cookie verification cached", fake.verifies == n, f"{fake.verifies - n} extra")

        os.environ["STUDIO_AUTH_ALLOW"] = "someone@else.com"
        auth._cache.clear()
        check("allowlist tightened → existing session refused", c.get("/stats").status_code == 401)
        os.environ["STUDIO_AUTH_ALLOW"] = "alice@x.com"
        auth._cache.clear()

        cookie = c.cookies.get(auth.COOKIE)
        r = c.post("/auth/logout")
        check("logout clears cookie + revokes", r.status_code == 200 and "alice@x.com" in fake.revoked
              and not c.cookies.get(auth.COOKIE))
        stolen = TestClient(app, cookies={auth.COOKIE: cookie})
        check("revoked cookie refused after logout", stolen.get("/stats").status_code == 401)

        # --- fail closed ---
        app, _, _ = _app({"STUDIO_AUTH": "on"})
        check("empty allowlist → nobody admitted", _login(TestClient(app), "id:alice@x.com").status_code == 403)
        app, _, _ = _app({"STUDIO_AUTH": "on", "STUDIO_AUTH_ALLOW": "*"})
        check("'*' admits any signed-in user", _login(TestClient(app), "id:who@ever.org").status_code == 200)
    finally:
        auth._admin_auth = real_admin
        auth._cache.clear()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return R


def main():
    results = collect()
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({detail})" if detail and not ok else ""))
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    assert not bad, bad


if __name__ == "__main__":
    main()
