"""Agent-code policy (occ/studio/codepolicy.py) — offline regression.

The agent's "sandbox" (restricted builtins + cv2/np) can read any file the server can, so
agent-authored code must be OFF unless STUDIO_AGENT_CODE=on. Checks: the default refuses
create_overlay / run_cv_code / run_cv_video before running anything, while repo presets still
work; opting in restores them; the SimRunner can't be steered into code injection through
chat text; yt-dlp can't receive an agent-supplied option. No server, models, or network.

    .venv/bin/python test_studio_codepolicy.py
"""

from __future__ import annotations

import ast
import os
from types import SimpleNamespace

import numpy as np

from occ.studio import runtime
from occ.studio import tools as T
from occ.studio.agent import SimRunner
from occ.studio.codepolicy import DISABLED
from occ.studio.overlays import BUILTINS, OverlayEngine, compile_overlay

# Would read a file if it ever ran — the exact capability the policy exists to stop.
_READS_FILE = ("def draw(ctx):\n"
               "    ctx.state['leak'] = np.loadtxt('pyproject.toml', dtype=str,"
               " delimiter='\\x00', comments=None)[:1].tolist()\n")


def collect() -> list[tuple[str, bool, str]]:
    R: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        R.append((name, bool(cond), detail))

    saved_env = os.environ.get("STUDIO_AGENT_CODE")
    saved_ctx = runtime.CTX
    saved_run = T.subprocess.run
    try:
        eng = OverlayEngine()
        frame = np.zeros((8, 8, 3), np.uint8)
        runtime.CTX = SimpleNamespace(overlays=eng, brain=None, ledger=None, settings=None,
                                      pipe=SimpleNamespace(snapshot_clean=lambda: frame))

        # --- default: OFF ---
        os.environ.pop("STUDIO_AGENT_CODE", None)
        try:
            eng.add("evil", "read a file", _READS_FILE)
            check("default: engine refuses agent overlay", False, "added")
        except PermissionError:
            check("default: engine refuses agent overlay", "evil" not in eng.overlays)
        r = T.create_overlay("evil", "read a file", _READS_FILE)
        check("default: create_overlay tool → error", r["status"] == "error"
              and "disabled" in r["error"], str(r))
        r = T.run_cv_code("raise SystemExit('ran')")
        check("default: run_cv_code refused before running",
              r == {"status": "error", "error": DISABLED}, str(r))
        r = T.run_cv_video("raise SystemExit('ran')")
        check("default: run_cv_video refused before running",
              r == {"status": "error", "error": DISABLED}, str(r))
        key = next(iter(BUILTINS))
        check(f"default: repo preset still loads ({key})", eng.add_builtin(key).builtin)
        for v in ("off", "0", "", "maybe"):
            os.environ["STUDIO_AGENT_CODE"] = v
            try:
                eng.add("evil", "x", _READS_FILE)
                ok = False
            except PermissionError:
                ok = True
            check(f"STUDIO_AGENT_CODE={v!r} → still off", ok)

        # --- opted in: ON ---
        os.environ["STUDIO_AGENT_CODE"] = "on"
        r = T.create_overlay("ok", "ring", "def draw(ctx):\n    pass\n")
        check("on: create_overlay works", r["status"] == "success", str(r))
        r = T.run_cv_code("result = 6 * 7")
        check("on: run_cv_code actually runs the code", "42" in str(r), str(r))

        # --- SimRunner: chat text spliced into overlay SOURCE must stay a literal ---
        hostile = "car')\n    __import__('os').system('touch /tmp/pwned')\n    m=ctx.mask('car"
        spec = SimRunner._highlight_overlay(None, hostile)
        tree = ast.parse(spec["code"])
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", None) == "mask"]
        arg = calls[0].args[0].value if calls else None
        check("sim: hostile chat → one function, no injected statements",
              len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
              and len(tree.body[0].body) == 4, spec["code"])
        check("sim: label sanitized to a plain literal",
              isinstance(arg, str) and set(arg) <= set(
                  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-")
              and "import" in arg and "(" not in arg, repr(arg))
        fn = compile_overlay(SimRunner._highlight_overlay(None, "car")["code"])
        calls_made = []
        fake = SimpleNamespace(t=3, anchors=np.array([[10, 40]]),
                               mask=lambda *c: np.array([True]),
                               ring=lambda *a, **k: calls_made.append("ring"),
                               text=lambda *a, **k: calls_made.append("text"))
        try:
            fn(fake)
            drew = calls_made == ["ring", "text"]
        except Exception as e:
            drew = False
            calls_made.append(f"{type(e).__name__}: {e}")
        check("sim: highlight overlay actually draws", drew, str(calls_made))

        # --- sandbox imports: numpy ≥2.5 lazily imports inside C methods via the caller's
        #     builtins, so `__import__` must exist — but only for numpy/cv2/math ---
        probe = compile_overlay("def draw(ctx):\n"
                                "    import math\n"
                                "    return int(np.array([True, True]).sum()) + int(math.floor(0.5))\n")
        check("sandbox: numpy method + `import math` work", probe(None) == 2)
        evil = compile_overlay("def draw(ctx):\n    import os\n    return os.getcwd()\n")
        try:
            evil(None)
            check("sandbox: `import os` refused", False, "imported os")
        except ImportError:
            check("sandbox: `import os` refused", True)
        check("sim: ordinary label unchanged",
              "ctx.mask('forklift')" in SimRunner._highlight_overlay(None, "forklift")["code"])

        # --- yt-dlp: an agent "url" can never be parsed as an option (e.g. --exec) ---
        seen = []
        T.subprocess.run = lambda args, **k: (seen.append(args)
                                              or SimpleNamespace(returncode=1, stdout="", stderr="x"))
        bad = "--exec=touch /tmp/pwned"
        T.load_youtube(bad, seconds=5)
        T._resolve_youtube(bad)
        check("yt-dlp: url always after '--'", seen and all(
            a[-2:] == ["--", bad] and a.count(bad) == 1 for a in seen), str(seen))
    finally:
        T.subprocess.run = saved_run
        runtime.CTX = saved_ctx
        if saved_env is None:
            os.environ.pop("STUDIO_AGENT_CODE", None)
        else:
            os.environ["STUDIO_AGENT_CODE"] = saved_env
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
