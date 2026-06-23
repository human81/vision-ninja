"""Airplane-mode switch for the studio — `STUDIO_OFFLINE=1`.

The local stack is genuinely on-device (litert Gemma brain on localhost,
faster-whisper STT, gemma-3 / medgemma vision via transformers/MPS), but two
things still reach for the network and break BIDI when wifi is off:

  1. HuggingFace pings huggingface.co to *revalidate the on-disk cache* before
     loading a model. Wifi on → instant; wifi off → hangs on connect-timeout or
     throws, stalling the STT / vision load mid-turn.
  2. The agent + vision paths fall back to **cloud Gemini** on any local hiccup
     (`agent.stream`, `tools.describe_image`) — exactly what can't connect offline.

`STUDIO_OFFLINE=1` fixes both: force every HF load to cache-only (no round-trip),
and make `offline()` True so the fallbacks degrade to LOCAL/sim instead of cloud.
Weights must already be cached (they are, after one online run).

`apply()` is called at package import so the env flags are set *before* any
transformers / huggingface_hub import; `offline()` is the runtime predicate the
loaders and fallbacks consult.
"""

from __future__ import annotations

import os

_TRUTHY = ("1", "true", "yes", "on")


def offline() -> bool:
    """True when the studio should never touch the network (cache-only + local fallbacks)."""
    return os.environ.get("STUDIO_OFFLINE", "").strip().lower() in _TRUTHY


def apply() -> bool:
    """Idempotent. If STUDIO_OFFLINE is set, force HF/transformers to cache-only and flip
    the flags on any already-imported HF modules (their constants are read at import).
    Returns the resolved offline state."""
    if not offline():
        return False
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # If huggingface_hub / transformers were imported before this ran, their module-level
    # offline constants were already snapshotted from the env — overwrite them too.
    try:
        import huggingface_hub.constants as _c
        _c.HF_HUB_OFFLINE = True
    except Exception:
        pass
    try:
        import transformers.utils.hub as _h
        _h._is_offline_mode = True
    except Exception:
        pass
    return True
