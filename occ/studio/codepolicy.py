"""Whether the agent may run code it wrote itself (create_overlay / run_cv_code / run_cv_video).

That code runs in-process with a restricted-builtins "sandbox" that stops accidents, not
attacks: through `np`/`cv2` it can read any file the server can (e.g. `.env` and its API
keys). So it is OFF unless explicitly enabled:

  STUDIO_AGENT_CODE=on     allow it — local use only (set in the gitignored .env)

Built-in presets and native face filters ship with the repo and are always allowed.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("studio.codepolicy")

DISABLED = ("Agent-authored code is disabled on this server (STUDIO_AGENT_CODE is not 'on'). "
            "Use a built-in preset instead (list_overlays shows them; toggle_overlay turns one "
            "on) or apply_face_filter.")


def agent_code_enabled() -> bool:
    on = os.environ.get("STUDIO_AGENT_CODE", "").strip().lower() in ("1", "on", "true", "yes")
    if on and os.environ.get("K_SERVICE"):          # set by Cloud Run
        log.warning("STUDIO_AGENT_CODE=on on Cloud Run — agent code can read server secrets")
    return on
