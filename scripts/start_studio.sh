#!/usr/bin/env bash
# start_studio.sh — launch the Vision Ninja Studio + sidecars on port 8011
#
# Services started:
#   :8011  Studio UI      http://localhost:8011     (main browser interface — sign in)
#   :9379  litert-lm      local Gemma voice brain   (skipped if not on PATH)
#   :9393  LocateAnything-3B grounding sidecar      (skipped if .venv-la3b missing)
#
# Usage:
#   ./scripts/start_studio.sh                         # default source (vehicles-2.mp4)
#   ./scripts/start_studio.sh assets/videos/market-square.mp4
#   ./scripts/start_studio.sh rtsp://192.168.1.10/stream
#   ./scripts/start_studio.sh --no-voice              # skip litert-lm
#   ./scripts/start_studio.sh --no-locate             # skip LocateAnything sidecar
#
# Requirements:
#   uv pip install -e ".[cv,studio]"   (core + studio deps)
#   .venv/bin/playwright install chromium  (only needed for fishfood e2e tests)
#
# Optional sidecars:
#   litert-lm  : uv tool install litert-lm   (local Gemma, $0 voice brain)
#   la3b       : .venv-la3b created by scripts/la3b_server.py setup
#
# API keys (put these in a gitignored .env — loaded automatically):
#   GOOGLE_API_KEY or GEMINI_API_KEY   — Gemini brain + Live voice
#   OPENAI_API_KEY                     — OpenAI Realtime voice (better Haitian Creole)
#   STRIPE_SECRET_KEY                  — agentic checkout (test mode: sk_test_...)
#   STRIPE_WEBHOOK_SECRET              — Stripe webhook verification
#
# Stop: Ctrl-C  (run.py studio shuts down the sidecars automatically on exit)

set -euo pipefail

# ── locate project root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── sanity checks ──────────────────────────────────────────────────────────────
if [[ ! -f ".venv/bin/python" ]]; then
  echo "ERROR: .venv not found. Run:"
  echo "  uv venv --python 3.12 .venv && uv pip install -e '.[cv,studio]'"
  exit 1
fi

if ! .venv/bin/python -c "import google.adk" 2>/dev/null; then
  echo "ERROR: [studio] extra not installed. Run:"
  echo "  uv pip install -e '.[cv,studio]'"
  exit 1
fi

# ── load .env if present ───────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

# ── parse args: first non-flag arg = source URI; flags passed through ──────────
EXTRA_ARGS=()
SOURCE_URI=""
for arg in "$@"; do
  case "$arg" in
    --no-voice|--no-locate) EXTRA_ARGS+=("$arg") ;;
    --*)                    EXTRA_ARGS+=("$arg") ;;
    *)                      SOURCE_URI="$arg" ;;
  esac
done

if [[ -n "$SOURCE_URI" ]]; then
  EXTRA_ARGS+=("--source" "$SOURCE_URI")
fi

# ── ensure log dir exists ──────────────────────────────────────────────────────
mkdir -p out/studio/logs

# ── launch ────────────────────────────────────────────────────────────────────
echo "Starting Vision Ninja Studio..."
echo "  UI  → http://localhost:8011  (sign-in required; STUDIO_AUTH=off = this machine only)"
echo "  Logs → out/studio/logs/{studio,litert,la3b}.log"
echo ""

exec .venv/bin/python run.py studio ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
