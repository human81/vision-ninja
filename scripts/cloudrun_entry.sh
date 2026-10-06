#!/bin/sh
# Container entrypoint (Dockerfile CMD).
#
#   cpu image (STUDIO_LOCAL_MODELS=off): just the studio.
#   gpu image (STUDIO_LOCAL_MODELS=on):  the same stack as `run.py studio` on the Mac —
#     studio + LocateAnything-3B (:9393, spawned by run.py) + litert-lm Gemma 4 (:9379),
#     all on loopback, sharing the L4. Model weights live on the /app/out bucket mount, so
#     each downloads once (first boot is slow; later boots read from the bucket).
set -eu
cd /app
mkdir -p out/studio/logs

case "${STUDIO_LOCAL_MODELS:-off}" in
  on|1|true|yes)
    if command -v litert-lm >/dev/null 2>&1; then
      mkdir -p out/litert-lm && ln -sfn /app/out/litert-lm "$HOME/.litert-lm"
      (
        if ! litert-lm list 2>/dev/null | grep -q '^gemma-4-12b-it'; then
          echo "[entry] importing gemma-4-12b-it (~6.5GB, first boot only)"
          litert-lm import --from-huggingface-repo litert-community/gemma-4-12B-it-litert-lm \
            gemma-4-12B-it.litertlm gemma-4-12b-it
        fi
        exec litert-lm serve --host 127.0.0.1 --port 9379
      ) >> out/studio/logs/litert.log 2>&1 &
    fi
    # --no-voice: litert is handled above (import first, then serve).
    exec python run.py studio --host 0.0.0.0 --port "${PORT:-8080}" --no-voice
    ;;
  *)
    exec uvicorn occ.studio.server:app --host 0.0.0.0 --port "${PORT:-8080}" \
      --timeout-keep-alive 75
    ;;
esac
