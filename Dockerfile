# Vision Ninja Studio — Cloud Run image.
#
#   VARIANT=cpu (default)  CPU torch. Local GPU models (Gemma/MedGemma vision, litert, LA3B,
#                          Qwen-Image) are switched off; the agent + vision run on Gemini.
#   VARIANT=gpu            CUDA torch for an NVIDIA L4 (Cloud Run GPU). Adds the local
#                          vision-model deps; weights download on first use to $HF_HOME.
#
#   docker build -t vision-ninja .                         # cpu
#   docker build --build-arg VARIANT=gpu --build-arg LOCAL_MODELS=on -t vision-ninja:gpu .
#   scripts/deploy_cloudrun.sh {cpu|gpu}                  # Cloud Build + Cloud Run
#
# Secrets never enter the image (.env is in .dockerignore); Cloud Run injects them.

FROM python:3.12-slim-bookworm

ARG VARIANT=cpu
# Local GPU models (Gemma/MedGemma vision): off for cpu; the gpu build passes LOCAL_MODELS=on.
ARG LOCAL_MODELS=off
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    PORT=8080 \
    HOME=/app \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    HF_HOME=/app/out/hf

# ffmpeg: NLE/clip tools + video decode; libgl/glib/egl: OpenCV + MediaPipe on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libegl1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app

# torch first, from the variant's wheel index (the CPU wheels skip ~2GB of CUDA libs).
RUN if [ "$VARIANT" = "gpu" ]; then IDX=https://download.pytorch.org/whl/cu126; \
    else IDX=https://download.pytorch.org/whl/cpu; fi; \
    uv pip install --index-url "$IDX" torch torchvision

# Then the app deps (dependency layer cached until pyproject.toml changes).
COPY pyproject.toml ./
RUN EXTRAS="--extra cv --extra web --extra studio --extra face --extra seg --extra pay"; \
    if [ "$VARIANT" = "gpu" ]; then EXTRAS="$EXTRAS --extra rfdetr --extra vlm --extra med"; fi; \
    uv pip install -r pyproject.toml $EXTRAS

# gpu: the Mac's sidecars too. litert-lm (local Gemma 4 brain) + a LocateAnything-3B env that
# reuses the system torch/cu126 but pins the transformers its remote code needs.
ENV LA3B_PYTHON=/opt/la3b/bin/python
RUN if [ "$VARIANT" = "gpu" ]; then \
      uv pip install litert-lm \
      && uv venv --system-site-packages --python /usr/local/bin/python3.12 /opt/la3b \
      && uv pip install --python /opt/la3b/bin/python "transformers==4.57.1"; \
    fi

# Demo clips (gitignored locally) + the default YOLO weights, baked in so a cold start
# needs no downloads.
RUN mkdir -p assets/videos && for f in vehicles-2.mp4 people-walking.mp4 market-square.mp4; do \
      curl -fsSL --retry 3 -o "assets/videos/$f" \
        "https://media.roboflow.com/supervision/video-examples/$f"; done

COPY . .

# Saved zones/lines live with the rest of the state on the out/ bucket mount.
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" \
    && mkdir -p out/studio \
    && ln -sf /app/out/studio/studio_annotations.json configs/studio_annotations.json \
    && useradd --uid 1000 --home-dir /app --no-create-home app \
    && chown -R app:app /app

ENV STUDIO_LOCAL_MODELS=${LOCAL_MODELS}

USER app
EXPOSE 8080

# Auth is ON by default (occ/studio/auth.py); agent-authored code is OFF (codepolicy.py).
# cpu: the studio alone. gpu: studio + LocateAnything-3B + litert-lm, like `run.py studio`.
CMD ["sh", "scripts/cloudrun_entry.sh"]
