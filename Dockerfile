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
    HF_HOME=/app/out/hf \
    HF_MODULES_CACHE=/tmp/hf_modules

# HF_MODULES_CACHE stays on local disk: transformers chmods the remote-code files it copies
# there, and the GCS FUSE bucket mount (HF_HOME) doesn't support chmod.
# ffmpeg: NLE/clip tools + video decode; libgl/glib/egl: OpenCV + MediaPipe on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libegl1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app

# Parity with the working local .venv: the SAME versions, never "whatever is newest at build
# time" (that drift broke Live Voice in production). torch/torchvision match the local venv,
# from the variant's wheel index (the CPU wheels skip ~2GB of CUDA libs).
ARG TORCH=2.12.0
ARG TORCHVISION=0.27.0
RUN if [ "$VARIANT" = "gpu" ]; then IDX=https://download.pytorch.org/whl/cu126; \
    else IDX=https://download.pytorch.org/whl/cpu; fi; \
    uv pip install --index-url "$IDX" "torch==$TORCH" "torchvision==$TORCHVISION"

# Everything else from requirements.lock (scripts/lock_requirements.sh freezes the local venv).
# --no-deps: replicate the frozen set EXACTLY. The working venv isn't strictly resolvable
# (e.g. opentelemetry-api 1.45.1 vs google-adk 2.2.0's declared <=1.41.1), so a fresh resolve
# would refuse — or "fix" it into a set that was never tested. The import check fails the
# build if the replica is broken.
COPY requirements.lock ./
RUN uv pip install --no-deps -r requirements.lock \
    && python -c "import cv2, mediapipe, ultralytics, supervision, trackers, google.adk, \
         google.genai, fastapi, firebase_admin, stripe, rembg; print('deps ok')"

# gpu: the Mac's sidecars too. litert-lm (local Gemma 4 brain) + a LocateAnything-3B env that
# reuses the system torch/cu126 but pins the transformers its remote code needs (plus lmdb/peft,
# which that remote code imports). --no-deps on purpose: resolving peft/transformers normally
# pulls a SECOND torch (CUDA 13) from PyPI that breaks torchvision ("torchvision::nms does not
# exist"); everything not listed here comes from the system site-packages.
ENV LA3B_PYTHON=/opt/la3b/bin/python
RUN if [ "$VARIANT" = "gpu" ]; then \
      uv pip install litert-lm \
      && uv venv --system-site-packages --python /usr/local/bin/python3.12 /opt/la3b \
      && uv pip install --python /opt/la3b/bin/python --no-deps "transformers==4.57.1" \
           "tokenizers==0.22.2" "huggingface-hub==0.36.2" "lmdb==2.2.1" "peft==0.19.1" \
      && /opt/la3b/bin/python -c "import torch, torchvision, transformers; \
           from torchvision.ops import nms; from transformers import AutoModel, AutoProcessor, AutoTokenizer; print('la3b env:', torch.__version__, \
           torchvision.__version__, transformers.__version__)"; \
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
