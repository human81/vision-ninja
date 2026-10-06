#!/usr/bin/env bash
# seed_models.sh — copy the local model weights this Mac already has into the GPU service's
# bucket, so Cloud Run runs the SAME weights cache-only (HF_HUB_OFFLINE=1): no Hugging Face
# rate limits (Cloud Run's shared IPs get 429s) and no gated-model token at runtime.
#
#   scripts/seed_models.sh                       # → gs://vision-b5c97-vision-ninja-gpu-out
#   BUCKET=gs://other-bucket scripts/seed_models.sh
#
# The HF cache keeps snapshot files as symlinks into blobs/; the bucket (GCS FUSE) wants real
# files, so each snapshot is staged as hard links (no extra disk) in the hub layout the
# loaders expect: hub/models--org--name/{refs/main, snapshots/<rev>/…}.
set -euo pipefail

BUCKET="${BUCKET:-gs://vision-b5c97-vision-ninja-gpu-out}"
PROJECT="${PROJECT:-vision-b5c97}"
STAGE="${STAGE:-$(mktemp -d)/models_stage}"
MODELS=(
  models--nvidia--LocateAnything-3B          # 🔍 locate sidecar
  models--google--gemma-3-4b-it              # local vision
  models--google--medgemma-4b-it             # /medical
  models--google--owlv2-base-patch16-ensemble
  models--Roboflow--rf-detr-nano
  models--Roboflow--rf-detr-medium
)
LITERT=gemma-4-12b-it                        # local Gemma 4 brain (litert-lm)

mkdir -p "$STAGE"
python3 - "$STAGE" "$LITERT" "${MODELS[@]}" <<'EOF'
import os, sys
stage, litert, models = sys.argv[1], sys.argv[2], sys.argv[3:]
hub = os.path.expanduser("~/.cache/huggingface/hub")
for m in models:
    src = os.path.join(hub, m)
    if not os.path.isdir(src):
        print(f"  · {m} not cached locally — skipped"); continue
    rev = open(os.path.join(src, "refs", "main")).read().strip()
    out = os.path.join(stage, "hf/hub", m)
    os.makedirs(os.path.join(out, "refs"), exist_ok=True)
    open(os.path.join(out, "refs", "main"), "w").write(rev)
    snap = os.path.join(src, "snapshots", rev)
    for root, _, files in os.walk(snap):
        for f in files:
            p = os.path.join(root, f)
            dst = os.path.join(out, "snapshots", rev, os.path.relpath(p, snap))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                os.link(os.path.realpath(p), dst)
    print(f"  ✓ staged {m} @ {rev[:8]}")
lm = os.path.expanduser(f"~/.litert-lm/models/{litert}/model.litertlm")
if os.path.exists(lm):        # the mldrift *_cache.bin files are Mac-GPU specific — skipped
    d = os.path.join(stage, "litert-lm/models", litert); os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "model.litertlm")):
        os.link(lm, os.path.join(d, "model.litertlm"))
    print(f"  ✓ staged litert {litert}")
EOF

for d in "$STAGE"/hf/hub/* "$STAGE/litert-lm"; do
  [[ -e "$d" ]] || continue
  rel="${d#"$STAGE"/}"
  echo "▸ $rel"
  gcloud storage rsync -r "$d" "$BUCKET/$rel" --project "$PROJECT"
done
echo "✓ seeded $BUCKET — redeploy (or restart) the GPU service to pick the weights up"
