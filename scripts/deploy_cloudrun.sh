#!/usr/bin/env bash
# deploy_cloudrun.sh — build + deploy Vision Ninja Studio to Cloud Run.
#
#   scripts/deploy_cloudrun.sh cpu all        # setup → secrets → build → deploy (CPU)
#   scripts/deploy_cloudrun.sh gpu all        # same, NVIDIA L4 variant (separate service)
#   scripts/deploy_cloudrun.sh cpu <step>     # one step: setup | secrets | build | deploy | domain
#
# What it creates (project $PROJECT, region $REGION):
#   service      vision-ninja (cpu) / vision-ninja-gpu (gpu) — ONE always-on instance
#                (the studio holds one live pipeline + agent in process memory)
#   image        $REGION-docker.pkg.dev/$PROJECT/studio/vision-ninja:{cpu,gpu}
#   bucket       gs://$PROJECT-<service>-out, mounted at /app/out (recordings, library,
#                settings, saved zones, model cache) — the container disk is ephemeral
#   identity     vision-ninja@$PROJECT.iam.gserviceaccount.com (Firebase auth admin,
#                secret accessor, object admin on its bucket)
#   secrets      GEMINI_API_KEY / OPENAI_API_KEY / STRIPE_SECRET_KEY / HF_TOKEN copied from .env
#
# Access: the service is publicly REACHABLE (--no-invoker-iam-check: works under the org's
# domain-restricted-sharing policy, which forbids an allUsers grant), but the app requires
# Firebase sign-in
# (occ/studio/auth.py) for everything; STUDIO_AUTH_ALLOW comes from .env. Agent-authored
# code stays OFF (STUDIO_AGENT_CODE is never set here).
#
# Needs: gcloud logged in (`gcloud auth login`) as an owner of $PROJECT, billing enabled;
# the GPU variant also needs Cloud Run L4 quota in $REGION.
#
# RUN_PROJECT=<id> runs the SERVICE in another project (e.g. one that has L4 quota) while the
# image, secrets, bucket and Firebase sign-in stay in $PROJECT:
#   RUN_PROJECT=serious-glyph scripts/deploy_cloudrun.sh gpu setup   (then: deploy)
set -euo pipefail

VARIANT="${1:-}"; STEP="${2:-all}"
PROJECT="${PROJECT:-vision-b5c97}"
RUN_PROJECT="${RUN_PROJECT:-$PROJECT}"
REGION="${REGION:-us-central1}"
case "$VARIANT" in
  cpu) SERVICE=vision-ninja;     LOCAL_MODELS=off ;;
  gpu) SERVICE=vision-ninja-gpu; LOCAL_MODELS=on ;;
  *)   sed -n '2,8p' "$0"; exit 2 ;;
esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/studio/vision-ninja:$VARIANT"
BUCKET="$PROJECT-$SERVICE-out"
SA="vision-ninja@$RUN_PROJECT.iam.gserviceaccount.com"
SECRETS=(GEMINI_API_KEY OPENAI_API_KEY STRIPE_SECRET_KEY HF_TOKEN)   # HF_TOKEN: gated Gemma/MedGemma weights
G=(gcloud --project "$PROJECT" --quiet)        # image, secrets, bucket, Firebase
GR=(gcloud --project "$RUN_PROJECT" --quiet)   # the Cloud Run service + its identity

envval() {   # value of KEY in .env (never echoed)
  [[ -f .env ]] && grep -E "^$1=" .env | tail -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//' || true
}

setup() {
  echo "▸ enabling APIs"
  "${G[@]}" services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com secretmanager.googleapis.com storage.googleapis.com \
    identitytoolkit.googleapis.com
  echo "▸ Artifact Registry repo 'studio'"
  "${G[@]}" artifacts repositories describe studio --location "$REGION" >/dev/null 2>&1 ||
    "${G[@]}" artifacts repositories create studio --location "$REGION" --repository-format docker
  echo "▸ bucket gs://$BUCKET"
  "${G[@]}" storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 ||
    "${G[@]}" storage buckets create "gs://$BUCKET" --location "$REGION" \
      --uniform-bucket-level-access --public-access-prevention
  if [[ "$RUN_PROJECT" != "$PROJECT" ]]; then
    "${GR[@]}" services enable run.googleapis.com iam.googleapis.com
    # its Cloud Run service agent must pull the image from $PROJECT's registry
    local rnum; rnum=$("${GR[@]}" projects describe "$RUN_PROJECT" --format 'value(projectNumber)')
    "${G[@]}" projects add-iam-policy-binding "$PROJECT" --role roles/artifactregistry.reader \
      --member "serviceAccount:service-$rnum@serverless-robot-prod.iam.gserviceaccount.com" \
      --condition None >/dev/null
  fi
  echo "▸ service account $SA"
  "${GR[@]}" iam service-accounts describe "$SA" >/dev/null 2>&1 ||
    "${GR[@]}" iam service-accounts create vision-ninja --display-name "Vision Ninja Studio"
  for role in roles/firebaseauth.admin roles/secretmanager.secretAccessor; do
    "${G[@]}" projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" \
      --role "$role" --condition None >/dev/null
  done
  "${G[@]}" storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member "serviceAccount:$SA" --role roles/storage.objectAdmin >/dev/null
  # Cloud Build runs as the Compute default SA on newer projects: let it push + log.
  local num; num=$("${G[@]}" projects describe "$PROJECT" --format 'value(projectNumber)')
  for role in roles/artifactregistry.writer roles/logging.logWriter roles/storage.objectViewer; do
    "${G[@]}" projects add-iam-policy-binding "$PROJECT" \
      --member "serviceAccount:$num-compute@developer.gserviceaccount.com" \
      --role "$role" --condition None >/dev/null
  done
}

secrets() {
  for name in "${SECRETS[@]}"; do
    local v; v=$(envval "$name")
    if [[ -z "$v" ]]; then echo "  · $name not in .env — skipped"; continue; fi
    "${G[@]}" secrets describe "$name" >/dev/null 2>&1 ||
      "${G[@]}" secrets create "$name" --replication-policy automatic >/dev/null
    printf %s "$v" | "${G[@]}" secrets versions add "$name" --data-file=- >/dev/null
    echo "  ✓ $name → Secret Manager (new version)"
  done
  echo "  · STRIPE_WEBHOOK_SECRET is per-endpoint: create a Stripe webhook for"
  echo "    <service-url>/stripe/webhook, then: gcloud secrets create STRIPE_WEBHOOK_SECRET …"
  echo "    (until then the webhook answers 503 — orders aren't auto-marked paid)"
}

build() {
  echo "▸ Cloud Build → $IMAGE (amd64; several minutes)"
  "${G[@]}" builds submit --config cloudbuild.yaml \
    --substitutions "_VARIANT=$VARIANT,_LOCAL_MODELS=$LOCAL_MODELS,_IMAGE=$IMAGE"
}

deploy() {
  local allow; allow=$(envval STUDIO_AUTH_ALLOW)
  [[ -n "$allow" ]] || { echo "✗ STUDIO_AUTH_ALLOW is empty in .env — nobody could sign in"; exit 1; }
  local env="STUDIO_AUTH_ALLOW=$allow|GOOGLE_GENAI_USE_VERTEXAI=0"
  for k in API_KEY AUTH_DOMAIN PROJECT_ID APP_ID; do
    env="$env|STUDIO_FIREBASE_$k=$(envval "STUDIO_FIREBASE_$k")"
  done
  local secrets_flag=() list="" ref=""
  if [[ "$RUN_PROJECT" != "$PROJECT" ]]; then      # cross-project secret references
    ref="projects/$("${G[@]}" projects describe "$PROJECT" --format 'value(projectNumber)')/secrets/"
  fi
  for name in "${SECRETS[@]}" STRIPE_WEBHOOK_SECRET; do
    "${G[@]}" secrets describe "$name" >/dev/null 2>&1 && list="${list:+$list,}$name=$ref$name:latest"
  done
  [[ -n "$list" ]] && secrets_flag=(--set-secrets "$list")
  local size=()
  if [[ "$VARIANT" == gpu ]]; then
    # Weights are pre-seeded into the bucket (scripts/seed_models.sh) → cache-only: no HF
    # rate limits (429 from Cloud Run's shared IPs) and no gated-model token needed.
    env="$env|OCC_SET=detector.device=cuda|HF_HUB_OFFLINE=1|TRANSFORMERS_OFFLINE=1|STUDIO_PRELOAD=gemma-3-4b-it"
    size=(--cpu 8 --memory 32Gi --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy)
  else
    # CPU levers: nano model, detect every 2nd frame (tracker predicts between), ≤1280px.
    env="$env|OCC_SET=detector.device=cpu;detector.model=yolo26n.pt;runtime.detect_every=2;source.max_long_side=1280|OCC_TORCH_THREADS=3"
    size=(--cpu 4 --memory 8Gi)
  fi
  # liveness: /healthz is a sync route (threadpool) behind the auth middleware, so a frozen
  # loop or an exhausted threadpool fails it and Cloud Run replaces the instance in ~30s.
  # concurrency 1000: MJPEG streams + WebSockets are long-lived and each holds a request slot;
  # at 80 the single instance ran out and Cloud Run refused everything (BIDI included).
  echo "▸ deploying $SERVICE ($VARIANT) in $RUN_PROJECT"
  "${GR[@]}" run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" \
    --service-account "$SA" --execution-environment gen2 \
    --no-invoker-iam-check \
    --min-instances 1 --max-instances 1 --no-cpu-throttling \
    --concurrency 1000 --timeout 3600 --port 8080 "${size[@]}" \
    --liveness-probe "httpGet.path=/healthz,httpGet.port=8080,periodSeconds=10,timeoutSeconds=5,failureThreshold=3" \
    --add-volume "name=out,type=cloud-storage,bucket=$BUCKET" \
    --add-volume-mount "volume=out,mount-path=/app/out" \
    --set-env-vars "^|^$env" ${secrets_flag[@]+"${secrets_flag[@]}"}
  local url; url=$(urls | head -1)
  # Stripe redirects back here after checkout (commerce.py defaults to localhost).
  "${GR[@]}" run services update "$SERVICE" --region "$REGION" \
    --update-env-vars "STUDIO_PUBLIC_URL=$url" >/dev/null
  echo "✓ $SERVICE → $url"
  domain
}

urls() {     # every URL Cloud Run serves the service on (deterministic one first)
  "${GR[@]}" run services describe "$SERVICE" --region "$REGION" --format json | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("\n".join(json.loads(d["metadata"]["annotations"].get("run.googleapis.com/urls", "[]"))
                or [d["status"]["url"]]))'
}

domain() {   # allow Google sign-in popups on every service URL (Firebase authorized domains)
  local hosts; hosts=$(urls | sed 's|^https://||' | tr '\n' ' ')
  local api="https://identitytoolkit.googleapis.com/admin/v2/projects/$PROJECT/config"
  local tok; tok=$(gcloud auth print-access-token)
  local cur; cur=$(curl -fsS -H "Authorization: Bearer $tok" -H "x-goog-user-project: $PROJECT" "$api")
  local body; body=$(HOSTS="$hosts" python3 -c '
import json, os, sys
d = json.load(sys.stdin).get("authorizedDomains", [])
print(json.dumps({"authorizedDomains": d + [h for h in os.environ["HOSTS"].split() if h not in d]}))' <<<"$cur")
  curl -fsS -X PATCH -H "Authorization: Bearer $tok" -H "x-goog-user-project: $PROJECT" \
    -H "Content-Type: application/json" "$api?updateMask=authorizedDomains" -d "$body" >/dev/null
  echo "✓ Firebase authorized domains: $hosts"
}

case "$STEP" in
  all)    setup; secrets; build; deploy ;;
  setup|secrets|build|deploy|domain) "$STEP" ;;
  *)      sed -n '2,8p' "$0"; exit 2 ;;
esac
