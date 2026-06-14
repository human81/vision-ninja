#!/usr/bin/env bash
# Download the verified, diverse test-video set into assets/videos/.
# All sources probed live + decodable (HTTP 200, valid h264) on 2026-06-14.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/videos"
BASE="https://media.roboflow.com/supervision/video-examples"
mkdir -p "$DIR"

# name                scene / specs
FILES=(
  "vehicles.mp4"        # 4K 25fps  highway vehicles
  "vehicles-2.mp4"      # 1080p 30  road vehicles
  "people-walking.mp4"  # 1080p 25  pedestrian street
  "market-square.mp4"   # 4K 60 vertical  crowded plaza (people + some vehicles)
  "grocery-store.mp4"   # 4K 30  indoor retail (people)
  "subway.mp4"          # 4K 30 vertical  transit (people)
)

for f in "${FILES[@]}"; do
  if [[ -s "$DIR/$f" ]]; then
    echo "✓ $f (already present)"
  else
    echo "↓ $f"
    curl -fSL --retry 3 -o "$DIR/$f" "$BASE/$f"
  fi
done
echo "Done → $DIR"
