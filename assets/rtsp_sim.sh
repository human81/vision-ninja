#!/usr/bin/env bash
# Serve a local video file as a looping RTSP stream for FUNCTIONAL RTSP testing.
# Public RTSP test streams are unreliable; this is deterministic, offline, free.
#
# Setup (once):   brew install mediamtx
# Terminal A:     mediamtx                      # starts RTSP server on :8554
# Terminal B:     ./assets/rtsp_sim.sh assets/videos/vehicles.mp4 cam
# Then point the app at:  rtsp://localhost:8554/cam
set -euo pipefail

VIDEO="${1:-assets/videos/vehicles.mp4}"
PATH_NAME="${2:-cam}"
URL="rtsp://localhost:8554/${PATH_NAME}"

[[ -s "$VIDEO" ]] || { echo "Missing: $VIDEO (run assets/fetch_assets.sh)"; exit 1; }

echo "Publishing $VIDEO → $URL  (Ctrl-C to stop)"
# Re-encode to h264 + zerolatency so any codec/profile works over RTSP.
# -re = real-time pacing, -stream_loop -1 = loop forever.
exec ffmpeg -hide_banner -loglevel warning \
  -re -stream_loop -1 -i "$VIDEO" \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -g 30 \
  -an -f rtsp -rtsp_transport tcp "$URL"
