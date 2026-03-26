#!/usr/bin/env bash
# Test server-side detection locally against the live streaming container.
# Requires: trained model weights in this directory
#
# Usage:
#   ./test_server_detect.sh                          # default model
#   ./test_server_detect.sh thermal_person_best.pt   # custom model

set -euo pipefail

MODEL=${1:-"thermal_person_best.pt"}
RTMP_URL="rtmp://ga-drone-stream.eastus.azurecontainer.io/live/drone1"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo ""
    echo "Download from Colab training or specify path:"
    echo "  ./test_server_detect.sh /path/to/best.pt"
    exit 1
fi

echo "=== Server-Side Thermal Detection Test ==="
echo "Model: $MODEL"
echo "Stream: $RTMP_URL"
echo "FPS: 2"
echo ""
echo "Press Ctrl+C to stop"
echo ""

source .venv/bin/activate 2>/dev/null || true

python server_detect.py \
    --stream-url "$RTMP_URL" \
    --model "$MODEL" \
    --fps 2 \
    --conf 0.25 \
    --classes 0
