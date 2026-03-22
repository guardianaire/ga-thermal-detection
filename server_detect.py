#!/usr/bin/env python3
"""Server-side thermal person detection on the RTMP video stream.

Reads frames from the HLS stream served by the nginx-rtmp container,
runs YOLO inference, and pushes detections to the SA pipeline via
the SignalR broadcast endpoint.

Architecture:
  Drone → RTMP → nginx-rtmp container → HLS stream
                                           ↓
                              This script (frame extraction + YOLO)
                                           ↓
                              SignalR broadcast → mobile app

Usage:
  # With default HLS URL and model
  python server_detect.py

  # Custom options
  python server_detect.py --hls-url http://localhost:8080/hls/drone1.m3u8 \
                          --model thermal_person_best.pt \
                          --fps 2 \
                          --conf 0.25

  # Use TFLite model (lighter, for CPU inference)
  python server_detect.py --model thermal_person_int8.tflite --fps 1
"""

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# SignalR broadcast endpoint (same one used by MyTutorialFunctionApp02)
SIGNALR_BROADCAST_URL = "https://signalrconnectorapp.azurewebsites.net/api/broadcast"

# Detection class names (must match training)
CLASS_NAMES = {0: "person", 1: "car", 2: "bicycle"}


def send_detections(detections: list, broadcast_url: str):
    """Send detections to the SignalR broadcast endpoint.

    Formats detections as SituationAwarenessUpdate messages
    matching the existing mobile app protocol.
    """
    if not detections:
        return

    payload = {
        "target": "SituationAwarenessUpdate",
        "arguments": [detections]
    }

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            broadcast_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
        log.debug(f"Sent {len(detections)} detections to SignalR")
    except Exception as e:
        log.warning(f"Failed to send detections: {e}")


def format_detection(cls_id: int, confidence: float, bbox: list,
                     frame_w: int, frame_h: int) -> dict:
    """Format a YOLO detection as an SA update matching the mobile app protocol.

    Since we don't have the user's GPS position on the server, we send
    pixel-space bounding box info. The mobile app or a downstream service
    can project these to geographic coordinates if needed.
    """
    cls_name = CLASS_NAMES.get(cls_id, "unknown")
    x1, y1, x2, y2 = bbox

    # Normalize bbox to 0-1 range
    cx = ((x1 + x2) / 2) / frame_w
    cy = ((y1 + y2) / 2) / frame_h

    return {
        "id": f"server-{cls_name}-{int(cx*1000)}-{int(cy*1000)}",
        "type": cls_name,
        "count": 1,
        "confidence": round(confidence, 3),
        "bbox": {
            "x1": round(x1 / frame_w, 4),
            "y1": round(y1 / frame_h, 4),
            "x2": round(x2 / frame_w, 4),
            "y2": round(y2 / frame_h, 4),
        },
        "source": "server",
        "threatLevel": "monitor",
        "direction": "unknown",
        "distance": 0,
        "bearing": 0,
    }


def run_detection_loop(args):
    """Main detection loop: read frames from HLS, run YOLO, send results."""

    from ultralytics import YOLO

    log.info(f"Loading model: {args.model}")
    model = YOLO(args.model)

    log.info(f"Connecting to HLS stream: {args.hls_url}")
    log.info(f"Target FPS: {args.fps}, confidence threshold: {args.conf}")
    log.info(f"Broadcast URL: {args.broadcast_url}")

    frame_interval = 1.0 / args.fps
    consecutive_failures = 0
    max_failures = 30  # Give up after 30 consecutive failures

    while True:
        cap = cv2.VideoCapture(args.hls_url)

        if not cap.isOpened():
            consecutive_failures += 1
            if consecutive_failures > max_failures:
                log.error("Too many consecutive failures, exiting")
                break
            log.warning(f"Cannot open HLS stream (attempt {consecutive_failures}/{max_failures}), retrying in 5s...")
            time.sleep(5)
            continue

        consecutive_failures = 0
        log.info("Connected to HLS stream!")

        stream_fps = cap.get(cv2.CAP_PROP_FPS) or 24
        frame_skip = max(1, int(stream_fps / args.fps))
        frame_count = 0
        detection_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Lost HLS stream, reconnecting...")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            t0 = time.perf_counter()

            # Run inference
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=args.classes,
                verbose=False,
            )

            inference_ms = (time.perf_counter() - t0) * 1000

            # Extract detections
            detections = []
            for r in results:
                h, w = r.orig_shape
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    det = format_detection(cls_id, conf, xyxy, w, h)
                    detections.append(det)

            if detections:
                detection_count += len(detections)
                send_detections(detections, args.broadcast_url)
                log.info(f"Frame {frame_count}: {len(detections)} detections "
                        f"({inference_ms:.0f}ms inference)")
            elif frame_count % (frame_skip * 10) == 0:
                # Log periodically even when no detections
                log.debug(f"Frame {frame_count}: no detections ({inference_ms:.0f}ms)")

            # Throttle to target FPS
            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        log.info(f"Stream session ended. Frames: {frame_count}, Detections: {detection_count}")
        log.info("Reconnecting in 3s...")
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="Server-side thermal person detection")
    parser.add_argument("--hls-url",
                       default="http://ga-drone-stream.eastus.azurecontainer.io:8080/hls/drone1.m3u8",
                       help="HLS stream URL")
    parser.add_argument("--model", default="thermal_person_best.pt",
                       help="YOLO model path (.pt or .tflite)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Inference image size")
    parser.add_argument("--fps", type=float, default=2.0,
                       help="Target detection FPS (default: 2)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                       help="Class IDs to detect (default: 0 = Person only)")
    parser.add_argument("--broadcast-url", default=SIGNALR_BROADCAST_URL,
                       help="SignalR broadcast endpoint URL")
    args = parser.parse_args()

    run_detection_loop(args)


if __name__ == "__main__":
    main()
