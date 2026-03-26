#!/usr/bin/env python3
"""Server-side thermal person detection on the RTMP video stream.

Reads frames from the HLS stream served by the nginx-rtmp container,
runs YOLO inference, and pushes detections to:
  1. SignalR broadcast → front-end mobile app
  2. MQTT (Event Grid) → Autel remote controller app
  3. Azure Blob Storage → annotated detection snapshots

Architecture:
  Drone → RTMP → nginx-rtmp container → HLS stream
                                           ↓
                              This script (frame extraction + YOLO)
                                           ↓
                              SignalR broadcast → mobile app
                              MQTT publish     → controller app
                              Blob snapshots   → review & training

Usage:
  # With default HLS URL and model
  python server_detect.py

  # Custom options
  python server_detect.py --hls-url http://localhost:8080/hls/drone1.m3u8 \
                          --model thermal_person_best.pt \
                          --fps 2 \
                          --conf 0.25

  # Disable MQTT (SignalR only)
  python server_detect.py --no-mqtt

  # Disable snapshots
  python server_detect.py --no-snapshots
"""

import argparse
import json
import logging
import os
import ssl
import time
import threading
import urllib.request
from datetime import datetime, timezone
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

# MQTT broker (Azure Event Grid)
MQTT_HOST = "tutorialeventgridnamespace.canadacentral-1.ts.eventgrid.azure.net"
MQTT_PORT = 8883
MQTT_CLIENT_ID = "client4-authnID"
MQTT_TOPIC = "api/status/custom-detections"

# Snapshot storage
SNAPSHOT_STORAGE_ACCOUNT = "guardianairevidstore"
SNAPSHOT_CONTAINER = "detections"

# Detection class names (must match training)
CLASS_NAMES = {0: "person", 1: "car", 2: "bicycle"}

# Colors (BGR) for bounding boxes on snapshots
BBOX_COLORS = {
    "person": (0, 255, 0),
    "car": (255, 165, 0),
    "bicycle": (255, 255, 0),
    "unknown": (128, 128, 128),
}


class MqttPublisher:
    """Publishes detections to Azure Event Grid MQTT broker with TLS client cert auth."""

    def __init__(self, host, port, client_id, cert_path, key_path):
        import paho.mqtt.client as mqtt

        self.topic = MQTT_TOPIC
        self.connected = False

        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        # TLS with client certificate — Event Grid requires username = client auth name
        self.client.username_pw_set(username=client_id)
        self.client.tls_set(
            certfile=cert_path,
            keyfile=key_path,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )

        log.info(f"MQTT connecting to {host}:{port} as {client_id}")
        self.client.connect_async(host, port)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            log.info("MQTT connected")
        else:
            log.warning(f"MQTT connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            log.warning(f"MQTT disconnected unexpectedly: rc={rc}")

    def publish(self, payload: dict):
        if not self.connected:
            return
        try:
            data = json.dumps(payload)
            self.client.publish(self.topic, data, qos=1)
            log.debug(f"MQTT published to {self.topic}")
        except Exception as e:
            log.warning(f"MQTT publish failed: {e}")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


class SnapshotUploader:
    """Uploads annotated detection frames to Azure Blob Storage.

    Throttles uploads to at most one every `interval` seconds to avoid
    flooding storage during sustained detection periods.
    """

    def __init__(self, connection_string: str, container_name: str,
                 interval: float = 5.0, min_confidence: float = 0.50):
        from azure.storage.blob import BlobServiceClient, ContentSettings

        self.container_name = container_name
        self.interval = interval
        self.min_confidence = min_confidence
        self.last_upload_time = 0.0
        self.upload_count = 0
        self._content_settings = ContentSettings(content_type="image/png")

        # Each detection session gets its own folder (timestamped at start)
        self.session_prefix = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")

        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        # Ensure container exists
        try:
            self.blob_service.create_container(container_name)
            log.info(f"Created blob container: {container_name}")
        except Exception:
            pass  # Already exists

        log.info(f"Snapshots: container={container_name}/{self.session_prefix}/, "
                 f"interval={interval}s, min_confidence={min_confidence:.0%}")

    def maybe_upload(self, frame, raw_detections, frame_count):
        """Upload an annotated snapshot if enough time has passed and confidence is high enough.

        Args:
            frame: Original BGR frame from the stream
            raw_detections: List of dicts with class_id, confidence, bbox (pixel coords)
            frame_count: Current frame number
        """
        # Filter to high-confidence detections
        high_conf = [d for d in raw_detections if d["confidence"] >= self.min_confidence]
        if not high_conf:
            return

        now = time.monotonic()
        if now - self.last_upload_time < self.interval:
            return

        self.last_upload_time = now

        # Draw bounding boxes on frame copy
        annotated = draw_detection_boxes(frame, high_conf)

        # Encode as PNG
        _, png_buf = cv2.imencode(".png", annotated)

        # Upload in background thread to avoid blocking detection loop
        ts = datetime.now(timezone.utc)
        best = max(high_conf, key=lambda d: d["confidence"])
        blob_name = (f"{self.session_prefix}/"
                     f"det_{ts.strftime('%H%M%S')}_"
                     f"{best['class_name']}_{best['confidence']:.0%}.png")

        thread = threading.Thread(
            target=self._upload_blob,
            args=(blob_name, png_buf.tobytes()),
            daemon=True,
        )
        thread.start()
        self.upload_count += 1

    def _upload_blob(self, blob_name, data):
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container_name, blob=blob_name)
            blob_client.upload_blob(data, overwrite=True,
                                    content_settings=self._content_settings)
            log.info(f"Snapshot uploaded: {self.container_name}/{blob_name}")
        except Exception as e:
            log.warning(f"Snapshot upload failed: {e}")


def draw_detection_boxes(frame, detections):
    """Draw bounding boxes with confidence labels on a frame copy.

    Args:
        frame: BGR numpy array
        detections: List of dicts with class_name, confidence, bbox (x1,y1,x2,y2 pixels)
    """
    annotated = frame.copy()
    for det in detections:
        cls_name = det["class_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = BBOX_COLORS.get(cls_name, BBOX_COLORS["unknown"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y - 2), font, font_scale, (0, 0, 0), thickness)

    return annotated


def send_detections_signalr(detections: list, broadcast_url: str):
    """Send detections to the SignalR broadcast endpoint."""
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
        log.warning(f"Failed to send detections to SignalR: {e}")


def send_detections_mqtt(detections: list, mqtt_pub: "MqttPublisher"):
    """Send detections to MQTT in the same format as the controller app."""
    if not detections or not mqtt_pub:
        return

    # Format matching publishCustomDetectionsToMqtt() in MainActivity.kt
    det_list = []
    for det in detections:
        det_list.append({
            "id": det["id"],
            "timestamp": str(int(time.time() * 1000)),
            "confidence": str(det["confidence"]),
            "bbox_x1": str(det["bbox"]["x1"]),
            "bbox_y1": str(det["bbox"]["y1"]),
            "bbox_x2": str(det["bbox"]["x2"]),
            "bbox_y2": str(det["bbox"]["y2"]),
            "source": "server",
        })

    payload = {"detections": det_list}
    mqtt_pub.publish(payload)


def format_detection(cls_id: int, confidence: float, bbox: list,
                     frame_w: int, frame_h: int) -> dict:
    """Format a YOLO detection as an SA update matching the mobile app protocol.

    Bounding boxes are normalized to 0-1. Each receiver (front-end, controller)
    has drone GPS and can project to lat/lon locally.
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

    # Set up MQTT publisher
    mqtt_pub = None
    if not args.no_mqtt:
        try:
            mqtt_pub = MqttPublisher(
                host=args.mqtt_host,
                port=args.mqtt_port,
                client_id=args.mqtt_client_id,
                cert_path=args.mqtt_cert,
                key_path=args.mqtt_key,
            )
        except Exception as e:
            log.warning(f"MQTT setup failed (continuing with SignalR only): {e}")

    # Set up snapshot uploader
    snapshot_uploader = None
    if not args.no_snapshots:
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        if conn_str:
            try:
                snapshot_uploader = SnapshotUploader(
                    connection_string=conn_str,
                    container_name=args.snapshot_container,
                    interval=args.snapshot_interval,
                    min_confidence=args.snapshot_min_conf,
                )
            except Exception as e:
                log.warning(f"Snapshot setup failed (continuing without): {e}")
        else:
            log.warning("Snapshots enabled but AZURE_STORAGE_CONNECTION_STRING not set — skipping")

    stream_url = args.stream_url
    is_rtmp = stream_url.startswith("rtmp://")
    log.info(f"Connecting to {'RTMP' if is_rtmp else 'HLS'} stream: {stream_url}")
    log.info(f"Target FPS: {args.fps}, confidence threshold: {args.conf}")
    log.info(f"SignalR: {args.broadcast_url}")
    log.info(f"MQTT: {'enabled' if mqtt_pub else 'disabled'}")
    log.info(f"Snapshots: {'enabled' if snapshot_uploader else 'disabled'}")

    frame_interval = 1.0 / args.fps
    consecutive_failures = 0

    # Low-latency ffmpeg options for RTMP (minimize buffering)
    if is_rtmp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32768"

    try:
        while True:
            cap = cv2.VideoCapture(stream_url)
            if is_rtmp:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                consecutive_failures += 1
                log.warning(f"Cannot open stream (attempt {consecutive_failures}), retrying in 5s...")
                time.sleep(5)
                continue

            consecutive_failures = 0
            log.info(f"Connected to {'RTMP' if is_rtmp else 'HLS'} stream!")

            stream_fps = cap.get(cv2.CAP_PROP_FPS) or 24
            frame_skip = max(1, int(stream_fps / args.fps))
            frame_count = 0
            detection_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Lost stream, reconnecting...")
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
                detections = []      # Normalized (for SignalR/MQTT)
                raw_detections = []  # Pixel coords (for snapshots)
                for r in results:
                    h, w = r.orig_shape
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        det = format_detection(cls_id, conf, xyxy, w, h)
                        detections.append(det)
                        raw_detections.append({
                            "class_name": CLASS_NAMES.get(cls_id, "unknown"),
                            "confidence": conf,
                            "bbox": xyxy,
                        })

                if detections:
                    detection_count += len(detections)
                    send_detections_signalr(detections, args.broadcast_url)
                    send_detections_mqtt(detections, mqtt_pub)
                    if snapshot_uploader:
                        snapshot_uploader.maybe_upload(frame, raw_detections, frame_count)
                    log.info(f"Frame {frame_count}: {len(detections)} detections "
                            f"({inference_ms:.0f}ms inference)")
                    for det in detections:
                        b = det["bbox"]
                        log.info(f"  {det['type']} conf={det['confidence']:.3f} "
                                f"bbox=({b['x1']:.4f},{b['y1']:.4f},{b['x2']:.4f},{b['y2']:.4f})")
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
    finally:
        if mqtt_pub:
            mqtt_pub.stop()


def main():
    parser = argparse.ArgumentParser(description="Server-side thermal person detection")
    parser.add_argument("--stream-url",
                       default="rtmp://ga-drone-stream.eastus.azurecontainer.io/live/drone1",
                       help="Video stream URL (RTMP for low-latency, HLS for compatibility)")
    parser.add_argument("--hls-url",
                       default=None,
                       help="(deprecated) Alias for --stream-url")
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

    # MQTT options
    parser.add_argument("--no-mqtt", action="store_true",
                       help="Disable MQTT publishing (SignalR only)")
    parser.add_argument("--mqtt-host", default=MQTT_HOST,
                       help="MQTT broker hostname")
    parser.add_argument("--mqtt-port", type=int, default=MQTT_PORT,
                       help="MQTT broker port")
    parser.add_argument("--mqtt-client-id", default=MQTT_CLIENT_ID,
                       help="MQTT client ID")
    parser.add_argument("--mqtt-cert", default="/app/certs/client4.pem",
                       help="Path to MQTT client certificate")
    parser.add_argument("--mqtt-key", default="/app/certs/client4.key",
                       help="Path to MQTT client private key")

    # Snapshot options
    parser.add_argument("--no-snapshots", action="store_true",
                       help="Disable snapshot uploads to blob storage")
    parser.add_argument("--snapshot-container", default=SNAPSHOT_CONTAINER,
                       help=f"Blob container for snapshots (default: {SNAPSHOT_CONTAINER})")
    parser.add_argument("--snapshot-interval", type=float, default=5.0,
                       help="Min seconds between snapshot uploads (default: 5)")
    parser.add_argument("--snapshot-min-conf", type=float, default=0.50,
                       help="Min confidence to trigger snapshot (default: 0.50)")

    args = parser.parse_args()

    # Support deprecated --hls-url flag (overrides --stream-url if provided)
    if args.hls_url:
        args.stream_url = args.hls_url

    run_detection_loop(args)


if __name__ == "__main__":
    main()
