#!/usr/bin/env python3
"""Replay a drone recording with YOLO detection and produce an annotated video.

Reads frames from a local video, runs thermal person detection, draws
bounding boxes with confidence labels on frames that have detections,
and writes an output video containing only the detection frames.

Usage:
  python replay_detect.py downloads/drone1_20260324_101311.mp4

  # Custom options
  python replay_detect.py downloads/drone1_20260324_101311.mp4 \
      --model weights/thermal_custom_best.pt \
      --fps 2 --conf 0.25 \
      --output runs/replay/

  # Also save individual snapshot PNGs
  python replay_detect.py downloads/drone1_20260324_101311.mp4 --snapshots
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "person", 1: "car", 2: "bicycle"}

# Colors (BGR) for bounding boxes
COLORS = {
    "person": (0, 255, 0),    # green
    "car": (255, 165, 0),     # orange
    "bicycle": (255, 255, 0), # cyan
    "unknown": (128, 128, 128),
}


def draw_detections(frame, detections):
    """Draw bounding boxes and confidence labels on a frame.

    Args:
        frame: BGR numpy array
        detections: list of dicts with keys: class_name, confidence, bbox (x1,y1,x2,y2 in pixels)

    Returns:
        Annotated frame (copy)
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        cls_name = det["class_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]

        color = COLORS.get(cls_name, COLORS["unknown"])

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label with confidence
        label = f"{cls_name} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background rectangle for label
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y - 2), font, font_scale, (0, 0, 0), thickness)

    return annotated


def add_frame_info(frame, frame_num, timestamp, det_count, fps_actual=None):
    """Add frame info overlay to bottom-left corner."""
    h, w = frame.shape[:2]
    info = f"Frame {frame_num} | t={timestamp:.1f}s | {det_count} detection(s)"
    if fps_actual is not None:
        info += f" | {fps_actual:.1f} FPS"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    (tw, th), _ = cv2.getTextSize(info, font, font_scale, thickness)

    # Semi-transparent background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - th - 12), (tw + 12, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, info, (6, h - 6), font, font_scale, (200, 200, 200), thickness)
    return frame


def replay_detect(args):
    """Main replay loop."""
    from ultralytics import YOLO

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Set up output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    if args.snapshots:
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps

    frame_skip = max(1, int(video_fps / args.fps))

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {frame_w}x{frame_h}, FPS: {video_fps:.1f}, Duration: {duration:.1f}s")
    print(f"  Detection rate: {args.fps} FPS (every {frame_skip} frames)")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Model: {args.model}")
    print(f"  Output: {output_dir}")
    print()

    # Set up video writer — use H.264 for macOS QuickTime compatibility
    output_video = output_dir / f"{video_path.stem}_detections.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_fps = args.fps  # playback at detection rate
    writer = cv2.VideoWriter(str(output_video), fourcc, out_fps, (frame_w, frame_h))

    # Also prepare an "all analyzed frames" video (includes frames with no detections)
    output_video_all = output_dir / f"{video_path.stem}_all_analyzed.mp4"
    writer_all = cv2.VideoWriter(str(output_video_all), fourcc, out_fps, (frame_w, frame_h))

    # Detection log
    detection_log = []

    frame_count = 0
    analyzed_count = 0
    detection_frame_count = 0
    total_detections = 0
    t_start = time.perf_counter()

    print("Running detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        analyzed_count += 1
        timestamp = frame_count / video_fps

        # Run inference
        t0 = time.perf_counter()
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
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_name = CLASS_NAMES.get(cls_id, "unknown")
                detections.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                })

        # Annotate frame (always for the "all" video)
        annotated = draw_detections(frame, detections) if detections else frame.copy()
        annotated = add_frame_info(annotated, frame_count, timestamp, len(detections))
        writer_all.write(annotated)

        if detections:
            detection_frame_count += 1
            total_detections += len(detections)

            # Write to detections-only video
            writer.write(annotated)

            # Save snapshot PNG
            if args.snapshots:
                snap_path = snapshots_dir / f"det_{frame_count:06d}_t{timestamp:.1f}s.png"
                cv2.imwrite(str(snap_path), annotated)

            # Log
            det_summary = ", ".join(
                f"{d['class_name']} {d['confidence']:.0%}" for d in detections
            )
            print(f"  [{timestamp:6.1f}s] Frame {frame_count}: {len(detections)} detection(s) — {det_summary} ({inference_ms:.0f}ms)")

            # Record to log
            for d in detections:
                detection_log.append({
                    "frame": frame_count,
                    "timestamp": round(timestamp, 2),
                    "class": d["class_name"],
                    "confidence": round(d["confidence"], 4),
                    "bbox_x1": d["bbox"][0],
                    "bbox_y1": d["bbox"][1],
                    "bbox_x2": d["bbox"][2],
                    "bbox_y2": d["bbox"][3],
                    "inference_ms": round(inference_ms, 1),
                })

        # Progress update every 50 analyzed frames
        if analyzed_count % 50 == 0:
            elapsed = time.perf_counter() - t_start
            pct = frame_count / total_frames * 100
            print(f"  ... {pct:.0f}% complete ({analyzed_count} frames analyzed, "
                  f"{detection_frame_count} with detections, {elapsed:.0f}s elapsed)")

    cap.release()
    writer.release()
    writer_all.release()

    elapsed_total = time.perf_counter() - t_start

    # Save detection log as JSON
    log_path = output_dir / f"{video_path.stem}_detections.json"
    with open(log_path, "w") as f:
        json.dump({
            "video": video_path.name,
            "model": args.model,
            "fps": args.fps,
            "conf_threshold": args.conf,
            "video_duration_s": round(duration, 1),
            "frames_analyzed": analyzed_count,
            "frames_with_detections": detection_frame_count,
            "total_detections": total_detections,
            "processing_time_s": round(elapsed_total, 1),
            "detections": detection_log,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Video duration:          {duration:.1f}s")
    print(f"  Frames analyzed:         {analyzed_count}")
    print(f"  Frames with detections:  {detection_frame_count}")
    print(f"  Total detections:        {total_detections}")
    print(f"  Processing time:         {elapsed_total:.1f}s")
    print(f"  Avg inference:           {elapsed_total/max(analyzed_count,1)*1000:.0f}ms/frame")
    print()
    print(f"  Output video (detections only): {output_video}")
    print(f"  Output video (all frames):      {output_video_all}")
    print(f"  Detection log:                  {log_path}")
    if args.snapshots:
        print(f"  Snapshots:                      {snapshots_dir}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Replay drone video with YOLO detection overlay")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="weights/thermal_custom_best.pt",
                       help="YOLO model path (default: weights/thermal_custom_best.pt)")
    parser.add_argument("--fps", type=float, default=2.0,
                       help="Detection FPS (default: 2)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Inference image size (default: 640)")
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                       help="Class IDs to detect (default: 0 = person)")
    parser.add_argument("--output", default="runs/replay",
                       help="Output directory (default: runs/replay)")
    parser.add_argument("--snapshots", action="store_true",
                       help="Also save individual detection frames as PNGs")

    args = parser.parse_args()
    replay_detect(args)


if __name__ == "__main__":
    main()
