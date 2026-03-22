#!/usr/bin/env python3
"""Extract frames from drone recording videos for training data.

Downloads recordings from Azure Blob Storage and extracts frames
at a configurable interval for labeling and training.

Usage:
  # Extract frames every 2 seconds from all recordings
  python extract_frames.py

  # Extract from a specific video at 1 FPS
  python extract_frames.py --video drone1_20260322_113638.mp4 --fps 1

  # Extract from a local file
  python extract_frames.py --local /path/to/video.mp4 --fps 0.5

  # Just list available recordings
  python extract_frames.py --list
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2


STORAGE_ACCOUNT = "guardianairevidstore"
CONTAINER_NAME = "recordings"
OUTPUT_DIR = Path(__file__).parent / "datasets" / "custom-thermal"


def list_recordings():
    """List available recordings in blob storage."""
    print("Fetching recordings from Azure Blob Storage...")
    result = subprocess.run(
        ["az", "storage", "blob", "list",
         "--container-name", CONTAINER_NAME,
         "--account-name", STORAGE_ACCOUNT,
         "--query", "[].{name:name, size:properties.contentLength, created:properties.creationTime}",
         "-o", "table"],
        capture_output=True, text=True
    )
    print(result.stdout)
    # Filter out test recordings (29094590 bytes = test sample)
    print("Note: Files with size 29094590 are test recordings (sample-drone.mp4)")


def download_recording(blob_name: str, dest_dir: Path) -> Path:
    """Download a recording from Azure Blob Storage."""
    dest = dest_dir / blob_name
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {blob_name}...")
    subprocess.run(
        ["az", "storage", "blob", "download",
         "--container-name", CONTAINER_NAME,
         "--account-name", STORAGE_ACCOUNT,
         "--name", blob_name,
         "--file", str(dest)],
        check=True, capture_output=True
    )
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Downloaded: {dest} ({size_mb:.1f} MB)")
    return dest


def extract_frames(video_path: Path, output_dir: Path, fps: float = 0.5):
    """Extract frames from a video at the given FPS rate.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (0.5 = 1 frame every 2 seconds)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    frame_interval = int(video_fps / fps)
    prefix = video_path.stem

    print(f"Video: {video_path.name}")
    print(f"  Duration: {duration:.1f}s, FPS: {video_fps:.0f}, Total frames: {total_frames}")
    print(f"  Extracting 1 frame every {1/fps:.1f}s ({fps} FPS)")
    print(f"  Expected output: ~{int(duration * fps)} frames")
    print(f"  Output dir: {output_dir}")
    print()

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        timestamp = frame_count / video_fps
        filename = f"{prefix}_t{timestamp:06.1f}s.jpg"
        filepath = output_dir / filename

        cv2.imwrite(str(filepath), frame)
        saved_count += 1

        if saved_count % 10 == 0:
            print(f"  Extracted {saved_count} frames (t={timestamp:.1f}s)...")

    cap.release()
    print(f"  Done! Extracted {saved_count} frames to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from drone recordings")
    parser.add_argument("--video", type=str, default=None,
                       help="Specific blob name to process (default: all real recordings)")
    parser.add_argument("--local", type=str, default=None,
                       help="Path to a local video file instead of downloading")
    parser.add_argument("--fps", type=float, default=0.5,
                       help="Frames per second to extract (default: 0.5 = every 2 seconds)")
    parser.add_argument("--list", action="store_true",
                       help="List available recordings and exit")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: datasets/custom-thermal/images)")
    args = parser.parse_args()

    if args.list:
        list_recordings()
        return

    output_dir = Path(args.output) if args.output else OUTPUT_DIR / "images" / "unlabeled"

    if args.local:
        video_path = Path(args.local)
        if not video_path.exists():
            print(f"ERROR: File not found: {video_path}")
            sys.exit(1)
        extract_frames(video_path, output_dir, args.fps)
        return

    # Download and process recordings from Azure
    downloads_dir = Path(__file__).parent / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    # Get list of recordings
    TEST_SIZE = 29094590  # Size of test sample recordings

    if args.video:
        blobs = [args.video]
    else:
        # List all blobs and filter out test recordings
        result = subprocess.run(
            ["az", "storage", "blob", "list",
             "--container-name", CONTAINER_NAME,
             "--account-name", STORAGE_ACCOUNT,
             "--query", "[?properties.contentLength != '29094590'].name",
             "-o", "tsv"],
            capture_output=True, text=True
        )
        blobs = [b.strip() for b in result.stdout.splitlines() if b.strip()]
        print(f"Found {len(blobs)} real drone recordings")

    total_frames = 0
    for blob_name in blobs:
        video_path = download_recording(blob_name, downloads_dir)
        total_frames += extract_frames(video_path, output_dir, args.fps)

    print(f"\n=== Summary ===")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Label frames with Roboflow (https://roboflow.com) or CVAT (https://cvat.ai)")
    print(f"  2. Export labels in YOLO format")
    print(f"  3. Combine with HIT-UAV dataset for training")


if __name__ == "__main__":
    main()
