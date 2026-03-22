#!/usr/bin/env python3
"""Download and prepare thermal person detection datasets for YOLO training.

Datasets:
  - HIT-UAV: High-altitude infrared thermal from drones (primary)
  - FLIR ADAS: Large thermal dataset (supplementary pretraining)

Usage:
  python download_datasets.py --dataset hit-uav
  python download_datasets.py --dataset flir
  python download_datasets.py --dataset all
"""

import argparse
import os
import subprocess
import sys
import zipfile
import shutil
from pathlib import Path


DATASETS_DIR = Path(__file__).parent / "datasets"


def download_hit_uav():
    """Download HIT-UAV high-altitude infrared thermal dataset.

    Paper: https://www.nature.com/articles/s41597-023-02066-6
    Contains 2,898 thermal images from UAVs with person/vehicle/bicycle annotations.
    Already in YOLO format.
    """
    dest = DATASETS_DIR / "hit-uav"
    if dest.exists():
        print(f"HIT-UAV already exists at {dest}")
        return dest

    print("Downloading HIT-UAV dataset...")
    dest.mkdir(parents=True, exist_ok=True)

    # Clone the HIT-UAV repo (contains images + YOLO annotations)
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/suojeong/HIT-UAV.git",
        str(dest / "raw")
    ], check=True)

    print(f"HIT-UAV downloaded to {dest}")
    return dest


def download_flir():
    """Download FLIR ADAS thermal dataset from Roboflow.

    Contains 10,228 thermal images with person/car/bicycle/dog annotations.
    Available in YOLO format via Roboflow.
    """
    dest = DATASETS_DIR / "flir-adas"
    if dest.exists():
        print(f"FLIR ADAS already exists at {dest}")
        return dest

    print("Downloading FLIR ADAS dataset from Roboflow...")
    print("Note: You may need a Roboflow API key for large downloads.")
    print("Alternative: Visit https://universe.roboflow.com/thermal-imaging-0hwfw/flir-data-set")
    print("and download in YOLOv8 format manually to datasets/flir-adas/")

    dest.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
        rf = Roboflow()
        project = rf.workspace("thermal-imaging-0hwfw").project("flir-data-set")
        version = project.version(1)
        version.download("yolov8", location=str(dest))
        print(f"FLIR ADAS downloaded to {dest}")
    except ImportError:
        print("Install roboflow package for automatic download: pip install roboflow")
        print("Or download manually from the URL above.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Download manually from the URL above.")

    return dest


def main():
    parser = argparse.ArgumentParser(description="Download thermal detection datasets")
    parser.add_argument("--dataset", choices=["hit-uav", "flir", "all"], default="hit-uav",
                       help="Which dataset to download (default: hit-uav)")
    args = parser.parse_args()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("hit-uav", "all"):
        download_hit_uav()

    if args.dataset in ("flir", "all"):
        download_flir()

    print("\nDone! Next step: python train.py")


if __name__ == "__main__":
    main()
