#!/usr/bin/env python3
"""Train YOLO model for thermal person detection.

Fine-tunes a YOLO nano model on the HIT-UAV thermal dataset.
Optimized for deployment on Android (Autel drone tablet).

Usage:
  # Train with defaults (YOLO11-N, HIT-UAV, 100 epochs)
  python train.py

  # Custom options
  python train.py --model yolo11n.pt --epochs 200 --imgsz 640 --batch 16

  # Use YOLO26 (if available)
  python train.py --model yolo26n.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(args):
    print(f"=== Thermal Person Detection Training ===")
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.data}")
    print(f"Epochs:   {args.epochs}")
    print(f"Img Size: {args.imgsz}")
    print(f"Batch:    {args.batch}")
    print(f"Device:   {args.device}")
    print()

    # Load pretrained model
    model = YOLO(args.model)

    # Fine-tune on thermal dataset
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/thermal",
        name=f"train_{Path(args.model).stem}",
        # Augmentation tuned for thermal imagery
        hsv_h=0.0,      # No hue augmentation (thermal is grayscale)
        hsv_s=0.0,      # No saturation augmentation
        hsv_v=0.3,      # Brightness variation (thermal intensity varies)
        degrees=10.0,    # Slight rotation (drone perspective)
        translate=0.1,   # Translation
        scale=0.5,       # Scale variation (altitude changes)
        fliplr=0.5,      # Horizontal flip
        flipud=0.1,      # Slight vertical flip (aerial view)
        mosaic=0.8,      # Mosaic augmentation
        mixup=0.1,       # Mixup augmentation
        # Training params
        patience=20,     # Early stopping
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print(f"\nTraining complete!")
    print(f"Best model: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python evaluate.py --weights {results.save_dir / 'weights' / 'best.pt'}")
    print(f"  2. Export:   python export_tflite.py --weights {results.save_dir / 'weights' / 'best.pt'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train thermal person detection model")
    parser.add_argument("--model", default="yolo11n.pt",
                       help="Pretrained model to fine-tune (default: yolo11n.pt)")
    parser.add_argument("--data", default="hit_uav.yaml",
                       help="Dataset config YAML (default: hit_uav.yaml)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=-1,
                       help="Batch size, -1 for auto (default: -1)")
    parser.add_argument("--device", default="",
                       help="Device: '' for auto, 'cpu', '0' for GPU (default: auto)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
