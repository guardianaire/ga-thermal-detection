#!/usr/bin/env python3
"""Fine-tune the HIT-UAV trained model on custom drone thermal data.

Starts from the best weights trained on HIT-UAV and fine-tunes on
your own labeled thermal images from the Autel drone.

Usage:
  python finetune.py
  python finetune.py --weights weights/thermal_person_best.pt --epochs 50
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def finetune(args):
    print(f"=== Fine-tuning on Custom Thermal Data ===")
    print(f"Base model:  {args.weights}")
    print(f"Dataset:     {args.data}")
    print(f"Epochs:      {args.epochs}")
    print()

    model = YOLO(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/finetune",
        name=f"custom_{Path(args.weights).stem}",
        # Lower learning rate for fine-tuning (don't forget HIT-UAV knowledge)
        lr0=0.001,
        lrf=0.01,
        # Thermal augmentation
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.1,
        mosaic=0.8,
        mixup=0.1,
        patience=15,
        save=True,
        plots=True,
    )

    print(f"\nFine-tuning complete!")
    print(f"Best model: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"\nExport: python export_tflite.py --weights {results.save_dir / 'weights' / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on custom thermal data")
    parser.add_argument("--weights", default="weights/thermal_person_best.pt",
                       help="Base model weights from HIT-UAV training")
    parser.add_argument("--data", default="custom_thermal.yaml",
                       help="Custom dataset YAML")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Fine-tuning epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size")
    parser.add_argument("--batch", type=int, default=-1,
                       help="Batch size")
    parser.add_argument("--device", default="",
                       help="Device")
    args = parser.parse_args()

    finetune(args)


if __name__ == "__main__":
    main()
