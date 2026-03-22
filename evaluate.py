#!/usr/bin/env python3
"""Evaluate trained model on test set and visualize results.

Usage:
  python evaluate.py --weights runs/thermal/train_yolo11n/weights/best.pt
  python evaluate.py --weights runs/thermal/train_yolo11n/weights/best.pt --show
"""

import argparse
from ultralytics import YOLO


def evaluate(args):
    print(f"=== Evaluating Thermal Detection Model ===")
    print(f"Weights: {args.weights}")
    print()

    model = YOLO(args.weights)

    # Run validation
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
        verbose=True,
    )

    # Print key metrics
    print(f"\n=== Results ===")
    print(f"mAP50:     {results.box.map50:.3f}")
    print(f"mAP50-95:  {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall:    {results.box.mr:.3f}")

    # Per-class results
    if hasattr(results.box, 'ap_class_index'):
        print(f"\n=== Per-Class AP50 ===")
        names = model.names
        for i, cls_idx in enumerate(results.box.ap_class_index):
            print(f"  {names[cls_idx]}: {results.box.ap50[i]:.3f}")

    # Run inference on sample images if --show
    if args.show:
        print(f"\nRunning inference on test images...")
        model.predict(
            source=f"datasets/hit-uav/raw/normal/images/test",
            imgsz=args.imgsz,
            save=True,
            project="runs/thermal",
            name="predict_samples",
            conf=0.25,
        )
        print(f"Predictions saved to runs/thermal/predict_samples/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate thermal detection model")
    parser.add_argument("--weights", required=True,
                       help="Path to trained model weights")
    parser.add_argument("--data", default="hit_uav.yaml",
                       help="Dataset config YAML")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--show", action="store_true",
                       help="Run inference on test images and save results")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
