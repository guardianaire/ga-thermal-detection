#!/usr/bin/env python3
"""Export trained YOLO model to TFLite for Android deployment.

Exports the best trained model to TFLite format with INT8 or FP16
quantization for efficient inference on Android tablets.

Usage:
  # Default: FP16 quantization
  python export_tflite.py --weights runs/thermal/train_yolo11n/weights/best.pt

  # INT8 quantization (smaller, faster, needs calibration data)
  python export_tflite.py --weights runs/thermal/train_yolo11n/weights/best.pt --int8

  # Also export ONNX for testing
  python export_tflite.py --weights runs/thermal/train_yolo11n/weights/best.pt --onnx
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export(args):
    print(f"=== Exporting Model to TFLite ===")
    print(f"Weights: {args.weights}")
    print(f"Image Size: {args.imgsz}")
    print(f"Quantization: {'INT8' if args.int8 else 'FP16'}")
    print()

    model = YOLO(args.weights)

    # Export to TFLite
    tflite_path = model.export(
        format="tflite",
        imgsz=args.imgsz,
        int8=args.int8,
        half=not args.int8,  # FP16 if not INT8
    )
    print(f"\nTFLite model exported: {tflite_path}")

    # Optionally export ONNX
    if args.onnx:
        onnx_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
        )
        print(f"ONNX model exported: {onnx_path}")

    # Print model info
    tflite_file = Path(tflite_path)
    if tflite_file.exists():
        size_mb = tflite_file.stat().st_size / (1024 * 1024)
        print(f"\nModel size: {size_mb:.1f} MB")
        print(f"\nNext step: Copy {tflite_path} to the Android app's assets folder")
        print(f"  cp {tflite_path} ~/repos/autel-mobilesdk-2.0/app/src/main/assets/")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to TFLite for Android")
    parser.add_argument("--weights", required=True,
                       help="Path to trained model weights (best.pt)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Input image size (default: 640)")
    parser.add_argument("--int8", action="store_true",
                       help="Use INT8 quantization (default: FP16)")
    parser.add_argument("--onnx", action="store_true",
                       help="Also export ONNX format")
    args = parser.parse_args()

    export(args)


if __name__ == "__main__":
    main()
