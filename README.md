# GuardianAire Thermal Person Detection

Custom YOLO model for detecting people in thermal/IR imagery from drone aerial view.

## Goal
Replace/supplement the Autel SDK's built-in AI detection with a custom model optimized for thermal person detection from UAV altitude.

## Approach
- Fine-tune YOLO11-N (or YOLO26-N) on UAV thermal datasets
- Export to TFLite for Android deployment on the Autel tablet
- Feed detections into the existing Situation Awareness pipeline

## Datasets
- HIT-UAV: High-altitude infrared thermal from drones
- POP: Partially occluded persons from UAV thermal
- FLIR ADAS: Large thermal dataset for pretraining

## Quick Start
```bash
pip install -r requirements.txt
python train.py
```

