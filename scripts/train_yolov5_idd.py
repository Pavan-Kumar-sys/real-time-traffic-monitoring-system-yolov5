"""
YOLOv5 Fine-Tuning & Export Pipeline for Indian Driving Dataset (IDD)
=====================================================================
Trains YOLOv5s natively on Indian road classes (including Auto-Rickshaw / 3-Wheeler)
and exports FP16 ONNX weights for sub-10ms edge inference.

Usage:
    python scripts/train_yolov5_idd.py --data idd_traffic.yaml --epochs 50 --batch 16
"""

import argparse
from pathlib import Path

IDD_DATASET_YAML = """# Indian Driving Dataset (IDD) Traffic Configuration for YOLOv5
path: ./datasets/idd_traffic
train: images/train
val: images/val

nc: 7
names:
  0: Pedestrian
  1: Bicycle
  2: Car
  3: Motorcycle
  4: Auto-Rickshaw
  5: Bus
  6: Truck
"""


def generate_idd_yaml(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(IDD_DATASET_YAML, encoding="utf-8")
    return output_path


def train_and_export(epochs: int = 50, imgsz: int = 480, batch: int = 16) -> None:
    yaml_path = generate_idd_yaml(Path("configs/idd_traffic.yaml"))
    print(f"[SentinAI] Generated IDD dataset config at: {yaml_path}")

    from ultralytics import YOLO

    model = YOLO("yolov5su.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        mosaic=1.0,
        mixup=0.15,
        project="runs/idd_yolov5",
        name="yolov5s_idd_3wheeler",
    )
    onnx_path = model.export(format="onnx", half=True, dynamic=False, imgsz=imgsz)
    print(f"[SentinAI] Exported optimized FP16 ONNX model to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv5s on IDD Traffic Dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    train_and_export(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
