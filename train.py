"""
=============================================================================
EcoSort — YOLO26 General Training Script (Local, CUDA)
=============================================================================
Train YOLO26 to detect 4 waste classes: Plastic, Paper, Metal, Others.
This is the general-purpose script. For dataset-specific tuning, use:
  - train_10k.py  (10K dataset, stability-focused)
  - train_60k.py  (60K dataset, performance-focused)

SETUP:
  1. pip install ultralytics torch torchvision opencv-python
  2. Put your dataset in:
       datasets/ecosort/
       ├── train/images/    ← Training images
       ├── train/labels/    ← YOLO labels (.txt)
       ├── val/images/      ← Validation images
       └── val/labels/      ← Validation labels

     Each label .txt line: <class_id> <x_center> <y_center> <width> <height>
     Class IDs: 0=Plastic, 1=Paper, 2=Metal, 3=Others

  3. Train:
       python train.py                          # GPU (default)
       python train.py --device cpu             # CPU only
       python train.py --model yolo26s.pt       # Bigger model
       python train.py --batch 8 --epochs 200   # Adjust for hardware
=============================================================================
"""

import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="EcoSort YOLO26 Training")
    p.add_argument("--model", default="yolo26n.pt",
                   help="yolo26n.pt (Nano) or yolo26s.pt (Small)")
    p.add_argument("--data", default="data.yaml", help="Dataset config YAML")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Reduce to 8 if OOM")
    p.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    p.add_argument("--project", default="runs/ecosort")
    p.add_argument("--name", default="train")
    return p.parse_args()


def train(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,

        # YOLO26 optimizer — MuSGD (Momentum-Updated SGD)
        optimizer="MuSGD",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        momentum=0.937,
        weight_decay=0.0005,

        # Loss weights
        # NOTE: YOLO26 has NO DFL module — uses direct coordinate regression.
        # Do not set a dfl parameter; it does not exist in YOLO26's loss.
        # STAL and ProgLoss are handled internally by the architecture.
        box=7.5,
        cls=0.5,

        # Augmentation for trash can conditions
        hsv_h=0.02,
        hsv_s=0.75,
        hsv_v=0.50,
        degrees=180.0,
        translate=0.2,
        scale=0.5,
        shear=5.0,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        close_mosaic=20,

        # Training settings
        workers=8,
        cache=True,
        amp=True,
        cos_lr=True,
        patience=50,
        save_period=25,
        exist_ok=True,
        verbose=True,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"\nNext steps:")
    print(f"  python evaluate.py")
    print(f"  python inference.py --source 0")
    print(f"  python export.py --format onnx")
    print("=" * 60)
    return results


if __name__ == "__main__":
    train(parse_args())
