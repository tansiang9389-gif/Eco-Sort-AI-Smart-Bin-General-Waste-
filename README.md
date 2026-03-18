# EcoSort — YOLO26 Waste Detection System

Real-time waste detection and classification using YOLO26 (YOLOv26) for embedded edge deployment. Detects **4 waste categories** inside a physical trash can using a top-down camera.

| Class ID | Category | Bin Recommendation |
|----------|----------|-------------------|
| 0 | Plastic | Recycling (Yellow) |
| 1 | Paper | Recycling (Blue) |
| 2 | Metal | Recycling (Yellow) |
| 3 | Others | General Waste (Black) |

---

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA (RTX 3050 or higher recommended)
- **PyTorch** with CUDA support

```bash
# Install PyTorch with CUDA 12.1 (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

Prepare your dataset in standard YOLO format:

```
datasets/
└── ecosort_10k/          # or ecosort_60k/
    ├── train/
    │   ├── images/       # Training images (.jpg, .png)
    │   └── labels/       # YOLO labels (.txt)
    ├── val/
    │   ├── images/       # Validation images (15-20% of data)
    │   └── labels/
    └── test/             # (Optional) Hold-out test set
        ├── images/
        └── labels/
```

Each label `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized to `[0, 1]`.

---

## Commands

### 1. Train the Model

Two dataset-specific training scripts are provided, each calibrated to avoid under/oversampling based on the power-law error model `E(n) = a * n^(-b)`.

#### Train on 10K dataset (stability-focused)

```bash
python train_10k.py
```

- **150 epochs** x 625 iter/epoch = 93,750 total iterations
- **Aggressive augmentation** (mosaic, mixup, copy-paste, extreme HSV jitter) to compensate for being below the 15K reliability benchmark
- **Strong regularization** (dropout 0.1, label smoothing 0.05, weight decay 0.001)
- **Conservative LR** (lr0=0.005, half default) to avoid overfitting
- **Early stopping** with patience=30
- Uses `data_10k.yaml` — place dataset in `datasets/ecosort_10k/`

#### Train on 60K dataset (performance-focused)

```bash
python train_60k.py
```

- **80 epochs** x 3,750 iter/epoch = 300,000 total iterations
- **Standard augmentation** — the dataset's own diversity handles generalization
- **No extra regularization** (no dropout, no label smoothing) — data is the regularizer
- **Standard LR** (lr0=0.01) — stable gradients from large batch diversity
- **Early stopping** with patience=20
- Uses `data_60k.yaml` — place dataset in `datasets/ecosort_60k/`

#### Train (general purpose, custom args)

```bash
python train.py
python train.py --model yolo26s.pt --epochs 200 --batch 8
python train.py --device cpu
```

Both dataset-specific scripts use:
- **YOLO26 Nano** (`yolo26n.pt`) pretrained weights
- **MuSGD optimizer** (YOLO26 native)
- **CUDA** with FP16 mixed precision (AMP)
- **Batch 16** (use `--batch 8` for 4GB VRAM laptops)

Outputs are saved to `runs/ecosort/train_10k/` or `runs/ecosort/train_60k/`.

---

### 2. Evaluate the Model

Computes mAP@0.50, mAP@0.50:0.95, per-class AP, confusion matrix, and checks against the 98% accuracy target.

```bash
# Evaluate 10K model on its validation set
python evaluate.py --model runs/ecosort/train_10k/weights/best.pt --data data_10k.yaml

# Evaluate 60K model on its validation set
python evaluate.py --model runs/ecosort/train_60k/weights/best.pt --data data_60k.yaml

# Evaluate on the test split instead of val
python evaluate.py --model runs/ecosort/train_60k/weights/best.pt --data data_60k.yaml --split test

# Save metrics summary to JSON
python evaluate.py --model runs/ecosort/train_60k/weights/best.pt --data data_60k.yaml --save
```

**Generated outputs** (saved to `runs/ecosort/evaluate/`):
- `confusion_matrix.png` — raw counts
- `confusion_matrix_normalized.png` — percentages
- `PR_curve.png` — Precision-Recall curve
- `F1_curve.png` — F1 vs confidence threshold
- `metrics_summary.json` — all metrics in JSON (with `--save`)

---

### 3. Run Inference (Live Detection)

Real-time detection with bounding boxes, class labels, confidence scores, and bin recommendations drawn on-screen. Uses YOLO26's **NMS-free architecture** — no Non-Maximum Suppression post-processing.

```bash
# USB webcam (live)
python inference.py --source 0

# Raspberry Pi Camera
python inference.py --source picam

# Single image
python inference.py --source path/to/image.jpg

# Video file
python inference.py --source path/to/video.mp4

# Custom confidence threshold
python inference.py --source 0 --conf_thresh 0.4

# Specify which trained model to use
python inference.py --weights runs/ecosort/train_10k/weights/best.pt --source 0

# Headless mode (no display) + save output video
python inference.py --source 0 --no-show --save

# Faster inference (lower resolution)
python inference.py --source 0 --imgsz 320
```

**Keyboard controls during live inference:**
- `q` or `ESC` — quit
- `c` — clear rolling confidence buffer

**On-screen display:**
- Bounding boxes colored by class (Orange=Plastic, Green=Paper, Cyan=Metal, Gray=Others)
- Confidence percentage per detection
- Bin recommendation below each box
- Summary panel (top-left) with confirmed detections
- FPS counter (top-right)

---

### 4. Export for Edge Deployment

Export the trained model to optimized formats for embedded hardware. YOLO26's **DFL-free architecture** (no Distribution Focal Loss / no softmax in detection head) makes export cleaner and faster on microcontrollers.

```bash
# ONNX (default — broadest compatibility)
python export.py --model runs/ecosort/train_10k/weights/best.pt

# INT8 TFLite for ESP32-S3 / ARM Cortex-M
python export.py --model runs/ecosort/train_10k/weights/best.pt --format tflite --int8

# TensorRT FP16 for NVIDIA Jetson
python export.py --model runs/ecosort/train_10k/weights/best.pt --format engine --half

# Smaller input size for extreme edge constraints
python export.py --model runs/ecosort/train_10k/weights/best.pt --format tflite --int8 --imgsz 320

# OpenVINO for Intel edge devices
python export.py --model runs/ecosort/train_10k/weights/best.pt --format openvino

# NCNN for mobile / embedded ARM
python export.py --model runs/ecosort/train_10k/weights/best.pt --format ncnn
```

| Format | Target Hardware | Notes |
|--------|----------------|-------|
| `onnx` | CPU/GPU, broadest | Standard runtime, no custom ops |
| `tflite` | ESP32-S3, ARM, Android | INT8 quantized ~1.5MB |
| `engine` | Jetson Nano/Orin | Must export ON target device |
| `openvino` | Intel NCS2, iGPU | IR format |
| `coreml` | Apple iOS/macOS | Neural Engine |
| `ncnn` | Android, RPi, ARM | No dependencies |

---

## File Structure

```
EcoSort/
├── data.yaml              # Dataset config (general, 4 classes)
├── data_10k.yaml          # Dataset config for 10K dataset
├── data_60k.yaml          # Dataset config for 60K dataset
├── train.py               # General training script (CLI args)
├── train_10k.py           # 10K-optimized training (stability-focused)
├── train_60k.py           # 60K-optimized training (performance-focused)
├── evaluate.py            # mAP, confusion matrix, 98% threshold check
├── inference.py           # Real-time detection with on-screen display
├── export.py              # Edge export (ONNX, TFLite, TensorRT, etc.)
├── requirements.txt       # Python dependencies
└── .claude/
    └── launch.json        # Launch configurations
```

---

## YOLO26 Architecture Highlights

| Feature | Benefit |
|---------|---------|
| **MuSGD Optimizer** | Momentum-updated SGD — stable convergence on detection tasks |
| **NMS-Free Head** | No Non-Maximum Suppression — saves 2-10ms per frame on edge |
| **No DFL Module** | Direct coordinate regression — no softmax, cleaner INT8 quantization |
| **STAL** | Small-Target-Aware Label Assignment — better detection of tiny/occluded waste |
| **ProgLoss** | Progressive Loss — gradually shifts focus from easy to hard objects |

---

## Quick Start

```bash
# 1. Install
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Prepare dataset in datasets/ecosort_10k/ (or ecosort_60k/)

# 3. Train
python train_10k.py

# 4. Evaluate
python evaluate.py --model runs/ecosort/train_10k/weights/best.pt --data data_10k.yaml --save

# 5. Test with webcam
python inference.py --weights runs/ecosort/train_10k/weights/best.pt --source 0

# 6. Export for edge
python export.py --model runs/ecosort/train_10k/weights/best.pt --format tflite --int8
```
