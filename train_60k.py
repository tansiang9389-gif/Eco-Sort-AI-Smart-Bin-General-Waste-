"""
=============================================================================
EcoSort — YOLO26 Training Script for 60K Dataset (CUDA / RTX 3050)
=============================================================================
PERFORMANCE-FOCUSED configuration for a ~60,000-image dataset.

WHY THIS IS DIFFERENT FROM 10K:
  The power-law error model E(n) = a * n^(-b) shows that 60K is 4× above
  the 15,000-image reliability benchmark. With this much data:
    - The dataset itself is the regularizer (no dropout needed)
    - Augmentation supplements diversity but doesn't carry the load
    - Higher LR is safe because gradient estimates are stable
    - Fewer epochs needed (more iterations per epoch)

  Error comparison (for typical b ≈ 0.5):
    E(10K)  = a × 10000^(-0.5)  = a × 0.01
    E(60K)  = a × 60000^(-0.5)  = a × 0.00408
    → 60K dataset has ~59% LESS error than 10K at same training effort

EPOCH CALIBRATION (no under/oversampling):
  ┌────────────────────────────────────────────────────────────────────┐
  │  60,000 images ÷ batch 16 = 3,750 iterations/epoch               │
  │  80 epochs × 3,750 = 300,000 total iterations                    │
  │  With mosaic (4 images/tile): ~1,200,000 effective image views    │
  │                                                                    │
  │  Compare to 10K script:                                           │
  │    10K: 150 epochs × 625 iter = 93,750 iterations                │
  │    60K:  80 epochs × 3,750 iter = 300,000 iterations             │
  │                                                                    │
  │  The 60K model sees 3.2× MORE total iterations despite fewer     │
  │  epochs. This is correct: each epoch already shows the model     │
  │  6× more unique images, so fewer passes are needed.              │
  │                                                                    │
  │  Going beyond 80 epochs would oversample — the model would see   │
  │  each image 80+ times with diminishing returns. Early stopping   │
  │  at patience=20 will halt even sooner if the model converges.    │
  └────────────────────────────────────────────────────────────────────┘

YOLO26-SPECIFIC ARCHITECTURE NOTES:
  • MuSGD optimizer  — Stable convergence; standard LR works with 60K data
  • NMS-Free head    — Direct detection output, no post-processing sorting
  • No DFL module    — Direct coordinate regression (no softmax in head)
  • STAL             — Small-Target-Aware Label Assignment (internal)
  • ProgLoss         — Progressive Loss scheduling (internal)

HARDWARE:
  GPU: NVIDIA RTX 3050 (4GB or 8GB VRAM)
  Batch 16 → fits in 8GB VRAM with FP16 (AMP)
  60K images do NOT fit in RAM cache → loaded from disk via workers

Run:
  python train_60k.py

Classes: Plastic (0), Paper (1), Metal (2), Others (3)
=============================================================================
"""

import sys
import torch
from ultralytics import YOLO


def check_cuda():
    """Verify CUDA is available and print GPU info."""
    if not torch.cuda.is_available():
        print("=" * 60)
        print("ERROR: CUDA not available!")
        print("Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 60)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 5:
        print("WARNING: 4GB VRAM detected. Change batch to 8 in this script.")
    return gpu_name, vram_gb


def train():
    """
    Train YOLO26 Nano on a 60,000-image waste dataset.

    With 60K images (4× above the 15K benchmark), the strategy shifts
    from "prevent overfitting" to "maximize feature extraction."
    Regularization is minimal; the data provides generalization.
    """

    gpu_name, vram_gb = check_cuda()

    # ── Load YOLO26 Nano pretrained weights ──────────────────────────
    # Same yolo26n.pt as 10K — the architecture stays compact for edge
    # deployment. With 60K images, the model has enough data to fully
    # utilize its ~2.5M parameters without overfitting.
    model = YOLO("yolo26n.pt")

    # ── Train ────────────────────────────────────────────────────────
    results = model.train(

        # ═══════════════════════════════════════════════════════════════
        # DATASET & CORE SETTINGS
        # ═══════════════════════════════════════════════════════════════
        data="data_60k.yaml",       # 4 classes: Plastic, Paper, Metal, Others
        epochs=80,                  # 80 × 3,750 iter = 300,000 total iterations
                                    # Each epoch sees 60K images — far more
                                    # than the 10K script's 150 epochs.
                                    # Going beyond 80 = oversampling
                                    # (diminishing returns, risk of overfitting
                                    # to augmentation artifacts)
        imgsz=640,                  # Standard YOLO input resolution
        batch=16,                   # 16 for 8GB VRAM; change to 8 for 4GB
        device="cuda",              # Use NVIDIA GPU

        # ═══════════════════════════════════════════════════════════════
        # OUTPUT DIRECTORY
        # ═══════════════════════════════════════════════════════════════
        project="runs/ecosort",
        name="train_60k",
        exist_ok=True,

        # ═══════════════════════════════════════════════════════════════
        # YOLO26 OPTIMIZER: MuSGD
        # ═══════════════════════════════════════════════════════════════
        # With 60K images, gradient estimates are 6× more stable than
        # with 10K. This means:
        #   - Standard LR (0.01) is safe — no overshooting risk
        #   - Standard warmup is sufficient
        #   - Standard weight decay — data diversity prevents overfitting
        optimizer="MuSGD",
        lr0=0.01,                   # Standard initial LR — stable gradients
                                    # from 3,750 iterations/epoch allow it
        lrf=0.01,                   # Final LR factor: 0.01 × 0.01 = 0.0001
                                    # Standard decay — model refines in
                                    # last epochs without stalling
        warmup_epochs=5.0,          # 5 epochs = 18,750 iterations warmup
                                    # (3× more warmup iterations than 10K
                                    # due to larger iter/epoch — proportional)
        warmup_momentum=0.8,        # Standard warmup momentum
        momentum=0.937,             # Standard momentum
        weight_decay=0.0005,        # Standard L2 — no extra needed because
                                    # 60K diverse images prevent memorization

        # ═══════════════════════════════════════════════════════════════
        # LOSS WEIGHTS
        # ═══════════════════════════════════════════════════════════════
        # Same as 10K — YOLO26 architecture is identical regardless of
        # dataset size. DFL does not exist in YOLO26 (direct regression).
        # STAL and ProgLoss are handled internally.
        box=7.5,                    # Bounding box regression loss weight
        cls=0.5,                    # Classification loss weight

        # ═══════════════════════════════════════════════════════════════
        # STANDARD DATA AUGMENTATION
        # ═══════════════════════════════════════════════════════════════
        # With 60K images, the dataset already contains natural diversity.
        # Augmentation supplements but doesn't need to compensate for
        # missing data like the 10K script.
        #
        # Key differences from 10K:
        #   hsv_s:   0.7 (not 0.8)  — less extreme saturation needed
        #   hsv_v:   0.4 (not 0.5)  — less extreme brightness needed
        #   scale:   0.5 (not 0.6)  — narrower scale range (less synthetic)
        #   mixup:   0.15 (not 0.2) — less blending needed
        #   copy_paste: 0.1 (not 0.15) — less synthetic pasting
        #   close_mosaic: 10 (not 20) — shorter clean fine-tuning phase

        # ── Color & Lighting ─────────────────────────────────────────
        hsv_h=0.015,                # Hue shift ±1.5% — standard
        hsv_s=0.7,                  # Saturation ±70% — still covers bin lighting
        hsv_v=0.4,                  # Brightness ±40% — standard

        # ── Geometric Transforms ─────────────────────────────────────
        degrees=180.0,              # Full rotation — waste orientation is random
        translate=0.2,              # ±20% position shift
        scale=0.5,                  # ±50% scale — standard range
        shear=5.0,                  # ±5° shear
        perspective=0.001,          # Slight perspective warp
        flipud=0.5,                 # 50% vertical flip
        fliplr=0.5,                 # 50% horizontal flip

        # ── Mosaic & MixUp ───────────────────────────────────────────
        mosaic=1.0,                 # 100% mosaic — still valuable for occlusion
        mixup=0.15,                 # 15% blend — less than 10K since dataset
                                    # already has natural overlap examples
        copy_paste=0.1,             # 10% — less synthetic pasting needed
        close_mosaic=10,            # Disable mosaic for last 10 epochs
                                    # Shorter fine-tuning: model already
                                    # generalizes well with 60K data

        # ═══════════════════════════════════════════════════════════════
        # NO EXTRA REGULARIZATION
        # ═══════════════════════════════════════════════════════════════
        # 60K images above the 15K benchmark means the dataset itself
        # prevents overfitting. Adding dropout or label smoothing would
        # REDUCE accuracy by unnecessarily constraining the model.
        #
        # Think of it this way:
        #   10K: model has more capacity than data → needs constraints
        #   60K: data matches or exceeds model capacity → let it learn freely
        dropout=0.0,                # No dropout — data is the regularizer
        label_smoothing=0.0,        # No smoothing — let the model be confident

        # ═══════════════════════════════════════════════════════════════
        # TRAINING INFRASTRUCTURE
        # ═══════════════════════════════════════════════════════════════
        workers=8,                  # DataLoader workers (adjust to CPU cores)
        cache=False,                # 60K images do NOT fit in RAM
                                    # (~12-24GB needed). Disk loading via
                                    # workers=8 provides sufficient throughput.
                                    # Set cache="ram" ONLY if you have 32GB+ RAM.
        amp=True,                   # FP16 mixed precision — essential for
                                    # RTX 3050 VRAM management
        cos_lr=True,                # Cosine annealing LR schedule
        patience=20,                # Early stopping: halt if val mAP doesn't
                                    # improve for 20 epochs.
                                    # 20 × 3,750 = 75,000 iterations
                                    # Tighter than 10K (30 epochs) because
                                    # each epoch already covers 6× more data.
                                    # If 75K iterations show no improvement,
                                    # the model has genuinely plateaued.
        save_period=10,             # Checkpoint every 10 epochs
        verbose=True,
        seed=42,                    # Reproducibility
    )

    # ── Report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("60K TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best weights:  runs/ecosort/train_60k/weights/best.pt")
    print(f"Last weights:  runs/ecosort/train_60k/weights/last.pt")
    print(f"Results CSV:   runs/ecosort/train_60k/results.csv")
    print(f"Curves:        runs/ecosort/train_60k/results.png")
    print()
    print("Next steps:")
    print("  1. python evaluate.py --model runs/ecosort/train_60k/weights/best.pt --data data_60k.yaml")
    print("  2. python inference.py --weights runs/ecosort/train_60k/weights/best.pt --source 0")
    print("  3. python export.py --model runs/ecosort/train_60k/weights/best.pt --format onnx")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train()
