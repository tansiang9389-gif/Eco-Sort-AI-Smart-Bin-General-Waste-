"""
=============================================================================
EcoSort — YOLO26 Training Script for 10K Dataset (CUDA / RTX 3050)
=============================================================================
STABILITY-FOCUSED configuration for a ~10,000-image dataset.

WHY THIS NEEDS SPECIAL CARE:
  The power-law error model E(n) = a * n^(-b) tells us that 10K images is
  BELOW the 15,000-image reliability threshold for complex, non-uniform
  objects like crushed/dirty/overlapping trash. A naive training run will
  overfit: the model memorizes specific training items and fails on unseen
  waste in the real bin.

COUNTERMEASURES (lessons from high-accuracy CNN research):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. AGGRESSIVE AUGMENTATION                                        │
  │    Mosaic (4-image tiles), MixUp, copy-paste, extreme color jitter │
  │    → Pushes effective dataset diversity above 15K threshold        │
  │                                                                    │
  │ 2. STRONG REGULARIZATION                                          │
  │    Dropout 0.1, weight_decay 0.001, label_smoothing 0.05          │
  │    → Prevents memorization of training samples                    │
  │                                                                    │
  │ 3. CONSERVATIVE LEARNING RATE                                     │
  │    lr0=0.005 (half default), cosine annealing, 5-epoch warmup     │
  │    → Avoids sharp local minima that overfit small datasets         │
  │                                                                    │
  │ 4. EARLY STOPPING (patience=30)                                   │
  │    → Catches overfitting the moment val mAP plateaus              │
  └─────────────────────────────────────────────────────────────────────┘

EPOCH CALIBRATION (no under/oversampling):
  10,000 images ÷ batch 16 = 625 iterations/epoch
  150 epochs × 625 = 93,750 total iterations
  With mosaic (4 images/tile): ~375,000 effective image views
  With all augmentation: ~600K+ unique views
  → Matches the ~600K target (15K × 40 views) from the power-law model
  → 150 epochs is the sweet spot: enough to converge, not enough to overfit

YOLO26-SPECIFIC ARCHITECTURE NOTES:
  • MuSGD optimizer  — Momentum-updated SGD, native to YOLO26, provides
                       more stable convergence than Adam on detection tasks
  • NMS-Free head    — No Non-Maximum Suppression needed; the model outputs
                       up to 300 direct detections (no post-processing sort)
  • No DFL module    — YOLO26 uses direct coordinate regression instead of
                       Distribution Focal Loss. This removes softmax ops
                       from the detection head, making it simpler and faster
  • STAL             — Small-Target-Aware Label Assignment is built into
                       the YOLO26 architecture's label matching strategy.
                       It improves detection of small/overlapping waste
  • ProgLoss         — Progressive Loss scheduling is handled internally
                       by the YOLO26 training loop in ultralytics

HARDWARE:
  GPU: NVIDIA RTX 3050 (4GB or 8GB VRAM)
  Batch 16 → fits in 8GB VRAM with FP16 (AMP)
  Batch 8  → use this if you have the 4GB laptop variant

Run:
  python train_10k.py

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

    # Recommend batch size based on VRAM
    if vram_gb < 5:
        print("WARNING: 4GB VRAM detected. Change batch to 8 in this script.")
        print("         Or set imgsz=480 to fit batch 16.")
    return gpu_name, vram_gb


def train():
    """
    Train YOLO26 Nano on a 10,000-image waste dataset.

    The entire training configuration is tuned for SMALL DATASET STABILITY.
    Every parameter choice is justified by the E(n) = a*n^(-b) model
    and best practices from high-accuracy CNN research.
    """

    gpu_name, vram_gb = check_cuda()

    # ── Load YOLO26 Nano pretrained weights ──────────────────────────
    # yolo26n.pt = Nano variant (~2.5M params), ideal for edge deployment.
    # Pretrained on COCO — provides strong feature initialization so the
    # model doesn't have to learn basic patterns from only 10K images.
    model = YOLO("yolo26n.pt")

    # ── Train ────────────────────────────────────────────────────────
    results = model.train(

        # ═══════════════════════════════════════════════════════════════
        # DATASET & CORE SETTINGS
        # ═══════════════════════════════════════════════════════════════
        data="data_10k.yaml",       # 4 classes: Plastic, Paper, Metal, Others
        epochs=150,                 # 150 × 625 iter = 93,750 total iterations
                                    # Sweet spot: converges fully without
                                    # oversampling the 10K images
        imgsz=640,                  # Standard YOLO input resolution
        batch=16,                   # 16 for 8GB VRAM; change to 8 for 4GB
        device="cuda",              # Use NVIDIA GPU

        # ═══════════════════════════════════════════════════════════════
        # OUTPUT DIRECTORY
        # ═══════════════════════════════════════════════════════════════
        project="runs/ecosort",
        name="train_10k",
        exist_ok=True,

        # ═══════════════════════════════════════════════════════════════
        # YOLO26 OPTIMIZER: MuSGD (Momentum-Updated SGD)
        # ═══════════════════════════════════════════════════════════════
        # MuSGD is YOLO26's native optimizer. It differs from standard
        # SGD by using a momentum-based update rule that reduces the
        # variance of gradient estimates — critical when each mini-batch
        # is only 16 images from a small 10K pool.
        #
        # WHY lr0=0.005 (half default)?
        #   With only 10K images, gradient noise is high. A large LR
        #   would cause the model to jump between sharp local minima
        #   (each one = memorizing a subset of training images).
        #   Lower LR = smoother trajectory = flatter minimum = better
        #   generalization to unseen waste.
        optimizer="MuSGD",
        lr0=0.005,                  # Initial LR: half default (0.01)
        lrf=0.005,                  # Final LR factor: 0.005 × 0.005 = 2.5e-5
                                    # Very low final LR for fine-grained
                                    # convergence in the last epochs
        warmup_epochs=5.0,          # 5 epochs = 3,125 iterations warmup
                                    # Lets BatchNorm statistics stabilize
                                    # before the optimizer takes big steps
        warmup_momentum=0.5,        # Gentler warmup momentum
        momentum=0.937,             # Standard momentum for main training
        weight_decay=0.001,         # 2× default (0.0005) — stronger L2
                                    # regularization to penalize large weights
                                    # that memorize specific training images

        # ═══════════════════════════════════════════════════════════════
        # LOSS WEIGHTS
        # ═══════════════════════════════════════════════════════════════
        # YOLO26 architecture removes the DFL (Distribution Focal Loss)
        # module entirely. Previous YOLO versions (v8, v11) predicted a
        # probability distribution over 16 discretized bbox offsets and
        # needed a DFL loss to train it. YOLO26 uses direct coordinate
        # regression (predicts x,y,w,h scalars), so:
        #   - No DFL loss term exists in YOLO26's loss function
        #   - No softmax in detection head = faster + simpler
        #   - Do NOT set dfl=0.0 — the parameter doesn't exist in YOLO26
        #
        # ProgLoss (Progressive Loss) is handled internally by YOLO26:
        #   it gradually shifts emphasis from easy-to-detect large objects
        #   to harder small objects as training progresses.
        #
        # STAL (Small-Target-Aware Label Assignment) is also internal:
        #   YOLO26's label matching uses an IoU-aware assignment that
        #   gives higher priority to small targets, preventing them from
        #   being ignored by larger object gradients.
        box=7.5,                    # Bounding box regression loss weight
        cls=0.5,                    # Classification loss weight

        # ═══════════════════════════════════════════════════════════════
        # AGGRESSIVE DATA AUGMENTATION
        # ═══════════════════════════════════════════════════════════════
        # This is the PRIMARY defense against the 15K benchmark gap.
        #
        # Every augmentation creates images the model hasn't memorized,
        # pushing effective diversity past the 15K threshold.
        # The math: 10K × mosaic(4×) × augment(~1.6×) = ~64K effective
        # unique training views per epoch.
        #
        # These settings simulate the harsh, unpredictable interior of
        # a trash bin: poor lighting, random orientations, occlusion.

        # ── Color & Lighting ─────────────────────────────────────────
        # Trash bins have wildly varying lighting: sometimes a phone
        # flashlight, sometimes overhead fluorescent, sometimes dark.
        hsv_h=0.025,                # Hue shift ±2.5% — broader color variation
        hsv_s=0.8,                  # Saturation ±80% — simulate deep shadows
                                    # and washed-out fluorescent lighting
        hsv_v=0.5,                  # Brightness ±50% — dark bin interior
                                    # to bright overhead light

        # ── Geometric Transforms ─────────────────────────────────────
        # Waste falls randomly; the top-down camera sees every angle.
        degrees=180.0,              # Full 360° rotation — waste has no
                                    # "correct" orientation in a bin
        translate=0.2,              # ±20% position shift — waste isn't
                                    # always centered under the camera
        scale=0.6,                  # ±60% scale — wider range than 60K
                                    # script to simulate distance variation
                                    # (camera at different bin heights)
        shear=5.0,                  # ±5° shear — perspective distortion
                                    # from non-perpendicular camera mount
        perspective=0.001,          # Slight perspective warp
        flipud=0.5,                 # 50% vertical flip
        fliplr=0.5,                 # 50% horizontal flip

        # ── Mosaic & MixUp ───────────────────────────────────────────
        # Mosaic tiles 4 images into one, which:
        #   1. Simulates densely packed garbage (occlusion)
        #   2. Shows 4× more images per iteration (effective batch = 64)
        #   3. Forces the model to detect at multiple scales simultaneously
        # MixUp blends 2 images together, simulating semi-transparent
        # overlapping trash bags.
        mosaic=1.0,                 # 100% of batches use 4-image mosaic
        mixup=0.2,                  # 20% blend probability (higher than 60K
                                    # to generate more synthetic combos)
        copy_paste=0.15,            # 15% chance to paste objects from one
                                    # image onto another — crucial for
                                    # generating rare class combinations
        close_mosaic=20,            # Disable mosaic for LAST 20 epochs
                                    # (longer clean fine-tuning for small
                                    # dataset — lets model refine bbox
                                    # accuracy without mosaic distortion)

        # ═══════════════════════════════════════════════════════════════
        # REGULARIZATION
        # ═══════════════════════════════════════════════════════════════
        # The #1 lesson from the Kaggle MNIST CNN notebook and the
        # research paper: small datasets MUST have regularization.
        # Without it, the model achieves 99% training accuracy but
        # fails on unseen data (the notebook itself says "Not good
        # enough!!" at 97% without CNN regularization).
        #
        # For YOLO26, regularization is applied in the detection head:
        dropout=0.1,                # 10% dropout in detection head
                                    # Forces the network to learn redundant
                                    # features — no single neuron can
                                    # memorize a specific training image
        label_smoothing=0.05,       # Targets become 0.95 instead of 1.0
                                    # Prevents overconfident predictions
                                    # on memorized samples. The model
                                    # learns "this is PROBABLY plastic"
                                    # instead of "this is DEFINITELY plastic"

        # ═══════════════════════════════════════════════════════════════
        # TRAINING INFRASTRUCTURE
        # ═══════════════════════════════════════════════════════════════
        workers=8,                  # DataLoader workers (adjust to CPU cores)
        cache="ram",                # 10K images fit in RAM (~2-4GB)
                                    # Eliminates disk I/O bottleneck
        amp=True,                   # FP16 mixed precision — halves VRAM usage
                                    # RTX 3050 has good FP16 tensor cores
        cos_lr=True,                # Cosine annealing LR schedule:
                                    #   LR follows cos(π × epoch/total_epochs)
                                    #   Smooth decay from lr0 to lr0×lrf
                                    #   Acts like ReduceLROnPlateau but
                                    #   deterministic (no val dependency)
        patience=30,                # Early stopping: halt if val mAP doesn't
                                    # improve for 30 epochs.
                                    # 30 × 625 = 18,750 iterations
                                    # This is generous enough to survive
                                    # temporary plateaus but tight enough
                                    # to catch real overfitting
        save_period=10,             # Checkpoint every 10 epochs — lets you
                                    # pick the best one if early stopping
                                    # triggers too late
        verbose=True,
        seed=42,                    # Reproducibility
    )

    # ── Report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("10K TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best weights:  runs/ecosort/train_10k/weights/best.pt")
    print(f"Last weights:  runs/ecosort/train_10k/weights/last.pt")
    print(f"Results CSV:   runs/ecosort/train_10k/results.csv")
    print(f"Curves:        runs/ecosort/train_10k/results.png")
    print()
    print("Next steps:")
    print("  1. python evaluate.py --model runs/ecosort/train_10k/weights/best.pt --data data_10k.yaml")
    print("  2. python inference.py --weights runs/ecosort/train_10k/weights/best.pt --source 0")
    print("  3. python export.py --model runs/ecosort/train_10k/weights/best.pt --format onnx")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train()
