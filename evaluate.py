"""
=============================================================================
EcoSort — YOLO26 Evaluation & Metrics Script
=============================================================================
Evaluates the trained YOLO26 model and produces:
  - mAP@0.50 and mAP@0.50:0.95 (primary accuracy metrics)
  - Per-class AP for all 4 classes (Plastic, Paper, Metal, Others)
  - Confusion matrix (raw counts + normalized percentages)
  - Precision-Recall curves, F1 curves
  - 98% accuracy threshold pass/fail check
  - Inference speed breakdown (preprocess / inference / postprocess)

YOLO26 NMS-FREE NOTE:
  Validation is also NMS-free. The model outputs up to 300 direct
  detections per image. The standard COCO mAP evaluation protocol
  is applied to these direct outputs.

Usage:
    python evaluate.py
    python evaluate.py --split test
    python evaluate.py --model runs/ecosort/train_10k/weights/best.pt --data data_10k.yaml
    python evaluate.py --model runs/ecosort/train_60k/weights/best.pt --data data_60k.yaml --save
=============================================================================
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# 4 waste classes
CLASS_NAMES = ["Plastic", "Paper", "Metal", "Others"]


def parse_args():
    parser = argparse.ArgumentParser(description="EcoSort YOLO26 Evaluation")
    parser.add_argument("--model", type=str,
                        default="runs/ecosort/train/weights/best.pt",
                        help="Path to trained YOLO26 weights.")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="Path to dataset YAML (data.yaml, data_10k.yaml, or data_60k.yaml).")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for evaluation.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for evaluation.")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for mAP calculation (standard: 0.001).")
    parser.add_argument("--save", action="store_true",
                        help="Save evaluation results to JSON file.")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU.")
    return parser.parse_args()


def evaluate(args):
    """
    Run YOLO26 validation to compute mAP50, mAP50-95, per-class metrics,
    and confusion matrix. Checks against 98% accuracy target.
    """

    model = YOLO(args.model)

    # ── Run validation ───────────────────────────────────────────────
    # YOLO26's NMS-free head means validation produces direct detections.
    # plots=True generates confusion matrix, PR curves, F1 curves.
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device,
        save_json=True,          # COCO-format results for analysis
        plots=True,              # Generates all evaluation plots
        project="runs/ecosort",
        name="evaluate",
        exist_ok=True,
        verbose=True,
    )

    # ── Extract key metrics ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS — EcoSort YOLO26 Waste Detector (4 Classes)")
    print("=" * 70)

    map50 = results.box.map50       # mAP at IoU=0.50
    map50_95 = results.box.map      # mAP at IoU=0.50:0.95
    precision = results.box.mp      # Mean precision across classes
    recall = results.box.mr         # Mean recall across classes

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'mAP@0.50':<25} {map50:>10.4f}")
    print(f"{'mAP@0.50:0.95':<25} {map50_95:>10.4f}")
    print(f"{'Mean Precision':<25} {precision:>10.4f}")
    print(f"{'Mean Recall':<25} {recall:>10.4f}")

    # ── Per-class metrics ────────────────────────────────────────────
    per_class_map50 = results.box.all_ap[:, 0]  # AP@50 per class
    per_class_map = results.box.maps             # mAP@50:95 per class

    print(f"\n{'Class':<15} {'AP@50':>10} {'AP@50:95':>10}")
    print("-" * 37)
    for i, name in enumerate(CLASS_NAMES):
        if i < len(per_class_map):
            ap50 = per_class_map50[i] if i < len(per_class_map50) else 0
            ap = per_class_map[i]
            print(f"{name:<15} {ap50:>10.4f} {ap:>10.4f}")

    # ── 98% Threshold Verification ───────────────────────────────────
    print("\n" + "=" * 70)
    TARGET = 0.98
    passed = map50 >= TARGET
    status = "PASS" if passed else "FAIL"
    symbol = ">>>" if passed else "XXX"
    print(f"[{symbol}] 98% ACCURACY THRESHOLD: {status}")
    print(f"    mAP@0.50 = {map50:.4f} {'≥' if passed else '<'} {TARGET}")

    if not passed:
        gap = TARGET - map50
        print(f"\n    Gap to target: {gap:.4f}")
        print("    Recommendations:")
        print("      1. Increase dataset size (target 5,000+ images per class)")
        print("      2. Increase epochs (10K: try 200; 60K: try 100)")
        print("      3. Use yolo26s.pt (Small) for more model capacity")
        print("      4. Enable Test-Time Augmentation (TTA) during eval")
        print("      5. Audit label quality — mislabeled data caps mAP")
        print("      6. Balance class distribution (check for underrepresented classes)")

    # ── Confusion Matrix Location ────────────────────────────────────
    print(f"\nGenerated evaluation plots:")
    print(f"  runs/ecosort/evaluate/confusion_matrix.png           (raw counts)")
    print(f"  runs/ecosort/evaluate/confusion_matrix_normalized.png (percentages)")
    print(f"  runs/ecosort/evaluate/PR_curve.png                   (Precision-Recall)")
    print(f"  runs/ecosort/evaluate/F1_curve.png                   (F1 vs confidence)")
    print(f"  runs/ecosort/evaluate/P_curve.png                    (Precision vs conf)")
    print(f"  runs/ecosort/evaluate/R_curve.png                    (Recall vs conf)")

    # ── Inference Speed ──────────────────────────────────────────────
    speed = results.speed
    print(f"\nInference Speed:")
    print(f"  Preprocess:   {speed.get('preprocess', 0):.1f}ms")
    print(f"  Inference:    {speed.get('inference', 0):.1f}ms")
    print(f"  Postprocess:  {speed.get('postprocess', 0):.1f}ms  (NMS-free = near 0)")
    total_ms = sum(speed.values())
    print(f"  Total:        {total_ms:.1f}ms  ({1000/max(total_ms, 1):.0f} FPS)")

    # ── Save summary to JSON ─────────────────────────────────────────
    if args.save:
        summary = {
            "model": args.model,
            "dataset": args.data,
            "split": args.split,
            "classes": CLASS_NAMES,
            "metrics": {
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            },
            "per_class_AP50": {
                name: round(float(per_class_map50[i]), 4)
                for i, name in enumerate(CLASS_NAMES)
                if i < len(per_class_map50)
            },
            "per_class_AP50_95": {
                name: round(float(per_class_map[i]), 4)
                for i, name in enumerate(CLASS_NAMES)
                if i < len(per_class_map)
            },
            "threshold_98_passed": passed,
            "speed_ms": {k: round(v, 2) for k, v in speed.items()},
        }

        out_path = Path("runs/ecosort/evaluate/metrics_summary.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nMetrics saved to: {out_path}")

    print("=" * 70)
    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
