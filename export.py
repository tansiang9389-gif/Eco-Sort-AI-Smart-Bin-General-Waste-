"""
=============================================================================
EcoSort — YOLO26 Edge Export Script
=============================================================================
Exports the trained YOLO26 model (4 classes: Plastic, Paper, Metal, Others)
to edge-optimized formats for deployment on embedded hardware.

YOLO26 DFL-FREE EXPORT ADVANTAGE:
  Previous YOLO versions (v8, v10, v11) used Distribution Focal Loss (DFL)
  in their detection heads. DFL predicts a probability distribution over
  16 discretized bounding-box offsets, requiring:
    - Softmax per box coordinate (4 × 16 = 64 values per anchor)
    - Additional conv layers for the distribution
    - ~30-40% more parameters in the detection head

  YOLO26 REMOVES DFL entirely. It uses direct coordinate regression
  (predicting x, y, w, h scalars directly), which means:
    - Simpler computation graph — fewer ops to convert/quantize
    - No softmax in detection head — softmax is SLOW on MCUs without FPU
    - Smaller model file — fewer params = less Flash/ROM needed
    - Better INT8 quantization — direct regression quantizes cleanly
      (softmax distributions lose precision at int8)
    - Broader format support — TFLite Micro, CoreML, OpenVINO all
      handle the simplified graph without custom ops

Supported export formats:
  ONNX       → CPU/GPU inference, broadest compatibility
  TensorRT   → NVIDIA Jetson / GPU edge devices
  TFLite     → ARM Cortex-M / ESP32 / Android
  OpenVINO   → Intel edge devices (NCS2, Movidius)
  CoreML     → Apple devices (iOS/macOS)
  NCNN       → Mobile / embedded ARM (Tencent framework)

Usage:
    python export.py                                   # ONNX (default)
    python export.py --format tflite --int8            # INT8 TFLite for ESP32
    python export.py --format engine --half            # TensorRT FP16 for Jetson
    python export.py --model runs/ecosort/train_10k/weights/best.pt --format onnx
=============================================================================
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="EcoSort YOLO26 Edge Export")
    parser.add_argument("--model", type=str,
                        default="runs/ecosort/train/weights/best.pt",
                        help="Path to trained YOLO26 best weights.")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "engine", "tflite", "openvino",
                                 "coreml", "ncnn"],
                        help="Export format. 'tflite' for ESP32, 'engine' for TensorRT.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size. Use 320/256 for extreme edge constraints.")
    parser.add_argument("--half", action="store_true",
                        help="FP16 export. Halves model size.")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 quantization. Smallest size, needs calibration data.")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic input shapes (ONNX only).")
    return parser.parse_args()


def export_model(args):
    """
    Export trained YOLO26 to an edge-optimized format.

    The DFL-free architecture means the exported graph:
      - Has no softmax ops in the detection head
      - Has ~30% fewer head parameters than YOLOv11
      - Quantizes cleanly to INT8 without distribution-collapse artifacts
      - Requires no custom ops for any target framework
    """

    model = YOLO(args.model)

    print("=" * 70)
    print(f"EXPORTING YOLO26 -> {args.format.upper()}")
    print(f"  Model:         {args.model}")
    print(f"  Classes:       4 (Plastic, Paper, Metal, Others)")
    print(f"  Input size:    {args.imgsz}x{args.imgsz}")
    print(f"  FP16 (half):   {args.half}")
    print(f"  INT8 quantize: {args.int8}")
    print(f"  Dynamic batch: {args.dynamic}")
    print("=" * 70)

    # ── Export ───────────────────────────────────────────────────────
    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=True,          # ONNX graph simplification (remove redundant ops)
        opset=17,               # ONNX opset 17 — well-supported across runtimes
    )

    print(f"\nExported model: {export_path}")
    print_deployment_notes(args.format)


def print_deployment_notes(fmt: str):
    """Print hardware-specific deployment guidance for each format."""

    notes = {
        "onnx": """
    ONNX Deployment Notes:
    ----------------------
    * Use onnxruntime (CPU) or onnxruntime-gpu for inference.
    * YOLO26's DFL-free head = pure standard ONNX, no custom ops.
    * Output tensor shape: (1, 300, 6) — direct detections, no NMS needed.
      Each row: [x1, y1, x2, y2, confidence, class_id]
    * For further optimization: onnxruntime quantization tools for INT8.
        """,
        "tflite": """
    TFLite Deployment Notes (ESP32 / ARM Cortex-M):
    ------------------------------------------------
    * INT8 quantized TFLite is the target for ESP32-S3.
    * ESP32-S3: 512KB SRAM + 8MB PSRAM — yolo26n INT8 (~1.5MB) fits.
    * Deploy with ESP-IDF + TFLite Micro C++ API:
        #include "tensorflow/lite/micro/micro_interpreter.h"
    * Flash the .tflite to ESP32 SPI Flash partition.
    * Camera: OV2640/OV5640 module via ESP32-CAM interface.
    * Frame pipeline: Camera -> JPEG decode -> Resize 320x320 -> Quantize
      -> TFLite Invoke -> Parse (1,300,6) output -> Serial/WiFi.
    * YOLO26's NMS-free output is critical: ESP32 cannot run NMS sorting.
      The 300x6 tensor is directly usable without post-processing.
    * DFL-free head = no softmax quantization issues — INT8 is clean.
    * Target: ~200-500ms/frame on ESP32-S3 at 320x320.
        """,
        "engine": """
    TensorRT Deployment Notes (NVIDIA Jetson):
    ------------------------------------------
    * TensorRT engines are device-specific — export ON the target Jetson.
    * FP16 (--half) recommended for Jetson Nano/Orin Nano.
    * YOLO26n + TensorRT FP16 achieves <5ms inference on Jetson Orin.
    * Output: (1, 300, 6) — zero post-processing latency.
    * Use DeepStream SDK for multi-camera pipeline integration.
        """,
        "openvino": """
    OpenVINO Deployment Notes (Intel Edge):
    ----------------------------------------
    * Target: Intel NCS2 (Movidius), Intel iGPU, or CPU.
    * Benchmark: benchmark_app -m model.xml -d MYRIAD
    * YOLO26's simplified head exports cleanly to IR format.
        """,
        "coreml": """
    CoreML Deployment Notes (Apple):
    ---------------------------------
    * Deploy via Vision framework or Core ML API on iOS/macOS.
    * Use Neural Engine on Apple Silicon for best performance.
        """,
        "ncnn": """
    NCNN Deployment Notes (Mobile/Embedded ARM):
    ---------------------------------------------
    * Lightweight framework by Tencent — runs on ARM without dependencies.
    * Suitable for Android, Raspberry Pi, or custom ARM boards.
    * YOLO26's DFL-free head avoids NCNN's softmax performance issues.
        """,
    }

    print(notes.get(fmt, ""))


def export_all_formats(model_path: str, imgsz: int = 640):
    """
    Convenience: export to all major edge formats at once.
    Useful for benchmarking across deployment targets.
    """
    model = YOLO(model_path)

    formats = {
        "onnx":      {"half": False, "int8": False},
        "onnx_fp16": {"half": True,  "int8": False},
        "tflite":    {"half": False, "int8": True},
        "openvino":  {"half": True,  "int8": False},
        "ncnn":      {"half": True,  "int8": False},
    }

    for fmt_key, kwargs in formats.items():
        fmt = fmt_key.split("_")[0]
        print(f"\n{'=' * 50}")
        print(f"Exporting: {fmt_key}")
        try:
            path = model.export(format=fmt, imgsz=imgsz, simplify=True, **kwargs)
            print(f"  -> {path}")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    args = parse_args()
    export_model(args)
