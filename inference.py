"""
=============================================================================
EcoSort — YOLO26 Real-Time Waste Detection & Display
=============================================================================
Detects and classifies waste into 4 categories, drawing bounding boxes
and labels on-screen in real time.

Classes: Plastic (0), Paper (1), Metal (2), Others (3)

YOLO26 NMS-FREE ARCHITECTURE:
  Unlike YOLOv8/v11, YOLO26 does NOT use Non-Maximum Suppression (NMS).
  The model outputs up to 300 direct detections per image with no
  duplicate-removal post-processing. This means:
    - No NMS sorting latency (saves 2-10ms per frame on edge devices)
    - No conf/iou NMS thresholds to tune
    - Deterministic output shape: always (1, 300, 6)
    - Each detection = [x1, y1, x2, y2, confidence, class_id]

  In this script, we simply threshold by confidence — no NMS call needed.

Usage:
    python inference.py --source 0                     # USB webcam (live)
    python inference.py --source picam                 # Raspberry Pi Camera
    python inference.py --source image.jpg             # Single image
    python inference.py --source video.mp4             # Video file
    python inference.py --source 0 --conf_thresh 0.4   # Custom confidence
    python inference.py --source 0 --no-show --save    # Headless + save video
=============================================================================
"""

import argparse
import time
from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not installed. Run: pip install ultralytics")


# =============================================================================
# 4-CLASS DEFINITIONS
# =============================================================================
CLASS_NAMES = {
    0: "Plastic",
    1: "Paper",
    2: "Metal",
    3: "Others",
}

# BGR colors for each class (visually distinct)
CLASS_COLORS = {
    0: (0, 165, 255),    # Orange  — Plastic
    1: (0, 200, 0),      # Green   — Paper
    2: (255, 200, 0),    # Cyan    — Metal
    3: (128, 128, 128),  # Gray    — Others
}

# Bin sorting recommendation per class
BIN_MAP = {
    "Plastic": "RECYCLING (Yellow Bin)",
    "Paper":   "RECYCLING (Blue Bin)",
    "Metal":   "RECYCLING (Yellow Bin)",
    "Others":  "GENERAL WASTE (Black Bin)",
}


# =============================================================================
# Rolling Confidence Smoother — prevents flickering detections on video
# =============================================================================
class RollingConfidence:
    """
    Smooths per-class confidence over a sliding window of frames.
    Prevents single-frame false positives from triggering actions.
    A detection is only "confirmed" when the rolling average exceeds
    the smooth_thresh for that class.
    """
    def __init__(self, window=6):
        self.window = window
        self.buffers = defaultdict(lambda: deque(maxlen=self.window))

    def add(self, cls, conf):
        self.buffers[cls].append(conf)

    def mean(self, cls):
        buf = self.buffers[cls]
        return float(sum(buf) / len(buf)) if buf else 0.0

    def clear(self):
        self.buffers.clear()


# =============================================================================
# ARGUMENT PARSER
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="EcoSort YOLO26 Real-Time Inference")
    p.add_argument("--weights", default="runs/ecosort/train/weights/best.pt",
                   help="Path to trained YOLO26 weights (.pt file)")
    p.add_argument("--source", default="0",
                   help="'0' for USB webcam, 'picam' for Pi Camera, or file path")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size (use 320 for faster Pi inference)")
    p.add_argument("--conf_thresh", type=float, default=0.35,
                   help="Minimum confidence for a detection to be drawn")
    p.add_argument("--smooth_window", type=int, default=6,
                   help="Rolling average window (frames) for smoothing")
    p.add_argument("--smooth_thresh", type=float, default=0.5,
                   help="Rolling avg must exceed this to 'confirm' a detection")
    p.add_argument("--cooldown", type=float, default=1.5,
                   help="Seconds between repeated console logs of same class")
    p.add_argument("--resize_width", type=int, default=640,
                   help="Resize frame width for performance. 0 to disable.")
    p.add_argument("--save", action="store_true",
                   help="Save output video to runs/ecosort/inference/output.mp4")
    p.add_argument("--no-show", action="store_true",
                   help="Headless mode — no display window (for servers/CI)")
    return p.parse_args()


# =============================================================================
# DETECTION PROCESSING — NMS-FREE
# =============================================================================
def process_detections(results, conf_thresh=0.25):
    """
    Extract detections from YOLO26's NMS-free output.

    YOLO26 outputs up to 300 detections per image with NO duplicate
    removal. Each detection is already a unique object prediction.
    We only need to threshold by confidence — no NMS sorting needed.

    Returns a list of detection dicts with keys:
      xyxy:     [x1, y1, x2, y2] pixel coordinates
      conf:     float confidence score
      cls:      int class index (0-3)
      label:    str class name ("Plastic", "Paper", "Metal", "Others")
    """
    detections = []

    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    # Extract tensors from YOLO26 output — no NMS post-processing
    xyxys = boxes.xyxy.cpu().numpy()     # Bounding box coordinates
    confs = boxes.conf.cpu().numpy()     # Confidence scores
    clss = boxes.cls.cpu().numpy().astype(int)  # Class indices

    for xyxy, conf, cls_id in zip(xyxys, confs, clss):
        if conf < conf_thresh:
            continue
        detections.append({
            "xyxy": xyxy,
            "conf": float(conf),
            "cls": int(cls_id),
            "label": CLASS_NAMES.get(int(cls_id), "Unknown"),
        })

    return detections


# =============================================================================
# ON-SCREEN DRAWING
# =============================================================================
def draw_frame(frame, detections, confirmed, fps=0.0):
    """
    Draw bounding boxes, class labels, confidence scores, bin
    recommendation, and a summary panel onto the frame.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Draw bounding boxes + labels for each detection ──────────────
    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        color = CLASS_COLORS.get(det["cls"], (255, 255, 255))
        label_text = f"{det['label']} {det['conf']:.0%}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with filled background for readability
        (lw, lh), _ = cv2.getTextSize(label_text, font, 0.55, 2)
        label_y = y1 - 8 if y1 - lh - 10 > 0 else y2 + lh + 8
        cv2.rectangle(frame, (x1, label_y - lh - 4),
                      (x1 + lw + 6, label_y + 4), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, label_y),
                    font, 0.55, (0, 0, 0), 2)

        # Bin recommendation below the box
        bin_text = BIN_MAP.get(det["label"], "")
        cv2.putText(frame, bin_text, (x1, y2 + 16), font, 0.35, color, 1)

    # ── Summary panel (top-left corner) ──────────────────────────────
    panel_h = 40 + max(len(confirmed), 1) * 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (300, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "EcoSort AI", (14, 30), font, 0.6, (255, 255, 255), 2)

    if confirmed:
        y_off = 54
        for c in confirmed:
            color = CLASS_COLORS.get(c["cls"], (255, 255, 255))
            text = f"{c['label']} ({c['conf']:.0%})"
            cv2.putText(frame, text, (18, y_off), font, 0.48, color, 1)
            y_off += 26
    else:
        cv2.putText(frame, "Scanning...", (18, 54),
                    font, 0.45, (100, 100, 255), 1)

    # ── FPS counter (top-right corner) ───────────────────────────────
    fps_text = f"{fps:.1f} FPS"
    (fw, fh), _ = cv2.getTextSize(fps_text, font, 0.6, 2)
    cv2.rectangle(frame, (w - fw - 18, 6), (w - 6, 6 + fh + 12),
                  (20, 20, 20), -1)
    cv2.putText(frame, fps_text, (w - fw - 14, 6 + fh + 5),
                font, 0.6, (0, 255, 0), 2)

    return frame


# =============================================================================
# Pi CAMERA SUPPORT (Raspberry Pi with libcamera)
# =============================================================================
def open_picamera(width=640, height=480):
    """Open Raspberry Pi camera using picamera2 (libcamera backend)."""
    try:
        from picamera2 import Picamera2
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        return picam
    except ImportError:
        print("picamera2 not installed. Run: sudo apt install python3-picamera2")
        print("Falling back to USB webcam (index 0)...")
        return None


# =============================================================================
# SINGLE IMAGE INFERENCE
# =============================================================================
def run_image(args, model):
    """Run inference on a single image and display/save results."""
    # YOLO26 NMS-free: no NMS parameters needed in the predict call
    results = model(args.source, imgsz=args.imgsz, verbose=False)[0]
    detections = process_detections(results, args.conf_thresh)

    frame = results.orig_img.copy()
    confirmed = [{
        "cls": d["cls"], "label": d["label"],
        "conf": d["conf"], "xyxy": d["xyxy"],
    } for d in detections]

    # Estimate FPS from inference speed
    speed = results.speed
    fps = 1000.0 / max(speed.get("inference", 1), 1)
    frame = draw_frame(frame, detections, confirmed, fps)

    # Console output
    print(f"\nResults for: {args.source}")
    for d in detections:
        bin_rec = BIN_MAP.get(d["label"], "")
        print(f"  {d['label']:>10s}  conf={d['conf']:.0%}  -> {bin_rec}")
    if not detections:
        print("  No waste detected.")

    # Display
    show = not getattr(args, "no_show", False)
    if show:
        cv2.imshow("EcoSort AI", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save
    if args.save:
        out_dir = Path("runs/ecosort/inference")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"result_{Path(args.source).name}"
        cv2.imwrite(str(out_path), frame)
        print(f"  Saved: {out_path}")


# =============================================================================
# MAIN VIDEO INFERENCE LOOP
# =============================================================================
def run(args):
    """
    Main inference loop for webcam / video / Pi Camera.

    YOLO26 NMS-Free flow:
      Frame → YOLO26 → 300 direct detections → confidence threshold → draw
      No NMS sorting step. This saves 2-10ms per frame on edge hardware.
    """
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # ── Open video source ────────────────────────────────────────────
    use_picam = False
    picam = None

    if args.source == "picam":
        picam = open_picamera(args.resize_width or 640, 480)
        if picam:
            use_picam = True
        else:
            cap = cv2.VideoCapture(0)
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        # Image or video file
        source_path = Path(args.source)
        if source_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            run_image(args, model)
            return
        cap = cv2.VideoCapture(args.source)

    if not use_picam and not cap.isOpened():
        print(f"ERROR: Could not open source: {args.source}")
        return

    print("EcoSort AI running. Press 'q' to quit, 'c' to clear buffer.")
    rolling = RollingConfidence(window=args.smooth_window)
    cooldown_last = {}
    show = not getattr(args, "no_show", False)
    writer = None
    prev_time = time.perf_counter()
    frame_count = 0

    try:
        while True:
            # ── Grab frame ───────────────────────────────────────────
            if use_picam:
                frame = picam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    if args.source.isdigit():
                        time.sleep(0.1)
                        continue
                    break

            # Resize for performance (especially on Pi)
            if args.resize_width and frame.shape[1] > args.resize_width:
                scale_f = args.resize_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale_f, fy=scale_f)

            frame_count += 1

            # ── YOLO26 NMS-free inference ────────────────────────────
            # No agnostic_nms, no iou threshold — YOLO26 outputs direct
            # detections. We just threshold by confidence.
            results = model(frame, imgsz=args.imgsz, verbose=False)[0]
            detections = process_detections(results, args.conf_thresh)

            # ── Rolling confidence smoothing ─────────────────────────
            # Keeps the best detection per class, smooths over frames
            best_per_class = {}
            for det in detections:
                cls = det["cls"]
                if cls not in best_per_class or det["conf"] > best_per_class[cls]["conf"]:
                    best_per_class[cls] = det

            confirmed = []
            for cls, det in best_per_class.items():
                rolling.add(cls, det["conf"])
                avg_conf = rolling.mean(cls)
                if avg_conf >= args.smooth_thresh:
                    confirmed.append({
                        "cls": cls,
                        "label": det["label"],
                        "conf": avg_conf,
                        "xyxy": det["xyxy"],
                    })

            # ── Console log with cooldown ────────────────────────────
            now = time.time()
            for c in confirmed:
                label = c["label"]
                if now - cooldown_last.get(label, 0) >= args.cooldown:
                    cooldown_last[label] = now
                    bin_rec = BIN_MAP.get(label, "?")
                    print(f"[DETECTED] {label} ({c['conf']:.0%}) -> {bin_rec}")

            # ── Calculate FPS ────────────────────────────────────────
            curr_time = time.perf_counter()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # ── Draw on frame ────────────────────────────────────────
            frame = draw_frame(frame, detections, confirmed, fps)

            # ── Display ──────────────────────────────────────────────
            if show:
                cv2.imshow("EcoSort AI", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:   # q or ESC
                    break
                elif key == ord("c"):
                    rolling.clear()
                    print("Buffer cleared.")

            # ── Save video ───────────────────────────────────────────
            if args.save:
                if writer is None:
                    out_dir = Path("runs/ecosort/inference")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fh, fw = frame.shape[:2]
                    out_path = out_dir / "output.mp4"
                    writer = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0, (fw, fh)
                    )
                writer.write(frame)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if writer:
            writer.release()
        if use_picam and picam:
            picam.stop()
        elif not use_picam:
            cap.release()
        if show:
            cv2.destroyAllWindows()

    print(f"Done. {frame_count} frames processed.")


if __name__ == "__main__":
    run(parse_args())
