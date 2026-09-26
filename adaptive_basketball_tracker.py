"""Video experiment runner for uncertainty/rim-aware YOLO + KCF + Kalman."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--ring", type=float, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    p.add_argument("--mode", choices=("adaptive", "fixed", "yolo", "no-ring", "no-uncertainty"), default="adaptive")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--class-id", type=int)
    p.add_argument("--person-class-id", type=int)
    p.add_argument("--confidence", type=float, default=0.15)
    p.add_argument("--image-size", type=int, default=960)
    p.add_argument("--device", default="auto")
    p.add_argument("--detect-interval", type=int, default=6)
    p.add_argument("--max-prediction", type=int, default=12)
    p.add_argument("--uncertainty-threshold", type=float, default=0.8)
    p.add_argument("--ring-enter", type=float, default=1.5)
    p.add_argument("--ring-exit", type=float, default=2.0)
    p.add_argument("--fps", type=float, help="Override source FPS for missing/incorrect metadata")
    p.add_argument("--output", type=Path, help="New experiment directory (must not already exist)")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--record", action="store_true")
    return p


def find_class(names, requested, aliases, required=True):
    items = dict(names) if isinstance(names, dict) else dict(enumerate(names))
    if requested is not None:
        if requested not in items:
            raise ValueError(f"Unknown class ID {requested}: {items}")
        return requested
    for alias in aliases:
        for index, name in items.items():
            if str(name).strip().lower() == alias:
                return int(index)
    if required:
        raise ValueError(f"Ball class not found; specify --class-id: {items}")
    return None


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    import cv2
    import numpy as np
    import torch
    root = Path(__file__).resolve().parent
    for env, directory in (("YOLO_CONFIG_DIR", ".ultralytics"), ("MPLCONFIGDIR", ".matplotlib"), ("TORCH_HOME", ".torch")):
        (root / directory).mkdir(exist_ok=True)
        os.environ.setdefault(env, str(root / directory))
    import ultralytics
    from ultralytics import YOLO
    from adaptive_tracking import AdaptiveTracker, Config

    if not args.video.is_file():
        raise ValueError(f"Video does not exist: {args.video}")
    model = YOLO(args.model)
    ball = find_class(model.names, args.class_id, ("sports ball", "basketball", "ball"))
    person = find_class(model.names, args.person_class_id, ("person", "player"), required=False)
    if person is None:
        print("Person class unavailable: player-overlap occlusion cue is disabled.")
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    def detector(frame):
        result = model.predict(frame, classes=[ball] + ([person] if person is not None else []),
                               conf=args.confidence, imgsz=args.image_size, device=device,
                               verbose=False)[0]
        balls, people = [], []
        if result.boxes is not None:
            for xyxy, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                x1, y1, x2, y2 = xyxy
                box = (x1, y1, x2-x1, y2-y1)
                if int(cls) == ball:
                    balls.append((box, score))
                elif int(cls) == person:
                    people.append(box)
        return balls, people

    factory = getattr(cv2, "TrackerKCF_create", None) or getattr(getattr(cv2, "legacy", None), "TrackerKCF_create", None)
    if factory is None and args.mode != "yolo":
        raise RuntimeError("KCF requires opencv-contrib-python in the active environment.")

    def kcf_factory(frame, box):
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(math.ceil(x+bw))), min(h, int(math.ceil(y+bh)))
        if x2-x1 < 2 or y2-y1 < 2:
            return None
        tracker = factory()
        ok = tracker.init(frame, (x1, y1, x2-x1, y2-y1))
        return None if ok is False else tracker

    capture = cv2.VideoCapture(str(args.video))
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open video")
        fps = args.fps or capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Source FPS unavailable; specify --fps")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video contains no readable frames")
        h, w = frame.shape[:2]
        x, y, rw, rh = args.ring
        if x < 0 or y < 0 or x+rw > w or y+rh > h:
            raise ValueError("Ring rectangle must be inside the input frame")
        config = Config(mode=args.mode, detect_interval=args.detect_interval, max_prediction=args.max_prediction,
                        uncertainty_threshold=args.uncertainty_threshold, ring_enter=args.ring_enter, ring_exit=args.ring_exit)
        tracker = AdaptiveTracker(detector, kcf_factory, tuple(args.ring), config)
        # Warm-up excluded from processing timings; every method uses the same detector settings.
        detector(frame)
        output = args.output or root / "results" / "adaptive" / f"{time.time_ns()}_{args.mode}"
        output.mkdir(parents=True, exist_ok=False)
        weight_path = Path(getattr(model, "ckpt_path", args.model))
        metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        config=asdict(config), video=str(args.video.resolve()), video_sha256=file_hash(args.video),
                        model_sha256=file_hash(weight_path) if weight_path.is_file() else None,
                        source_fps=fps, width=w, height=h, ball_class=ball, person_class=person,
                        device=device, opencv=cv2.__version__, numpy=np.__version__,
                        ultralytics=ultralytics.__version__, torch=torch.__version__)
        (output / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.record:
            writer = cv2.VideoWriter(str(output / "tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Cannot create output video")
        index, csv_writer = 0, None
        with (output / "tracking.csv").open("w", newline="", encoding="utf-8") as handle:
            while ok:
                start = time.perf_counter()
                row = tracker.step(frame, index, 1/fps)
                row.update(timestamp_s=index/fps, processing_ms=(time.perf_counter()-start)*1000)
                if csv_writer is None:
                    csv_writer = csv.DictWriter(handle, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                if args.record or not args.no_preview:
                    cv2.rectangle(frame, (int(x), int(y)), (int(x+rw), int(y+rh)), (255, 255, 0), 2)
                    if row["center_x"] != "":
                        color = (0, 255, 0) if row["observed"] else (0, 165, 255)
                        cv2.circle(frame, (round(row["center_x"]), round(row["center_y"])), 6, color, 2)
                    cv2.putText(frame, f'{index} {row["mode"]} {row["reason"]}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                    if writer is not None:
                        writer.write(frame)
                    if not args.no_preview:
                        cv2.imshow("Adaptive basketball tracking", frame)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                index += 1
                ok, frame = capture.read()
        print(f"Saved experiment: {output.resolve()}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    p = parser()
    args = p.parse_args()
    if not all(math.isfinite(v) for v in [*args.ring, args.confidence, args.uncertainty_threshold, args.ring_enter, args.ring_exit]):
        p.error("Numeric options must be finite")
    if min(args.ring[2:]) <= 0 or args.detect_interval < 1 or args.max_prediction < 0 or args.image_size < 32:
        p.error("Invalid rectangle, interval, prediction limit, or image size")
    if not 0 < args.confidence <= 1 or not 0 < args.ring_enter < args.ring_exit or args.uncertainty_threshold <= 0:
        p.error("Invalid confidence, uncertainty threshold, or ring hysteresis")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        p.error("--fps must be positive and finite")
    from project_runtime import ensure_project_venv
    ensure_project_venv()
    run(args)


if __name__ == "__main__":
    main()
