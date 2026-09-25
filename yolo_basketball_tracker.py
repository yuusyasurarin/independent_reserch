"""Ultralytics YOLO + ByteTrackによるバスケットボール追跡システム。

標準COCO学習済みモデルでは ``sports ball`` クラスだけを検出・追跡する。
独自学習済みモデル（best.pt）も ``--model`` で指定できる。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

# Ultralyticsがユーザーディレクトリ外へ設定を書かず、プロジェクト内で完結するようにする。
PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = PROJECT_DIR / ".ultralytics"
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT))
MATPLOTLIB_ROOT = PROJECT_DIR / ".matplotlib"
MATPLOTLIB_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_ROOT))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


WINDOW_NAME = "YOLO Basketball Tracking"
TRAIL_LENGTH = 30


@dataclass
class Detection:
    track_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


@dataclass
class TrackingStats:
    frames: int = 0
    detected_frames: int = 0
    total_detections: int = 0
    unique_ids: set[int] | None = None

    def __post_init__(self) -> None:
        self.unique_ids = set()

    def update(self, detections: list[Detection]) -> None:
        self.frames += 1
        if detections:
            self.detected_frames += 1
        self.total_detections += len(detections)
        self.unique_ids.update(item.track_id for item in detections if item.track_id >= 0)

    @property
    def detection_rate(self) -> float:
        return 100.0 * self.detected_frames / self.frames if self.frames else 0.0


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def resolve_ball_class(model: YOLO, requested_class: int | None) -> int:
    if requested_class is not None:
        return requested_class
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    normalized = [(int(index), str(name).lower().strip()) for index, name in items]
    for candidate in ("sports ball", "basketball", "ball"):
        for index, name in normalized:
            if name == candidate:
                return index
    available = ", ".join(f"{index}:{name}" for index, name in normalized)
    raise RuntimeError(
        "モデル内に sports ball / basketball / ball クラスがありません。"
        f"--class-id を指定してください。利用可能クラス: {available}"
    )


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video is not None:
        capture = cv2.VideoCapture(str(args.video))
        source = str(args.video)
    else:
        capture = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(args.camera_index)
        source = f"camera {args.camera_index}"
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"映像を開けませんでした: {source}")
    return capture


def extract_detections(result: Any) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    coordinates = boxes.xyxy.int().cpu().tolist()
    confidences = boxes.conf.float().cpu().tolist()
    if boxes.id is None:
        track_ids = [-1] * len(coordinates)
    else:
        track_ids = boxes.id.int().cpu().tolist()
    return [
        Detection(int(track_id), float(confidence), *map(int, xyxy))
        for xyxy, confidence, track_id in zip(coordinates, confidences, track_ids)
    ]


def draw_detections(
    frame: Any,
    detections: list[Detection],
    histories: dict[int, deque[tuple[int, int]]],
    inference_ms: float,
    display_fps: float,
    stats: TrackingStats,
    device: str,
) -> None:
    for detection in detections:
        center = detection.center
        history_key = detection.track_id if detection.track_id >= 0 else -1
        histories[history_key].append(center)
        label_id = str(detection.track_id) if detection.track_id >= 0 else "pending"
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"ball ID:{label_id} {detection.confidence:.2f}",
            (detection.x1, max(20, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        points = list(histories[history_key])
        for start, end in zip(points, points[1:]):
            cv2.line(frame, start, end, (255, 100, 0), 2)

    state = "DETECTED" if detections else "NOT DETECTED"
    color = (0, 255, 0) if detections else (0, 0, 255)
    cv2.putText(
        frame,
        f"YOLO + ByteTrack | {state} | {device}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"FPS {display_fps:.1f} | inference {inference_ms:.1f} ms | detection {stats.detection_rate:.1f}%",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "S: snapshot  Q/ESC: quit",
        (15, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def create_writer(path: Path, frame: Any, source_fps: float, fallback_fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else fallback_fps
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"結果動画の保存先を開けませんでした: {path}")
    return writer


def print_summary(stats: TrackingStats, elapsed: float, output_path: Path) -> None:
    average_fps = stats.frames / elapsed if elapsed > 0 else 0.0
    print("\n--- YOLO追跡結果 ---")
    print(f"処理フレーム数: {stats.frames}")
    print(f"ボール検出フレーム率: {stats.detection_rate:.2f}%")
    print(f"総検出数: {stats.total_detections}")
    print(f"追跡ID数: {len(stats.unique_ids)}")
    print(f"平均処理速度: {average_fps:.2f} FPS")
    print(f"CSV: {output_path.resolve()}")


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device == "cpu":
        print("警告: CUDAが利用できないためCPUで実行します。リアルタイム性能が低下します。")
    print(f"モデルを読み込んでいます: {args.model}")
    model = YOLO(args.model)
    ball_class = resolve_ball_class(model, args.class_id)
    print(f"対象クラス: {ball_class} ({model.names[ball_class]}) / device={device}")

    capture = open_capture(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stats = TrackingStats()
    histories: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    video_writer: cv2.VideoWriter | None = None
    raw_writer: cv2.VideoWriter | None = None
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else args.fps
    print(f"入力FPS（申告値）: {source_fps:.2f} / CSV: {output_path.resolve()}")
    frame_number = 0
    started = time.perf_counter()
    last_tick = started
    smoothed_fps = 0.0
    tracker_config = f"{args.tracker}.yaml"

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame", "elapsed_seconds", "detected", "track_id", "confidence",
                "center_x", "center_y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                "inference_ms", "processing_fps", "source_seconds",
            ]
        )
        try:
            while True:
                grabbed, frame = capture.read()
                if not grabbed:
                    if args.video is None:
                        print("フレーム取得に失敗しました。カメラ接続を確認してください。")
                    break
                frame_number += 1
                acquired = time.perf_counter()
                source_seconds = ((frame_number - 1) / source_fps if args.video is not None
                                  else acquired - started)
                if args.video is not None:
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if math.isfinite(timestamp) and timestamp > 0:
                        source_seconds = timestamp
                if args.raw_record is not None:
                    if raw_writer is None:
                        raw_writer = create_writer(args.raw_record, frame, source_fps, args.fps)
                    raw_writer.write(frame)
                inference_started = time.perf_counter()
                result = model.track(
                    frame,
                    persist=True,
                    tracker=tracker_config,
                    classes=[ball_class],
                    conf=args.confidence,
                    iou=args.iou,
                    imgsz=args.image_size,
                    device=device,
                    quantize=16 if args.half and device != "cpu" else None,
                    verbose=False,
                )[0]
                inference_ms = 1000.0 * (time.perf_counter() - inference_started)
                detections = extract_detections(result)

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_tick, 1e-9)
                smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
                last_tick = now
                stats.update(detections)
                draw_detections(frame, detections, histories, inference_ms, smoothed_fps, stats, device)

                if detections:
                    for detection in detections:
                        center_x, center_y = detection.center
                        writer.writerow(
                            [
                                frame_number, round(now - started, 6), 1, detection.track_id,
                                round(detection.confidence, 6), center_x, center_y,
                                detection.x1, detection.y1, detection.x2, detection.y2,
                                round(inference_ms, 3), round(smoothed_fps, 3), round(source_seconds, 6),
                            ]
                        )
                else:
                    writer.writerow(
                        [frame_number, round(now - started, 6), 0, "", "", "", "", "", "", "", "",
                         round(inference_ms, 3), round(smoothed_fps, 3), round(source_seconds, 6)]
                    )
                csv_file.flush()

                if args.record is not None:
                    if video_writer is None:
                        video_writer = create_writer(
                            Path(args.record), frame, capture.get(cv2.CAP_PROP_FPS), args.fps
                        )
                    video_writer.write(frame)

                if args.max_frames and frame_number >= args.max_frames:
                    break
                if not args.no_display:
                    cv2.imshow(WINDOW_NAME, frame)
                key = (cv2.waitKey(1) & 0xFF) if not args.no_display else -1
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_path = snapshot_dir / f"yolo_{stamp}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    print(f"画像を保存しました: {image_path}")
        finally:
            capture.release()
            if video_writer is not None:
                video_writer.release()
            if raw_writer is not None:
                raw_writer.release()
            cv2.destroyAllWindows()

    print_summary(stats, time.perf_counter() - started, output_path)


def parse_args(mode: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + ByteTrackによるバスケットボール追跡")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera-index", type=int, default=0, help="カメラ番号（既定: 0）")
    source.add_argument("--video", type=Path, help="カメラの代わりに使用する動画")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLOモデル（独自best.ptも指定可能）")
    parser.add_argument("--class-id", type=int, help="追跡対象クラスID（未指定時はballクラスを検索）")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort"), default="bytetrack")
    parser.add_argument("--confidence", type=float, default=0.10, help="検出信頼度閾値")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU閾値")
    parser.add_argument("--image-size", type=int, default=960, help="YOLO入力画像サイズ")
    parser.add_argument("--device", default="auto", help="auto、cpu、0など")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="GPUでFP16を使用")
    parser.add_argument("--width", type=int, default=1920, help="要求するカメラ横解像度")
    parser.add_argument("--height", type=int, default=1080, help="要求するカメラ縦解像度")
    parser.add_argument("--fps", type=float, default=60.0, help="要求するカメラFPS")
    parser.add_argument("--output", default="results/yolo/yolo_tracking.csv", help="CSV出力先")
    parser.add_argument("--record", type=Path, help="可視化済みMP4の保存先")
    parser.add_argument("--raw-record", type=Path, help="描画前の映像の保存先")
    parser.add_argument("--no-display", action="store_true", help="画面を表示しない")
    parser.add_argument("--max-frames", type=int, default=0, help="処理フレーム数の上限（0:制限なし）")
    parser.add_argument("--snapshot-dir", default="captures", help="画像保存先")
    if mode is not None:
        run_dir = PROJECT_DIR / "results" / "yolo" / (mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        parser.set_defaults(output=str(run_dir / "tracking.csv"), record=run_dir / "tracked.mp4",
                            snapshot_dir=str(run_dir / "snapshots"),
                            raw_record=run_dir / "raw.mp4" if mode == "live" else None,
                            model=str(PROJECT_DIR / "yolo26n.pt"))
    args = parser.parse_args()
    if mode == "live" and args.video is not None:
        parser.error("リアルタイム用では --camera-index を指定してください")
    if mode == "video" and args.video is None:
        parser.error("録画動画用では --video を指定してください")
    if mode is not None and args.tracker != "bytetrack":
        parser.error("この実験プログラムはByteTrack専用です")
    if args.video is not None and not args.video.is_file():
        parser.error(f"動画が見つかりません: {args.video}")
    paths = [Path(p).resolve() for p in (args.video, args.output, args.record, args.raw_record) if p is not None]
    if len(paths) != len(set(paths)):
        parser.error("入力動画・CSV・結果動画・元映像は別のパスを指定してください")
    if not math.isfinite(args.fps) or args.fps <= 0 or args.width <= 0 or args.height <= 0 or args.max_frames < 0:
        parser.error("FPS・幅・高さは正の値、max-framesは0以上にしてください")
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence は0より大きく1以下にしてください")
    if not 0.0 < args.iou <= 1.0:
        parser.error("--iou は0より大きく1以下にしてください")
    if args.image_size <= 0:
        parser.error("--image-size は正の整数にしてください")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("終了しました。")
    except (RuntimeError, OSError) as error:
        cv2.destroyAllWindows()
        raise SystemExit(f"エラー: {error}") from error
