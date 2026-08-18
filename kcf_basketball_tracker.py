"""KCFによるリアルタイム・バスケットボール追跡システム。

KCFは高速な単一物体トラッカーであり、本研究では比較用ベースラインとして使う。
遮蔽後の自動再検出は行わず、失跡は明示的に記録して R キーで再初期化する。
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


WINDOW_NAME = "KCF Basketball Tracking"


@dataclass
class TrackingStats:
    processed: int = 0
    tracked: int = 0
    lost: int = 0
    loss_events: int = 0
    reinitializations: int = 0
    longest_streak: int = 0
    current_streak: int = 0
    was_tracked: bool | None = None

    def update(self, success: bool) -> None:
        self.processed += 1
        if success:
            self.tracked += 1
            self.current_streak += 1
            self.longest_streak = max(self.longest_streak, self.current_streak)
        else:
            self.lost += 1
            self.current_streak = 0
            if self.was_tracked is not False:
                self.loss_events += 1
        self.was_tracked = success

    @property
    def success_rate(self) -> float:
        return 100.0 * self.tracked / self.processed if self.processed else 0.0


def create_kcf(detect_threshold: float, learning_rate: float) -> Any:
    """OpenCV contribのKCFを、利用可能なAPIから生成する。"""
    params_class = getattr(cv2, "TrackerKCF_Params", None)
    factory = getattr(cv2, "TrackerKCF_create", None)
    if factory is not None:
        if params_class is None:
            return factory()
        params = params_class()
        params.detect_thresh = detect_threshold
        params.interp_factor = learning_rate
        return factory(params)

    legacy = getattr(cv2, "legacy", None)
    factory = getattr(legacy, "TrackerKCF_create", None) if legacy is not None else None
    if factory is not None:
        return factory()
    raise RuntimeError(
        "KCFが利用できません。opencv-pythonとの競合を除去し、"
        "opencv-contrib-pythonをインストールしてください。"
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


def expand_roi(
    roi: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    padding: float,
) -> tuple[int, int, int, int]:
    """小さいボールの周囲の特徴もKCFに渡すため、ROIを中心基準で拡張する。"""
    x, y, width, height = roi
    frame_height, frame_width = frame_shape[:2]
    expanded_width = max(2, round(width * padding))
    expanded_height = max(2, round(height * padding))
    center_x, center_y = x + width / 2, y + height / 2
    left = max(0, round(center_x - expanded_width / 2))
    top = max(0, round(center_y - expanded_height / 2))
    right = min(frame_width, left + expanded_width)
    bottom = min(frame_height, top + expanded_height)
    left = max(0, right - expanded_width)
    top = max(0, bottom - expanded_height)
    return left, top, right - left, bottom - top


def select_and_initialize(
    frame: Any,
    args: argparse.Namespace,
) -> tuple[Any, tuple[int, int, int, int]]:
    selection_frame = frame.copy()
    cv2.putText(
        selection_frame,
        "Select the ball, then press ENTER/SPACE",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    selected = cv2.selectROI(WINDOW_NAME, selection_frame, True, False)
    selected = tuple(int(value) for value in selected)
    if selected[2] <= 0 or selected[3] <= 0:
        raise KeyboardInterrupt("ROI selection cancelled")

    tracking_roi = expand_roi(selected, frame.shape, args.roi_padding)
    tracker = create_kcf(args.detect_threshold, args.learning_rate)
    initialized = tracker.init(frame, tracking_roi)
    if initialized is False:
        raise RuntimeError("KCFを初期化できませんでした。ROIを選び直してください。")
    return tracker, tracking_roi


def create_writer(path: Path, frame: Any, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"結果動画の保存先を開けませんでした: {path}")
    return writer


def draw_overlay(
    frame: Any,
    success: bool,
    box: tuple[float, float, float, float],
    display_fps: float,
    stats: TrackingStats,
) -> tuple[int | None, int | None]:
    center_x = center_y = None
    if success:
        x, y, width, height = (int(value) for value in box)
        center_x, center_y = x + width // 2, y + height // 2
        color = (0, 255, 0)
        state = "TRACKING"
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    else:
        color = (0, 0, 255)
        state = "LOST - press R to reselect"

    cv2.putText(frame, f"KCF | {state}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        frame,
        f"FPS {display_fps:.1f} | success {stats.success_rate:.1f}% | losses {stats.loss_events}",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "R: reselect  S: snapshot  Q/ESC: quit",
        (15, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return center_x, center_y


def print_summary(stats: TrackingStats, elapsed: float, output_path: Path) -> None:
    average_fps = stats.processed / elapsed if elapsed > 0 else 0.0
    print("\n--- KCF追跡結果 ---")
    print(f"処理フレーム数: {stats.processed}")
    print(f"追跡成功率: {stats.success_rate:.2f}%")
    print(f"失跡フレーム数: {stats.lost}")
    print(f"失跡イベント数: {stats.loss_events}")
    print(f"最長連続追跡: {stats.longest_streak} フレーム")
    print(f"再初期化回数: {stats.reinitializations}")
    print(f"平均処理速度: {average_fps:.2f} FPS")
    print(f"CSV: {output_path.resolve()}")


def run(args: argparse.Namespace) -> None:
    capture = open_capture(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    video_writer: cv2.VideoWriter | None = None
    stats = TrackingStats()
    tracker = None
    frame_number = 0
    started = time.perf_counter()
    last_tick = started
    smoothed_fps = 0.0

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame", "elapsed_seconds", "tracked", "center_x", "center_y",
                "bbox_x", "bbox_y", "bbox_width", "bbox_height", "processing_fps",
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

                if tracker is None:
                    tracker, box = select_and_initialize(frame, args)
                    success = True
                else:
                    success, box = tracker.update(frame)
                    success = bool(success)

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_tick, 1e-9)
                smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
                last_tick = now
                stats.update(success)

                center_x, center_y = draw_overlay(frame, success, box, smoothed_fps, stats)
                if success:
                    x, y, width, height = (round(float(value), 2) for value in box)
                else:
                    x = y = width = height = ""
                csv_writer.writerow(
                    [
                        frame_number, round(now - started, 6), int(success),
                        center_x if center_x is not None else "",
                        center_y if center_y is not None else "",
                        x, y, width, height, round(smoothed_fps, 3),
                    ]
                )
                csv_file.flush()

                if args.record is not None:
                    if video_writer is None:
                        source_fps = capture.get(cv2.CAP_PROP_FPS)
                        video_writer = create_writer(Path(args.record), frame, source_fps or args.fps)
                    video_writer.write(frame)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    tracker, _ = select_and_initialize(frame, args)
                    stats.reinitializations += 1
                    stats.was_tracked = None
                    stats.current_streak = 0
                if key == ord("s"):
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_path = snapshot_dir / f"kcf_{stamp}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    print(f"画像を保存しました: {image_path}")
        finally:
            capture.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()

    print_summary(stats, time.perf_counter() - started, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KCFによるリアルタイム・バスケットボール追跡")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera-index", type=int, default=0, help="カメラ番号（既定: 0）")
    source.add_argument("--video", type=Path, help="カメラの代わりに使用する動画")
    parser.add_argument("--width", type=int, default=1920, help="要求するカメラ横解像度")
    parser.add_argument("--height", type=int, default=1080, help="要求するカメラ縦解像度")
    parser.add_argument("--fps", type=float, default=60.0, help="要求するカメラFPS")
    parser.add_argument("--roi-padding", type=float, default=1.3, help="選択ROIの拡大率（既定: 1.3）")
    parser.add_argument("--detect-threshold", type=float, default=0.5, help="KCF失跡判定閾値")
    parser.add_argument("--learning-rate", type=float, default=0.075, help="KCFモデル更新率")
    parser.add_argument("--output", default="results/kcf_tracking.csv", help="CSV出力先")
    parser.add_argument("--record", type=Path, help="可視化済みMP4の保存先")
    parser.add_argument("--snapshot-dir", default="captures", help="画像保存先")
    args = parser.parse_args()
    if args.roi_padding < 1.0:
        parser.error("--roi-padding は1.0以上にしてください")
    if not 0.0 < args.detect_threshold <= 1.0:
        parser.error("--detect-threshold は0より大きく1以下にしてください")
    if not 0.0 < args.learning_rate <= 1.0:
        parser.error("--learning-rate は0より大きく1以下にしてください")
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
