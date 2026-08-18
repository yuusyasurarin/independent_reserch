"""OpenCV の既存手法でバスケットボールをリアルタイム追跡する。

起動後、最初のフレームでボールを囲んで Enter/Space を押す。
追跡中は r で追跡対象を選び直し、s で画像保存、q/Esc で終了する。
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from project_runtime import ensure_project_venv

ensure_project_venv()

import cv2


TRACKERS = ("csrt", "kcf", "mosse", "mil")


def tracker_factory(name: str) -> Any:
    """OpenCV 4.x の通常/legacy APIのどちらでもトラッカーを生成する。"""
    class_name = f"Tracker{name.upper()}_create"
    factory = getattr(cv2, class_name, None)
    if factory is None and hasattr(cv2, "legacy"):
        factory = getattr(cv2.legacy, class_name, None)
    if factory is None:
        raise RuntimeError(
            f"{name.upper()} は現在のOpenCVに含まれていません。"
            " 'pip install opencv-contrib-python' を実行してください。"
        )
    return factory()


def open_source(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video:
        capture = cv2.VideoCapture(str(args.video))
        source_name = str(args.video)
    else:
        capture = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(args.camera_index)
        source_name = f"camera {args.camera_index}"
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        # UVCカメラで遅延が蓄積しにくいよう、可能ならバッファを短くする。
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError(f"映像を開けませんでした: {source_name}")
    return capture


def choose_ball(frame: Any, tracker_name: str) -> tuple[Any, tuple[int, int, int, int]]:
    message = "Drag around the ball, then press ENTER/SPACE (C cancels)"
    cv2.putText(frame, message, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    roi = cv2.selectROI("Basketball tracking", frame, showCrosshair=True, fromCenter=False)
    roi = tuple(int(value) for value in roi)
    if roi[2] <= 0 or roi[3] <= 0:
        raise KeyboardInterrupt("ROI selection was cancelled")
    tracker = tracker_factory(tracker_name)
    initialized = tracker.init(frame, roi)
    if initialized is False:  # 一部のOpenCV版では None が成功を表す。
        raise RuntimeError("トラッカーを初期化できませんでした。ROIを確認してください。")
    return tracker, roi


def draw_status(
    frame: Any,
    ok: bool,
    box: tuple[float, float, float, float],
    tracker_name: str,
    display_fps: float,
    recording_state: str | None,
) -> tuple[int | None, int | None]:
    center_x = center_y = None
    color = (0, 255, 0) if ok else (0, 0, 255)
    if ok:
        x, y, width, height = (int(value) for value in box)
        center_x, center_y = x + width // 2, y + height // 2
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        state = "TRACKING"
    else:
        state = "LOST - press R to select again"

    cv2.putText(
        frame,
        f"{tracker_name.upper()} | {state} | {display_fps:.1f} FPS",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )
    record_label = f" | {recording_state}" if recording_state is not None else ""
    cv2.putText(frame, f"R: reselect  S: snapshot  P: pause/resume{record_label}  Q/ESC: quit", (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return center_x, center_y


def create_writer(path: Path, frame: Any, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps, 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"結果動画の保存先を開けませんでした: {path}")
    return writer


def run(args: argparse.Namespace) -> None:
    capture = open_source(args)
    output_path = Path(args.output) if args.output else Path("results") / args.tracker / "realtime_tracking.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = Path(args.snapshot_dir)
    snapshots.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    tracker = None
    video_writer: cv2.VideoWriter | None = None
    recording_paused = False
    start_time = time.perf_counter()
    last_tick = start_time
    smoothed_fps = 0.0

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "elapsed_seconds", "tracked", "center_x", "center_y",
                         "bbox_x", "bbox_y", "bbox_width", "bbox_height"])
        try:
            while True:
                grabbed, frame = capture.read()
                if not grabbed:
                    if args.video:
                        break
                    print("フレーム取得に失敗しました。カメラ接続を確認してください。")
                    break

                frame_number += 1
                if tracker is None:
                    tracker, box = choose_ball(frame.copy(), args.tracker)
                    ok = True
                else:
                    ok, box = tracker.update(frame)

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_tick, 1e-9)
                smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
                last_tick = now

                if args.record is None:
                    recording_state = None
                else:
                    recording_state = "PAUSED" if recording_paused else "REC"
                center_x, center_y = draw_status(
                    frame,
                    bool(ok),
                    box,
                    args.tracker,
                    smoothed_fps,
                    recording_state,
                )
                if ok:
                    x, y, width, height = (round(float(value), 2) for value in box)
                else:
                    x = y = width = height = ""
                writer.writerow([
                    frame_number, round(now - start_time, 6), int(bool(ok)),
                    center_x if center_x is not None else "",
                    center_y if center_y is not None else "", x, y, width, height,
                ])

                if args.record is not None and not recording_paused:
                    if video_writer is None:
                        source_fps = capture.get(cv2.CAP_PROP_FPS)
                        video_writer = create_writer(Path(args.record), frame, source_fps or args.fps)
                    video_writer.write(frame)

                cv2.imshow("Basketball tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    tracker, _ = choose_ball(frame.copy(), args.tracker)
                if key == ord("s"):
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = snapshots / f"ball_{stamp}.jpg"
                    cv2.imwrite(str(path), frame)
                    print(f"画像を保存しました: {path}")
                if key == ord("p") and args.record is not None:
                    recording_paused = not recording_paused
                    print("録画を一時停止しました。" if recording_paused else "録画を再開しました。")
        finally:
            capture.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()

    print(f"追跡ログを保存しました: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="既存のOpenCV手法によるリアルタイム・ボール追跡")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera-index", type=int, default=0, help="カメラ番号（既定: 0）")
    source.add_argument("--video", type=Path, help="カメラの代わりに使う動画ファイル")
    parser.add_argument("--tracker", choices=TRACKERS, default="csrt", help="追跡手法（既定: csrt）")
    parser.add_argument("--width", type=int, default=1920, help="要求するカメラ横解像度")
    parser.add_argument("--height", type=int, default=1080, help="要求するカメラ縦解像度")
    parser.add_argument("--fps", type=float, default=60.0, help="要求するカメラFPS")
    parser.add_argument(
        "--output",
        help="CSV出力先（既定: results/<tracker>/realtime_tracking.csv）",
    )
    parser.add_argument("--record", type=Path, help="追跡表示を重ねたMP4の保存先")
    parser.add_argument("--snapshot-dir", default="captures", help="スナップショット保存先")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("終了しました。")
    except (RuntimeError, OSError) as error:
        cv2.destroyAllWindows()
        raise SystemExit(f"エラー: {error}") from error
