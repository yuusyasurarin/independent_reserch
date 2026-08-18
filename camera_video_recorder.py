"""カメラ映像をプレビューしながらMP4へ保存する単純な録画プログラム。"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

from project_runtime import ensure_project_venv

ensure_project_venv()

import cv2


WINDOW_NAME = "Camera Video Recorder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="カメラ映像をMP4動画として録画")
    parser.add_argument("--camera-index", type=int, default=0, help="カメラ番号（既定: 0）")
    parser.add_argument("--width", type=int, default=1920, help="要求する横解像度（既定: 1920）")
    parser.add_argument("--height", type=int, default=1080, help="要求する縦解像度（既定: 1080）")
    parser.add_argument("--fps", type=float, default=60.0, help="要求・保存FPS（既定: 60）")
    parser.add_argument(
        "--output",
        type=Path,
        help="MP4保存先（省略時はresults/recordings内へ日時名で保存）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="指定秒数で自動終了（省略時はQ、Esc、またはCtrl+Cまで録画）",
    )
    parser.add_argument("--no-preview", action="store_true", help="プレビュー画面を表示しない")
    args = parser.parse_args()

    if args.camera_index < 0:
        parser.error("--camera-index は0以上にしてください")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width と --height は正の整数にしてください")
    if args.fps <= 0:
        parser.error("--fps は0より大きくしてください")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration は0より大きくしてください")
    return args


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "recordings" / f"camera_{timestamp}.mp4"


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: camera_index={args.camera_index}")

    # 高解像度・高FPSのUVCカメラで帯域を抑えられる場合はMJPGを要求する。
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_writer(path: Path, frame, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"録画ファイルを作成できませんでした: {path}")
    return writer


def run(args: argparse.Namespace) -> None:
    output_path = args.output or default_output_path()
    capture = open_camera(args)
    writer: cv2.VideoWriter | None = None
    started = 0.0
    paused_started: float | None = None
    paused_total = 0.0
    recording_paused = False
    frame_count = 0

    def recording_elapsed(now: float) -> float:
        current_pause = now - paused_started if recording_paused and paused_started is not None else 0.0
        return max(0.0, now - started - paused_total - current_pause)

    try:
        grabbed, frame = capture.read()
        if not grabbed:
            raise RuntimeError("カメラから最初のフレームを取得できませんでした")

        reported_fps = capture.get(cv2.CAP_PROP_FPS)
        recording_fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else args.fps
        writer = create_writer(output_path, frame, recording_fps)
        actual_width = frame.shape[1]
        actual_height = frame.shape[0]
        started = time.perf_counter()

        print(f"録画を開始しました: {output_path.resolve()}")
        print(f"映像: {actual_width}x{actual_height} / 保存FPS: {recording_fps:.2f}")
        if args.no_preview:
            print("終了するにはCtrl+Cを押してください。")
        else:
            print("Pで一時停止・再開、QまたはEscで録画を終了します。")

        while True:
            now = time.perf_counter()
            elapsed = recording_elapsed(now)
            if not recording_paused:
                writer.write(frame)
                frame_count += 1

            if not args.no_preview:
                preview = frame.copy()
                state = "PAUSED" if recording_paused else "REC"
                state_color = (0, 255, 255) if recording_paused else (0, 0, 255)
                cv2.circle(preview, (24, 27), 8, state_color, -1)
                cv2.putText(
                    preview,
                    f"{state} {elapsed:07.1f}s | {frame_count} frames",
                    (42, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    state_color,
                    2,
                )
                cv2.putText(
                    preview,
                    "P: pause/resume  Q/ESC: stop",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    toggle_time = time.perf_counter()
                    if recording_paused:
                        if paused_started is not None:
                            paused_total += toggle_time - paused_started
                        paused_started = None
                        recording_paused = False
                        print("録画を再開しました。")
                    else:
                        paused_started = toggle_time
                        recording_paused = True
                        print("録画を一時停止しました。")

            if args.duration is not None and elapsed >= args.duration:
                break

            grabbed, frame = capture.read()
            if not grabbed:
                print("フレームを取得できなかったため録画を終了します。")
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = recording_elapsed(time.perf_counter()) if started else 0.0
    print("録画を終了しました。")
    print(f"保存先: {output_path.resolve()}")
    print(f"録画時間: {elapsed:.2f}秒 / 保存フレーム数: {frame_count}")


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\n録画を終了しました。")
    except (RuntimeError, OSError) as error:
        raise SystemExit(f"エラー: {error}") from error
