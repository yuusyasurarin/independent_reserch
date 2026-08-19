"""カメラ映像をプレビューしながらMP4へ保存する単純な録画プログラム。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from project_runtime import ensure_project_venv

ensure_project_venv()

import cv2

from synchronized_video_writer import SynchronizedVideoWriter


WINDOW_NAME = "Camera Video Recorder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="カメラ映像をMP4動画として録画")
    parser.add_argument("--camera-index", type=int, default=0, help="カメラ番号（既定: 0）")
    parser.add_argument("--width", type=int, default=1280, help="要求する横解像度（既定: 1280）")
    parser.add_argument("--height", type=int, default=720, help="要求する縦解像度（既定: 720）")
    parser.add_argument("--fps", type=float, default=30.0, help="要求・保存FPS（既定: 30）")
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


def run(args: argparse.Namespace) -> None:
    output_path = args.output or default_output_path()
    capture = open_camera(args)
    writer: SynchronizedVideoWriter | None = None
    recording_paused = False

    try:
        grabbed, frame = capture.read()
        if not grabbed:
            raise RuntimeError("カメラから最初のフレームを取得できませんでした")

        reported_fps = capture.get(cv2.CAP_PROP_FPS)
        writer = SynchronizedVideoWriter(output_path, frame, reported_fps, args.fps)
        actual_width = frame.shape[1]
        actual_height = frame.shape[0]

        print(f"録画を開始しました: {output_path.resolve()}")
        print(f"映像: {actual_width}x{actual_height} / 保存FPS: {writer.fps:.2f}")
        if args.no_preview:
            print("終了するにはCtrl+Cを押してください。")
        else:
            print("Pで一時停止・再開、QまたはEscで録画を終了します。")

        while True:
            elapsed = writer.elapsed
            if not recording_paused:
                writer.write(frame)

            if not args.no_preview:
                preview = frame.copy()
                state = "PAUSED" if recording_paused else "REC"
                state_color = (0, 255, 255) if recording_paused else (0, 0, 255)
                cv2.circle(preview, (24, 27), 8, state_color, -1)
                cv2.putText(
                    preview,
                    f"{state} {elapsed:07.1f}s | input {writer.input_frames} frames",
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
                    if recording_paused:
                        writer.resume()
                        recording_paused = False
                        print("録画を再開しました。")
                    else:
                        writer.pause()
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
            writer.close()
        cv2.destroyAllWindows()

    elapsed = writer.elapsed if writer is not None else 0.0
    print("録画を終了しました。")
    print(f"保存先: {output_path.resolve()}")
    print(f"録画時間: {elapsed:.2f}秒 / 保存フレーム数: {writer.frames_written if writer else 0}")


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\n録画を終了しました。")
    except (RuntimeError, OSError) as error:
        raise SystemExit(f"エラー: {error}") from error
