"""ELPカメラの映像を表示・保存する最小構成のPythonプログラム

ステップバイステップ:
1. カメラを開く
2. フレームを1枚ずつ読み込む
3. 画面に表示する
4. 's' で保存
5. 'q' で終了

使い方:
    python elp_camera_demo.py
    python elp_camera_demo.py --camera-index 1
    python elp_camera_demo.py --save-dir captures
"""

import argparse
import os
from datetime import datetime

from project_runtime import ensure_project_venv

ensure_project_venv()

import cv2


def open_camera(camera_index=0, width=1280, height=720, fps=30.0):
    """カメラを開いて、成功した場合は VideoCapture を返す"""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # DSHOW が使えない環境向けのフォールバック
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"カメラの起動に失敗しました。camera_index={camera_index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def save_frame(frame, save_dir="captures"):
    """現在のフレームを画像として保存する"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"elp_capture_{timestamp}.png")
    cv2.imwrite(file_path, frame)
    print(f"保存しました: {file_path}")
    return file_path


def main():
    parser = argparse.ArgumentParser(description="ELPカメラのライブ表示と保存")
    parser.add_argument("--camera-index", type=int, default=0, help="カメラのデバイス番号 (例: 0, 1, 2)")
    parser.add_argument("--width", type=int, default=1280, help="要求する横解像度（既定: 1280）")
    parser.add_argument("--height", type=int, default=720, help="要求する縦解像度（既定: 720）")
    parser.add_argument("--fps", type=float, default=30.0, help="要求するFPS（既定: 30）")
    parser.add_argument("--save-dir", type=str, default="captures", help="保存先ディレクトリ")
    args = parser.parse_args()

    print("[Step 1] カメラを起動しています...")
    cap = open_camera(args.camera_index, args.width, args.height, args.fps)

    print("[Step 2] 映像を表示します. 's' = 保存, 'q' = 終了")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームを取得できませんでした。カメラが接続されているか確認してください。")
                break

            cv2.imshow("ELP Camera Preview", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("[Step 3] 現在のフレームを保存します")
                save_frame(frame, args.save_dir)

            elif key == ord('q'):
                print("[Step 4] 終了します")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
