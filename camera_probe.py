"""接続中のカメラを探し、各解像度・fpsで実際に何fps出るかを測る（当日の最初に実行）。

OpenCVは、カメラが対応していない解像度やfpsを要求されてもエラーにせず、別の値で
動くことがある。そのため「要求した値」ではなく「実際に読めた画像の大きさ」と
「2秒間に読めた枚数」を表示する。録画・追跡には、ここで実測30fps近く出た設定を使う。

使い方:
    python camera_probe.py                 # 番号0〜4を調べる
    python camera_probe.py --indices 0 1   # 番号を指定
    python camera_probe.py --save probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from project_runtime import ensure_project_venv

ensure_project_venv()

import cv2

MODES = [(1920, 1080, 30), (1280, 720, 60), (1280, 720, 30), (640, 480, 30)]


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap if cap.isOpened() else None


def measure(index, width, height, fps, seconds):
    cap = open_camera(index)
    if cap is None:
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok:
            return dict(requested=[width, height, fps], ok=False)
        for _ in range(5):  # 自動露出などが落ち着くまで捨てる
            cap.read()
        n, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
        elapsed = time.perf_counter() - start
        return dict(requested=[width, height, fps], ok=True, actual=[frame.shape[1], frame.shape[0]],
                    reported_fps=cap.get(cv2.CAP_PROP_FPS), measured_fps=n / elapsed if elapsed else 0.,
                    brightness=float(frame.mean()))
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seconds", type=float, default=2.)
    p.add_argument("--save", type=Path)
    a = p.parse_args()
    results = {}
    for index in a.indices:
        cap = open_camera(index)
        if cap is None:
            print(f"カメラ {index}: なし")
            continue
        cap.release()
        rows = [measure(index, w, h, f, a.seconds) for w, h, f in MODES]
        results[index] = rows
        print(f"カメラ {index}:")
        for r in rows:
            if r is None or not r["ok"]:
                print(f"  要求 {r['requested'] if r else '?'} → 画像を取得できず")
                continue
            w, h, f = r["requested"]
            mark = "" if r["actual"] == [w, h] else "  ※要求と違う解像度"
            print(f"  要求 {w}x{h}@{f} → 実際 {r['actual'][0]}x{r['actual'][1]}  実測 {r['measured_fps']:.1f}fps"
                  f"（カメラ申告 {r['reported_fps']:.0f}）{mark}")
    if not results:
        print("カメラが1台も見つかりませんでした。USBの接続と、他のアプリ（カメラアプリ・Zoomなど）が使っていないかを確認してください。")
    if a.save:
        a.save.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
