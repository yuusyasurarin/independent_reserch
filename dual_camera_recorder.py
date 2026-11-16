"""2台のカメラを同時に録画し、共通の実時間基準でMP4と時刻ログを保存する。

両方の録画は同じ開始時刻からの実時間に合わせて書き込むため、同じフレーム番号は
ほぼ同じ時刻（±1フレーム＋カメラ内部の遅延差）を指す。確認用に、拍手やフラッシュの
瞬間に M キーを押すと markers.csv に時刻を記録できる。後で両動画の該当フレームを
目視で合わせ、triangulate_ball.py の --offset-s を決める。

使い方:
    python dual_camera_recorder.py --cameras 0 1 --width 1280 --height 720 --fps 30 --output results/recordings/game01
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import threading
import time

from project_runtime import ensure_project_venv

ensure_project_venv()

import cv2

from synchronized_video_writer import SynchronizedVideoWriter


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: camera_index={index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"カメラ {index} から最初のフレームを取得できませんでした")
    return cap, frame


class CameraRecorder:
    def __init__(self, name, index, args, output_dir, session_start):
        self.name, self.index, self.session_start = name, index, session_start
        self.cap, first = open_camera(index, args.width, args.height, args.fps)
        self.writer = SynchronizedVideoWriter(output_dir / f"{name}.mp4", first, self.cap.get(cv2.CAP_PROP_FPS), args.fps)
        self.log = (output_dir / f"{name}_timestamps.csv").open("w", encoding="utf-8", newline="")
        self.csv = csv.writer(self.log)
        self.csv.writerow(["input_index", "host_time_s", "writer_elapsed_s"])
        self.latest = first
        self.frames = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, name=f"capture-{name}", daemon=True)
        self.info = dict(index=index, width=first.shape[1], height=first.shape[0], reported_fps=self.cap.get(cv2.CAP_PROP_FPS))

    def loop(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError(f"カメラ {self.index} からフレームを取得できなくなりました")
                now = time.perf_counter()
                self.writer.write(frame)
                self.csv.writerow([self.frames, f"{now - self.session_start:.6f}", f"{self.writer.elapsed:.6f}"])
                self.frames += 1
                self.latest = frame
        except Exception as error:  # noqa: BLE001 - surfaced to the main loop
            self.error = error
        finally:
            self.cap.release()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        self.writer.close()
        self.log.close()


def run(args):
    output_dir = args.output or Path("results") / "recordings" / f"dual_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_start = time.perf_counter()
    recorders = [CameraRecorder(f"cam{k+1}", index, args, output_dir, session_start) for k, index in enumerate(args.cameras)]
    for r in recorders:
        r.thread.start()
    markers = []
    print(f"録画中: {output_dir.resolve()}")
    for r in recorders:
        print(f"  {r.name}: index {r.info['index']} {r.info['width']}x{r.info['height']} reported {r.info['reported_fps']:.1f}fps")
    print("M: 同期マーカー（拍手・フラッシュの瞬間）  Q/Esc: 終了")
    try:
        while True:
            if any(r.error for r in recorders):
                break
            if args.duration and time.perf_counter() - session_start >= args.duration:
                break
            if not args.no_preview:
                tiles = [cv2.resize(r.latest, (args.preview_width, round(args.preview_width * r.latest.shape[0] / r.latest.shape[1]))) for r in recorders]
                height = max(t.shape[0] for t in tiles)
                tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
                canvas = cv2.hconcat(tiles)
                elapsed = time.perf_counter() - session_start
                text = f"REC {elapsed:07.1f}s  " + "  ".join(f"{r.name}:{r.frames}" for r in recorders) + f"  markers:{len(markers)}"
                cv2.putText(canvas, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                cv2.imshow("Dual camera recorder", canvas)
                key = cv2.waitKey(30) & 0xFF
            else:
                time.sleep(.03)
                key = 255
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                markers.append(dict(host_time_s=time.perf_counter() - session_start, **{r.name + "_frames": r.frames for r in recorders}))
                print(f"マーカー {len(markers)}: {markers[-1]}")
    finally:
        for r in recorders:
            r.close()
        cv2.destroyAllWindows()
        with (output_dir / "markers.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["host_time_s"] + [r.name + "_frames" for r in recorders])
            writer.writeheader()
            writer.writerows(markers)
        session = dict(output=str(output_dir.resolve()), requested=dict(width=args.width, height=args.height, fps=args.fps),
                       cameras={r.name: dict(**r.info, input_frames=r.frames, written_frames=r.writer.frames_written, saved_fps=r.writer.fps) for r in recorders},
                       elapsed_s=time.perf_counter() - session_start, markers=len(markers),
                       errors=[str(r.error) for r in recorders if r.error],
                       time_basis="Both MP4s are written against the same host clock: frame k of each file is time k/fps after this session start, with frames duplicated or dropped to hold that rate. Camera-internal latency is not measured.")
        (output_dir / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(session, ensure_ascii=False, indent=2))
    if session["errors"]:
        raise SystemExit("エラー: " + "; ".join(session["errors"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cameras", type=int, nargs=2, default=[0, 1], metavar=("A", "B"), help="2台のカメラ番号")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.)
    p.add_argument("--output", type=Path, help="保存先フォルダ（省略時は results/recordings/dual_日時）")
    p.add_argument("--duration", type=float, default=0., help="指定秒数で自動終了（0は手動終了）")
    p.add_argument("--preview-width", type=int, default=640)
    p.add_argument("--no-preview", action="store_true")
    a = p.parse_args()
    if a.cameras[0] == a.cameras[1] or a.width <= 0 or a.height <= 0 or a.fps <= 0 or a.duration < 0:
        p.error("設定が不正です")
    try:
        run(a)
    except KeyboardInterrupt:
        print("\n録画を終了しました。")
