"""実時間と再生時間を一致させる非同期MP4ライター。"""

from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2


class SynchronizedVideoWriter:
    """処理速度が変動しても、実時間基準でフレームを複製・間引きして保存する。"""

    def __init__(
        self,
        path: Path,
        frame: Any,
        source_fps: float,
        fallback_fps: float,
        realtime: bool = True,
        queue_size: int = 8,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self.path = path
        self.fps = self._select_fps(source_fps, fallback_fps)
        self._realtime = realtime
        self._writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"録画ファイルを作成できませんでした: {path}")

        self._queue: queue.Queue[tuple[float | None, Any]] = queue.Queue(maxsize=queue_size)
        self._started = time.perf_counter()
        self._paused_started: float | None = None
        self._paused_total = 0.0
        self._closed = False
        self._closed_elapsed: float | None = None
        self._error: BaseException | None = None
        self._frames_written = 0
        self._input_frames = 0
        self._thread = threading.Thread(target=self._worker, name="video-writer", daemon=True)
        self._thread.start()

    @staticmethod
    def _select_fps(source_fps: float, fallback_fps: float) -> float:
        if math.isfinite(source_fps) and source_fps > 0:
            return float(source_fps)
        if math.isfinite(fallback_fps) and fallback_fps > 0:
            return float(fallback_fps)
        return 30.0

    @property
    def paused(self) -> bool:
        return self._paused_started is not None

    @property
    def elapsed(self) -> float:
        if self._closed_elapsed is not None:
            return self._closed_elapsed
        now = time.perf_counter()
        current_pause = now - self._paused_started if self._paused_started is not None else 0.0
        return max(0.0, now - self._started - self._paused_total - current_pause)

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def input_frames(self) -> int:
        return self._input_frames

    def write(self, frame: Any) -> None:
        if self._closed or self.paused:
            return
        self._raise_worker_error()
        timestamp = self.elapsed if self._realtime else self._input_frames / self.fps
        item = (timestamp, frame.copy())
        self._input_frames += 1
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # 中間フレームを捨て、最新フレームと時刻を優先する。
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def pause(self) -> None:
        if not self._closed and not self.paused:
            self._paused_started = time.perf_counter()

    def resume(self) -> None:
        if not self._closed and self._paused_started is not None:
            self._paused_total += time.perf_counter() - self._paused_started
            self._paused_started = None

    def close(self) -> None:
        if self._closed:
            return
        end_elapsed = self.elapsed if self._realtime else self._input_frames / self.fps
        self._closed_elapsed = end_elapsed
        self._closed = True
        if self._thread.is_alive():
            self._queue.put((None, end_elapsed))
        self._thread.join()
        self._raise_worker_error()

    def _worker(self) -> None:
        last_frame = None
        try:
            while True:
                elapsed, frame = self._queue.get()
                if elapsed is None:
                    expected_frames = max(1, round(float(frame) * self.fps))
                    while last_frame is not None and self._frames_written < expected_frames:
                        self._writer.write(last_frame)
                        self._frames_written += 1
                    break

                target_index = max(0, round(elapsed * self.fps))
                if last_frame is None:
                    self._writer.write(frame)
                    self._frames_written = 1
                    last_frame = frame
                    continue

                while self._frames_written < target_index:
                    self._writer.write(last_frame)
                    self._frames_written += 1

                if self._frames_written == target_index:
                    self._writer.write(frame)
                    self._frames_written += 1
                last_frame = frame
        except BaseException as error:
            self._error = error
        finally:
            self._writer.release()

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"動画保存中にエラーが発生しました: {self.path}") from self._error
