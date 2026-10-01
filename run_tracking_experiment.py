"""Run PDF experiment models M1--M6 with common logging and bounded async replay."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import csv
import json
import math
import os
from pathlib import Path
import platform
import queue
import subprocess
import threading
import time


class LatestWorker:
    """One in-flight task, one replaceable pending frame; no unbounded image queue."""
    def __init__(self, detector):
        self.detector = detector
        self.pending = queue.Queue(1)
        self.results = queue.Queue()
        self.replaced = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            task = self.pending.get()
            if task is None:
                return
            index, timestamp, received, frame, context = task
            try:
                result = self.detector(frame, timestamp, context)
                self.results.put((index, received, result, None))
            except Exception as error:
                self.results.put((index, received, None, error))

    def submit(self, task):
        try:
            self.pending.put_nowait(task)
        except queue.Full:
            try:
                self.pending.get_nowait()
                self.replaced += 1
            except queue.Empty:
                pass
            self.pending.put_nowait(task)

    def close(self):
        self.pending.put(None)
        self.thread.join()


def replay_detection(tracker, history, source_index, detection):
    """Correct the source state, then re-run image updates. Never revise published CSV rows."""
    entries = list(history)
    position = next((i for i, e in enumerate(entries) if e['index'] == source_index), None)
    if position is None:
        return False, 0
    tracker.restore(entries[position]['before'])
    entries[position]['detection'] = detection
    for entry in entries[position:]:
        entry['before'] = tracker.snapshot()
        tracker.step(entry['frame'], entry['index'], entry['timestamp'], entry['detection'])
    return True, len(entries)-position


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--camera-index', type=int)
    p.add_argument('--method', choices=[f'M{i}' for i in range(1, 7)], required=True)
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--class-id', type=int)
    p.add_argument('--person-class-id', type=int)
    p.add_argument('--ring-class-id', type=int)
    p.add_argument('--ring', nargs=4, type=float, metavar=('X', 'Y', 'W', 'H'))
    p.add_argument('--background', type=Path)
    p.add_argument('--cnn', type=Path)
    p.add_argument('--cnn-confidence', type=float, default=.7)
    p.add_argument('--confidence', type=float, default=.15)
    p.add_argument('--image-size', type=int, default=960)
    p.add_argument('--device', default='auto')
    p.add_argument('--s2-hz', type=float, default=15., help='0: every input frame')
    p.add_argument('--s1-stride', type=int, default=1)
    p.add_argument('--ring-hz', type=float, default=1.)
    p.add_argument('--fps', type=float, help='Camera requested FPS / video metadata override')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--execution', choices=('offline', 'realtime'), default='offline')
    p.add_argument('--async-detection', action='store_true')
    p.add_argument('--history-frames', type=int, default=30)
    p.add_argument('--max-replay-frames', type=int, default=15)
    p.add_argument('--max-prediction-s', type=float, default=.2)
    p.add_argument('--max-kcf-age-s', type=float, default=.5)
    p.add_argument('--association-px', type=float, default=160.)
    p.add_argument('--gate', type=float, default=9.21)
    p.add_argument('--confirm-frames', type=int, default=2)
    p.add_argument('--players', choices=('P0', 'P1', 'P2'), default='P0')
    p.add_argument('--search', choices=('G0', 'G1', 'G2', 'G3'), default='G0')
    p.add_argument('--ring-policy', choices=('R0', 'R1', 'R2', 'R3', 'R4'), default='R0')
    p.add_argument('--alpha', type=float, default=1.5)
    p.add_argument('--beta-up', type=float, default=2.)
    p.add_argument('--beta-down', type=float, default=2.)
    p.add_argument('--intervention-hz', type=float, default=30.)
    p.add_argument('--full-interval', type=float, default=1.)
    p.add_argument('--pure-blob', action='store_true')
    p.add_argument('--diff-threshold', type=int, default=25)
    p.add_argument('--min-blob-area', type=int, default=4)
    p.add_argument('--blob-padding', type=float, default=1.5)
    p.add_argument('--max-rois', type=int, default=8)
    p.add_argument('--max-roi-area', type=float, default=.6)
    p.add_argument('--roi-batch', type=int, default=4)
    p.add_argument('--grid-size', type=int, default=48)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--record', action='store_true')
    p.add_argument('--no-preview', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    return p


def validate(a):
    numeric = [v for v in vars(a).values() if isinstance(v, float)]
    if not all(math.isfinite(v) for v in numeric):
        raise ValueError('Numeric settings must be finite')
    if a.method in ('M1', 'M6') and a.background is None:
        raise ValueError('M1/M6 requires --background prepared before the experiment')
    if a.method == 'M1' and a.cnn is None:
        raise ValueError('M1 requires a trained --cnn checkpoint')
    if a.pure_blob and a.method not in ('M1', 'M6'):
        raise ValueError('--pure-blob is for M1/M6')
    if a.async_detection and (a.method not in ('M3', 'M4', 'M5', 'M6') or a.execution != 'realtime'):
        raise ValueError('Async is for realtime M3--M6; use synchronous M1/M2 baselines')
    if a.camera_index is not None and a.execution != 'realtime':
        raise ValueError('Camera requires --execution realtime')
    if min(a.history_frames, a.max_replay_frames, a.s1_stride, a.confirm_frames, a.roi_batch,
           a.width, a.height, a.image_size, a.grid_size, a.max_rois, a.min_blob_area) < 1:
        raise ValueError('Sizes/counts must be positive')
    if a.s2_hz < 0 or a.max_frames < 0 or not 0 < a.confidence <= 1 or not 0 < a.cnn_confidence <= 1:
        raise ValueError('Invalid rate, frame limit or confidence')
    if min(a.ring_hz, a.max_prediction_s, a.max_kcf_age_s, a.association_px, a.gate,
           a.alpha, a.beta_up, a.beta_down, a.full_interval, a.intervention_hz, a.blob_padding) <= 0:
        raise ValueError('Durations, scales and thresholds must be positive')
    if not 0 < a.max_roi_area <= 1 or not 1 <= a.diff_threshold <= 255 or (a.fps is not None and a.fps <= 0):
        raise ValueError('Invalid ROI area, difference threshold or FPS')
    if a.ring is not None and (not all(math.isfinite(v) for v in a.ring) or min(a.ring[2:]) <= 0):
        raise ValueError('Invalid manual ring rectangle')
    if a.output.exists():
        raise FileExistsError(f'Use a new experiment directory: {a.output}')


def run(a):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from adaptive_basketball_tracker import file_hash
    from experiment_detectors import Detector, kcf_factory
    from experiment_tracking import ExperimentTracker, Settings

    started = time.perf_counter()
    detector = Detector(a)
    cfg = Settings(**{k: getattr(a, k) for k in Settings.__dataclass_fields__ if hasattr(a, k)})
    tracker = ExperimentTracker(cfg, kcf_factory, a.ring)
    cap = cv2.VideoCapture(str(a.video) if a.video else a.camera_index)
    worker = writer = None
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open input')
        if a.video is None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
            cap.set(cv2.CAP_PROP_FPS, a.fps or 60.)
        fps = a.fps or cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('No valid source FPS; specify --fps')
        ok, frame = cap.read()
        if not ok:
            raise ValueError('No readable frames')
        h, w = frame.shape[:2]
        if a.ring is not None:
            x, y, rw, rh = a.ring
            if min(x, y) < 0 or x+rw > w or y+rh > h:
                raise ValueError('Manual ring must fit input image')
        detector(frame, 0., tracker.search_context())  # warm-up, no tracking initialization
        detector.last_full = detector.last_ring = -math.inf
        a.output.mkdir(parents=True, exist_ok=False)
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                        settings=asdict(cfg), fps=fps, width=w, height=h, platform=platform.platform(),
                        processor=platform.processor(), torch=torch.__version__, opencv=cv2.__version__,
                        numpy=np.__version__, ultralytics=ultralytics.__version__, git_commit=commit,
                        device=detector.device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        hashes={str(p): file_hash(p) for p in (a.video, a.background, a.cnn, Path(a.model)) if p and Path(p).is_file()},
                        startup_ms=(time.perf_counter()-started)*1000,
                        timestamp_basis='video POS_MSEC (fallback frame/FPS)' if a.video else 'host receipt, not sensor exposure',
                        latency_basis='offline read-to-output' if a.execution == 'offline' else 'video scheduled arrival / camera host receipt',
                        queue_policy='latest pending detector frame; bounded source-frame replay' if a.async_detection else 'synchronous',
                        ring_condition='manual fixed, not learned ring evaluation' if a.ring else 'detected')
        (a.output/'config.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        if a.record:
            writer = cv2.VideoWriter(str(a.output/'tracking.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps/a.s1_stride, (w, h))
            if not writer.isOpened():
                raise RuntimeError('Cannot create recording')
        worker = LatestWorker(detector) if a.async_detection else None
        history = deque(maxlen=a.history_frames)
        origin = time.perf_counter()
        input_count = emitted = calls = stale = replayed = 0
        completed_times = []
        last_timestamp, next_detection = -1., 0.
        last_applied = -1
        with (a.output/'tracking.csv').open('w', encoding='utf-8', newline='') as stream, \
             (a.output/'detections.jsonl').open('w', encoding='utf-8') as detection_log:
            csv_writer = None

            def log_result(index, result, applied, age):
                nonlocal calls
                calls += 1
                result = dict(result)
                for key in ('started_clock_s', 'finished_clock_s'):
                    result[key] -= origin
                completed_times.append(result['finished_clock_s'])
                detection_log.write(json.dumps(dict(frame_index=index, applied=applied,
                                                   age_ms=age, **result), allow_nan=False)+'\n')

            while ok and (not a.max_frames or input_count < a.max_frames):
                index = input_count
                input_count += 1
                now = time.perf_counter()
                if a.video:
                    raw_t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    timestamp = index/fps if a.fps is not None or not math.isfinite(raw_t) or raw_t <= last_timestamp else raw_t
                    if timestamp <= last_timestamp:
                        timestamp = last_timestamp+1/fps
                    if a.execution == 'realtime':
                        delay = origin+timestamp-now
                        if delay > 0:
                            time.sleep(delay)
                        received = origin+timestamp
                    else:
                        received = now
                else:
                    timestamp, received = now-origin, now
                last_timestamp = timestamp
                # Latest-frame policy for realtime file playback; keep true source IDs.
                late = a.video and a.execution == 'realtime' and time.perf_counter()-received > 1/fps
                if index % a.s1_stride or late:
                    ok, frame = cap.read()
                    continue
                processing_start = time.perf_counter()
                age, replay_ms, replay_frames = '', 0., 0
                if worker:
                    while True:
                        try:
                            source_index, source_received, result, error = worker.results.get_nowait()
                        except queue.Empty:
                            break
                        if error:
                            raise error
                        length = sum(e['index'] >= source_index for e in history)
                        begin = time.perf_counter()
                        if source_index <= last_applied or length > a.max_replay_frames:
                            applied, n = False, 0
                        else:
                            applied, n = replay_detection(tracker, history, source_index, result)
                        if applied:
                            last_applied = source_index
                        else:
                            stale += 1
                            next_detection = timestamp
                        age = (time.perf_counter()-source_received)*1000
                        replay_ms += (time.perf_counter()-begin)*1000
                        replay_frames += n
                        log_result(source_index, result, applied, age)
                context = tracker.search_context()
                hz = a.intervention_hz if context['near_ring'] and a.ring_policy == 'R3' else a.s2_hz
                due = hz == 0 or timestamp+1e-8 >= next_detection
                result = None
                if due:
                    next_detection = timestamp+(1/hz if hz else 0)
                    if worker:
                        worker.submit((index, timestamp, received, frame.copy(), context))
                    else:
                        result = detector(frame, timestamp, context)
                        age = (time.perf_counter()-received)*1000
                        log_result(index, result, True, age)
                before = tracker.snapshot() if worker else None
                row = tracker.step(frame, index, timestamp, result)
                if worker:
                    history.append(dict(index=index, timestamp=timestamp, frame=frame.copy(), before=before, detection=result))
                row.update(detector_requested=int(due), detector_source_frame=index if result else last_applied,
                           detector_age_ms=age, replay_ms=replay_ms, replay_frames=replay_frames,
                           processing_ms=(time.perf_counter()-processing_start)*1000,
                           received_clock_s=received-origin, output_clock_s=time.perf_counter()-origin,
                           latency_ms=(time.perf_counter()-received)*1000)
                if csv_writer is None:
                    csv_writer = csv.DictWriter(stream, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                emitted += 1
                replayed += replay_frames
                if writer or not a.no_preview:
                    display = frame.copy()
                    if row['center_x'] != '':
                        cv2.circle(display, (round(row['center_x']), round(row['center_y'])), 6, (0, 255, 0), 2)
                    cv2.putText(display, f'{a.method} {index} {row["source"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                    if writer:
                        writer.write(display)
                    if not a.no_preview:
                        cv2.imshow('M1-M6 experiment', display)
                        if cv2.waitKey(1) & 255 in (ord('q'), 27):
                            break
                ok, frame = cap.read()
            elapsed = time.perf_counter()-origin
            if worker:
                worker.close()
                while not worker.results.empty():
                    index, received, result, error = worker.results.get_nowait()
                    if error:
                        raise error
                    log_result(index, result, False, (time.perf_counter()-received)*1000)
                    stale += 1
            summary = dict(input_frames=input_count, output_frames=emitted, wall_seconds=elapsed,
                           input_received_fps=input_count/elapsed if elapsed else None,
                           output_fps=emitted/elapsed if elapsed else None,
                           drop_rate=1-emitted/input_count if input_count else None,
                           detector_jobs=calls, detector_results_discarded=stale,
                           detector_jobs_completed_in_window=sum(t <= elapsed for t in completed_times),
                           detector_job_fps=sum(t <= elapsed for t in completed_times)/elapsed if elapsed else None,
                           detector_pending_replaced=worker.replaced if worker else 0,
                           replay_frames=replayed, recording='processed frames at nominal FPS; dropped frames are not duplicated')
            (a.output/'runtime.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'Saved: {a.output.resolve()}')
    finally:
        if worker and worker.thread.is_alive():
            worker.close()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    p = parser()
    a = p.parse_args()
    try:
        validate(a)
        root = Path(__file__).resolve().parent
        for key, folder in (('YOLO_CONFIG_DIR', '.ultralytics'), ('MPLCONFIGDIR', '.matplotlib'), ('TORCH_HOME', '.torch')):
            os.environ.setdefault(key, str(root/folder))
        run(a)
    except (ValueError, FileExistsError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
