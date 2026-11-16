"""Real-time fixed-delay ball tracking on top of the evaluated condition-G parts.

Every captured frame gets a consecutive id. Frames the processor skips under
load are still pushed into the fixed-lag smoother as explicit empty rows, so
bridge interpolation can fill them from neighbouring YOLO anchors. Display
happens a fixed delay after capture. The margin between the algorithmic
lookahead (0.4 s for bridge) and the display delay (3 s) absorbs processing
jitter; a bounded backlog drops stale frames instead of letting delay grow.
No ground truth or future frames beyond the lookahead are used.
"""
from collections import OrderedDict, deque
import threading
import time

from .common import stats


def placeholder(frame_id, fps):
    """Row for a frame the processor skipped. It carries no visual evidence."""
    return dict(frame_id=frame_id, timestamp_s=frame_id/fps, bbox=None, track_id=None, state='dropped',
                source='dropped', confidence=None, kcf_success_flag=None,
                events=['dropped_by_backlog_control'], age_since_yolo_s=None, yolo_executed=False)


class TrackingWorker:
    """Causal System 1/2 plus bounded-lag smoother with backlog-controlled dropping.

    Call step() once per captured frame, in capture order, from a single thread.
    """

    def __init__(self, policy, smoother, detect, fps, max_backlog_s, clock=time.perf_counter, history=18000):
        if fps <= 0 or max_backlog_s < 0:
            raise ValueError('fps must be positive and max_backlog_s non-negative')
        self.policy, self.smoother, self.detect = policy, smoother, detect
        self.fps, self.max_backlog_s, self.clock = fps, max_backlog_s, clock
        self.counters = dict(received=0, processed=0, dropped=0, yolo_calls=0, emitted=0)
        self.processing_ms = deque(maxlen=history)
        self.backlog_s = deque(maxlen=history)
        self.last_seq = -1
        self.last_source = 'none'

    def step(self, seq, t_capture, image, newer_waiting):
        if seq != self.last_seq+1:
            raise ValueError('Frames must be stepped exactly once in capture order')
        self.last_seq = seq
        self.counters['received'] += 1
        started = self.clock()
        backlog = started-t_capture
        self.backlog_s.append(backlog)
        if newer_waiting and backlog > self.max_backlog_s:
            # Stale and a fresher frame is already waiting: skip the work, keep the slot.
            self.counters['dropped'] += 1
            row, candidates, detected = placeholder(seq, self.fps), [], False
        else:
            detected = self.policy.needs_detection(seq, image)
            candidates = []
            if detected:
                candidates, _ = self.detect(image)
                self.counters['yolo_calls'] += 1
            row = self.policy.update(image, seq, candidates, detected)
            row['yolo_executed'] = detected
            self.counters['processed'] += 1
            self.last_source = row['source']
        out = self.smoother.push(row, candidates, detected)
        finished = self.clock()
        if row['source'] != 'dropped':
            self.processing_ms.append((finished-started)*1000)
        return self._stamp([out] if out is not None else [], finished)

    def flush(self):
        return self._stamp(list(self.smoother.flush()), self.clock())

    def _stamp(self, outputs, t_result):
        for out in outputs:
            out['t_result'] = t_result
            out['processed'] = out['original_source'] != 'dropped'
            self.counters['emitted'] += 1
        return outputs

    def recent_yolo_hz(self):
        return len(self.policy.calls)/(self.policy.window/self.fps)

    def summary(self):
        received = self.counters['received']
        return dict(**self.counters,
                    drop_rate=self.counters['dropped']/received if received else 0.,
                    processing_ms=stats(list(self.processing_ms)), backlog_s=stats(list(self.backlog_s)))


class DelayedOutputStore:
    """Thread-safe frame/result buffer released at a fixed delay after capture.

    Frames older than max_hold_s that were never released (display stalled)
    are discarded so memory stays bounded.
    """

    def __init__(self, delay_s, max_hold_s):
        if delay_s < 0 or max_hold_s <= delay_s:
            raise ValueError('Require 0 <= delay_s < max_hold_s')
        self.delay, self.max_hold = delay_s, max_hold_s
        self.frames = OrderedDict()
        self.results = {}
        self.lock = threading.Lock()
        self.last_released = -1
        self.discarded = 0

    def add_frame(self, seq, t_capture, image):
        with self.lock:
            if self.frames and seq <= next(reversed(self.frames)):
                raise ValueError('Frames must be added in increasing order')
            self.frames[seq] = (t_capture, image)
            while self.frames:
                first, (t_first, _) = next(iter(self.frames.items()))
                if t_capture-t_first <= self.max_hold:
                    break
                self.frames.popitem(last=False)
                self.results.pop(first, None)
                self.last_released = max(self.last_released, first)
                self.discarded += 1

    def add_result(self, row):
        with self.lock:
            if row['frame_id'] > self.last_released:
                self.results[row['frame_id']] = row

    def due(self, now):
        """Newest frame whose display time has arrived, or None."""
        with self.lock:
            latest = None
            for seq, (t_capture, _) in self.frames.items():
                if t_capture > now-self.delay:
                    break
                latest = seq
            return latest

    def release(self, upto):
        """Pop all frames up to and including `upto` in order, with their results."""
        released = []
        with self.lock:
            while self.frames:
                seq, (t_capture, image) = next(iter(self.frames.items()))
                if seq > upto:
                    break
                self.frames.popitem(last=False)
                released.append((seq, t_capture, image, self.results.pop(seq, None)))
                self.last_released = seq
            return released

    def buffered_frames(self):
        with self.lock:
            return len(self.frames)

    def next_due_time(self):
        with self.lock:
            for _, (t_capture, _) in self.frames.items():
                return t_capture+self.delay
            return None
