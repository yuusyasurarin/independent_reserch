"""Single-ball tracking policy; detector and KCF adapters are injected for testing."""
from dataclasses import dataclass
import math
import numpy as np


def center(box):
    x, y, w, h = box
    return np.array([x + w / 2, y + h / 2], dtype=float)


def valid_box(box):
    return box is not None and np.isfinite(box).all() and min(box[2:]) > 0


def overlap(ball, person):
    x, y, w, h = ball
    a, b, c, d = person
    return max(0, min(x+w, a+c)-max(x, a)) * max(0, min(y+h, b+d)-max(y, b)) / (w*h)


@dataclass
class Config:
    mode: str = "adaptive"
    detect_interval: int = 6
    max_prediction: int = 12
    uncertainty_threshold: float = 0.8
    ring_enter: float = 1.5
    ring_exit: float = 2.0
    gate: float = 9.21
    process_noise: float = 1800.0


class MotionFilter:
    def __init__(self, point):
        self.x = np.array([*point, 0., 0.])
        self.P = np.diag([25., 25., 10000., 10000.])
        self.H = np.eye(2, 4)

    def predict(self, dt, noise):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + noise**2 * G @ G.T

    def innovation(self, point, variance):
        residual = point - self.x[:2]
        S = self.P[:2, :2] + np.eye(2) * variance
        return float(residual @ np.linalg.solve(S, residual))

    def correct(self, point, variance):
        R = np.eye(2) * variance
        K = np.linalg.solve(self.P[:2, :2] + R, self.P[:2, :].copy()).T
        self.x += K @ (point - self.x[:2])
        A = np.eye(4) - K @ self.H
        self.P = A @ self.P @ A.T + K @ R @ K.T


class AdaptiveTracker:
    def __init__(self, detector, kcf_factory, ring, config=None):
        self.detector, self.kcf_factory = detector, kcf_factory
        self.ring, self.cfg = ring, config or Config()
        self.kf = self.kcf = self.box = None
        self.people = []
        self.people_frame = -100000
        self.last_detection = -100000
        self.missing = 0
        self.near_ring = False
        self.previous_mode = None

    def ring_distance(self, point):
        return float(np.linalg.norm(point - center(self.ring)) / self.ring[2])

    def step(self, frame, index, dt):
        cfg = self.cfg
        adaptive = cfg.mode in ("adaptive", "no-ring", "no-uncertainty")
        ring_enabled = adaptive and cfg.mode != "no-ring"
        if self.kf is not None:
            self.kf.predict(dt, cfg.process_noise)
            w, h = self.box[2:]
            predicted = (* (self.kf.x[:2] - [w/2, h/2]), w, h)
            distance = min(self.ring_distance(self.kf.x[:2]), self.ring_distance(center(self.box)))
            self.near_ring = ring_enabled and distance < (cfg.ring_exit if self.near_ring else cfg.ring_enter)
            sigma = math.sqrt(float(np.linalg.eigvalsh(self.kf.P[:2, :2]).max()))
            uncertainty = sigma / max(w, h, 1) + (index-self.last_detection) / (2*cfg.detect_interval)
        else:
            predicted, uncertainty = None, 0.
            self.near_ring = False
        suspect = bool(predicted is not None and index-self.people_frame <= cfg.detect_interval and
                       any(overlap(predicted, p) >= 0.3 for p in self.people))
        kcf_box, kcf_ok, kcf_nis = None, False, None
        # OpenCV KCF.update also learns appearance: do not call it during suspected occlusion.
        if self.kcf is not None and not (adaptive and suspect):
            ok, candidate = self.kcf.update(frame)
            if ok and valid_box(candidate):
                kcf_nis = self.kf.innovation(center(candidate), 100.)
                kcf_ok = self.near_ring or kcf_nis <= cfg.gate
                if kcf_ok:
                    kcf_box = candidate
        if self.kf is None:
            reason = "initial_or_lost"
        elif cfg.mode == "yolo":
            reason = "every_frame"
        elif self.near_ring:
            reason = "near_ring"
        elif adaptive and suspect:
            reason = "occlusion_suspected"
        elif adaptive and cfg.mode != "no-uncertainty" and uncertainty >= cfg.uncertainty_threshold:
            reason = "high_uncertainty"
        elif adaptive and not kcf_ok:
            reason = "kcf_failed_or_innovation"
        elif index-self.last_detection >= cfg.detect_interval:
            reason = "periodic"
        else:
            reason = "stable"
        detect = reason != "stable"
        observation, confidence, nis = None, None, None
        if detect:
            balls, self.people = self.detector(frame)
            self.people_frame = index
            self.last_detection = index
            candidates = []
            for box, score in balls:
                if not valid_box(box):
                    continue
                value = self.kf.innovation(center(box), 25.) if self.kf is not None else 0.
                if self.kf is None or cfg.mode == "yolo":
                    cost = -score
                elif self.near_ring:
                    # A finite visual gate tolerates a rebound without accepting any ball in the image.
                    delta = np.linalg.norm(center(box)-center(self.box)) / self.ring[2]
                    if delta > cfg.ring_exit:
                        continue
                    cost = delta - 0.2*score
                else:
                    if value > cfg.gate:
                        continue
                    cost = value - score
                candidates.append((cost, box, score, value))
            if candidates:
                _, observation, confidence, nis = min(candidates, key=lambda item: item[0])
        source = "yolo" if observation is not None else "none"
        # A failed YOLO verification invalidates KCF rather than hiding a likely drift.
        if not detect and kcf_ok:
            observation, source, nis = kcf_box, "kcf", kcf_nis
        if observation is not None:
            point = center(observation)
            if self.kf is None or (self.near_ring and nis is not None and nis > cfg.gate):
                self.kf = MotionFilter(point)
            else:
                self.kf.correct(point, 25. if source == "yolo" else 100.)
            self.box = tuple(observation)
            self.missing = 0
            if source == "yolo" and cfg.mode != "yolo":
                self.kcf = self.kcf_factory(frame, self.box)
            point_out = point if self.near_ring or cfg.mode == "yolo" else self.kf.x[:2]
            mode = "ring_visual" if self.near_ring else "detector" if source == "yolo" else "fast"
        else:
            self.missing += 1
            self.kcf = None
            point_out = None
            mode = "lost"
            if self.kf is not None and self.missing <= cfg.max_prediction:
                mode = "occluded" if suspect else "prediction"
                # At the rim a motion-only extrapolation is deliberately not published.
                if not self.near_ring and cfg.mode != "yolo":
                    point_out, source = self.kf.x[:2].copy(), "prediction"
            elif self.kf is not None:
                self.kf = self.box = None
        relative = (point_out-center(self.ring))/self.ring[2] if point_out is not None else None
        row = dict(frame_index=index, mode=mode, reason=reason, source=source,
                   switched=int(mode != self.previous_mode), detector_called=int(detect),
                   observed=int(observation is not None), center_x="", center_y="",
                   ring_x="", ring_y="", near_ring=int(self.near_ring),
                   occlusion_suspected=int(suspect), uncertainty=uncertainty,
                   innovation_nis=nis if nis is not None else "", confidence=confidence if confidence is not None else "",
                   missing_frames=self.missing)
        if point_out is not None:
            row.update(center_x=float(point_out[0]), center_y=float(point_out[1]),
                       ring_x=float(relative[0]), ring_y=float(relative[1]))
        self.previous_mode = mode
        return row
