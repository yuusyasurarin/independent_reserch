"""M1--M6 single-ball policies. No detector, GUI or ground truth dependency."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import numpy as np

from adaptive_tracking import MotionFilter, center, valid_box


def iou(a, b):
    if not valid_box(a) or not valid_box(b):
        return 0.0
    x = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    intersection = x*y
    return intersection / (a[2]*a[3]+b[2]*b[3]-intersection)


def expand(box, factor=2.):
    x, y, w, h = box
    return x-(factor-1)*w/2, y-(factor-1)*h/2, w*factor, h*factor


def union(boxes):
    x = min(b[0] for b in boxes)
    y = min(b[1] for b in boxes)
    return x, y, max(b[0]+b[2] for b in boxes)-x, max(b[1]+b[3] for b in boxes)-y


def ring_zone(ring, alpha=1.5, up=2., down=2., scale=1.):
    x, y = center(ring)
    w = ring[2]
    return x-alpha*w*scale, y-up*w*scale, 2*alpha*w*scale, (up+down)*w*scale


@dataclass
class Settings:
    method: str = 'M2'
    max_prediction_s: float = .2
    gate: float = 9.21
    process_noise: float = 1800.
    yolo_variance: float = 25.
    kcf_variance: float = 100.
    association_px: float = 160.
    max_kcf_age_s: float = .5
    confirm_frames: int = 2
    players: str = 'P0'
    ring_policy: str = 'R0'
    alpha: float = 1.5
    beta_up: float = 2.
    beta_down: float = 2.


class ExperimentTracker:
    def __init__(self, settings, kcf_factory, manual_ring=None):
        self.cfg = settings
        self.factory = kcf_factory
        self.kcf = self.kf = self.box = None
        self.ring = tuple(manual_ring) if manual_ring is not None else None
        self.manual_ring = manual_ring is not None
        self.people = {}
        self.next_person = 0
        self.player_ids = []
        self.last_observation = self.last_detection = None
        self.time = None
        self.near_ring = False
        self.pending = None
        self.confirmations = 0

    def snapshot(self):
        # KCF cannot be cloned. Replay always starts on a new detector result,
        # which reinitializes/discards KCF on the matching source image.
        return copy.deepcopy({k: v for k, v in vars(self).items() if k not in ('kcf', 'factory')})

    def restore(self, state):
        factory = self.factory
        vars(self).clear()
        vars(self).update(copy.deepcopy(state))
        self.factory, self.kcf = factory, None

    def _players(self, boxes, t):
        unmatched = set(self.people)
        result = {}
        for box in boxes:
            matches = [(iou(box, self.people[k][0]), k) for k in unmatched]
            score, key = max(matches, default=(0, -1))
            if score < .1:
                key = self.next_person
                self.next_person += 1
            else:
                unmatched.remove(key)
            result[key] = (tuple(box), t)
        for key in unmatched:
            if t-self.people[key][1] <= .5:
                result[key] = self.people[key]
        self.people = result

    def search_context(self):
        t = self.time or 0.
        people = [self.people[k][0] for k in self.player_ids
                  if k in self.people and t-self.people[k][1] <= .5]
        return dict(box=self.box, ring=self.ring, near_ring=self.near_ring,
                    lost=self.box is None or self.last_observation is None or t-self.last_observation > .05,
                    player_roi=union(people) if people else None)

    def _choose(self, candidates):
        candidates = [(tuple(b), float(s)) for b, s in candidates if valid_box(b) and math.isfinite(s)]
        if not candidates:
            return None
        if self.box is None:
            return max(candidates, key=lambda v: v[1])
        choices = []
        for box, confidence in candidates:
            distance = float(np.linalg.norm(center(box)-center(self.box)))
            if self.kf is not None:
                cost = self.kf.innovation(center(box), self.cfg.yolo_variance)
                if cost > self.cfg.gate and not (self.near_ring and self.cfg.ring_policy in ('R1', 'R4')):
                    continue
            else:
                cost = distance / self.cfg.association_px
                if distance > self.cfg.association_px:
                    continue
                if self.cfg.method == 'M1':
                    cost -= iou(expand(box), expand(self.box))
            choices.append((cost-confidence*.1, box, confidence))
        return min(choices, key=lambda v: v[0])[1:] if choices else None

    def step(self, frame, index, timestamp, detection=None):
        c = self.cfg
        use_kf = c.method in ('M3', 'M5', 'M6')
        use_kcf = c.method in ('M4', 'M5', 'M6')
        dt = 0 if self.time is None else timestamp-self.time
        if dt < 0:
            raise ValueError('Frame timestamps must be increasing')
        self.time = timestamp
        if detection is not None:
            if detection.get('ring') is not None and not self.manual_ring:
                self.ring = tuple(detection['ring'])
            self._players(detection.get('people', []), timestamp)
        if self.last_observation is not None and timestamp-self.last_observation > max(c.max_prediction_s, c.max_kcf_age_s):
            self.box = self.kf = self.kcf = None
        if self.kf is not None:
            self.kf.predict(dt, c.process_noise)
        probe = self.box
        if self.kf is not None and self.box is not None:
            probe = (* (self.kf.x[:2]-np.array(self.box[2:])/2), *self.box[2:])
        self.near_ring = bool(self.ring is not None and probe is not None and
                              iou(probe, ring_zone(self.ring, c.alpha, c.beta_up, c.beta_down,
                                                  1.2 if self.near_ring else 1.)) > 0)
        visual_priority = self.near_ring and c.ring_policy in ('R1', 'R4')
        observation = None
        confidence = None
        source = 'missing'
        # One observation per frame: a detector result takes precedence over KCF.
        if detection is not None:
            selected = self._choose(detection.get('balls', []))
            if selected is not None:
                observation, confidence = selected
                if c.method == 'M1' and self.box is None:
                    consistent = self.pending is not None and iou(expand(self.pending), expand(observation)) > 0
                    self.confirmations = self.confirmations+1 if consistent else 1
                    self.pending = observation
                    if self.confirmations < c.confirm_frames:
                        observation = None
                if observation is not None:
                    source = 'cnn' if c.method == 'M1' else 'yolo'
                    self.last_detection = timestamp
                    self.kcf = self.factory(frame, observation) if use_kcf else None
            else:
                self.confirmations = 0
                self.pending = None
            if observation is None:
                self.kcf = None
        elif use_kcf and self.kcf is not None:
            if self.last_detection is not None and timestamp-self.last_detection <= c.max_kcf_age_s:
                ok, candidate = self.kcf.update(frame)
                if ok and valid_box(candidate):
                    accepted = (self.kf is None or visual_priority or
                                self.kf.innovation(center(candidate), c.kcf_variance) <= c.gate)
                    if accepted and self.box is not None:
                        ratio = candidate[2]*candidate[3]/(self.box[2]*self.box[3])
                        accepted = .25 <= ratio <= 4 and np.linalg.norm(center(candidate)-center(self.box)) <= c.association_px
                    if accepted:
                        observation, source = tuple(candidate), 'kcf'
            if observation is None:
                self.kcf = None
        output_box = None
        if observation is not None:
            self.box = tuple(observation)
            self.last_observation = timestamp
            self.player_ids = []
            if use_kf:
                if self.kf is None or visual_priority:
                    self.kf = MotionFilter(center(observation))
                else:
                    self.kf.correct(center(observation), c.kcf_variance if source == 'kcf' else c.yolo_variance)
                output_box = (* (self.kf.x[:2]-np.array(self.box[2:])/2), *self.box[2:])
            else:
                output_box = self.box
        else:
            if probe is not None and c.players != 'P0':
                nearby = [(np.linalg.norm(center(box)-center(probe)), key) for key, (box, t) in self.people.items()
                          if timestamp-t <= .5 and iou(expand(probe, 4), box) > 0]
                if nearby:
                    self.player_ids = [key for _, key in sorted(nearby)[:1 if c.players == 'P1' else len(nearby)]]
            if use_kf and self.kf is not None and self.last_observation is not None:
                if timestamp-self.last_observation <= c.max_prediction_s and not visual_priority:
                    output_box, source = probe, 'prediction'
                elif timestamp-self.last_observation > c.max_prediction_s:
                    self.box = self.kf = self.kcf = None
        row = dict(frame_index=index, timestamp_s=timestamp, source=source, observed=int(observation is not None),
                   bbox_x='', bbox_y='', bbox_w='', bbox_h='', center_x='', center_y='',
                   ring_bbox_x='', ring_bbox_y='', ring_bbox_w='', ring_bbox_h='', ring_x='', ring_y='',
                   confidence='' if confidence is None else confidence, near_ring=int(self.near_ring),
                   player_ids=';'.join(map(str, self.player_ids)),
                   observation_age_s='' if self.last_observation is None else timestamp-self.last_observation)
        if self.ring is not None:
            row.update(zip(('ring_bbox_x', 'ring_bbox_y', 'ring_bbox_w', 'ring_bbox_h'), self.ring))
        if output_box is not None:
            row.update(zip(('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'), map(float, output_box)))
            p = center(output_box)
            row.update(center_x=float(p[0]), center_y=float(p[1]))
            if self.ring is not None:
                relative = (p-center(self.ring))/self.ring[2]
                row.update(ring_x=float(relative[0]), ring_y=float(relative[1]))
        return row
