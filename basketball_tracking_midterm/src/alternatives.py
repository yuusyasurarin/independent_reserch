"""Two alternatives to condition G's front end (development study, experiment10 only).

H-template : System 1 uses colour template matching (normalised cross-correlation)
             of the last YOLO ball patch instead of sparse Lucas-Kanade feature points.
             A ~30 px ball has few corners, so LK often keeps <4 points; a template
             needs no corners.
ROI YOLO   : while the ball is tracked, YOLO runs on a crop around the predicted
             position instead of the whole frame. The crop is resized with the SAME
             pixel scale as the full-frame imgsz (default 960 for a 1280-wide frame,
             i.e. 0.75), so the ball looks exactly as large as during training.
             Lost-ball search and every N-th call still use the whole frame.

Scheduling, budget, candidate choice, expiry and the fixed-lag smoother are unchanged
(inherited from AdaptiveBallPolicy / FixedLagBall). Thresholds are engineering defaults.
"""
import math

from .adaptive_tracking import AdaptiveBallPolicy
from .common import center, valid_box

TEMPLATE_DEFAULTS = dict(template_threshold=.6, template_pad=.25, template_min_px=8)


class TemplateBallPolicy(AdaptiveBallPolicy):
    """AdaptiveBallPolicy with template matching replacing LK in prepare()."""

    def __init__(self, cfg, fps):
        super().__init__(cfg, fps)
        self.options = TEMPLATE_DEFAULTS | self.options
        if not 0 < self.options['template_threshold'] < 1:
            raise ValueError('template_threshold must be in (0, 1)')
        self.template = None
        self.color = None

    def clear(self):
        super().clear()
        self.template = None

    def seed(self, gray):
        # Called by update() right after a YOLO box is accepted; store its padded patch.
        self.template = None
        if self.box is not None and self.color is not None:
            x1, y1, x2, y2 = self.box
            pad = self.options['template_pad']*max(x2-x1, y2-y1)
            h, w = self.color.shape[:2]
            a, b = max(0, math.floor(x1-pad)), max(0, math.floor(y1-pad))
            c, d = min(w, math.ceil(x2+pad)), min(h, math.ceil(y2+pad))
            if min(c-a, d-b) >= self.options['template_min_px']:
                self.template = (self.color[b:d, a:c].copy(), (x1-a, y1-b, x2-a, y2-b))
        return super().seed(gray)

    def prepare(self, frame, frame_id):
        if self.prepared_frame == frame_id:
            return
        self.prepared_frame = frame_id
        self.color = frame
        self.gray = self.cv.cvtColor(frame, self.cv.COLOR_BGR2GRAY)
        self.flow_box = self.flow_points = None
        self.flow_fraction, self.flow_error = 0., None
        self.events, self.request_reason = [], 'interval'
        h, w = frame.shape[:2]
        self.predicted_box = self.translated(self.box, self.velocity) if self.box else None
        if self.box is None or self.template is None:
            return
        patch, (ox1, oy1, ox2, oy2) = self.template
        ph, pw = patch.shape[:2]
        ref = self.predicted_box
        diameter = max(self.box[2]-self.box[0], self.box[3]-self.box[1])
        margin = max(24., diameter*1.5, float(self.np.linalg.norm(self.velocity))*3)
        cx, cy = (ref[0]+ref[2])/2, (ref[1]+ref[3])/2
        x1 = max(0, math.floor(cx-pw/2-margin)); y1 = max(0, math.floor(cy-ph/2-margin))
        x2 = min(w, math.ceil(cx+pw/2+margin)); y2 = min(h, math.ceil(cy+ph/2+margin))
        if x2-x1 < pw or y2-y1 < ph:
            return
        score = self.cv.matchTemplate(frame[y1:y2, x1:x2], patch, self.cv.TM_CCOEFF_NORMED)
        _, best, _, (mx, my) = self.cv.minMaxLoc(score)
        self.flow_fraction, self.flow_error = float(best), float(1-best)
        if not math.isfinite(best) or best < self.options['template_threshold']:
            return
        box = [x1+mx+ox1, y1+my+oy1, x1+mx+ox2, y1+my+oy2]
        shift = self.np.array([box[0]-self.box[0], box[1]-self.box[1]])
        if self.np.linalg.norm(shift-self.velocity) > max(8., diameter*.75):
            self.events.append('motion_change')
        if valid_box(box, w, h):
            # update() copies flow_points into self.points; keep it non-None.
            self.flow_box, self.flow_points = box, self.np.zeros((1, 1, 2), self.np.float32)


def roi_window(reference, frame_w, frame_h, imgsz, roi_imgsz):
    """Square crop (x1, y1, side) around reference at the full-frame inference scale."""
    scale = imgsz/max(frame_w, frame_h)
    side = min(round(roi_imgsz/scale), frame_w, frame_h)
    cx, cy = (reference[0]+reference[2])/2, (reference[1]+reference[3])/2
    x1 = int(min(max(0, round(cx-side/2)), frame_w-side))
    y1 = int(min(max(0, round(cy-side/2)), frame_h-side))
    return x1, y1, side


class RoiScheduler:
    """Decides full-frame vs ROI for each YOLO call the policy has already requested."""

    def __init__(self, imgsz, roi_imgsz, full_every):
        if roi_imgsz <= 0 or roi_imgsz % 32 or roi_imgsz > imgsz or full_every < 1:
            raise ValueError('roi_imgsz must be a positive multiple of 32 <= imgsz, full_every >= 1')
        self.imgsz, self.roi_imgsz, self.full_every = imgsz, roi_imgsz, full_every
        self.since_full = math.inf
        self.counts = dict(full=0, roi=0)

    def window(self, policy, frame):
        """None = whole frame; else (x1, y1, side)."""
        reference = policy.flow_box or policy.predicted_box or policy.box
        if (reference is None or policy.request_reason == 'lost_search'
                or self.since_full+1 >= self.full_every):
            self.since_full = 0
            self.counts['full'] += 1
            return None
        self.since_full += 1
        self.counts['roi'] += 1
        h, w = frame.shape[:2]
        return roi_window(reference, w, h, self.imgsz, self.roi_imgsz)

    def detect(self, detector, policy, frame):
        win = self.window(policy, frame)
        if win is None:
            candidates, dt = detector.detect(frame)
            return candidates, dict(dt, roi=False)
        x1, y1, side = win
        candidates, dt = detector.detect(frame[y1:y1+side, x1:x1+side], imgsz=self.roi_imgsz)
        for c in candidates:
            b = c['bbox']
            c['bbox'] = [b[0]+x1, b[1]+y1, b[2]+x1, b[3]+y1]
        return candidates, dict(dt, roi=True, roi_window=[x1, y1, side])

    def relative_cost(self):
        """Inference cost relative to all-full-frame calls (pixels ~ FLOPs)."""
        n = self.counts['full']+self.counts['roi']
        return (self.counts['full']+self.counts['roi']*(self.roi_imgsz/self.imgsz)**2)/n if n else 0.


# --- Recovery when the ROI locks onto a head / shoe (2026-11-25 development study) ---
# All options default to "off", which reproduces the behaviour above exactly.
RECOVERY_DEFAULTS = dict(keep_challenger_on_roi=False, instant_switch_conf=0.)


class RecoveryMixin:
    """Changes only how a distant candidate may replace the current track.

    keep_challenger_on_roi : an ROI call cannot see a distant ball, so it no longer
                             cancels a pending distant challenger from a full-frame call.
    instant_switch_conf    : on a full-frame call, a distant candidate with confidence
                             >= this value (and challenge_margin above the best local one)
                             replaces the track at once instead of after two detections.
    """
    roi_call = False

    def choose(self, candidates, reference, frame_id):
        o = RECOVERY_DEFAULTS | self.options
        if reference is not None and not self.roi_call and o['instant_switch_conf'] > 0:
            high = [c for c in candidates if c['confidence'] >= self.cfg['confidence']]
            top = max(high, key=lambda c: c['confidence'], default=None)
            diameter = max(reference[2]-reference[0], reference[3]-reference[1])
            gate = max(12., diameter*o['motion_gate_diameters'])
            def distance(c):
                return math.dist(center(c['bbox']), center(reference))
            local = max((c['confidence'] for c in high if distance(c) <= gate), default=0.)
            if (top and top['confidence'] >= o['instant_switch_conf'] and distance(top) > gate
                    and top['confidence'] > local+o['challenge_margin']):
                self.challenger = None
                self.events += ['global_reacquisition', 'instant_switch']
                return top, True
        saved, before = self.challenger, len(self.events)
        choice, strong = super().choose(candidates, reference, frame_id)
        added = self.events[before:]
        if (self.roi_call and o['keep_challenger_on_roi'] and saved is not None
                and not {'global_candidate_pending', 'global_reacquisition'} & set(added)):
            self.challenger = saved
        return choice, strong


def with_recovery(policy_class):
    return type('Recovery'+policy_class.__name__, (RecoveryMixin, policy_class), {})


class RecoveryRoiScheduler(RoiScheduler):
    """RoiScheduler with time-based full frames and escalation after a doubtful ROI call.

    full_interval : whole frame when >= this many frames passed since the last whole
                    frame (replaces the every-N-th-call rule when > 0).
    escalate      : after an ROI call that gave no high-confidence ball, or whose box
                    area changed by more than size_jump times, the next call is whole-frame.
    """

    def __init__(self, imgsz, roi_imgsz, full_every, full_interval=0, escalate=False, size_jump=2.):
        super().__init__(imgsz, roi_imgsz, full_every)
        if full_interval < 0 or size_jump <= 1:
            raise ValueError('full_interval >= 0 and size_jump > 1 required')
        self.full_interval, self.escalate, self.size_jump = full_interval, escalate, size_jump
        self.last_full_frame = None
        self.force_full = False
        self.reference = None
        self.counts.update(escalated=0)

    def window(self, policy, frame):
        frame_id = policy.prepared_frame
        reference = policy.flow_box or policy.predicted_box or policy.box
        self.reference = list(reference) if reference else None
        if self.full_interval:
            due = self.last_full_frame is None or frame_id-self.last_full_frame >= self.full_interval
        else:
            due = self.since_full+1 >= self.full_every
        if self.force_full and reference is not None and policy.request_reason != 'lost_search':
            self.counts['escalated'] += 1
        if reference is None or policy.request_reason == 'lost_search' or due or self.force_full:
            self.since_full, self.last_full_frame, self.force_full = 0, frame_id, False
            self.counts['full'] += 1
            return None
        self.since_full += 1
        self.counts['roi'] += 1
        h, w = frame.shape[:2]
        return roi_window(reference, w, h, self.imgsz, self.roi_imgsz)

    def after_update(self, result, used_roi):
        """Call after policy.update() on a frame where YOLO ran."""
        if not (self.escalate and used_roi):
            return
        box, ref = result['bbox'], self.reference
        if result['source'] != 'YOLO' or box is None:
            self.force_full = True
        elif ref is not None:
            area = max((box[2]-box[0])*(box[3]-box[1]), 1.)/max((ref[2]-ref[0])*(ref[3]-ref[1]), 1.)
            self.force_full = not 1/self.size_jump <= area <= self.size_jump
