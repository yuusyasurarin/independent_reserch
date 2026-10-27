"""F: causal sparse optical flow, velocity prediction and budgeted detection.

No ground truth, future frames, person detector or calibrated 3D physics is used.
All thresholds are engineering defaults, not independently validated settings.
"""
from collections import deque
import math

from .common import center, valid_box


DEFAULTS = dict(max_interval=3, max_yolo_hz=15, min_points=4, max_points=30,
                fb_error_px=1.5, min_flow_fraction=.5, motion_gate_diameters=2.5,
                velocity_alpha=.7, max_prediction_frames=2, low_confidence=.10,
                motion_bonus=.12, size_penalty=.04, challenge_margin=.15)


class AdaptiveBallPolicy:
    def __init__(self, cfg, fps):
        import cv2
        import numpy as np
        self.cv, self.np = cv2, np
        self.cfg, self.fps = cfg, fps
        self.options = DEFAULTS | cfg.get('adaptive', {})
        o = self.options
        if (o['max_interval'] < 1 or o['max_yolo_hz'] <= 0 or o['min_points'] < 2
                or o['max_points'] < o['min_points'] or o['fb_error_px'] <= 0
                or not 0 < o['min_flow_fraction'] <= 1 or not 0 < o['velocity_alpha'] <= 1
                or o['motion_gate_diameters'] <= 0 or o['max_prediction_frames'] < 0
                or not cfg['candidate_conf'] <= o['low_confidence'] <= cfg['confidence']):
            raise ValueError('Invalid adaptive tracking settings')
        self.window = max(1, round(fps))
        self.budget = max(1, min(self.window, math.floor(o['max_yolo_hz'] * self.window / fps)))
        self.calls = deque()
        self.last_detection = None
        self.last_high = None
        self.last_observed = None
        self.last_yolo_box = None
        self.last_yolo_frame = None
        self.box = None
        self.velocity = np.zeros(2)
        self.track_id = None
        self.next_id = 1
        self.previous_gray = None
        self.points = None
        self.prepared_frame = None
        self.challenger = None

    def translated(self, box, shift):
        return [box[0]+float(shift[0]), box[1]+float(shift[1]),
                box[2]+float(shift[0]), box[3]+float(shift[1])]

    def clear(self):
        self.box = None
        self.points = None
        self.velocity[:] = 0
        self.track_id = None
        self.last_yolo_box = self.last_yolo_frame = None

    def seed(self, gray):
        if self.box is None:
            return None
        x1,y1,x2,y2 = self.box
        x,y=max(0,math.ceil(x1)),max(0,math.ceil(y1))
        crop=gray[y:math.floor(y2),x:math.floor(x2)]
        if crop.shape[0]<3 or crop.shape[1]<3:
            return None
        points=self.cv.goodFeaturesToTrack(crop, maxCorners=self.options['max_points'],
            qualityLevel=.01, minDistance=2, blockSize=3)
        return points+self.np.array([x,y],dtype=self.np.float32) if points is not None else None

    def prepare(self, frame, frame_id):
        """Read only the current/past image; called before deciding on YOLO."""
        if self.prepared_frame == frame_id:
            return
        self.prepared_frame = frame_id
        self.gray = self.cv.cvtColor(frame, self.cv.COLOR_BGR2GRAY)
        self.flow_box = None
        self.flow_points = None
        self.flow_fraction = 0.
        self.flow_error = None
        self.events = []
        self.request_reason = 'interval'
        h,w = frame.shape[:2]
        self.predicted_box = self.translated(self.box, self.velocity) if self.box else None
        if self.box is None or self.previous_gray is None or self.points is None:
            return
        if len(self.points) < self.options['min_points']:
            return
        # Build pyramids only around the ball, retaining full pixel resolution.
        # Cropping both frames identically preserves image coordinates after offsetting.
        diameter = max(self.box[2]-self.box[0], self.box[3]-self.box[1])
        margin=max(24.,diameter*1.5,float(self.np.linalg.norm(self.velocity))*3)
        x1=max(0,math.floor(self.box[0]-margin));y1=max(0,math.floor(self.box[1]-margin))
        x2=min(w,math.ceil(self.box[2]+margin));y2=min(h,math.ceil(self.box[3]+margin))
        offset=self.np.array([x1,y1],dtype=self.np.float32)
        local_points=self.points-offset
        prev_crop=self.previous_gray[y1:y2,x1:x2]
        next_crop=self.gray[y1:y2,x1:x2]
        nxt, status, _ = self.cv.calcOpticalFlowPyrLK(prev_crop, next_crop, local_points, None,
            winSize=(21,21), maxLevel=3,
            criteria=(self.cv.TERM_CRITERIA_EPS | self.cv.TERM_CRITERIA_COUNT, 20, .01))
        if nxt is None:
            return
        back, reverse_status, _ = self.cv.calcOpticalFlowPyrLK(next_crop, prev_crop, nxt, None,
            winSize=(21,21), maxLevel=3,
            criteria=(self.cv.TERM_CRITERIA_EPS | self.cv.TERM_CRITERIA_COUNT, 20, .01))
        if back is None:
            return
        errors = self.np.linalg.norm(back.reshape(-1,2)-local_points.reshape(-1,2), axis=1)
        nxt=nxt+offset
        old, new = self.points.reshape(-1,2), nxt.reshape(-1,2)
        good = (status.ravel()==1) & (reverse_status.ravel()==1) & self.np.isfinite(errors)
        good &= (errors <= self.options['fb_error_px']) & self.np.isfinite(new).all(axis=1)
        good &= (new[:,0]>=0) & (new[:,0]<w) & (new[:,1]>=0) & (new[:,1]<h)
        if good.sum() < self.options['min_points']:
            return
        shift = self.np.median((new-old)[good], axis=0)
        residual = self.np.linalg.norm((new-old)-shift, axis=1)
        good &= residual <= max(2., diameter*.15)
        self.flow_fraction = float(good.sum()/len(old))
        if good.sum() < self.options['min_points'] or self.flow_fraction < self.options['min_flow_fraction']:
            return
        shift = self.np.median((new-old)[good], axis=0)
        box = self.translated(self.box, shift)
        self.flow_error = float(self.np.median(errors[good]))
        # A rapid bounce/catch triggers detection rather than forcing constant velocity.
        if self.np.linalg.norm(shift-self.velocity) > max(8., diameter*.75):
            self.events.append('motion_change')
        if valid_box(box,w,h):
            self.flow_box, self.flow_points = box, nxt[good].reshape(-1,1,2)

    def needs_detection(self, frame_id, frame=None):
        if frame is None:
            raise ValueError('Condition F requires the current frame for causal scheduling')
        self.prepare(frame, frame_id)
        while self.calls and frame_id-self.calls[0] >= self.window:
            self.calls.popleft()
        since = frame_id-self.last_detection if self.last_detection is not None else math.inf
        if self.box is None:
            request = since >= self.options['max_interval']
            self.request_reason = 'lost_search'
        elif self.flow_box is None or 'motion_change' in self.events:
            request = True
            self.request_reason = 'flow_unreliable' if self.flow_box is None else 'motion_change'
        else:
            request = since >= self.options['max_interval']
        if request and len(self.calls) >= self.budget:
            self.events.append('detection_budget_deferred')
            return False
        return request

    def choose(self, candidates, reference, frame_id):
        high = [c for c in candidates if c['confidence'] >= self.cfg['confidence']]
        top = max(high, key=lambda c:c['confidence'], default=None)
        if reference is None:
            self.challenger = None
            return top, top is not None
        x,y = center(reference)
        diameter = max(reference[2]-reference[0], reference[3]-reference[1])
        gate = max(12.,diameter*self.options['motion_gate_diameters'])
        def distance(c):
            return math.dist(center(c['bbox']), (x,y))
        def score(c):
            b=c['bbox']
            ratio=max((b[2]-b[0])*(b[3]-b[1]),1.)/max((reference[2]-reference[0])*(reference[3]-reference[1]),1.)
            return c['confidence'] + self.options['motion_bonus']*max(0.,1-distance(c)/gate) - self.options['size_penalty']*abs(math.log(ratio))
        local = [c for c in high if distance(c)<=gate]
        choice = max(local,key=score,default=None)
        # A distant stronger candidate must persist across two detection frames.
        # This lets a background lock release without allowing a one-frame teleport.
        if top and distance(top)>gate and (choice is None or top['confidence']>choice['confidence']+self.options['challenge_margin']):
            if self.challenger and frame_id-self.challenger[0]<=self.fps*self.cfg['max_age_s']:
                previous=self.challenger[1]
                if math.dist(center(previous),center(top['bbox'])) <= gate:
                    self.events.append('global_reacquisition')
                    self.challenger=None
                    return top,True
            self.challenger=(frame_id,list(top['bbox']))
            self.events.append('global_candidate_pending')
        else:
            self.challenger=None
        if choice:
            return choice,True
        # Weak candidates never initialize a track or extend its high-confidence TTL.
        if self.last_high is not None and (frame_id-self.last_high)/self.fps <= self.cfg['max_age_s']:
            low=[c for c in candidates if self.options['low_confidence']<=c['confidence']<self.cfg['confidence']
                 and distance(c)<=max(6.,diameter*.75)]
            choice=max(low,key=score,default=None)
            if choice:
                self.events.append('low_confidence_recovery')
                return choice,False
        return None,False

    def update(self, frame, frame_id, candidates, detected):
        self.prepare(frame,frame_id)
        h,w=frame.shape[:2]
        source,state,confidence='none','lost',None
        old_box=list(self.box) if self.box else None
        expired=self.last_high is not None and (frame_id-self.last_high)/self.fps>self.cfg['max_age_s']+1e-9
        if expired:
            self.clear()
            self.flow_box=self.predicted_box=None
            self.events.append('expired_since_high_confidence_yolo')
        reference=self.flow_box or self.predicted_box
        choice=None;strong=False
        if detected:
            self.calls.append(frame_id)
            self.last_detection=frame_id
            self.events.append('detect_'+self.request_reason)
            choice,strong=self.choose(candidates,reference,frame_id)
        if choice is not None:
            new=self.box is None or 'global_reacquisition' in self.events
            if new:
                self.clear()
                self.track_id=self.next_id;self.next_id+=1
            box=list(choice['bbox'])
            if self.last_yolo_box is not None and frame_id>self.last_yolo_frame:
                measured=(self.np.array(center(box))-self.np.array(center(self.last_yolo_box)))/(frame_id-self.last_yolo_frame)
                self.velocity=self.options['velocity_alpha']*measured+(1-self.options['velocity_alpha'])*self.velocity
            self.box=box
            self.last_yolo_box=list(box);self.last_yolo_frame=frame_id
            self.last_observed=frame_id
            if strong:self.last_high=frame_id
            source='YOLO' if strong else 'YOLO_low'
            state='reinitialized' if new and self.next_id>2 else ('initialized' if new else 'corrected')
            confidence=choice['confidence']
            self.points=self.seed(self.gray)
        elif self.box is not None and self.flow_box is not None:
            shift=self.np.array(center(self.flow_box))-self.np.array(center(old_box))
            self.velocity=self.options['velocity_alpha']*shift+(1-self.options['velocity_alpha'])*self.velocity
            self.box=self.flow_box
            self.last_observed=frame_id
            self.points=self.flow_points
            source,state='LK','tracking'
        elif (self.box is not None and self.predicted_box is not None and self.last_observed is not None
              and frame_id-self.last_observed<=self.options['max_prediction_frames']):
            self.box=self.predicted_box
            self.points=None  # Do not turn an unobserved prediction into new visual evidence.
            source,state='motion_prediction','predicted'
        else:
            self.clear()
        if self.box is not None and not valid_box(self.box,w,h):
            self.clear();source,state,confidence='none','lost',None
            self.events.append('out_of_bounds')
        self.previous_gray=self.gray
        return dict(frame_id=frame_id,timestamp_s=frame_id/self.fps,bbox=list(self.box) if self.box else None,
            track_id=self.track_id,state=state,source=source,confidence=confidence,kcf_success_flag=None,
            events=list(self.events),age_since_yolo_s=(frame_id-self.last_high)/self.fps if self.last_high is not None else None,
            flow_fraction=self.flow_fraction,flow_fb_error_px=self.flow_error,
            velocity_px_per_frame=self.velocity.tolist(),detection_reason=self.request_reason if detected else None,
            detection_budget=self.budget,detection_window_frames=self.window)
