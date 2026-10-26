from __future__ import annotations

import math
from .common import BYTE_VERSION, center, iou, valid_box


def highest(candidates):
    return max(candidates,key=lambda x:x['confidence'],default=None)


def associate(candidates,box,dt,base_px,speed_px_s):
    if box is None:
        return highest(candidates)
    x,y=center(box)
    gate=base_px+speed_px_s*max(0.,dt)
    ranked=[]
    for candidate in candidates:
        cx,cy=center(candidate['bbox'])
        distance=math.hypot(cx-x,cy-y)
        if distance<=gate:
            # Distance gate permits non-overlapping fast balls. IoU is a weak tie-breaker.
            cost=distance/gate-.1*iou(box,candidate['bbox'])-.05*candidate['confidence']
            ranked.append((cost,candidate))
    return min(ranked,key=lambda x:x[0])[1] if ranked else None


def create_kcf():
    import cv2
    if hasattr(cv2,'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    if hasattr(getattr(cv2,'legacy',None),'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError('KCF unavailable: install only the pinned opencv-contrib-python wheel')


class TrackingPolicy:
    def __init__(self,condition,cfg,fps,kcf_factory=None):
        if condition not in 'ABCDE':
            raise ValueError('Unknown condition')
        self.condition,self.cfg,self.fps=condition,cfg,fps
        self.box=None
        self.last_yolo_time=None
        self.current_id=None
        self.next_id=1
        self.ever_initialized=False
        self.kcf=None
        self.kcf_factory=kcf_factory or create_kcf
        self.byte=None
        if condition=='E' and kcf_factory is None:
            create_kcf()  # fail before processing if unavailable
        if condition=='C':
            import supervision as sv
            if sv.__version__!=BYTE_VERSION:
                raise ValueError(f'Expected supervision {BYTE_VERSION}, found {sv.__version__}')
            self.sv=sv
            self.byte=sv.ByteTrack(track_activation_threshold=cfg['byte_activation_threshold'],
                lost_track_buffer=max(1,round(cfg['byte_lost_buffer_s']*30)),
                minimum_matching_threshold=cfg['byte_matching_threshold'],frame_rate=round(fps),minimum_consecutive_frames=1)

    def needs_detection(self,frame_id):
        return self.condition in 'ABC' or frame_id%self.cfg['detect_interval']==0

    def lose(self):
        self.box=None
        self.kcf=None
        self.current_id=None

    def update(self,frame,frame_id,candidates,detected):
        now=frame_id/self.fps
        h,w=frame.shape[:2]
        state,source,confidence='lost','none',None
        kcf_ok=None
        events=[]
        if self.last_yolo_time is not None and now-self.last_yolo_time>self.cfg['max_age_s']+1e-9:
            if self.box is not None: events.append('expired_since_last_yolo')
            self.lose()
        if self.condition in 'AB':
            choice=highest([c for c in candidates if c['confidence']>=self.cfg['confidence']])
            self.box=choice['bbox'] if choice else None
            if choice:
                source,state,confidence='YOLO','detected',choice['confidence']
            self.current_id=None  # no concealed identity tracking in baselines
        elif self.condition=='C':
            import numpy as np
            eligible=[c for c in candidates if c['confidence']>.1]
            detections=self.sv.Detections(xyxy=np.asarray([c['bbox'] for c in eligible],dtype=float).reshape(-1,4),
                confidence=np.asarray([c['confidence'] for c in eligible],dtype=float),class_id=np.zeros(len(eligible),dtype=int))
            tracked=self.byte.update_with_detections(detections)
            tracks=[dict(bbox=b.tolist(),confidence=float(s),track_id=int(i)) for b,s,i in zip(tracked.xyxy,tracked.confidence,tracked.tracker_id)]
            same=[c for c in tracks if c['track_id']==self.current_id]
            choice=highest(same or tracks)
            if choice:
                new=self.current_id!=choice['track_id'] or self.box is None
                state=('reinitialized' if self.ever_initialized else 'initialized') if new else 'tracking'
                self.current_id=choice['track_id']; self.box=choice['bbox']
                confidence=choice['confidence']; source='YOLO+ByteTrack'
                self.last_yolo_time=now; self.ever_initialized=True
            else:
                self.box=None  # do not emit a lost Kalman prediction as an observation
        else:
            if detected:
                eligible=[c for c in candidates if c['confidence']>=self.cfg['confidence']]
                age=now-self.last_yolo_time if self.last_yolo_time is not None else 0
                choice=associate(eligible,self.box,age,self.cfg['association_base_px'],self.cfg['association_speed_px_s'])
                if choice:
                    initializing=self.box is None
                    if initializing:
                        self.current_id=self.next_id; self.next_id+=1
                    state=('reinitialized' if self.ever_initialized else 'initialized') if initializing else 'corrected'
                    self.box=list(choice['bbox']); self.last_yolo_time=now
                    self.ever_initialized=True; source='YOLO'; confidence=choice['confidence']
                    if self.condition=='E':
                        self.kcf=self.kcf_factory()
                        x1,y1,x2,y2=self.box
                        x,y=max(0,int(x1)),max(0,int(y1))
                        bw,bh=min(w-x,max(1,math.ceil(x2)-x)),min(h-y,max(1,math.ceil(y2)-y))
                        try:
                            initialized=self.kcf.init(frame,(x,y,bw,bh))
                            if initialized is False: self.kcf=None
                        except Exception as e:
                            self.kcf=None; events.append('kcf_init_failed:'+type(e).__name__)
                else:
                    # A scheduled YOLO miss or gate rejection invalidates the old location immediately.
                    if self.box is not None: events.append('yolo_miss_or_gate_rejection')
                    self.lose()
            elif self.box is not None:
                if self.condition=='D':
                    source,state='hold','holding'
                elif self.kcf is not None:
                    try:
                        kcf_ok,xywh=self.kcf.update(frame)
                        x,y,bw,bh=map(float,xywh)
                        box=[x,y,x+bw,y+bh]
                        if kcf_ok and valid_box(box,w,h):
                            self.box=box;source,state='KCF','tracking'
                        else:
                            self.lose();events.append('kcf_failed_or_out_of_bounds')
                    except Exception as e:
                        self.lose(); kcf_ok=False; events.append('kcf_update_failed:'+type(e).__name__)
                else:
                    self.lose();events.append('kcf_unavailable_after_initialization')
        if self.box is not None and not valid_box(self.box,w,h):
            self.lose();source,state,confidence='none','lost',None
        return dict(frame_id=frame_id,timestamp_s=now,bbox=list(self.box) if self.box else None,
                    track_id=self.current_id,state=state,source=source,confidence=confidence,
                    kcf_success_flag=kcf_ok,events=events,
                    age_since_yolo_s=now-self.last_yolo_time if self.last_yolo_time is not None else None)
