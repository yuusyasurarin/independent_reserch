"""Bounded-lookahead estimates from arriving states and optional current images.

The online System 1 / System 2 scheduler stays causal. This module only revises
not-yet-emitted positions. No GT or whole-video access is available here.
"""
from collections import deque
import math
import numpy as np
from .common import center, valid_box


METHODS = ('forward', 'bridge', 'rts', 'beam', 'bidir')


def geometry(box):
    return np.array([*center(box),box[2]-box[0],box[3]-box[1]],dtype=float)


def rectangle(values):
    x,y,w,h=values
    return [float(x-w/2),float(y-h/2),float(x+w/2),float(y+h/2)]


class FixedLagBall:
    def __init__(self, fps, width, height, lag_s=2., method='bridge', max_bridge_s=.4,
                 beam_width=8, candidate_limit=4):
        if method not in METHODS or lag_s<0 or fps<=0 or max_bridge_s<=0:
            raise ValueError('Invalid fixed-lag configuration')
        self.fps,self.width,self.height=fps,width,height
        self.lag=round(lag_s*fps)
        self.method=method
        self.gap=max(1,round(max_bridge_s*fps))
        self.history=self.gap
        self.beam_width,self.candidate_limit=beam_width,candidate_limit
        self.rows=deque()
        self.latest=-1
        self.next_emit=0
        self.x=self.P=None
        self.segment=0
        self.last_id=None
        self.last_observation=None
        self.A=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],float)
        self.H=np.array([[1,0,0,0],[0,1,0,0]],float)
        self.peak_rows=0
        self.images=deque(maxlen=self.gap+1)
        self.last_anchor=None

    def reverse_updates(self,frame,row):
        """System 1 visual evidence propagated backwards from a System 2 anchor."""
        import cv2
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        i=row['online']['frame_id']
        self.images.append((i,gray))
        p=row['online']
        if p['source']!='YOLO' or p['bbox'] is None:return
        box=list(p['bbox']); points=None;previous=gray
        proposals={}
        left=self.last_anchor[0] if self.last_anchor else max(0,i-self.gap)
        for j,other in list(self.images)[-2::-1]:
            if j<left:break
            x,y=math.ceil(box[0]),math.ceil(box[1]);x2,y2=math.floor(box[2]),math.floor(box[3])
            if points is None:
                crop=previous[max(0,y):y2,max(0,x):x2]
                if min(crop.shape)<3:break
                points=cv2.goodFeaturesToTrack(crop,30,.01,2,blockSize=3)
                if points is None:break
                points+=np.array([max(0,x),max(0,y)],np.float32)
            if len(points)<4:break
            diameter=max(box[2]-box[0],box[3]-box[1]);margin=max(24.,diameter*2)
            x0=max(0,math.floor(box[0]-margin));y0=max(0,math.floor(box[1]-margin))
            x3=min(self.width,math.ceil(box[2]+margin));y3=min(self.height,math.ceil(box[3]+margin))
            offset=np.array([x0,y0],np.float32);local=points-offset
            a,b=previous[y0:y3,x0:x3],other[y0:y3,x0:x3]
            new,ok,_=cv2.calcOpticalFlowPyrLK(a,b,local,None,winSize=(21,21),maxLevel=3)
            if new is None:break
            back,reverse_ok,_=cv2.calcOpticalFlowPyrLK(b,a,new,None,winSize=(21,21),maxLevel=3)
            if back is None:break
            error=np.linalg.norm((back-local).reshape(-1,2),axis=1)
            delta=(new-local).reshape(-1,2)
            good=(ok.ravel()==1)&(reverse_ok.ravel()==1)&np.isfinite(error)&(error<=1.5)&np.isfinite(delta).all(axis=1)
            if good.sum()<4:break
            shift=np.median(delta[good],axis=0)
            good&=np.linalg.norm(delta-shift,axis=1)<=max(2.,diameter*.15)
            if good.sum()<4 or good.mean()<.5:break
            shift=np.median(delta[good],axis=0)
            box=[box[0]+float(shift[0]),box[1]+float(shift[1]),box[2]+float(shift[0]),box[3]+float(shift[1])]
            if not valid_box(box,self.width,self.height):break
            proposals[j]=list(box)
            points=(new+offset)[good].reshape(-1,1,2);previous=other
        coherent=False
        if self.last_anchor and left in proposals:
            old=self.last_anchor[1]
            coherent=math.dist(center(old),center(proposals[left]))<=max(6.,max(old[2]-old[0],old[3]-old[1])*.75)
        for stored in self.rows:
            t=stored['online']['frame_id']
            if t in proposals and t>=self.next_emit:
                stored['reverse']=dict(bbox=proposals[t],left=left,right=i,coherent=coherent)
        self.last_anchor=(i,list(p['bbox']))

    def filter_row(self,row):
        p=row['online']; b=p['bbox']; i=p['frame_id']
        observed=b is not None and p['source'] not in ('motion_prediction','hold','none')
        reset=observed and (self.x is None or (self.last_observation is not None and i-self.last_observation>self.gap)
                           or (p.get('track_id') is not None and self.last_id is not None and p['track_id']!=self.last_id))
        if reset:
            self.x=np.array([*center(b),0.,0.])
            self.P=np.diag([25.,25.,100.,100.]);self.segment+=1
        if self.x is None:return
        diameter=max(b[2]-b[0],b[3]-b[1]) if b else 20.
        acceleration=max(2.,diameter*.15)
        q=acceleration**2*np.array([[.25,0,.5,0],[0,.25,0,.5],[.5,0,1,0],[0,.5,0,1]])
        xp=self.x.copy() if reset else self.A@self.x
        pp=self.P.copy() if reset else self.A@self.P@self.A.T+q
        row['xp'],row['pp']=xp,pp
        row['back_gain']=None if reset else np.linalg.solve(pp,self.A@self.P).T
        self.x,self.P=xp.copy(),pp.copy()
        if observed:
            sigma=max(2.,diameter*(.10 if p['source']=='YOLO' else .30))
            R=np.eye(2)*sigma**2
            K=np.linalg.solve(self.H@pp@self.H.T+R,self.H@pp).T
            self.x=xp+K@(np.array(center(b))-self.H@xp)
            residual=np.eye(4)-K@self.H
            self.P=residual@pp@residual.T+K@R@K.T
            self.last_observation=i;self.last_id=p.get('track_id')
        row['xf'],row['pf'],row['segment']=self.x.copy(),self.P.copy(),self.segment
        row['observed']=observed

    def push(self, online, candidates, detected, frame=None):
        i=online['frame_id']
        if i!=self.latest+1:raise ValueError('Frames must arrive exactly once in order')
        self.latest=i
        row=dict(online=dict(online),candidates=list(candidates),detected=bool(detected))
        if self.method=='rts':self.filter_row(row)
        self.rows.append(row)
        if self.method=='bidir':
            if frame is None:raise ValueError('Bidirectional LK requires the current image')
            self.reverse_updates(frame,row)
        self.peak_rows=max(self.peak_rows,len(self.rows))
        if i<self.lag:return None
        return self.emit(i-self.lag)

    def interpolate(self, nodes, target):
        before=[n for n in nodes if n[0]<=target and n[1] is not None]
        after=[n for n in nodes if n[0]>=target and n[1] is not None]
        if not before or not after:return None
        a,b=before[-1],after[0]
        if a[0]==b[0]:return list(a[1])
        if b[0]-a[0]>self.gap:return None
        # No very long jumps or extreme size changes forced through occlusion.
        ga,gb=geometry(a[1]),geometry(b[1])
        diameter=max(ga[2],ga[3],gb[2],gb[3])
        if np.linalg.norm(ga[:2]-gb[:2])>max(30.,diameter*2)*(b[0]-a[0]):return None
        if max(ga[2]/gb[2],gb[2]/ga[2],ga[3]/gb[3],gb[3]/ga[3])>3:return None
        f=(target-a[0])/(b[0]-a[0])
        return rectangle(ga*(1-f)+gb*f)

    def beam_nodes(self, rows):
        """Bounded beam search over detector events, including explicit missing nodes.

        Scores combine detector confidence and capped motion/size costs. The
        bounded acceleration penalty permits bounces rather than forcing a parabola.
        This is a beam approximation, not an exact second-order Viterbi solver.
        """
        hypotheses=[(0.,None,None,None,[])]
        for row in rows:
            if not row['detected']:continue
            i=row['online']['frame_id']
            candidates=sorted((c for c in row['candidates'] if c['confidence']>=.1),
                              key=lambda c:c['confidence'],reverse=True)[:self.candidate_limit]
            choices=candidates+[None]
            branches=[]
            for cost,last_i,last_box,velocity,path in hypotheses:
                for c in choices:
                    if c is None:
                        branches.append((cost+1.8,last_i,last_box,velocity,path+[(i,None)]))
                        continue
                    box=c['bbox'];g=geometry(box);new_velocity=None
                    penalty=-math.log(max(c['confidence'],1e-6))
                    if last_box is not None and i-last_i<=self.gap:
                        old=geometry(last_box);dt=i-last_i
                        diameter=max(8.,g[2],g[3],old[2],old[3])
                        delta=g[:2]-old[:2]
                        new_velocity=delta/dt
                        if np.linalg.norm(delta)>max(30.,diameter*2)*dt:penalty+=4.
                        if velocity is not None:
                            error=np.linalg.norm(delta-velocity*dt)/(diameter+8.*dt)
                            penalty+=.7*min(error**2,4.)
                        penalty+=.15*min(abs(math.log(g[2]*g[3]/(old[2]*old[3]))),3.)
                    branches.append((cost+penalty,i,box,new_velocity,path+[(i,box)]))
            branches.sort(key=lambda h:h[0])
            hypotheses=branches[:self.beam_width]
            minimum=hypotheses[0][0]
            hypotheses=[(c-minimum,i,b,v,p) for c,i,b,v,p in hypotheses]
        return hypotheses[0][-1]

    def estimate(self, rows, index):
        row=rows[index];p=row['online'];target=p['frame_id']
        if self.method=='forward':return p['bbox'],p['source']
        if self.method=='bidir':
            reverse=row.get('reverse')
            if reverse is None or p['source']=='YOLO':return p['bbox'],p['source']
            if p['bbox'] is None or p['source']=='motion_prediction':return reverse['bbox'],'G_bidir'
            if reverse['coherent']:
                f=(target-reverse['left'])/max(1,reverse['right']-reverse['left'])
                return rectangle(geometry(p['bbox'])*(1-f)+geometry(reverse['bbox'])*f),'G_bidir'
            return p['bbox'],p['source']
        if self.method in ('bridge','beam'):
            nodes=(self.beam_nodes(rows) if self.method=='beam' else
                   [(r['online']['frame_id'],r['online']['bbox']) for r in rows
                    if r['online']['source'] in ('YOLO','YOLO_low')])
            box=self.interpolate(nodes,target)
            return (box,'G_'+self.method) if box is not None else (p['bbox'],p['source'])
        # Fixed-lag RTS on stored causal filter states, stopping at reset boundaries.
        if 'xf' not in row:return p['bbox'],p['source']
        future=[]
        for r in rows[index:]:
            if r.get('segment')!=row['segment']:break
            future.append(r)
        if not future:return p['bbox'],p['source']
        x=future[-1]['xf'].copy()
        for j in range(len(future)-2,-1,-1):
            a,b=future[j],future[j+1]
            gain=b['back_gain']
            x=a['xf']+gain@(x-b['xp'])
        base=p['bbox']
        if base is None:
            base=self.interpolate([(r['online']['frame_id'],r['online']['bbox']) for r in rows
                                   if r.get('observed')],target)
        if base is None:return None,'none'
        g=geometry(base);g[:2]=x[:2]
        return rectangle(g),'G_rts'

    def emit(self,target):
        if target!=self.next_emit:raise ValueError('Each target must be emitted once in order')
        rows=list(self.rows)
        index=next(j for j,r in enumerate(rows) if r['online']['frame_id']==target)
        p=rows[index]['online']
        box,source=self.estimate(rows,index)
        if box is not None and not valid_box(box,self.width,self.height):box,source=None,'none'
        out=dict(frame_id=target,timestamp_s=target/self.fps,bbox=box,source=source,
            state='smoothed' if source.startswith('G_') else p['state'],track_id=None,
            confidence=None if source.startswith('G_') else p['confidence'],
            yolo_executed=rows[index]['detected'],events=[],kcf_success_flag=None,
            available_through_frame=self.latest,lookahead_frames=self.latest-target,
            full_lookahead=self.latest-target==self.lag,method=self.method,
            original_source=p['source'])
        self.next_emit+=1
        while self.rows and self.rows[0]['online']['frame_id']<self.next_emit-self.history:
            self.rows.popleft()
        return out

    def flush(self):
        """EOF cannot supply unseen future frames; mark the shortened tail explicitly."""
        while self.next_emit<=self.latest:
            yield self.emit(self.next_emit)
