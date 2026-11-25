"""Real-time fixed-delay ball tracking: YOLOv5s + local LK + bridge (condition G).

Input is a live camera, or a video file paced as if it were a camera. Each frame
is shown exactly --delay seconds after capture. The processor never works on a
backlog older than --max-backlog seconds: stale frames are skipped and later
filled by bridge interpolation from neighbouring YOLO anchors. Accuracy on
experiment10 was measured offline; camera latency itself is not measured here.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import environment, power_state, resolve, save_json, stats
from src.adaptive_tracking import AdaptiveBallPolicy
from src.alternatives import RoiScheduler
from src.shot import ShotJudge
from src.detector import Detector
from src.lag_tracking import FixedLagBall
from src.realtime import DelayedOutputStore, TrackingWorker
from scripts.research_g import PROFILES, profile_config


WINDOW='Fixed-delay ball tracking'


class CameraSource:
    def __init__(self,index,width,height,fps):
        import cv2
        cap=cv2.VideoCapture(index,cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release();cap=cv2.VideoCapture(index)
        if not cap.isOpened():raise RuntimeError(f'Cannot open camera index {index}')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        cap.set(cv2.CAP_PROP_FPS,fps);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        ok,frame=cap.read()
        if not ok:raise RuntimeError('Camera returned no frame')
        self.cap=cap;self.fps=float(fps)
        self.height,self.width=frame.shape[:2]
        self.info=dict(kind='camera',index=index,requested=dict(width=width,height=height,fps=fps),
                       actual=dict(width=self.width,height=self.height,reported_fps=cap.get(cv2.CAP_PROP_FPS)),
                       timestamp_basis='host clock right after cap.read(); camera/USB latency not measured')
    def read(self):return self.cap.read()
    def release(self):self.cap.release()


class PacedVideoSource:
    """Reads a file at its own frame rate so it arrives like a camera stream."""
    def __init__(self,path,clock):
        import cv2
        self.cap=cv2.VideoCapture(str(path))
        if not self.cap.isOpened():raise RuntimeError(f'Cannot open video {path}')
        self.fps=self.cap.get(cv2.CAP_PROP_FPS)
        self.width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH));self.height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.fps>0 or min(self.width,self.height)<=0:raise ValueError('Invalid video metadata')
        self.clock=clock;self.start=None;self.i=0
        self.info=dict(kind='video',path=str(path),fps=self.fps,width=self.width,height=self.height,
                       frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                       timestamp_basis='simulated arrival at start + i/fps (sleep-paced); not a camera')
    def read(self):
        if self.start is None:self.start=self.clock()
        wait=self.start+self.i/self.fps-self.clock()
        if wait>0:time.sleep(wait)
        self.i+=1
        return self.cap.read()
    def release(self):self.cap.release()


class RecordingWriter:
    """Encodes delayed frames on its own thread. A full queue drops recording
    frames rather than stalling the display loop; frames.jsonl stays complete."""
    def __init__(self,path,fps,size,capacity=90,raw=False):
        import cv2
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
        if not self.writer.isOpened():raise RuntimeError('Cannot open video writer')
        self.queue=queue.Queue(maxsize=capacity);self.written=0;self.dropped=0;self.trace=deque(maxlen=30);self.raw=raw;self.rim=None
        self.thread=threading.Thread(target=self.loop,name='record',daemon=True);self.thread.start()
    def put(self,image,result,hud):
        try:self.queue.put_nowait((image,result,hud))
        except queue.Full:self.dropped+=1
    def loop(self):
        while True:
            item=self.queue.get()
            if item is None:break
            image,result,hud=item
            if self.raw:canvas=image
            else:canvas=image.copy();draw(canvas,result,self.trace,hud,self.rim)
            self.writer.write(canvas);self.written+=1
    def close(self):
        self.queue.put(None);self.thread.join();self.writer.release()


def draw(image,result,trace,hud,rim=None):
    import cv2
    if rim is not None:
        x1,y1,x2,y2=map(round,rim);cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),1)
    if result is not None and result['bbox']:
        x1,y1,x2,y2=map(round,result['bbox'])
        colour=(0,150,255) if result['source'].startswith('G_') else (0,220,0) if result['source']=='YOLO' else (255,200,0)
        cv2.rectangle(image,(x1,y1),(x2,y2),colour,2)
        trace.append(((x1+x2)//2,(y1+y2)//2))
    else:trace.clear()
    for a,b in zip(trace,list(trace)[1:]):cv2.line(image,a,b,(255,200,0),1)
    for k,line in enumerate(hud):
        cv2.putText(image,line,(12,26+22*k),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3)
        cv2.putText(image,line,(12,26+22*k),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)


def parse_rim(text):
    values=[float(v) for v in text.split(',')]
    if len(values)!=4 or values[2]<=values[0] or values[3]<values[1]:raise ValueError('--rim needs x1,y1,x2,y2')
    return values


def select_rim(source):
    """Drag a box tightly around the ring opening on the first frame (Enter to accept)."""
    import cv2
    title='Mark the rim: drag around the ring, then Enter'
    ok,frame=source.read()
    if not ok:raise RuntimeError('No frame to mark the rim on')
    box=cv2.selectROI(title,frame,showCrosshair=True,fromCenter=False)
    cv2.destroyWindow(title)
    x,y,w,h=map(float,box)
    if w<=0:raise RuntimeError('No rim marked')
    print(f'rim = {x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}  (reuse with --rim)',flush=True)
    return [x,y,x+w,y+h]


def run(args):
    import cv2
    import numpy as np
    import torch
    clock=time.perf_counter
    cfg=profile_config(args.profile,args.config);cfg['device']=args.device
    if args.max_interval:cfg['adaptive']['max_interval']=args.max_interval
    if args.max_yolo_hz:cfg['adaptive']['max_yolo_hz']=args.max_yolo_hz
    if args.imgsz!=cfg['imgsz']:
        cfg['g_development_override']=dict(imgsz=args.imgsz,reason='Explicit real-time development override; not val-selected')
        cfg['imgsz']=args.imgsz
    if args.device=='cpu':torch.set_num_threads(args.cpu_threads);torch.set_num_interop_threads(1)
    detector=Detector(cfg,args.size,'G')
    source=(PacedVideoSource(resolve(args.video),clock) if args.video
            else CameraSource(args.camera_index,args.width,args.height,args.fps))
    fps,width,height=source.fps,source.width,source.height
    rim=parse_rim(args.rim) if args.rim else None
    if args.select_rim:rim=select_rim(source)
    judge=ShotJudge(rim,fps) if rim else None
    power_start=power_state()
    output=None;recorder=None;framelog=None;raw_recorder=None;shotlog=None
    if args.record:
        output=resolve(args.record)/f'realtime_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
        output.mkdir(parents=True,exist_ok=False)
        recorder=RecordingWriter(output/'delayed.mp4',fps,(width,height));recorder.rim=rim
        raw_recorder=RecordingWriter(output/'raw.mp4',fps,(width,height),capacity=300,raw=True) if args.record_raw else None
        shotlog=(output/'shots.jsonl').open('w',encoding='utf-8') if judge else None
        framelog=(output/'frames.jsonl').open('w',encoding='utf-8')
        save_json(output/'run_config.json',dict(args=vars(args),config=cfg,source=source.info,weights=detector.metadata,
            weights_path=str(detector.path),method='bridge',rim=rim,
            roi=dict(roi_imgsz=args.roi_imgsz,full_every=args.roi_full_every) if args.roi_imgsz else None,environment=environment(),power_at_start=power_start,command=sys.argv,
            notes='Display delay is fixed after capture; the smoother looks ahead --lookahead seconds; '
                  'frames older than --max-backlog are skipped when newer frames wait.'))
    policy=AdaptiveBallPolicy(cfg,fps)
    smoother=FixedLagBall(fps,width,height,args.lookahead,'bridge',max_bridge_s=args.max_bridge)
    roi=RoiScheduler(cfg['imgsz'],args.roi_imgsz,args.roi_full_every) if args.roi_imgsz else None
    detect=(lambda image:roi.detect(detector,policy,image)) if roi else detector.detect
    worker=TrackingWorker(policy,smoother,detect,fps,args.max_backlog,clock)
    store=DelayedOutputStore(args.delay,args.delay+args.hold_margin)
    detector.warmup(width,height)
    if args.device!='cpu':torch.cuda.reset_peak_memory_stats(detector.device)
    pending=queue.Queue();stop=threading.Event();eof=threading.Event();drained=threading.Event()
    errors=[]
    def capture_loop():
        seq=0
        try:
            while not stop.is_set():
                ok,frame=source.read()
                if not ok:break
                t=clock();store.add_frame(seq,t,frame);pending.put((seq,t,frame));seq+=1
        except Exception:errors.append(traceback.format_exc())
        finally:
            eof.set();pending.put(None);source.release()
    def process_loop():
        try:
            while True:
                item=pending.get()
                if item is None:break
                if stop.is_set():continue
                seq,t,frame=item
                for out in worker.step(seq,t,frame,not pending.empty()):store.add_result(out)
            if not stop.is_set():
                for out in worker.flush():store.add_result(out)
        except Exception:errors.append(traceback.format_exc());stop.set()
        finally:drained.set()
    threads=[threading.Thread(target=capture_loop,name='capture',daemon=True),
             threading.Thread(target=process_loop,name='process',daemon=True)]
    session_start=clock()
    for th in threads:th.start()
    if not args.no_display:cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
    trace=deque(maxlen=30);display_delay=[];result_margin=[];missing=0;shown=0;last_shot=None
    print(f'Running: {args.size}/G {args.profile} bridge delay={args.delay}s lookahead={args.lookahead}s '
          f'max_backlog={args.max_backlog}s {width}x{height}@{fps:g} imgsz={cfg["imgsz"]} device={args.device} '
          f'power={power_start["source"]} {power_start["battery_percent"]}%; q/Esc to quit',flush=True)
    try:
        while True:
            now=clock();upto=store.due(now)
            if upto is not None:
                counters=worker.counters
                recent=list(worker.processing_ms)[-60:]
                status=(f"proc {len(recent)*1000/sum(recent) if recent else 0:.1f}fps  YOLO {worker.recent_yolo_hz():.1f}/s"
                        f"  backlog {worker.backlog_s[-1] if worker.backlog_s else 0:.2f}s",
                        f"dropped {counters['dropped']}/{counters['received']} ({counters['dropped']/max(1,counters['received']):.1%})"
                        f"  buffered {store.buffered_frames()}  missing {missing}")
                for seq,t_capture,image,result in store.release(upto):
                    delay=now-t_capture;display_delay.append(delay)
                    margin=None
                    if result is None:missing+=1
                    else:margin=t_capture+args.delay-result['t_result'];result_margin.append(margin)
                    shot=None
                    if judge is not None:
                        bbox=result['bbox'] if result else None
                        shot=judge.update(seq,((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2) if bbox else None)
                        if shot is not None:
                            last_shot=(seq,shot)
                            print(f"SHOT {shot['result'].upper()} ({shot['reason']}) at {shot['time_s']:.1f}s  score {judge.score()}",flush=True)
                            if shotlog is not None:shotlog.write(json.dumps(shot,allow_nan=False)+'\n');shotlog.flush()
                    if raw_recorder is not None:raw_recorder.put(image,None,None)
                    if framelog is not None:
                        framelog.write(json.dumps(dict(frame_id=seq,capture_s=t_capture-session_start,release_s=now-session_start,
                            display_delay_s=delay,result_ready=result is not None,result_margin_s=margin,
                            processed=result['processed'] if result else None,source=result['source'] if result else None,
                            bbox=result['bbox'] if result else None,confidence=result['confidence'] if result else None,
                            yolo_executed=result['yolo_executed'] if result else None),allow_nan=False)+'\n')
                    if recorder is not None or seq==upto:
                        hud=[f"f={seq} delay {delay:.2f}s/{args.delay:.1f}s  {'result '+result['source'] if result else 'RESULT MISSING'}",*status]
                        if judge is not None:
                            sc=judge.score()
                            recent=f"  last {last_shot[1]['result'].upper()}" if last_shot and seq-last_shot[0]<2*fps else ''
                            hud.append(f"shots: made {sc['made']} / {sc['attempts']}{recent}")
                        if recorder is not None:recorder.put(image,result,hud)
                        if seq==upto and not args.no_display:
                            canvas=image.copy();draw(canvas,result,trace,hud,rim)
                            if args.display_scale!=1:canvas=cv2.resize(canvas,None,fx=args.display_scale,fy=args.display_scale)
                            cv2.imshow(WINDOW,canvas);shown+=1
            elif shown==0 and not args.no_display:
                canvas=np.zeros((round(height*args.display_scale),round(width*args.display_scale),3),np.uint8)
                cv2.putText(canvas,f'buffering {max(0.,args.delay-(now-session_start)):.1f}s',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                cv2.imshow(WINDOW,canvas)
            if errors or (eof.is_set() and drained.is_set() and store.buffered_frames()==0):break
            if args.duration and now-session_start>=args.duration:break
            nxt=store.next_due_time()
            wait=max(1,min(40,int((nxt-clock())*1000))) if nxt is not None else 5
            if args.no_display:time.sleep(wait/1000)
            else:
                key=cv2.waitKey(wait)&0xFF
                if key in (ord('q'),27):break
                try:
                    if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
                except cv2.error:break
    finally:
        stop.set()
        for th in threads:th.join(timeout=5)
        if recorder is not None:recorder.close()
        if raw_recorder is not None:raw_recorder.close()
        if shotlog is not None:shotlog.close()
        if framelog is not None:framelog.close()
        if not args.no_display:cv2.destroyAllWindows()
    elapsed=clock()-session_start
    summary=dict(session_seconds=elapsed,captured=worker.counters['received'],displayed=shown,
        recorded=recorder.written if recorder else 0,recording_dropped=recorder.dropped if recorder else 0,
        raw_recorded=raw_recorder.written if raw_recorder else 0,raw_recording_dropped=raw_recorder.dropped if raw_recorder else 0,
        shots=dict(rim=rim,**judge.score(),events=judge.events) if judge else None,
        tracking=worker.summary(),display_delay_s=stats(display_delay),result_margin_s=stats(result_margin),
        missing_results=missing,discarded_unreleased=store.discarded,
        display_delay_target_s=args.delay,lookahead_s=args.lookahead,max_backlog_s=args.max_backlog,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(detector.device) if args.device!='cpu' else None,
        power=dict(start=power_start,end=power_state()),settings=dict(size=args.size,profile=args.profile,imgsz=cfg['imgsz']),
        errors=errors,
        scope='Capture-to-release delay on the host clock; camera and monitor latency excluded. Accuracy not evaluated here.')
    if output is not None:save_json(output/'session.json',summary);print('Saved:',output)
    print(json.dumps({k:summary[k] for k in ('session_seconds','captured','missing_results','discarded_unreleased')},indent=1))
    print('tracking:',json.dumps({k:v for k,v in summary['tracking'].items() if k not in ('processing_ms','backlog_s')}))
    print('processing_ms:',summary['tracking']['processing_ms']);print('backlog_s:',summary['tracking']['backlog_s'])
    print('display_delay_s:',summary['display_delay_s']);print('result_margin_s:',summary['result_margin_s'])
    if errors:print('\n'.join(errors),file=sys.stderr);sys.exit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera-index',type=int,default=0)
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    p.add_argument('--fps',type=float,default=30.,help='nominal camera rate used as the tracker time base')
    p.add_argument('--video',help='process a video file paced at its own fps instead of a camera')
    p.add_argument('--size',choices=['s','m'],default='s');p.add_argument('--profile',choices=PROFILES,default='f')
    p.add_argument('--config',help='experiment config JSON (weights, thresholds); default configs/experiment_f.json')
    p.add_argument('--imgsz',type=int,default=960);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--delay',type=float,default=3.,help='fixed display delay after capture, seconds')
    p.add_argument('--lookahead',type=float,default=.4,help='smoother lookahead, seconds (at least --max-bridge)')
    p.add_argument('--max-bridge',type=float,default=.4,help='longest gap bridge interpolation fills, seconds')
    p.add_argument('--max-backlog',type=float,default=1.,help='skip frames older than this when newer ones wait, seconds')
    p.add_argument('--hold-margin',type=float,default=1.,help='extra seconds frames may wait for display before being discarded')
    p.add_argument('--display-scale',type=float,default=1.);p.add_argument('--no-display',action='store_true')
    p.add_argument('--duration',type=float,default=0.,help='stop automatically after this many seconds (0 = until q/Esc or end of file)')
    p.add_argument('--record',help='directory; saves delayed.mp4, frames.jsonl, run_config.json, session.json')
    p.add_argument('--record-raw',action='store_true',help='also save the camera frames without drawings as raw.mp4 (needs --record)')
    p.add_argument('--roi-imgsz',type=int,default=0,help='YOLO on a crop around the predicted ball at this input size (0 = whole frame)')
    p.add_argument('--roi-full-every',type=int,default=5,help='every N-th YOLO call uses the whole frame even while tracking')
    p.add_argument('--max-interval',type=int,default=0,help='override adaptive max_interval (frames)')
    p.add_argument('--max-yolo-hz',type=float,default=0,help='override adaptive max_yolo_hz')
    p.add_argument('--rim',help='shot judgement: rim box x1,y1,x2,y2 in frame pixels')
    p.add_argument('--select-rim',action='store_true',help='shot judgement: drag a box around the rim on the first frame')
    a=p.parse_args()
    if (a.imgsz<32 or a.imgsz%32 or a.delay<0 or a.lookahead<0 or a.max_bridge<=0 or a.max_backlog<0
            or a.hold_margin<=0 or a.fps<=0 or a.display_scale<=0 or a.duration<0):
        p.error('Invalid settings')
    if a.lookahead<a.max_bridge:p.error('--lookahead must be at least --max-bridge, or bridge cannot see its later anchor')
    if a.lookahead+a.max_backlog>a.delay:p.error('--lookahead + --max-backlog must not exceed --delay, or results arrive after display time')
    if a.no_display and not a.record:p.error('--no-display needs --record, otherwise nothing is produced')
    if a.record_raw and not a.record:p.error('--record-raw needs --record')
    if a.select_rim and a.no_display:p.error('--select-rim needs a display; use --rim instead')
    if a.roi_imgsz and (a.roi_imgsz%32 or a.roi_imgsz>a.imgsz):p.error('--roi-imgsz must be a multiple of 32 and <= --imgsz')
    run(a)
