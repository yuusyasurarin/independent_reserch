"""Fresh YOLO + cheap causal tracking + bounded-lag correction (condition G).

--realtime simulates constant-rate file arrival; it is not a camera latency test.
The saved video retains original timestamps; wall-release times are logged.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, resolve, save_json, save_csv, sha256, stats, environment
from src.adaptive_tracking import AdaptiveBallPolicy
from src.tracking import TrackingPolicy
from src.data import inspect_video, load_gt
from src.detector import Detector
from src.evaluation import tracking_metrics, detection_metrics
from src.lag_tracking import FixedLagBall, METHODS
from src.reporting import plots, save_failures
from scripts.research_g import profile_config, PROFILES
from scripts.run_experiment import draw


def run(args):
    import cv2
    import psutil
    import torch
    cv2.setNumThreads(0)
    if args.device=='cpu':
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    cfg=profile_config(args.profile)
    cfg['device']=args.device
    if args.imgsz!=cfg['imgsz']:
        cfg['g_development_override']=dict(imgsz=args.imgsz,
            reason='Explicit experiment10 development study of lower-cost inference; not val-selected')
        cfg['imgsz']=args.imgsz
    cfg['save_video']=not args.no_video
    video=resolve(cfg['video']);info=inspect_video(video)
    gt_path=resolve(cfg['ground_truth']);gt=load_gt(gt_path,info)
    name=f"{args.size}_{args.condition}_{args.profile}_{args.method}_{args.imgsz}_{args.device.replace(':','')}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    output=resolve(args.output)/name;output.mkdir(parents=True,exist_ok=False)
    save_json(output/'status.json',dict(status='initializing'))
    cap=writer=None
    try:
        detector=Detector(cfg,args.size,args.condition)
        env=environment()
        assert env['versions']['torch'].split('+')[0]=='2.5.1' and env['opencv']=='4.10.0'
        manifest=dict(condition=args.condition,model_size=args.size,profile=args.profile,config=cfg,
            video=info,ground_truth=str(gt_path),gt_sha256=sha256(gt_path),weights=detector.metadata,
            weights_path=str(detector.path),method=args.method,lag_s=args.lag,
            rendering=cfg['save_video'],realtime_pacing=args.realtime,loop_seconds=0,
            environment=env,cpu_threads=torch.get_num_threads(),pinned_runtime=True,
            cache='None; actual YOLO calls on causally requested frames only',
            interpretation='experiment10 development; fixed-lag smoothing, not independent generalization',
            operating_settings_scope='Explicit G development override' if 'g_development_override' in cfg else 'Exported val settings retained',
            source_sha256={str(p.relative_to(ROOT)):sha256(p) for folder in ('src','scripts') for p in (ROOT/folder).glob('*.py')},
            command=sys.argv,warmup_calls=3,precision='FP32',
            kcf_initialization='init plus first update on the same clean detection image',
            frontend='AdaptiveBallPolicy' if args.condition=='G' else 'TrackingPolicy',
            comparison_runner_version=2)
        save_json(output/'run_config.json',manifest)
        policy=(AdaptiveBallPolicy(cfg,info['fps']) if args.condition=='G'
                else TrackingPolicy(args.condition,cfg,info['fps']))
        smoother=FixedLagBall(info['fps'],info['width'],info['height'],args.lag,args.method)
        detector.warmup(info['width'],info['height'])
        if args.device!='cpu':torch.cuda.reset_peak_memory_stats(detector.device)
        cap=cv2.VideoCapture(str(video));assert cap.isOpened()
        if cfg['save_video']:
            writer=cv2.VideoWriter(str(output/'tracked.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),info['fps'],(info['width'],info['height']))
            assert writer.isOpened()
        image_queue=deque();trace=deque(maxlen=30)
        outputs=[];online=[];logs=[];timings=[];releases=[]
        calls=0;sleep_s=0.;peak_frames=0;peak_rss=0
        process=psutil.Process()
        start=time.perf_counter()
        n=min(args.max_frames,info['frame_count']) if args.max_frames else info['frame_count']
        rawlog=(output/'candidates.jsonl').open('w',encoding='utf-8')
        onlinelog=(output/'online.jsonl').open('w',encoding='utf-8')
        tracklog=(output/'tracking.jsonl').open('w',encoding='utf-8')
        def pace(deadline):
            nonlocal sleep_s
            wait=start+deadline-time.perf_counter()
            if args.realtime and wait>0:
                t=time.perf_counter();time.sleep(wait);sleep_s+=time.perf_counter()-t
        def emit(pred):
            i=pred['frame_id']
            pace(i/info['fps']+args.lag)
            if writer is not None:
                queued_i,display=image_queue.popleft();assert queued_i==i
                draw(display,pred,trace,gt[i])
                text=f"{args.size}/{args.condition} {args.method} lag={args.lag:.2f}s available={pred['available_through_frame']} tail={not pred['full_lookahead']}"
                cv2.putText(display,text,(12,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                writer.write(display)
            pred.update(cycle=0,global_frame_id=i)
            release=time.perf_counter()-start
            releases.append(dict(frame_id=i,source_time_s=i/info['fps'],
                available_through_frame=pred['available_through_frame'],lookahead_frames=pred['lookahead_frames'],
                full_lookahead=pred['full_lookahead'],wall_release_s=release,
                simulated_arrival_to_release_s=release-i/info['fps'] if args.realtime else None))
            tracklog.write(json.dumps(pred,allow_nan=False)+'\n');outputs.append(pred)
        try:
            for i in range(n):
                pace(i/info['fps'])
                begin=time.perf_counter();sleep_before=sleep_s
                ok,frame=cap.read();assert ok
                if writer is not None:image_queue.append((i,frame))
                peak_frames=max(peak_frames,len(image_queue))
                detection_start=time.perf_counter()
                detected=policy.needs_detection(i,frame)
                prepare_ms=(time.perf_counter()-detection_start)*1000
                candidates=[];dt=dict(preprocess_ms=0.,inference_ms=0.,postprocess_ms=0.)
                if detected:
                    candidates,dt=detector.detect(frame);calls+=1
                a=time.perf_counter();p=policy.update(frame,i,candidates,detected)
                frontend_ms=prepare_ms+(time.perf_counter()-a)*1000
                p.update(yolo_executed=detected,cycle=0,global_frame_id=i)
                a=time.perf_counter();out=smoother.push(p,candidates,detected,frame)
                backend_ms=(time.perf_counter()-a)*1000
                a=time.perf_counter()
                if out is not None:emit(out)
                render_ms=(time.perf_counter()-a-(sleep_s-sleep_before))*1000
                log=dict(frame_id=i,cycle=0,global_frame_id=i,yolo_executed=detected,candidates=candidates)
                rawlog.write(json.dumps(log,allow_nan=False)+'\n');onlinelog.write(json.dumps(p,allow_nan=False)+'\n')
                online.append(p);logs.append(log)
                if i%30==0:
                    memory=process.memory_info();peak_rss=max(peak_rss,getattr(memory,'peak_wset',memory.rss))
                timings.append(dict(frame_id=i,yolo_executed=detected,**dt,frontend_ms=frontend_ms,
                    backend_ms=backend_ms,render_save_ms=render_ms,
                    processing_ms=(time.perf_counter()-begin-(sleep_s-sleep_before))*1000))
                if (i+1)%100==0:print(name, i+1,'frames; YOLO',calls,flush=True)
            tail_start=time.perf_counter();tail_sleep=sleep_s
            for out in smoother.flush():emit(out)
            tail_processing=time.perf_counter()-tail_start-(sleep_s-tail_sleep)
        finally:
            rawlog.close();onlinelog.close();tracklog.close()
        if writer is not None:writer.release();writer=None
        cap.release();cap=None
        elapsed=time.perf_counter()-start
        assert len(outputs)==n
        tracking,per_frame=tracking_metrics(gt,outputs,info['fps'])
        forward,_=tracking_metrics(gt,online,info['fps'])
        full,_=tracking_metrics(gt,[p for p in outputs if p['full_lookahead']],info['fps'])
        detection,curve=detection_metrics(gt,logs,cfg['confidence'])
        performance=dict(processed_frames=n,yolo_calls=calls,warmup_yolo_calls=3,
            realized_yolo_calls_per_video_second=calls/(n/info['fps']),processing_fps=n/(elapsed-sleep_s),
            wall_fps=n/elapsed,wall_seconds=elapsed,pacing_sleep_seconds=sleep_s,
            processing_ms=stats([r['processing_ms'] for r in timings]),
            frontend_ms=stats([r['frontend_ms'] for r in timings]),backend_ms=stats([r['backend_ms'] for r in timings]),
            tail_processing_s=tail_processing,peak_buffer_frames=peak_frames,
            peak_rgb_buffer_bytes=peak_frames*info['width']*info['height']*3,
            peak_state_rows=smoother.peak_rows,peak_process_rss_or_peak_wset_bytes=peak_rss,
            simulated_release_latency_s=stats([r['simulated_arrival_to_release_s'] for r in releases if args.realtime]),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(detector.device) if args.device!='cpu' else None,
            timing_scope='Fresh inference + decode + frontend + smoother + drawing + log/video writes and EOF tail; excludes setup/warmup/post-run evaluation; processing FPS excludes explicit pacing waits',
            latency_scope='Simulated file arrival to video-write completion; not camera/display latency' if args.realtime else 'Unpaced file benchmark: wall times are NOT live latency')
        metrics=dict(condition=args.condition,model_size=args.size,complete_video=n==info['frame_count'],tracking=tracking,
            online_tracking=forward,full_lookahead_tracking=full,detection=detection,performance=performance)
        save_json(output/'metrics.json',metrics)
        save_csv(output/'per_frame.csv',per_frame);save_csv(output/'timing.csv',timings);save_csv(output/'release.csv',releases)
        save_csv(output/'tracking.csv',[{k:p[k] for k in ('frame_id','timestamp_s','source','confidence','lookahead_frames','full_lookahead')} |
            {k:p['bbox'][j] if p['bbox'] else None for j,k in enumerate(('x1','y1','x2','y2'))} for p in outputs])
        plots(output,per_frame,curve);save_failures(video,gt,outputs,per_frame,output,8,info['fps'])
        save_json(output/'status.json',dict(status='complete',complete_video=n==info['frame_count']))
        print('Saved:',output,'TSR',tracking['tsr_iou_05'],'fps',performance['processing_fps'],flush=True)
        return output
    except Exception as error:
        save_json(output/'status.json',dict(status='failed',error=repr(error)))
        raise
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',choices=['B','D','E','G'],default='G')
    p.add_argument('--size',choices=['s','m'],default='s')
    p.add_argument('--profile',choices=PROFILES,default='f')
    p.add_argument('--method',choices=METHODS,default='bridge')
    p.add_argument('--lag',type=float,default=2.)
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--output',default='results/condition_g/fresh')
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--realtime',action='store_true')
    args=p.parse_args()
    if args.imgsz<32 or args.imgsz%32 or args.lag<0 or args.cpu_threads<1 or args.max_frames<0:p.error('Invalid settings')
    if args.condition in ('B','D') and (args.method!='forward' or args.lag!=0):
        p.error('B/D baselines require --method forward --lag 0')
    if args.condition!='G' and args.imgsz!=960:p.error('Only G permits explicit input-resolution development overrides')
    import socket
    def no_network(sock,address):raise RuntimeError('Fresh G experiment has no network access')
    socket.socket.connect=no_network
    run(args)
