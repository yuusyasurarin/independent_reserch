"""Sequential offline A-E experiment runner. Missing trained weights are an error."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, environment, hardware_sample, load_config, resolve, save_csv, save_json, sha256, stats
from src.data import inspect_video, load_gt
from src.detector import Detector, verify_weight
from src.evaluation import detection_metrics, tracking_metrics
from src.reporting import plots, save_failures
from src.tracking import TrackingPolicy


def draw(frame,pred,trace,truth=None):
    import cv2
    if pred['bbox']:
        x1,y1,x2,y2=map(round,pred['bbox'])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,150,255),2)
        trace.append(((x1+x2)//2,(y1+y2)//2))
    else:trace.clear()
    if truth and truth['bbox']:
        b=list(map(round,truth['bbox']));cv2.rectangle(frame,tuple(b[:2]),tuple(b[2:]),(0,220,0),2)
    for a,b in zip(trace,list(trace)[1:]):cv2.line(frame,a,b,(255,200,0),1)
    text=f"f={pred['frame_id']} t={pred['timestamp_s']:.2f}s {pred['state']} {pred['source']} id={pred['track_id']}"
    cv2.putText(frame,text,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)


def run_one(cfg,size,condition,args):
    import cv2
    video=resolve(cfg['video'])
    info=inspect_video(video)
    gt_path=None if args.no_gt else resolve(cfg['ground_truth'])
    if gt_path is not None and not gt_path.is_file():
        raise FileNotFoundError(f'{gt_path}; run audit_data.py, or use --no-gt for performance/video only')
    gt=load_gt(gt_path,info)
    output=resolve(args.output)/f'{size}_{condition}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'status.json',{'status':'initializing'})
    detector=None;cap=None;writer=None
    try:
        detector=Detector(cfg,size,condition)
        policy=TrackingPolicy(condition,cfg,info['fps'])
        env=environment()
        pinned=(env['versions']['torch'].split('+')[0]=='2.5.1' and env['versions']['torchvision'].split('+')[0]=='0.20.1'
                and env['versions']['numpy']=='1.26.4' and env['opencv']=='4.10.0')
        if not pinned and not args.allow_version_mismatch:
            raise RuntimeError('Runtime differs from pinned environment. Install the dedicated venv, or explicitly --allow-version-mismatch for a separately labeled smoke test.')
        manifest=dict(config=cfg,condition=condition,model_size=size,video=info,ground_truth=str(gt_path) if gt_path else None,
            gt_sha256=sha256(gt_path) if gt_path else None,weights=detector.metadata,weights_path=str(detector.path),
            command=sys.argv,environment=env,pinned_runtime=pinned,
            cache='No detection or image cache; frames decoded sequentially. OS filesystem cache is not controlled.',
            rendering=cfg['save_video'],realtime_pacing=args.realtime,loop_seconds=args.loop_seconds,
            warmup_calls=3,precision='FP16' if cfg['half'] else 'FP32',
            source_sha256={str(p.relative_to(ROOT)):sha256(p) for folder in ('src','scripts') for p in (ROOT/folder).glob('*.py')},
            interpretation='Repeated-video stability test' if args.loop_seconds else 'Single pass through experiment10',
            timing='CUDA synchronized at each stage; includes transfers/NMS in pre/post stages; not camera-to-display latency')
        overlap=ROOT/'data/audit/overlap.json'
        if overlap.is_file():
            overlap_report=json.loads(overlap.read_text(encoding='utf-8'))
            if overlap_report.get('evaluation_video_sha256')==info['sha256']:
                manifest['dataset_overlap_audit']=overlap_report
        save_json(output/'run_config.json',manifest)
        freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
        (output/'environment.txt').write_text(freeze.stdout,encoding='utf-8')
        detector.warmup(info['width'],info['height'])
        if detector.device.type=='cuda':detector.torch.cuda.reset_peak_memory_stats(detector.device)
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():raise ValueError('Could not open video')
        if cfg['save_video']:
            writer=cv2.VideoWriter(str(output/'tracked.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),info['fps'],(info['width'],info['height']))
            if not writer.isOpened():raise ValueError('Could not create video writer')
        all_timing=[];first_predictions=[];first_logs=[];telemetry=[]
        cycle=frame_id=total=yolo_calls=completed_cycles=0
        trace=deque(maxlen=30)
        started=time.perf_counter();next_telemetry=started
        with (output/'candidates.jsonl').open('w',encoding='utf-8') as rawlog, (output/'tracking.jsonl').open('w',encoding='utf-8') as tracklog:
            while True:
                now=time.perf_counter()
                if args.loop_seconds and total and now-started>=args.loop_seconds:break
                if args.max_frames and total>=args.max_frames:break
                work_start=time.perf_counter()
                ok,frame=cap.read()
                if not ok:
                    if frame_id!=info['frame_count']:raise ValueError(f'Video ended early: {frame_id}')
                    completed_cycles+=1
                    if not args.loop_seconds:break
                    cap.release();cap=cv2.VideoCapture(str(video))
                    cycle+=1;frame_id=0;trace.clear()
                    policy=TrackingPolicy(condition,cfg,info['fps'])
                    # Reopen and tracker reset overhead remains in overall wall timing.
                    continue
                if frame_id>=info['frame_count']:raise ValueError('Video has more frames than expected')
                read_done=time.perf_counter()
                detected=policy.needs_detection(frame_id)
                candidates=[];timing=dict(preprocess_ms=0.,inference_ms=0.,postprocess_ms=0.)
                if detected:
                    candidates,timing=detector.detect(frame);yolo_calls+=1
                track_start=time.perf_counter()
                pred=policy.update(frame,frame_id,candidates,detected)
                track_end=time.perf_counter()
                pred.update(cycle=cycle,global_frame_id=total,yolo_executed=detected)
                # Ground truth is used only for drawing/reporting AFTER target selection.
                if writer is not None:
                    draw(frame,pred,trace,gt[frame_id]);writer.write(frame)
                render_end=time.perf_counter()
                log=dict(frame_id=frame_id,cycle=cycle,global_frame_id=total,yolo_executed=detected,candidates=candidates)
                rawlog.write(json.dumps(log,allow_nan=False)+'\n');tracklog.write(json.dumps(pred,allow_nan=False)+'\n')
                if cycle==0:
                    first_predictions.append(pred);first_logs.append(log)
                if time.perf_counter()>=next_telemetry:
                    sample_start=time.perf_counter()
                    telemetry.append(dict(elapsed_s=sample_start-started,cycle=cycle,frame_id=frame_id,nvidia_smi=hardware_sample()))
                    next_telemetry=time.perf_counter()+cfg['telemetry_interval_s']
                work_done=time.perf_counter()
                total+=1;frame_id+=1
                sleep_s=0
                if args.realtime:
                    remaining=started+total/info['fps']-time.perf_counter()
                    if remaining>0:
                        sleep_start=time.perf_counter();time.sleep(remaining);sleep_s=time.perf_counter()-sleep_start
                all_timing.append(dict(global_frame_id=total-1,cycle=cycle,frame_id=frame_id-1,yolo_executed=detected,
                    decode_ms=(read_done-work_start)*1000,**timing,tracking_ms=(track_end-track_start)*1000,
                    render_save_ms=(render_end-track_end)*1000,log_telemetry_ms=(work_done-render_end)*1000,
                    processing_ms=(work_done-work_start)*1000,pacing_sleep_ms=sleep_s*1000,
                    schedule_lag_ms=max(0.,(time.perf_counter()-started-total/info['fps'])*1000) if args.realtime else None))
                if total%100==0:print(f'{size}/{condition}: {total} frames, cycle {cycle}, YOLO {yolo_calls}',flush=True)
        if writer is not None:writer.release();writer=None
        cap.release();cap=None
        elapsed=time.perf_counter()-started
        sleeps=sum(r['pacing_sleep_ms'] for r in all_timing)/1000
        complete_first=len(first_predictions)==info['frame_count']
        detection,curve=detection_metrics(gt,first_logs,cfg['confidence'])
        tracking,per_frame=tracking_metrics(gt,first_predictions,info['fps'])
        memory=dict(max_allocated_bytes=detector.torch.cuda.max_memory_allocated(detector.device),
                    max_reserved_bytes=detector.torch.cuda.max_memory_reserved(detector.device)) if detector.device.type=='cuda' else None
        performance=dict(processed_frames=total,yolo_calls=yolo_calls,warmup_yolo_calls=detector.warmup_calls,
            video_fps=info['fps'],scheduled_detection_hz=info['fps']/(cfg['detect_interval'] if condition in 'DE' else 1),
            realized_yolo_calls_per_video_second=yolo_calls/(total/info['fps']) if total else None,
            realized_yolo_calls_per_wall_second=yolo_calls/elapsed,wall_seconds=elapsed,pacing_sleep_seconds=sleeps,
            wall_fps=total/elapsed,processing_fps=total/(elapsed-sleeps),
            processing_ms=stats([r['processing_ms'] for r in all_timing]),
            yolo_stage_ms={k:stats([r[k] for r in all_timing if r['yolo_executed']]) for k in ('preprocess_ms','inference_ms','postprocess_ms')},
            tracking_ms_all_frames=stats([r['tracking_ms'] for r in all_timing]),
            render_save_ms=stats([r['render_save_ms'] for r in all_timing]),gpu_memory=memory,
            completed_loops=completed_cycles,first_pass_complete=complete_first,
            reached_requested_loop_seconds=elapsed>=args.loop_seconds if args.loop_seconds else None,
            note='processing_fps excludes deliberate pacing sleeps; wall_fps includes them. Both exclude setup/warmup and post-run evaluation/graphs. Telemetry overhead is included. No frames dropped.')
        summary=dict(condition=condition,model_size=size,detection=detection,tracking=tracking,performance=performance,
                     accuracy_scope='First pass only (partial)' if not complete_first else 'First complete pass only',
                     stability_scope='Repeated same video, not independent 15-minute footage' if args.loop_seconds else 'Single short video; no 15-minute stability conclusion',
                     complete_video=complete_first)
        save_json(output/'metrics.json',summary);save_json(output/'telemetry.json',telemetry)
        save_csv(output/'timing.csv',all_timing);save_csv(output/'per_frame.csv',per_frame);save_csv(output/'detector_pr.csv',curve)
        flat=[]
        for pred in first_predictions:
            flat.append({k:pred[k] for k in ('frame_id','timestamp_s','track_id','state','source','confidence','yolo_executed')} |
                        {k:pred['bbox'][i] if pred['bbox'] else None for i,k in enumerate(('x1','y1','x2','y2'))})
        save_csv(output/'tracking.csv',flat)
        save_csv(output/'summary.csv',[dict(model_size=size,condition=condition,complete_video=complete_first,
            evaluated_frames=tracking['evaluated_position_frames'],tsr03=tracking['tsr_iou_03'],tsr05=tracking['tsr_iou_05'],
            cle_mean_px=tracking['cle_px']['mean'],output_rate=tracking['output_rate_visible'],
            detector_precision=detection['precision'],detector_recall=detection['recall'],detector_f1=detection['f1'],ap50=detection['ap50'],
            yolo_calls=yolo_calls,wall_fps=performance['wall_fps'],processing_fps=performance['processing_fps'])])
        plots(output,per_frame,curve)
        save_failures(video,gt,first_predictions,per_frame,output,cfg['save_failures'],info['fps'])
        save_json(output/'status.json',dict(status='complete',complete_video=complete_first))
        print('Saved:',output)
        return output
    except Exception as e:
        save_json(output/'status.json',dict(status='failed',error=repr(e)))
        raise
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()
        del detector
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except ImportError:pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/experiment.json'))
    p.add_argument('--conditions',nargs='+',choices=list('ABCDE'),required=True)
    p.add_argument('--sizes',nargs='+',choices=['s','m'],required=True)
    p.add_argument('--output',default='results')
    p.add_argument('--max-frames',type=int,default=0,help='Partial smoke test only')
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--no-gt',action='store_true')
    p.add_argument('--loop-seconds',type=float,default=0)
    p.add_argument('--realtime',action='store_true',help='Pace by original FPS, without dropping frames')
    p.add_argument('--allow-version-mismatch',action='store_true',help='Explicitly label a non-pinned runtime smoke test')
    p.add_argument('--offline-guard',action='store_true',help='Reject Python internet socket connections during the run')
    a=p.parse_args()
    if a.max_frames<0 or a.loop_seconds<0 or (a.max_frames and a.loop_seconds):p.error('Invalid duration/frame limit')
    if a.loop_seconds and not a.realtime:p.error('Stability loops require --realtime; single-pass default is maximum-throughput processing')
    cfg=load_config(a.config)
    if a.offline_guard:
        import socket
        connect=socket.socket.connect
        def guarded_connect(sock,address):
            if sock.family in (socket.AF_INET,socket.AF_INET6):
                raise RuntimeError(f'Offline guard rejected a network connection: {address}')
            return connect(sock,address)
        socket.socket.connect=guarded_connect
    if a.no_video:cfg['save_video']=False
    # Fail all requested conditions upfront; never silently run A in place of B-E.
    for size in a.sizes:
        for condition in a.conditions:verify_weight(cfg,size,condition)
    for size in a.sizes:
        for condition in a.conditions:run_one(cfg,size,condition,a)


if __name__=='__main__':main()
