"""G: compare bounded-lag algorithms with causal, budgeted System 1/2 inputs.

frontend and sweep are cached-detector development experiments, never inference
FPS benchmarks. Full fresh-inference measurements use run_fixed_lag.py.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load_config, resolve, save_json, save_csv, sha256
from src.adaptive_tracking import AdaptiveBallPolicy
from src.data import load_gt
from src.evaluation import tracking_metrics
from src.lag_tracking import FixedLagBall, METHODS


PROFILES={'f':(3,15),'budget10':(6,10),'budget5':(12,5)}


def profile_config(profile):
    cfg=copy.deepcopy(load_config(ROOT/'configs/experiment_f.json'))
    interval,hz=PROFILES[profile]
    cfg['adaptive'].update(max_interval=interval,max_yolo_hz=hz)
    return cfg


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(line) for line in f]


def write_jsonl(path,rows):
    with Path(path).open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')


def frontend(cache_run, profile, output):
    import cv2
    cv2.setNumThreads(0)
    cfg=profile_config(profile)
    manifest=json.loads((cache_run/'run_config.json').read_text(encoding='utf-8'))
    assert manifest['condition']=='B' and manifest['model_size']=='s'
    info=manifest['video']
    for key in ('imgsz','confidence','candidate_conf','nms_iou','half'):
        assert cfg[key]==manifest['config'][key]
    assert sha256(resolve(cfg['video']))==info['sha256']
    assert sha256(resolve(cfg['models']['s']['finetuned']))==manifest['weights']['sha256']
    cache=read_jsonl(cache_run/'candidates.jsonl')
    policy=AdaptiveBallPolicy(cfg,info['fps'])
    cap=cv2.VideoCapture(str(resolve(cfg['video'])))
    predictions=[];logs=[];start=time.perf_counter()
    try:
        for i in range(info['frame_count']):
            ok,frame=cap.read();assert ok
            detected=policy.needs_detection(i,frame)
            # Access only THIS frame's cached candidates, only after causal scheduling.
            candidates=cache[i]['candidates'] if detected else []
            assert cache[i]['frame_id']==i and cache[i]['yolo_executed']
            pred=policy.update(frame,i,candidates,detected)
            pred['yolo_executed']=detected
            predictions.append(pred)
            logs.append(dict(frame_id=i,yolo_executed=detected,candidates=candidates))
    finally:cap.release()
    elapsed=time.perf_counter()-start
    output.mkdir(parents=True,exist_ok=False)
    write_jsonl(output/'tracking.jsonl',predictions)
    write_jsonl(output/'candidates.jsonl',logs)
    gt=load_gt(resolve(cfg['ground_truth']),info)
    metrics,_=tracking_metrics(gt,predictions,info['fps'])
    save_json(output/'frontend.json',dict(profile=profile,config=cfg,video=info,
        cache_run=str(cache_run.resolve()),cache_sha256=sha256(cache_run/'candidates.jsonl'),
        weights=manifest['weights'],gt_sha256=sha256(resolve(cfg['ground_truth'])),
        mode='cached detector replay; no inference FPS claim',
        yolo_calls=sum(p['yolo_executed'] for p in predictions),replay_wall_s=elapsed,
        tracking=metrics,source_sha256={n:sha256(ROOT/n) for n in ('src/adaptive_tracking.py','scripts/research_g.py')}))
    print('Frontend:',profile,'TSR',metrics['tsr_iou_05'],'calls',sum(p['yolo_executed'] for p in predictions),flush=True)


def source_data(front):
    if (front/'frontend.json').is_file():
        manifest=json.loads((front/'frontend.json').read_text(encoding='utf-8'))
    else:
        manifest=json.loads((front/'run_config.json').read_text(encoding='utf-8'))
        assert manifest['condition']=='F' and manifest['model_size']=='s'
        manifest=dict(manifest,profile='f',yolo_calls=sum(p['yolo_executed'] for p in read_jsonl(front/'tracking.jsonl')))
    return manifest,read_jsonl(front/'tracking.jsonl'),read_jsonl(front/'candidates.jsonl')


def sweep(fronts,lags,methods,output):
    import cv2
    cv2.setNumThreads(0)
    output.mkdir(parents=True,exist_ok=False)
    all_rows=[]
    for front in fronts:
        manifest,predictions,logs=source_data(front)
        info=manifest['video'];cfg=manifest['config']
        gt=load_gt(resolve(cfg['ground_truth']),info)
        for lag in lags:
            for method in methods:
                smoother=FixedLagBall(info['fps'],info['width'],info['height'],lag,method)
                results=[];seconds=0.
                cap=cv2.VideoCapture(info['path']) if method=='bidir' else None
                for p,log in zip(predictions,logs):
                    frame=None
                    if cap is not None:
                        ok,frame=cap.read();assert ok
                    started=time.perf_counter()
                    out=smoother.push(p,log['candidates'],log['yolo_executed'],frame)
                    seconds+=time.perf_counter()-started
                    if out is not None:results.append(out)
                started=time.perf_counter()
                results.extend(smoother.flush())
                seconds+=time.perf_counter()-started
                if cap is not None:cap.release()
                assert len(results)==len(predictions)==info['frame_count']
                metrics,per_frame=tracking_metrics(gt,results,info['fps'])
                full=[p for p in results if p['full_lookahead']]
                full_metrics,_=tracking_metrics(gt,full,info['fps'])
                key=f"{manifest['profile']}_{method}_{lag:g}s"
                run=output/key;run.mkdir()
                save_json(run/'metrics.json',dict(tracking=metrics,full_lookahead_tracking=full_metrics,
                    algorithm_seconds=seconds,algorithm_ms_per_frame=seconds/len(results)*1000,
                    timing_scope='smoother only (includes local reverse LK for bidir); excludes YOLO, decoding and rendering',
                    peak_state_rows=smoother.peak_rows))
                save_json(run/'manifest.json',dict(method=method,lag_s=lag,profile=manifest['profile'],
                    source_frontend=str(front.resolve()),source_tracking_sha256=sha256(front/'tracking.jsonl'),
                    source_candidates_sha256=sha256(front/'candidates.jsonl'),
                    source_sha256=sha256(ROOT/'src/lag_tracking.py'),gt_sha256=sha256(resolve(cfg['ground_truth'])),
                    config=cfg,video=info,selection_scope='experiment10 development only',
                    yolo_calls=manifest['yolo_calls']))
                write_jsonl(run/'tracking.jsonl',results)
                save_csv(run/'per_frame.csv',per_frame)
                row=dict(profile=manifest['profile'],method=method,lag_s=lag,tsr05=metrics['tsr_iou_05'],
                    tsr03=metrics['tsr_iou_03'],cle_mean=metrics['cle_px']['mean'],cle_n=metrics['cle_px']['n'],
                    output_rate=metrics['output_rate_visible'],longest_failure_s=metrics['longest_failure_at_05_s'],
                    full_lookahead_tsr05=full_metrics['tsr_iou_05'],full_lookahead_n=full_metrics['evaluated_position_frames'],
                    yolo_calls=manifest['yolo_calls'],algorithm_ms=seconds/len(results)*1000,
                    run=str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run))
                all_rows.append(row)
                print(key,'TSR',round(row['tsr05'],4),'algorithm ms',round(row['algorithm_ms'],3),flush=True)
                save_csv(output/'comparison.csv',all_rows)
    save_json(output/'comparison.json',dict(rows=all_rows,
        note='Matched detector calls within each profile; cached development sweep, not end-to-end speed or independent test.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('frontend');p.add_argument('--cache-run',type=Path,required=True)
    p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('sweep');p.add_argument('--frontends',nargs='+',type=Path,required=True)
    p.add_argument('--lags',nargs='+',type=float,default=[2.]);p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='frontend':frontend(a.cache_run,a.profile,a.output)
    else:sweep(a.frontends,a.lags,a.methods,a.output)
