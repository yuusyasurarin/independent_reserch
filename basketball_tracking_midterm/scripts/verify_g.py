"""Verify G time boundaries, metrics, full videos and causal replay without GT."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, resolve, sha256, save_json, valid_box
from src.adaptive_tracking import AdaptiveBallPolicy
from src.lag_tracking import FixedLagBall
from src.data import load_gt
from src.evaluation import tracking_metrics, detection_metrics
from scripts.research_g import read_jsonl


def basic(predictions,info,lag):
    n=info['frame_count'];steps=round(lag*info['fps'])
    assert len(predictions)==n
    for i,p in enumerate(predictions):
        assert p['frame_id']==i and abs(p['timestamp_s']-i/info['fps'])<1e-9
        assert p['available_through_frame']==min(i+steps,n-1)
        assert p['lookahead_frames']==min(steps,n-1-i)
        assert p['full_lookahead']==(p['lookahead_frames']==steps)
        assert p['bbox'] is None or valid_box(p['bbox'],info['width'],info['height'])
        if p['source'].startswith('G_'):assert p['confidence'] is None


def verify_sweep(root):
    root=root.resolve()
    reports=[]
    for path in sorted(root.glob('*/manifest.json')):
        run=path.parent;m=json.loads(path.read_text(encoding='utf-8'))
        metrics=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
        pred=read_jsonl(run/'tracking.jsonl');info=m['video']
        basic(pred,info,m['lag_s'])
        gt_path=resolve(m['config']['ground_truth'])
        assert sha256(gt_path)==m['gt_sha256']
        gt=load_gt(gt_path,info)
        assert tracking_metrics(gt,pred,info['fps'])[0]==metrics['tracking']
        assert tracking_metrics(gt,[p for p in pred if p['full_lookahead']],info['fps'])[0]==metrics['full_lookahead_tracking']
        source=Path(m['source_frontend'])
        assert sha256(source/'tracking.jsonl')==m['source_tracking_sha256']
        assert sha256(source/'candidates.jsonl')==m['source_candidates_sha256']
        logs=read_jsonl(source/'candidates.jsonl')
        assert all(p['yolo_executed']==l['yolo_executed'] for p,l in zip(pred,logs))
        assert sum(l['yolo_executed'] for l in logs)==m['yolo_calls']
        if m['method']=='forward':
            assert [p['bbox'] for p in pred]==[p['bbox'] for p in read_jsonl(source/'tracking.jsonl')]
        reports.append(dict(run=str(run.relative_to(ROOT)),status='passed',frames=len(pred),
                            temporal_boundaries_valid=True,reevaluated_metrics_identical=True))
    return reports


def verify_fresh(run):
    run=run.resolve()
    import cv2
    cv2.setNumThreads(0)
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    m,metrics,status=read('run_config.json'),read('metrics.json'),read('status.json')
    assert m['condition']=='G' and status==dict(status='complete',complete_video=True)
    cfg,info=m['config'],m['video']
    pred=read_jsonl(run/'tracking.jsonl');online=read_jsonl(run/'online.jsonl');logs=read_jsonl(run/'candidates.jsonl')
    n=info['frame_count'];assert len(online)==len(logs)==n
    basic(pred,info,m['lag_s'])
    assert sha256(Path(info['path']))==info['sha256']
    assert sha256(Path(m['weights_path']))==m['weights']['sha256']
    gt_path=Path(m['ground_truth']);assert sha256(gt_path)==m['gt_sha256']
    gt=load_gt(gt_path,info)
    assert tracking_metrics(gt,pred,info['fps'])[0]==metrics['tracking']
    assert tracking_metrics(gt,online,info['fps'])[0]==metrics['online_tracking']
    assert tracking_metrics(gt,[p for p in pred if p['full_lookahead']],info['fps'])[0]==metrics['full_lookahead_tracking']
    assert detection_metrics(gt,logs,cfg['confidence'])[0]==metrics['detection']
    policy=AdaptiveBallPolicy(cfg,info['fps'])
    smoother=FixedLagBall(info['fps'],info['width'],info['height'],m['lag_s'],m['method'])
    cap=cv2.VideoCapture(info['path']);assert cap.isOpened()
    replay=[]
    try:
        for i,log in enumerate(logs):
            ok,frame=cap.read();assert ok
            assert log['frame_id']==log['global_frame_id']==i and log['cycle']==0
            detected=policy.needs_detection(i,frame)
            assert detected==log['yolo_executed']
            if not detected:assert log['candidates']==[]
            p=policy.update(frame,i,log['candidates'],detected)
            for key in ('bbox','source','state','events','track_id','confidence'):
                assert p[key]==online[i][key],f'Frontend {i}/{key}'
            recent=logs[max(0,i-policy.window+1):i+1]
            assert sum(x['yolo_executed'] for x in recent)<=policy.budget
            out=smoother.push(p,log['candidates'],detected,frame)
            if out is not None:replay.append(out)
    finally:cap.release()
    replay.extend(smoother.flush())
    for a,b in zip(pred,replay):
        for key in b:assert a[key]==b[key],f"Smoother {a['frame_id']}/{key}"
    assert metrics['performance']['yolo_calls']==sum(l['yolo_executed'] for l in logs)
    assert metrics['performance']['processed_frames']==n
    assert metrics['performance']['peak_buffer_frames']<=round(m['lag_s']*info['fps'])+1
    assert metrics['performance']['peak_state_rows']<=smoother.lag+smoother.history+1
    decoded=0
    if m['rendering']:
        cap=cv2.VideoCapture(str(run/'tracked.mp4'));assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FPS)==info['fps']
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                assert frame.shape[:2]==(info['height'],info['width']);decoded+=1
        finally:cap.release()
        assert decoded==n
    for name in ('timing.csv','release.csv','per_frame.csv','tracking.csv'):
        with (run/name).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==n
        if name=='release.csv' and m['realtime_pacing']:
            assert all(float(r['simulated_arrival_to_release_s'])>=m['lag_s']-1e-6 for r in rows)
    report=dict(run=str(run.relative_to(ROOT)),status='passed',frames=n,decoded_video_frames=decoded,
        causal_frontend_replay_identical=True,fixed_lag_replay_identical=True,rolling_yolo_budget_valid=True,
        temporal_boundaries_valid=True,reevaluated_metrics_identical=True,
        yolo_calls=metrics['performance']['yolo_calls'],metrics_sha256=sha256(run/'metrics.json'),
        run_config_sha256=sha256(run/'run_config.json'))
    save_json(run/'verification.json',report)
    print('Verified fresh:',run.name,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweeps',nargs='*',type=Path,default=[])
    p.add_argument('--fresh-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reports=[]
    for root in a.sweeps:
        verified=verify_sweep(root);reports.extend(verified)
        print('Verified cached trials:',root,len(verified),flush=True)
    if a.fresh_root:
        for run in sorted(a.fresh_root.iterdir()):
            if run.is_dir():reports.append(verify_fresh(run))
    save_json(a.output,dict(status='passed',runs=reports))
