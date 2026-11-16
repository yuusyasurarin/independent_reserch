"""No-drop equivalence: TrackingWorker over experiment10 must reproduce the offline s/G bridge run.

The clock is frozen at each frame's nominal time, so the backlog rule never
skips a frame. With lookahead 0.4 s (bridge's own horizon) the released boxes
must equal the offline run that used a 2 s lookahead. Ground truth is used
only for the final score.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, resolve, save_json
from src.detector import Detector
from src.adaptive_tracking import AdaptiveBallPolicy
from src.lag_tracking import FixedLagBall
from src.realtime import TrackingWorker
from src.data import inspect_video, load_gt
from src.evaluation import tracking_metrics
from scripts.research_g import profile_config


def main(reference,output):
    import cv2
    cfg=profile_config('f');cfg['device']='cuda:0'
    video=resolve(cfg['video']);info=inspect_video(video);gt=load_gt(resolve(cfg['ground_truth']),info)
    detector=Detector(cfg,'s','G');detector.warmup(info['width'],info['height'])
    fps=info['fps']
    policy=AdaptiveBallPolicy(cfg,fps);smoother=FixedLagBall(fps,info['width'],info['height'],.4,'bridge')
    frozen={'now':0.}
    worker=TrackingWorker(policy,smoother,detector.detect,fps,1.,lambda:frozen['now'])
    cap=cv2.VideoCapture(str(video));outputs=[];started=time.perf_counter()
    try:
        for i in range(info['frame_count']):
            ok,frame=cap.read();assert ok
            frozen['now']=i/fps
            outputs+=worker.step(i,i/fps,frame,False)
            if (i+1)%500==0:print(i+1,'frames',flush=True)
        outputs+=worker.flush()
    finally:cap.release()
    metrics,_=tracking_metrics(gt,outputs,fps)
    result=dict(tsr05=metrics['tsr_iou_05'],tsr03=metrics['tsr_iou_03'],output_rate=metrics['output_rate_visible'],
        yolo_calls=worker.counters['yolo_calls'],dropped=worker.counters['dropped'],seconds=time.perf_counter()-started)
    if reference is not None and reference.is_file():
        offline=[json.loads(l) for l in reference.open(encoding='utf-8')]
        result.update(reference=str(reference),offline_frames=len(offline),
            identical_bbox=sum(a['bbox']==b['bbox'] for a,b in zip(outputs,offline)))
    print(json.dumps(result,indent=1))
    if output:save_json(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,
        default=ROOT/'results/comparison_fixed_lag_20261028/s_G_f_bridge_960_cuda0_20261027T165238603535Z/tracking.jsonl')
    p.add_argument('--output',type=Path)
    a=p.parse_args();main(a.reference,a.output)
