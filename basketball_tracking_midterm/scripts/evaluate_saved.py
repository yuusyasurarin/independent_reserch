"""Evaluate saved inference after human GT becomes available, without running YOLO again."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import save_json,save_csv,sha256
from src.data import load_gt
from src.evaluation import detection_metrics,tracking_metrics
from src.reporting import plots


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    config=json.loads((a.run/'run_config.json').read_text(encoding='utf-8'))
    def first_cycle(name):
        with (a.run/name).open(encoding='utf-8') as f:return [r for line in f if (r:=json.loads(line))['cycle']==0]
    predictions=first_cycle('tracking.jsonl');logs=first_cycle('candidates.jsonl')
    if [p['frame_id'] for p in predictions]!=list(range(len(predictions))):raise ValueError('Missing or duplicate prediction rows')
    if [r['frame_id'] for r in logs]!=[p['frame_id'] for p in predictions]:raise ValueError('Candidate/prediction alignment mismatch')
    gt=load_gt(a.gt,config['video'])
    detection,curve=detection_metrics(gt,logs,config['config']['confidence'])
    tracking,rows=tracking_metrics(gt,predictions,config['video']['fps'])
    a.output.mkdir(parents=True,exist_ok=False)
    save_json(a.output/'metrics.json',dict(detection=detection,tracking=tracking,gt_sha256=sha256(a.gt),
        source_run=str(a.run.resolve()),complete_video=len(predictions)==len(gt)))
    save_csv(a.output/'per_frame.csv',rows);save_csv(a.output/'detector_pr.csv',curve);plots(a.output,rows,curve)
    print(a.output)
