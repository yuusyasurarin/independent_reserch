"""Score a realtime_tracking.py recording of experiment10 against the ground truth.

Positions come from frames.jsonl (what was actually released for display at the
fixed delay, including skipped frames filled by bridge). Ground truth is used
only here, after the run. Frames the display stage discarded count as missing.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load_config, resolve, save_json, sha256
from src.data import inspect_video, load_gt
from src.evaluation import tracking_metrics


def evaluate(run,config=None):
    manifest=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    source=manifest['source']
    if source['kind']!='video':raise ValueError('Only paced video runs of the annotated video can be scored')
    cfg=load_config(config or ROOT/'configs/experiment_f.json')
    video=resolve(cfg['video']);info=inspect_video(video)
    if Path(source['path']).resolve()!=video:raise ValueError(f'Run used {source["path"]}, not the annotated {video}')
    gt_path=resolve(cfg['ground_truth']);gt=load_gt(gt_path,info)
    rows={}
    with (run/'frames.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);rows[r['frame_id']]=r
    predictions=[]
    for i in range(info['frame_count']):
        r=rows.get(i)
        predictions.append(dict(frame_id=i,timestamp_s=i/info['fps'],bbox=r['bbox'] if r else None,
            state=('missing' if r is None else 'no_result' if not r['result_ready'] else 'released'),
            source=(r['source'] if r and r['source'] else 'none'),track_id=None))
    metrics,per_frame=tracking_metrics(gt,predictions,info['fps'])
    session=json.loads((run/'session.json').read_text(encoding='utf-8'))
    result=dict(run=str(run),video_sha256=info['sha256'],gt_sha256=sha256(gt_path),tracking=metrics,
        released_frames=len(rows),captured=session['captured'],
        dropped_by_backlog_control=session['tracking']['dropped'],drop_rate=session['tracking']['drop_rate'],
        yolo_calls=session['tracking']['yolo_calls'],missing_results=session['missing_results'],
        display_delay_s=session['display_delay_s'],result_margin_s=session['result_margin_s'],
        scope='Paced-file development check on experiment10; not an independent test and not a camera latency measurement')
    save_json(run/'evaluation.json',result)
    print(json.dumps(dict(tsr05=metrics['tsr_iou_05'],tsr03=metrics['tsr_iou_03'],output_rate=metrics['output_rate_visible'],
        drop_rate=result['drop_rate'],yolo_calls=result['yolo_calls'],delay_median=session['display_delay_s']['median'],
        delay_p95=session['display_delay_s']['p95']),indent=1))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path,help='results/.../realtime_<timestamp> folder with run_config.json, frames.jsonl, session.json')
    p.add_argument('--config',type=Path,help='experiment config whose video/ground_truth the run used; default configs/experiment_f.json')
    a=p.parse_args();evaluate(a.run.resolve(),a.config)
