"""Ball-tracking metric suite on experiment10 (one table row per run).

Reads either a run_fixed_lag.py folder (tracking.csv, offline file benchmark) or a
realtime_tracking.py folder (frames.jsonl, what was released for display) and scores it
against data/experiment10_gt.csv. Ground truth is used only here.

Metrics (denominator = frames with a visible GT ball unless noted)
  tsr05 / tsr03       share of frames whose output box overlaps GT with IoU >= 0.5 / 0.3
                      (no output counts as failure)
  mean_iou            mean IoU (no output = 0)
  cle_median/p95_px   centre distance in pixels, frames with an output
  output_rate         share of frames with any output
  wrong_object_rate   share of frames whose output does not overlap GT at all (IoU = 0)
  longest_wrong_s     longest run of such frames (time to come back to the ball)
  rim_tsr05           tsr05 restricted to frames whose GT centre is near the rim
                      (within --rim-zone px of the rim box); this is what shot judgement needs
  drop_rate, measured_fps, delay_p95_s   real-time runs only (from session.json)

  python scripts/evaluate_tracking_suite.py RUN [RUN ...] --rim 510,172,545,182 --json out.json
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, center, iou, load_config, resolve
from src.data import inspect_video, load_gt


def outputs(run):
    run = Path(run)
    if (run/'frames.jsonl').is_file():
        rows = [json.loads(line) for line in (run/'frames.jsonl').open(encoding='utf-8')]
        return {r['frame_id']: r['bbox'] for r in rows}, 'realtime'
    rows = list(csv.DictReader((run/'tracking.csv').open(encoding='utf-8-sig')))
    return {int(r['frame_id']): [float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')] if r['x1'] else None
            for r in rows}, 'offline'


def near_rim(box, rim, zone):
    x, y = center(box)
    dx = max(rim[0]-x, 0, x-rim[2])
    dy = max(rim[1]-y, 0, y-rim[3])
    return math.hypot(dx, dy) <= zone


def evaluate(run, gt, rim, zone, fps):
    pred, kind = outputs(run)
    ious, errors, wrong_runs, run_len, rim_hits = [], [], [], 0, []
    for truth in gt:
        if truth['status'] != 'visible':
            if run_len: wrong_runs.append(run_len); run_len = 0
            continue
        box = pred.get(truth['frame_id'])
        value = iou(truth['bbox'], box) if box else 0.
        ious.append(value)
        if box:
            errors.append(math.dist(center(truth['bbox']), center(box)))
        wrong = box is not None and value == 0
        if wrong:
            run_len += 1
        elif run_len:
            wrong_runs.append(run_len); run_len = 0
        if near_rim(truth['bbox'], rim, zone):
            rim_hits.append(value >= .5)
    if run_len: wrong_runs.append(run_len)
    n = len(ious)
    errors.sort()
    result = dict(run=Path(run).name, kind=kind, visible_frames=n,
                  tsr05=sum(v >= .5 for v in ious)/n, tsr03=sum(v >= .3 for v in ious)/n,
                  mean_iou=statistics.fmean(ious), cle_median_px=statistics.median(errors),
                  cle_p95_px=errors[min(len(errors)-1, math.ceil(.95*len(errors))-1)],
                  output_rate=len(errors)/n, wrong_object_rate=sum(wrong_runs)/n,
                  longest_wrong_s=max(wrong_runs, default=0)/fps,
                  rim_frames=len(rim_hits), rim_tsr05=sum(rim_hits)/len(rim_hits) if rim_hits else None)
    session = Path(run)/'session.json'
    if session.is_file():
        s = json.loads(session.read_text(encoding='utf-8'))
        t = s['tracking']
        result.update(drop_rate=t['dropped']/t['received'], measured_fps=t['processed']*fps/t['received'],
                      delay_p95_s=s['display_delay_s']['p95'], power=s.get('power', {}).get('start', {}).get('source'))
    metrics = Path(run)/'metrics.json'
    if metrics.is_file():
        m = json.loads(metrics.read_text(encoding='utf-8'))
        result.update(processing_fps=m['performance']['processing_fps'])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--rim', default='510,172,545,182', help='experiment10 rim box')
    p.add_argument('--rim-zone', type=float, default=80.)
    p.add_argument('--json')
    a = p.parse_args()
    cfg = load_config(ROOT/'configs/experiment_f.json')
    info = inspect_video(resolve(cfg['video']))
    gt = load_gt(resolve(cfg['ground_truth']), info)
    rim = [float(v) for v in a.rim.split(',')]
    rows = [evaluate(r, gt, rim, a.rim_zone, info['fps']) for r in a.runs]
    for r in rows:
        print(f"{r['run'][:58]:58s} TSR.5 {r['tsr05']:.1%} TSR.3 {r['tsr03']:.1%} IoU {r['mean_iou']:.3f} "
              f"CLE {r['cle_median_px']:.1f}/{r['cle_p95_px']:.0f}px out {r['output_rate']:.1%} "
              f"wrong {r['wrong_object_rate']:.1%} ({r['longest_wrong_s']:.2f}s) rim {r['rim_tsr05']:.1%} n={r['rim_frames']}"
              + (f" drop {r['drop_rate']:.1%}" if 'drop_rate' in r else ''))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
