"""Run the shot judgement (src/shot.py) on a saved trajectory and list the events.

Inputs (one of):
  frames.jsonl   from scripts/realtime_tracking.py --record  (delayed, bridge output)
  tracking.csv   from scripts/run_fixed_lag.py
  *_gt.csv       ground-truth boxes (status=visible rows only)

  python scripts/judge_shots.py results/.../frames.jsonl --rim 728,128,782,144 [--json out.json]

The events list (frame range, result, reason) is what you check against the video by eye.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.shot import ShotJudge


def centres(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        for line in path.open(encoding='utf-8'):
            r = json.loads(line); b = r['bbox']
            yield r['frame_id'], ((b[0]+b[2])/2, (b[1]+b[3])/2) if b else None
        return
    for r in csv.DictReader(path.open(encoding='utf-8-sig')):
        visible = r.get('status', 'visible') == 'visible' and r.get('x1')
        box = [float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')] if visible else None
        yield int(r['frame_id']), ((box[0]+box[2])/2, (box[1]+box[3])/2) if box else None


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('trajectory')
    p.add_argument('--rim', required=True, help='x1,y1,x2,y2 around the ring opening')
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--json')
    a = p.parse_args()
    judge = ShotJudge([float(v) for v in a.rim.split(',')], a.fps)
    for frame_id, centre in centres(a.trajectory):
        if (e := judge.update(frame_id, centre)):
            print(f"{e['result']:4s} {e['reason']:20s} frames {e['start_frame']}-{e['frame_id']} "
                  f"({e['start_frame']/a.fps:.1f}-{e['time_s']:.1f}s) crossed={e['crossed_frame']} "
                  f"closest {e['min_rim_distance_px']:.1f}px")
    print(judge.score())
    if a.json:
        Path(a.json).write_text(json.dumps(dict(trajectory=Path(a.trajectory).name, rim=judge.rim, options=judge.o,
                                                score=judge.score(), events=judge.events), indent=2), encoding='utf-8')
