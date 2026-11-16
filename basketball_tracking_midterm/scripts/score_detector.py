"""Goal events from a triangulated 3D ball trajectory, with optional scoring against manual truth.

A goal is a downward crossing of the rim plane inside the rim radius; a crossing
just outside is reported as rim_miss. Truth CSV columns: time_s, event (goal/miss).
Tune --rim-radius / --near-m only on development games, then freeze them.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import save_csv, save_json
from src.stereo import detect_goals, match_events


def main(args):
    court=json.loads(args.court.read_text(encoding='utf-8'));hoop=court['hoops'][args.hoop]
    with args.trajectory.open(encoding='utf-8-sig',newline='') as f:
        rows=[dict(time_s=float(r['time_s']),x=float(r['x']),y=float(r['y']),z=float(r['z'])) for r in csv.DictReader(f)]
    rows.sort(key=lambda r:r['time_s'])
    radius=args.rim_radius if args.rim_radius else hoop['rim_radius']
    events=detect_goals(rows,hoop['center'],radius,args.near_m,args.max_gap_s,args.min_interval_s)
    save_csv(args.output,events)
    result=dict(hoop=args.hoop,hoop_center=hoop['center'],rim_radius=radius,near_m=args.near_m,samples=len(rows),
                goals=sum(e['event']=='goal' for e in events),rim_misses=sum(e['event']=='rim_miss' for e in events))
    if args.truth:
        with args.truth.open(encoding='utf-8-sig',newline='') as f:
            truth=[float(r['time_s']) for r in csv.DictReader(f) if r['event'].strip().lower()=='goal']
        result['evaluation']=match_events([e for e in events if e['event']=='goal'],truth,args.tolerance_s)
        result['evaluation']['tolerance_s']=args.tolerance_s
    save_json(args.output.with_suffix('.summary.json'),result)
    print(json.dumps(result,indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trajectory',type=Path,required=True,help='CSV from triangulate_ball.py')
    p.add_argument('--court',type=Path,required=True);p.add_argument('--hoop',choices=['A','B'],required=True)
    p.add_argument('--rim-radius',type=float,default=0.,help='override the court file (metres); 0 = use court file')
    p.add_argument('--near-m',type=float,default=.3);p.add_argument('--max-gap-s',type=float,default=.2)
    p.add_argument('--min-interval-s',type=float,default=1.)
    p.add_argument('--truth',type=Path,help='manual events CSV: time_s,event');p.add_argument('--tolerance-s',type=float,default=1.)
    p.add_argument('--output',type=Path,required=True,help='events CSV')
    main(p.parse_args())
