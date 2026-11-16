"""Triangulate the ball in court coordinates from two cameras' 2D tracking logs.

Accepts frames.jsonl (realtime_tracking.py --record), tracking.jsonl or
tracking.csv (run_fixed_lag.py / run_experiment.py). Camera B time is
camera A time + --offset-s; measure the offset from a clap or flash visible
in both recordings (dual_camera_recorder.py logs marker times to help).
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import save_csv, save_json, stats
from src.stereo import load_camera, triangulate_tracks


def load_track(path,fps):
    path=Path(path);rows=[]
    if path.suffix=='.jsonl':
        for line in path.open(encoding='utf-8'):
            r=json.loads(line)
            t=r.get('capture_s',r.get('timestamp_s'));t=r['frame_id']/fps if t is None else t
            b=r.get('bbox');rows.append((float(t),[(b[0]+b[2])/2,(b[1]+b[3])/2] if b else None))
    elif path.suffix=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                t=float(r['timestamp_s']) if r.get('timestamp_s') else int(r['frame_id'])/fps
                b=[r.get(k,'') for k in ('x1','y1','x2','y2')]
                rows.append((t,[(float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2] if all(v.strip() for v in b) else None))
    else:raise ValueError('Track must be .jsonl or .csv')
    return sorted(rows,key=lambda r:r[0])


def main(args):
    cam_a=load_camera(args.camera_a,args.pose_a);cam_b=load_camera(args.camera_b,args.pose_b)
    track_a=load_track(args.track_a,args.fps);track_b=load_track(args.track_b,args.fps)
    rows=triangulate_tracks(track_a,track_b,cam_a,cam_b,args.offset_s,args.max_gap_s)
    kept=[r for r in rows if max(r['residual_a_px'],r['residual_b_px'])<=args.max_residual_px and r['ray_angle_deg']>=args.min_angle_deg]
    save_csv(args.output,kept)
    summary=dict(samples_a=sum(p is not None for _,p in track_a),samples_b=sum(p is not None for _,p in track_b),
                 triangulated=len(rows),kept=len(kept),offset_s=args.offset_s,
                 residual_a_px=stats([r['residual_a_px'] for r in rows]),residual_b_px=stats([r['residual_b_px'] for r in rows]),
                 ray_angle_deg=stats([r['ray_angle_deg'] for r in rows]),
                 height_m=stats([r['z'] for r in kept]),cameras=dict(a=cam_a['pose'],b=cam_b['pose']),
                 note='Residuals are reprojection distances of the triangulated point; a large median usually means a wrong time offset or pose.')
    save_json(args.output.with_suffix('.summary.json'),summary)
    print(json.dumps({k:summary[k] for k in ('samples_a','samples_b','triangulated','kept','residual_a_px','residual_b_px','ray_angle_deg')},indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for side in 'ab':
        p.add_argument(f'--track-{side}',type=Path,required=True);p.add_argument(f'--camera-{side}',type=Path,required=True)
        p.add_argument(f'--pose-{side}',type=Path,required=True)
    p.add_argument('--fps',type=float,default=30.,help='used when a track has frame ids but no timestamps')
    p.add_argument('--offset-s',type=float,default=0.,help='camera B time minus camera A time for the same instant')
    p.add_argument('--max-gap-s',type=float,default=.2,help='longest gap in camera B bridged by interpolation')
    p.add_argument('--max-residual-px',type=float,default=15.)
    p.add_argument('--min-angle-deg',type=float,default=5.,help='reject near-parallel rays')
    p.add_argument('--output',type=Path,required=True,help='CSV with time_s, x, y, z, residuals, ray angle')
    a=p.parse_args()
    if a.fps<=0 or a.max_gap_s<=0 or a.max_residual_px<=0:p.error('Invalid settings')
    main(a)
