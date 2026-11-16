"""Convert a CVAT "YOLO 1.1" export of a whole-video task into the tracking ground-truth CSV.

Frames without a label file become `unknown_position` only when --all-frames-reviewed
is given (every frame was looked at and the ball was not visible); otherwise they
stay `unannotated` and are excluded from evaluation. The metadata JSON binds the
CSV to the video and the archive so load_gt() refuses mismatched files.
"""
import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import save_csv, save_json, sha256, valid_box
from src.data import inspect_video


def main(args):
    info=inspect_video(args.video)
    labels={}
    with zipfile.ZipFile(args.annotations) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP entries')
        for name in names:
            m=re.search(r'(?:^|/)frame_(\d+)\.txt$',name)
            if not m or not name.startswith('labels/'):continue
            i=int(m.group(1))
            if i not in range(info['frame_count']) or i in labels:raise ValueError(f'Out-of-range or duplicate frame label: {name}')
            lines=[s for s in z.read(name).decode('utf-8-sig').splitlines() if s.strip()]
            if len(lines)>1:raise ValueError(f'More than one ball box in {name}; single-ball GT expected')
            box=None
            if lines:
                values=list(map(float,lines[0].split()))
                if len(values)!=5 or int(values[0])!=args.class_id:raise ValueError(f'Bad label line in {name}: {lines[0]}')
                _,x,y,w,h=values
                box=[(x-w/2)*info['width'],(y-h/2)*info['height'],(x+w/2)*info['width'],(y+h/2)*info['height']]
                if not valid_box(box,info['width'],info['height']):raise ValueError(f'Box outside frame in {name}')
            labels[i]=box
    if not labels:raise ValueError('No labels/frame_*.txt entries found in the archive')
    rows=[]
    for i in range(info['frame_count']):
        box=labels.get(i)
        status='visible' if box else ('unknown_position' if args.all_frames_reviewed else 'unannotated')
        rows.append(dict(frame_id=i,timestamp_s=i/info['fps'],status=status,
                         x1=box[0] if box else '',y1=box[1] if box else '',x2=box[2] if box else '',y2=box[3] if box else '',
                         segment_id='',note='' if box else ('Reviewed, ball not visible' if args.all_frames_reviewed else 'Not annotated')))
    save_csv(args.output,rows)
    save_json(args.output.with_suffix('.metadata.json'),dict(video_sha256=info['sha256'],annotation_sha256=sha256(args.annotations),
        gt_sha256=sha256(args.output),video=info,annotations=str(args.annotations),counts=dict(Counter(r['status'] for r in rows)),
        all_frames_reviewed=args.all_frames_reviewed))
    print(dict(output=str(args.output),frames=info['frame_count'],**Counter(r['status'] for r in rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--annotations',type=Path,required=True,help='CVAT YOLO 1.1 export ZIP (labels/.../frame_000123.txt)')
    p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True,help='e.g. data/fisheye_game01_cam1_gt.csv')
    p.add_argument('--class-id',type=int,default=0)
    p.add_argument('--all-frames-reviewed',action='store_true',help='unlabeled frames were checked and the ball is not visible')
    main(p.parse_args())
