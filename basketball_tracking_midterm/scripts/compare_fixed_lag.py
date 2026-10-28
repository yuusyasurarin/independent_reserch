"""Run B/D baselines and corrected bidirectional-KCF E / bridge G sequentially.

All conditions use fresh inference and the same runner, exported detector
settings, image/log saving, and simulated 30 fps arrival. Never run in parallel.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, resolve, save_json

PLAN=(('B','forward',0.),('D','forward',0.),('E','bidir_kcf',2.),('G','bridge',2.))


def run(args):
    root=resolve(args.output)
    root.mkdir(parents=True,exist_ok=False)
    commands=[]
    for size in args.sizes:
        for condition,method,lag in PLAN:
            command=[sys.executable,str(ROOT/'scripts/run_fixed_lag.py'),
                '--size',size,'--condition',condition,'--method',method,'--lag',str(lag),
                '--profile','f','--imgsz','960','--device','cuda:0','--realtime',
                '--output',str(root),'--max-frames',str(args.max_frames)]
            commands.append(command)
    save_json(root/'comparison_plan.json',dict(commands=commands,sizes=args.sizes,
        conditions=[dict(condition=c,method=m,lag_s=l) for c,m,l in PLAN],
        primary_metric='TSR@0.5 on all visible frames, no temporal realignment',
        selection='Frozen before these runs; experiment10 development evaluation',
        kcf_fix='Prime first update on same clean detection frame, forward and reverse',
        reverse_kcf_gap_s=.4,reverse_lag_s=2.,repetitions=1,
        performance='Single sequential run per condition; identical recording and pacing'))
    for command in commands:
        print('Running:',subprocess.list2cmdline(command),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',nargs='+',choices=['s','m'],default=['s','m'])
    p.add_argument('--output',default='results/comparison_fixed_lag_20261028')
    p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args()
    if a.max_frames<0:p.error('max-frames must not be negative')
    run(a)
