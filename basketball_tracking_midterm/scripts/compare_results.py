"""Build tables/plots from actual completed runs; never invent missing B-F results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import save_csv,save_json


def compare(runs,output):
    output.mkdir(parents=True,exist_ok=False)
    rows=[];signatures={};issues=[]
    for run in runs:
        status=json.loads((run/'status.json').read_text(encoding='utf-8'))
        if status['status']!='complete':raise ValueError(f'Run failed or unfinished: {run}')
        m=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
        c=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
        t,d,p=m['tracking'],m['detection'],m['performance']
        config=c['config']
        key=c['model_size']
        signature=(c['video']['sha256'],c['gt_sha256'],config['imgsz'],config['confidence'],config['candidate_conf'],
                   config['nms_iou'],config['half'],c['rendering'],c['realtime_pacing'],c['pinned_runtime'],
                   json.dumps(c['environment'],sort_keys=True))
        # Dynamic memory/temperature are not configuration differences.
        env=dict(c['environment']);env.pop('nvidia_smi',None)
        signature=signature[:-1]+(json.dumps(env,sort_keys=True),)
        if key in signatures and signatures[key]!=signature:issues.append(f'{key}: differing runtime/input/detection/rendering settings')
        signatures.setdefault(key,signature)
        if not m['complete_video']:issues.append(f'{run.name}: partial video, not a final comparison')
        if c['loop_seconds']:issues.append(f'{run.name}: repeated-video stability timing; do not mix with single-pass FPS')
        if not c['pinned_runtime']:issues.append(f'{run.name}: non-pinned smoke runtime')
        rows.append(dict(run=str(run.resolve()),size=key,condition=c['condition'],complete_video=m['complete_video'],
            gt_frames=t['evaluated_position_frames'],tsr03=t['tsr_iou_03'],tsr05=t['tsr_iou_05'],
            cle_mean_px=t['cle_px']['mean'],cle_n=t['cle_px']['n'],output_rate=t['output_rate_visible'],
            longest_failure_s=t['longest_failure_at_05_s'],precision=d['precision'],recall=d['recall'],f1=d['f1'],
            ap50=d['ap50'],detector_gt_boxes=d['gt_boxes'],yolo_calls=p['yolo_calls'],
            yolo_hz_video=p['realized_yolo_calls_per_video_second'],processing_fps=p['processing_fps'],wall_fps=p['wall_fps'],
            frame_ms_mean=p['processing_ms']['mean'],frame_ms_p95=p['processing_ms']['p95']))
    # B-F must share identical trained weights for each model size.
    for size in ('s','m'):
        hashes=set()
        for run in runs:
            c=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
            if c['model_size']==size and c['condition']!='A':hashes.add(c['weights']['sha256'])
        if len(hashes)>1:issues.append(f'{size}: B-F trained weights differ')
    save_csv(output/'comparison.csv',rows)
    save_json(output/'comparison.json',dict(rows=rows,comparability_issues=sorted(set(issues)),
        note='No missing condition is imputed. D/E/F detector PR is conditional on executed detection frames (adaptive for F); compare tracking TSR on common full-video visible GT.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=[r['size']+'/'+r['condition']+(' (partial)' if not r['complete_video'] else '') for r in rows]
    axes[0].bar(labels,[r['tsr05'] if r['tsr05'] is not None else float('nan') for r in rows])
    axes[0].set(ylabel='Tracking success IoU >= 0.5',ylim=(0,1))
    axes[1].bar(labels,[r['processing_fps'] for r in rows]);axes[1].set(ylabel='Processing FPS (excludes pacing)')
    fig.suptitle('Comparability issues: see comparison.json' if issues else 'Measured results')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print('Saved:',output)
    for issue in sorted(set(issues)):print('CHECK:',issue)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.runs,a.output)
