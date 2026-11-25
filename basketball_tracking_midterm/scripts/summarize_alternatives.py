"""Comparison table for the 2026-11-25 alternatives runs (reads saved run folders only).

  python scripts/summarize_alternatives.py results/alt_20261125 [--json out.json] [--csv out.csv]
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_yolo_budget import analyze
from scripts.analyze_wrong_lock import analyze as wrong_lock


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def ms(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values),
                p95=values[min(len(values)-1, int(.95*len(values)))], mean=statistics.fmean(values))


def summarize(run):
    run = Path(run)
    metrics = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    config = json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    fps = config['video']['fps']
    budget = analyze(jsonl(run/'online.jsonl'), fps)
    roi_flag = {r['frame_id']: bool(r.get('roi')) for r in jsonl(run/'candidates.jsonl') if r['yolo_executed']}
    timing = list(csv.DictReader((run/'timing.csv').open(encoding='utf-8-sig')))
    per_call = {'full': [], 'roi': []}
    for r in timing:
        if r['yolo_executed'] == 'True':
            total = sum(float(r[k]) for k in ('preprocess_ms', 'inference_ms', 'postprocess_ms'))
            per_call['roi' if roi_flag[int(r['frame_id'])] else 'full'].append(
                dict(total=total, inference=float(r['inference_ms'])))
    processing = [float(r['processing_ms']) for r in timing]
    perf, track = metrics['performance'], metrics['tracking']
    return dict(run=run.name, command=' '.join(config['command'][1:]),
                yolo_calls_per_s=perf['realized_yolo_calls_per_video_second'],
                roi_calls=perf['roi_calls'], relative_inference_pixels=perf['relative_inference_pixels'],
                budget_refused_frames=budget['deferred_frames'], budget_refused_rate=budget['deferred_rate'],
                window_full_frames=budget['frames_with_full_window'],
                tsr05_bridge=track['tsr_iou_05'], tsr05_online=metrics['online_tracking']['tsr_iou_05'],
                tsr03_bridge=track['tsr_iou_03'], longest_failure_s=track['longest_failure_at_05_s'],
                call_ms_full=ms([c['total'] for c in per_call['full']]),
                call_ms_roi=ms([c['total'] for c in per_call['roi']]),
                infer_ms_full=ms([c['inference'] for c in per_call['full']]),
                infer_ms_roi=ms([c['inference'] for c in per_call['roi']]),
                processing_fps=perf['processing_fps'], frame_ms=ms(processing),
                frames_over_33ms=sum(v > 1000/fps for v in processing), frames=len(processing),
                wrong_lock=wrong_lock(run, fps), source_sha256=config['source_sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    p.add_argument('--json')
    a = p.parse_args()
    rows = [summarize(r) for r in sorted(Path(a.folder).glob('s_G*')) if (r/'metrics.json').is_file()]
    for r in rows:
        f, q = r['call_ms_full'], r['call_ms_roi']
        print(f"{r['run'][:60]:60s} calls/s {r['yolo_calls_per_s']:5.1f} refused {r['budget_refused_frames']:4d} "
              f"TSR {r['tsr05_bridge']:.3f}/{r['tsr05_online']:.3f} full {f['median']:.1f}ms "
              f"roi {q['median'] if q else float('nan'):.1f}ms fps {r['processing_fps']:.1f} >33ms {r['frames_over_33ms']} "
              f"wrong {r['wrong_lock']['wrong_object_frames']} longest {r['wrong_lock']['longest_episode_s']:.2f}s")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
