"""Check complete single-pass logs, schedules, saved videos and recomputed metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, save_json, sha256, valid_box
from src.data import load_gt
from src.evaluation import detection_metrics, tracking_metrics


def verify_run(run):
    read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
    manifest, metrics, status = read('run_config.json'), read('metrics.json'), read('status.json')
    if status != dict(status='complete', complete_video=True) or manifest['loop_seconds']:
        raise ValueError(f'Expected a completed single-pass full video: {run}')
    cfg, video = manifest['config'], manifest['video']
    n, fps, condition = video['frame_count'], video['fps'], manifest['condition']
    with (run/'tracking.jsonl').open(encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f]
    with (run/'candidates.jsonl').open(encoding='utf-8') as f:
        logs = [json.loads(line) for line in f]
    assert len(predictions) == len(logs) == n
    scheduled_count = 0
    for i, (p, log) in enumerate(zip(predictions, logs)):
        assert p['frame_id'] == log['frame_id'] == p['global_frame_id'] == log['global_frame_id'] == i
        assert p['cycle'] == log['cycle'] == 0
        assert abs(p['timestamp_s'] - i/fps) < 1e-9
        scheduled = condition in 'ABC' or i % cfg['detect_interval'] == 0
        assert p['yolo_executed'] == log['yolo_executed'] == scheduled
        scheduled_count += scheduled
        if not scheduled:
            assert log['candidates'] == []
        if p['bbox'] is not None:
            assert valid_box(p['bbox'], video['width'], video['height'])
            if condition in 'DE':
                assert p['age_since_yolo_s'] <= cfg['max_age_s'] + 1e-9
        if condition in 'AB':
            choice = max((c for c in log['candidates'] if c['confidence'] >= cfg['confidence']),
                         key=lambda c: c['confidence'], default=None)
            assert p['bbox'] == (choice['bbox'] if choice else None)
            assert p['confidence'] == (choice['confidence'] if choice else None)
            assert p['track_id'] is None
        if p['source'] in ('hold', 'KCF'):
            assert not scheduled and p['confidence'] is None
        if p['source'] == 'hold':
            assert condition == 'D' and i > 0 and p['bbox'] == predictions[i-1]['bbox']
        if p['source'] == 'KCF':
            assert condition == 'E' and p['kcf_success_flag'] is True
    assert metrics['performance']['processed_frames'] == n
    assert metrics['performance']['yolo_calls'] == scheduled_count
    for name in ('tracking.csv', 'timing.csv', 'per_frame.csv'):
        with (run/name).open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n
    gt_path = Path(manifest['ground_truth'])
    assert sha256(gt_path) == manifest['gt_sha256']
    assert sha256(Path(manifest['weights_path'])) == manifest['weights']['sha256']
    gt = load_gt(gt_path, video)
    detection, _ = detection_metrics(gt, logs, cfg['confidence'])
    tracking, _ = tracking_metrics(gt, predictions, fps)
    assert detection == metrics['detection'] and tracking == metrics['tracking']
    decoded = None
    if manifest['rendering']:
        import cv2
        cap = cv2.VideoCapture(str(run/'tracked.mp4'))
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == fps
        decoded = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                assert frame.shape[:2] == (video['height'], video['width'])
                decoded += 1
        finally:
            cap.release()
        assert decoded == n
    manifest_path = run/'run_config.json'
    try:
        relative = run.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        relative = str(run.resolve())
    report = dict(run=relative, condition=condition, model_size=manifest['model_size'],
        status='passed', frames=n, video_frames_decoded=decoded, yolo_calls=scheduled_count,
        log_alignment_and_schedule_valid=True, saved_prediction_reevaluation_identical=True,
        original_coordinate_bounds_valid=True, confidence_and_output_semantics_valid=True,
        sources=dict(Counter(p['source'] for p in predictions)),
        events=dict(Counter(event for p in predictions for event in p['events'])),
        run_config_sha256=sha256(manifest_path), metrics_sha256=sha256(run/'metrics.json'),
        weights_sha256=manifest['weights']['sha256'], gt_sha256=manifest['gt_sha256'])
    save_json(run/'verification.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for run in args.runs:
        reports.append(verify_run(run))
        print('Verified:', run.name, flush=True)
    save_json(args.output, dict(status='passed', runs=reports))


if __name__ == '__main__':
    main()
