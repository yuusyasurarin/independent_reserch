"""PDF tracking metrics: IoU TSR/AUC, CLE/NCLE, coverage, censored Lost/recovery."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiment_tracking import center, iou, ring_zone


VISIBILITIES = ('visible', 'partial', 'occluded', 'out_of_frame')


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        index = int(row['frame_index'])
        if index < 0 or index in result:
            raise ValueError(f'Duplicate/negative frame_index: {index}')
        result[index] = row
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    return result


def box(row, prefix='bbox_'):
    if row is None:
        return None
    values = [row.get(prefix+k, '') for k in ('x', 'y', 'w', 'h')]
    if all(v == '' for v in values):
        return None
    if any(v == '' for v in values):
        raise ValueError('A bbox must have all four values or all blank')
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result) or min(result[2:]) <= 0:
        raise ValueError('Bbox values must be finite, with positive width/height')
    return result


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95))) if values else dict(n=0, mean=None, median=None, p95=None)


def summarize(samples, threshold=.5, epsilon=.05):
    eligible = [s for s in samples if s['eligible']]
    n = len(eligible)
    overlaps = [s['iou'] for s in eligible]
    thresholds = np.linspace(0, 1, 101)
    curve = [sum(v >= t for v in overlaps)/n if n else None for t in thresholds]
    return dict(frames=len(samples), eligible_frames=n,
                tsr=sum(s['iou'] >= threshold for s in eligible)/n if n else None,
                auc_mean_iou=float(np.mean(overlaps)) if n else None,
                auc_trapezoid_001=float(np.trapezoid(curve, thresholds)) if n else None,
                success_curve=curve, cle_px=stats([s['cle'] for s in eligible if s['cle'] is not None]),
                ncle=stats([s['ncle'] for s in eligible if s['ncle'] is not None]),
                relative_error=stats([s['relative'] for s in eligible if s['relative'] is not None]),
                output_coverage=sum(s['output'] for s in eligible)/n if n else None,
                valid_coverage=sum(s['ncle'] is not None and s['ncle'] <= epsilon for s in eligible)/n if n else None)


def events(samples, threshold=.5, k=3, m=3, window=.5):
    """Events only cross adjacent annotated frames in the same shot and eligible run."""
    blocks, block = [], []
    previous = None
    for sample in samples:
        contiguous = previous is not None and sample['index'] == previous['index']+1 and sample['shot'] == previous['shot']
        if not sample['eligible'] or not contiguous:
            if block:
                blocks.append(block)
                block = []
        if sample['eligible']:
            sample = dict(sample)
            sample['after_occlusion'] = bool(contiguous and previous['visibility'] == 'occluded')
            block.append(sample)
        previous = sample
    if block:
        blocks.append(block)
    lost, revisible = [], []
    for block in blocks:
        def finish(event, recovery):
            event = dict(event)
            event['recovery_s'] = recovery
            event['followup_s'] = block[-1]['timestamp']-event['start_s']
            event['censored'] = event['followup_s']+1e-9 < window
            return event

        if block[0].get('after_occlusion'):
            success_run, recovery = 0, None
            for j, sample in enumerate(block):
                success_run = success_run+1 if sample['iou'] >= threshold else 0
                if success_run >= m:
                    recovery = block[j-m+1]['timestamp']-block[0]['timestamp']
                    break
            revisible.append(finish(dict(shot=block[0]['shot'], start_s=block[0]['timestamp'],
                                         frame_index=block[0]['index']), recovery))
        fail_run = success_run = 0
        active = None
        for j, sample in enumerate(block):
            if sample['iou'] >= threshold:
                success_run += 1
                fail_run = 0
                if active and success_run >= m:
                    recovery = block[j-m+1]['timestamp']-active['start_s']
                    lost.append(finish(active, recovery))
                    active = None
            else:
                success_run = 0
                fail_run += 1
                if active is None and fail_run >= k:
                    first = block[j-k+1]
                    active = dict(shot=first['shot'], start_s=first['timestamp'], frame_index=first['index'])
        if active:
            lost.append(finish(active, None))

    def report(items):
        eligible = [e for e in items if not e['censored']]
        recovered = [e for e in items if e['recovery_s'] is not None]
        return dict(events=len(items), eligible_window_events=len(eligible),
                    censored=sum(e['censored'] for e in items),
                    recovered_within_window=sum(e['recovery_s'] is not None and e['recovery_s'] <= window for e in eligible),
                    recovery_rate=sum(e['recovery_s'] is not None and e['recovery_s'] <= window for e in eligible)/len(eligible) if eligible else None,
                    unrecovered=sum(e['recovery_s'] is None for e in items),
                    recovery_seconds=stats([e['recovery_s'] for e in recovered]), details=items)

    shots = {s['shot'] for s in samples}
    return dict(lost=report(lost), revisibility=report(revisible),
                lost_per_shot=len(lost)/len(shots) if shots else None,
                lost_shot_rate=len({e['shot'] for e in lost})/len(shots) if shots else None)


def evaluate(gt, predictions, threshold=.5, epsilon=.05, k=3, m=3, window=.5, zone=(1.5, 2., 2.)):
    samples, previous_time = [], -math.inf
    for index, truth in sorted(gt.items()):
        visibility = truth['visibility']
        if visibility not in VISIBILITIES or not truth.get('shot_id'):
            raise ValueError('GT needs visibility and shot_id on every frame')
        t = float(truth['timestamp_s'])
        if not math.isfinite(t) or t <= previous_time:
            raise ValueError('GT timestamps must be finite and strictly increasing')
        previous_time = t
        target, ring = box(truth), box(truth, 'ring_bbox_')
        eligible = visibility in ('visible', 'partial')
        if eligible and (target is None or ring is None):
            raise ValueError(f'Eligible GT needs ball AND ring bbox: frame {index}')
        if not eligible and target is not None:
            raise ValueError(f'Do not invent ball GT during occlusion/out-of-frame: {index}')
        pred = predictions.get(index)
        output, predicted_ring = box(pred), box(pred, 'ring_bbox_')
        cle = float(np.linalg.norm(center(target)-center(output))) if eligible and output else None
        ncle = cle/ring[2] if cle is not None else None
        relative = None
        if cle is not None and predicted_ring is not None:
            relative = float(np.linalg.norm((center(output)-center(predicted_ring))/predicted_ring[2] -
                                            (center(target)-center(ring))/ring[2]))
        near = eligible and iou(target, ring_zone(ring, *zone)) > 0
        for flag in ('fast', 'small'):
            if truth.get(flag, '0') not in ('0', '1'):
                raise ValueError(f'{flag} must be 0 or 1')
        samples.append(dict(index=index, timestamp=t, shot=truth['shot_id'], visibility=visibility,
                            eligible=eligible, output=output is not None, iou=iou(target, output),
                            cle=cle, ncle=ncle, relative=relative, near_ring=near,
                            fast=truth.get('fast') == '1', small=truth.get('small') == '1'))
    groups = dict(all=samples, normal=[s for s in samples if s['visibility'] == 'visible' and not s['near_ring'] and not s['fast']],
                  partial=[s for s in samples if s['visibility'] == 'partial'],
                  near_ring=[s for s in samples if s['near_ring']],
                  fast=[s for s in samples if s['fast']], small=[s for s in samples if s['small']])
    report = {name: summarize(rows, threshold, epsilon) for name, rows in groups.items()}
    shots = {s['shot'] for s in samples}
    report['by_shot'] = {shot: summarize([s for s in samples if s['shot'] == shot], threshold, epsilon) for shot in sorted(shots)}
    report['macro_shot'] = {}
    rng = np.random.default_rng(42)
    for metric in ('tsr', 'output_coverage', 'valid_coverage', 'auc_mean_iou'):
        values = [r[metric] for r in report['by_shot'].values() if r[metric] is not None]
        means = rng.choice(values, size=(1000, len(values)), replace=True).mean(1) if len(values) >= 2 else None
        report['macro_shot'][metric] = dict(mean=float(np.mean(values)) if values else None,
                                            bootstrap_shot_ci95=np.percentile(means, [2.5, 97.5]).tolist() if means is not None else None)
    report['events'] = events(samples, threshold, k, m, window)
    occluded = [s for s in samples if s['visibility'] == 'occluded']
    report['occluded_output_rate'] = sum(s['output'] for s in occluded)/len(occluded) if occluded else None
    report['missing_prediction_rows'] = len(set(gt)-set(predictions))
    report['extra_prediction_rows'] = len(set(predictions)-set(gt))
    report['latency_ms'] = stats([float(r['latency_ms']) for r in predictions.values() if r.get('latency_ms', '') != ''])
    report['processing_ms'] = stats([float(r['processing_ms']) for r in predictions.values() if r.get('processing_ms', '') != ''])
    report['detector_age_ms'] = stats([float(r['detector_age_ms']) for r in predictions.values() if r.get('detector_age_ms', '') != ''])
    return report


def candidate_recall(gt, detections):
    report = {}
    for field in ('blob_rois', 'proposals'):
        centers = contained = count = 0
        for detection in detections:
            truth = gt.get(detection['frame_index'])
            if not truth or truth['visibility'] not in ('visible', 'partial'):
                continue
            target = box(truth)
            if target is None:
                continue
            count += 1
            x, y = center(target)
            rois = detection[field]
            centers += any(b[0] <= x <= b[0]+b[2] and b[1] <= y <= b[1]+b[3] for b in rois)
            contained += any(max(0, min(b[0]+b[2], target[0]+target[2])-max(b[0], target[0])) *
                             max(0, min(b[1]+b[3], target[1]+target[3])-max(b[1], target[1])) >= .9*target[2]*target[3] for b in rois)
        report[field] = dict(eligible_detector_frames=count, center_recall=centers/count if count else None,
                             bbox_90percent_recall=contained/count if count else None)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ground-truth', type=Path, required=True)
    p.add_argument('--predictions', type=Path, nargs='+', required=True)
    p.add_argument('--iou-threshold', type=float, default=.5)
    p.add_argument('--epsilon', type=float, default=.05)
    p.add_argument('--lost-k', type=int, default=3)
    p.add_argument('--recovery-m', type=int, default=3)
    p.add_argument('--window', type=float, default=.5)
    p.add_argument('--zone', type=float, nargs=3, default=(1.5, 2., 2.))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not all(math.isfinite(v) for v in [a.iou_threshold, a.epsilon, a.window, *a.zone]) or not 0 < a.iou_threshold <= 1 or min(a.epsilon, a.window, a.lost_k, a.recovery_m, *a.zone) <= 0:
        p.error('Invalid evaluation thresholds')
    gt = read_rows(a.ground_truth)
    result = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != 'predictions'}, methods={})
    for path in a.predictions:
        report = evaluate(gt, read_rows(path), a.iou_threshold, a.epsilon, a.lost_k, a.recovery_m, a.window, a.zone)
        events_path = path.parent/'detections.jsonl'
        if events_path.is_file():
            detections = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
            report['candidate_recall'] = candidate_recall(gt, detections)
            report['detector_jobs'] = len(detections)
            report['detector_ms'] = stats([d['detector_ms'] for d in detections])
            report['yolo_batches'] = sum(d['yolo_batches'] for d in detections)
            report['common_batches'] = sum(d['common_batches'] for d in detections)
        runtime = path.parent/'runtime.json'
        if runtime.is_file():
            report['runtime'] = json.loads(runtime.read_text(encoding='utf-8'))
        result['methods'][str(path)] = report
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f'Saved: {a.output}')


if __name__ == '__main__':
    main()
