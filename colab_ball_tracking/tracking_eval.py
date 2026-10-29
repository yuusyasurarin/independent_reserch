"""Ball-only evaluation: zero-based frames, no GT-dependent prediction selection."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import zipfile


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write empty table')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frame_number(name):
    match = re.fullmatch(r'frame_(\d+)', PurePosixPath(name).stem)
    if not match:
        raise ValueError(f'Unsupported frame filename: {name}')
    return int(match[1])


def load_annotations(path, count, width, height, missing_policy='unknown'):
    """Missing files are unknown unless the annotator explicitly confirmed invisibility."""
    import yaml
    if missing_policy not in ('unknown', 'not_visible'):
        raise ValueError('Invalid missing-label policy')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive paths')
        config = yaml.safe_load(archive.read('data.yaml'))
        if config.get('names') not in ({0: 'ball'}, ['ball']):
            raise ValueError('Expected class 0: ball only')
        listed = [frame_number(s.strip()) for s in archive.read('train.txt').decode('utf-8-sig').splitlines() if s.strip()]
        if len(listed) != count or sorted(listed) != list(range(count)):
            raise ValueError('Image list must contain every video frame exactly once, starting at 0')
        labels = {}
        for name in names:
            if not name.startswith('labels/') or not name.endswith('.txt'):
                continue
            index = frame_number(name)
            if index not in range(count) or index in labels:
                raise ValueError(f'Duplicate/out-of-range label: {name}')
            lines = [s for s in archive.read(name).decode('utf-8-sig').splitlines() if s.strip()]
            if len(lines) > 1:
                raise ValueError(f'Single target evaluation requires at most one ball per frame: {name}')
            target = None
            if lines:
                parts = lines[0].split()
                if len(parts) != 5 or parts[0] != '0':
                    raise ValueError(f'Expected YOLO detection format (class xc yc w h): {name}')
                xc, yc, w, h = map(float, parts[1:])
                if not all(math.isfinite(v) for v in (xc, yc, w, h)) or not (0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'Invalid normalized bbox: {name}')
                edges = (xc-w/2, yc-h/2, xc+w/2, yc+h/2)
                if min(edges) < -2e-6 or max(edges) > 1+2e-6:
                    raise ValueError(f'BBox outside image: {name}')
                x1, y1, x2, y2 = [min(1., max(0., v)) for v in edges]
                target = [x1*width, y1*height, (x2-x1)*width, (y2-y1)*height]
            labels[index] = target
    gt = []
    for i in range(count):
        target = labels.get(i)
        # Even empty files require a declared policy; exporters do not preserve review status.
        status = 'bbox_labeled' if target is not None else missing_policy
        gt.append(dict(frame_index=i, status=status, bbox=target))
    return gt, dict(frames=count, bbox_labeled=sum(r['bbox'] is not None for r in gt),
                    missing_files=[i for i in range(count) if i not in labels],
                    empty_label_files=[i for i, b in labels.items() if b is None],
                    missing_policy=missing_policy, annotation_sha256=sha256(path))


def iou(a, b):
    if a is None or b is None:
        return 0.
    overlap = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return overlap/(a[2]*a[3]+b[2]*b[3]-overlap)


def select_target(candidates, previous_id):
    """Keep prior confirmed ID when present, else highest confidence; never inspect GT."""
    tracked = [c for c in candidates if c['track_id'] is not None]
    same = [c for c in tracked if c['track_id'] == previous_id]
    return max(same or tracked, key=lambda c: (c['confidence'], -c['track_id'])) if tracked else None


def stats(values):
    if not values:
        return dict(n=0, mean=None, median=None, p95=None)
    ordered = sorted(values)
    pos = .95*(len(ordered)-1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo))


def evaluate(gt, predictions, fps, threshold=.5):
    if not math.isfinite(fps) or fps <= 0 or not 0 < threshold <= 1:
        raise ValueError('Invalid FPS or IoU threshold')
    if [r['frame_index'] for r in gt] != list(range(len(gt))):
        raise ValueError('GT must contain dense zero-based frames')
    if len(predictions) != len(gt) or [r['frame_index'] for r in predictions] != list(range(len(gt))):
        raise ValueError('Predictions must include every frame, including missing outputs')
    rows, overlaps, errors, fail_runs = [], [], [], []
    run = 0
    previous_id = None
    id_changes = 0
    for truth, pred in zip(gt, predictions):
        target, output = truth['bbox'], pred['bbox']
        eligible = truth['status'] == 'bbox_labeled'
        overlap = iou(target, output) if eligible else None
        error = math.hypot(target[0]+target[2]/2-output[0]-output[2]/2,
                           target[1]+target[3]/2-output[1]-output[3]/2) if eligible and output is not None else None
        success = overlap >= threshold if eligible else None
        if eligible:
            overlaps.append(overlap)
            if error is not None:
                errors.append(error)
        if eligible and not success:
            run += 1
        elif run:
            fail_runs.append(run)
            run = 0
        current_id = pred['track_id'] if output is not None else None
        if current_id is not None and previous_id is not None and current_id != previous_id:
            id_changes += 1
        previous_id = current_id  # gaps break adjacency; this is NOT MOT IDSW
        rows.append(dict(frame_index=truth['frame_index'], timestamp_s=truth['frame_index']/fps,
                         gt_status=truth['status'], eligible=int(eligible), output=int(output is not None),
                         track_id=current_id, confidence=pred['confidence'], iou=overlap,
                         success_at_threshold=int(success) if eligible else None, cle_px=error,
                         **{f'gt_{k}': target[j] if target else None for j,k in enumerate(('x','y','w','h'))},
                         **{f'pred_{k}': output[j] if output else None for j,k in enumerate(('x','y','w','h'))}))
    if run:
        fail_runs.append(run)
    eligible_rows = [r for r in rows if r['eligible']]
    invisible = [r for r in rows if r['gt_status'] == 'not_visible']
    n = len(eligible_rows)
    summary = dict(total_frames=len(gt), evaluated_bbox_frames=n,
                   not_visible_frames=len(invisible), unknown_frames=sum(r['gt_status']=='unknown' for r in rows),
                   iou_threshold=threshold, tsr=sum(v >= threshold for v in overlaps)/n if n else None,
                   tsr_iou_025=sum(v >= .25 for v in overlaps)/n if n else None,
                   mean_iou=statistics.mean(overlaps) if n else None,
                   output_coverage=sum(r['output'] for r in eligible_rows)/n if n else None,
                   cle_px=stats(errors), missing_output_on_bbox_frames=sum(not r['output'] for r in eligible_rows),
                   output_on_not_visible_frames=sum(r['output'] for r in invisible),
                   output_rate_when_not_visible=sum(r['output'] for r in invisible)/len(invisible) if invisible else None,
                   failure_runs_ge3=sum(v>=3 for v in fail_runs), longest_failure_frames=max(fail_runs,default=0),
                   longest_failure_duration_s=max(fail_runs,default=0)/fps,
                   adjacent_selected_id_changes=id_changes,
                   unique_selected_ids=len({r['track_id'] for r in rows if r['track_id'] is not None}),
                   limitations=[
                       'Ball-only labels: no ring NCLE, ring-zone scores, visibility subtypes, shot macro scores, IDF1 or HOTA.',
                       'Not-visible outputs are reported separately; invisible ball positions cannot be judged as correct/incorrect.',
                       'Failure runs are descriptive within contiguous bbox-labeled frames; shot boundaries are not annotated.',
                       'Adjacent ID changes are diagnostic, not standard MOT ID switches.',
                       'CLE excludes missing outputs; TSR and mean IoU include them as failures/zero.',
                   ])
    return summary, rows
