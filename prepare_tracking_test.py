"""Prepare held-out video metadata and explicitly UNANNOTATED tracking GT templates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
FIELDS = [
    'frame_index', 'timestamp_s', 'shot_id', 'visibility',
    'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
    'ring_bbox_x', 'ring_bbox_y', 'ring_bbox_w', 'ring_bbox_h',
    'fast', 'small', 'score',
]


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or expected <= 0:
            raise ValueError(f'Invalid video metadata: {path}')
        width, height = int(cap.get(3)), int(cap.get(4))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f'Invalid frame: {path}, {count}')
            count += 1
        if count != expected:
            raise ValueError(f'Decode count mismatch: {path}: {count} != {expected}')
    finally:
        cap.release()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.resolve()), sha256=digest, fps=fps,
                frame_count=count, width=width, height=height,
                duration_s=count/fps, timestamp_basis='frame_index / nominal_fps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / '編集済み元データ')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/tracking_test_experiment09_10')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing data: {args.output}')
    videos = []
    for name in ('experiment09', 'experiment10'):
        paths = sorted(args.source_dir.glob(name + '_*.mp4'))
        if len(paths) != 1:
            raise ValueError(f'Expected one {name}_*.mp4, found {len(paths)}')
        info = inspect_video(paths[0])
        info['name'] = name
        videos.append(info)
    if len({v['sha256'] for v in videos}) != len(videos):
        raise ValueError('Duplicate test videos')
    args.output.mkdir(parents=True, exist_ok=False)
    for info in videos:
        dest = args.output / info['name']
        dest.mkdir()
        (dest / 'annotations').mkdir()
        with (dest / 'ground_truth.template.csv').open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for index in range(info['frame_count']):
                writer.writerow(dict(frame_index=index, timestamp_s=f"{index/info['fps']:.9f}",
                                     visibility='unannotated'))
    report = dict(status='prepared_unannotated', purpose='held_out_tracking_test',
                  evaluation_executed=False, opencv_version=cv2.__version__, videos=videos,
                  note='Blank annotations are unknown, NOT background. Complete and review GT before evaluation.')
    (args.output / 'preparation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
