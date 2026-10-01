"""Prepare pre-experiment backgrounds and reviewable frame/blob annotation manifests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from adaptive_basketball_tracker import file_hash


GT_FIELDS = ['frame_index', 'timestamp_s', 'shot_id', 'visibility', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
             'ring_bbox_x', 'ring_bbox_y', 'ring_bbox_w', 'ring_bbox_h', 'fast', 'small', 'score']
PATCH_FIELDS = ['path', 'session_id', 'split', 'video', 'frame_index', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                'label', 'reviewed', 'ignore_reason']


def load_sessions(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty session manifest')
    sessions, videos = {}, {}
    for row in rows:
        session, split = row['session_id'], row['split']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', session):
            raise ValueError('session_id must use letters, digits, underscore or hyphen')
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if session in sessions and sessions[session] != split:
            raise ValueError(f'Session leakage across splits: {session}')
        sessions[session] = split
        video = (path.parent/row['video']).resolve()
        if not video.is_file():
            raise ValueError(f'Video missing: {video}')
        digest = file_hash(video)
        if digest in videos:
            raise ValueError(f'Duplicate source video, even if renamed: {video}')
        videos[digest] = split
        row.update(video=str(video), video_sha256=digest)
    return rows


def extract(args):
    import cv2
    from experiment_detectors import Blobs, crop, save_image
    rows = load_sessions(args.sessions)
    if args.stride < 1 or args.min_blob_area < 1 or not 1 <= args.diff_threshold <= 255 or args.blob_padding <= 0:
        raise ValueError('Invalid stride, blob area, difference threshold or padding')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with (args.output/'patches.csv').open('w', encoding='utf-8', newline='') as patches_file:
        patches = csv.DictWriter(patches_file, fieldnames=PATCH_FIELDS)
        patches.writeheader()
        for video_number, row in enumerate(rows):
            background = (args.sessions.parent/row['background']).resolve() if row.get('background') else args.background
            blobs = Blobs(background, args.diff_threshold, args.min_blob_area, args.blob_padding) if background else None
            cap = cv2.VideoCapture(row['video'])
            prefix = f'{row["session_id"]}_{video_number:03d}'
            gt_path = args.output/(prefix+'_gt_template.csv')
            try:
                if not cap.isOpened():
                    raise ValueError(f'Cannot open {row["video"]}')
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    raise ValueError('Source FPS unavailable')
                with gt_path.open('w', encoding='utf-8', newline='') as gt_file:
                    gt = csv.DictWriter(gt_file, fieldnames=GT_FIELDS)
                    gt.writeheader()
                    index = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        # Complete frame table; no guessed visibility, coordinates or outcomes.
                        gt.writerow(dict(frame_index=index, timestamp_s=index/fps, shot_id='', visibility='', fast=0, small=0))
                        if index % args.stride == 0:
                            image_path = Path('images')/row['split']/f'{prefix}_{index:07d}.jpg'
                            save_image(args.output/image_path, frame)
                            if blobs:
                                for number, box in enumerate(blobs.rois(frame)):
                                    path = Path('patches')/f'{prefix}_{index:07d}_{number:04d}.png'
                                    save_image(args.output/path, crop(frame, box))
                                    patches.writerow(dict(path=path.as_posix(), session_id=row['session_id'], split=row['split'],
                                                          video=row['video'], frame_index=index,
                                                          **dict(zip(('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'), box)),
                                                          label='', reviewed=0, ignore_reason=''))
                        index += 1
            finally:
                cap.release()
    config = dict(sessions=rows, stride=args.stride, background=str(args.background) if args.background else None,
                  background_sha256=file_hash(args.background) if args.background else None,
                  diff_threshold=args.diff_threshold, min_blob_area=args.min_blob_area, blob_padding=args.blob_padding)
    (args.output/'preparation.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Images and UNLABELED review manifest: {args.output}')


def validate_patches(path, require_splits=True):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    accepted, sessions, hashes = [], {}, {}
    for row in rows:
        split, session = row['split'], row['session_id']
        if split not in ('train', 'val', 'test') or not session:
            raise ValueError('Every patch needs session_id and train/val/test split')
        if session in sessions and sessions[session] != split:
            raise ValueError(f'Session leakage: {session}')
        sessions[session] = split
        if row['reviewed'] != '1':
            raise ValueError(f'Unreviewed patch: {row["path"]}')
        if row['label'] == 'ignore':
            if not row.get('ignore_reason'):
                raise ValueError('Ignored patches need a reason')
            continue
        if row['label'] not in ('background', 'ball', 'player'):
            raise ValueError(f'Invalid label: {row["label"]}')
        image = (path.parent/row['path']).resolve()
        digest = file_hash(image)
        if digest in hashes and hashes[digest] != split:
            raise ValueError(f'Identical patch in different splits: {image}')
        hashes[digest] = split
        accepted.append({**row, 'path': str(image)})
    if require_splits and any(not any(r['split'] == s for r in accepted) for s in ('train', 'val', 'test')):
        raise ValueError('Reviewed train, val, test patches are all required')
    return accepted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    bg = sub.add_parser('background')
    bg.add_argument('--video', type=Path, required=True)
    bg.add_argument('--output', type=Path, required=True)
    bg.add_argument('--size1', type=int, default=18)
    bg.add_argument('--size2', type=int, default=18)
    ex = sub.add_parser('extract')
    ex.add_argument('--sessions', type=Path, required=True)
    ex.add_argument('--output', type=Path, required=True)
    ex.add_argument('--stride', type=int, default=6)
    ex.add_argument('--background', type=Path)
    ex.add_argument('--diff-threshold', type=int, default=25)
    ex.add_argument('--min-blob-area', type=int, default=4)
    ex.add_argument('--blob-padding', type=float, default=1.5)
    check = sub.add_parser('validate-patches')
    check.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'background':
        from experiment_detectors import build_background
        build_background(args.video, args.output, args.size1, args.size2)
    elif args.command == 'extract':
        extract(args)
    else:
        rows = validate_patches(args.manifest)
        print(json.dumps({s: {c: sum(r['split'] == s and r['label'] == c for r in rows)
                              for c in ('background', 'ball', 'player')} for s in ('train', 'val', 'test')}, indent=2))


if __name__ == '__main__':
    main()
