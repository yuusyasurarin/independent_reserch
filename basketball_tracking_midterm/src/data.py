from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from .common import save_csv, save_json, sha256, valid_box

STATUSES = {'visible', 'absent', 'unknown_position', 'unannotated'}


def validate_dataset(root, output, allow_background=False):
    """Read only. Errors never create or modify labels in the supplied dataset."""
    from PIL import Image
    import yaml
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors, warnings, rows = [], [], []
    hashes, pixel_hashes = defaultdict(list), defaultdict(list)
    config = yaml.safe_load((root/'data.yaml').read_text(encoding='utf-8'))
    if config.get('names') not in ({0:'ball'}, ['ball']):
        errors.append('data.yaml must define exactly class 0: ball')
    for split in ('train','val','test'):
        if config.get(split) != f'images/{split}':
            errors.append(f'data.yaml {split} must point to images/{split} for this dataset layout')
        images = sorted(p for p in (root/'images'/split).rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
        if not images:
            errors.append(f'No images in {split}')
        expected_labels = set()
        for image in images:
            label = root/'labels'/split/image.relative_to(root/'images'/split).with_suffix('.txt')
            if label.resolve() in expected_labels:
                errors.append(f'Multiple images map to the same label filename: {label}')
            expected_labels.add(label.resolve())
            item = dict(split=split, image=image.relative_to(root).as_posix(), label=label.relative_to(root).as_posix(),
                        image_sha256=sha256(image), label_sha256=None, status='missing', boxes=0,
                        width=None, height=None, pixel_sha256=None)
            hashes[item['image_sha256']].append((split,item['image']))
            try:
                with Image.open(image) as im:
                    im.load()
                    item['width'], item['height'] = im.size
                    rgb = im.convert('RGB')
                    item['pixel_sha256'] = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                    pixel_hashes[item['pixel_sha256']].append((split,item['image']))
            except Exception as e:
                errors.append(f'Broken image {image}: {e}')
            if not label.is_file():
                errors.append(f'Missing label (not background): {label}')
            else:
                item['label_sha256'] = sha256(label)
                lines = [x for x in label.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
                item['status'] = 'labeled' if lines else 'explicit_empty'
                if not lines and not allow_background:
                    errors.append(f'Empty label requires intentional-background confirmation: {label}')
                seen = set()
                for line in lines:
                    try:
                        values = [float(v) for v in line.split()]
                        if len(values)!=5 or values[0]!=0 or not all(math.isfinite(v) for v in values):
                            raise ValueError('Expected class 0 and four finite coordinates')
                        _, x,y,w,h = values
                        if not (0<w<=1 and 0<h<=1 and min(x-w/2,y-h/2)>=-2e-6 and max(x+w/2,y+h/2)<=1+2e-6):
                            raise ValueError('Invalid normalized box')
                        if tuple(values) in seen:
                            raise ValueError('Duplicate box')
                        seen.add(tuple(values))
                    except ValueError as e:
                        errors.append(f'{label}: {e}: {line}')
                item['boxes'] = len(lines)
            rows.append(item)
        extras = set(p.resolve() for p in (root/'labels'/split).rglob('*.txt'))-expected_labels
        errors.extend(f'Orphan label: {p}' for p in sorted(extras))
    duplicates = [items for items in pixel_hashes.values() if len(items)>1]
    for items in duplicates:
        if len({s for s,_ in items})>1:
            errors.append(f'Cross-split duplicate pixels: {items}')
        else:
            warnings.append(f'Within-split duplicate pixels: {items}')
    source_manifest = root/'manifest.csv'
    sources = []
    if source_manifest.is_file():
        with source_manifest.open(encoding='utf-8-sig', newline='') as f:
            source_rows = list(csv.DictReader(f))
        sources = sorted({r['source'] for r in source_rows})
        lookup = {r['image']:r for r in rows}
        if len(source_rows)!=len(rows):
            errors.append('Existing manifest row count differs from image inventory')
        for record in source_rows:
            current = lookup.get(record['image'])
            if current is None or any(current[k]!=record[k] for k in ('image_sha256','label_sha256')):
                errors.append(f'Existing manifest mismatch: {record["image"]}')
    counts = {split:dict(images=sum(r['split']==split for r in rows),
                        empty=sum(r['split']==split and r['status']=='explicit_empty' for r in rows),
                        boxes=sum(r['boxes'] for r in rows if r['split']==split)) for split in ('train','val','test')}
    report = dict(valid=not errors, root=str(root), counts=counts, errors=errors, warnings=warnings,
                  source_names=sources, allow_background=allow_background, image_cache=False,
                  data_yaml_sha256=sha256(root/'data.yaml'),
                  limitation='Exact file/pixel checks do not detect edited replays or near-duplicate scenes.')
    save_csv(output/'dataset_inventory.csv', rows)
    save_json(output/'dataset_validation.json', report)
    if errors:
        raise ValueError(f'Dataset validation failed ({len(errors)} issues); see {output}/dataset_validation.json')
    yaml_path = output/'data.yaml'
    yaml_path.write_text(yaml.safe_dump(dict(path=str(root),train='images/train',val='images/val',
                                            test='images/test',nc=1,names={0:'ball'}),allow_unicode=True),encoding='utf-8')
    return report, yaml_path


def inspect_video(path, decode_all=False, reference_pixels=None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps<=0 or count<1 or min(width,height)<=0:
            raise ValueError('Invalid video metadata')
        decoded, matches = 0, []
        if decode_all:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (height,width):
                    raise ValueError('Video dimensions changed')
                if reference_pixels and hashlib.sha256(frame.tobytes()).hexdigest() in reference_pixels:
                    matches.append(decoded)
                decoded += 1
            if decoded!=count:
                raise ValueError(f'Frame count differs: {decoded} decoded, {count} metadata')
        return dict(path=str(Path(path).resolve()),sha256=sha256(path),fps=fps,frame_count=count,width=width,height=height,
                    duration_s=count/fps,decoded_frames=decoded,exact_source_pixel_matches=matches,
                    timestamp_basis='frame_id / nominal fps; constant-fps source required')
    finally:
        cap.release()


def import_confirmed_gt(bundle, output):
    """Convert existing reviewed labels; invisible != confirmed absent."""
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    annotation = bundle/manifest['annotations']['file']
    if sha256(annotation)!=manifest['annotations']['sha256']:
        raise ValueError('Annotation archive changed')
    if manifest['annotations']['missing_policy']!='not_visible':
        raise ValueError('Expected the already-confirmed experiment10 manifest')
    # Independently parse the original annotation ZIP rather than trusting an editable derived CSV.
    import zipfile
    import yaml
    info = manifest['video']
    labels = {}
    with zipfile.ZipFile(annotation) as z:
        if len(set(z.namelist())) != len(z.namelist()):
            raise ValueError('Duplicate ZIP entries')
        if yaml.safe_load(z.read('data.yaml'))['names'] not in ({0:'ball'},['ball']):
            raise ValueError('Expected ball class 0')
        ids = [int(Path(s).stem.removeprefix('frame_')) for s in z.read('train.txt').decode('utf-8-sig').splitlines() if s.strip()]
        if sorted(ids)!=list(range(info['frame_count'])):
            raise ValueError('Video / annotation frame list differs')
        for name in z.namelist():
            if name.startswith('labels/') and name.endswith('.txt'):
                index = int(Path(name).stem.removeprefix('frame_'))
                if index not in range(info['frame_count']) or index in labels:
                    raise ValueError('Duplicate or out-of-range label frame')
                lines = [s for s in z.read(name).decode('utf-8-sig').splitlines() if s.strip()]
                if len(lines)>1:
                    raise ValueError('Single-ball experiment requires at most one GT box')
                box = None
                if lines:
                    values = list(map(float,lines[0].split()))
                    if len(values)!=5 or values[0]!=0:
                        raise ValueError('Bad YOLO label')
                    _,x,y,w,h=values
                    box=[(x-w/2)*info['width'],(y-h/2)*info['height'],(x+w/2)*info['width'],(y+h/2)*info['height']]
                    if not valid_box(box,info['width'],info['height']):
                        raise ValueError('Invalid GT coordinates')
                labels[index]=box
    rows=[]
    for i in range(info['frame_count']):
        box=labels.get(i)
        rows.append(dict(frame_id=i,timestamp_s=i/info['fps'],status='visible' if box else 'unknown_position',
                         x1=box[0] if box else '', y1=box[1] if box else '', x2=box[2] if box else '', y2=box[3] if box else '',
                         segment_id='',note='' if box else 'Reviewed not visible; absence versus occlusion not distinguished'))
    output.parent.mkdir(parents=True,exist_ok=True)
    save_csv(output,rows)
    save_json(output.with_suffix('.metadata.json'),dict(video_sha256=info['sha256'],annotation_sha256=sha256(annotation),
              gt_sha256=sha256(output),source_manifest=manifest,counts=dict(Counter(r['status'] for r in rows)),
              visibility_confirmation='All 629 unlabeled frames confirmed not visible by user on 2026-10-26; not asserted absent.'))


def load_gt(path, info):
    # Sparse CSV: missing rows are explicitly unannotated, never background.
    result=[dict(frame_id=i,status='unannotated',bbox=None,segment_id='') for i in range(info['frame_count'])]
    if path is None:
        return result
    path=Path(path)
    metadata=path.with_suffix('.metadata.json')
    if metadata.is_file():
        meta=json.loads(metadata.read_text(encoding='utf-8'))
        if meta['video_sha256']!=info['sha256'] or meta['gt_sha256']!=sha256(path):
            raise ValueError('GT / video checksum mismatch')
    seen=set()
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            i=int(row['frame_id'])
            if i in seen or i not in range(len(result)) or row['status'] not in STATUSES:
                raise ValueError('Invalid / duplicate frame_id or status')
            seen.add(i)
            if not math.isclose(float(row['timestamp_s']),i/info['fps'],abs_tol=1e-5):
                raise ValueError('GT time does not match video frame_id')
            coords=[row.get(k,'') for k in ('x1','y1','x2','y2')]
            box=list(map(float,coords)) if row['status']=='visible' else None
            if box is not None and not valid_box(box,info['width'],info['height']):
                raise ValueError('GT box outside original frame')
            if box is None and any(x.strip() for x in coords):
                raise ValueError('Non-visible GT must not assert a location')
            result[i]=dict(frame_id=i,status=row['status'],bbox=box,segment_id=row.get('segment_id',''))
    return result
