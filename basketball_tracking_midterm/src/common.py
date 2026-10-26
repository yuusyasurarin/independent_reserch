from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
YOLO_COMMIT = '915bbf294bb74c859f0b41f1c23bc395014ea679'
BYTE_VERSION = '0.25.0'


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg['imgsz'] % 32 or cfg['imgsz'] < 32 or cfg['detect_interval'] < 1:
        raise ValueError('imgsz must be a multiple of 32; detect_interval must be >= 1')
    if not 0 < cfg['candidate_conf'] <= cfg['confidence'] <= 1:
        raise ValueError('Require 0 < candidate_conf <= confidence <= 1')
    if not 0 < cfg['nms_iou'] <= 1 or cfg['max_age_s'] <= 0:
        raise ValueError('Invalid NMS threshold / maximum age')
    return cfg


def stats(values):
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'p95': None}
    seq = sorted(values)
    x = .95 * (len(seq)-1)
    lo, hi = math.floor(x), math.ceil(x)
    return dict(n=len(seq), mean=statistics.mean(seq), median=statistics.median(seq),
                p95=seq[lo]+(seq[hi]-seq[lo])*(x-lo))


def iou(a, b):
    if a is None or b is None:
        return 0.0
    area = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0.0


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def valid_box(box, width, height):
    return (box is not None and len(box)==4 and all(math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def hardware_sample():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,temperature.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    versions = {}
    for name in ('torch','torchvision','numpy','opencv-contrib-python','opencv-python','supervision',
                 'scipy','Pillow','PyYAML','pandas','matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  processor=platform.processor(), versions=versions, nvidia_smi=hardware_sample())
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                      cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except ImportError:
        result['cuda_available'] = False
    try:
        import cv2
        result['opencv'] = cv2.__version__
        result['kcf_available'] = hasattr(cv2,'TrackerKCF_create') or hasattr(getattr(cv2,'legacy',None),'TrackerKCF_create')
    except ImportError:
        result['kcf_available'] = False
    return result


def offline_environment():
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    os.environ['YOLOv5_VERBOSE'] = 'false'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['COMET_MODE'] = 'DISABLED'
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache' / 'matplotlib')
    os.environ['YOLOV5_CONFIG_DIR'] = str(ROOT / '.cache' / 'yolov5')


def check_repo(path):
    path = Path(path)
    marker = path / 'midterm_source.json'
    if not marker.is_file():
        raise FileNotFoundError('Run scripts/setup_assets.py to prepare the pinned YOLOv5 code')
    info = json.loads(marker.read_text(encoding='utf-8'))
    if info['commit'] != YOLO_COMMIT:
        raise ValueError('YOLOv5 commit differs from the fixed training/inference version')
    for name, digest in info['files'].items():
        if sha256(path/name) != digest:
            raise ValueError(f'YOLOv5 source modified: {name}')
    return info
