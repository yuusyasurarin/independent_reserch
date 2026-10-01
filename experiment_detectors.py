"""Two-stage median background, blob CNN, and full-frame/ROI YOLO adapters."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from experiment_tracking import expand, iou, ring_zone, union


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode: {path}')
    data.tofile(path)


def clip(box, shape):
    h, w = shape[:2]
    x, y, bw, bh = box
    x1, y1 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    x2, y2 = min(w, int(math.ceil(x+bw))), min(h, int(math.ceil(y+bh)))
    return (x1, y1, x2-x1, y2-y1) if x2 > x1 and y2 > y1 else None


def crop(frame, box):
    x, y, w, h = box
    return frame[y:y+h, x:x+w]


def build_background(video, output, size1=18, size2=18):
    if min(size1, size2) < 1:
        raise ValueError('Median stage sizes must be positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video))
    medians = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        for _ in range(size2):
            group = []
            for _ in range(size1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Background clip needs at least {size1*size2} frames')
                group.append(frame)
            medians.append(np.median(group, axis=0).astype(np.uint8))
        background = np.median(medians, axis=0).astype(np.uint8)
        save_image(output, background)
        from adaptive_basketball_tracker import file_hash
        output.with_suffix('.json').write_text(json.dumps(dict(
            video=str(Path(video).resolve()), video_sha256=file_hash(video), size1=size1, size2=size2,
            frames=size1*size2, source_fps=fps, preparation_ms=(time.perf_counter()-started)*1000,
            acquisition_seconds=size1*size2/fps if fps > 0 else None,
            rule='First size1*size2 frames of a separate pre-experiment background clip; static thereafter.'
        ), indent=2), encoding='utf-8')
    finally:
        cap.release()


class Blobs:
    def __init__(self, background, threshold=25, min_area=4, padding=1.5):
        self.background = read_image(background)
        self.threshold, self.min_area, self.padding = threshold, min_area, padding

    def boxes(self, frame):
        if frame.shape != self.background.shape:
            raise ValueError('Background and input must have the same dimensions')
        difference = cv2.absdiff(frame, self.background).max(axis=2)
        mask = (difference >= self.threshold).astype(np.uint8)*255
        # No erosion: a tiny ball must not disappear before classification.
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes = [tuple(map(int, stats[k, :4])) for k in range(1, count)
                 if stats[k, cv2.CC_STAT_AREA] >= self.min_area]
        return boxes

    def rois(self, frame):
        return [clip(expand(b, self.padding), frame.shape) for b in self.boxes(frame)]


def merge_rois(boxes):
    result = []
    for box in boxes:
        if box is None:
            continue
        changed = True
        while changed:
            changed = False
            for existing in result[:]:
                if iou(box, existing) > 0:
                    box = union([box, existing])
                    result.remove(existing)
                    changed = True
        result.append(box)
    return result


def tiles(shape, boundary=False):
    h, w = shape[:2]
    if boundary:
        return [(0, 0, w, max(1, h//4)), (0, 3*h//4, w, h-3*h//4),
                (0, 0, max(1, w//4), h), (3*w//4, 0, w-3*w//4, h)]
    return [clip((x*w*.4, y*h*.4, w*.6, h*.6), shape) for y in range(2) for x in range(2)]


def suppress(candidates, threshold=.5):
    kept = []
    for candidate in sorted(candidates, key=lambda x: x[1], reverse=True):
        if all(iou(candidate[0], other[0]) < threshold for other in kept):
            kept.append(candidate)
    return kept


def patch_transform(size=107, augment=False):
    from torchvision import transforms as T
    operations = [T.Resize((size, size))]
    if augment:
        operations += [T.RandomHorizontalFlip(), T.ColorJitter(.15, .15, .15, .02)]
    return T.Compose(operations+[T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class BlobClassifier:
    def __init__(self, checkpoint, device='cpu'):
        import torch
        from torchvision.models import resnet18
        self.device = torch.device(device)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['classes'] != ['background', 'ball', 'player']:
            raise ValueError('Expected classes: background, ball, player')
        self.classes = state['classes']
        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 3)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device).eval()
        self.transform = patch_transform(state['image_size'])

    def predict(self, patches):
        import torch
        from PIL import Image
        result = []
        for start in range(0, len(patches), 32):
            batch = torch.stack([self.transform(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)))
                                 for p in patches[start:start+32]]).to(self.device)
            with torch.inference_mode():
                result.extend(self.model(batch).softmax(1).cpu().tolist())
        return result


class Detector:
    def __init__(self, args):
        import torch
        from ultralytics import YOLO
        from adaptive_basketball_tracker import find_class
        self.args = args
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        if self.device.isdigit():
            self.device = 'cuda:'+self.device
        self.yolo = YOLO(args.model)
        self.ball = find_class(self.yolo.names, args.class_id, ('ball', 'basketball', 'sports ball'), required=args.method != 'M1')
        self.person = find_class(self.yolo.names, args.person_class_id, ('player', 'person'), required=False)
        self.ring = find_class(self.yolo.names, args.ring_class_id, ('ring', 'rim', 'hoop'), required=False)
        if args.ring is None and self.ring is None:
            raise ValueError('Use a ball/player/ring model or explicitly supply --ring X Y W H (manual-ring condition)')
        if args.players != 'P0' and self.person is None and args.method != 'M1':
            raise ValueError('P1/P2 requires a player/person class')
        self.blobs = Blobs(args.background, args.diff_threshold, args.min_blob_area, args.blob_padding) if args.background else None
        self.cnn = BlobClassifier(args.cnn, self.device) if args.method == 'M1' else None
        self.last_full = self.last_ring = -math.inf

    def _yolo(self, frame, rois, classes):
        if not rois or not classes:
            return [], 0
        predictions = []
        # predict returns xyxy in each original crop's coordinate system.
        # It already undoes letterboxing; only the crop offset remains.
        for start in range(0, len(rois), self.args.roi_batch):
            batch = rois[start:start+self.args.roi_batch]
            results = self.yolo.predict([crop(frame, b) for b in batch], classes=classes,
                                       imgsz=self.args.image_size, conf=self.args.confidence,
                                       device=self.device, verbose=False)
            for roi, result in zip(batch, results):
                for coords, score, label in zip(result.boxes.xyxy.cpu().tolist(),
                                                result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                    x1, y1, x2, y2 = coords
                    predictions.append(((x1+roi[0], y1+roi[1], x2-x1, y2-y1), score, int(label)))
        return predictions, math.ceil(len(rois)/self.args.roi_batch)

    def __call__(self, frame, timestamp, context):
        a = self.args
        started = time.perf_counter()
        full = (0, 0, frame.shape[1], frame.shape[0])
        blob_boxes = self.blobs.boxes(frame) if self.blobs else []
        rois = [clip(expand(b, a.blob_padding), frame.shape) for b in blob_boxes]
        fallback, kind = False, 'full'
        periodic = timestamp-self.last_full >= a.full_interval
        if a.method in ('M1', 'M6'):
            kind = 'blob'
            if a.method == 'M6':
                rois = [clip(b, frame.shape) for b in merge_rois(rois)]
                if (not rois or len(rois) > a.max_rois or
                    sum(b[2]*b[3] for b in rois) > frame.shape[0]*frame.shape[1]*a.max_roi_area or
                    context['lost'] or periodic) and not a.pure_blob:
                    rois, kind, fallback = [full], 'full_fallback', True
        else:
            rois = [full]
        if a.method != 'M1' and not a.pure_blob:
            if context['lost'] and a.search != 'G0' and not periodic:
                if a.search == 'G1':
                    rois, kind = tiles(frame.shape), 'full_grid'
                elif a.search == 'G2':
                    rois, kind = tiles(frame.shape, True), 'boundary'
                else:
                    hints = [context.get('box'), context.get('player_roi')]
                    rois = [clip(expand(b, 3), frame.shape) for b in hints if b is not None]
                    rois = [b for b in rois if b]
                    kind = 'prediction_player'
                    if not rois:
                        rois, kind = [full], 'full_fallback'
            if context['near_ring'] and a.ring_policy in ('R2', 'R4') and context['ring'] is not None and not periodic:
                rois = [clip(ring_zone(context['ring'], a.alpha, a.beta_up, a.beta_down, 1.2), frame.shape)]
                rois, kind = [r for r in rois if r], 'ring_roi'
        if rois == [full]:
            self.last_full = timestamp
        proposals = list(rois)
        ring_due = self.ring is not None and timestamp-self.last_ring >= 1/a.ring_hz
        predictions, calls, cnn_patches = [], 0, 0
        balls, people = [], []
        if a.method == 'M1':
            # Blob bbox is the location estimate; padded crop is for classification only.
            boxes = blob_boxes
            if (not boxes or context['lost']) and not a.pure_blob:
                # Sliding windows are an explicit coarse localization fallback, not bbox regression.
                size, stride = a.grid_size, max(1, a.grid_size//2)
                boxes = [clip((x, y, size, size), frame.shape)
                         for y in range(0, frame.shape[0], stride) for x in range(0, frame.shape[1], stride)]
                kind, fallback = 'cnn_grid', True
                rois = boxes
            patches = [crop(frame, b) for b in rois]
            scores = self.cnn.predict(patches) if patches else []
            cnn_patches = len(patches)
            proposals = rois
            for box, prob in zip(boxes, scores):
                label = int(np.argmax(prob))
                if label == 1 and prob[1] >= a.cnn_confidence:
                    balls.append((box, prob[1]))
                elif label == 2:
                    people.append(box)
        else:
            classes = [v for v in (self.ball, self.person, self.ring if ring_due else None) if v is not None]
            predictions, calls = self._yolo(frame, rois, classes)
        # Common full-frame ring/player refresh, including for M1, separately counted.
        common_calls = 0
        if ring_due and (a.method == 'M1' or rois != [full]):
            common, common_calls = self._yolo(frame, [full], [v for v in (self.ring, self.person) if v is not None])
            predictions += common
        if ring_due:
            self.last_ring = timestamp
        if a.method != 'M1':
            balls = [(b, s) for b, s, label in predictions if label == self.ball]
        people += [b for b, _, label in predictions if label == self.person]
        rings = [(s, b) for b, s, label in predictions if label == self.ring]
        # .cpu() above synchronizes CUDA results, so wall timing includes completed inference.
        return dict(balls=suppress(balls), people=[b for b, _ in suppress([(b, 1.) for b in people])],
                    ring=max(rings, key=lambda v: v[0])[1] if rings else None,
                    proposals=proposals, blob_rois=[clip(expand(b, a.blob_padding), frame.shape) for b in blob_boxes],
                    search_kind=kind, fallback=fallback, roi_count=len(rois),
                    roi_area=sum(b[2]*b[3] for b in rois)/(frame.shape[0]*frame.shape[1]),
                    yolo_batches=calls, common_batches=common_calls, cnn_patches=cnn_patches,
                    started_clock_s=started, finished_clock_s=time.perf_counter(),
                    detector_ms=(time.perf_counter()-started)*1000)


def kcf_factory(frame, box):
    factory = getattr(cv2, 'TrackerKCF_create', None) or getattr(getattr(cv2, 'legacy', None), 'TrackerKCF_create', None)
    if factory is None:
        raise RuntimeError('KCF needs opencv-contrib-python')
    box = clip(box, frame.shape)
    if box is None or min(box[2:]) < 2:
        return None
    tracker = factory()
    return None if tracker.init(frame, box) is False else tracker
