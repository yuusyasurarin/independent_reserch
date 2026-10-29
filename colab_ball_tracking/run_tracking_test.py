"""Run a frozen YOLO model + ByteTrack on the packaged test video."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from tracking_eval import evaluate, load_annotations, save_csv, save_json, select_target, sha256, stats

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--conf', type=float, default=.10)
    p.add_argument('--nms-iou', type=float, default=.5)
    p.add_argument('--eval-iou', type=float, default=.5)
    p.add_argument('--device', default='0')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--class-id', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=0, help='Smoke test only: nonzero results are explicitly partial')
    a = p.parse_args()
    if a.imgsz < 32 or a.imgsz % 32 or a.max_frames < 0 or a.class_id < 0:
        p.error('Invalid image size / frame count / class ID')
    if not all(math.isfinite(v) and 0 < v <= 1 for v in (a.conf,a.nms_iou,a.eval_iou)):
        p.error('Thresholds must be in (0,1]')
    if not a.model.is_file():
        p.error(f'Provide the trained best.pt file: {a.model}')
    if a.output.exists():
        p.error('Output already exists; use a new run directory')
    return a


def main():
    a = parse_args()
    bundle = a.bundle.resolve()
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    video = bundle / manifest['video']['file']
    annotations = bundle / manifest['annotations']['file']
    tracker = bundle / 'bytetrack.yaml'
    for path, digest in ((video,manifest['video']['sha256']), (annotations,manifest['annotations']['sha256'])):
        if sha256(path) != digest:
            raise ValueError(f'Input checksum mismatch: {path}')
    a.output.mkdir(parents=True, exist_ok=False)
    for variable, folder in (('YOLO_CONFIG_DIR','ultralytics_config'),('MPLCONFIGDIR','matplotlib_config')):
        directory = (a.output/folder).resolve()
        directory.mkdir(exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if a.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('GPU is unavailable. Select a GPU runtime in Colab or --device cpu.')
    info = manifest['video']
    fps, count, width, height = info['fps'],info['frame_count'],info['width'],info['height']
    gt, annotation_report = load_annotations(annotations,count,width,height,manifest['annotations']['missing_policy'])
    limit = min(count,a.max_frames) if a.max_frames else count
    gt = gt[:limit]
    model = YOLO(str(a.model.resolve()))
    class_name = model.names.get(a.class_id) if isinstance(model.names,dict) else model.names[a.class_id]
    if class_name not in ('ball','sports ball','basketball'):
        raise ValueError(f'Chosen model class is not a ball: {a.class_id}: {class_name}')
    if a.class_id == 0 and model.names != {0:'ball'}:
        raise ValueError('Default evaluation requires the fine-tuned single-class ball model')
    config = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  model_sha256=sha256(a.model),model_names=model.names,
                  video=info,annotations=annotation_report,source_manifest=manifest,
                  tracker_sha256=sha256(tracker),selection_rule='prior confirmed ID if present, else highest confidence; unconfirmed detections excluded',
                  versions=dict(python=platform.python_version(),ultralytics=ultralytics.__version__,
                                torch=torch.__version__,opencv=cv2.__version__,numpy=np.__version__,cuda=torch.version.cuda),
                  device_name=torch.cuda.get_device_name(torch.device('cuda:'+a.device.split(',')[0])) if a.device != 'cpu' else platform.processor(),
                  complete_video=limit==count, timing='Warmed model.track + CPU result retrieval; wall FPS also includes video reading/drawing/writing/logging. Not camera latency.')
    save_json(a.output/'run_config.json',config)
    shutil.copy2(tracker,a.output/'bytetrack.yaml')
    freeze = subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (a.output/'environment.txt').write_text(freeze.stdout,encoding='utf-8')
    inference = dict(classes=[a.class_id],conf=a.conf,iou=a.nms_iou,imgsz=a.imgsz,device=a.device,
                     quantize=16 if a.fp16 and a.device!='cpu' else None,verbose=False)
    model.predict(np.zeros((height,width,3),dtype=np.uint8),**inference)  # no tracker state in warm-up
    if a.device != 'cpu':
        torch.cuda.synchronize()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    writer = None
    predictions, timings = [], []
    previous_id = None
    started = time.perf_counter()
    try:
        if (int(capture.get(3)),int(capture.get(4)),int(capture.get(7))) != (width,height,count) or not math.isclose(capture.get(5),fps,rel_tol=1e-6):
            raise ValueError('Video metadata differs from prepared annotations')
        writer = cv2.VideoWriter(str(a.output/'tracked.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened():
            raise RuntimeError('Cannot create tracked.mp4')
        with (a.output/'candidates.jsonl').open('w',encoding='utf-8') as log:
            for index in range(limit):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'Video ended early at frame {index}')
                before = time.perf_counter()
                result = model.track(frame,persist=True,tracker=str(tracker),**inference)[0]
                boxes = result.boxes
                candidates = []
                if boxes is not None:
                    coords, scores = boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()
                    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(coords)
                    for xyxy, score, track_id in zip(coords,scores,ids):
                        x1,y1,x2,y2 = xyxy
                        if not all(math.isfinite(v) for v in (*xyxy,score)) or x2<=x1 or y2<=y1:
                            raise ValueError('Invalid model output')
                        candidates.append(dict(bbox=[x1,y1,x2-x1,y2-y1],confidence=score,track_id=track_id))
                elapsed_ms = (time.perf_counter()-before)*1000
                timings.append(elapsed_ms)
                selected = select_target(candidates,previous_id)
                if selected is not None:
                    previous_id = selected['track_id']
                pred = dict(frame_index=index,bbox=selected['bbox'] if selected else None,
                            track_id=selected['track_id'] if selected else None,
                            confidence=selected['confidence'] if selected else None)
                predictions.append(pred)
                log.write(json.dumps(dict(frame_index=index,candidates=candidates,selected=selected,track_ms=elapsed_ms),allow_nan=False)+'\n')
                # GT is first read here, after prediction and target selection.
                truth = gt[index]
                for box, color, label in ((truth['bbox'],(0,220,0),'GT'),(pred['bbox'],(0,160,255),f"ID {pred['track_id']}")):
                    if box is not None:
                        x,y,w,h = box
                        cv2.rectangle(frame,(round(x),round(y)),(round(x+w),round(y+h)),color,2)
                        cv2.putText(frame,label,(max(0,round(x)),max(16,round(y)-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                cv2.putText(frame,f"frame {index} | GT: {truth['status']} | green=GT orange=selected track",(12,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                writer.write(frame)
                if index%50==0 or index==limit-1:
                    print(f'{index+1}/{limit} frames',flush=True)
            if limit==count and capture.read()[0]:
                raise ValueError('Video has more frames than the annotation manifest')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    wall = time.perf_counter()-started
    summary, rows = evaluate(gt,predictions,fps,a.eval_iou)
    summary.update(complete_video=limit==count,track_ms=stats(timings),
                   track_only_fps=1000*len(timings)/sum(timings),wall_fps=len(predictions)/wall)
    save_csv(a.output/'per_frame.csv',rows)
    save_json(a.output/'metrics.json',summary)
    save_json(a.output/'predictions.json',predictions)
    save_csv(a.output/'summary.csv',[{k:v for k,v in summary.items() if not isinstance(v,(list,dict))}])
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f'Saved: {a.output.resolve()}')


if __name__ == '__main__':
    main()
