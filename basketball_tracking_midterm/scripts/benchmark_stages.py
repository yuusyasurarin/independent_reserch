"""Per-stage latency benchmark and real-time budget check (speed only, no accuracy).

Measures, on the current machine, the stages used by scripts/realtime_tracking.py:
  decode  : reading one 1280x720 frame (video file) or a synthetic frame
  pre     : YOLOv5 letterbox + HWC->CHW + tensor upload (same code as src/detector.py)
  infer   : YOLOv5 forward pass
  post    : NMS + box rescale + .cpu()
  system1 : grayscale + goodFeaturesToTrack + forward/backward pyramidal LK around the ball
  draw    : drawing one box + mp4 encode of the frame (the recording thread's work)

It then checks the real-time budget at 30 fps:
  per-second load = yolo_hz * (pre+infer+post) + fps * (decode+system1+draw)
which must stay below 1000 ms (one CPU/GPU pipeline, no overlap assumed; conservative).

Weights are only used for timing, so the official yolov5s.pt is acceptable: the fine-tuned
1-class model has the same backbone/neck and differs only in the final 1x1 conv channels.
This script does NOT verify provenance and its numbers must not be reported as accuracy.

Example (Windows laptop, AC power):
  .\\.venv\\Scripts\\python.exe scripts/benchmark_stages.py --yolov5 third_party/yolov5 ^
      --weights weights/finetuned/s/best.pt --device cuda:0 --imgsz 960 640 --video "../編集済み元データ/experiment10_編集.mp4"
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path


def stats(values):
    values = sorted(values)
    if not values:
        return None
    p95 = values[min(len(values)-1, math.ceil(.95*len(values))-1)]
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values), p95=p95)


def load_frames(args, np, cv2):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        frames, decode = [], []
        while len(frames) < args.frames:
            t = time.perf_counter()
            ok, frame = cap.read()
            dt = (time.perf_counter()-t)*1000
            if not ok:
                break
            frames.append(frame)
            decode.append(dt)
        cap.release()
        if not frames:
            raise SystemExit(f'Could not read {args.video}')
        return frames, decode, 'video'
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    frames = []
    for i in range(args.frames):
        f = base.copy()
        x, y = 200+6*i % (args.width-400), 300+int(80*math.sin(i/8))
        cv2.circle(f, (x, y), 14, (30, 90, 220), -1)
        cv2.circle(f, (x-4, y-4), 4, (200, 220, 255), -1)
        frames.append(f)
    return frames, None, 'synthetic'


def bench_system1(frames, np, cv2):
    """Same OpenCV calls and parameters as AdaptiveBallPolicy.prepare/seed."""
    times, prev_gray = [], None
    cx, cy = frames[0].shape[1]//2, frames[0].shape[0]//2
    box = [cx-15, cy-15, cx+15, cy+15]  # ~30 px ball, as in experiment10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .01)
    for frame in frames:
        t = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = box
        crop = gray[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(crop, maxCorners=30, qualityLevel=.01, minDistance=2, blockSize=3)
        if prev_gray is not None and pts is not None:
            m = 45
            h, w = gray.shape
            cx1, cy1, cx2, cy2 = max(0, x1-m), max(0, y1-m), min(w, x2+m), min(h, y2+m)
            local = (pts+np.array([x1-cx1, y1-cy1], dtype=np.float32)).astype(np.float32)
            a, b = prev_gray[cy1:cy2, cx1:cx2], gray[cy1:cy2, cx1:cx2]
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, local, None, winSize=(21, 21), maxLevel=3, criteria=criteria)
            cv2.calcOpticalFlowPyrLK(b, a, nxt, None, winSize=(21, 21), maxLevel=3, criteria=criteria)
        prev_gray = gray
        times.append((time.perf_counter()-t)*1000)
    return times


def bench_draw(frames, cv2):
    times = []
    with tempfile.TemporaryDirectory() as d:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(Path(d)/'b.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for frame in frames:
            t = time.perf_counter()
            f = frame.copy()
            cv2.rectangle(f, (100, 100), (130, 130), (0, 200, 0), 2)
            cv2.putText(f, 'ball 0.91', (100, 92), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 0), 2)
            writer.write(f)
            times.append((time.perf_counter()-t)*1000)
        writer.release()
    return times


def bench_yolo(args, frames, imgsz, torch, np):
    sys.path.insert(0, str(Path(args.yolov5).resolve()))
    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    device = torch.device(args.device)
    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **({'weights_only': False} | k))  # trusted local checkpoint
    try:
        model = attempt_load(args.weights, device=device, inplace=True, fuse=True)
    finally:
        torch.load = orig
    model = model.half() if args.half else model.float()
    model.eval()
    stride = int(model.stride.max())
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    out = dict(pre=[], infer=[], post=[])

    def detect(frame):
        sync(); t0 = time.perf_counter()
        img = letterbox(frame, new_shape=(imgsz, imgsz), auto=False, stride=stride)[0]
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        x = torch.from_numpy(img).to(device)
        x = (x.half() if args.half else x.float()).unsqueeze(0)/255
        sync(); t1 = time.perf_counter()
        with torch.inference_mode():
            raw = model(x, augment=False)[0]
        sync(); t2 = time.perf_counter()
        r = non_max_suppression(raw, conf_thres=.01, iou_thres=.45, max_det=300)[0]
        if len(r):
            r[:, :4] = scale_boxes(x.shape[2:], r[:, :4], frame.shape[:2])
        r.detach().cpu().tolist()
        sync(); t3 = time.perf_counter()
        return (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000

    for _ in range(args.warmup):
        detect(frames[0])
    n = min(len(frames), args.yolo_frames)
    for frame in frames[:n]:
        a, b, c = detect(frame)
        out['pre'].append(a); out['infer'].append(b); out['post'].append(c)
    sys.path.pop(0)
    return out


def budget(yolo_ms, other_ms, fps, yolo_hz):
    load = yolo_hz*yolo_ms + fps*other_ms
    return dict(yolo_hz=yolo_hz, load_ms_per_s=load, utilisation=load/1000,
                sustainable_fps=fps*1000/load if load else math.inf, realtime=load < 1000)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--yolov5', required=True, help='YOLOv5 v7.0 source folder')
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--imgsz', type=int, nargs='+', default=[960])
    p.add_argument('--half', action='store_true', help='FP16 (CUDA only)')
    p.add_argument('--threads', type=int, default=0, help='torch CPU threads (0 = default)')
    p.add_argument('--video')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--frames', type=int, default=200)
    p.add_argument('--yolo-frames', type=int, default=60)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--yolo-hz', type=float, nargs='+', default=[30., 15., 12.85],
                   help='YOLO calls per second to check (30=every frame/B, 15=cap of G, 12.85=measured s/G)')
    p.add_argument('--out', help='write JSON here')
    args = p.parse_args()
    if args.half and not args.device.startswith('cuda'):
        p.error('--half needs CUDA')

    import cv2
    import numpy as np
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    frames, decode, source = load_frames(args, np, cv2)
    s1 = bench_system1(frames, np, cv2)
    draw = bench_draw(frames, cv2)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    report = dict(platform=platform.platform(), processor=platform.processor() or platform.machine(),
                  python=sys.version.split()[0], torch=torch.__version__, opencv=cv2.__version__,
                  torch_threads=torch.get_num_threads(), cuda_device=gpu, device=args.device, half=args.half,
                  source=source, frame_size=list(frames[0].shape[1::-1]),
                  stages_ms=dict(decode=stats(decode) if decode else None, system1=stats(s1), draw_encode=stats(draw)),
                  yolo={})
    other = (statistics.median(decode) if decode else 0.) + statistics.median(s1) + statistics.median(draw)
    print(f'device={args.device} gpu={gpu} threads={torch.get_num_threads()} source={source}')
    print(f'decode={(f"{statistics.median(decode):.2f}ms") if decode else "n/a (synthetic)"}  '
          f'system1={statistics.median(s1):.2f}ms  draw+encode={statistics.median(draw):.2f}ms (medians)')
    for imgsz in args.imgsz:
        y = bench_yolo(args, frames, imgsz, torch, np)
        med = {k: statistics.median(v) for k, v in y.items()}
        total = sum(med.values())
        checks = [budget(total, other, args.fps, hz) for hz in args.yolo_hz]
        report['yolo'][str(imgsz)] = dict(stages_ms={k: stats(v) for k, v in y.items()}, total_median_ms=total, budget=checks)
        print(f'\nimgsz={imgsz}: pre={med["pre"]:.1f} infer={med["infer"]:.1f} post={med["post"]:.1f} -> YOLO {total:.1f}ms/call')
        for c in checks:
            print(f'  YOLO {c["yolo_hz"]:5.2f}/s: load {c["load_ms_per_s"]:7.0f} ms/s ({c["utilisation"]*100:5.1f}%) '
                  f'-> sustainable {c["sustainable_fps"]:5.1f} fps  {"OK" if c["realtime"] else "NOT real-time"}')
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print('\nwrote', args.out)


if __name__ == '__main__':
    main()
