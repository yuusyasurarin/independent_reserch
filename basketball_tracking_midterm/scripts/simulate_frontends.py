"""Synthetic sanity check: LK vs template System 1, with and without ROI YOLO.

NOT an accuracy result for experiment10. It renders a 1280x720 synthetic scene (textured
floor, ~28 px shaded ball with motion blur, dribbles, passes and arcs, moving "players"
that occlude it) and replaces YOLO with a noisy oracle (jittered true box, random misses,
random false positives). It isolates the question "how often does System 1 fail and force
a YOLO call?" and checks that the code paths run. Real accuracy needs run_fixed_lag.py on
experiment10 (see reports/2026-11-25-yolo-budget-alternatives/README.md).

  python scripts/simulate_frontends.py --seeds 0 1 2 --frames 900 --out results/sim.json
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adaptive_tracking import AdaptiveBallPolicy
from src.alternatives import TemplateBallPolicy, RoiScheduler, roi_window
from src.common import iou, load_config, ROOT
from src.lag_tracking import FixedLagBall

W, H, FPS = 1280, 720, 30


def trajectory(rng, n):
    """Piecewise motion: dribble (vertical bounces), pass (fast straight), shot (parabola), hold."""
    pts, x, y = [], 640., 500.
    while len(pts) < n:
        kind = rng.choice(['dribble', 'pass', 'shot', 'hold'], p=[.4, .3, .2, .1])
        if kind == 'dribble':
            vx = rng.uniform(-6, 6)
            for _ in range(int(rng.integers(30, 90))):
                t = len(pts) % 16 / 16
                pts.append((x, 430+140*abs(math.sin(math.pi*t)))); x = float(np.clip(x+vx, 80, W-80))
            y = pts[-1][1]
        elif kind == 'pass':
            tx, ty = rng.uniform(100, W-100), rng.uniform(300, 600)
            k = int(rng.integers(10, 25))
            for i in range(1, k+1):
                pts.append((x+(tx-x)*i/k, y+(ty-y)*i/k))
            x, y = tx, ty
        elif kind == 'shot':
            tx = rng.uniform(150, W-150); k = int(rng.integers(30, 50)); peak = rng.uniform(150, 300)
            for i in range(1, k+1):
                s = i/k
                pts.append((x+(tx-x)*s, y-4*peak*s*(1-s)+(380-y)*s))
            x, y = tx, 380.
        else:
            for _ in range(int(rng.integers(10, 30))):
                pts.append((x+rng.normal(0, 1), y+rng.normal(0, 1)))
    return np.array(pts[:n])


def floor(rng):
    base = rng.integers(90, 160, (H, W, 3)).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2)
    base[..., 0] = (base[..., 0]*.6).astype(np.uint8); base[..., 2] = np.clip(base[..., 2]*1.3, 0, 255).astype(np.uint8)
    for yy in (200, 360, 520):
        cv2.line(base, (0, yy), (W, yy+30), (230, 230, 230), 3)
    cv2.circle(base, (640, 360), 120, (230, 230, 230), 3)
    return base


def render(bg, c, v, players, r=14):
    f = bg.copy()
    layer = np.zeros_like(f); mask = np.zeros((H, W), np.float32)
    cx, cy = int(round(c[0])), int(round(c[1]))
    cv2.circle(layer, (cx, cy), r, (20, 90, 210), -1); cv2.circle(mask, (cx, cy), r, 1., -1)
    cv2.circle(layer, (cx-4, cy-5), 5, (90, 170, 250), -1)
    cv2.line(layer, (cx-r, cy), (cx+r, cy), (10, 30, 60), 1); cv2.line(layer, (cx, cy-r), (cx, cy+r), (10, 30, 60), 1)
    speed = float(np.hypot(*v))
    if speed > 3:  # motion blur along the velocity
        k = int(min(31, speed*.6)) | 1
        kern = np.zeros((k, k), np.float32); kern[k//2, :] = 1/k
        rot = cv2.getRotationMatrix2D((k/2-.5, k/2-.5), -math.degrees(math.atan2(v[1], v[0])), 1)
        kern = cv2.warpAffine(kern, rot, (k, k)); kern /= max(kern.sum(), 1e-6)
        layer = cv2.filter2D(layer, -1, kern); mask = cv2.filter2D(mask, -1, kern)
    f = (f*(1-mask[..., None])+layer*mask[..., None]).astype(np.uint8)
    visible = True
    for (px, py, pw, ph, col) in players:
        cv2.rectangle(f, (int(px), int(py)), (int(px+pw), int(py+ph)), col, -1)
        if px-r < c[0] < px+pw+r and py-r < c[1] < py+ph+r:
            visible = False
    return f, visible


def oracle(rng, gt, visible, window=None, miss=.1, fp=.05):
    out = []
    if visible and rng.random() > miss:
        j = rng.normal(0, 1.5, 4)
        out.append(dict(bbox=[gt[0]+j[0], gt[1]+j[1], gt[2]+j[2], gt[3]+j[3]], confidence=float(rng.uniform(.5, .95)), class_id=0))
    if rng.random() < fp:
        x, y = rng.uniform(0, W-30), rng.uniform(0, H-30)
        out.append(dict(bbox=[x, y, x+28, y+28], confidence=float(rng.uniform(.3, .8)), class_id=0))
    if window is not None:
        x1, y1, s = window
        out = [c for c in out if x1 <= (c['bbox'][0]+c['bbox'][2])/2 < x1+s and y1 <= (c['bbox'][1]+c['bbox'][3])/2 < y1+s]
    return out


def run(seed, n, frontend, roi_full_every, cfg, fast=False):
    import copy
    cfg = copy.deepcopy(cfg)
    if fast:  # ROI calls are ~1/9 the cost: allow one every frame
        cfg['adaptive'].update(max_interval=1, max_yolo_hz=30)
    rng = np.random.default_rng(seed)
    traj = trajectory(rng, n); bg = floor(rng)
    players = [[rng.uniform(0, W), rng.uniform(250, 450), 70, 200, tuple(int(v) for v in rng.integers(20, 200, 3)), rng.uniform(-5, 5)] for _ in range(4)]
    det_rng = np.random.default_rng(seed+1000)
    policy = (TemplateBallPolicy if frontend == 'template' else AdaptiveBallPolicy)(cfg, FPS)
    roi = RoiScheduler(960, 320, roi_full_every) if roi_full_every else None
    smoother = FixedLagBall(FPS, W, H, .4, 'bridge')
    gts, vis, online, outs, s1_ms = [], [], [], [], []
    for i in range(n):
        for p in players:
            p[0] = (p[0]+p[5]) % W
        v = traj[i]-traj[i-1] if i else np.zeros(2)
        frame, visible = render(bg, traj[i], v, [p[:5] for p in players])
        gt = [traj[i][0]-14, traj[i][1]-14, traj[i][0]+14, traj[i][1]+14]
        t = time.perf_counter(); detected = policy.needs_detection(i, frame); s1_ms.append((time.perf_counter()-t)*1000)
        cands = []
        if detected:
            win = roi.window(policy, frame) if roi else None
            cands = oracle(det_rng, gt, visible, win)
        row = policy.update(frame, i, cands, detected); row['yolo_executed'] = detected
        online.append(row)
        o = smoother.push(row, cands, detected, frame)
        if o: outs.append(o)
        gts.append(gt); vis.append(visible)
    outs += list(smoother.flush())
    def tsr(rows):
        ok = [iou(r['bbox'], g) >= .5 for r, g, v in zip(rows, gts, vis) if v]
        return sum(ok)/len(ok)
    calls = sum(r['yolo_executed'] for r in online)
    reasons = Counter(r['detection_reason'] for r in online if r['yolo_executed'])
    return dict(seed=seed, frontend=frontend, roi_full_every=roi_full_every, fast=fast,
                yolo_per_s=calls/(n/FPS), deferred_frames=sum('detection_budget_deferred' in r['events'] for r in online),
                reasons=dict(reasons), sources=dict(Counter(r['source'] for r in online)),
                tsr05_online=tsr(online), tsr05_bridge=tsr(outs),
                relative_inference_pixels=roi.relative_cost() if roi else 1.,
                system1_ms_median=float(np.median(s1_ms)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--frames', type=int, default=900)
    p.add_argument('--out')
    args = p.parse_args()
    cfg = load_config(ROOT/'configs/experiment_f.json')
    variants = [('lk', 0, False), ('template', 0, False), ('lk', 5, False), ('template', 5, False), ('lk', 5, True), ('template', 5, True)]
    results = [run(s, args.frames, f, r, cfg, fast) for f, r, fast in variants for s in args.seeds]
    print(f"{'variant':16} {'YOLO/s':>7} {'deferred':>8} {'TSR online':>10} {'TSR bridge':>10} {'pixels':>6}  reasons")
    summary = {}
    for f, r, fast in variants:
        rs = [x for x in results if x['frontend'] == f and x['roi_full_every'] == r and x['fast'] == fast]
        m = lambda k: float(np.mean([x[k] for x in rs]))
        reasons = sum((Counter(x['reasons']) for x in rs), Counter())
        tot = sum(reasons.values())
        name = f + ('+ROI' if r else '') + (' 30/s' if fast else '')
        summary[name] = dict(yolo_per_s=m('yolo_per_s'), deferred_frames=m('deferred_frames'), tsr05_online=m('tsr05_online'),
                             tsr05_bridge=m('tsr05_bridge'), relative_inference_pixels=m('relative_inference_pixels'),
                             system1_ms_median=m('system1_ms_median'), reasons={k: v/tot for k, v in reasons.items()})
        s = summary[name]
        print(f"{name:16} {s['yolo_per_s']:7.2f} {s['deferred_frames']:8.1f} {s['tsr05_online']*100:9.1f}% {s['tsr05_bridge']*100:9.1f}% "
              f"{s['relative_inference_pixels']:6.2f}  " + ', '.join(f'{k} {v*100:.0f}%' for k, v in sorted(s['reasons'].items(), key=lambda kv: -kv[1])))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(summary=summary, runs=results), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
