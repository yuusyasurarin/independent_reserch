"""Feasibility study: an LSTM that classifies a shot attempt as made / miss.

Only 27 labelled attempts exist (data/shot_labels_visual.csv), far too few to train a
sequence model from scratch. Two small experiments:

  sim   train on synthetic rim-relative trajectories (simple physics: arc, net, rim and
        backboard bounces, tracker noise, missing frames), test on the 27 real attempts
  lovo  train on the real attempts of two videos, test on the third (leave one video out)

Input per attempt: 60 frames (2 s at 30 fps) centred on the frame where the tracked ball
is closest to the rim; each frame = (dx, dy, visible) with dx = (x - rim centre)/rim half
width, dy = (y - rim line)/rim half width (positive = below), clipped to +-15.
The attempt itself (where the window is) comes from the label, so this measures
outcome classification only, not attempt detection.

  python scripts/shot_lstm.py --traj experiment10=... experiment11=... experiment12=... --json out.json
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.judge_shots import centres
from scripts.evaluate_shots import load_labels

WINDOW, BEFORE = 60, 30
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- real windows
def real_windows(trajectories, labels):
    data = []
    for video, path in trajectories:
        track = dict(centres(path))
        for lab in labels[video]:
            x1, y1, x2, y2 = lab['rim']
            cx, cy, half = (x1+x2)/2, (y1+y2)/2, (x2-x1)/2
            frames = [f for f in range(lab['start'], lab['end']+1) if track.get(f)]
            if frames:
                mid = min(frames, key=lambda f: math.hypot(track[f][0]-cx, track[f][1]-cy))
            else:
                mid = (lab['start']+lab['end'])//2
            seq = []
            for f in range(mid-BEFORE, mid-BEFORE+WINDOW):
                c = track.get(f)
                seq.append([np.clip((c[0]-cx)/half, -15, 15), np.clip((c[1]-cy)/half, -15, 15), 1.] if c else [0., 0., 0.])
            data.append(dict(video=video, start=lab['start'], y=int(lab['result'] == 'made'), x=np.array(seq, np.float32)))
    return data


# ---------------------------------------------------------------- synthetic
def simulate(rng):
    """One attempt in rim units (rim half width = 1, +y = down). Returns (T,3), label."""
    g = rng.uniform(.012, .045)
    side = rng.choice([-1, 1])
    kind = rng.choice(['made', 'made', 'made_bounce', 'side', 'rim_out', 'board', 'short'])
    r = rng.uniform(.5, 1.1)                       # ball radius in rim half widths
    apex_h = rng.uniform(3, 14)
    vy_down = math.sqrt(2*g*apex_h)                # speed when coming down through the rim line
    t_fall = vy_down/g
    xt = {'made': rng.uniform(-.6, .6), 'made_bounce': rng.uniform(-.9, .9), 'side': side*rng.uniform(1.3, 4.),
          'rim_out': side*rng.uniform(.6, 1.4), 'board': side*rng.uniform(-.5, 1.), 'short': rng.uniform(-3, 3)}[kind]
    vx = side*rng.uniform(.05, .6)*(-1 if rng.random() < .5 else 1)
    pts = []
    x, y, t = xt-vx*t_fall*1.6, -apex_h-rng.uniform(0, 8), 0
    vy = -math.sqrt(max(2*g*(abs(y)-apex_h), 0)) if abs(y) > apex_h else 0.
    if kind == 'short':                            # never reaches the rim line from above
        y, vy = rng.uniform(3, 8), -math.sqrt(2*g*rng.uniform(2, 6))
    state, bounced = 'air', 0
    for t in range(140):
        pts.append((x, y))
        vy += g; x += vx; y += vy
        if state == 'air' and vy > 0 and y >= 0 and pts[-1][1] < 0:
            if kind in ('made',) or (kind == 'made_bounce' and bounced >= 1):
                state = 'net'; net_t = 0
            elif kind == 'made_bounce' and bounced == 0:
                vy = -abs(vy)*rng.uniform(.2, .4); vx = -vx*.5; bounced = 1; y = -.01
            elif kind == 'rim_out':
                vy = -abs(vy)*rng.uniform(.3, .7); vx = side*rng.uniform(.3, 1.5); state = 'out'
            elif kind == 'board':
                vx = side*rng.uniform(.3, 1.2); x = side*rng.uniform(1.4, 2.5)+vx; state = 'out'
        if state == 'net':
            net_t += 1
            vx *= .6
            if net_t < 8:
                vy = min(vy, rng.uniform(.15, .4))
        if y > 25 or abs(x) > 30:
            break
    pts = np.array(pts)
    label = int(kind.startswith('made'))
    # centre the window on the frame closest to the rim, add tracker noise / gaps
    d = np.hypot(np.maximum(np.abs(pts[:, 0])-1, 0), pts[:, 1])
    mid = int(np.argmin(d))+rng.integers(-3, 4)
    seq = np.zeros((WINDOW, 3), np.float32)
    noise = rng.uniform(.03, .2)
    for k in range(WINDOW):
        i = mid-BEFORE+k
        if 0 <= i < len(pts):
            seq[k] = [np.clip(pts[i, 0]+rng.normal(0, noise), -15, 15), np.clip(pts[i, 1]+rng.normal(0, noise), -15, 15), 1.]
    vis = seq[:, 2] > 0
    for _ in range(rng.integers(0, 4)):            # missing stretches (occlusion / tracker loss)
        a = rng.integers(0, WINDOW); b = a+rng.integers(1, 12)
        vis[a:b] = False
    if label and rng.random() < .5:                # ball often vanishes in the net
        a = BEFORE+rng.integers(1, 6); vis[a:a+rng.integers(3, 20)] = False
    seq[~vis] = 0
    return seq, label


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    xs, ys = zip(*(simulate(rng) for _ in range(n)))
    return np.stack(xs), np.array(ys)


# ---------------------------------------------------------------- model
def train(x, y, epochs, seed, device):
    import torch
    torch.manual_seed(seed)
    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    xt, yt = torch.tensor(x).to(device), torch.tensor(y, dtype=torch.float32).to(device)
    pos = yt.mean().clamp(.05, .95)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=(1-pos)/pos)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 64):
            b = perm[i:i+64]
            opt.zero_grad(); loss = lossf(model(xt[b]), yt[b]); loss.backward(); opt.step()
    return model


def predict(model, x, device):
    import torch
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x).to(device))).cpu().numpy()


def Net():
    import torch

    class LSTMClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(3, 32, num_layers=2, batch_first=True, dropout=.1)
            self.head = torch.nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)
    return LSTMClassifier()


def metrics(y, p):
    pred = (p >= .5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return dict(n=len(y), accuracy=float((pred == y).mean()), made_precision=tp/(tp+fp) if tp+fp else None,
                made_recall=tp/(tp+fn) if tp+fn else None, predicted_made=int(pred.sum()), true_made=int(y.sum()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traj', nargs='+', required=True, help='video=trajectory file')
    ap.add_argument('--labels', default=str(ROOT/'data/shot_labels_visual.csv'))
    ap.add_argument('--n-sim', type=int, default=8000)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--json')
    a = ap.parse_args()
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    real = real_windows([t.split('=', 1) for t in a.traj], load_labels(a.labels))
    xr, yr = np.stack([d['x'] for d in real]), np.array([d['y'] for d in real])
    videos = sorted({d['video'] for d in real})
    out = dict(real_attempts=len(real), real_made=int(yr.sum()), sim={}, lovo={}, per_attempt=[])
    probs = []
    for seed in range(a.seeds):
        xs, ys = synthetic(a.n_sim, seed)
        model = train(xs, ys, a.epochs, seed, device)
        xv, yv = synthetic(2000, 1000+seed)
        out['sim'].setdefault('synthetic_val', []).append(metrics(yv, predict(model, xv, device)))
        p = predict(model, xr, device); probs.append(p)
        out['sim'].setdefault('real', []).append(metrics(yr, p))
        # leave one video out, real data only
        lp = np.zeros(len(real))
        for v in videos:
            tr = np.array([d['video'] != v for d in real])
            m = train(xr[tr], yr[tr], 200, seed, device)
            lp[~tr] = predict(m, xr[~tr], device)
        out['lovo'].setdefault('real', []).append(metrics(yr, lp))
    mean_p = np.mean(probs, axis=0)
    out['per_attempt'] = [dict(video=d['video'], start=d['start'], made=d['y'], sim_prob_mean=float(q)) for d, q in zip(real, mean_p)]
    for key in ('sim', 'lovo'):
        for part, rows in out[key].items():
            print(key, part, 'accuracy per seed', [round(r['accuracy'], 3) for r in rows],
                  'made P/R', [(r['made_precision'] and round(r['made_precision'], 2), r['made_recall'] and round(r['made_recall'], 2)) for r in rows])
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2), encoding='utf-8')
