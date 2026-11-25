"""How often does the adaptive policy (F/G) run into its YOLO call budget?

Reads a run folder (or a .jsonl file) and reports
  * calls per second and how close that is to the budget (default 15 calls / 30 frames)
  * the fraction of frames whose trailing 1-second window is already full ("at the cap")
  * frames where a wanted detection was refused by the budget, and how many separate episodes
  * why YOLO was called (interval / flow_unreliable / motion_change / lost_search)
  * the gap between consecutive YOLO calls

Input, in order of preference inside a run folder:
  online.jsonl   (run_fixed_lag.py; causal System 1/2 rows with events -> full report)
  tracking.jsonl (run_experiment.py F runs keep events; run_fixed_lag.py's has none)
  frames.jsonl   (realtime_tracking.py; only yolo_executed -> cap/gap part only)

Usage:
  python scripts/analyze_yolo_budget.py results/condition_g/fresh/s_G_f_bridge_960_cuda0_<time>
  python scripts/analyze_yolo_budget.py run_a run_b --json reports/budget.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path

REASONS = ('interval', 'flow_unreliable', 'motion_change', 'lost_search')


def pick(path):
    path = Path(path)
    if path.is_file():
        return path
    for name in ('online.jsonl', 'tracking.jsonl', 'frames.jsonl'):
        if (path/name).is_file():
            return path/name
    raise FileNotFoundError(f'No online.jsonl / tracking.jsonl / frames.jsonl in {path}')


def read(path):
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r['frame_id'])
    return rows


def analyze(rows, fps=30., budget=None, window=None):
    n = len(rows)
    yolo = [bool(r.get('yolo_executed', r.get('detection_reason') is not None)) for r in rows]
    budget = budget or next((r['detection_budget'] for r in rows if r.get('detection_budget')), 15)
    window = window or next((r['detection_window_frames'] for r in rows if r.get('detection_window_frames')), round(fps))
    has_events = any(r.get('events') for r in rows)
    # Same counting rule as AdaptiveBallPolicy: calls in frames (i-window, i].
    count, full, trailing = 0, 0, []
    for i, y in enumerate(yolo):
        count += y
        if i >= window:
            count -= yolo[i-window]
        trailing.append(count)
        full += count >= budget
    calls = sum(yolo)
    ids = [rows[i]['frame_id'] for i, y in enumerate(yolo) if y]
    gaps = Counter(b-a for a, b in zip(ids, ids[1:]))
    out = dict(frames=n, seconds=n/fps, yolo_calls=calls, calls_per_s=calls/(n/fps) if n else 0.,
               budget_calls=budget, window_frames=window, budget_per_s=budget*fps/window,
               utilisation=(calls/(n/fps))/(budget*fps/window) if n else 0.,
               frames_with_full_window=full, frames_with_full_window_rate=full/n if n else 0.,
               gap_frames_histogram=dict(sorted(gaps.items())),
               max_calls_in_any_window=max(trailing, default=0))
    if has_events:
        deferred = [any(e == 'detection_budget_deferred' for e in r.get('events', [])) for r in rows]
        episodes = sum(1 for i, d in enumerate(deferred) if d and (i == 0 or not deferred[i-1]))
        reasons = Counter(r.get('detection_reason') for r, y in zip(rows, yolo) if y)
        wanted = Counter()
        for r, d in zip(rows, deferred):
            if d:
                ev = r.get('events', [])
                wanted['motion_change' if 'motion_change' in ev else
                       'lost_search' if r.get('bbox') is None else 'flow_unreliable_or_interval'] += 1
        out.update(deferred_frames=sum(deferred), deferred_rate=sum(deferred)/n if n else 0.,
                   deferred_episodes=episodes,
                   deferred_while=dict(wanted),
                   call_reasons={k: reasons.get(k, 0) for k in REASONS} |
                                {k: v for k, v in reasons.items() if k not in REASONS},
                   sources=dict(Counter(r.get('source') for r in rows)))
    else:
        out['note'] = 'No per-frame events in this file: deferrals and call reasons are unavailable (use online.jsonl).'
    return out


def show(name, r):
    print(f'\n== {name}')
    print(f"frames {r['frames']} ({r['seconds']:.1f}s)  YOLO {r['yolo_calls']}  "
          f"{r['calls_per_s']:.2f}/s of {r['budget_per_s']:.1f}/s budget ({r['utilisation']*100:.0f}%)")
    print(f"trailing 1s window already full: {r['frames_with_full_window']} frames "
          f"({r['frames_with_full_window_rate']*100:.1f}%)")
    if 'deferred_frames' in r:
        print(f"detection wanted but refused by budget: {r['deferred_frames']} frames "
              f"({r['deferred_rate']*100:.1f}%), {r['deferred_episodes']} episodes; {r['deferred_while']}")
        total = max(1, r['yolo_calls'])
        print('call reasons: ' + ', '.join(f'{k} {v} ({v/total*100:.0f}%)' for k, v in r['call_reasons'].items()))
    else:
        print(r['note'])
    print('gap between YOLO calls (frames: count):', r['gap_frames_histogram'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', help='run folders or jsonl files')
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--json', help='also save all results here')
    args = p.parse_args()
    results = {}
    for run in args.runs:
        path = pick(run)
        results[str(path)] = r = analyze(read(path), args.fps)
        show(path, r)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
