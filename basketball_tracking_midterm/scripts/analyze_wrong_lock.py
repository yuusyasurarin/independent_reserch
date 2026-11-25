"""How long the output stays on a wrong object (head, shoe, ...) before recovering.

Reads per_frame.csv of run_fixed_lag.py runs. A "wrong-object frame" is a frame with a
visible ground-truth ball where the tracker outputs a box that does not overlap it at all
(IoU = 0). Consecutive wrong-object frames form an episode; the episode length is the time
until the output returns to the ball, is dropped, or the GT stops being visible.
Visual identity of the wrong object (head vs shoe) is not checked here.

  python scripts/analyze_wrong_lock.py RUN_DIR [RUN_DIR ...] [--json out.json]
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def analyze(run, fps=30.):
    rows = list(csv.DictReader((Path(run)/'per_frame.csv').open(encoding='utf-8')))
    episodes, current = [], 0
    wrong = missing = visible = 0
    for r in rows:
        eligible = r['eligible'] == '1'
        is_wrong = eligible and r['output'] == '1' and float(r['iou'] or 0) == 0
        visible += eligible
        missing += eligible and r['output'] == '0'
        wrong += is_wrong
        if is_wrong:
            current += 1
        elif current:
            episodes.append(current); current = 0
    if current:
        episodes.append(current)
    long = [e for e in episodes if e >= 3]
    return dict(run=Path(run).name, visible_frames=visible, wrong_object_frames=wrong,
                wrong_object_rate=wrong/visible if visible else None, no_output_frames=missing,
                episodes=len(episodes), episodes_ge3=len(long),
                median_episode_s=statistics.median(episodes)/fps if episodes else 0.,
                mean_episode_s=statistics.fmean(episodes)/fps if episodes else 0.,
                longest_episode_s=max(episodes, default=0)/fps,
                frames_in_episodes_ge10=sum(e for e in episodes if e >= 10))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--json')
    a = p.parse_args()
    results = [analyze(r) for r in a.runs if (Path(r)/'per_frame.csv').is_file()]
    for r in results:
        print(f"{r['run']}: wrong {r['wrong_object_frames']}/{r['visible_frames']} ({r['wrong_object_rate']:.1%}), "
              f"episodes {r['episodes']} (>=3f: {r['episodes_ge3']}), median {r['median_episode_s']:.2f}s, "
              f"longest {r['longest_episode_s']:.2f}s, no-output {r['no_output_frames']}")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
