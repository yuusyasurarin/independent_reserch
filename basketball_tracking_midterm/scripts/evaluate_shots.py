"""Shot-judgement metrics against data/shot_labels_visual.csv.

Each judged event is matched one-to-one to a labelled attempt of the same video when
their frame ranges overlap (labels widened by --tolerance frames).

Metrics
  attempt precision   matched events / judged events      (judged shots that were real attempts)
  attempt recall      matched labels / labelled attempts  (real attempts the system counted)
  attempt F1          harmonic mean of the two
  outcome accuracy    matched events whose made/miss equals the label
  made precision      judged 'made' that were labelled made
  made recall         labelled made that were judged 'made' (uncounted made shots are misses)
  end-to-end accuracy labelled attempts counted AND judged correctly / labelled attempts
  score error         |judged made - labelled made| and |judged attempts - labelled| per video

  python scripts/evaluate_shots.py --events experiment11=path/frames.jsonl experiment12=... [--json out]
Events are produced on the fly with src/shot.py from each trajectory file (frames.jsonl,
tracking.csv or a GT csv), using the rim box written in the labels.
"""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.judge_shots import centres
from src.shot import ShotJudge

ROOT = Path(__file__).resolve().parents[1]


def load_labels(path):
    labels = defaultdict(list)
    for r in csv.DictReader(open(path, encoding='utf-8')):
        labels[r['video']].append(dict(start=int(r['start_frame']), end=int(r['end_frame']), result=r['result'],
                                       rim=[float(r[k]) for k in ('rim_x1', 'rim_y1', 'rim_x2', 'rim_y2')]))
    return labels


def judge(trajectory, rim, fps=30., **options):
    j = ShotJudge(rim, fps, **options)
    for frame_id, c in centres(trajectory):
        j.update(frame_id, c)
    return [dict(start=e['start_frame'], end=e['frame_id'], result=e['result'], reason=e['reason']) for e in j.events]


def match(events, labels, tolerance):
    pairs, used = [], set()
    for i, e in enumerate(events):
        best = None
        for k, l in enumerate(labels):
            if k in used or e['end'] < l['start']-tolerance or e['start'] > l['end']+tolerance:
                continue
            overlap = min(e['end'], l['end'])-max(e['start'], l['start'])
            if best is None or overlap > best[0]:
                best = (overlap, k)
        if best is not None:
            used.add(best[1]); pairs.append((i, best[1]))
    return pairs


def score(per_video):
    n_events = sum(len(v['events']) for v in per_video.values())
    n_labels = sum(len(v['labels']) for v in per_video.values())
    matched = correct = made_pred = made_true = made_hit = 0
    for v in per_video.values():
        made_pred += sum(e['result'] == 'made' for e in v['events'])
        made_true += sum(l['result'] == 'made' for l in v['labels'])
        for i, k in v['pairs']:
            matched += 1
            ok = v['events'][i]['result'] == v['labels'][k]['result']
            correct += ok
            made_hit += ok and v['labels'][k]['result'] == 'made'
    p = matched/n_events if n_events else None
    r = matched/n_labels if n_labels else None
    return dict(labelled_attempts=n_labels, judged_events=n_events, matched=matched,
                attempt_precision=p, attempt_recall=r,
                attempt_f1=2*p*r/(p+r) if p and r else 0., outcome_accuracy=correct/matched if matched else None,
                made_precision=made_hit/made_pred if made_pred else None, made_recall=made_hit/made_true if made_true else None,
                end_to_end_accuracy=correct/n_labels if n_labels else None,
                score_error={k: dict(made=abs(sum(e['result'] == 'made' for e in v['events'])-sum(l['result'] == 'made' for l in v['labels'])),
                                     attempts=abs(len(v['events'])-len(v['labels']))) for k, v in per_video.items()})


def evaluate(pairs_arg, labels_path, tolerance=15, **options):
    labels = load_labels(labels_path)
    per_video = {}
    for video, trajectory in pairs_arg:
        lab = labels[video]
        events = judge(trajectory, lab[0]['rim'], **options)
        per_video[video] = dict(events=events, labels=lab, pairs=match(events, lab, tolerance))
    return score(per_video), per_video


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--events', nargs='+', required=True, help='video=trajectory_file')
    p.add_argument('--labels', default=str(ROOT/'data/shot_labels_visual.csv'))
    p.add_argument('--tolerance', type=int, default=15)
    p.add_argument('--json')
    a = p.parse_args()
    summary, per_video = evaluate([x.split('=', 1) for x in a.events], a.labels, a.tolerance)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(summary=summary, per_video=per_video), indent=2), encoding='utf-8')
