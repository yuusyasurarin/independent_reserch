"""Evaluate one Score/Miss/unknown decision per shot, including tracking failures."""
import argparse
import csv
import json
from pathlib import Path


def evaluate(rows):
    counts = dict(tp=0, fp=0, fn=0, tn=0, unknown_score=0, unknown_miss=0)
    ids = set()
    for row in rows:
        shot = row['shot_id']
        if not shot or shot in ids:
            raise ValueError('One unique shot_id per row is required')
        ids.add(shot)
        actual, predicted = row['actual'], row['predicted']
        if actual not in ('score', 'miss') or predicted not in ('score', 'miss', 'unknown'):
            raise ValueError('actual: score/miss; predicted: score/miss/unknown')
        if predicted == 'unknown':
            counts['unknown_'+actual] += 1
        else:
            counts[{('score', 'score'): 'tp', ('score', 'miss'): 'fn', ('miss', 'score'): 'fp', ('miss', 'miss'): 'tn'}[actual, predicted]] += 1
    if not ids:
        raise ValueError('No shots')

    def metric(tp, fp, fn, support):
        return dict(support=support, precision=tp/(tp+fp) if tp+fp else 0. if support else None,
                    recall=tp/support if support else None, f1=2*tp/(2*tp+fp+fn) if support else None)

    tp, fp, fn, tn = (counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    score = metric(tp, fp, fn+counts['unknown_score'], tp+fn+counts['unknown_score'])
    miss = metric(tn, fn, fp+counts['unknown_miss'], tn+fp+counts['unknown_miss'])
    return dict(shots=len(ids), confusion=counts, score=score, miss=miss,
                accuracy=(tp+tn)/len(ids), unknown_rate=(counts['unknown_score']+counts['unknown_miss'])/len(ids),
                macro_f1=(score['f1']+miss['f1'])/2 if score['f1'] is not None and miss['f1'] is not None else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shots', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.shots.open(encoding='utf-8-sig', newline='') as f:
        report = evaluate(list(csv.DictReader(f)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
