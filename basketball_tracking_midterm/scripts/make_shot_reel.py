"""Cut every labelled shot out of the source videos and join them into one video.

Each clip runs from --before seconds before the labelled start to --after seconds after
the labelled end. A caption shows the shot number, source video and frame range, the
labelled result (by eye) and, if --judged is given, the automatic judgement. The rim
box from the labels is drawn. An index CSV gives where each shot starts in the reel.

  python scripts/make_shot_reel.py --output results/shot_reel/shot_scenes.mp4 \
      --judged experiment11=.../shots.json ...
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_shots import load_labels, match

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT.parent/'編集済み元データ'


def caption(img, lines, colour):
    import cv2
    y = 30
    for text, scale in lines:
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)
        y += int(36*scale/.8)


def main():
    import cv2
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', default=str(ROOT/'data/shot_labels_visual.csv'))
    p.add_argument('--videos', nargs='+', default=['experiment10', 'experiment11', 'experiment12'])
    p.add_argument('--judged', nargs='*', default=[], help='video=evaluate_shots.py JSON (per_video events used)')
    p.add_argument('--before', type=float, default=1.)
    p.add_argument('--after', type=float, default=1.)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    labels = load_labels(a.labels)
    judged = {}
    for item in a.judged:
        video, path = item.split('=', 1)
        events = json.loads(Path(path).read_text(encoding='utf-8'))['per_video'][video]['events']
        judged[video] = (events, {k: i for i, k in match(events, labels[video], 15)})
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    writer = None; index = []; n = 0; written = 0
    for video in a.videos:
        cap = cv2.VideoCapture(str(VIDEOS/f'{video}_編集.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if writer is None:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            assert writer.isOpened()
        for k, lab in enumerate(labels[video]):
            n += 1
            a0 = max(0, lab['start']-round(a.before*fps)); a1 = min(total-1, lab['end']+round(a.after*fps))
            auto = None
            if video in judged:
                events, by_label = judged[video]
                auto = events[by_label[k]]['result'] if k in by_label else 'not counted'
            index.append(dict(shot=n, video=video, start_frame=lab['start'], end_frame=lab['end'], label=lab['result'],
                              auto=auto, reel_start_s=round(written/fps, 2), reel_end_s=round((written+a1-a0+1)/fps, 2)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, a0)
            x1, y1, x2, y2 = map(round, lab['rim'])
            for f in range(a0, a1+1):
                ok, img = cap.read()
                if not ok:
                    break
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                colour = (80, 220, 80) if lab['result'] == 'made' else (80, 160, 255)
                lines = [(f"Shot {n}/{sum(len(labels[v]) for v in a.videos)}  {video}  frame {f} ({lab['start']}-{lab['end']})", .7),
                         (f"label: {lab['result'].upper()}", .9)]
                if auto is not None:
                    lines.append((f"auto: {auto.upper()}", .7))
                caption(img, lines, colour)
                writer.write(img); written += 1
            for _ in range(round(fps*.3)):      # short black gap between shots
                writer.write(img*0); written += 1
        cap.release()
    writer.release()
    with open(out.with_suffix('.csv'), 'w', encoding='utf-8-sig', newline='') as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
    print('Saved', out, f'{written/30:.1f}s', len(index), 'shots')


if __name__ == '__main__':
    main()
