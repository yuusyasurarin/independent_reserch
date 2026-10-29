"""Build a standalone test package without changing source annotations or other experiments."""
import argparse
from pathlib import Path
import json
import shutil
import zipfile

from tracking_eval import load_annotations, save_csv, save_json, sha256

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('experiment09', 'experiment10'), default='experiment09')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--confirm-not-visible', action='store_true',
                        help='Use only after the annotator confirms ALL unlabeled frames are not visible')
    args = parser.parse_args()
    experiment = args.experiment
    number = experiment[-2:]
    source_dir = PROJECT/'datasets/tracking_test_experiment09_10'
    default_annotation = (source_dir/'ball_only_アノテーション_test09.zip' if experiment=='experiment09'
                          else source_dir/'experiment10/annotations/ball_tracking_experiment10.zip')
    annotation = args.annotations or default_annotation
    prep = json.loads((source_dir/'preparation.json').read_text(encoding='utf-8'))
    info = next(v for v in prep['videos'] if v['name']==experiment)
    video = PROJECT/f'編集済み元データ/{experiment}_編集.mp4'
    if sha256(video)!=info['sha256']:
        raise ValueError('Video changed since full-frame preparation')
    policy = 'not_visible' if experiment=='experiment09' or args.confirm_not_visible else 'unknown'
    gt,report=load_annotations(annotation,info['frame_count'],info['width'],info['height'],policy)
    if experiment=='experiment09' and (report['missing_files']!=list(range(18)) or report['bbox_labeled']!=447 or report['empty_label_files']):
        raise ValueError('Annotations changed: re-confirm missing-frame policy before rebuilding')
    # Keep the original experiment09 package and local manifest intact when building experiment10.
    destination = ROOT if experiment=='experiment09' else ROOT/'bundles'/experiment
    data = destination/'data'
    data.mkdir(parents=True,exist_ok=True)
    shutil.copy2(video,data/f'{experiment}.mp4')
    shutil.copy2(annotation,data/f'annotations{number}.zip')
    confirmation = ('User confirmed frames 0-17 reviewed and ball not visible (2026-10-14). No occlusion/out-of-frame subtype asserted.'
                    if experiment=='experiment09' else
                    'Annotator confirmed all unlabeled frames reviewed and ball not visible; recorded via --confirm-not-visible.'
                    if args.confirm_not_visible else
                    'Unlabeled frames remain unknown; no invisibility assumption has been made.')
    manifest = dict(experiment=experiment,
                    video={**{k:info[k] for k in ('sha256','fps','frame_count','width','height','duration_s')},'file':f'data/{experiment}.mp4'},
                    annotations=dict(file=f'data/annotations{number}.zip',sha256=sha256(annotation),missing_policy=policy,
                                     original_filename=annotation.name,
                                     confirmation=confirmation),
                    purpose=f'Held-out {experiment}; inference and evaluation only, no training')
    save_json(destination/'manifest.json',manifest)
    save_json(destination/'annotation_report.json',report)
    rows=[]
    for r in gt:
        rows.append(dict(frame_index=r['frame_index'],timestamp_s=r['frame_index']/info['fps'],status=r['status'],
                         **{f'bbox_{k}':r['bbox'][j] if r['bbox'] else None for j,k in enumerate(('x','y','w','h'))}))
    save_csv(data/'ground_truth_ball.csv',rows)
    make_preview(video,gt,destination/'annotation_preview.jpg')
    files = ['tracking_eval.py','run_tracking_test.py','bytetrack.yaml','requirements-colab.txt',
             'test_tracking_eval.py','manifest.json','annotation_report.json','README.md',
             f'data/{experiment}.mp4',f'data/annotations{number}.zip','data/ground_truth_ball.csv','annotation_preview.jpg']
    for name in ('tracking_eval.py','run_tracking_test.py','bytetrack.yaml','requirements-colab.txt','test_tracking_eval.py','README.md'):
        if destination != ROOT:
            shutil.copy2(ROOT/name,destination/name)
    archive=ROOT/f'ball_tracking_{experiment}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for name in files:
            z.write(destination/name,'ball_tracking_colab/'+name)
    checksum = ROOT/'bundle_sha256.txt' if experiment=='experiment09' else ROOT/f'ball_tracking_{experiment}.sha256'
    checksum.write_text(sha256(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps(dict(bundle=str(archive),bytes=archive.stat().st_size,frames=report['frames'],
                         bbox_labeled=report['bbox_labeled'],missing_files=len(report['missing_files']),
                         missing_policy=policy),ensure_ascii=False,indent=2))


def make_preview(video,gt,output):
    import cv2
    import numpy as np
    canvas=np.full((408,1280,3),240,dtype=np.uint8)
    cap=cv2.VideoCapture(str(video))
    try:
        indices = (0,17,18,80,161,242,323,440) if len(gt)==465 else np.linspace(0,len(gt)-1,8,dtype=int)
        for slot,index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok:raise ValueError(f'Cannot read preview frame {index}')
            if gt[index]['bbox']:
                x,y,w,h=gt[index]['bbox']
                cv2.rectangle(frame,(round(x),round(y)),(round(x+w),round(y+h)),(0,220,0),3)
            x,y=(slot%4)*320,(slot//4)*204
            canvas[y:y+180,x:x+320]=cv2.resize(frame,(320,180))
            cv2.putText(canvas,f"frame {index}: {gt[index]['status']}",(x+4,y+198),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    finally:
        cap.release()
    cv2.imencode('.jpg',canvas)[1].tofile(str(output))


if __name__=='__main__':
    main()
