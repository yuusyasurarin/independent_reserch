"""Read-only audit of the original dataset/video; output only into this project."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, environment, resolve, save_json
from src.data import import_confirmed_gt, inspect_video, validate_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='../colab_ball_yolo/dataset')
    p.add_argument('--video',default='../編集済み元データ/experiment10_編集.mp4')
    p.add_argument('--source-gt-bundle',default='../colab_ball_tracking/bundles/experiment10')
    p.add_argument('--allow-background',action='store_true',help='Existing 161 explicit empty labels are documented confirmed background')
    a=p.parse_args()
    dataset=resolve(a.dataset)
    report,_=validate_dataset(dataset,ROOT/'data/audit',a.allow_background)
    reference_pixels=set()
    provenance=dataset/'provenance/manifest.csv'
    if provenance.is_file():
        with provenance.open(encoding='utf-8-sig') as f:
            reference_pixels={r['pixel_sha256'] for r in csv.DictReader(f)}
    video=inspect_video(resolve(a.video),True,reference_pixels)
    source_report=dataset/'provenance/preparation.json'
    sources=json.loads(source_report.read_text(encoding='utf-8'))['sources'] if source_report.is_file() else []
    overlap=[s['name'] for s in sources if s['sha256']==video['sha256']]
    source_hashes={s['name']:s['sha256'] for s in sources}
    confirmation_path=ROOT/'configs/data_provenance.json'
    confirmation=json.loads(confirmation_path.read_text(encoding='utf-8')) if confirmation_path.is_file() else None
    independent=bool(confirmation and confirmation['independent_recording']
                     and confirmation['evaluation_video_sha256']==video['sha256']
                     and confirmation['training_source_sha256']==source_hashes and not overlap
                     and not video['exact_source_pixel_matches'])
    save_json(ROOT/'data/audit/video.json',video)
    save_json(ROOT/'data/audit/overlap.json',dict(evaluation_video_sha256=video['sha256'],source_video_hash_matches=overlap,
        exact_source_pixel_match_frames=video['exact_source_pixel_matches'],dataset_sources=report['source_names'],
        source_video_hashes=source_hashes,user_confirmation=confirmation,
        independent_recording_verified=independent,
        independence_basis='User confirmed separate recordings and no repeated play; hashes bind this confirmation to the audited files.' if independent else None,
        conclusion='Separate evaluation recording confirmed by user; scope is one 83.733-second clip.' if independent else
                   'Different source filenames/hashes and exact pixels alone do not establish different original recordings.'))
    if a.source_gt_bundle:
        import_confirmed_gt(resolve(a.source_gt_bundle),ROOT/'data/experiment10_gt.csv')
    save_json(ROOT/'data/audit/environment.json',environment())
    print(json.dumps(dict(dataset=report['counts'],video=video,source_video_hash_matches=overlap),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
