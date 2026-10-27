"""Verify the delivery without training or inventing experiment results."""
import ast
from collections import Counter
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,save_json,sha256


def main():
    report={}
    notebook=json.loads((ROOT/'colab/train_yolov5_colab.ipynb').read_text(encoding='utf-8'))
    guarded={'RUN_TRAIN','RUN_RESUME','RUN_VAL','RUN_EXPORT','RUN_FINAL_TEST'}
    found=set()
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            code=''.join(cell['source']);compile(code,f'cell_{i}','exec')
            for node in ast.walk(ast.parse(code)):
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id in guarded:
                            assert node.value.value is False
                            found.add(target.id)
    assert found==guarded
    with zipfile.ZipFile(ROOT/'colab_upload.zip') as z:
        assert z.testzip() is None
        for name in ('src/tracking.py','scripts/train_yolov5.py','constraints-common.txt','README.md'):
            assert z.read('basketball_tracking_midterm/'+name)==(ROOT/name).read_bytes()
        assert not any(n.endswith(('.pt','.mp4')) or '/dataset/' in n for n in z.namelist())
    report['notebook']=dict(code_cells=sum(c['cell_type']=='code' for c in notebook['cells']),
                            all_cells_compile=True,expensive_actions_default_off=True,colab_execution_by_this_check=False)
    report['colab_upload_sha256']=sha256(ROOT/'colab_upload.zip')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        result=unittest.TextTestRunner(stream=captured,verbosity=2).run(suite)
    (ROOT/'data/verification_tests.txt').write_text(captured.getvalue(),encoding='utf-8')
    report['tests']=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    assert result.wasSuccessful(),captured.getvalue()
    run_reports=[]
    for run in sorted((ROOT/'results').rglob('run_config.json')):
        folder=run.parent
        status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
        if status['status']!='complete':continue
        with (folder/'tracking.jsonl').open(encoding='utf-8') as f:tracking=[json.loads(x) for x in f]
        with (folder/'candidates.jsonl').open(encoding='utf-8') as f:candidates=[json.loads(x) for x in f]
        assert [r['global_frame_id'] for r in tracking]==list(range(len(tracking)))
        assert [(r['cycle'],r['frame_id']) for r in candidates]==[(r['cycle'],r['frame_id']) for r in tracking]
        counts=Counter(r['cycle'] for r in tracking)
        for cycle in counts:
            assert [r['frame_id'] for r in tracking if r['cycle']==cycle]==list(range(counts[cycle]))
        m=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        assert m['performance']['processed_frames']==len(tracking)
        assert m['performance']['yolo_calls']==sum(r['yolo_executed'] for r in candidates)
        video=folder/'tracked.mp4'
        if video.exists():
            import cv2
            cap=cv2.VideoCapture(str(video));assert cap.isOpened()
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(tracking)
            cap.release()
        run_reports.append(dict(run=folder.relative_to(ROOT).as_posix(),condition=m['condition'],size=m['model_size'],
            frames=len(tracking),cycles=len(counts),complete_video=m['complete_video'],
            processing_fps=m['performance']['processing_fps'],tsr05=m['tracking']['tsr_iou_05']))
    baseline=ROOT/'results/baseline/s_A_20261026T060623998365Z/metrics.json'
    reevaluated=ROOT/'results/reevaluated_s_A/metrics.json'
    if baseline.is_file() and reevaluated.is_file():
        original=json.loads(baseline.read_text(encoding='utf-8'));repeat=json.loads(reevaluated.read_text(encoding='utf-8'))
        for section in ('detection','tracking'):assert original[section]==repeat[section]
        report['saved_prediction_reevaluation_identical']=True
    report['actual_runs']=run_reports
    report['external_colab_evidence']={}
    for size in ('s','m'):
        export=ROOT/'weights/finetuned'/size
        metadata_path=export/'best.pt.metadata.json'
        if not metadata_path.is_file():continue
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        selection=json.loads((export/'final_selection.json').read_text(encoding='utf-8'))
        assert sha256(export/'best.pt')==metadata['sha256']==selection['weights_sha256']
        assert sha256(export/'training_metadata.json')==metadata['training_metadata_sha256']
        assert sha256(export/'val_metrics.json')==selection['validation_evidence_sha256']
        tests=list(export.glob('test_evaluation_*/metrics.json'))
        for test in tests:
            evidence=json.loads(test.read_text(encoding='utf-8'))
            assert evidence['split']=='test' and evidence['weights_sha256']==metadata['sha256']
        report['external_colab_evidence'][size]=dict(export_hashes_verified=True,
            test_evidence_files=[p.relative_to(ROOT).as_posix() for p in tests],
            note='User ran Colab training/test; verified exported evidence, not re-executed by this check.')
    complete={(r['size'],r['condition']) for r in run_reports if r['complete_video']}
    report['missing_full_video_conditions']=[f'{s}/{c}' for s in ('s','m') for c in 'ABCDEF' if (s,c) not in complete]
    report['not_executed']=['Real optimizer resume after Colab disconnect','15-minute stability trial']
    save_json(ROOT/'data/verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
