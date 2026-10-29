import tempfile
import unittest
from pathlib import Path
import zipfile

from tracking_eval import evaluate, load_annotations, select_target


class EvaluationTests(unittest.TestCase):
    def test_missing_predictions_are_failures_but_not_zero_cle(self):
        gt = [dict(frame_index=i,status='bbox_labeled',bbox=[10,10,10,10]) for i in range(2)]
        pred = [dict(frame_index=0,bbox=[10,10,10,10],track_id=1,confidence=.9),
                dict(frame_index=1,bbox=None,track_id=None,confidence=None)]
        result,_ = evaluate(gt,pred,30)
        self.assertEqual(result['tsr'],.5)
        self.assertEqual(result['mean_iou'],.5)
        self.assertEqual(result['output_coverage'],.5)
        self.assertEqual(result['cle_px']['n'],1)
        self.assertEqual(result['cle_px']['mean'],0)

    def test_invisible_unknown_and_failure_run_boundaries(self):
        statuses = ['bbox_labeled']*3+['not_visible','unknown']+['bbox_labeled']*2
        gt = [dict(frame_index=i,status=s,bbox=[0,0,10,10] if s=='bbox_labeled' else None) for i,s in enumerate(statuses)]
        pred = [dict(frame_index=i,bbox=[0,0,10,10] if i==3 else None,track_id=1 if i==3 else None,confidence=.9 if i==3 else None) for i in range(7)]
        result,_ = evaluate(gt,pred,30)
        self.assertEqual(result['evaluated_bbox_frames'],5)
        self.assertEqual(result['output_on_not_visible_frames'],1)
        self.assertEqual(result['unknown_frames'],1)
        self.assertEqual(result['failure_runs_ge3'],1)
        self.assertEqual(result['longest_failure_frames'],3)

    def test_selector_keeps_id_and_rejects_unconfirmed_detections(self):
        a = dict(track_id=3,confidence=.4,bbox=[100,100,10,10])
        b = dict(track_id=5,confidence=.9,bbox=[0,0,10,10])
        self.assertIs(select_target([a,b],3),a)
        self.assertIs(select_target([a,b],None),b)
        self.assertIsNone(select_target([dict(track_id=None,confidence=.99)],None))

    def make_archive(self,path,label='0 0.5 0.5 0.2 0.2\n',listed=2):
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('data.yaml','names: {0: ball}\n')
            z.writestr('train.txt',''.join(f'data/images/train/frame_{i:06d}.png\n' for i in range(listed)))
            z.writestr('labels/train/frame_000001.txt',label)

    def test_import_is_zero_based_and_missing_policy_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.zip';self.make_archive(p)
            gt,report=load_annotations(p,2,1280,720)
            self.assertEqual(gt[0]['status'],'unknown')
            for actual,expected in zip(gt[1]['bbox'],[512.,288.,256.,144.]):
                self.assertAlmostEqual(actual,expected)
            self.assertEqual(report['missing_files'],[0])
            gt,_=load_annotations(p,2,1280,720,'not_visible')
            self.assertEqual(gt[0]['status'],'not_visible')

    def test_reject_invalid_and_multiple_boxes_and_frame_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.zip'
            for bad in ('0 nan .5 .1 .1\n','1 .5 .5 .1 .1\n','0 .5 .5 .1 .1\n0 .6 .6 .1 .1\n','0 .99 .5 .5 .1\n'):
                self.make_archive(p,bad)
                with self.assertRaises(ValueError):load_annotations(p,2,1280,720)
            self.make_archive(p,listed=3)
            with self.assertRaises(ValueError):load_annotations(p,2,1280,720)

    def test_missing_prediction_rows_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([dict(frame_index=0,status='unknown',bbox=None)],[],30)


if __name__ == '__main__':
    unittest.main()
