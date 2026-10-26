import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, iou, sha256
from src.data import load_gt, validate_dataset
from src.evaluation import detection_metrics, match_detections, tracking_metrics
from src.tracking import TrackingPolicy, associate


def candidate(box=None,score=.9):
    return dict(bbox=box or [10.,10.,20.,20.],confidence=score,class_id=0)


def config():
    return json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))


class TrackingTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.frame=np.zeros((100,100,3),dtype=np.uint8)

    def test_schedule_and_hold_no_future_detections(self):
        policy=TrackingPolicy('D',config(),30)
        self.assertEqual([i for i in range(10) if policy.needs_detection(i)],[0,3,6,9])
        self.assertEqual(policy.update(self.frame,0,[candidate()],True)['state'],'initialized')
        held=policy.update(self.frame,1,[],False)
        self.assertEqual(held['source'],'hold');self.assertIsNone(held['confidence'])
        lost=policy.update(self.frame,3,[],True)
        self.assertIsNone(lost['bbox']);self.assertEqual(lost['state'],'lost')
        self.assertEqual(policy.update(self.frame,6,[candidate()],True)['state'],'reinitialized')

    def test_expiry_prevents_indefinite_old_outputs(self):
        cfg=config();cfg['max_age_s']=.04
        policy=TrackingPolicy('D',cfg,30)
        policy.update(self.frame,0,[candidate()],True)
        self.assertIsNotNone(policy.update(self.frame,1,[],False)['bbox'])
        self.assertIsNone(policy.update(self.frame,2,[],False)['bbox'])

    def test_fast_nonoverlapping_ball_and_gate_rejection(self):
        fast=candidate([30,10,40,20]);far=candidate([900,10,910,20])
        self.assertEqual(iou(fast['bbox'],[10,10,20,20]),0)
        self.assertIs(associate([far,fast],[10,10,20,20],.1,10,300),fast)
        self.assertIsNone(associate([far],[10,10,20,20],.1,10,300))

    def test_ab_use_highest_only_and_do_not_track_id(self):
        for condition in 'AB':
            p=TrackingPolicy(condition,config(),30)
            p.update(self.frame,0,[candidate()],True)
            chosen=p.update(self.frame,1,[candidate(score=.6),candidate([50,50,60,60],.95)],True)
            self.assertEqual(chosen['bbox'],[50,50,60,60]);self.assertIsNone(chosen['track_id'])
            self.assertIsNone(p.update(self.frame,2,[],True)['bbox'])

    def test_kcf_confidence_absent_and_failure_clears(self):
        class KCF:
            def init(self,frame,box):pass
            def update(self,frame):return True,(11,10,10,10)
        p=TrackingPolicy('E',config(),30,kcf_factory=KCF)
        p.update(self.frame,0,[candidate()],True)
        r=p.update(self.frame,1,[],False)
        self.assertEqual(r['source'],'KCF');self.assertIsNone(r['confidence']);self.assertTrue(r['kcf_success_flag'])
        p.kcf.update=lambda frame:(False,(0,0,0,0))
        self.assertIsNone(p.update(self.frame,2,[],False)['bbox'])

    def test_bytetrack_empty_frame_and_identity_continuation(self):
        p=TrackingPolicy('C',config(),30)
        a=p.update(self.frame,0,[candidate()],True)
        b=p.update(self.frame,1,[candidate([11,10,21,20])],True)
        self.assertIsNotNone(a['track_id']);self.assertEqual(a['track_id'],b['track_id'])
        self.assertIsNone(p.update(self.frame,2,[],True)['bbox'])
        c=p.update(self.frame,3,[candidate([12,10,22,20])],True)
        self.assertEqual(c['track_id'],a['track_id'])
        self.assertEqual(c['state'],'reinitialized')


class EvaluationTests(unittest.TestCase):
    def test_wrong_object_is_fp_and_fn_not_success(self):
        flags,fn=match_detections([candidate([50,50,60,60])],[[10,10,20,20]])
        self.assertEqual(flags,[(.9,0)]);self.assertEqual(fn,1)

    def test_one_to_one_duplicate_predictions(self):
        flags,fn=match_detections([candidate(),candidate(score=.8)],[[10,10,20,20]])
        self.assertEqual([p for _,p in flags],[1,0]);self.assertEqual(fn,0)

    def test_detection_excludes_unknown_and_skipped_frames(self):
        statuses=['visible','unknown_position','absent','visible']
        gt=[dict(frame_id=i,status=s,bbox=[10,10,20,20] if s=='visible' else None) for i,s in enumerate(statuses)]
        logs=[dict(frame_id=i,yolo_executed=i!=3,candidates=[candidate()]) for i in range(4)]
        m,_=detection_metrics(gt,logs)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,1,0))
        self.assertEqual(m['eligible_detection_frames'],2);self.assertEqual(m['gt_boxes'],1)

    def test_perfect_ap_and_wrong_object_ap(self):
        gt=[dict(frame_id=0,status='visible',bbox=[10,10,20,20])]
        good=[dict(frame_id=0,yolo_executed=True,candidates=[candidate()])]
        self.assertEqual(detection_metrics(gt,good)[0]['ap50'],1)
        good[0]['candidates']=[candidate([50,50,60,60])]
        m,_=detection_metrics(gt,good)
        self.assertEqual((m['ap50'],m['tp'],m['fp'],m['fn']),(0,0,1,1))

    def test_missing_output_fails_tsr_but_not_zero_cle(self):
        gt=[dict(frame_id=i,status='visible',bbox=[10,10,20,20],segment_id='') for i in range(2)]
        pred=[dict(frame_id=i,timestamp_s=i/30,bbox=[10,10,20,20] if i==0 else None,state='tracking',source='YOLO',track_id=None) for i in range(2)]
        m,_=tracking_metrics(gt,pred,30)
        self.assertEqual(m['tsr_iou_05'],.5);self.assertEqual(m['cle_px']['n'],1)
        self.assertEqual(m['output_rate_visible'],.5)

    def test_gaps_and_segments_break_failure_runs(self):
        gt=[dict(frame_id=i,status='visible' if i!=2 else 'unannotated',bbox=[10,10,20,20] if i!=2 else None,
                 segment_id='a' if i<4 else 'b') for i in range(6)]
        pred=[dict(frame_id=i,timestamp_s=i/30,bbox=None,state='lost',source='none',track_id=None) for i in range(6)]
        m,_=tracking_metrics(gt,pred,30)
        self.assertAlmostEqual(m['longest_failure_at_05_s'],2/30)


class DataTests(unittest.TestCase):
    def make_dataset(self,root):
        from PIL import Image
        for i,split in enumerate(('train','val','test')):
            (root/'images'/split).mkdir(parents=True)
            (root/'labels'/split).mkdir(parents=True)
            Image.new('RGB',(64,64),(i*60,20,50)).save(root/f'images/{split}/example.png')
            (root/f'labels/{split}/example.txt').write_text('0 .5 .5 .2 .2\n')
        (root/'data.yaml').write_text('names: {0: ball}\ntrain: images/train\nval: images/val\ntest: images/test\n')

    def test_missing_vs_empty_labels_no_repair(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input';self.make_dataset(root)
            label=root/'labels/train/example.txt';label.unlink()
            with self.assertRaises(ValueError):validate_dataset(root,Path(d)/'audit',True)
            self.assertFalse(label.exists())
            label.write_text('')
            with self.assertRaises(ValueError):validate_dataset(root,Path(d)/'audit',False)
            self.assertTrue(validate_dataset(root,Path(d)/'audit',True)[0]['valid'])

    def test_class_coordinates_corruption_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input';self.make_dataset(root)
            label=root/'labels/train/example.txt'
            for line in ('1 .5 .5 .2 .2','0 nan .5 .2 .2','0 .99 .5 .2 .2','0 .5 .5 0 .2'):
                label.write_text(line)
                with self.assertRaises(ValueError):validate_dataset(root,Path(d)/'audit',True)
            label.write_text('0 .5 .5 .2 .2')
            (root/'images/val/example.png').write_bytes((root/'images/train/example.png').read_bytes())
            with self.assertRaises(ValueError):validate_dataset(root,Path(d)/'audit',True)

    def test_sparse_gt_statuses_and_frame_times(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gt.csv'
            p.write_text('frame_id,timestamp_s,status,x1,y1,x2,y2\n1,0.033333333,visible,10,10,20,20\n')
            info=dict(frame_count=3,fps=30,width=100,height=100,sha256='unused')
            rows=load_gt(p,info)
            self.assertEqual([r['status'] for r in rows],['unannotated','visible','unannotated'])
            p.write_text('frame_id,timestamp_s,status,x1,y1,x2,y2\n1,1.0,visible,10,10,20,20\n')
            with self.assertRaises(ValueError):load_gt(p,info)

    def test_confirmed_experiment10_status_counts(self):
        p=ROOT/'data/experiment10_gt.csv'
        if not p.exists():self.skipTest('Local experiment10 audit not present in this environment')
        info=json.loads((ROOT/'data/audit/video.json').read_text(encoding='utf-8'))
        gt=load_gt(p,info)
        self.assertEqual(sum(g['status']=='visible' for g in gt),1883)
        self.assertEqual(sum(g['status']=='unknown_position' for g in gt),629)
        self.assertFalse(any(g['status']=='absent' for g in gt))


class IntegrationTests(unittest.TestCase):
    def test_legacy_yolo_pillow_plotting(self):
        from src.common import offline_environment
        from src.compat import enable_legacy_font_size
        offline_environment();enable_legacy_font_size()
        sys.path.insert(0,str(ROOT/'vendor/yolov5'))
        import numpy as np
        import matplotlib
        from utils.plots import Annotator
        font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
        annotator=Annotator(np.zeros((100,100,3),dtype=np.uint8),pil=True,font=str(font),font_size=12)
        annotator.box_label([20,20,60,60],'ball')
        self.assertGreater(annotator.result().sum(),0)

    def test_weight_role_size_class_hash_and_val_settings(self):
        from src.common import YOLO_COMMIT,save_json
        from src.detector import verify_weight
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'model.pt';p.write_bytes(b'test fixture, not a real model')
            cfg=config();cfg['models']['s']['pretrained']=str(p)
            info=dict(role='pretrained',model_size='s',yolov5_commit=YOLO_COMMIT,sha256=sha256(p),class_id=32)
            save_json(p.with_suffix('.pt.metadata.json'),info)
            self.assertEqual(verify_weight(cfg,'s','A')[2],32)
            cfg['models']['s']['pretrained_class']=0
            with self.assertRaises(ValueError):verify_weight(cfg,'s','A')
            cfg['models']['s']['finetuned']=str(root/'missing.pt')
            with self.assertRaises(FileNotFoundError):verify_weight(cfg,'s','B')
            cfg['models']['s']['finetuned']=str(p)
            with self.assertRaises(ValueError):verify_weight(cfg,'s','B')
            info.update(role='finetuned',class_id=0);save_json(p.with_suffix('.pt.metadata.json'),info)
            cfg['selection_split']='val'
            selection=dict(weights_sha256=sha256(p),selection_split='val',operating_settings={k:cfg[k] for k in ('imgsz','confidence','nms_iou','half')})
            save_json(root/'final_selection.json',selection)
            self.assertEqual(verify_weight(cfg,'s','B')[2],0)
            cfg['confidence']=.4
            with self.assertRaises(ValueError):verify_weight(cfg,'s','B')
            cfg['confidence']=selection['operating_settings']['confidence'];p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_weight(cfg,'s','B')

    def test_original_coordinate_roundtrip_letterbox(self):
        from src.common import check_repo,offline_environment
        repo=ROOT/'vendor/yolov5'
        if not repo.exists():self.skipTest('Prepare pinned source via setup_assets.py')
        check_repo(repo);offline_environment();sys.path.insert(0,str(repo))
        import numpy as np
        import torch
        from utils.augmentations import letterbox
        from utils.general import scale_boxes
        image=np.zeros((720,1280,3),dtype=np.uint8)
        resized,ratio,pad=letterbox(image,new_shape=(960,960),auto=False)
        original=np.array([[10,20,44,58]],dtype=float)
        transformed=original.copy()
        transformed[:,[0,2]]=transformed[:,[0,2]]*ratio[0]+pad[0]
        transformed[:,[1,3]]=transformed[:,[1,3]]*ratio[1]+pad[1]
        actual=scale_boxes(resized.shape[:2],torch.tensor(transformed),image.shape[:2]).numpy()
        np.testing.assert_allclose(actual,original,atol=1e-5)

    def test_real_kcf_init_and_update(self):
        import numpy as np
        rng=np.random.default_rng(0)
        frame=np.zeros((100,100,3),dtype=np.uint8)
        frame[10:30,10:30]=rng.integers(0,255,(20,20,3),dtype=np.uint8)
        p=TrackingPolicy('E',config(),30)
        p.update(frame,0,[candidate([10,10,30,30])],True)
        result=p.update(frame,1,[],False)
        self.assertEqual(result['source'],'KCF');self.assertTrue(result['kcf_success_flag'])

    def test_resume_copy_keeps_full_checkpoint_bytes(self):
        from scripts.train_yolov5 import checkpoint_callback
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'weights').mkdir()
            last=root/'weights/last.pt';last.write_bytes(b'checkpoint including optimizer')
            checkpoint_callback(root)(last,3,False,0.,0.)
            self.assertEqual(sha256(last),sha256(root/'weights/resume_latest.pt'))
            self.assertEqual(json.loads((root/'resume_checkpoint.json').read_text())['epoch_zero_based'],3)


if __name__=='__main__':unittest.main()
