"""Synthetic invariants for PDF models, time alignment, labeling and metrics."""
import csv
from collections import deque
import json
from pathlib import Path
import tempfile
import threading
import unittest
import cv2
import numpy as np
from experiment_tracking import ExperimentTracker, Settings
from experiment_detectors import Blobs, build_background, clip, merge_rois, Detector
from evaluate_tracking_experiment import evaluate, candidate_recall
from prepare_experiment_data import validate_patches
from run_tracking_experiment import LatestWorker, replay_detection


class FakeKCF:
    def __init__(self, box):
        self.box = box

    def update(self, frame):
        x, y, w, h = self.box
        self.box = x+1, y, w, h
        return True, self.box


def detection(x=10, y=10):
    return dict(balls=[((x, y, 10, 10), .9)], people=[], ring=(40, 10, 20, 5))


def truth(index, visible='visible', shot='a'):
    box = dict(bbox_x=10, bbox_y=10, bbox_w=10, bbox_h=10) if visible in ('visible', 'partial') else {}
    return dict(frame_index=index, timestamp_s=index*.1, shot_id=shot, visibility=visible,
                ring_bbox_x=40, ring_bbox_y=10, ring_bbox_w=20, ring_bbox_h=5, **box)


class TrackingTests(unittest.TestCase):
    def tracker(self, method, **kwargs):
        return ExperimentTracker(Settings(method=method, **kwargs), lambda f, b: FakeKCF(b), (40, 10, 20, 5))

    def test_models_are_distinct(self):
        expected = {'M2': 'missing', 'M3': 'prediction', 'M4': 'kcf', 'M5': 'kcf', 'M6': 'kcf'}
        for method, source in expected.items():
            tracker = self.tracker(method)
            self.assertEqual(tracker.step(None, 0, 0., detection())['source'], 'yolo')
            self.assertEqual(tracker.step(None, 1, .02)['source'], source)
            if method == 'M4':
                self.assertIsNone(tracker.kf)
            if method == 'M3':
                self.assertIsNone(tracker.kcf)

    def test_m1_confirmation(self):
        tracker = self.tracker('M1')
        self.assertEqual(tracker.step(None, 0, 0, detection())['source'], 'missing')
        self.assertEqual(tracker.step(None, 1, .02, detection(11))['source'], 'cnn')
        self.assertEqual(tracker.step(None, 2, .04)['source'], 'missing')

    def test_prediction_expiry_and_reacquisition(self):
        tracker = self.tracker('M3')
        tracker.step(None, 0, 0., detection())
        self.assertEqual(tracker.step(None, 1, .1)['source'], 'prediction')
        self.assertEqual(tracker.step(None, 2, .3)['source'], 'missing')
        self.assertEqual(tracker.step(None, 3, .4, detection(1000))['center_x'], 1005)

    def test_ring_priority_ablation(self):
        for policy, expected in [('R0', 'prediction'), ('R1', 'missing')]:
            tracker = self.tracker('M3', ring_policy=policy)
            tracker.step(None, 0, 0, detection(40))
            self.assertEqual(tracker.step(None, 1, .05)['source'], expected)

    def test_delayed_replay_no_retroactive_output(self):
        tracker = self.tracker('M4')
        history, published = deque(), []
        for i in range(4):
            history.append(dict(index=i, timestamp=i*.02, frame=None, before=tracker.snapshot(), detection=None))
            published.append(tracker.step(None, i, i*.02))
        self.assertEqual(replay_detection(tracker, history, 1, detection()), (True, 3))
        self.assertTrue(all(r['source'] == 'missing' for r in published))
        self.assertEqual(tracker.step(None, 4, .08)['center_x'], 18)
        self.assertEqual(replay_detection(tracker, history, 99, detection()), (False, 0))

    def test_player_not_ball(self):
        tracker = self.tracker('M2', players='P2')
        tracker.step(None, 0, 0, detection())
        row = tracker.step(None, 1, .1, dict(balls=[], people=[(5, 0, 30, 80)], ring=None))
        self.assertEqual(row['center_x'], '')
        self.assertNotEqual(row['player_ids'], '')

    def test_latest_worker(self):
        started, release = threading.Event(), threading.Event()
        def detect(frame, timestamp, context):
            if timestamp == 0:
                started.set()
                release.wait(5)
            return timestamp
        worker = LatestWorker(detect)
        try:
            worker.submit((0, 0, 0, None, {}))
            self.assertTrue(started.wait(5))
            worker.submit((1, 1, 0, None, {}))
            worker.submit((2, 2, 0, None, {}))
            release.set()
            worker.close()
            self.assertEqual(worker.replaced, 1)
            self.assertEqual([worker.results.get()[0] for _ in range(2)], [0, 2])
        finally:
            release.set()
            if worker.thread.is_alive():
                worker.close()


class EvaluationTests(unittest.TestCase):
    def test_score_unknown_is_not_dropped(self):
        from evaluate_score_events import evaluate as score
        result = score([dict(shot_id='a', actual='score', predicted='score'),
                        dict(shot_id='b', actual='score', predicted='unknown'),
                        dict(shot_id='c', actual='miss', predicted='unknown')])
        self.assertEqual(result['accuracy'], 1/3)
        self.assertEqual(result['score']['f1'], 2/3)
        self.assertEqual(result['confusion']['unknown_miss'], 1)

    def test_missing_is_failure(self):
        result = evaluate({i: truth(i) for i in range(2)}, {0: truth(0)})
        self.assertEqual(result['all']['tsr'], .5)
        self.assertEqual(result['all']['auc_mean_iou'], .5)
        self.assertEqual(result['all']['cle_px']['n'], 1)
        self.assertEqual(result['all']['cle_px']['mean'], 0)
        self.assertEqual(result['all']['valid_coverage'], .5)

    def test_gt_ring_normalization(self):
        pred = truth(0)
        pred['bbox_x'], pred['ring_bbox_w'] = 11, 100
        result = evaluate({0: truth(0)}, {0: pred}, epsilon=.05)
        self.assertAlmostEqual(result['all']['ncle']['mean'], .05)
        self.assertEqual(result['all']['valid_coverage'], 1)
        self.assertGreater(result['all']['relative_error']['mean'], 0)

    def test_recovery_stable_first_success_time(self):
        result = evaluate({i: truth(i) for i in range(12)}, {i: truth(i) for i in range(3, 12)})
        event = result['events']['lost']
        self.assertEqual(event['events'], 1)
        self.assertAlmostEqual(event['details'][0]['recovery_s'], .3)
        self.assertEqual(event['recovery_rate'], 1)

    def test_censored_clip(self):
        result = evaluate({i: truth(i) for i in range(3)}, {})
        self.assertEqual(result['events']['lost']['censored'], 1)
        self.assertIsNone(result['events']['lost']['recovery_rate'])

    def test_occlusion_and_revisibility(self):
        gt = {i: truth(i, 'occluded' if i < 3 else 'visible') for i in range(12)}
        result = evaluate(gt, {i: truth(i) for i in range(3, 12)})
        self.assertEqual(result['all']['eligible_frames'], 9)
        self.assertEqual(result['events']['revisibility']['details'][0]['recovery_s'], 0)
        self.assertEqual(result['events']['lost']['events'], 0)

    def test_occluded_gt_rejected(self):
        row = truth(0)
        row['visibility'] = 'occluded'
        with self.assertRaises(ValueError):
            evaluate({0: row}, {})

    def test_no_cross_shot_events(self):
        gt = {0: truth(0), 1: truth(1), 3: truth(3, shot='b'), 4: truth(4, shot='b')}
        self.assertEqual(evaluate(gt, {})['events']['lost']['events'], 0)

    def test_candidate_recalls(self):
        d = dict(frame_index=0, blob_rois=[(14, 14, 2, 2)], proposals=[(0, 0, 100, 100)])
        result = candidate_recall({0: truth(0)}, [d])
        self.assertEqual(result['blob_rois']['center_recall'], 1)
        self.assertEqual(result['blob_rois']['bbox_90percent_recall'], 0)
        self.assertEqual(result['proposals']['bbox_90percent_recall'], 1)


class DataTests(unittest.TestCase):
    def test_session_and_identical_image_leakage_rejected(self):
        from experiment_detectors import save_image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_image(root/'a.png', np.zeros((8, 8, 3), np.uint8))
            save_image(root/'b.png', np.ones((8, 8, 3), np.uint8))
            path = root/'patches.csv'
            def write(second_session, second_path):
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['path', 'session_id', 'split', 'label', 'reviewed'])
                    writer.writeheader()
                    writer.writerow(dict(path='a.png', session_id='one', split='train', label='ball', reviewed=1))
                    writer.writerow(dict(path=second_path, session_id=second_session, split='val', label='ball', reviewed=1))
            write('one', 'b.png')
            with self.assertRaisesRegex(ValueError, 'Session leakage'):
                validate_patches(path, require_splits=False)
            write('two', 'a.png')
            with self.assertRaisesRegex(ValueError, 'Identical patch'):
                validate_patches(path, require_splits=False)

    def test_background_tiny_blob(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            writer = cv2.VideoWriter(str(root/'bg.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 64))
            for _ in range(4):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            build_background(root/'bg.avi', root/'bg.png', 2, 2)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[30:33, 30:33] = 255
            self.assertEqual(Blobs(root/'bg.png', min_area=4).boxes(frame), [(30, 30, 3, 3)])
            self.assertEqual(json.loads((root/'bg.json').read_text())['frames'], 4)

    def test_unreviewed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'patches.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'session_id', 'split', 'label', 'reviewed'])
                writer.writeheader()
                writer.writerow(dict(path='x.png', session_id='s', split='train', label='ball', reviewed=0))
            with self.assertRaisesRegex(ValueError, 'Unreviewed'):
                validate_patches(path)

    def test_roi_merge_clip(self):
        self.assertEqual(clip((-5, -5, 10, 10), (20, 20, 3)), (0, 0, 5, 5))
        self.assertEqual(merge_rois([(0, 0, 10, 10), (8, 0, 10, 10)]), [(0, 0, 18, 10)])

    def test_crop_coordinate_restore(self):
        from types import SimpleNamespace
        import torch
        class Model:
            def predict(self, frames, **kwargs):
                return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[1., 2., 6., 8.]]),
                        conf=torch.tensor([.9]), cls=torch.tensor([0.]))) for _ in frames]
        detector = Detector.__new__(Detector)
        detector.args = SimpleNamespace(roi_batch=4, image_size=64, confidence=.1)
        detector.device, detector.yolo = 'cpu', Model()
        predictions, count = detector._yolo(np.zeros((100, 100, 3), np.uint8), [(20, 30, 10, 10)], [0])
        self.assertEqual(predictions[0][0], (21., 32., 5., 6.))
        self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
