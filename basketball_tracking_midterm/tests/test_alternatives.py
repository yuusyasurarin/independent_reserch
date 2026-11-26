"""Template System 1, ROI windows and the budget analysis script."""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alternatives import TemplateBallPolicy, RoiScheduler, roi_window, RecoveryRoiScheduler, with_recovery
from src.adaptive_tracking import AdaptiveBallPolicy
from src.common import center
from scripts.analyze_yolo_budget import analyze
from test_core import candidate, config


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.patch = np.random.default_rng(3).integers(0, 255, (24, 24, 3), dtype=np.uint8)

    def frame(self, x=60, y=50, visible=True):
        f = np.full((200, 300, 3), 40, np.uint8)
        if visible:
            f[y:y+24, x:x+24] = self.patch
        return f

    def start(self):
        p = TemplateBallPolicy(self.cfg, 30)
        f = self.frame()
        self.assertTrue(p.needs_detection(0, f))
        p.update(f, 0, [candidate([60, 50, 84, 74])], True)
        return p

    def test_template_follows_translation_without_yolo(self):
        p = self.start()
        f = self.frame(67, 54)
        self.assertFalse(p.needs_detection(1, f))
        r = p.update(f, 1, [], False)
        self.assertEqual(r['source'], 'LK')  # same output slot as G's System 1
        np.testing.assert_allclose(center(r['bbox']), (79, 66), atol=.6)

    def test_occlusion_requests_detection(self):
        p = self.start()
        f = self.frame(visible=False)
        self.assertTrue(p.needs_detection(1, f))
        self.assertEqual(p.request_reason, 'flow_unreliable')


class RoiTests(unittest.TestCase):
    def test_window_keeps_full_frame_scale_and_stays_inside(self):
        x1, y1, side = roi_window([1270, 5, 1280, 15], 1280, 720, 960, 320)
        self.assertEqual(side, 427)
        self.assertEqual((x1, y1), (1280-427, 0))

    def test_full_frame_every_nth_call_and_when_lost(self):
        class P:
            flow_box = [100, 100, 130, 130]; predicted_box = box = None; request_reason = 'interval'
        s = RoiScheduler(960, 320, 3)
        f = np.zeros((720, 1280, 3), np.uint8)
        kinds = [s.window(P, f) is None for _ in range(6)]
        self.assertEqual(kinds, [True, False, False, True, False, False])
        P.request_reason = 'lost_search'
        self.assertIsNone(s.window(P, f))


class BudgetAnalysisTests(unittest.TestCase):
    def test_full_window_and_reasons(self):
        rows = []
        for i in range(60):
            y = i % 2 == 0
            rows.append(dict(frame_id=i, yolo_executed=y, detection_reason='interval' if y else None,
                             events=['detection_budget_deferred'] if i == 41 else [],
                             detection_budget=15, detection_window_frames=30, bbox=[0, 0, 1, 1]))
        r = analyze(rows)
        self.assertEqual(r['yolo_calls'], 30)
        self.assertEqual(r['frames_with_full_window'], 32)  # every window ending at frame 28..59 holds 15 calls
        self.assertEqual(r['deferred_frames'], 1)
        self.assertEqual(r['call_reasons']['interval'], 30)


if __name__ == '__main__':
    unittest.main()


class RecoveryTests(unittest.TestCase):
    """ROI lock-on recovery options (all off = original behaviour)."""

    def policy(self, **options):
        cfg = config()
        cfg['adaptive'] = dict(cfg.get('adaptive', {}), **options)
        p = with_recovery(AdaptiveBallPolicy)(cfg, 30)
        p.box = [100., 100., 130., 130.]
        p.events = []
        return p

    far = candidate([600, 400, 630, 430], .9)
    near = candidate([102, 101, 132, 131], .3)

    def test_default_needs_two_detections(self):
        p = self.policy()
        self.assertIs(p.choose([self.near, self.far], p.box, 10)[0], self.near)
        self.assertIs(p.choose([self.near, self.far], p.box, 12)[0], self.far)

    def test_instant_switch_on_full_frame_only(self):
        p = self.policy(instant_switch_conf=.5)
        p.roi_call = True
        self.assertIs(p.choose([self.near, self.far], p.box, 10)[0], self.near)
        p.roi_call = False
        self.assertIs(p.choose([self.near, self.far], p.box, 11)[0], self.far)
        self.assertIn('instant_switch', p.events)

    def test_roi_call_keeps_pending_challenger(self):
        for keep, expected in ((False, None), (True, 'far')):
            p = self.policy(keep_challenger_on_roi=keep)
            p.events = []
            p.choose([self.near, self.far], p.box, 10)   # full frame: challenger pending
            p.roi_call = True
            p.choose([self.near], p.box, 11)              # ROI: far ball outside the crop
            p.roi_call = False
            got = p.choose([self.near, self.far], p.box, 12)[0]
            self.assertIs(got, self.far if expected else self.near)


class RecoverySchedulerTests(unittest.TestCase):
    class P:
        flow_box = [100, 100, 130, 130]; predicted_box = box = None
        request_reason = 'interval'; prepared_frame = 0

    def test_time_based_full_frame(self):
        s = RecoveryRoiScheduler(960, 320, 5, full_interval=4)
        f = np.zeros((720, 1280, 3), np.uint8)
        kinds = []
        for i in range(9):
            self.P.prepared_frame = i
            kinds.append(s.window(self.P, f) is None)
        self.assertEqual(kinds, [True, False, False, False, True, False, False, False, True])

    def test_escalates_after_weak_or_resized_roi_detection(self):
        s = RecoveryRoiScheduler(960, 320, 100, escalate=True)
        f = np.zeros((720, 1280, 3), np.uint8)
        for i, result, full_next in ((0, None, None),
                                     (1, dict(source='YOLO_low', bbox=[100, 100, 130, 130]), True),
                                     (3, dict(source='YOLO', bbox=[100, 100, 200, 200]), True),
                                     (5, dict(source='YOLO', bbox=[101, 101, 131, 131]), False)):
            self.P.prepared_frame = i
            win = s.window(self.P, f)
            if result:
                self.assertIsNotNone(win)
                s.after_update(result, True)
                self.P.prepared_frame = i+1
                self.assertEqual(s.window(self.P, f) is None, full_next)
