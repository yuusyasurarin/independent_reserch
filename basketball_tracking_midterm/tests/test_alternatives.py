"""Template System 1, ROI windows and the budget analysis script."""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alternatives import TemplateBallPolicy, RoiScheduler, roi_window
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
