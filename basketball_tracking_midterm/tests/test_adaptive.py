"""Causal tracking behavior on synthetic motion, disappearance and distractors."""
import unittest
from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.adaptive_tracking import AdaptiveBallPolicy
from src.common import center
from test_core import candidate, config


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.cfg=config()
        self.patch=np.random.default_rng(7).integers(0,255,(24,24,3),dtype=np.uint8)

    def frame(self,x=30,y=40,visible=True):
        frame=np.zeros((160,240,3),np.uint8)
        if visible:frame[y:y+24,x:x+24]=self.patch
        return frame

    def initialize(self):
        p=AdaptiveBallPolicy(self.cfg,30)
        f=self.frame()
        self.assertTrue(p.needs_detection(0,f))
        p.update(f,0,[candidate([30,40,54,64])],True)
        return p

    def test_translation_tracks_observed_motion_without_yolo(self):
        p=self.initialize()
        f=self.frame(35,43)
        self.assertFalse(p.needs_detection(1,f))
        r=p.update(f,1,[],False)
        self.assertEqual(r['source'],'LK')
        np.testing.assert_allclose(center(r['bbox']),(47,55),atol=.7)
        self.assertIsNone(r['confidence'])

    def test_disappearance_requests_detection_and_prediction_expires(self):
        p=self.initialize()
        f=self.frame(visible=False)
        self.assertTrue(p.needs_detection(1,f))
        r=p.update(f,1,[],True)
        self.assertEqual(r['source'],'motion_prediction')
        for i in range(2,5):
            detected=p.needs_detection(i,f)
            r=p.update(f,i,[],detected)
        self.assertIsNone(r['bbox'])

    def test_low_score_never_initializes_or_extends_high_confidence_ttl(self):
        p=AdaptiveBallPolicy(self.cfg,30)
        f=self.frame()
        p.needs_detection(0,f)
        self.assertIsNone(p.update(f,0,[candidate([30,40,54,64],.15)],True)['bbox'])
        p=self.initialize()
        for i in range(1,12):
            p.needs_detection(i,f)
            r=p.update(f,i,[candidate([30,40,54,64],.15)],True)
        self.assertIsNone(r['bbox'])

    def test_distant_distractor_requires_confirmation_then_releases_lock(self):
        p=self.initialize()
        local=candidate([30,40,54,64],.3)
        far=candidate([180,40,204,64],.9)
        f=self.frame()
        p.needs_detection(3,f)
        r=p.update(f,3,[local,far],True)
        self.assertEqual(r['bbox'],local['bbox'])
        self.assertIn('global_candidate_pending',r['events'])
        p.needs_detection(6,f)
        r=p.update(f,6,[local,far],True)
        self.assertEqual(r['bbox'],far['bbox'])
        self.assertIn('global_reacquisition',r['events'])
        self.assertEqual(r['track_id'],2)

    def test_rolling_detection_budget_even_under_constant_flow_failure(self):
        p=AdaptiveBallPolicy(self.cfg,30)
        f=self.frame(visible=False)
        calls=deque()
        for i in range(100):
            detected=p.needs_detection(i,f)
            if detected:calls.append(i)
            p.update(f,i,[candidate([30,40,54,64])] if detected else [],detected)
            while calls and i-calls[0]>=30:calls.popleft()
            self.assertLessEqual(len(calls),15)

    def test_output_leaves_frame_without_clamping_to_false_location(self):
        p=self.initialize()
        p.velocity[:]=300
        f=self.frame(visible=False)
        p.needs_detection(1,f)
        r=p.update(f,1,[],False)
        self.assertIsNone(r['bbox'])
        self.assertIn('out_of_bounds',r['events'])

    def test_high_confidence_expiry_even_when_optical_flow_is_valid(self):
        p=self.initialize()
        for i in range(1,12):
            f=self.frame()
            p.needs_detection(i,f)
            r=p.update(f,i,[],False)
        self.assertIsNone(r['bbox'])

    def test_current_frame_required_and_invalid_settings_rejected(self):
        p=AdaptiveBallPolicy(self.cfg,30)
        with self.assertRaises(ValueError):p.needs_detection(0)
        self.cfg['adaptive']={'max_yolo_hz':0}
        with self.assertRaises(ValueError):AdaptiveBallPolicy(self.cfg,30)


if __name__=='__main__':unittest.main()
