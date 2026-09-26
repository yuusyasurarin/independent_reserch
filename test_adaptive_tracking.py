"""Deterministic policy and evaluation regressions; no model download required."""
import unittest
import numpy as np
from adaptive_tracking import AdaptiveTracker, Config, MotionFilter
from evaluate_adaptive_tracking import evaluate


class FakeKCF:
    def __init__(self, box):
        self.box = box
    def update(self, frame):
        return True, self.box


class PolicyTests(unittest.TestCase):
    def tracker(self, detections, ring=(500, 200, 50, 10), **kwargs):
        iterator = iter(detections)
        detector = lambda frame: next(iterator)
        return AdaptiveTracker(detector, lambda frame, box: FakeKCF(box), ring, Config(**kwargs))

    def test_stable_uses_kcf(self):
        tracker = self.tracker([([((10, 10, 20, 20), .9)], [])])
        self.assertEqual(tracker.step(None, 0, 1/30)["source"], "yolo")
        row = tracker.step(None, 1, 1/30)
        self.assertEqual(row["source"], "kcf")
        self.assertEqual(row["detector_called"], 0)

    def test_occlusion_invalidates_appearance_and_expires_prediction(self):
        tracker = self.tracker([([((10, 10, 20, 20), .9)], [(0, 0, 100, 100)]), ([], []), ([], []), ([], [])], max_prediction=1)
        tracker.step(None, 0, 1/30)
        row = tracker.step(None, 1, 1/30)
        self.assertEqual(row["reason"], "occlusion_suspected")
        self.assertEqual(row["source"], "prediction")
        self.assertEqual(row["observed"], 0)
        self.assertIsNone(tracker.kcf)
        row = tracker.step(None, 2, 1/30)
        self.assertEqual(row["mode"], "lost")
        self.assertEqual(row["center_x"], "")
        self.assertIsNone(tracker.kf)

    def test_rim_accepts_rebound_and_suppresses_extrapolation(self):
        tracker = self.tracker([([((495, 195, 10, 10), .9)], []),
                                ([((530, 195, 10, 10), .9)], []), ([], [])])
        tracker.step(None, 0, 1/60)
        row = tracker.step(None, 1, 1/60)
        self.assertEqual(row["reason"], "near_ring")
        self.assertEqual(row["center_x"], 535)
        self.assertGreater(row["innovation_nis"], tracker.cfg.gate)
        row = tracker.step(None, 2, 1/60)
        self.assertEqual(row["center_x"], "")
        self.assertEqual(row["observed"], 0)

    def test_uncertainty_triggers_detection(self):
        tracker = self.tracker([([((10, 10, 20, 20), .9)], [])]*2, uncertainty_threshold=.01)
        tracker.step(None, 0, 1/30)
        self.assertEqual(tracker.step(None, 1, 1/30)["reason"], "high_uncertainty")

    def test_fixed_disables_adaptive_triggers(self):
        tracker = self.tracker([([((495, 195, 10, 10), .9)], [(490, 190, 100, 100)])], mode="fixed")
        tracker.step(None, 0, 1/30)
        self.assertEqual(tracker.step(None, 1, 1/30)["reason"], "stable")

    def test_reject_far_false_detection_then_reacquire_after_loss(self):
        tracker = self.tracker([([((10, 10, 20, 20), .9)], []),
                               ([((900, 900, 20, 20), .99)], []),
                               ([((900, 900, 20, 20), .99)], [])], detect_interval=1, max_prediction=0)
        tracker.step(None, 0, 1/30)
        self.assertEqual(tracker.step(None, 1, 1/30)["mode"], "lost")
        self.assertEqual(tracker.step(None, 2, 1/30)["source"], "yolo")

    def test_yolo_baseline_has_no_motion_gate(self):
        tracker = self.tracker([([((10, 10, 20, 20), .9)], []),
                               ([((900, 900, 20, 20), .9)], [])], mode="yolo")
        tracker.step(None, 0, 1/30)
        self.assertEqual(tracker.step(None, 1, 1/30)["center_x"], 910)

    def test_ablation_controls_remove_only_selected_trigger(self):
        tracker = self.tracker([([((495, 195, 20, 20), .9)], [])], mode="no-ring", uncertainty_threshold=10)
        tracker.step(None, 0, 1/30)
        row = tracker.step(None, 1, 1/30)
        self.assertEqual(row["near_ring"], 0)
        self.assertEqual(row["reason"], "stable")
        tracker = self.tracker([([((10, 10, 20, 20), .9)], [])], mode="no-uncertainty", uncertainty_threshold=.01)
        tracker.step(None, 0, 1/30)
        self.assertEqual(tracker.step(None, 1, 1/30)["reason"], "stable")

    def test_ring_hysteresis(self):
        sequence = [([((505, 200, 10, 10), .9)], [])] * 4
        tracker = self.tracker(sequence)
        tracker.step(None, 0, 1/30)
        self.assertEqual(tracker.step(None, 1, 1/30)["near_ring"], 1)
        tracker.box = (605, 200, 10, 10)  # 1.7 rim widths from the rim center
        tracker.kf = MotionFilter([610, 205])
        self.assertEqual(tracker.step(None, 2, 1/30)["near_ring"], 1)
        tracker.box = (640, 200, 10, 10)  # 2.4 widths: exit
        tracker.kf = MotionFilter([645, 205])
        self.assertEqual(tracker.step(None, 3, 1/30)["near_ring"], 0)

    def test_covariance_stays_positive(self):
        kf = MotionFilter([0, 0])
        for i in range(200):
            kf.predict(1/60, 1800)
            kf.correct(np.array([i, 2*i]), 25)
        self.assertTrue(np.linalg.eigvalsh(kf.P).min() > 0)
        self.assertTrue(np.allclose(kf.P, kf.P.T))


class EvaluationTests(unittest.TestCase):
    def test_missing_predictions_count_as_failures_and_predictions_are_not_observations(self):
        truth = {i: dict(center_x="10", center_y="10", visible="1", occluded="0", near_ring="0") for i in range(3)}
        pred = {0: dict(center_x="10", center_y="10", observed="0", processing_ms="10", detector_called="1", switched="1")}
        result = evaluate(truth, pred, 20)
        self.assertEqual(result["all"]["success_rate"], 1/3)
        self.assertEqual(result["all"]["observed_success_rate"], 0)
        self.assertEqual(result["runtime"]["missing_prediction_rows"], 2)

    def test_recovery_reports_censoring(self):
        truth = {i: dict(center_x="10", center_y="10", visible="1", occluded=str(int(i == 0)), near_ring="0") for i in range(3)}
        pred = {2: dict(center_x="10", center_y="10", observed="1", processing_ms="10", detector_called="1", switched="1")}
        self.assertEqual(evaluate(truth, pred, 20)["runtime"]["recovery_frames"], [1])
        pred[2]["observed"] = "0"
        self.assertEqual(evaluate(truth, pred, 20)["runtime"]["recovery_censored"], 1)


if __name__ == "__main__":
    unittest.main()
