"""Backlog-controlled dropping, in-order emission and fixed-delay release."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.adaptive_tracking import AdaptiveBallPolicy
from src.common import center
from src.lag_tracking import FixedLagBall
from src.realtime import DelayedOutputStore, TrackingWorker, placeholder


FPS=30


class FakeClock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now


class Scene:
    """A textured patch moving 2 px per frame; the fake detector reports its true box."""
    def __init__(self):
        self.patch=np.random.default_rng(3).integers(0,255,(24,24,3),dtype=np.uint8)
    def box(self,i):return [30+2*i,40,54+2*i,64]
    def frame(self,i):
        f=np.zeros((160,240,3),np.uint8);b=self.box(i);f[b[1]:b[3],b[0]:b[2]]=self.patch;return f
    def detector(self,i):
        return lambda image:([dict(bbox=self.box(i),confidence=.9,class_id=0)],{})


def make_worker(clock,max_backlog=1.,detect=None):
    cfg=dict(confidence=.25,candidate_conf=.001,max_age_s=.3,adaptive={})
    policy=AdaptiveBallPolicy(cfg,FPS)
    smoother=FixedLagBall(FPS,240,160,.4,'bridge')
    return TrackingWorker(policy,smoother,detect or (lambda image:([],{})),FPS,max_backlog,clock)


class WorkerTests(unittest.TestCase):
    def test_keeps_every_frame_when_processing_is_fast(self):
        clock=FakeClock();scene=Scene();holder={}
        worker=make_worker(clock,detect=lambda image:holder['detect'](image))
        emitted=[]
        for i in range(60):
            t=i/FPS;clock.now=t+.005;holder['detect']=scene.detector(i)
            emitted+=worker.step(i,t,scene.frame(i),newer_waiting=False)
        self.assertEqual([o['frame_id'] for o in emitted],list(range(48)))
        emitted+=worker.flush()
        self.assertEqual([o['frame_id'] for o in emitted],list(range(60)))
        self.assertEqual(worker.counters['dropped'],0)
        self.assertEqual(worker.counters['processed'],60)
        self.assertTrue(all(o['processed'] for o in emitted))
        self.assertTrue(all(o['bbox'] is not None for o in emitted[:48]))
        self.assertLessEqual(worker.recent_yolo_hz(),15)

    def test_stale_frame_is_dropped_only_when_a_newer_frame_waits(self):
        clock=FakeClock();scene=Scene()
        worker=make_worker(clock,max_backlog=1.)
        clock.now=2.;worker.step(0,0.,scene.frame(0),newer_waiting=False)
        self.assertEqual(worker.counters['processed'],1)
        clock.now=2.5;worker.step(1,1/FPS,scene.frame(1),newer_waiting=True)
        self.assertEqual(worker.counters['dropped'],1)
        clock.now=2.6;worker.step(2,2.55,scene.frame(2),newer_waiting=True)
        self.assertEqual(worker.counters['processed'],2)  # fresh enough despite a queue
        outputs=worker.flush()
        self.assertEqual([o['frame_id'] for o in outputs],[0,1,2])
        self.assertEqual(outputs[1]['original_source'],'dropped')
        self.assertFalse(outputs[1]['processed'])
        self.assertAlmostEqual(worker.summary()['drop_rate'],1/3)

    def test_dropped_frames_are_bridged_between_yolo_anchors(self):
        clock=FakeClock();scene=Scene();holder={}
        worker=make_worker(clock,max_backlog=.5,detect=lambda image:holder['detect'](image))
        outputs=[]
        for i in range(8):
            holder['detect']=scene.detector(i);t=i/FPS
            stale=i in (1,2,4,5)
            clock.now=t+(1. if stale else .005)
            outputs+=worker.step(i,t,scene.frame(i),newer_waiting=stale)
        outputs+=worker.flush()
        self.assertEqual(worker.counters['dropped'],4)
        self.assertEqual([o['frame_id'] for o in outputs],list(range(8)))
        for i in (0,3):  # anchors keep their YOLO box; bridge relabels them like the offline G run
            self.assertEqual(outputs[i]['bbox'],scene.box(i))
            self.assertEqual(outputs[i]['original_source'],'YOLO')
        for i in (1,2,4,5):
            self.assertEqual(outputs[i]['source'],'G_bridge')
            np.testing.assert_allclose(center(outputs[i]['bbox']),center(scene.box(i)),atol=.5)
            self.assertIsNone(outputs[i]['confidence'])

    def test_order_is_enforced(self):
        worker=make_worker(FakeClock())
        worker.step(0,0.,np.zeros((160,240,3),np.uint8),False)
        with self.assertRaises(ValueError):worker.step(2,0.,np.zeros((160,240,3),np.uint8),False)
        self.assertEqual(placeholder(5,FPS)['source'],'dropped')


class StoreTests(unittest.TestCase):
    def test_release_at_fixed_delay_in_order_with_results(self):
        store=DelayedOutputStore(3.,4.)
        for i in range(120):store.add_frame(i,i/FPS,i)
        for i in range(0,120,2):store.add_result(dict(frame_id=i,bbox=[0,0,1,1]))
        self.assertIsNone(store.due(2.99))
        self.assertEqual(store.due(3.),0)
        self.assertEqual(store.due(3.05),1)
        self.assertEqual(store.due(3.5),15)
        released=store.release(15)
        self.assertEqual([r[0] for r in released],list(range(16)))
        self.assertEqual([r[3] is not None for r in released[:3]],[True,False,True])
        self.assertEqual(store.buffered_frames(),104)
        self.assertIsNone(store.due(3.5))
        self.assertAlmostEqual(store.next_due_time(),16/FPS+3.)
        with self.assertRaises(ValueError):store.add_frame(100,5.,None)

    def test_unreleased_frames_are_discarded_after_max_hold(self):
        store=DelayedOutputStore(3.,4.)
        for i in range(300):store.add_frame(i,i/FPS,None)
        self.assertLessEqual(store.buffered_frames(),4*FPS+1)
        self.assertGreater(store.discarded,0)
        store.add_result(dict(frame_id=0))  # already discarded: must not resurrect
        self.assertEqual(store.release(0),[])
        self.assertNotIn(0,store.results)
        with self.assertRaises(ValueError):DelayedOutputStore(3.,3.)


if __name__=='__main__':unittest.main()
