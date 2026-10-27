import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.lag_tracking import FixedLagBall, METHODS
from src.common import center


def observation(i, box=None, source='YOLO'):
    return dict(frame_id=i,timestamp_s=i/30,bbox=box,source=source if box else 'none',
                state='detected' if box else 'lost',confidence=.9 if box else None,track_id=1)


class FixedLagTests(unittest.TestCase):
    def test_output_deadline_order_tail_and_bounded_storage(self):
        for method in (m for m in METHODS if m!='bidir'):
            g=FixedLagBall(30,640,480,1,method)
            emitted=[]
            for i in range(100):
                p=observation(i,[20+i,50,40+i,70])
                out=g.push(p,[dict(bbox=p['bbox'],confidence=.9)],True)
                if i<30:self.assertIsNone(out)
                else:
                    self.assertEqual(out['frame_id'],i-30)
                    self.assertEqual(out['available_through_frame'],i)
                    self.assertTrue(out['full_lookahead'])
                    emitted.append(out)
            emitted.extend(g.flush())
            self.assertEqual([p['frame_id'] for p in emitted],list(range(100)))
            self.assertEqual(sum(not p['full_lookahead'] for p in emitted),30)
            self.assertLessEqual(g.peak_rows,30+g.history+1)

    def test_bridge_uses_future_anchor_only_after_arrival(self):
        g=FixedLagBall(30,640,480,.1,'bridge')
        out=[]
        for i in range(7):
            box=[20+i*2,50,40+i*2,70] if i in (0,3,6) else None
            result=g.push(observation(i,box),[],box is not None)
            if result:out.append(result)
        out.extend(g.flush())
        self.assertEqual(center(out[1]['bbox']),(32.,60.))
        self.assertEqual(out[1]['source'],'G_bridge')
        self.assertIsNone(out[1]['confidence'])

    def test_long_unobserved_gap_is_not_filled(self):
        for method in ('bridge','beam','rts'):
            g=FixedLagBall(30,640,480,1,method)
            results=[]
            for i in range(40):
                box=[20,50,40,70] if i in (0,35) else None
                out=g.push(observation(i,box),[dict(bbox=box,confidence=.9)] if box else [],True)
                if out:results.append(out)
            results.extend(g.flush())
            self.assertIsNone(results[20]['bbox'])

    def test_rts_reduces_noise_on_constant_motion(self):
        g=FixedLagBall(30,640,480,.3,'rts')
        estimates=[];raw=[]
        for i in range(60):
            noise=3 if i%2 else -3
            box=[20+2*i+noise,50,40+2*i+noise,70]
            raw.append(center(box)[0])
            out=g.push(observation(i,box),[],True)
            if out:estimates.append(out)
        estimates.extend(g.flush())
        error=[(center(p['bbox'])[0]-(30+2*p['frame_id']))**2 for p in estimates[10:50]]
        self.assertLess(np.mean(error),9.)

    def test_beam_rejects_isolated_high_score_distractor(self):
        g=FixedLagBall(30,640,480,.2,'beam')
        outputs=[]
        for i in range(12):
            box=[20+i*2,50,40+i*2,70]
            candidates=[dict(bbox=box,confidence=.8)]
            if i==4:candidates.append(dict(bbox=[400,300,420,320],confidence=.99))
            out=g.push(observation(i,box),candidates,True)
            if out:outputs.append(out)
        outputs.extend(g.flush())
        self.assertEqual(outputs[4]['bbox'],[28,50,48,70])

    def test_committed_output_is_not_changed_by_future_frames(self):
        g=FixedLagBall(30,640,480,.1,'bridge')
        first=None
        for i in range(4):first=g.push(observation(i,[20+i,50,40+i,70]),[],True)
        saved=dict(first)
        for i in range(4,30):g.push(observation(i,[400,200,420,220]),[],True)
        self.assertEqual(first,saved)
        self.assertEqual(first['available_through_frame'],3)

    def test_invalid_method_and_duplicate_frames(self):
        with self.assertRaises(ValueError):FixedLagBall(30,640,480,2,'unknown')
        g=FixedLagBall(30,640,480,2,'bridge')
        g.push(observation(0),[],False)
        with self.assertRaises(ValueError):g.push(observation(0),[],False)

    def test_backward_lk_recovers_missing_positions_from_later_anchor(self):
        g=FixedLagBall(30,240,160,.2,'bidir')
        patch=np.random.default_rng(12).integers(0,255,(24,24,3),dtype=np.uint8)
        outputs=[]
        for i in range(10):
            frame=np.zeros((160,240,3),np.uint8)
            frame[40:64,30+i*2:54+i*2]=patch
            box=[30+i*2,40,54+i*2,64] if i in (0,4,8) else None
            out=g.push(observation(i,box),[],box is not None,frame)
            if out:outputs.append(out)
        outputs.extend(g.flush())
        np.testing.assert_allclose(center(outputs[2]['bbox']),(46,52),atol=.7)
        self.assertEqual(outputs[2]['source'],'G_bidir')


if __name__=='__main__':unittest.main()
