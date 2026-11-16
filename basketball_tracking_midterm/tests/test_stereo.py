"""Synthetic two-camera fisheye scene: triangulation accuracy, time offset handling, goal detection."""
import math
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.stereo import (detect_goals, interpolate_track, look_at, match_events, project,
                        ray_angle_deg, triangulate_tracks, undistort, projection_matrix)

HOOP=np.array([12.425,0.,3.05])


def camera(position):
    R,t=look_at(position,HOOP)
    return dict(K=np.array([[420.,0,640],[0,420.,360],[0,0,1]]),D=np.array([.06,-.02,.004,-.001]).reshape(4,1),
                image_size=(1280,720),R=R,t=t)


def arc(start,end,apex_dz,n=40,t0=0.):
    """Parabolic arc in time from start to end, sampled at 30 Hz-ish intervals."""
    rows=[]
    for k in range(n):
        f=k/(n-1);p=start*(1-f)+end*f;p=p.copy();p[2]+=apex_dz*4*f*(1-f)
        rows.append((t0+k/30,p))
    return rows


class StereoTests(unittest.TestCase):
    def setUp(self):
        self.a=camera([6.,-9.,4.]);self.b=camera([6.,9.,4.])
        self.rng=np.random.default_rng(0)

    def test_undistort_inverts_projection(self):
        X=np.array([[8.,2.,2.]]);px=project(X,self.a)
        n=undistort(px,self.a)[0]
        cam=self.a['R']@X[0]+self.a['t']
        np.testing.assert_allclose(n,cam[:2]/cam[2],atol=1e-6)

    def test_triangulation_recovers_trajectory_with_pixel_noise(self):
        truth=arc(np.array([6.,2.,2.]),HOOP,2.)
        ta=[(t,project(p,self.a)[0]+self.rng.normal(0,.5,2)) for t,p in truth]
        tb=[(t,project(p,self.b)[0]+self.rng.normal(0,.5,2)) for t,p in truth]
        rows=triangulate_tracks(ta,tb,self.a,self.b)
        self.assertEqual(len(rows),len(truth))
        err=[np.linalg.norm([r['x'],r['y'],r['z']]-p) for r,(_,p) in zip(rows,truth)]
        self.assertLess(max(err),.15);self.assertLess(np.median(err),.04)  # 0.5 px noise at ~10 m range
        self.assertLess(np.median([r['residual_a_px'] for r in rows]),1.5)
        self.assertTrue(all(5<r['ray_angle_deg']<175 for r in rows))  # cameras face each other across the hoop

    def test_time_offset_and_missing_samples(self):
        truth=arc(np.array([2.,1.,2.]),HOOP,2.)
        ta=[(t,project(p,self.a)[0]) for t,p in truth]
        tb=[(t+.1,project(p,self.b)[0]) for t,p in truth]
        tb[10]=(tb[10][0],None)  # one lost frame in camera B, bridged by interpolation
        rows=triangulate_tracks(ta,tb,self.a,self.b,offset_s=.1)
        self.assertEqual(len(rows),len(truth))
        err=[np.linalg.norm([r['x'],r['y'],r['z']]-p) for r,(_,p) in zip(rows,truth)]
        self.assertLess(max(err),.05)
        wrong=triangulate_tracks(ta,tb,self.a,self.b,offset_s=0.)
        self.assertGreater(np.median([r['residual_a_px'] for r in wrong]),np.median([r['residual_a_px'] for r in rows]))
        self.assertEqual(interpolate_track([0,1],[[0,0],[2,2]],[.5,3.],1.)[1],None)

    def test_goal_and_miss_classification(self):
        # The arc peaks above the rim and its last sample sits just below the rim plane at the hoop.
        made=[dict(time_s=t,x=p[0],y=p[1],z=p[2]) for t,p in arc(np.array([5.,0.,2.]),HOOP+[0,0,-.05],1.6)]
        events=detect_goals(made,HOOP)
        self.assertEqual([e['event'] for e in events],['goal'])
        self.assertLess(events[0]['distance_from_center_m'],.1)
        rim=[dict(time_s=t,x=p[0],y=p[1],z=p[2]) for t,p in arc(np.array([5.,.4,2.]),HOOP+[0,.4,-.05],1.6)]
        self.assertEqual([e['event'] for e in detect_goals(rim,HOOP)],['rim_miss'])
        airball=[dict(time_s=t,x=p[0],y=p[1],z=p[2]) for t,p in arc(np.array([5.,2.,2.]),HOOP+[0,2.,-.05],1.6)]
        self.assertEqual(detect_goals(airball,HOOP),[])
        rising=[dict(time_s=t,x=p[0],y=p[1],z=p[2]) for t,p in arc(HOOP+[0,0,-.05],np.array([5.,0.,2.]),1.6)]
        self.assertEqual(detect_goals(rising,HOOP),[])
        m=match_events(events,[events[0]['time_s']+.3,20.],tolerance_s=1.)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,0,1))

    def test_same_camera_gives_zero_ray_angle(self):
        X=np.array([8.,0.,2.])
        self.assertAlmostEqual(ray_angle_deg(X,self.a,self.a),0.)
        self.assertGreater(ray_angle_deg(X,self.a,self.b),30.)
        self.assertEqual(projection_matrix(self.a).shape,(3,4))


if __name__=='__main__':unittest.main()
