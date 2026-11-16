"""Two-camera fisheye geometry for 3D ball tracking and goal detection.

Conventions: world coordinates are the court frame defined in configs/court_*.json
(metres, Z up). A camera pose stores R, t with X_cam = R @ X_world + t, as
returned by cv2.solvePnP. Pixel coordinates are undistorted with the OpenCV
fisheye model to normalised rays before any triangulation, so tracking itself
can run on the raw fisheye images.
"""
import json
import math
from pathlib import Path

import numpy as np


def load_camera(calibration, pose=None):
    c = json.loads(Path(calibration).read_text(encoding='utf-8'))
    cam = dict(K=np.array(c['K'], float), D=np.array(c['D'], float).reshape(4, 1),
               image_size=tuple(c['image_size']), calibration=str(calibration))
    if pose is not None:
        p = json.loads(Path(pose).read_text(encoding='utf-8'))
        cam.update(R=np.array(p['R'], float), t=np.array(p['t'], float).reshape(3), pose=str(pose))
    return cam


def look_at(position, target, up=(0., 0., 1.)):
    """R, t for a camera at `position` looking at `target` (OpenCV axes: x right, y down, z forward)."""
    position, target, up = (np.asarray(v, float) for v in (position, target, up))
    forward = target-position; forward /= np.linalg.norm(forward)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    return R, -R@position


def project(points_world, cam):
    """Fisheye projection of world points to pixels (N,2)."""
    import cv2
    rvec, _ = cv2.Rodrigues(cam['R'])
    pts = np.asarray(points_world, float).reshape(-1, 1, 3)
    out, _ = cv2.fisheye.projectPoints(pts, rvec, cam['t'].reshape(3, 1), cam['K'], cam['D'])
    return out.reshape(-1, 2)


def undistort(points_px, cam):
    """Pixels -> normalised image coordinates (x/z, y/z) on the undistorted pinhole plane."""
    import cv2
    pts = np.asarray(points_px, float).reshape(-1, 1, 2)
    return cv2.fisheye.undistortPoints(pts, cam['K'], cam['D']).reshape(-1, 2)


def projection_matrix(cam):
    return np.hstack([cam['R'], cam['t'].reshape(3, 1)])


def triangulate(P1, P2, x1, x2):
    """Linear DLT triangulation from normalised coordinates of one point in two views."""
    A = np.vstack([x1[0]*P1[2]-P1[0], x1[1]*P1[2]-P1[1], x2[0]*P2[2]-P2[0], x2[1]*P2[2]-P2[1]])
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3]/X[3]


def ray_angle_deg(X, cam_a, cam_b):
    ca = -cam_a['R'].T@cam_a['t']; cb = -cam_b['R'].T@cam_b['t']
    u, v = X-ca, X-cb
    cosine = np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))
    return math.degrees(math.acos(max(-1., min(1., cosine))))


def interpolate_track(times, points, query_times, max_gap_s):
    """Linear interpolation of a 2D track at query times; None where no two samples bracket the query within max_gap_s."""
    times = np.asarray(times, float); points = np.asarray(points, float)
    result = []
    for q in query_times:
        j = np.searchsorted(times, q)
        if j < len(times) and math.isclose(times[j], q, abs_tol=1e-6):
            result.append(points[j]); continue
        if j == 0 or j >= len(times) or times[j]-times[j-1] > max_gap_s:
            result.append(None); continue
        f = (q-times[j-1])/(times[j]-times[j-1])
        result.append(points[j-1]*(1-f)+points[j]*f)
    return result


def triangulate_tracks(track_a, track_b, cam_a, cam_b, offset_s=0., max_gap_s=.2):
    """track_*: list of (time_s, [x,y] pixel centre or None). Camera B time = camera A time + offset_s.

    Returns rows with world position, reprojection residuals in pixels and the ray angle.
    """
    P1, P2 = projection_matrix(cam_a), projection_matrix(cam_b)
    tb = [(t, p) for t, p in track_b if p is not None]
    times_b = [t for t, _ in tb]; points_b = [p for _, p in tb]
    valid_a = [(t, p) for t, p in track_a if p is not None]
    matched_b = interpolate_track(times_b, points_b, [t+offset_s for t, _ in valid_a], max_gap_s) if tb else [None]*len(valid_a)
    rows = []
    for (t, pa), pb in zip(valid_a, matched_b):
        if pb is None:
            continue
        xa, xb = undistort(pa, cam_a)[0], undistort(pb, cam_b)[0]
        X = triangulate(P1, P2, xa, xb)
        if X is None:
            continue
        depth_a = (cam_a['R']@X+cam_a['t'])[2]; depth_b = (cam_b['R']@X+cam_b['t'])[2]
        if depth_a <= 0 or depth_b <= 0:
            continue
        ra = float(np.linalg.norm(project(X, cam_a)[0]-np.asarray(pa)))
        rb = float(np.linalg.norm(project(X, cam_b)[0]-np.asarray(pb)))
        rows.append(dict(time_s=float(t), x=float(X[0]), y=float(X[1]), z=float(X[2]),
                         residual_a_px=ra, residual_b_px=rb, ray_angle_deg=ray_angle_deg(X, cam_a, cam_b)))
    return rows


def detect_goals(rows, hoop_center, rim_radius=.225, near_m=.3, max_gap_s=.2, min_interval_s=1.):
    """Downward crossings of the rim plane. Inside the rim -> goal; just outside -> rim_miss.

    `rows` are triangulated samples sorted by time_s. The crossing point is linearly
    interpolated between the two samples around the plane. No bounce physics.
    """
    cx, cy, h = (float(v) for v in hoop_center)
    events = []; last = -math.inf
    for a, b in zip(rows, rows[1:]):
        if b['time_s']-a['time_s'] > max_gap_s or not (a['z'] > h >= b['z']):
            continue
        f = (a['z']-h)/(a['z']-b['z'])
        x = a['x']+f*(b['x']-a['x']); y = a['y']+f*(b['y']-a['y']); t = a['time_s']+f*(b['time_s']-a['time_s'])
        d = math.hypot(x-cx, y-cy)
        kind = 'goal' if d <= rim_radius else 'rim_miss' if d <= rim_radius+near_m else None
        if kind is None or t-last < min_interval_s:
            continue
        events.append(dict(time_s=t, event=kind, distance_from_center_m=d, x=x, y=y,
                           descent_m_per_s=(a['z']-b['z'])/(b['time_s']-a['time_s'])))
        last = t
    return events


def match_events(detected, truth, tolerance_s=1.):
    """Greedy one-to-one matching of detected goals to true goal times."""
    pending = sorted(t for t in truth)
    tp = []
    for e in sorted(detected, key=lambda e: e['time_s']):
        best = min(pending, key=lambda t: abs(t-e['time_s']), default=None)
        if best is not None and abs(best-e['time_s']) <= tolerance_s:
            tp.append((e['time_s'], best)); pending.remove(best)
    fp = len(detected)-len(tp); fn = len(pending)
    precision = len(tp)/len(detected) if detected else None
    recall = len(tp)/len(truth) if truth else None
    return dict(tp=len(tp), fp=fp, fn=fn, precision=precision, recall=recall, matches=tp, unmatched_truth=pending)
