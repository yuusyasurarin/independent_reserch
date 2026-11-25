"""Shot judgement from a 2D ball trajectory and a hand-marked rim (development study).

The camera is fixed, so the rim is marked once per session as a box around the ring
opening (x1, y1, x2, y2). The rim line is the box's vertical centre, its opening is
[x1, x2]. Only ball centres are used; no 3D, net or player information.

States, per ball centre (None = no position this frame):
  idle      -> armed     the ball is above the rim inside the approach zone
                         (|x - rim centre| <= approach_halfwidths * half width,
                          rim_y - approach_height <= y < rim_y)
  armed     -> crossed   the centre moves downward through the rim line with its
                         (interpolated) crossing x inside the opening +- margin
  armed     -> MISS      it crosses the rim line outside the opening, or leaves the
                         zone / times out after having come within near_px of the rim
  armed     -> idle      it leaves without ever coming near the rim (not a shot)
  crossed   -> MADE      it is seen below the rim (y >= rim_y + confirm_px) near the
                         opening, or disappears for confirm_frames (lost in the net)
  crossed   -> MISS      it rises back above the rim line (rim-out), moves far to
                         the side, or is below the rim but outside the opening
                         (|x - centre| > confirm_halfwidths * half width; added
                         2026-11-26 after one false 'made' on experiment12)
All thresholds are engineering defaults in pixels of a 1280x720 frame and were not
tuned on labelled shots. A ball passing in front of / behind the ring cannot be told
apart in 2D; this is a known limitation.
"""
import math

DEFAULTS = dict(approach_halfwidths=4., approach_height=160., margin_halfwidths=.35,
                near_px=60., confirm_px=25., confirm_frames=8, side_halfwidths=2.5, confirm_halfwidths=1.1,
                leave_frames=6, timeout_s=2.)


class ShotJudge:
    def __init__(self, rim, fps, **options):
        x1, y1, x2, y2 = map(float, rim)
        if not (x2 > x1 and y2 >= y1 and fps > 0):
            raise ValueError('rim must be (x1, y1, x2, y2) with x2 > x1, y2 >= y1')
        self.o = DEFAULTS | options
        self.rim = (x1, y1, x2, y2)
        self.cx, self.y, self.half = (x1+x2)/2, (y1+y2)/2, (x2-x1)/2
        self.fps = fps
        self.state, self.events = 'idle', []
        self.last = None            # last (frame_id, x, y)
        self.start = self.crossed_at = None
        self.min_dist = math.inf
        self.outside = self.missing = 0

    def _in_zone(self, x, y):
        return (abs(x-self.cx) <= self.o['approach_halfwidths']*self.half
                and self.y-self.o['approach_height'] <= y < self.y)

    def _rim_dist(self, x, y):
        dx = max(0., abs(x-self.cx)-self.half)
        return math.hypot(dx, y-self.y)

    def _finish(self, frame_id, result, reason):
        event = dict(frame_id=frame_id, time_s=frame_id/self.fps, result=result, reason=reason,
                     start_frame=self.start, crossed_frame=self.crossed_at,
                     min_rim_distance_px=None if math.isinf(self.min_dist) else self.min_dist)
        self.events.append(event)
        self.state, self.start, self.crossed_at = 'idle', None, None
        self.min_dist, self.outside, self.missing = math.inf, 0, 0
        return event

    def update(self, frame_id, center):
        """Feed one frame (in order). Returns a finished event dict or None."""
        o = self.o
        if center is None:
            self.missing += 1
            if self.state == 'crossed' and self.missing >= o['confirm_frames']:
                return self._finish(frame_id, 'made', 'lost_below_rim')
            if self.state == 'armed' and frame_id-self.start > o['timeout_s']*self.fps:
                return self._timeout(frame_id)
            return None
        x, y = map(float, center)
        previous, self.last, self.missing = self.last, (frame_id, x, y), 0
        if self.state == 'idle':
            if self._in_zone(x, y):
                self.state, self.start = 'armed', frame_id
                self.min_dist, self.outside = self._rim_dist(x, y), 0
            return None
        self.min_dist = min(self.min_dist, self._rim_dist(x, y))
        if self.state == 'armed':
            if previous is not None and previous[2] < self.y <= y:
                t = (self.y-previous[2])/(y-previous[2])
                x_cross = previous[1]+t*(x-previous[1])
                if abs(x_cross-self.cx) <= self.half*(1+o['margin_halfwidths']):
                    self.state, self.crossed_at = 'crossed', frame_id
                    return self._check_crossed(frame_id, x, y)
                if self.min_dist > o['near_px']:
                    return self._timeout(frame_id)  # dropped far from the ring: not a shot
                return self._finish(frame_id, 'miss', 'passed_outside_rim')
            if self._in_zone(x, y):
                self.outside = 0
            else:
                self.outside += 1
                if self.outside >= o['leave_frames']:
                    return self._timeout(frame_id)
            if frame_id-self.start > o['timeout_s']*self.fps:
                return self._timeout(frame_id)
            return None
        return self._check_crossed(frame_id, x, y)

    def _check_crossed(self, frame_id, x, y):
        o = self.o
        if y < self.y:
            return self._finish(frame_id, 'miss', 'rim_out')
        if abs(x-self.cx) > o['side_halfwidths']*self.half:
            return self._finish(frame_id, 'miss', 'bounced_sideways')
        if y >= self.y+o['confirm_px']:
            if abs(x-self.cx) > o['confirm_halfwidths']*self.half:
                return self._finish(frame_id, 'miss', 'dropped_beside_rim')
            return self._finish(frame_id, 'made', 'through_rim')
        if frame_id-self.crossed_at > o['timeout_s']*self.fps:
            return self._finish(frame_id, 'made', 'stayed_at_rim')
        return None

    def _timeout(self, frame_id):
        if self.min_dist <= self.o['near_px']:
            return self._finish(frame_id, 'miss', 'left_near_rim')
        self.state, self.start, self.min_dist, self.outside = 'idle', None, math.inf, 0
        return None

    def score(self):
        made = sum(e['result'] == 'made' for e in self.events)
        return dict(attempts=len(self.events), made=made, missed=len(self.events)-made)
