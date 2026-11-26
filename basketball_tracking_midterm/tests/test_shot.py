"""Shot judgement state machine on synthetic trajectories."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.shot import ShotJudge

RIM = (500, 170, 540, 180)   # centre x 520, rim line y 175, half width 20


def feed(points, **options):
    judge = ShotJudge(RIM, 30, **options)
    events = [e for i, c in enumerate(points) if (e := judge.update(i, c))]
    return judge, events


def arc(x0, y0, x1, y1, n):
    return [(x0+(x1-x0)*k/(n-1), y0+(y1-y0)*k/(n-1)) for k in range(n)]


class ShotTests(unittest.TestCase):
    def test_ball_dropping_through_the_ring_is_made(self):
        _, events = feed(arc(560, 60, 522, 170, 10)+arc(522, 178, 521, 230, 6))
        self.assertEqual([(e['result'], e['reason']) for e in events], [('made', 'through_rim')])

    def test_ball_lost_in_the_net_is_made(self):
        _, events = feed(arc(560, 60, 522, 170, 10)+[(521, 180)]+[None]*10)
        self.assertEqual([(e['result'], e['reason']) for e in events], [('made', 'lost_below_rim')])

    def test_ball_falling_just_outside_the_ring_is_a_miss(self):
        _, events = feed(arc(600, 60, 552, 170, 10)+arc(551, 180, 550, 260, 5))
        self.assertEqual([(e['result'], e['reason']) for e in events], [('miss', 'passed_outside_rim')])

    def test_bounce_off_the_rim_back_up_is_a_miss(self):
        _, events = feed(arc(560, 60, 530, 165, 8)+arc(535, 160, 700, 40, 12))
        self.assertEqual([(e['result'], e['reason']) for e in events], [('miss', 'left_near_rim')])

    def test_rim_out_after_crossing_is_a_miss(self):
        _, events = feed(arc(560, 60, 522, 170, 10)+[(522, 177), (530, 168)])
        self.assertEqual([(e['result'], e['reason']) for e in events], [('miss', 'rim_out')])

    def test_pass_far_above_the_rim_is_not_a_shot(self):
        judge, events = feed(arc(300, 40, 800, 40, 30))
        self.assertEqual(events, [])
        self.assertEqual(judge.score(), dict(attempts=0, made=0, missed=0))

    def test_ball_far_to_the_side_dropping_is_not_a_shot(self):
        _, events = feed(arc(600, 20, 598, 170, 8)+arc(598, 180, 598, 300, 5), near_px=40.)
        self.assertEqual(events, [])

    def test_score_counts_several_shots(self):
        made = arc(560, 60, 522, 170, 10)+arc(522, 178, 521, 230, 6)
        miss = arc(600, 60, 552, 170, 10)+arc(551, 180, 550, 260, 5)
        judge, _ = feed(made+[None]*5+miss+[None]*5+made)
        self.assertEqual(judge.score(), dict(attempts=3, made=2, missed=1))


if __name__ == '__main__':
    unittest.main()
