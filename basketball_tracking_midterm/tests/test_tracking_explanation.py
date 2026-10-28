"""Check direction and pixel scale of the educational response-map coordinates."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.visualize_tracking_examples import gray_kcf


class CorrelationMapTests(unittest.TestCase):
    def test_known_translation_in_both_axes(self):
        texture=np.random.default_rng(42).integers(0,256,(24,24,3),dtype=np.uint8)
        previous=np.full((150,180,3),128,dtype=np.uint8)
        previous[60:84,70:94]=texture
        for dx,dy in ((5,-3),(-4,6),(0,0)):
            with self.subTest(dx=dx,dy=dy):
                current=np.full_like(previous,128)
                current[60+dy:84+dy,70+dx:94+dx]=texture
                result=gray_kcf(previous,current,[70,60,94,84])
                self.assertEqual(result['delta'],(dx,dy))


if __name__=='__main__':unittest.main()
