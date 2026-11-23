import unittest

import numpy as np
from PIL import Image

from src.robustness.degrade import Degradation, bicubic_control, make_variant


class DegradationTests(unittest.TestCase):
    def test_matched_variants_are_repeatable(self):
        image = Image.new("RGB", (64, 32), (128, 128, 128))
        spec = Degradation(divisor=4, blur=1.0, noise=10, jpeg_quality=60)
        left, metadata = make_variant(image, spec, 8)
        right, _ = make_variant(image, spec, 8)
        other, _ = make_variant(image, spec, 9)
        self.assertEqual(metadata["variant_size"], (16, 8))
        np.testing.assert_array_equal(left, right)
        self.assertFalse(np.array_equal(left, other))
        self.assertEqual(bicubic_control(left, image.size).size, image.size)
