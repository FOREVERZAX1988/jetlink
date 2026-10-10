"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Lossless frames: the reference codec gives every frame back bit for bit, and
the errors are what JPEG-LS's MED predictor says at the edges too.
"""
import unittest

import numpy as np

from jetlink import lossless


class TestLossless(unittest.TestCase):
  def test_round_trip(self):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:16, 0:24]
    frames = [
      rng.integers(0, 256, (2, 3, 16, 24), dtype=np.uint8),           # noise, every wrap
      np.broadcast_to(((xx * 7 + yy * 3) % 256).astype(np.uint8), (2, 3, 16, 24)),
      np.full((2, 3, 16, 24), 255, np.uint8),
      np.zeros((2, 3, 16, 24), np.uint8),
    ]
    for frame in frames:
      errors = lossless.encode(frame)
      self.assertEqual(errors.shape, frame.shape)
      self.assertEqual(errors.dtype, np.uint8)
      np.testing.assert_array_equal(lossless.decode(errors), frame)

  def test_fold_is_a_bijection_small_first(self):
    r = np.arange(256)
    z = lossless.fold(r)
    self.assertEqual(sorted(z.tolist()), list(range(256)))
    np.testing.assert_array_equal(lossless.unfold(z), r)
    # 0, -1, 1, -2, 2 -> 0, 1, 2, 3, 4
    np.testing.assert_array_equal(lossless.fold(np.array([0, -1, 1, -2, 2])), [0, 1, 2, 3, 4])

  def test_edges(self):
    x = np.array([[[[10, 12, 9],
                    [11, 30, 8]]]], np.uint8)
    e = lossless.unfold(lossless.encode(x))[0, 0]
    # first pixel from 0, the first row from the left, the first column from above
    self.assertEqual(e[0, 0], 10)
    self.assertEqual(e[0, 1], 2)
    self.assertEqual(e[0, 2], (9 - 12) % 256)
    self.assertEqual(e[1, 0], 1)
    # a=11, b=12, c=10: c <= min, so max(a, b) = 12
    self.assertEqual(e[1, 1], 30 - 12)
    # a=30, b=9, c=12: between, so a + b - c = 27
    self.assertEqual(e[1, 2], (8 - 27) % 256)

  def test_smooth_planes_pack_small(self):
    yy, xx = np.mgrid[0:128, 0:256]
    plane = ((xx // 3 + yy // 2) % 256).astype(np.uint8)
    errors = lossless.encode(plane[None, None])
    self.assertLess(np.count_nonzero(errors > 4), errors.size // 50)


if __name__ == '__main__':
  unittest.main()
