"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The cross-fade between two models: the swap's step spread over a few frames,
never a mean of a decision.
"""
import unittest

import numpy as np

from jetlink.openpilot import blend


def arr(*values, dtype=np.float32):
  return np.array(values, dtype=dtype)


class TestBlender(unittest.TestCase):
  def test_not_fading_is_the_running_model_unchanged(self):
    b = blend.SourceBlender(fade_frames=10)
    near = {'plan': arr(1.0, 2.0)}
    self.assertIs(b.step(near, {'plan': arr(9.0, 9.0)}, fading=False), near)
    self.assertEqual(b.w, 1.0)

  def test_a_fade_ramps_the_incoming_model_in_over_its_frames(self):
    b = blend.SourceBlender(fade_frames=4)
    b.reset(0.0)                    # the outgoing model alone
    incoming = {'plan': arr(1.0)}
    outgoing = {'plan': arr(5.0)}
    seen = [float(b.step(incoming, outgoing, fading=True)['plan'][0]) for _ in range(4)]
    # the incoming model's share: 0.25, 0.5, 0.75, then itself alone
    self.assertAlmostEqual(seen[0], 1.0 * 0.25 + 5.0 * 0.75)
    self.assertAlmostEqual(seen[1], 1.0 * 0.5 + 5.0 * 0.5)
    self.assertAlmostEqual(seen[2], 1.0 * 0.75 + 5.0 * 0.25)
    self.assertAlmostEqual(seen[3], 1.0)
    self.assertEqual(b.w, 1.0)

  def test_the_fade_starts_at_the_outgoing_models_value(self):
    b = blend.SourceBlender(fade_frames=10)
    b.reset(0.0)
    # a tenth of the way in still leans on the outgoing model (near 0, far 10)
    out = b.step({'p': arr(0.0)}, {'p': arr(10.0)}, fading=True)
    self.assertGreater(float(out['p'][0]), 5.0)

  def test_a_fade_without_a_far_leg_is_over_at_once(self):
    b = blend.SourceBlender(fade_frames=10)
    b.reset(0.0)
    near = {'plan': arr(2.0)}
    self.assertIs(b.step(near, None, fading=True), near)

  def test_a_key_only_the_running_model_has_is_kept_not_invented(self):
    b = blend.SourceBlender(fade_frames=2)
    out = b.blend({'plan': arr(1.0), 'only_near': arr(7.0)}, {'plan': arr(3.0)}, w=0.5)
    self.assertEqual(float(out['only_near'][0]), 7.0)

  def test_a_shape_mismatch_is_the_running_models_answer(self):
    b = blend.SourceBlender(fade_frames=2)
    out = b.blend({'plan': arr(1.0, 2.0)}, {'plan': arr(9.0)}, w=0.5)
    np.testing.assert_array_equal(out['plan'], arr(1.0, 2.0))

  def test_an_integer_output_is_not_averaged(self):
    b = blend.SourceBlender(fade_frames=2)
    near = {'idx': arr(1, 2, 3, dtype=np.int32)}
    far = {'idx': arr(9, 9, 9, dtype=np.int32)}
    np.testing.assert_array_equal(b.blend(near, far, w=0.5)['idx'], near['idx'])

  def test_a_named_decision_follows_the_owning_model(self):
    b = blend.SourceBlender(fade_frames=2, discrete=('shouldStop',))
    near = {'shouldStop': arr(0.0)}
    far = {'shouldStop': arr(1.0)}
    # below the mid-point the outgoing model's decision stands whole
    self.assertEqual(float(b.blend(near, far, w=0.4)['shouldStop'][0]), 1.0)
    # at or past it, the incoming model's, whole and not a mean
    self.assertEqual(float(b.blend(near, far, w=0.5)['shouldStop'][0]), 0.0)
    self.assertEqual(float(b.blend(near, far, w=0.9)['shouldStop'][0]), 0.0)

  def test_a_decision_is_a_copy_not_the_models_array(self):
    b = blend.SourceBlender(fade_frames=2, discrete=('shouldStop',))
    near = {'shouldStop': arr(1.0)}
    out = b.blend(near, {'shouldStop': arr(0.0)}, w=0.9)
    out['shouldStop'][0] = 0.0
    self.assertEqual(float(near['shouldStop'][0]), 1.0, 'the model output was written through')

  def test_a_decision_the_outgoing_model_lacks_falls_back_to_the_running_one(self):
    b = blend.SourceBlender(fade_frames=2, discrete=('shouldStop',))
    near = {'shouldStop': arr(1.0)}
    out = b.blend(near, {'plan': arr(0.0)}, w=0.1)
    self.assertEqual(float(out['shouldStop'][0]), 1.0)

  def test_a_raw_dump_is_never_blended(self):
    b = blend.SourceBlender(fade_frames=2)
    near = {'raw_pred': arr(1.0, 2.0, 3.0)}
    far = {'raw_pred': arr(9.0, 9.0, 9.0)}
    np.testing.assert_array_equal(b.blend(near, far, w=0.5)['raw_pred'], near['raw_pred'])

  def test_w_is_clipped_and_zero_is_the_outgoing_leg_alone(self):
    b = blend.SourceBlender(fade_frames=2)
    np.testing.assert_array_equal(b.blend({'p': arr(1.0)}, {'p': arr(5.0)}, w=0.0)['p'], arr(5.0))
    np.testing.assert_array_equal(b.blend({'p': arr(1.0)}, {'p': arr(5.0)}, w=-3.0)['p'], arr(5.0))

  def test_a_zero_frame_fade_is_refused(self):
    with self.assertRaises(ValueError):
      blend.SourceBlender(fade_frames=0)


if __name__ == '__main__':
  unittest.main()
