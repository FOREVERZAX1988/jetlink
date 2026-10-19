"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Where a frame's time went: the six segments, finished where the reply is read
-- which on a pipelined link is a later frame than the one that spent them --
and the percentiles a drive's summary line is read off.
"""
import unittest

from jetlink.openpilot import stages


class StageTimesTest(unittest.TestCase):
  def frame(self, t, frame_id, *, pickup=0.050, warp=0.001, pair=0.004, encode=None, send=0.003, reply_at=0.043):
    slot = t.start(pickup)
    slot.warp, slot.pair, slot.encode, slot.send = warp, pair, encode, send
    t.track(slot, frame_id, 100.0)
    return t.answer(frame_id, 100.0 + reply_at)

  def test_the_six_are_every_frame_and_reply_is_the_round_trip(self):
    t = stages.StageTimes()
    row = self.frame(t, 7, encode=0.002)
    self.assertEqual(row['frame'], 7)
    self.assertEqual([row[f'{name}_ms'] for name in stages.STAGES], [50.0, 1.0, 4.0, 2.0, 3.0, 43.0])
    # the order the frame spends them in, and the same six in the line
    self.assertEqual(stages.text(row),
                     'frame 7 pickup 50.0 warp 1.0 pair 4.0 encode 2.0 send 3.0 reply 43.0')

  def test_a_segment_that_did_not_happen_is_marked_not_invented(self):
    # the first frame of a session has nothing to be picked up after, and a
    # warp with no errors has nothing to pack
    t = stages.StageTimes()
    row = self.frame(t, 1, pickup=None, encode=None)
    self.assertIsNone(row['pickup_ms'])
    self.assertIsNone(row['encode_ms'])
    self.assertIn(' pickup - ', stages.text(row))
    self.assertIn(' encode - ', stages.text(row))

  def test_a_late_reply_is_a_later_frames_to_read_and_still_timed(self):
    # a pipelined link: frame 1's reply is read while frame 3 is on its way,
    # which is the frame that never waited and the one the four-segment line
    # cannot report
    t = stages.StageTimes()
    for frame_id in (1, 2):
      slot = t.start(0.050)
      t.track(slot, frame_id, 100.0 + frame_id * 0.05)
    row = t.answer(1, 100.0 + 3 * 0.05)
    self.assertEqual(row['reply_ms'], 100.0, 'two frames of the host being behind')
    self.assertEqual(t.frames, 1)

  def test_a_reply_past_the_ring_is_counted_and_not_guessed_at(self):
    t = stages.StageTimes(ring=4)
    self.frame(t, 1)
    for frame_id in range(2, 7):
      slot = t.start(0.050)
      t.track(slot, frame_id, 100.0)
    self.assertIsNone(t.answer(1, 300.0), 'its slot is frame 5"s now')
    self.assertEqual((t.frames, t.untimed), (1, 1))
    self.assertEqual(t.answer(5, 100.043)['frame'], 5)

  def test_the_summary_is_p50_p99_max_of_each_segment(self):
    t = stages.StageTimes()
    for frame_id in range(1, 100):
      self.frame(t, frame_id, reply_at=0.040)
    self.frame(t, 100, reply_at=0.200)
    summary = t.summary()
    self.assertEqual((summary['frames'], summary['untimed']), (100, 0))
    self.assertEqual(summary['reply'], (40.0, 40.0, 200.0))
    self.assertEqual(summary['pickup'], (50.0, 50.0, 50.0))
    # a segment no kept frame had reads as no data, and the line says so
    t2 = stages.StageTimes()
    self.frame(t2, 1, encode=None)
    self.assertIsNone(t2.summary()['encode'])
    self.assertIn('encode -', stages.text(t2.summary()))

  def test_the_percentiles_are_over_the_frames_kept(self):
    t = stages.StageTimes(kept=4)
    for frame_id in range(1, 5):
      self.frame(t, frame_id, reply_at=0.040)
    self.frame(t, 5, reply_at=0.200)
    self.assertEqual(t.summary()['frames'], 5)
    # the window is the last four answers: one slow frame of four is its max,
    # not its median
    self.assertEqual(t.summary()['reply'], (40.0, 40.0, 200.0))

  def test_an_untimed_summary_is_still_a_line(self):
    self.assertEqual(stages.text(stages.StageTimes().summary()),
                     'frames 0 untimed 0 pickup - warp - pair - encode - send - reply -')


if __name__ == '__main__':
  unittest.main()
