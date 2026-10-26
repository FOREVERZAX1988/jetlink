"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The late-reply watermark: one number that says a reply is late, and the counts
that separate lateness from loss.
"""
import unittest

from jetlink.openpilot import latency


class TestLateTracker(unittest.TestCase):
  def test_nothing_is_late_before_a_frame_is_given_up_on(self):
    t = latency.LateTracker()
    self.assertIsNone(t.watermark)
    self.assertFalse(t.arrived(1))
    self.assertFalse(t.arrived(2))
    self.assertEqual(t.late, 0)
    self.assertEqual(t.answered, 2)

  def test_a_reply_at_or_below_the_watermark_is_late(self):
    t = latency.LateTracker()
    t.gave_up(5)
    self.assertTrue(t.arrived(5))      # the frame itself
    self.assertTrue(t.arrived(3))      # anything older, in order or not
    self.assertFalse(t.arrived(6))     # above it, answered in time
    self.assertEqual(t.late, 2)
    self.assertEqual(t.answered, 3)

  def test_the_watermark_only_moves_forward(self):
    t = latency.LateTracker()
    t.gave_up(5)
    t.gave_up(3)
    self.assertEqual(t.watermark, 5)
    t.gave_up(9)
    self.assertEqual(t.watermark, 9)

  def test_a_frame_given_up_on_and_answered_is_not_a_loss(self):
    t = latency.LateTracker()
    t.gave_up(1)
    self.assertEqual(t.settled(), 1)
    t.arrived(1)
    self.assertEqual(t.settled(), 0)

  def test_a_frame_given_up_on_and_never_answered_stays_settled(self):
    t = latency.LateTracker()
    t.gave_up(1)
    t.gave_up(2)
    t.arrived(2)
    self.assertEqual(t.settled(), 1)
    self.assertEqual(t.late, 1)

  def test_the_summary_counts_lateness_apart_from_loss(self):
    t = latency.LateTracker()
    for fid in (1, 2, 3):
      t.gave_up(fid)
    t.arrived(1, now=0.0)
    t.arrived(2, now=0.1)
    # 3 never came back
    s = t.summary()
    self.assertEqual(s['answered'], 2)
    self.assertEqual(s['late'], 2)
    self.assertEqual(s['settled'], 1)
    self.assertIn('late_gap_p50_ms', s)
    self.assertAlmostEqual(s['late_max_ms'] if 'late_max_ms' in s else s['late_gap_max_ms'], 100.0)

  def test_the_summary_reports_no_latency_in_ms_but_a_rhythm(self):
    # the tracker sees when a reply was read, not when its frame started, so
    # the only ms it can stand behind is the gap between late replies
    t = latency.LateTracker()
    t.gave_up(1)
    t.arrived(1, now=10.0)
    self.assertEqual(t.summary()['late'], 1)
    self.assertNotIn('late_p50_ms', t.summary())   # nothing to measure a gap from

  def test_the_gaps_are_between_consecutive_late_replies(self):
    t = latency.LateTracker()
    for fid in (1, 2, 3):
      t.gave_up(fid)
    t.arrived(1, now=0.0)
    t.arrived(2, now=0.05)
    t.arrived(3, now=0.20)
    gaps = list(t._gaps())
    self.assertEqual(len(gaps), 2)
    self.assertAlmostEqual(gaps[0], 50.0)
    self.assertAlmostEqual(gaps[1], 150.0)

  def test_the_replies_held_are_bounded(self):
    t = latency.LateTracker(kept=4)
    for fid in range(1, 11):
      t.gave_up(fid)
      t.arrived(fid, now=float(fid))
    self.assertEqual(len(t._lately), 4)
    self.assertEqual(t.late, 10)          # the counts are not bounded, only the history


if __name__ == '__main__':
  unittest.main()
