"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's own six stages: where a frame's time went before the server saw it,
and that a stage the link does not have reads as zero.
"""
import unittest

from jetlink.openpilot import stages


def run(t, fid, *, start=None, eof=None, captured=None, warped=None, sent=None, replied=None, done=0.0):
  """One frame through the tracker. Every stamp is absolute seconds; None
  leaves that stage unstamped (a stage the link does not have)."""
  first = next(v for v in (start, eof, captured, warped, sent, replied, done) if v is not None)
  t.start(fid, now=first if start is None else start, t_eof=eof)
  for name, at in (('captured', captured), ('warped', warped), ('sent', sent), ('replied', replied)):
    if at is not None:
      getattr(t, name)(fid, at)
  return t.finished(fid, done)


class TestFrameStages(unittest.TestCase):
  def test_the_six_stages_are_the_seams_and_sum_to_the_frame(self):
    t = stages.FrameStages()
    s = run(t, 0, start=10.0, eof=10.0, captured=10.010, warped=10.014, sent=10.020, replied=10.060, done=10.065)
    # capture 10 ms, warp 4, send 6, reply_wait 40, publish 5, overhead 0
    self.assertAlmostEqual(s.capture, 10.0, places=3)
    self.assertAlmostEqual(s.warp, 4.0, places=3)
    self.assertAlmostEqual(s.send, 6.0, places=3)
    self.assertAlmostEqual(s.reply_wait, 40.0, places=3)
    self.assertAlmostEqual(s.publish, 5.0, places=3)
    self.assertAlmostEqual(s.overhead, 0.0, places=3)
    self.assertAlmostEqual(s.total_ms, 65.0, places=3)

  def test_the_capture_stage_is_measured_from_the_sensors_own_stamp(self):
    # the frame thread picks the frame up 12 ms after the sensor finished
    # reading it out: that 12 ms is the link's to account for, not the sensor's
    t = stages.FrameStages()
    s = run(t, 0, start=0.012, eof=0.0, captured=0.012, warped=0.013, done=0.014)
    self.assertAlmostEqual(s.capture, 12.0, places=3)
    self.assertAlmostEqual(s.warp, 1.0, places=3)

  def test_a_frame_with_no_eof_measures_capture_from_its_own_start(self):
    t = stages.FrameStages()
    s = run(t, 0, start=0.0, captured=0.0, warped=0.001, sent=0.002, replied=0.003, done=0.004)
    self.assertEqual(s.capture, 0.0)   # no eof: the frame's start stands in
    self.assertAlmostEqual(s.warp, 1.0, places=3)

  def test_a_stage_that_never_happened_reads_zero(self):
    t = stages.FrameStages()
    # a raw link has no reply_wait before it is set, and no publish either
    s = run(t, 0, start=0.0, eof=0.0, captured=0.001, warped=0.002, sent=0.003, done=0.004)
    self.assertEqual(s.reply_wait, 0.0)
    self.assertEqual(s.publish, 0.0)

  def test_a_stamp_out_of_order_reads_zero_not_negative(self):
    t = stages.FrameStages()
    t.start(0, now=0.0, t_eof=0.0)
    t.captured(0, 0.010)
    t.warped(0, 0.005)     # before the capture: a clock that is not monotonic
    s = t.finished(0, 0.020)
    self.assertGreaterEqual(s.warp, 0.0)
    self.assertGreaterEqual(s.total_ms, 0.0)

  def test_a_reused_slot_drops_a_late_event_for_the_old_frame(self):
    t = stages.FrameStages(ring=2)
    run(t, 0, start=0.0, eof=0.0, captured=0.001, done=0.002)
    run(t, 2, start=0.010, eof=0.010, captured=0.011, done=0.012)   # 2 collides with 0's slot
    # a late stamp for frame 0 must not land on frame 2's slot
    t.warped(0, 0.050)
    self.assertIsNone(t.finished(0, 0.060), 'a reused id must not report stages')

  def test_finishing_twice_reports_once(self):
    t = stages.FrameStages()
    run(t, 0, start=0.0, eof=0.0, captured=0.001, done=0.002)
    self.assertIsNone(t.finished(0, 0.003))

  def test_last_is_the_most_recent_finished(self):
    t = stages.FrameStages()
    self.assertIsNone(t.last())
    run(t, 0, start=0.0, eof=0.0, captured=0.001, done=0.002)
    run(t, 1, start=0.010, eof=0.010, captured=0.011, done=0.012)
    self.assertIsNotNone(t.last())

  def test_reset_forgets_everything(self):
    t = stages.FrameStages()
    run(t, 0, start=0.0, eof=0.0, captured=0.001, done=0.002)
    t.reset()
    self.assertIsNone(t.last())
    self.assertEqual(t.summary()['frames'], 0)

  def test_the_summary_says_which_stage_the_time_goes_to(self):
    t = stages.FrameStages()
    # 20 frames with a long reply_wait and a short warp
    for i in range(20):
      base = i * 0.050
      run(t, i, start=base, eof=base, captured=base + 0.002, warped=base + 0.003,
          sent=base + 0.004, replied=base + 0.044, done=base + 0.045)
    s = t.summary()
    self.assertEqual(s['frames'], 20)
    self.assertGreater(s['reply_wait']['p50_ms'], s['warp']['p50_ms'])
    self.assertAlmostEqual(s['frame']['p50_ms'], 45.0, places=0)
    self.assertIn('held', s)

  def test_the_summary_is_safe_before_any_frame(self):
    s = stages.FrameStages().summary()
    self.assertEqual(s['frames'], 0)
    self.assertEqual(s['frame'], {'p50_ms': 0.0, 'p95_ms': 0.0})
    self.assertEqual(s['capture'], {'p50_ms': 0.0, 'p95_ms': 0.0})

  def test_the_window_bounds_what_is_summarised(self):
    t = stages.FrameStages()
    for i in range(10):
      base = i * 0.050
      run(t, i, start=base, eof=base, captured=base + 0.001, warped=base + 0.002,
          sent=base + 0.003, replied=base + 0.043, done=base + 0.044)
    self.assertEqual(t.summary(window=3)['frames'], 3)


class TestStageTracker(unittest.TestCase):
  def test_it_keeps_the_last_frames_only(self):
    tr = stages.StageTracker(kept=3)
    self.assertEqual(tr.summary()['frames'], 0)
    for i in range(5):
      tr.record(stages.StageMs(warp=float(i), reply_wait=1.0))
    s = tr.summary()
    self.assertEqual(s['frames'], 3)
    self.assertEqual(s['warp']['max_ms'], 4.0)

  def test_a_none_record_changes_nothing(self):
    tr = stages.StageTracker()
    tr.record(None)
    self.assertEqual(tr.summary()['frames'], 0)

  def test_the_frame_total_is_the_stages_summed(self):
    tr = stages.StageTracker()
    tr.record(stages.StageMs(capture=1.0, warp=2.0, send=3.0, reply_wait=4.0, publish=5.0, overhead=6.0))
    self.assertEqual(tr.summary()['frame']['med_ms'], 21.0)


if __name__ == '__main__':
  unittest.main()


class TestMeans(unittest.TestCase):
  """The O(1) reader the modeld record uses: it is written once a second and
  must not sort the ring to say what a frame costs."""

  def _run(self, t, i, **kw):
    t.start(i, kw['start'], t_eof=kw.get('eof'))
    if 'captured' in kw:
      t.captured(i, kw['captured'])
    if 'warped' in kw:
      t.warped(i, kw['warped'])
    if 'sent' in kw:
      t.sent(i, kw['sent'])
    if 'replied' in kw:
      t.replied(i, kw['replied'])
    return t.finished(i, kw['done'], held=kw.get('held', False))

  def test_empty_before_a_frame(self):
    self.assertEqual(stages.FrameStages().means(), {'frames': 0})

  def test_a_frames_stages_averaged(self):
    t = stages.FrameStages()
    for i in range(4):
      base = i * 0.050
      self._run(t, i, start=base, eof=base, captured=base + 0.001, warped=base + 0.002,
                sent=base + 0.004, replied=base + 0.008, done=base + 0.009)
    m = t.means()
    self.assertEqual(m['frames'], 4)
    # capture 1 ms, warp 1, send 2, reply_wait 4, publish 1, the rest overhead
    self.assertEqual(m['capture'], 1.0)
    self.assertEqual(m['warp'], 1.0)
    self.assertEqual(m['send'], 2.0)
    self.assertEqual(m['reply_wait'], 4.0)
    self.assertEqual(m['publish'], 1.0)
    self.assertEqual(m['frame'], 9.0)

  def test_the_ring_dropping_a_frame_drops_it_from_the_mean(self):
    # the sum has to follow the ring, or the mean counts frames it no longer holds
    t = stages.FrameStages(ring=2)
    for i in range(2):
      base = i * 0.050
      self._run(t, i, start=base, eof=base, captured=base + 0.001, warped=base + 0.001,
                sent=base + 0.001, replied=base + 0.001, done=base + 0.001)
    self.assertEqual(t.means()['capture'], 1.0)
    # a third frame, whose capture is 5 ms: the first is dropped, so the mean
    # is (1 + 5) / 2, not (1 + 1 + 5) / 3
    self._run(t, 2, start=10.0, eof=10.0, captured=10.005, warped=10.005,
              sent=10.005, replied=10.005, done=10.005)
    self.assertEqual(t.means()['frames'], 2)
    self.assertEqual(t.means()['capture'], 3.0)

  def test_a_reset_clears_the_sum(self):
    t = stages.FrameStages()
    self._run(t, 0, start=0.0, eof=0.0, captured=0.001, warped=0.001, sent=0.001, replied=0.001, done=0.001)
    self.assertEqual(t.means()['frames'], 1)
    t.reset()
    self.assertEqual(t.means(), {'frames': 0})

  def test_held_frames_are_counted(self):
    t = stages.FrameStages()
    self._run(t, 0, start=0.0, eof=0.0, captured=0.001, warped=0.001, sent=0.001, replied=0.001, done=0.001, held=True)
    self._run(t, 1, start=1.0, eof=1.0, captured=1.001, warped=1.001, sent=1.001, replied=1.001, done=1.001)
    m = t.means()
    self.assertEqual(m['frames'], 2)
    self.assertEqual(m['held'], 1)

  def test_the_mean_agrees_with_the_percentiles_on_one_frame(self):
    # with one frame, p50 and the mean are the same number: a cheap cross-check
    # that the sum is over the same stages
    t = stages.FrameStages()
    self._run(t, 0, start=0.0, eof=0.0, captured=0.002, warped=0.004, sent=0.006, replied=0.010, done=0.012)
    means, summary = t.means(), t.summary()
    for name in stages.STAGES:
      self.assertEqual(means[name], summary[name]['p50_ms'], name)
    self.assertEqual(means['frame'], summary['frame']['p50_ms'])
