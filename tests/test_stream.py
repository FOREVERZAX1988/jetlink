"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The codec negotiation and the stream gate: which carrier a link's frames go as,
and what an HEVC stream needs to stay decodable.
"""
import unittest

from jetlink import codecs
from jetlink.openpilot import stream


class TestCodecChoice(unittest.TestCase):
  def test_a_hello_with_no_list_offers_raw_alone(self):
    self.assertEqual(codecs.offers(None), [])
    self.assertEqual(codecs.offers({}), [])
    self.assertEqual(codecs.offers({'frame_codecs': 'med-zstd'}), [])   # not a list
    self.assertEqual(codecs.pick({}, prefer_lossy=True), codecs.RAW)
    self.assertEqual(codecs.pick({}, prefer_lossy=False), codecs.RAW)

  def test_offers_keeps_the_known_names_in_a_stable_order(self):
    self.assertEqual(codecs.offers({'frame_codecs': ['hevc', 'med-zstd']}),
                     [codecs.LOSSLESS, codecs.HEVC])
    self.assertEqual(codecs.offers({'frame_codecs': ['med-zstd', 'quantum']}), [codecs.LOSSLESS])

  def test_hevc_is_never_picked_without_the_users_consent(self):
    hello = {'frame_codecs': ['hevc', 'med-zstd']}
    self.assertEqual(codecs.pick(hello, prefer_lossy=False), codecs.LOSSLESS)

  def test_with_consent_hevc_wins_where_offered(self):
    hello = {'frame_codecs': ['hevc', 'med-zstd']}
    self.assertEqual(codecs.pick(hello, prefer_lossy=True), codecs.HEVC)

  def test_lossless_beats_raw_when_offered(self):
    self.assertEqual(codecs.pick({'frame_codecs': ['med-zstd']}, prefer_lossy=False), codecs.LOSSLESS)

  def test_a_server_that_offers_lossless_keeps_it_even_with_consent(self):
    # the consent is for the loss, and lossless is not lossy: it stays the pick
    self.assertEqual(codecs.pick({'frame_codecs': ['med-zstd']}, prefer_lossy=True), codecs.LOSSLESS)

  def test_cost_is_per_camera_a_frame(self):
    raw = codecs.cost(codecs.RAW)
    self.assertTrue(raw.lossless)
    self.assertEqual(raw.bytes_per_frame, codecs.RAW_BYTES)
    self.assertLess(codecs.cost(codecs.LOSSLESS).bytes_per_frame, raw.bytes_per_frame)
    hevc = codecs.cost(codecs.HEVC, hevc_bitrate=2_000_000)
    self.assertFalse(hevc.lossless)
    self.assertEqual(hevc.bytes_per_frame, 2_000_000 // 20 // 8)   # 2 Mb/s at 20 Hz
    # the rate is what sets it, so the same codec at a higher rate costs more
    self.assertGreater(codecs.cost(codecs.HEVC, hevc_bitrate=8_000_000).bytes_per_frame, hevc.bytes_per_frame)

  def test_describe_says_lossy_or_not(self):
    self.assertIn('lossy', codecs.describe(codecs.HEVC).split('(')[1].split(',')[0])
    self.assertIn('lossless', codecs.describe(codecs.RAW))


class TestStreamGate(unittest.TestCase):
  def gate(self):
    g = stream.StreamGate()
    g.open(1)
    return g

  def test_a_new_connection_owes_an_idr_on_every_stream_and_resets(self):
    g = stream.StreamGate()
    step = g.open(1)
    self.assertTrue(step.reset_encoders)
    self.assertFalse(step.send)
    self.assertFalse(g.is_open)
    self.assertEqual(set(g.awaiting()), set(stream.STREAMS))
    # the requests surface on the next begin(), the one place they are read
    self.assertEqual(set(g.begin(1, 0).request_idr), set(stream.STREAMS))

  def test_a_frame_from_another_epoch_is_refused(self):
    g = self.gate()
    step = g.begin(0, 0)
    self.assertFalse(step.send)
    self.assertIn('stale epoch', step.reason)
    self.assertFalse(g.confirm(0, 0, road_idr=True, wide_idr=True).send)

  def test_the_gate_opens_only_on_a_confirmed_keyframe_on_both(self):
    g = self.gate()
    # one camera has it, the other does not: the frame is dropped, not sent
    one = g.confirm(1, 0, road_idr=True, wide_idr=False)
    self.assertFalse(one.send)
    self.assertEqual(one.reason, f'waiting for an IDR on {stream.STREAMS[1]}')
    self.assertFalse(g.is_open)
    self.assertEqual(g.dropped, 1)
    # both cameras: the frame opens the gate and is sent
    both = g.confirm(1, 1, road_idr=True, wide_idr=True)
    self.assertTrue(both.send)
    self.assertTrue(g.is_open)
    self.assertEqual(g.opened_at, 1)

  def test_an_outstanding_request_is_asked_for_once(self):
    g = self.gate()
    first = g.begin(1, 0)
    self.assertEqual(set(first.request_idr), set(stream.STREAMS))
    second = g.begin(1, 1)
    self.assertEqual(second.request_idr, ())     # already asked, not asked again

  def test_a_gap_shuts_the_gate_and_owes_idrs_again(self):
    g = self.gate()
    g.confirm(1, 0, road_idr=True, wide_idr=True)
    self.assertTrue(g.is_open)
    step = g.gap('queue dropped a frame')
    self.assertFalse(step.send)
    self.assertFalse(g.is_open)
    self.assertEqual(g.gaps, 1)
    # the next begin() owes the IDRs again, and the frames until the next
    # confirmed pair are dropped
    self.assertEqual(set(g.begin(1, 1).request_idr), set(stream.STREAMS))
    self.assertFalse(g.begin(1, 2).send)
    self.assertFalse(g.confirm(1, 2, road_idr=True, wide_idr=False).send)

  def test_the_encoders_are_reset_on_open_not_on_a_periodic_idr(self):
    g = self.gate()
    g.confirm(1, 0, road_idr=True, wide_idr=True)
    # frame 20 is road's periodic phase: an IDR is owed, but the stream's
    # references are still valid, so the encoders are not reset
    step = g.begin(1, 20)
    self.assertIn(stream.STREAMS[0], step.request_idr)
    self.assertFalse(step.reset_encoders)

  def test_the_periodic_phases_do_not_land_together(self):
    g = self.gate()
    g.confirm(1, 0, road_idr=True, wide_idr=True)
    phases = {}
    for fid in range(40):
      requested = g.begin(1, fid).request_idr
      for name in requested:
        phases.setdefault(name, []).append(fid)
      g.confirm(1, fid, road_idr=stream.STREAMS[0] in requested, wide_idr=stream.STREAMS[1] in requested)
    # road at multiples of the GOP, wide ten frames later, never the same frame
    self.assertEqual(phases[stream.STREAMS[0]][:2], [0, 20])
    self.assertEqual(phases[stream.STREAMS[1]][:2], [10, 30])

  def test_a_periodic_idr_is_not_requested_twice(self):
    g = self.gate()
    g.confirm(1, 0, road_idr=True, wide_idr=True)
    self.assertIn(stream.STREAMS[0], g.begin(1, 20).request_idr)
    g.confirm(1, 20, road_idr=True, wide_idr=False)
    self.assertEqual(g.begin(1, 21).request_idr, ())

  def test_a_frame_the_encoders_made_no_keyframe_for_keeps_the_gate_shut(self):
    # the encoder can refuse: the gate waits, it does not open on the request
    g = self.gate()
    g.begin(1, 0)
    g.confirm(1, 0, road_idr=True, wide_idr=True)
    g.gap('dropped')
    g.begin(1, 1)
    # the encoder answered no keyframe at all
    step = g.confirm(1, 1, road_idr=False, wide_idr=False)
    self.assertFalse(step.send)
    self.assertFalse(g.is_open)


if __name__ == '__main__':
  unittest.main()
