"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Frames as HEVC: the six warp planes are a 4:2:0 image taken apart, the frame
cache never loses a live frame's context, and a keyframe is what the stream
says and not what was asked for.
"""
import itertools
import shutil
import unittest

import numpy as np

from jetlink import hevc


def six(h=4, w=6, seed=0):
  """Six planes with every phase distinct, so a transposed phase is caught."""
  rng = np.random.default_rng(seed)
  return rng.integers(0, 256, (6, h, w), dtype=np.uint8)


class TestPlaneRoundTrip(unittest.TestCase):
  def rt(self, planes: np.ndarray) -> np.ndarray:
    return hevc.from_nv12(hevc.to_nv12(planes), plane_shape=planes.shape[-2:])

  def test_the_six_planes_become_a_buffer_and_back(self):
    planes = six()
    buf = hevc.to_nv12(planes)
    self.assertEqual(buf.shape, (hevc.PLANES * planes.shape[-2] * planes.shape[-1],))
    np.testing.assert_array_equal(self.rt(planes), planes)

  def test_each_phase_lands_on_its_own_parity_of_the_luma(self):
    # plane p holds p at every pixel: the luma must show phase p exactly at the
    # (dx, dy) parity p names, and nothing else
    planes = np.zeros((6, 2, 3), np.uint8)
    for p in range(4):
      planes[p] = p + 1
    buf = hevc.to_nv12(planes)
    luma = buf[:4 * 2 * 3].reshape(4, 6)
    for dx, dy in itertools.product((0, 1), repeat=2):
      self.assertEqual(luma[dy::2, dx::2].min(), (dx + 2 * dy) + 1)
      self.assertEqual(luma[dy::2, dx::2].max(), (dx + 2 * dy) + 1)

  def test_the_chroma_is_the_last_two_planes_interleaved(self):
    planes = six()
    buf = hevc.to_nv12(planes)
    h, w = planes.shape[-2:]
    chroma = buf[4 * h * w:].reshape(h, w, 2)
    np.testing.assert_array_equal(chroma[..., 0], planes[4])
    np.testing.assert_array_equal(chroma[..., 1], planes[5])

  def test_the_round_trip_is_exact_for_every_value(self):
    # every byte 0..255 appears, wrapped: the fold must not lose a value
    planes = (np.arange(6 * 4 * 4, dtype=np.int64).reshape(6, 4, 4) * 7 % 256).astype(np.uint8)
    np.testing.assert_array_equal(self.rt(planes), planes)

  def test_a_wrong_plane_count_is_refused(self):
    with self.assertRaises(ValueError):
      hevc.to_nv12(np.zeros((5, 4, 6), np.uint8))

  def test_a_buffer_that_is_not_six_planes_is_refused(self):
    with self.assertRaises(ValueError):
      hevc.from_nv12(np.zeros(100, np.uint8), plane_shape=(4, 6))

  def test_the_geometry_must_be_given(self):
    # a flat buffer's size cannot say how H and W split, and a wrong guess is a
    # plausible but wrong frame rather than a failure
    with self.assertRaises(ValueError):
      hevc.from_nv12(np.zeros(6 * 4 * 6, np.uint8))

  def test_the_nv12_shape_is_the_planes_own_size(self):
    self.assertEqual(hevc.nv12_shape((128, 256)), (6 * 128 * 256,))

  def test_the_nv12_geometry_is_twice_the_plane_each_way(self):
    # the length above and this pair answer different questions: an encoder
    # needs the second, and cannot get it from the first
    self.assertEqual(hevc.nv12_geometry((128, 256)), (256, 512))
    self.assertEqual(hevc.nv12_geometry((128, 256))[0] * hevc.nv12_geometry((128, 256))[1] * 3 // 2,
                     hevc.nv12_shape((128, 256))[0])

  def test_the_big_models_geometry_round_trips(self):
    planes = six(128, 256, seed=7)
    self.assertEqual(hevc.to_nv12(planes).shape, (6 * 128 * 256,))
    np.testing.assert_array_equal(self.rt(planes), planes)


class TestMetaCache(unittest.TestCase):
  def test_a_frame_finds_its_own_context(self):
    c = hevc.MetaCache()
    c.put(3, 'ctx-3')
    self.assertEqual(c.take(3), ('ctx-3', None))
    self.assertEqual(c.hits, 1)

  def test_a_taken_frame_is_a_stale_miss_not_a_gap(self):
    c = hevc.MetaCache()
    c.put(3, 'ctx-3')
    c.take(3)
    got, why = c.take(3)
    self.assertIsNone(got)
    self.assertEqual(why, hevc.Missing.STALE)

  def test_a_reused_slot_is_a_stale_miss(self):
    c = hevc.MetaCache(ring=2)
    c.put(0, 'ctx-0')
    c.put(2, 'ctx-2')             # 2 collides with 0's slot
    got, why = c.take(0)
    self.assertIsNone(got)
    self.assertEqual(why, hevc.Missing.STALE)

  def test_a_frame_never_put_is_a_gap(self):
    c = hevc.MetaCache()
    got, why = c.take(9)
    self.assertIsNone(got)
    self.assertEqual(why, hevc.Missing.GAP)
    self.assertEqual(c.gaps, 1)

  def test_a_miss_never_clears_a_live_frames_context(self):
    # the reference fork's first version popped the whole queue here and ate
    # the new stream's opening frame
    c = hevc.MetaCache()
    c.put(1, 'one')
    c.put(2, 'two')
    c.take(99)                    # a gap
    self.assertEqual(c.take(1), ('one', None))
    self.assertEqual(c.take(2), ('two', None))

  def test_clear_forgets_everything(self):
    c = hevc.MetaCache()
    c.put(1, 'one')
    c.clear()
    self.assertEqual(c.take(1)[1], hevc.Missing.GAP)

  def test_the_summary_counts_the_three_outcomes(self):
    c = hevc.MetaCache(ring=2)
    c.put(0, 'a')
    c.take(0)
    c.put(2, 'b')
    c.put(4, 'c')                 # evicts 2
    c.take(2)                     # stale
    c.take(77)                    # gap
    s = c.summary()
    self.assertEqual((s['hits'], s['stale'], s['gaps']), (1, 1, 1))


class TestKeyframeScan(unittest.TestCase):
  def nal(self, nal_type: int, prefix: bytes = b'\x00\x00\x00\x01') -> bytes:
    return prefix + bytes([(nal_type << 1) & 0xFF]) + b'\xaa\xbb'

  def test_an_irap_picture_is_a_keyframe(self):
    for nal_type in (16, 17, 18, 19, 20, 21):
      self.assertTrue(hevc.keyframe_in(self.nal(nal_type)), f'NAL {nal_type}')

  def test_a_trail_picture_is_not(self):
    for nal_type in (0, 1, 32, 39):
      self.assertFalse(hevc.keyframe_in(self.nal(nal_type)), f'NAL {nal_type}')

  def test_a_four_byte_start_code_is_read_too(self):
    self.assertTrue(hevc.keyframe_in(self.nal(19, b'\x00\x00\x00\x01')))
    self.assertTrue(hevc.keyframe_in(self.nal(19, b'\x00\x00\x01')))

  def test_vps_sps_pps_alone_are_not_a_picture(self):
    blob = self.nal(32) + self.nal(33) + self.nal(34)
    self.assertFalse(hevc.keyframe_in(blob))

  def test_an_empty_or_truncated_buffer_is_not_a_keyframe(self):
    self.assertFalse(hevc.keyframe_in(b''))
    self.assertFalse(hevc.keyframe_in(b'\x00\x00'))
    # a start code with no header byte after it is truncated, not a picture
    self.assertFalse(hevc.keyframe_in(b'\x00\x00\x00\x01'))
    self.assertFalse(hevc.keyframe_in(b'\x00\x00\x01'))

  def test_the_header_comes_before_the_picture(self):
    blob = self.nal(32) + self.nal(33) + self.nal(19) + self.nal(1)
    self.assertTrue(hevc.keyframe_in(blob))


@unittest.skipUnless(shutil.which(hevc.FFMPEG_BIN), 'needs ffmpeg')
class TestFfmpegEncoder(unittest.TestCase):
  def test_it_opens_with_an_irap_and_keeps_encoding(self):
    enc = hevc.FfmpegEncoder(size=(256, 512), bitrate=2_000_000, gop=20)
    self.addCleanup(enc.close)
    planes = six(128, 256, seed=3)
    frame = hevc.to_nv12(planes)
    out, is_key = enc.encode(frame, force_keyframe=True)
    # ffmpeg buffers, so drain a few frames; the stream must open with an IRAP
    seen_key = is_key
    for _ in range(30):
      out, is_key = enc.encode(frame)
      seen_key = seen_key or is_key
      if seen_key:
        break
    self.assertTrue(seen_key, 'the stream never carried an IRAP picture')
    self.assertGreater(len(out), 0)

  def test_a_wrong_sized_frame_is_refused(self):
    enc = hevc.FfmpegEncoder(size=(256, 512))
    self.addCleanup(enc.close)
    with self.assertRaises(ValueError):
      enc.encode(np.zeros((10, 10), np.uint8))


class TestAvailability(unittest.TestCase):
  def test_available_says_whether_the_tool_is_there(self):
    present = shutil.which(hevc.FFMPEG_BIN)
    self.assertEqual(hevc.available(present), present is not None)
    self.assertFalse(hevc.available('/nonexistent/definitely-not-ffmpeg'))


if __name__ == '__main__':
  unittest.main()
