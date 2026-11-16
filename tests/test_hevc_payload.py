"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The HEVC frame's wire layout, as the server reads it.

`HevcFrame.unpack` (JetlinkKit/JetlinkServer/HevcFrame.swift) takes the frame
section as: roadLength at 0, wideLength at 4, then the two access units back
to back, and accepts it only where `8 + roadLength + wideLength` is exactly
the section's length. The comma used to write them interleaved instead
(length, unit, length, unit), so the wide length landed where the server read
the road unit's first four bytes and no HEVC frame was ever accepted:
measured 2026-11-16, `inference failed: BAD_SHAPE (frame 0)` on a link whose
encoder was working and whose payload was the right size.

These assert the layout itself, and `read_like_the_server` is the server's own
read transcribed, so the two cannot drift apart silently again.
"""
import unittest

from jetlink import client as jetlink_client


def read_like_the_server(payload: bytes, packed_bytes: int):
  """HevcFrame.unpack, minus the decoder: (accepted, road, wide).

  `packed_bytes` is what the server expects between the request header and
  the frame section (layout.packedBytes).
  """
  at = 8 + packed_bytes
  frame = payload[at:]
  if len(frame) < 8:
    return False, b'', b''
  road_len = int.from_bytes(frame[0:4], 'little')
  wide_len = int.from_bytes(frame[4:8], 'little')
  if 8 + road_len + wide_len != len(frame):
    return False, b'', b''
  return True, frame[8:8 + road_len], frame[8 + road_len:]


HEADER = (7).to_bytes(4, 'little') + (1 << 4).to_bytes(4, 'little')  # frame 7, Flag.hevc
PACKED = bytes(range(32))                                            # 8 float32s
ROAD = b'\x00\x00\x00\x01road-au-bytes'
WIDE = b'\x00\x00\x00\x01wide-access-unit'


class TestHevcPayload(unittest.TestCase):

  def test_the_two_lengths_come_first(self):
    parts = jetlink_client._hevc_parts(HEADER, PACKED, (ROAD, WIDE))
    payload = b''.join(bytes(p) for p in parts)
    at = len(HEADER) + len(PACKED)
    self.assertEqual(payload[at:at + 4], len(ROAD).to_bytes(4, 'little'))
    self.assertEqual(payload[at + 4:at + 8], len(WIDE).to_bytes(4, 'little'))
    self.assertEqual(payload[at + 8:at + 8 + len(ROAD)], ROAD)
    self.assertEqual(payload[at + 8 + len(ROAD):], WIDE)

  def test_the_server_accepts_it(self):
    parts = jetlink_client._hevc_parts(HEADER, PACKED, (ROAD, WIDE))
    payload = b''.join(bytes(p) for p in parts)
    ok, road, wide = read_like_the_server(payload, len(PACKED))
    self.assertTrue(ok)
    self.assertEqual(road, ROAD)
    self.assertEqual(wide, WIDE)

  def test_the_interleaved_layout_it_replaced_is_rejected(self):
    # the bug, kept as the reason the test above exists: lengths and units
    # alternating never adds up on the server's side
    interleaved = [HEADER, PACKED,
                   len(ROAD).to_bytes(4, 'little'), ROAD,
                   len(WIDE).to_bytes(4, 'little'), WIDE]
    payload = b''.join(bytes(p) for p in interleaved)
    ok, _, _ = read_like_the_server(payload, len(PACKED))
    self.assertFalse(ok)

  def test_an_empty_access_unit_still_lines_up(self):
    # a camera whose encoder had nothing yet is never sent (the gate holds
    # the frame), but the layout must not depend on both being non-empty
    parts = jetlink_client._hevc_parts(HEADER, PACKED, (ROAD, b''))
    payload = b''.join(bytes(p) for p in parts)
    ok, road, wide = read_like_the_server(payload, len(PACKED))
    self.assertTrue(ok)
    self.assertEqual(road, ROAD)
    self.assertEqual(wide, b'')


if __name__ == '__main__':
  unittest.main()
