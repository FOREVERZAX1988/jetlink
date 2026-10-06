"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink.transport.aio itself: buffer addresses everywhere, and the real
io_submit/io_getevents on Linux, against a pipe standing in for ep2.
"""
import ctypes
import os
import sys

import pytest

from jetlink.transport.aio import Aio, address


def test_an_address_is_where_the_bytes_are_for_read_only_buffers_too():
  data = bytes(range(200))
  for buf, want in ((data, data), (memoryview(data)[17:90], data[17:90]),
                    (bytearray(data), data), (memoryview(bytearray(data))[5:], data[5:])):
    addr, n = address(buf)
    assert n == len(want) and ctypes.string_at(addr, n) == want


def test_an_empty_buffer_has_no_bytes():
  assert address(b'')[1] == 0


@pytest.mark.real_aio
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux AIO')
def test_requests_reach_the_fd_in_order_and_are_reaped():
  r, w = os.pipe()
  aio = Aio(w, 8)
  try:
    payload = os.urandom(3000)
    addr, _ = address(payload)
    assert aio.submit(1, [[(addr, 1000)], [(addr + 1000, 1500), (addr + 2500, 500)]]) == 2
    assert sorted(aio.reap(2, 1.0)) == [(1, 1000), (2, 2000)]
    assert os.read(r, 4000) == payload
    assert aio.reap(0, 0.0) == []
  finally:
    aio.close()
    os.close(r)
    os.close(w)


@pytest.mark.real_aio
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux AIO')
def test_a_bad_fd_fails_the_submit_and_queues_nothing():
  aio = Aio(-1 & 0xFFFFFFFF, 4)
  try:
    addr, _ = address(b'x' * 16)
    with pytest.raises(OSError):
      aio.submit(1, [[(addr, 16)]])
  finally:
    aio.close()
