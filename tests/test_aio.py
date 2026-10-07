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


def test_a_context_fills_its_structs_where_the_kernel_reads_them(monkeypatch):
  # the real context only builds on Linux, so a mistake in its setup reached a
  # comma with every test here green (2026-10-06: a view made before its array
  # failed every link); this builds it anywhere, with the syscalls faked
  from types import SimpleNamespace

  from jetlink.transport import aio as module
  monkeypatch.setattr(module.platform, 'machine', lambda: 'aarch64')
  monkeypatch.setattr(module.ctypes, 'CDLL', lambda *a, **k: SimpleNamespace(syscall=SimpleNamespace(restype=None)))
  done = []

  def call(self, nr, *args):
    if nr == self._submit:
      return args[1].value
    if nr == self._getevents:
      for i, res in enumerate(done):
        self._events[i].res = res
      return len(done)
    return 0

  monkeypatch.setattr(Aio, '_call', call)
  a = Aio(7, 4)
  payload = bytes(range(100))
  addr, _ = address(payload)
  assert a.submit([[(addr, 60)], [(addr + 60, 30), (addr + 90, 10)]]) == 2
  first, second = a._iocbs[0], a._iocbs[1]
  assert (first.nbytes, first.fildes, first.opcode) == (1, 7, 8)
  assert second.nbytes == 2
  iovecs = [(v.base, v.len) for v in a._iovecs[8:10]]
  assert iovecs == [(addr + 60, 30), (addr + 90, 10)]
  assert ctypes.string_at(a._iovecs[0].base, a._iovecs[0].len) == payload[:60]
  done.extend([60, -108])
  assert a.reap(2, 0.0) == [60, -108]


@pytest.mark.real_aio
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux AIO')
def test_requests_reach_the_fd_in_order_and_are_reaped():
  r, w = os.pipe()
  aio = Aio(w, 8)
  try:
    payload = os.urandom(3000)
    addr, _ = address(payload)
    assert aio.submit([[(addr, 1000)], [(addr + 1000, 1500), (addr + 2500, 500)]]) == 2
    assert sorted(aio.reap(2, 1.0)) == [1000, 2000]
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
      aio.submit([[(addr, 16)]])
  finally:
    aio.close()
