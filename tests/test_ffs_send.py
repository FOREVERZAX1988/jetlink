"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The gadget's AIO send path, against tests/aio_fakes.FakeAio: framing into
aligned requests, room, refusals, failures and closing.
"""
import errno

import pytest

from jetlink import protocol as P
from jetlink.transport import ffs
from jetlink.transport.base import LinkError
from jetlink.transport.ffs import FfsTransport
from tests.aio_fakes import FakeAio

FRAME = 393216 + 65536   # a frame's payload, near the real 459 KB


@pytest.fixture
def sender():
  t = FfsTransport.__new__(FfsTransport)
  t._prepare('/nonexistent', gadget=None)
  t._ensure_epfiles = lambda: None
  t._udc_note = lambda: ''
  t.ep_in = 123
  t._aio = aio = FakeAio(None, ffs.AIO_DEPTH)
  unbound = []

  def unbind(gadget=None):
    unbound.append(gadget)
    aio.shutdown()   # disabling the endpoint completes everything queued

  t.unbind = unbind
  t._unbound = unbound
  return t, aio


def _payload(size: int) -> bytes:
  return bytes(range(256)) * (size // 256) + bytes(range(size % 256))


@pytest.mark.parametrize('size', [0, 1, 16352, 16384, 32768, 393216, FRAME, 4 << 20])
@pytest.mark.parametrize('quantum', [8192, 16384, 32768])
def test_a_message_crosses_whole_in_aligned_requests(sender, size, quantum):
  t, aio = sender
  t.write_chunk = quantum
  payload = _payload(size)
  t.send(P.Msg.INFER_REQ, 7, (payload,), timeout=1.0)
  wire = bytes(aio.wire)
  _, _, kind, seq, _, length, _ = P.unpack_header(wire[:P.HEADER_SIZE])
  assert (kind, seq, length) == (P.Msg.INFER_REQ, 7, size)
  assert wire[P.HEADER_SIZE:P.HEADER_SIZE + size] == payload
  assert not any(wire[P.HEADER_SIZE + size:]), 'padding is zeros'
  assert len(wire) % P.GADGET_TX_ALIGN == 0
  # never a short packet: every request but the last `quantum` bytes, and the
  # last a whole number of packets too
  assert all(n == quantum for n in aio.requests[:-1])
  assert 0 < aio.requests[-1] <= quantum and aio.requests[-1] % P.USB_MAX_PACKET == 0
  if len(wire) <= ffs.QUEUED_LIMIT:
    assert aio.submits == 1, 'a message that fits goes in one io_submit'
  assert t.last_send['bytes'] == len(wire) and t.last_send['requests'] == len(aio.requests)
  assert t._send_deadline is None


def test_parts_are_gathered_without_a_copy_of_the_message(sender):
  # the frame, the packed inputs and the header in their own buffers, as
  # JetlinkClient.infer_begin hands them over
  t, aio = sender
  frame, packed = bytearray(_payload(FRAME)), bytearray(_payload(5000))
  t.send(P.Msg.INFER_REQ, 1, (b'\x01' * 8, frame, packed))
  wire = bytes(aio.wire)
  assert wire[P.HEADER_SIZE:P.HEADER_SIZE + 8 + FRAME + 5000] == b'\x01' * 8 + frame + packed


def test_the_callers_buffers_are_free_once_send_returns(sender):
  # the kernel copies at io_submit, so the next frame's warp may overwrite
  # the readback straight away; this fake copies there too
  t, aio = sender
  frame = bytearray(_payload(FRAME))
  t.send(P.Msg.INFER_REQ, 1, (frame,))
  frame[:] = bytes(len(frame))
  assert bytes(aio.wire[P.HEADER_SIZE:P.HEADER_SIZE + 64]) == _payload(64)


def test_a_large_message_streams_through_bounded_room(sender):
  t, aio = sender
  t.send(P.Msg.UPLOAD_CHUNK, 1, (_payload(4 << 20),), timeout=1.0)
  assert aio.submits > 1
  assert len(aio.wire) >= 4 << 20
  # FakeAio asserts the context never held more than its depth


def test_try_send_refuses_a_third_frame_the_host_has_not_taken(sender):
  t, aio = sender
  aio.holding = True
  assert t.try_send(P.Msg.INFER_REQ, 1, (_payload(FRAME),))
  assert t.try_send(P.Msg.INFER_REQ, 2, (_payload(FRAME),)), 'one frame may go out behind another'
  sent = len(aio.wire)
  assert not t.try_send(P.Msg.INFER_REQ, 3, (_payload(FRAME),))
  assert len(aio.wire) == sent, 'a refused frame sends nothing'
  assert t.last_send['refused'] and t.last_send['backlog_kb'] > 0
  assert t.send_totals['refused'] == 1
  aio.release()
  assert t.try_send(P.Msg.INFER_REQ, 3, (_payload(FRAME),))
  assert not t._unbound, 'refusing is not a failure'


def test_a_blocking_send_waits_for_room_then_drops_the_gadget(sender):
  t, aio = sender
  aio.holding = True
  t.send(P.Msg.INFER_REQ, 1, (_payload(FRAME),))
  t.send(P.Msg.INFER_REQ, 2, (_payload(FRAME),))
  with pytest.raises(LinkError, match='took no USB data'):
    t.send(P.Msg.INFER_REQ, 3, (_payload(FRAME),), timeout=0.05)
  assert t._unbound and t.last_send['aborted']
  assert t.send_totals['aborts'] == 1
  with pytest.raises(LinkError, match='link abandoned'):
    t.send(P.Msg.PING, 4)


def test_a_write_the_host_did_not_complete_fails_the_next_send(sender):
  t, aio = sender
  aio.result = lambda token, n: -errno.EPIPE
  t.send(P.Msg.PING, 1)
  with pytest.raises(LinkError, match='gadget write failed'):
    t.send(P.Msg.PING, 2)
  with pytest.raises(LinkError, match='gadget write failed'):
    t.try_send(P.Msg.PING, 3)


def test_enomem_halves_the_request_size_and_repeats_nothing(sender):
  t, aio = sender
  t.write_chunk = 2 * P.GADGET_TX_ALIGN
  aio.fail.append(errno.ENOMEM)
  payload = _payload(100000)
  t.send(P.Msg.INFER_REQ, 1, (payload,))
  t.send(P.Msg.INFER_REQ, 2, (payload,))
  assert t.write_chunk == P.GADGET_TX_ALIGN, 'the smaller size holds for the session'
  wire = bytes(aio.wire)
  half = len(wire) // 2
  assert wire[P.HEADER_SIZE:P.HEADER_SIZE + len(payload)] == payload
  assert wire[half + P.HEADER_SIZE:half + P.HEADER_SIZE + len(payload)] == payload
  assert set(aio.requests) == {P.GADGET_TX_ALIGN}
  assert t.send_totals['enomem'] == 1


def test_enomem_at_the_smallest_size_is_a_link_error(sender):
  t, aio = sender
  aio.fail.append(errno.ENOMEM)   # 8 KB is already under the 16 KB floor
  with pytest.raises(LinkError, match='gadget write failed'):
    t.send(P.Msg.PING, 1)
  assert t.last_send['errno'] == errno.ENOMEM
  assert not t._unbound, 'nothing of the message was queued, so the host has nothing to discard'


def test_an_interrupted_submit_is_retried_and_repeats_nothing(sender):
  t, aio = sender
  aio.fail.append(errno.EINTR)
  payload = _payload(50000)
  t.send(P.Msg.INFER_REQ, 1, (payload,))
  assert aio.submits == 2
  assert bytes(aio.wire[P.HEADER_SIZE:P.HEADER_SIZE + len(payload)]) == payload


def test_a_failure_past_the_first_byte_drops_the_gadget(sender):
  # the host has half a message, which only a re-enumeration takes away
  t, aio = sender
  submit = aio.submit
  calls = []

  def second_fails(requests):
    calls.append(len(requests))
    if len(calls) == 2:
      raise OSError(errno.EIO, 'I/O error')
    return submit(requests)

  aio.submit = second_fails
  with pytest.raises(LinkError, match='gadget write failed'):
    t.send(P.Msg.UPLOAD_CHUNK, 1, (_payload(4 << 20),), timeout=1.0)
  assert t._unbound


def test_close_lets_queued_writes_finish_then_destroys_the_context(sender):
  t, aio = sender
  t.send(P.Msg.LEAVE, 1, (b'{}',))
  t._close_fds = lambda names, reader: None
  t.close()
  assert aio.closed and not t._write_aborted and t._aio is None


def test_close_drops_the_gadget_under_writes_the_host_never_takes(sender, monkeypatch):
  monkeypatch.setattr(ffs, 'CLOSE_FLUSH', 0.02)
  t, aio = sender
  aio.holding = True
  t.send(P.Msg.INFER_REQ, 1, (_payload(FRAME),))
  t._close_fds = lambda names, reader: None
  t.close()
  assert t._write_aborted and t._unbound, 'only the unbind completes them'
  assert aio.closed, 'destroyed once they completed'


def test_close_leaves_a_context_it_cannot_empty_to_the_process(sender, monkeypatch):
  # io_destroy would block on requests nothing completes
  monkeypatch.setattr(ffs, 'CLOSE_FLUSH', 0.02)
  monkeypatch.setattr(ffs, 'ABORT_DRAIN', 0.02)
  t, aio = sender
  t.unbind = lambda gadget=None: t._unbound.append(gadget)   # an unbind that frees nothing
  aio.holding = True
  t.send(P.Msg.INFER_REQ, 1, (_payload(FRAME),))
  t._close_fds = lambda names, reader: None
  t.close()
  assert not aio.closed and t._aio is None


def test_a_closing_transport_refuses_to_send(sender):
  t, _ = sender
  t._closing = True
  with pytest.raises(LinkError, match='closing'):
    t.send(P.Msg.PING, 1)


def test_send_totals_keep_the_maxima_between_log_samples(sender):
  t, aio = sender
  for seq in range(3):
    t.send(P.Msg.INFER_REQ, seq, (_payload(FRAME),))
  totals = t.send_totals
  assert totals['messages'] == 3 and totals['requests'] == len(aio.requests)
  assert totals['bytes'] == len(aio.wire)
  assert {'max_submit_ms', 'max_wait_ms', 'max_backlog_ms'} <= totals.keys()


@pytest.mark.parametrize('start', range(0, 7 * 16384, 16384))
def test_requests_cover_the_message_from_any_start_without_overlap(start):
  spans = [(1000, 32), (5000, 5 * 16384), (900000, 2 * 16384 - 32)]
  total = sum(n for _, n in spans)
  reqs = ffs._requests(spans, start, 32768)
  assert sum(n for n, _ in reqs) == total - start
  assert all(n == sum(length for _, length in req) for n, req in reqs)
  assert all(n == 32768 for n, _ in reqs[:-1])
  # the first byte asked for is the one at `start`
  first = reqs[0][1][0][0]
  at = 0
  for addr, n in spans:
    if at + n > start:
      assert first == addr + start - at
      break
    at += n
