"""Exercise the real framing and FFS send lifecycle without a USB device."""
import errno
import signal
from types import SimpleNamespace

import pytest

from jetlink import protocol as P
from jetlink.transport import ffs
from jetlink.transport.base import LinkError, StreamTransport
from jetlink.transport.ffs import FfsTransport


@pytest.fixture
def sender(monkeypatch):
  t = FfsTransport.__new__(FfsTransport)
  StreamTransport.__init__(t)
  t._ensure_epfiles = lambda: None
  t._udc_note = lambda: ''
  t._write_aborted = False
  t._had_host = False
  t.send_totals = {}
  t.ep_in = 123
  calls, chunks = [], []
  t._write_guard = SimpleNamespace(arm=lambda budget: calls.append(('arm', budget)) or True,
                                  disarm=lambda: calls.append(('disarm',)) or True)
  monkeypatch.setattr(ffs.signal, 'pthread_sigmask', lambda how, mask: calls.append(('mask', how)) or set())

  def write(fd, bufs):
    assert fd == 123
    chunks.append(b''.join(bufs))
    return len(chunks[-1])

  monkeypatch.setattr(ffs, 'os', SimpleNamespace(writev=write))
  return t, calls, chunks


@pytest.mark.parametrize('size', [0, 1, 16352, 16384, 32768, 393216, 458752, 4 << 20])
@pytest.mark.parametrize('quantum', [16384, 32768])
def test_chunked_message_preserves_bytes_and_guards_once(sender, size, quantum):
  t, calls, chunks = sender
  t.write_chunk = quantum
  payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
  t.send(P.Msg.INFER_REQ, 7, (payload,), timeout=.2)
  wire = b''.join(chunks)
  _, _, kind, seq, _, length, _ = P.unpack_header(wire[:P.HEADER_SIZE])
  assert (kind, seq, length) == (P.Msg.INFER_REQ, 7, size)
  assert wire[P.HEADER_SIZE:P.HEADER_SIZE + size] == payload
  assert not any(wire[P.HEADER_SIZE + size:])
  assert all(0 < len(c) <= quantum and len(c) % P.GADGET_TX_ALIGN == 0 for c in chunks)
  assert [c[0] for c in calls] == ['mask', 'arm', 'disarm', 'mask']
  assert calls[-1] == ('mask', signal.SIG_SETMASK)
  assert t.last_send['bytes'] == len(wire)
  assert t.last_send['writes'] == len(chunks)
  assert t._send_deadline is None


def test_shrunk_quantum_persists_and_never_retries_large_allocation(sender, monkeypatch):
  t, _, chunks = sender
  t.write_chunk = 32768
  write = ffs.os.writev
  attempted = []

  def allocate(fd, bufs):
    n = sum(b.nbytes for b in bufs)
    attempted.append(n)
    if n > 16384:
      raise OSError(errno.ENOMEM, 'fragmented')
    return write(fd, bufs)

  monkeypatch.setattr(ffs.os, 'writev', allocate)
  for seq in (1, 2):
    t.send(P.Msg.INFER_REQ, seq, (bytes(40000),))
  assert attempted.count(32768) == 1
  assert t.write_chunk == 16384
  assert len(chunks) == 6
  assert t.send_totals['messages'] == 2
  assert t.send_totals['enomem'] == 1
  assert t.send_totals['writes'] == 7
  assert t.send_totals['bytes'] == sum(len(chunk) for chunk in chunks)


def test_minimum_size_enomem_fails_without_spinning(sender, monkeypatch):
  t, calls, _ = sender
  t.write_chunk = P.GADGET_TX_ALIGN

  def fail(*args):
    raise OSError(errno.ENOMEM, 'no memory')

  monkeypatch.setattr(ffs.os, 'writev', fail)
  with pytest.raises(LinkError, match='gadget write failed') as exc:
    t.send(P.Msg.INFER_REQ, 1, (bytes(40000),))
  assert isinstance(exc.value.__cause__, OSError)
  assert t.last_send['writes'] == t.last_send['enomem'] == 1
  assert t.last_send['errno'] == errno.ENOMEM
  assert calls[-1] == ('mask', signal.SIG_SETMASK)


def test_deadline_is_shared_by_all_chunks(sender, monkeypatch):
  t, calls, chunks = sender
  clock = [1.0]
  monkeypatch.setattr(ffs.time, 'monotonic', lambda: clock[0])
  write = ffs.os.writev

  def slow(fd, bufs):
    clock[0] += .06
    return write(fd, bufs)

  monkeypatch.setattr(ffs.os, 'writev', slow)
  with pytest.raises(LinkError, match='timed out'):
    t.send(P.Msg.INFER_REQ, 1, (bytes(100000),), timeout=.1)
  assert len(chunks) == 2
  assert len([c for c in calls if c[0] == 'arm']) == 1
  assert t._send_deadline is None


def test_guard_expiry_wins_even_when_last_write_returns(sender):
  t, calls, _ = sender
  t._write_guard.disarm = lambda: False
  with pytest.raises(LinkError, match='exceeded deadline'):
    t.send(P.Msg.PING, 1)
  assert t.last_send['aborted']
  assert calls[-1] == ('mask', signal.SIG_SETMASK)


def test_no_more_chunks_after_abort(sender, monkeypatch):
  t, _, chunks = sender
  write = ffs.os.writev

  def abort(fd, bufs):
    t._write_aborted = True
    return write(fd, bufs)

  monkeypatch.setattr(ffs.os, 'writev', abort)
  with pytest.raises(LinkError, match='exceeded deadline'):
    t.send(P.Msg.INFER_REQ, 1, (bytes(40000),))
  assert len(chunks) == 1


def test_short_writes_advance_without_replaying_bytes(sender, monkeypatch):
  t, _, chunks = sender

  def short(fd, bufs):
    chunks.append(b''.join(bufs)[:1024])
    return len(chunks[-1])

  monkeypatch.setattr(ffs.os, 'writev', short)
  payload = bytes(range(256)) * 200
  t.send(P.Msg.INFER_REQ, 1, (payload,))
  assert b''.join(chunks)[P.HEADER_SIZE:P.HEADER_SIZE + len(payload)] == payload


def test_serialization_failure_never_arms_the_guard(sender):
  t, calls, _ = sender
  with pytest.raises(TypeError):
    t.send(P.Msg.PING, 1, (object(),))
  assert calls == []


def test_closed_guard_restores_signals_without_writing(sender):
  t, calls, chunks = sender
  t._write_guard.arm = lambda budget: False
  with pytest.raises(LinkError, match='expired or closed'):
    t.send(P.Msg.PING, 1)
  assert chunks == []
  assert calls[-1] == ('mask', signal.SIG_SETMASK)


def test_default_deadline_covers_the_entire_message(sender, monkeypatch):
  t, _, chunks = sender
  clock = [1.0]
  monkeypatch.setattr(ffs.time, 'monotonic', lambda: clock[0])
  monkeypatch.setattr(ffs, 'WRITE_TIMEOUT', .1)
  write = ffs.os.writev

  def slow(fd, bufs):
    clock[0] += .06
    return write(fd, bufs)

  monkeypatch.setattr(ffs.os, 'writev', slow)
  with pytest.raises(LinkError, match='timed out'):
    t.send(P.Msg.INFER_REQ, 1, (bytes(100000),))
  assert len(chunks) == 2


def test_an_active_host_error_is_never_retried(sender, monkeypatch):
  t, _, _ = sender
  t._had_host = True

  def fail(*args):
    raise OSError(errno.EIO, 'partial transfer possible')

  monkeypatch.setattr(ffs.os, 'writev', fail)
  with pytest.raises(LinkError, match='gadget write failed'):
    t.send(P.Msg.INFER_REQ, 1)
  assert t.last_send['writes'] == 1


def test_disarm_exception_still_restores_signals(sender):
  t, calls, _ = sender

  def fail():
    raise RuntimeError('disarm failed')

  t._write_guard.disarm = fail
  with pytest.raises(RuntimeError, match='disarm failed'):
    t.send(P.Msg.PING, 1)
  assert calls[-1] == ('mask', signal.SIG_SETMASK)
  assert t._send_deadline is None


def test_arm_exception_restores_signals_and_clears_deadline(sender):
  t, calls, chunks = sender

  def fail(_budget):
    raise RuntimeError('cannot arm')

  t._write_guard.arm = fail
  with pytest.raises(RuntimeError, match='cannot arm'):
    t.send(P.Msg.PING, 1)
  assert chunks == []
  assert calls[-1] == ('mask', signal.SIG_SETMASK)
  assert t._send_deadline is None


def test_prehost_retry_cannot_extend_deadline(sender, monkeypatch):
  t, _, _ = sender
  clock = [1.0]
  monkeypatch.setattr(ffs.time, 'monotonic', lambda: clock[0])

  def fail(*args):
    clock[0] += .06
    raise OSError(errno.EIO, 'host not ready')

  t._wait_for_host_ready = lambda: True
  monkeypatch.setattr(ffs.os, 'writev', fail)
  with pytest.raises(LinkError, match='timed out'):
    t.send(P.Msg.PING, 1, timeout=.1)
  assert t.last_send['writes'] == 2
