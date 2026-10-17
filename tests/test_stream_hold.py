"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A stream whose peer has stopped reading: the frame is held, not the link. This
is the comma over Wi-Fi (2026-10-17), where a phone quiet for a moment cost the
link every three minutes -- one 'send failed: timed out (tcp)' after another,
each one a demotion to the small model and a backoff the picture waited out.
"""
import socket
import threading
import unittest

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.transport.tcp import TcpTransport
from tests.test_protocol import _spec


class AStreamThePeerStoppedReading(unittest.TestCase):
  """The sender is the transport under test; the peer is the socket it writes
  to, read here by hand so a test can see every byte that reaches it."""

  BODY = b'\x5a' * 200_000

  def setUp(self):
    srv = TcpTransport.listen('127.0.0.1', 0)
    self.peer = socket.create_connection(srv.getsockname())
    conn, _ = srv.accept()
    srv.close()
    self.addCleanup(self.peer.close)
    self.ours = TcpTransport(conn)
    self.addCleanup(self.ours.close)
    # a phone that stopped draining: its window is what closes first, and this
    # end's send buffer fills behind it
    self.peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
    self.ours.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)

  def wire(self, seq: int, body: bytes | None = None) -> bytes:
    """The message this transport puts on the wire for `seq`."""
    return b''.join(bytes(m) for m in self.ours._frame(P.Msg.PING, seq, (self.BODY if body is None else body,), 0))

  def read_ready(self) -> bytes:
    """What the peer has right now and nothing more: it is not waiting for
    anything, which is the whole trouble."""
    self.peer.setblocking(False)
    out = bytearray()
    try:
      while (chunk := self.peer.recv(1 << 16)):
        out += chunk
    except (BlockingIOError, TimeoutError):
      pass
    finally:
      self.peer.setblocking(True)
    return bytes(out)

  def test_a_message_the_socket_only_half_took_is_committed_and_not_half_on_the_wire(self):
    # the first frame: the socket takes what it can and the transport keeps the
    # rest, because a stream cannot take back what is already out
    self.assertTrue(self.ours.try_send(P.Msg.PING, 1, (self.BODY,)))
    self.assertTrue(self.ours._tx, 'the send buffer took a frame whole, so nothing was held')
    # the second: the peer takes none of it, and None of it is committed
    self.assertFalse(self.ours.try_send(P.Msg.PING, 2, (self.BODY,)))
    seen = self.read_ready()
    self.assertGreater(len(seen), 0, 'the first frame never reached the peer')
    self.assertLess(len(seen), len(self.wire(1)), 'the peer read a frame whole')
    # a prefix of the first frame and not one byte of the held one
    self.assertEqual(seen, self.wire(1)[:len(seen)])

  def test_the_stream_carries_every_frame_the_transport_committed_and_no_other(self):
    # the frame loop: a new frame every time, the peer reading at its own pace
    expect, seen, held, seq = bytearray(), bytearray(), 0, 0
    for _ in range(60):
      seq += 1
      if self.ours.try_send(P.Msg.PING, seq, (self.BODY,)):
        expect += self.wire(seq)
      else:
        held += 1
      seen += self.read_ready()
      # never a byte the transport did not commit, in the order it committed
      # it: the peer reads no frame twice, and none it was told was held
      self.assertEqual(bytes(seen), bytes(expect[:len(seen)]))
    self.assertTrue(expect, 'no frame was ever committed')
    self.assertTrue(held, 'the peer never left a frame to be held')
    # what is owed is the tail of what was committed, and nothing else: the
    # frames that never went out left no gap in the stream
    self.assertTrue(expect.startswith(seen))

  def test_a_send_that_waits_finishes_an_owed_message_first(self):
    # a control message and a frame the socket half took do not interleave: the
    # peer reads one stream, and half a message is not one it can skip
    half = self.BODY[:len(self.BODY) // 2]
    self.assertTrue(self.ours.try_send(P.Msg.PING, 1, (half,)))
    self.assertTrue(self.ours._tx, 'the whole message was taken, so nothing is owed')
    expect = self.wire(1, half) + self.wire(2, half)
    # the phone reads again while the send runs, as its kernel does
    self.peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    self.ours.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    reader = bytearray()

    def drain():
      while len(reader) < len(expect):
        reader.extend(self.read_ready())

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    self.ours.send(P.Msg.PING, 2, (half,), timeout=5.0)
    t.join(5.0)
    self.assertEqual(bytes(reader), expect)


class APreviewOverAStreamThePeerStoppedReading(unittest.TestCase):
  """The comma's side of a preview over Wi-Fi, on a socket the phone has
  stopped draining: the frame is held and the link stays up."""

  def setUp(self):
    srv = TcpTransport.listen('127.0.0.1', 0)
    self.peer = socket.create_connection(srv.getsockname())
    conn, _ = srv.accept()
    srv.close()
    self.addCleanup(self.peer.close)
    self.ours = TcpTransport(conn)
    self.addCleanup(self.ours.close)
    self.ours.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
    self.client = JetlinkClient(self.ours, want_hidden=False)
    self.client.spec = _spec()
    self.client.deadline = 5.0
    self.client.patient = True

  def frame(self, frame_id: int) -> int | None:
    return self.client.infer_begin(bytes(self.client.spec.warped_nbytes), bytes(self.client.spec.packed_nbytes),
                                   frame_id=frame_id, skip_if_busy=True)

  def test_a_frame_the_stream_cannot_take_is_held_and_the_link_survives(self):
    sent = [self.frame(f) for f in range(1, 4)]
    self.assertTrue(any(f is not None for f in sent), 'no frame ever went out')
    self.assertIsNone(sent[-1], 'the peer never made the client hold a frame')
    # the frame the wheel would have handed the link back over: a preview holds
    # it, and what it publishes meanwhile is the newest answer it has
    self.assertFalse(self.client.dead, 'the link went down over a frame the caller can do without')

  def drain(self) -> None:
    """The phone reading again."""
    self.peer.setblocking(False)
    try:
      while self.peer.recv(1 << 16):
        pass
    except (BlockingIOError, TimeoutError):
      pass
    finally:
      self.peer.setblocking(True)

  def test_the_loop_goes_on_once_the_peer_reads_again(self):
    for f in range(1, 4):
      self.frame(f)
    # the phone comes back: its window opens, and the frames the loop keeps
    # offering place again
    self.peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    self.ours.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    placed = 0
    for f in range(4, 20):
      if self.frame(f) is not None:
        placed += 1
      self.drain()
    self.assertGreater(placed, 0, 'the link never placed a frame again after the peer drained it')
    self.assertFalse(self.client.dead)

  def test_a_frame_nothing_was_written_of_never_commits(self):
    # the peer takes none of it: not a byte, and the caller is told to hold
    self.peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2304)
    self.ours.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
    first = self.frame(1)
    self.assertIsNotNone(first)
    self.assertIsNone(self.frame(2), 'the peer had room, so the frame was not held')
    self.assertFalse(self.client.dead)
