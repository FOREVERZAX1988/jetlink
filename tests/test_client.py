"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's client over a live socket pair: the leave, and what a close does
to a connection another process still holds a copy of.
"""
import json
import os
import socket
import threading
import time
import unittest

import numpy as np

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.transport.base import LinkError
from tests.test_protocol import _spec, make_pair, reply


class SocketPairTest(unittest.TestCase):
  def setUp(self):
    self.ours, self.peer = make_pair()
    self.addCleanup(self.ours.close)
    self.addCleanup(self.peer.close)
    self.client = JetlinkClient(self.ours)

  def pong_once(self) -> threading.Thread:
    """The far end answers one PING."""
    def serve():
      msg = self.peer.recv(timeout=2.0)
      assert msg.msg_type == P.Msg.PING
      self.peer.send(P.Msg.PONG, msg.seq)
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return t


class TheLeave(SocketPairTest):
  def test_a_dead_link_carries_no_leave(self):
    self.client.dead = True
    self.client.leave('lost')
    with self.assertRaises(LinkError):
      self.peer.recv(timeout=0.2)

  def test_it_says_why_with_what_was_measured_and_wants_no_answer(self):
    self.client.leave('behind', frames=3, p50_ms=41.2)
    msg = self.peer.recv(timeout=1.0)
    self.assertEqual(msg.msg_type, P.Msg.LEAVE)
    self.assertEqual(json.loads(bytes(msg.payload)), {'reason': 'behind', 'frames': 3, 'p50_ms': 41.2})
    self.assertFalse(self.client.dead)

  def test_an_older_servers_error_to_it_is_not_this_links_failure(self):
    # a v0.8.0 server answers unknown_message; the next exchange, a hello or a
    # ping, reads the socket next and must not take that as its own failure
    self.client.leave('stopped')
    msg = self.peer.recv(timeout=1.0)
    self.peer.send_json(P.Msg.ERROR, msg.seq, {'error': 'unknown_message', 'detail': 'type 19'})
    t = self.pong_once()
    self.client.ping(timeout=2.0)
    t.join(2.0)
    self.assertFalse(self.client.dead)

  def test_an_error_to_anything_else_still_is(self):
    self.peer.send_json(P.Msg.ERROR, 999, {'error': 'no_hello', 'detail': 'say hello again'})
    with self.assertRaises(LinkError):
      self.client.ping(timeout=1.0)


class ClosingTheSocket(SocketPairTest):
  def test_the_peer_hears_a_close_though_another_process_holds_a_copy(self):
    # the comma's owner keeps a copy of the phone's dial for the drive; a
    # plain close here left the phone connected to nobody until the owner let
    # its copy go, at the next join attempt, seconds to a minute later
    copy = socket.socket(fileno=os.dup(self.ours.sock.fileno()))
    self.addCleanup(copy.close)
    self.ours.close()
    with self.assertRaises(LinkError) as closed:
      self.peer.recv(timeout=1.0)
    self.assertIn('peer closed', str(closed.exception))


class FramesInFlight(SocketPairTest):
  """Frames the comma sends without waiting (the small model driving) or stops
  waiting for (a held frame), and how their answers are read later."""

  def setUp(self):
    super().setUp()
    self.client.spec = _spec()
    self.client.deadline = 0.5

  def send(self, frame_id: int) -> int:
    spec = self.client.spec
    return self.client.infer_begin(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=frame_id)

  def answer(self, status=P.Status.OK) -> None:
    """The far end takes one request and answers it with its frame id in
    every output."""
    msg = self.peer.recv(timeout=2.0)
    self.assertEqual(msg.msg_type, P.Msg.INFER_REQ)
    frame_id = P.unpack_infer_req(msg.payload)[0]
    payload = reply(np.full(self.client.spec.reply_nelem, frame_id, np.float32), frame_id=frame_id)
    if status != P.Status.OK:
      payload = P.pack_infer_resp(frame_id, status, 0, 0, 0) + payload[P.INFER_RESP_SIZE:]
    self.peer.send(P.Msg.INFER_RESP, msg.seq, (payload,))

  def test_a_drain_takes_what_has_arrived_and_waits_for_nothing(self):
    t0 = time.monotonic()
    self.assertEqual(self.client.drain(), 0)
    self.assertLess(time.monotonic() - t0, 0.05, 'a drain with nothing to read waited')
    self.send(1)
    self.send(2)
    self.answer()
    self.answer()
    for _ in range(50):   # the replies cross a socket pair; give them a moment
      if self.client.drain():
        break
      time.sleep(0.005)
    self.assertEqual(self.client.waiting_for(), 0.0)
    self.assertEqual(self.client.last_output[0], 2.0, 'the newest reply is what a held frame publishes')
    self.assertFalse(self.client.dead)

  def test_a_frame_given_up_on_is_read_quietly_by_the_next(self):
    seq1 = self.send(1)
    request = self.peer.recv(timeout=2.0)   # not answered yet
    self.assertIsNone(self.client.infer_end(seq1, hold=0.01))
    self.assertFalse(self.client.dead, 'a hold that passed is not a failure')
    self.assertGreater(self.client.waiting_for(), 0.0)
    # the late answer lands, then the next frame goes out and is answered
    frame_id = P.unpack_infer_req(request.payload)[0]
    self.peer.send(P.Msg.INFER_RESP, request.seq, (reply(np.ones(self.client.spec.reply_nelem), frame_id=frame_id),))
    seq2 = self.send(2)
    self.answer()
    with self.assertNoLogs('jetlink.client', level='WARNING'):
      out = self.client.infer_end(seq2)
    self.assertEqual(out[0], 2.0)
    self.assertEqual(self.client.waiting_for(), 0.0)

  def test_a_quiet_host_shows_in_how_long_the_oldest_frame_has_waited(self):
    self.send(1)
    time.sleep(0.02)
    self.assertGreater(self.client.waiting_for(), 0.015)
    self.assertEqual(self.client.drain(), 0)

  def test_a_host_that_answers_nothing_for_the_deadline_fails_the_link_before_the_next_frame(self):
    # whichever model is driving: frames keep going out while the small one
    # does, so a quiet host shows here, not in a wait
    self.client.deadline = 0.05
    self.send(1)
    time.sleep(0.06)
    with self.assertRaises(LinkError) as quiet:
      self.send(2)
    self.assertIn('no answer to 1 frames', str(quiet.exception))
    self.assertTrue(self.client.dead)

  def test_a_frame_the_server_failed_fails_the_link_when_drained(self):
    self.send(1)
    self.answer(status=P.Status.INFER_FAILED)
    time.sleep(0.02)
    with self.assertRaises(LinkError):
      self.client.drain()
    self.assertTrue(self.client.dead)

  def test_a_hello_forgets_the_frames_of_the_session_before(self):
    self.send(1)

    def serve():
      msg = self.peer.recv(timeout=2.0)   # the frame
      msg = self.peer.recv(timeout=2.0)
      assert msg.msg_type == P.Msg.HELLO_REQ
      self.peer.send_json(P.Msg.HELLO_RESP, msg.seq, {'device': 'test'})
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    self.client.hello(timeout=2.0)
    t.join(2.0)
    self.assertEqual(self.client.waiting_for(), 0.0)
    self.assertIsNone(self.client.last_output)


if __name__ == '__main__':
  unittest.main()
