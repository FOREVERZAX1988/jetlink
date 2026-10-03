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
import unittest

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.transport.base import LinkError
from jetlink.transport.tcp import TcpTransport


class SocketPairTest(unittest.TestCase):
  def setUp(self):
    a, b = socket.socketpair()
    self.ours, self.peer = TcpTransport(a), TcpTransport(b)
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


if __name__ == '__main__':
  unittest.main()
