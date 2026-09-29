"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Protocol 3 on the comma's side: the hidden state stays on the server, and a
comma meeting a server of another version stops in one round trip, saying
which side to update.

The server of the protocol before is played from its bytes on the wire: it
answers the hello with its own version, and latches any header version but 2
as a desync, never answering again. tests/test_swift_server.py has the other
pair, a comma of the protocol before against the Swift server.
"""
from __future__ import annotations

import json
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError
from jetlink.transport.tcp import TcpTransport

QUEUED = {'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256), 'desire_pulse': (1, 25, 8),
          'traffic_convention': (1, 2), 'action_t': (1, 2), 'features_buffer': (1, 32, 32, 512)}
# Cinque Terre V3's, as the fork's tests read them off its ONNX
STATEFUL = {'new_img': (2, 6, 128, 256), 'desire': (8,), 'traffic_convention': (1, 2), 'action_t': (1, 2),
            'state_img_q': (2, 5, 6, 128, 256), 'state_desire_q': (132, 1, 8), 'state_feat_q': (128, 1, 16384)}
SLICES = {'plan': slice(917, 1907), 'hidden_state': slice(2066, 18450), 'pad': slice(18450, 18452)}


def big_spec(inputs: dict) -> ModelSpec:
  outputs = {'outputs': (1, 18452), **{f'next_{n}': s for n, s in inputs.items() if n.startswith('state_')}}
  return ModelSpec(sha256='c' * 64, nbytes=765953504, frame_skip=4, input_shapes=inputs, output_shapes=outputs,
                   output_slices=SLICES, checkpoint=None)


def gadget_bytes(nbytes: int) -> int:
  """What the comma's gadget puts on the wire for a message: whole 16 KB bursts."""
  return -(-nbytes // P.GADGET_TX_ALIGN) * P.GADGET_TX_ALIGN


@pytest.mark.parametrize('inputs', [QUEUED, STATEFUL], ids=['queued', 'stateful'])
def test_a_766_mb_model_frame_is_one_comma_read_down(inputs):
  """The sizes transport v3 is for: the reply fits the one 16 KB read the comma
  keeps posted (it was 73,860 B, five), and a queued model's request loses the
  64 KB of prev_feat (475,136 B on the wire before)."""
  spec = big_spec(inputs)
  request = P.HEADER_SIZE + spec.infer_req_nbytes
  reply = P.HEADER_SIZE + spec.infer_resp_nbytes
  assert (request, gadget_bytes(request)) == (393_304, 409_600)
  assert reply == 8_324 and reply <= P.GADGET_TX_ALIGN
  assert P.HEADER_SIZE + P.INFER_RESP_SIZE + spec.output_nbytes == 73_860   # the reply on WANT_HIDDEN


def test_the_envelope_is_framed_as_protocol_2():
  for msg in P.Msg:
    version = struct.unpack_from('<IH', P.pack_header(msg, 1, 0))[1]
    assert version == (2 if msg in (P.Msg.HELLO_REQ, P.Msg.HELLO_RESP, P.Msg.SHUTDOWN_REQ, P.Msg.SHUTDOWN_RESP,
                                    P.Msg.ERROR) else 3), msg.name
  # read whole either way; whether it belongs is the receiver's call
  for version in (2, 3):
    raw = struct.pack(P.HEADER_FMT, P.MAGIC, version, P.Msg.PING, 1, 0, 0, 0)
    assert P.unpack_header(raw)[1] == version
  assert P.same_protocol(3, P.Msg.PING) and P.same_protocol(2, P.Msg.HELLO_RESP)
  assert not P.same_protocol(2, P.Msg.INFER_RESP) and not P.same_protocol(2, P.Msg.PROGRESS)


# -- the reply ------------------------------------------------------------------------

def replying(spec: ModelSpec, floats: np.ndarray, tail: bytes = b'') -> JetlinkClient:
  """A client whose transport answers every frame with `floats`."""
  payload = P.pack_infer_resp(1, P.Status.OK, 0, 0, 0) + floats.astype(np.float32).tobytes() + tail
  transport = SimpleNamespace(send=lambda *a, **kw: None, recv=lambda **kw: SimpleNamespace(
    msg_type=P.Msg.INFER_RESP, seq=1, payload=memoryview(payload), version=P.VERSION))
  client = JetlinkClient(transport, want_hidden=False)
  client.spec = spec
  return client


def frame(client: JetlinkClient, **kw) -> np.ndarray:
  spec = client.spec
  return client.infer(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=1, **kw)


def test_the_output_keeps_its_layout_with_the_hidden_state_left_out():
  """The glue slices by the spec's output_slices, as when the whole vector
  crossed: the reply is expanded back, hidden_state reading as zeros."""
  spec = big_spec(QUEUED)
  reply = np.arange(spec.reply_nelem, dtype=np.float32) + 1
  out = frame(replying(spec, reply))
  assert out.shape == (18_452,)
  np.testing.assert_array_equal(out[:2066], reply[:2066])
  assert not out[2066:18450].any()
  np.testing.assert_array_equal(out[18450:], reply[2066:])


def test_want_hidden_returns_the_whole_vector_as_sent():
  spec = big_spec(STATEFUL)
  whole = np.arange(spec.output_nelem, dtype=np.float32)
  sent = []
  client = replying(spec, whole)
  client.t.send = lambda *a, **kw: sent.append(a)
  client.want_hidden = True
  np.testing.assert_array_equal(frame(client), whole)
  _, flags = P.unpack_infer_req(sent[0][2][0])
  assert flags & P.Flag.WANT_HIDDEN


def test_openpilots_raw_predictions_switch_asks_for_the_hidden_state(monkeypatch):
  monkeypatch.delenv('SEND_RAW_PRED', raising=False)
  assert not JetlinkClient(SimpleNamespace()).want_hidden
  monkeypatch.setenv('SEND_RAW_PRED', '1')
  assert JetlinkClient(SimpleNamespace()).want_hidden


def test_a_reply_with_the_hidden_state_left_in_is_refused():
  """A server that says 3 and answers like 2 would have every float after
  hidden_state misread: refused, not guessed at."""
  spec = big_spec(QUEUED)
  client = replying(spec, np.zeros(spec.output_nelem, np.float32))
  with pytest.raises(LinkError, match='73828 bytes, expected 8292'):
    frame(client)
  assert client.dead


def test_telemetry_still_follows_the_outputs():
  spec = big_spec(QUEUED)
  client = replying(spec, np.zeros(spec.reply_nelem, np.float32), json.dumps({'temp_c': 50}).encode())
  frame(client, want_state=True)
  assert client.last_state == {'temp_c': 50}


def test_the_feedback_is_the_servers_now():
  spec = big_spec(QUEUED)
  assert 'prev_feat' not in spec.packed_shapes
  packed = np.zeros(spec.packed_nelem, np.float32)
  spec.feed_back(packed, np.ones(spec.output_nelem, np.float32))   # what modeld still calls
  assert not packed.any()


# -- a server of the protocol before --------------------------------------------------

class OldServer:
  """The protocol-2 server as the comma meets it on the wire."""

  def __init__(self, first_reply: tuple[int, dict] | None = None):
    listener = TcpTransport.listen('127.0.0.1', 0)
    self.port = listener.getsockname()[1]
    self.seen: list[tuple[int, int]] = []   # (version, type) of each message
    self.first_reply = first_reply
    self.thread = threading.Thread(target=self._serve, args=(listener,), daemon=True)
    self.thread.start()

  def _serve(self, listener) -> None:
    sock, _ = listener.accept()
    listener.close()
    with sock:
      try:
        while True:
          header = self._exactly(sock, P.HEADER_SIZE)
          _, version, msg_type, seq, flags, length, _ = struct.unpack(P.HEADER_FMT, header)
          self.seen.append((version, msg_type))
          if version != 2:
            return   # latched as a desync: nothing is ever answered again
          body = self._exactly(sock, length + (1 if flags & P.Flag.PADDED else 0))[:length]
          if self.first_reply is not None:
            self._send(sock, *self.first_reply, seq)
            self.first_reply = None
          if msg_type == P.Msg.HELLO_REQ:
            self._send(sock, P.Msg.HELLO_RESP, {'protocol': 2, 'backend': 'trt', 'engine_state': 'ready'}, seq)
          elif msg_type == P.Msg.SHUTDOWN_REQ:
            self._send(sock, P.Msg.SHUTDOWN_RESP, {'ok': True, 'detail': json.loads(body)['reason']}, seq)
      except (ConnectionError, OSError):
        return

  @staticmethod
  def _exactly(sock, n: int) -> bytes:
    out = b''
    while len(out) < n:
      chunk = sock.recv(n - len(out))
      if not chunk:
        raise ConnectionError('closed')
      out += chunk
    return out

  @staticmethod
  def _send(sock, msg_type: int, obj: dict, seq: int) -> None:
    body = json.dumps(obj).encode()
    sock.sendall(struct.pack(P.HEADER_FMT, P.MAGIC, 2, msg_type, seq, 0, len(body), 0) + body)

  def client(self) -> JetlinkClient:
    return JetlinkClient(TcpTransport.connect('127.0.0.1', self.port), deadline=2.0)


def test_a_new_comma_stops_at_an_old_server_in_one_round_trip():
  old = OldServer()
  client = old.client()
  try:
    with pytest.raises(LinkError, match='protocol 2 and this comma 3: update jetlink on the Jetson'):
      client.hello(timeout=2.0)
    assert client.dead
  finally:
    client.close()
  old.thread.join(2.0)
  assert old.seen == [(2, P.Msg.HELLO_REQ)]   # the hello, and nothing it could not read


def test_an_old_servers_unsolicited_message_is_refused_by_name_not_as_a_desync():
  """A build's PROGRESS can reach a new client ahead of the hello's answer."""
  old = OldServer(first_reply=(P.Msg.PROGRESS, {'stage': 'build', 'frac': 0.5, 'msg': ''}))
  client = old.client()
  try:
    with pytest.raises(LinkError, match='update jetlink on the Jetson'):
      client.hello(timeout=2.0)
    assert client.dead and not client.t._desynced
  finally:
    client.close()


def test_a_new_comma_can_still_power_an_old_server_off():
  """The low-battery shutdown goes with no hello first, so it travels in the
  envelope every version reads."""
  old = OldServer()
  client = old.client()
  try:
    assert client.shutdown('car battery', timeout=2.0) == {'ok': True, 'detail': 'car battery'}
  finally:
    client.close()
  assert old.seen == [(2, P.Msg.SHUTDOWN_REQ)]
