"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Protocol 3 on the comma's side: the hidden state stays on the server, the
frames are smaller, and a header of any other version is a broken stream.
"""
from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError
from tests.test_protocol import make_pair

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


def test_a_header_of_another_version_is_a_broken_stream():
  """No special case: the stream is latched as desynced, as for any bad header,
  and the caller reopens the link."""
  a, b = make_pair()
  try:
    a.sock.sendall(struct.pack(P.HEADER_FMT, P.MAGIC, P.VERSION - 1, P.Msg.HELLO_RESP, 1, 0, 2, 0) + b'{}')
    with pytest.raises(LinkError, match='protocol error'):
      b.recv(timeout=5)
    with pytest.raises(LinkError, match='desynced'):
      b.recv(timeout=5)
  finally:
    a.close()
    b.close()


# -- the reply ------------------------------------------------------------------------

def replying(spec: ModelSpec, floats: np.ndarray, tail: bytes = b'') -> JetlinkClient:
  """A client whose transport answers every frame with `floats`."""
  payload = P.pack_infer_resp(1, P.Status.OK, 0, 0, 0) + floats.astype(np.float32).tobytes() + tail
  transport = SimpleNamespace(send=lambda *a, **kw: None, recv=lambda **kw: SimpleNamespace(
    msg_type=P.Msg.INFER_RESP, seq=1, payload=memoryview(payload)))
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
