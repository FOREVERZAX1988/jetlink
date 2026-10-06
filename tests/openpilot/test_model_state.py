"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the comma puts on the wire for each kind of big model, and what modeld
reads off the model.

Both kinds get the frame and the scalars only. A stateful graph (openpilot
#38916, Cinque Terre V3 on) keeps its hidden state inside itself, and the
Jetson feeds a queued one's (up to Cinque Terre V2) back (jetlink protocol 3).
The warp and the link are faked, and tinygrad is numpy; the packing is the
code that drives.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from jetlink.openpilot import model_state
from jetlink.spec import ModelSpec
from tests.openpilot import fakes

# Cinque Terre V3's output layout, read off its ONNX
SLICES = {'lane_lines': (0, 528), 'lane_lines_prob': (528, 536), 'road_edges': (536, 800), 'meta': (800, 855),
          'desire_pred': (855, 887), 'pose': (887, 899), 'wide_from_device_euler': (899, 905),
          'road_transform': (905, 917), 'plan': (917, 1907), 'lead': (1907, 2051), 'lead_prob': (2051, 2054),
          'desire_state': (2054, 2062), 'action': (2062, 2066), 'hidden_state': (2066, 18450), 'pad': (18450, 18452)}
STATEFUL = {
  'new_img': (2, 6, 128, 256), 'desire': (8,), 'traffic_convention': (1, 2), 'action_t': (1, 2),
  'state_img_q': (2, 5, 6, 128, 256), 'state_desire_q': (132, 1, 8), 'state_feat_q': (128, 1, 16384),
}
QUEUED = {
  'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256), 'desire_pulse': (1, 25, 8),
  'traffic_convention': (1, 2), 'action_t': (1, 2), 'features_buffer': (1, 32, 32, 512),
}


def spec_for(inputs: dict) -> ModelSpec:
  outputs = {'outputs': (1, 18452)}
  outputs.update({f'next_{n}': s for n, s in inputs.items() if n.startswith('state_')})
  return ModelSpec(sha256='a' * 64, nbytes=1, frame_skip=4, input_shapes=inputs, output_shapes=outputs,
                   output_slices={k: slice(*v) for k, v in SLICES.items()}, checkpoint=None)


class FakeClient:
  def __init__(self, kind: str = 'usb'):
    self.sent = []
    self.asked = []
    self.last_timings = (0, 0, 0)
    self.last_state = None
    self.dead = False
    # what the transport tells the server's hello: 'usb', or 'cable' for a phone
    self.t = SimpleNamespace(link_info=lambda: {'kind': kind})
    self.output = np.zeros(18452, np.float32)
    self.output[slice(*SLICES['hidden_state'])] = 0.5
    self.last_output = None
    self._in_flight = []
    # frames (by seq) whose reply is not back within the hold, and the holds
    # end() asked for
    self.late = set()
    self.holds = []
    self.drains = 0
    # frames (by frame_id) a drain leaves unanswered: a reply not back yet
    self.keep: set[int] = set()
    # what a reply asked for telemetry carries, when the server has any
    self.telemetry = None

  def infer_begin(self, data, packed, frame_id, reset=False, want_state=False):
    self.sent.append((np.frombuffer(bytes(data), np.uint8).copy(), np.array(packed, copy=True), frame_id, reset))
    self.asked.append(want_state)
    self._in_flight.append(frame_id)
    return frame_id

  def infer_end(self, seq, hold=None):
    self.holds.append(hold)
    if hold is not None and (seq in self.late or seq in self.keep):
      return None
    self._in_flight.remove(seq)
    self.last_output = self.output
    self._piggyback()
    return self.output

  def drain(self):
    # replies come in order: none past the first that is not back yet
    self.drains += 1
    back = next((i for i, f in enumerate(self._in_flight) if f in self.keep), len(self._in_flight))
    taken, self._in_flight = self._in_flight[:back], self._in_flight[back:]
    if taken:
      self.last_output = self.output
      self._piggyback()
    return len(taken)

  def _piggyback(self):
    if self.telemetry is not None and self.asked[-1]:
      self.last_state = self.telemetry

  @property
  def unanswered(self):
    return len(self._in_flight)


class ModelStateTest(unittest.TestCase):
  def setUp(self):
    # a phone's transport asks the UDC how fast the cable is
    fakes.isolate(self, Path(tempfile.mkdtemp()))
    p = mock.patch.dict(sys.modules, fakes.fake_tinygrad())
    p.start()
    self.addCleanup(p.stop)
    self.log = fakes.RecordingLog()
    self.events = []

  def make(self, spec, client, face=fakes.FACE):
    return model_state.JetlinkModelState(1928, 1208, client, spec, object(), face=face, log=self.log,
                                         event=lambda name, **fields: self.events.append((name, fields)))

  def run_frames(self, inputs: dict, n: int = 3, client=None, warp_output=None, after_enqueue=None):
    """`n` frames driven. Returns the spec, the state, the client and the
    warped frame every frame carried."""
    spec = spec_for(inputs)
    client = client or FakeClient()
    warped = np.arange(np.prod(spec.warped_shape), dtype=np.uint64).astype(np.uint8)
    warp_output = warp_output or SimpleNamespace(data=lambda: warped)
    with mock.patch.object(model_state, 'call_warp', return_value=warp_output):
      state = self.make(spec, client)
      self.frames(state, n, after_enqueue)
    return spec, state, client, warped

  @staticmethod
  def warping(warped):
    """The warp mocked, for frames run after run_frames() returned."""
    return mock.patch.object(model_state, 'call_warp', return_value=SimpleNamespace(data=lambda: warped))

  @staticmethod
  def frames(state, n: int, after_enqueue=None) -> list:
    bufs = {k: SimpleNamespace(data=np.zeros(8, np.uint8)) for k in ('img', 'big_img')}
    outs = []
    for i in range(n):
      desire = np.zeros(8, np.float32)
      desire[3] = 1.0 if i >= 1 else 0.0   # held from frame 1: a pulse on 1 only
      args = (bufs, {'img': np.eye(3), 'big_img': np.eye(3)},
              {'desire_pulse': desire, 'traffic_convention': np.array([1, 0], np.float32),
               'action_t': np.array([0.1, 0.2], np.float32)})
      outs.append(state.run(*args, after_enqueue))
    return outs


class TestWire(ModelStateTest):
  def test_a_stateful_model_gets_the_frame_and_twelve_floats(self):
    spec, state, client, warped = self.run_frames(STATEFUL)
    self.assertEqual(state.vision_input_names, ['img', 'big_img'])
    self.assertNotIn('prev_feat', state.npy)
    for i, (data, packed, frame_id, reset) in enumerate(client.sent):
      self.assertEqual(frame_id, i + 1)
      self.assertEqual(reset, i == 0)
      np.testing.assert_array_equal(data, warped)
      self.assertEqual(packed.shape, (12,))
      np.testing.assert_array_equal(packed[8:], np.array([1, 0, 0.1, 0.2], np.float32))
    # the desire pulse is the rising edge, as openpilot's own ModelState sends it
    self.assertEqual([p[3] for _, p, _, _ in client.sent], [0.0, 1.0, 0.0])

  def test_over_the_cable_the_frame_goes_out_of_the_gpu_mapping(self):
    # the socket copies the mapping while the first segments are on the wire;
    # no host copy first
    client = FakeClient('cable')
    spec = spec_for(STATEFUL)
    frame = np.arange(np.prod(spec.warped_shape), dtype=np.uint64).astype(np.uint8)
    mapping = SimpleNamespace(as_memoryview=mock.Mock(return_value=memoryview(frame)))
    warp_output = SimpleNamespace(data=mock.Mock(side_effect=AssertionError('copied on the host')),
                                  _buffer=lambda: mapping)
    _, state, client, _ = self.run_frames(STATEFUL, client=client, warp_output=warp_output)
    self.assertTrue(state.send_from_gpu)
    mapping.as_memoryview.assert_called_with(allow_zero_copy=True)
    for data, *_ in client.sent:
      np.testing.assert_array_equal(data, frame)

  def test_every_frame_is_measured_for_the_leave(self):
    # the whole frame as modeld waits on it, against the server's own total;
    # the phone's log carries only the latter without this
    spec, state, client, warped = self.run_frames(STATEFUL)
    self.assertEqual(state.trips.frames, 3)
    summary = state.trips.summary()
    self.assertEqual(summary['frames'], 3)
    self.assertEqual(summary['over'], 0)
    for key in ('span_s', 'p50_ms', 'p99_ms', 'max_ms', 'server_ms'):
      self.assertIn(key, summary)

  def test_trips_summarise_the_frames_kept(self):
    trips = model_state.Trips()
    for _ in range(99):
      trips.record(0.030, 25_000)
    trips.record(0.200, 25_000)
    summary = trips.summary()
    self.assertEqual((summary['frames'], summary['over']), (100, 1))
    self.assertEqual((summary['p50_ms'], summary['max_ms'], summary['server_ms']), (30.0, 200.0, 25.0))
    self.assertEqual(summary['p99_ms'], 30.0, 'p99 is the 99th of a hundred, as the server takes it')
    self.assertEqual(model_state.Trips().summary(), {'frames': 0, 'over': 0, 'held': 0, 'span_s': 0.0})

  def test_usb_keeps_the_host_copy(self):
    _, state, _, _ = self.run_frames(STATEFUL)
    self.assertFalse(state.send_from_gpu)

  def test_the_cable_is_a_phone_s_socket_as_the_transport_says(self):
    from jetlink.transport.tcp import CABLE_ADDRESS, TcpTransport
    sock = mock.Mock()
    sock.getsockname.return_value = (CABLE_ADDRESS, 5599)
    sock.getpeername.return_value = ('192.168.60.3', 50000)
    client = FakeClient()
    client.t = TcpTransport(sock)
    state = self.make(spec_for(STATEFUL), client)
    self.assertTrue(state.send_from_gpu)

  def test_a_queued_model_gets_the_frame_and_twelve_floats_too(self):
    spec, state, client, warped = self.run_frames(QUEUED)
    self.assertNotIn('prev_feat', state.npy)
    for data, packed, _, _ in client.sent:
      np.testing.assert_array_equal(data, warped)
      self.assertEqual(packed.shape, (12,))
      np.testing.assert_array_equal(packed[8:], np.array([1, 0, 0.1, 0.2], np.float32))

  def test_the_warp_is_sized_from_either_layout(self):
    for inputs in (STATEFUL, QUEUED):
      self.assertEqual(spec_for(inputs).model_hw, (128, 256))

  def test_the_first_frames_and_slow_ones_are_timed_in_the_log(self):
    self.run_frames(STATEFUL, n=4)
    timed = [line for line in self.log.lines('warning') if ' warp ' in line]
    self.assertEqual([line.split()[2] for line in timed], ['1', '2', '3'])


class TestProving(ModelStateTest):
  """The second after a swap is the large model's proof: a single held frame
  in its first PROVING_FRAMES is behind, before anyone can engage."""

  def late_at(self, late: set[int], n: int):
    client = FakeClient()
    client.late = late
    spec = spec_for(STATEFUL)
    warped = SimpleNamespace(data=lambda: np.zeros(np.prod(spec.warped_shape), np.uint8))
    behind = []
    with mock.patch.object(model_state, 'call_warp', return_value=warped):
      state = self.make(spec, client)
      for _ in range(n):
        self.frames(state, 1)
        behind.append(state.behind)
    return behind

  def test_a_held_frame_while_proving_is_behind(self):
    behind = self.late_at({7}, n=8)
    self.assertEqual(behind[:6], [None] * 6)
    self.assertEqual(behind[6], f'held frame 7 of the first {model_state.PROVING_FRAMES}')
    self.assertIsNone(behind[7], 'said once, for the frame that held')

  def test_the_last_proving_frame_still_counts_and_the_one_after_is_weather(self):
    n = model_state.PROVING_FRAMES
    self.assertIsNotNone(self.late_at({n}, n=n)[-1])
    self.assertEqual(self.late_at({n + 1}, n=n + 1), [None] * (n + 1))

  def test_a_host_that_keeps_up_proves_without_a_word(self):
    self.assertEqual(self.late_at(set(), n=model_state.PROVING_FRAMES + 5),
                     [None] * (model_state.PROVING_FRAMES + 5))


class TestHold(ModelStateTest):
  """A reply not back HOLD_FRAME into the frame is not waited for: the
  previous output goes out again and the camera frame is not dropped."""

  def test_a_late_reply_publishes_the_previous_output_once(self):
    client = FakeClient()
    client.late = {2}
    spec = spec_for(STATEFUL)
    warped = SimpleNamespace(data=lambda: np.zeros(np.prod(spec.warped_shape), np.uint8))
    with mock.patch.object(model_state, 'call_warp', return_value=warped):
      state = self.make(spec, client)
      client.output[slice(*SLICES['plan'])] = 1.0
      first, = self.frames(state, 1)
      client.output = client.output.copy()
      client.output[slice(*SLICES['plan'])] = 2.0
      held, after = self.frames(state, 2)
    self.assertTrue((first['plan'] == 1.0).all())
    self.assertTrue((held['plan'] == 1.0).all(), 'the frame before, again')
    self.assertTrue((after['plan'] == 2.0).all(), 'and the next frame its own')
    self.assertEqual(state.trips.held, 1)
    self.assertTrue(any('held' in line for line in self.log.lines('warning')))

  def test_the_hold_is_the_rest_of_the_frame_budget(self):
    _, state, client, _ = self.run_frames(STATEFUL, n=3)
    self.assertIsNone(client.holds[0], 'nothing to publish again yet: the first frame waits')
    for hold in client.holds[1:]:
      self.assertGreater(hold, 0.0)
      self.assertLessEqual(hold, model_state.HOLD_FRAME)

  def test_with_the_hold_off_every_frame_waits(self):
    with mock.patch.object(model_state, 'HOLD_FRAME', None):
      _, _, client, _ = self.run_frames(STATEFUL, n=3)
    self.assertEqual(client.holds, [None, None, None])

  def holding(self, late, n: int, **patches):
    """`n` driven frames, the replies to `late` seqs (a predicate) too late
    to wait for. The state, and what `behind` said after each frame. Past
    the proof unless `patches` say otherwise (TestProving)."""
    patches.setdefault('PROVING_FRAMES', 0)
    client = FakeClient()
    client.late = {seq for seq in range(1, n + 1) if late(seq)}
    spec = spec_for(STATEFUL)
    warped = SimpleNamespace(data=lambda: np.zeros(np.prod(spec.warped_shape), np.uint8))
    behind = []
    with mock.patch.object(model_state, 'call_warp', return_value=warped), contextlib.ExitStack() as patched:
      patched.enter_context(mock.patch.multiple(model_state, **patches))
      state = self.make(spec, client)
      for _ in range(n):
        self.frames(state, 1)
        behind.append(state.behind)
    return state, behind

  def test_holds_in_a_row_say_behind(self):
    # the first frame has nothing to hold, so it waits; the five after are held
    state, behind = self.holding(lambda seq: True, n=1 + model_state.HOLDS_IN_A_ROW)
    self.assertEqual(behind[:-1], [None] * model_state.HOLDS_IN_A_ROW)
    self.assertEqual(behind[-1], f'held {model_state.HOLDS_IN_A_ROW} frames in a row')
    self.assertEqual(state.trips.held, model_state.HOLDS_IN_A_ROW)

  def test_a_frame_that_is_not_held_ends_the_run(self):
    run = model_state.HOLDS_IN_A_ROW - 1
    _, behind = self.holding(lambda seq: seq != 1 and seq != run + 2, n=1 + 2 * run + 1)
    self.assertEqual(behind, [None] * len(behind))

  def test_holds_spread_over_the_window_say_behind_past_the_allowance(self):
    allowed = model_state.HOLDS_ALLOWED
    _, behind = self.holding(lambda seq: seq % 2 == 0, n=2 * allowed + 2)   # never two in a row
    self.assertEqual(behind[:2 * allowed + 1], [None] * (2 * allowed + 1), 'the allowance is forgiven')
    self.assertEqual(behind[-1], f'held {allowed + 1} frames in {model_state.HOLD_WINDOW:.0f} s')

  def test_holds_further_apart_than_the_window_are_weather(self):
    # a window of nothing: every hold is the only one in it
    _, behind = self.holding(lambda seq: seq % 2 == 0, n=4 * model_state.HOLDS_ALLOWED, HOLD_WINDOW=0.0)
    self.assertEqual(behind, [None] * len(behind))


class TestTheFace(ModelStateTest):
  """What modeld reads off the model is comma's, as the adapter hands it over."""

  def test_it_carries_the_adapters_face(self):
    state = self.make(spec_for(STATEFUL), FakeClient())
    face = fakes.FACE
    self.assertIs(state.constants, face.constants)
    self.assertEqual((state.LAT_SMOOTH_SECONDS, state.LONG_SMOOTH_SECONDS), (0.0, 0.3))
    self.assertEqual(state.PLANPLUS_CONTROL, 1.0)
    # the module function, not bound to the model
    self.assertEqual(state.get_action_from_model('out', 'prev'), ('action', ('out', 'prev')))
    self.assertEqual(state.lat_delay, 0.0, "the joining model's to set, from the small model")
    self.assertIsInstance(state.parser, fakes.FakeParser)
    self.assertEqual(state.prev_desire.shape, (face.desire_len,))
    self.assertEqual(state.frame_size, face.frame_size(1928, 1208))
    self.assertIs(state.chestnut, True)

  def test_outputs_go_through_the_parser_sliced(self):
    spec = spec_for(STATEFUL)
    client = FakeClient()
    client.output[slice(*SLICES['plan'])] = 2.0
    state = self.make(spec, client)
    warped = SimpleNamespace(data=lambda: np.zeros(np.prod(spec.warped_shape), np.uint8))
    bufs = {k: SimpleNamespace(data=np.zeros(8, np.uint8)) for k in ('img', 'big_img')}
    with mock.patch.object(model_state, 'call_warp', return_value=warped):
      out = state.run(bufs, {'img': np.eye(3), 'big_img': np.eye(3)},
                      {'desire': np.zeros(8, np.float32), 'traffic_convention': np.zeros(2, np.float32),
                       'action_t': np.zeros(2, np.float32)})
    self.assertEqual(set(out), set(SLICES))
    self.assertEqual(out['plan'].shape, (1, 990))
    self.assertTrue((out['plan'] == 2.0).all())

  def test_closing_it_closes_its_link(self):
    client = mock.Mock()
    client.t.link_info.return_value = {'kind': 'usb'}
    self.make(spec_for(STATEFUL), client).close()
    client.close.assert_called_once_with()


class TestTelemetry(ModelStateTest):
  """The server's health rides on the response to a frame that asked for it.
  modeld asked by passing a callback every second frame; with none, the model
  asks as often itself and logs what came back at 1 Hz."""

  def test_without_a_callback_a_frame_asks_when_the_log_is_due(self):
    # the log takes one a second; asking on every second frame cost nine
    # replies in ten a JSON decode nobody read
    client = FakeClient()
    client.telemetry = {'gpu_temp': 51.0}
    _, _, client, _ = self.run_frames(STATEFUL, n=6, client=client)
    self.assertEqual(client.asked, [True, False, False, False, False, False])
    self.assertEqual(len(self.events), 1)

  def test_what_came_back_is_logged_at_one_hertz(self):
    client = FakeClient()
    client.last_state = {'gpu_temp': 51.0}
    self.run_frames(STATEFUL, n=6, client=client)
    self.assertEqual(self.events, [('jetlinkTelemetry', {'dead': False, 'gpu_temp': 51.0})])

  def test_nothing_back_yet_is_nothing_logged(self):
    self.run_frames(STATEFUL, n=4)
    self.assertEqual(self.events, [])

  def test_a_callback_asks_and_is_called_instead(self):
    client = FakeClient()
    client.last_state = {'gpu_temp': 51.0}
    callback = mock.Mock()
    self.run_frames(STATEFUL, n=3, client=client, after_enqueue=callback)
    self.assertEqual(client.asked, [True, True, True])
    self.assertEqual(callback.call_count, 3)


if __name__ == '__main__':
  unittest.main()
