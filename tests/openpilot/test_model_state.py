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
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from jetlink.openpilot import model_state
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError
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
  def __init__(self):
    self.nonce = 'test-session'
    self.sent = []
    self.asked = []
    self.last_timings = (0, 0, 0)
    self.last_state = None
    self.dead = False
    # a preview's link: there a frame the server failed is that frame's failure
    # and not the link's, and the frames whose answer carries no output at all
    # (a NOT_FINITE from the server)
    self.tolerate_failed = False
    # and a preview's link: a host that is behind rather than gone holds the
    # frames it owes instead of costing the link (client.patient)
    self.patient = False
    self.last_failed = False
    self.failed: set[int] = set()
    self.t = SimpleNamespace()
    self.output = np.zeros(18452, np.float32)
    self.output[slice(*SLICES['hidden_state'])] = 0.5
    self.last_output = None
    self._in_flight = []
    # frames (by seq) whose reply is not back within the hold, and the holds
    # end() asked for
    self.late = set()
    self.holds = []
    # and the budget a frame that had nothing to hold asked for (end()'s
    # first-frame deadline), None when it held instead
    self.deadlines = []
    self.drains = 0
    # a held frame's late reply taken by the *next* frame's infer_end instead of
    # by drain(): what a phone tailing by one frame does on the wire
    # (client._take_reply), and the shape 2026-10-18 22:02 was made of
    self.late_read_by_next = False
    self.last_answer_at = 0.0
    # frames (by frame_id) a drain leaves unanswered: a reply not back yet
    self.keep: set[int] = set()
    # what a reply asked for telemetry carries, when the server has any
    self.telemetry = None
    # frames (by frame_id) the link refuses when asked to skip if busy, and
    # whether each frame asked
    self.busy: set[int] = set()
    self.skippable = []
    self.lossless = []

  def infer_begin(self, data, packed, frame_id, reset=False, want_state=False, skip_if_busy=False, lossless=None):
    self.skippable.append(skip_if_busy)
    self.lossless.append(lossless)
    if skip_if_busy and frame_id in self.busy:
      return None
    self.sent.append((None if data is None else np.frombuffer(bytes(data), np.uint8).copy(), np.array(packed, copy=True),
                      frame_id, reset))
    self.asked.append(want_state)
    self._in_flight.append(frame_id)
    return frame_id

  def infer_end(self, seq, hold=None, deadline=None):
    self.holds.append(hold)
    self.deadlines.append(deadline)
    if self.late_read_by_next and len(self._in_flight) > 1 and self._in_flight[0] in self.late:
      # the previous frame's answer lands while this one waits, as it does on a
      # phone one frame behind: taken here, and the clock stamped here, not by
      # the drain() that ran before it arrived
      self._in_flight.pop(0)
      self.last_output = self.output
      self.last_answer_at = time.monotonic()
    if seq in self.failed:
      # the server answered, and its answer carried no output (NOT_FINITE)
      self._in_flight.remove(seq)
      self.last_failed = True
      if not self.tolerate_failed:
        self.dead = True
        raise LinkError(f'inference failed: NOT_FINITE (frame {seq})')
      return None
    self.last_failed = False
    if hold is not None and (seq in self.late or seq in self.keep):
      return None
    self._in_flight.remove(seq)
    self.last_output = self.output
    self._piggyback()
    return self.output

  def drain(self):
    # replies come in order: none past the first that is not back yet
    self.drains += 1
    if self.late_read_by_next and self._in_flight and self._in_flight[0] in self.late:
      return 0   # it has not landed yet at the top of this frame
    back = next((i for i, f in enumerate(self._in_flight) if f in self.keep), len(self._in_flight))
    taken, self._in_flight = self._in_flight[:back], self._in_flight[back:]
    if taken:
      # a frame the server answered without an output is answered all the same
      # (client._take_reply with tolerate_failed): the link is there, and there
      # is nothing to publish off it
      self.last_failed = any(f in self.failed for f in taken)
      if not self.last_failed:
        self.last_output = self.output
        self._piggyback()
    return len(taken)

  def _piggyback(self):
    if self.telemetry is not None and self.asked[-1]:
      self.last_state = self.telemetry

  @property
  def unanswered(self):
    return len(self._in_flight)


class FakeClock:
  """model_state's clock, in steps. Everything else here runs a frame in
  microseconds, so a rule that reads time (the preview's silence, whole
  seconds) cannot be exercised against the wall clock at all: this advances a
  fixed amount per read instead, which makes it deterministic."""

  def __init__(self, step: float = 0.5):
    self.now = 0.0
    self.step = step

  def monotonic(self) -> float:
    self.now += self.step
    return self.now

  def perf_counter(self) -> float:
    return self.now


class FakeWarp:
  """warp.Warp as the model state uses it. `output` holds the warped frame
  only once wait() has returned: the GPU writes it in between."""

  def __init__(self, warped):
    self.warped = warped
    self.output = memoryview(bytearray(warped.nbytes))
    self.started = []

  def start(self, frame, big_frame, tfm, big_tfm):
    self.started.append((frame, big_frame, np.array(tfm), np.array(big_tfm)))
    self.output[:] = b'\xee' * self.output.nbytes

  def wait(self):
    self.output[:] = self.warped.tobytes()


class ModelStateTest(unittest.TestCase):
  def setUp(self):
    self.log = fakes.RecordingLog()
    self.events = []

  def make(self, spec, client, face=fakes.FACE, warped=None, preview=False, pipeline=None):
    warped = np.zeros(np.prod(spec.warped_shape), np.uint8) if warped is None else warped
    return model_state.JetlinkModelState(client, spec, FakeWarp(warped), face=face, log=self.log,
                                         event=lambda name, **fields: self.events.append((name, fields)),
                                         preview=preview, pipeline=pipeline)

  def run_frames(self, inputs: dict, n: int = 3, client=None, after_enqueue=None):
    """`n` frames driven. Returns the spec, the state, the client and the
    warped frame every frame carried."""
    spec = spec_for(inputs)
    client = client or FakeClient()
    warped = np.arange(np.prod(spec.warped_shape), dtype=np.uint64).astype(np.uint8)
    state = self.make(spec, client, warped=warped)
    self.frames(state, n, after_enqueue)
    return spec, state, client, warped

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

  def test_the_warp_gets_the_camera_buffers_where_they_are_and_their_transforms(self):
    # and the frame goes out of its output once it is done (FakeWarp)
    state = self.make(spec_for(STATEFUL), FakeClient())
    bufs = {k: SimpleNamespace(data=np.zeros(8, np.uint8)) for k in ('img', 'big_img')}
    tfm = {'img': np.eye(3) * 2, 'big_img': np.eye(3) * 3}
    state.run(bufs, tfm, {'desire': np.zeros(8, np.float32), 'traffic_convention': np.zeros(2, np.float32),
                          'action_t': np.zeros(2, np.float32)})
    (frame, big_frame, got_tfm, got_big), = state.warp.started
    self.assertEqual((frame, big_frame), (bufs['img'].data.ctypes.data, bufs['big_img'].data.ctypes.data))
    np.testing.assert_array_equal(got_tfm, tfm['img'])
    np.testing.assert_array_equal(got_big, tfm['big_img'])

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

  def test_the_whole_run_reaches_the_log_every_summary_period(self):
    # the per-frame line is one sample; p99 over the window is read off this,
    # not inferred from a stream of samples
    spec, state, _, _ = self.run_frames(STATEFUL)
    assert not [line for line in self.log.lines('warning') if 'trips ' in line], 'summarised on the first frames'
    state._last_summary_logged = -model_state.SUMMARY_PERIOD
    self.frames(state, 1)
    trips = [line for line in self.log.lines('warning') if 'trips ' in line]
    self.assertEqual(len(trips), 1)
    for key in ('frames', 'over', 'held', 'pipelined', 'span_s', 'p50_ms', 'p99_ms', 'max_ms', 'server_ms'):
      self.assertIn(key, trips[0])
    # a period, not a frame
    self.frames(state, 2)
    self.assertEqual(len([line for line in self.log.lines('warning') if 'trips ' in line]), 1)

  def test_trips_summarise_the_frames_kept(self):
    trips = model_state.Trips()
    for _ in range(99):
      trips.record(0.030, 25_000)
    trips.record(0.200, 25_000)
    summary = trips.summary()
    self.assertEqual((summary['frames'], summary['over']), (100, 1))
    self.assertEqual((summary['p50_ms'], summary['max_ms'], summary['server_ms']), (30.0, 200.0, 25.0))
    self.assertEqual(summary['p99_ms'], 30.0, 'p99 is the 99th of a hundred, as the server takes it')
    self.assertEqual(model_state.Trips().summary(), {'frames': 0, 'over': 0, 'held': 0, 'pipelined': 0, 'span_s': 0.0})

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


class TestLossless(ModelStateTest):
  """A lossless warp's errors go packed, to a server that takes them."""

  def lossless_state(self, takes: bool, errors: bool = True):
    spec = spec_for(STATEFUL)
    client = FakeClient()
    client.takes_lossless = takes
    warp = FakeWarp(np.zeros(np.prod(spec.warped_shape), np.uint8))
    warp.errors = memoryview(bytearray(warp.output.nbytes)) if errors else None
    packer = mock.Mock(**{'pack.return_value': (b'table', b'planes')})
    with mock.patch('jetlink.lossless.Packer', return_value=packer) as made:
      state = model_state.JetlinkModelState(client, spec, warp, face=fakes.FACE, log=self.log,
                                            event=lambda name, **fields: None)
    self.frames(state, 2)
    return spec, client, warp, packer, made

  def test_the_errors_go_packed_to_a_server_that_takes_them(self):
    spec, client, warp, packer, made = self.lossless_state(takes=True)
    made.assert_called_once_with(spec.warped_shape)
    self.assertEqual(client.lossless, [(b'table', b'planes')] * 2)
    self.assertTrue(all(c.args == (warp.errors,) for c in packer.pack.call_args_list))

  def test_a_server_that_does_not_take_them_gets_the_frame_raw(self):
    _, client, _, packer, made = self.lossless_state(takes=False)
    made.assert_not_called()
    self.assertEqual(client.lossless, [None, None])

  def test_a_plain_warp_sends_raw(self):
    _, client, _, _, made = self.lossless_state(takes=True, errors=False)
    made.assert_not_called()
    self.assertEqual(client.lossless, [None, None])


class TestProving(ModelStateTest):
  """The second after a swap is the large model's proof: a single held frame
  in its first PROVING_FRAMES is behind, before anyone can engage."""

  def late_at(self, late: set[int], n: int):
    client = FakeClient()
    client.late = late
    spec = spec_for(STATEFUL)
    behind = []
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

  def test_a_held_frame_while_settling_is_forgiven(self):
    # the first after a join carries the history reset, a Mac's is CoreML warm-up
    n = model_state.SETTLING_FRAMES
    self.assertEqual(self.late_at(set(range(2, n + 1)), n=n), [None] * n)
    self.assertIsNotNone(self.late_at({n + 1}, n=n + 1)[-1])

  def test_a_host_that_keeps_up_proves_without_a_word(self):
    self.assertEqual(self.late_at(set(), n=model_state.PROVING_FRAMES + 5),
                     [None] * (model_state.PROVING_FRAMES + 5))


class TestHold(ModelStateTest):
  """A reply not back HOLD_FRAME into the frame is not waited for: the
  previous output goes out again and the camera frame is not dropped."""

  def _second_frame_held(self, client) -> None:
    """Three frames planning 1, 2, 2, where `client` holds the second: it
    publishes the first's output again, and the third its own."""
    state = self.make(spec_for(STATEFUL), client)
    client.output[slice(*SLICES['plan'])] = 1.0
    first, = self.frames(state, 1)
    client.output = client.output.copy()
    client.output[slice(*SLICES['plan'])] = 2.0
    held, after = self.frames(state, 2)
    self.assertTrue((first['plan'] == 1.0).all())
    self.assertTrue((held['plan'] == 1.0).all(), 'the frame before, again')
    self.assertTrue((after['plan'] == 2.0).all(), 'and the next frame its own')
    self.assertEqual(state.trips.held, 1)

  def test_a_late_reply_publishes_the_previous_output_once(self):
    client = FakeClient()
    client.late = {2}
    self._second_frame_held(client)
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
    self.assertEqual(client.skippable, [False] * 3, 'and its send waits for the link too')

  def test_a_frame_skips_a_busy_link_only_with_an_output_to_hold(self):
    _, _, client, _ = self.run_frames(STATEFUL, n=3)
    self.assertEqual(client.skippable, [False, True, True])

  def test_a_frame_the_link_would_not_take_is_held_without_waiting(self):
    client = FakeClient()
    client.busy = {2}
    self._second_frame_held(client)
    self.assertEqual([f for _, _, f, _ in client.sent], [1, 3], 'frame 2 never went out')
    self.assertEqual(len(client.holds), 2, 'and nothing waited for it')
    self.assertTrue(any('not sent' in line for line in self.log.lines('warning')))

  def test_an_unsent_frame_leaves_the_reset_for_the_next_one(self):
    client = FakeClient()
    spec = spec_for(STATEFUL)
    state = self.make(spec, client)
    self.frames(state, 1)
    state._need_reset = True      # a swap asked the host to start over
    client.busy = {2}
    self.frames(state, 2)
    self.assertEqual([(f, reset) for _, _, f, reset in client.sent], [(1, True), (3, True)])

  def holding(self, late, n: int, **patches):
    """`n` driven frames, the replies to `late` seqs (a predicate) too late
    to wait for. The state, and what `behind` said after each frame. Past
    the proof unless `patches` say otherwise (TestProving)."""
    patches.setdefault('PROVING_FRAMES', 0)
    patches.setdefault('SETTLING_FRAMES', 0)
    client = FakeClient()
    client.late = {seq for seq in range(1, n + 1) if late(seq)}
    spec = spec_for(STATEFUL)
    behind = []
    with contextlib.ExitStack() as patched:
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


class TestWifiHold(ModelStateTest):
  """A link off the cable pays a network round trip per reply, so the hold
  waits the wider WIFI_HOLD_FRAME and a few late frames while proving are the
  radio, not a host that cannot keep up. On the cable nothing changes."""

  @staticmethod
  def over_wifi(client: FakeClient) -> FakeClient:
    """A client whose transport answers that it is not on the cable."""
    client.t.on_the_cable = lambda: False
    return client

  def budget(self, client):
    """The state, and what `_hold_left` allows a frame started now."""
    state = self.make(spec_for(STATEFUL), client)
    return state, state._hold_left(SimpleNamespace(t0=time.perf_counter()))

  def test_the_wifi_budget_reaches_past_the_cable_one(self):
    state, left = self.budget(self.over_wifi(FakeClient()))
    self.assertTrue(state._wifi)
    self.assertGreater(left, model_state.HOLD_FRAME, 'a Wi-Fi reply gets the wider budget')
    self.assertLessEqual(left, model_state.WIFI_HOLD_FRAME)

  def test_the_cable_keeps_the_tighter_budget(self):
    client = FakeClient()
    client.t.on_the_cable = lambda: True
    state, left = self.budget(client)
    self.assertFalse(state._wifi)
    self.assertLessEqual(left, model_state.HOLD_FRAME)

  def test_a_link_with_no_medium_to_ask_keeps_the_tighter_budget(self):
    # an older server, or a fake: no on_the_cable, so not Wi-Fi
    state, left = self.budget(FakeClient())
    self.assertFalse(state._wifi)
    self.assertLessEqual(left, model_state.HOLD_FRAME)

  def test_a_wifi_frame_holds_within_the_wider_budget(self):
    client = self.over_wifi(FakeClient())
    _, state, client, _ = self.run_frames(STATEFUL, n=3, client=client)
    self.assertTrue(state._wifi)
    self.assertIsNone(client.holds[0], 'nothing to publish again yet: the first frame waits')
    for hold in client.holds[1:]:
      self.assertGreater(hold, 0.0)
      self.assertLessEqual(hold, model_state.WIFI_HOLD_FRAME)

  def test_a_wifi_phone_one_answer_behind_keeps_the_link(self):
    """Measured 2026-10-18, SM8975 phone over a 5 GHz hotspot: ~55 ms a reply,
    so every frame after the first published the previous output, and the link
    was torn down about a second into every join ("held 15 frames in a row",
    "held 15 frames in a row of the first 20") with 0 lost on the wire. One
    answer behind is a one-frame-old plan: tailing, and over Wi-Fi not behind.
    `drain()` stamping _last_answer is what a phone answering late but answering
    looks like to the state, so the test stamps it as prepare() would."""
    client = self.over_wifi(FakeClient())
    client.late = set(range(1, 121))           # every reply past its hold
    with mock.patch.multiple(model_state, PROVING_FRAMES=0, SETTLING_FRAMES=0):
      state = self.make(spec_for(STATEFUL), client)
      behind = []
      for _ in range(120):
        client.last_answer_at = time.monotonic()   # what drain() stamps on the wire
        self.frames(state, 1)
        behind.append(state.behind)
    self.assertEqual(behind, [None] * 120, 'a tailing phone is never behind')
    self.assertGreaterEqual(state.trips.held, 119, 'and every frame of it is still logged as held')

  def test_a_late_reply_read_by_the_next_frame_is_still_an_answer(self):
    """2026-10-18 22:02, the fix before this one in place: nine joins and nine
    leaves in two and a half minutes, every leave "span_s 1.0-1.4 ... held 15
    pipelined 0 p50_ms 55.5", 0 lost and no net drops. The reply to frame N
    lands after frame N+1 has drained, so N+1's infer_end took it and prepare()
    never stamped _last_answer: the drain clock stood at the join and the guard
    never saw an answer at all. The client's own clock is that answer."""
    client = self.over_wifi(FakeClient())
    client.late = set(range(1, 121))
    client.late_read_by_next = True
    with mock.patch.multiple(model_state, PROVING_FRAMES=0, SETTLING_FRAMES=0):
      state = self.make(spec_for(STATEFUL), client)
      joined_at = state._last_answer
      behind = []
      for _ in range(120):
        self.frames(state, 1)
        behind.append(state.behind)
    self.assertGreater(client.drains, 100, 'prepare() ran every frame and found nothing')
    self.assertLess(state._last_answer - joined_at, 0.05, 'and the drain clock never moved off the join')
    self.assertGreater(client.last_answer_at, joined_at, 'while the answers kept landing')
    self.assertGreaterEqual(state.trips.held, 119, 'every frame of it still a held one')
    self.assertEqual(behind, [None] * 120, 'a phone tailing one frame is not a lost link')

  def test_the_wifi_counts_hand_back_once_the_answers_stop(self):
    """Tailing is an answer late, not silence. Past WIFI_TAILING_ANSWER no reply
    has landed, the published plan is stale, and the counts take the link back."""
    state = self.make(spec_for(STATEFUL), self.over_wifi(FakeClient()))
    state._holds_in_a_row_override = model_state.WIFI_HOLDS_IN_A_ROW
    state._frame_id = model_state.PROVING_FRAMES + 1
    state._last_answer = time.monotonic() - model_state.WIFI_TAILING_ANSWER - 1.0
    for _ in range(model_state.WIFI_HOLDS_IN_A_ROW - 1):
      self.assertIsNone(state._note_hold(True))
    self.assertEqual(state._note_hold(True), f'held {model_state.WIFI_HOLDS_IN_A_ROW} frames in a row')

  def test_a_wifi_link_that_holds_every_proving_frame_stays_up(self):
    """The proving window is a cable rule. On Wi-Fi a phone holds most of its
    first twenty frames by design, and the link still has to survive them."""
    behind = self.late_at(set(range(1, model_state.PROVING_FRAMES + 1)),
                          n=model_state.PROVING_FRAMES, wifi=True)
    self.assertEqual(behind, [None] * len(behind), 'every proving frame may be held on Wi-Fi')

  def late_at(self, late: set[int], n: int, wifi: bool):
    """`n` driven frames, the replies to `late` seqs too late to wait for."""
    client = self.over_wifi(FakeClient()) if wifi else FakeClient()
    client.late = late
    behind = []
    state = self.make(spec_for(STATEFUL), client)
    for _ in range(n):
      self.frames(state, 1)
      behind.append(state.behind)
    return behind

  def test_a_wifi_link_that_holds_every_proving_frame_stays_up(self):
    """The proving window is a cable rule. On Wi-Fi a phone holds most of its
    first twenty frames by design, and the link still has to survive them."""
    behind = self.late_at(set(range(1, model_state.PROVING_FRAMES + 1)),
                          n=model_state.PROVING_FRAMES, wifi=True)
    self.assertEqual(behind, [None] * len(behind), 'every proving frame may be held on Wi-Fi')

  def test_on_the_cable_one_late_frame_while_proving_is_still_behind(self):
    behind = self.late_at({7}, n=8, wifi=False)
    self.assertEqual(behind[6], f'held frame 7 of the first {model_state.PROVING_FRAMES}')
    self.assertIsNone(behind[7], 'said once, for the frame that held')


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
    self.assertIs(state.chestnut, True)

  def test_outputs_go_through_the_parser_sliced(self):
    spec = spec_for(STATEFUL)
    client = FakeClient()
    client.output[slice(*SLICES['plan'])] = 2.0
    state = self.make(spec, client)
    bufs = {k: SimpleNamespace(data=np.zeros(8, np.uint8)) for k in ('img', 'big_img')}
    out = state.run(bufs, {'img': np.eye(3), 'big_img': np.eye(3)},
                    {'desire': np.zeros(8, np.float32), 'traffic_convention': np.zeros(2, np.float32),
                     'action_t': np.zeros(2, np.float32)})
    self.assertEqual(set(out), set(SLICES))
    self.assertEqual(out['plan'].shape, (1, 990))
    self.assertTrue((out['plan'] == 2.0).all())

  def test_closing_it_closes_its_link(self):
    client = mock.Mock()
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

  def test_send_diagnostics_include_session_maxima_without_extra_log_frequency(self):
    client = FakeClient()
    client.last_state = {'gpu_temp': 51.0}
    client.t.last_send = {'submit_ms': 3.0, 'errno': None}
    client.t.send_totals = {'messages': 100, 'max_submit_ms': 19.0}
    self.run_frames(STATEFUL, n=6, client=client)
    sends = [fields for name, fields in self.events if name == 'jetlinkSend']
    self.assertEqual(len(sends), 1)
    self.assertEqual(sends[0]['nonce'], 'test-session')
    self.assertEqual(sends[0]['submit_ms'], 3.0)
    self.assertEqual(sends[0]['totals']['max_submit_ms'], 19.0)
    client.t.send_totals['messages'] += 1
    self.assertEqual(sends[0]['totals']['messages'], 100)

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


class TestTheHostsFirstAnswer(ModelStateTest):
  """The frames before the host has answered once have no output to publish
  again, so end() has nothing to hold: they wait their own budget
  (model_state.FIRST_FRAME_TIMEOUT). The client's deadline is one 20 Hz frame,
  and a first frame is the slowest frame there is (the history reset and the
  host's own cold start)."""

  def test_the_first_frame_waits_its_own_budget(self):
    spec, state, client, _ = self.run_frames(STATEFUL, n=3)
    self.assertIsNone(client.holds[0])
    self.assertEqual(client.deadlines[0], model_state.FIRST_FRAME_TIMEOUT)
    # the answer is in: every frame after it holds instead, as before
    self.assertEqual(client.deadlines[1:], [None, None])
    for hold in client.holds[1:]:
      self.assertIsNotNone(hold)

  def test_wifi_waits_the_wider_one(self):
    client = FakeClient()
    client.t.on_the_cable = lambda: False
    state = self.make(spec_for(STATEFUL), client)
    self.frames(state, 1)
    self.assertEqual(client.deadlines, [model_state.WIFI_FIRST_FRAME_TIMEOUT])

  def test_an_output_already_in_hand_is_not_a_first_frame(self):
    # drain() before the first frame has the host's output: there is something
    # to publish again, so the frame holds and never waits the budget
    client = FakeClient()
    client.last_output = client.output
    state = self.make(spec_for(STATEFUL), client)
    self.frames(state, 1)
    self.assertEqual(client.deadlines, [None])
    self.assertIsNotNone(client.holds[0])


class TestAPreview(ModelStateTest):
  """A preview never takes control, so a host that is slow but answering keeps
  the link: its frames publish the previous output and the picture refreshes at
  the host's rate. Only silence past PREVIEW_SILENCE hands it back."""

  def make_late(self, preview: bool):
    """A host slower than the hold on every frame after its first."""
    client = FakeClient()
    client.late = set(range(2, 20))
    return self.make(spec_for(STATEFUL), client, preview=preview), client

  def test_a_late_host_keeps_the_link_and_no_frame_waits_for_it(self):
    state, client = self.make_late(preview=True)
    self.frames(state, 8)
    # the frames after the first never waited for their own reply, so none of
    # them is late: the host is slow and keeps the link (state.pipeline)
    self.assertEqual((state.trips.held, state.trips.pipelined), (0, 7))
    self.assertIsNone(state.behind)

  def test_a_preview_that_still_waits_holds_its_frames(self):
    # pipeline=False: the same late host, the frame loop waiting for its own
    # reply as it did before the pipeline existed, one held frame each
    client = FakeClient()
    client.late = set(range(2, 20))
    state = self.make(spec_for(STATEFUL), client, preview=True, pipeline=False)
    self.assertFalse(state.pipeline)
    self.frames(state, 8)
    self.assertEqual((state.trips.held, state.trips.pipelined), (7, 0))
    self.assertIsNone(state.behind)

  def test_the_link_is_told_the_preview_is_patient(self):
    # a stall past the client's deadline is a host behind, not gone, so the
    # preview's link holds the frame; the wheel's still hands the picture back
    # over one (client.patient)
    _, client = self.make_late(preview=True)
    self.assertTrue(client.patient)
    self.assertTrue(client.tolerate_failed, 'both are set from state.preview')
    self.assertFalse(self.make_late(preview=False)[1].patient)

  def test_the_same_host_hands_a_driving_link_back(self):
    state, client = self.make_late(preview=False)
    self.frames(state, 8)
    self.assertTrue((state.behind or '').startswith('held frame'), state.behind)

  def test_silence_hands_the_preview_back(self):
    state, client = self.make_late(preview=True)
    with mock.patch.object(model_state, 'PREVIEW_SILENCE', 0.0):
      self.frames(state, 8)
    self.assertIn('no answer for', state.behind or '')


class TestAPipelinedPreview(ModelStateTest):
  """A preview's frame loop never waits for the frame's own reply: the newest
  answer is published, that reply is read by a later frame, and the host's work
  runs against the camera's next frame instead of adding to it. The wait is
  what put a phone's Wi-Fi frame past the camera's 50 ms, and a frame past it
  is a camera frame dropped behind the large model -- the demote that took the
  picture away (JETLINK_WIFI_LINK_1016)."""

  def late_state(self, preview: bool = True):
    """A host slower than the hold on every frame after its first."""
    client = FakeClient()
    client.late = set(range(2, 20))
    return self.make(spec_for(STATEFUL), client, preview=preview), client

  def test_no_frame_after_the_first_waits_for_its_own_reply(self):
    state, client = self.late_state()
    self.assertTrue(state.pipeline)
    self.frames(state, 8)
    # only the first frame waits, and it waits its own budget: there is no
    # output to publish yet, so it is the one frame that has nothing to fall
    # back on (model_state._first_frame_budget)
    self.assertEqual(client.holds, [None])
    self.assertEqual(client.deadlines, [model_state.FIRST_FRAME_TIMEOUT])

  def test_the_newest_answer_is_what_a_pipelined_frame_publishes(self):
    state, client = self.late_state()
    outs = self.frames(state, 4)
    # the answer in hand, the same array parsed once: the parser's softmax
    # works in place on it, so the dict is the one already made
    self.assertIs(outs[1], outs[0])
    self.assertEqual(state.trips.pipelined, 3)
    self.assertEqual(state.trips.frames, 4)

  def test_a_link_that_will_not_take_a_frame_is_not_waited_for_either(self):
    # skip_if_busy: a frame the link cannot take without draining earlier ones
    # is not sent at all, and the frame publishes the newest answer
    state, client = self.late_state()
    client.busy = {3, 4}
    outs = self.frames(state, 4)
    self.assertEqual([frame_id for _, _, frame_id, _ in client.sent], [1, 2])
    self.assertEqual(state.trips.pipelined, 1, 'only the frame that went out on a pipeline')
    self.assertIs(outs[2], outs[1], 'a frame that was not sent publishes the newest answer')

  def test_the_wheel_never_pipelines(self):
    state, client = self.late_state(preview=False)
    self.assertFalse(state.pipeline)
    self.frames(state, 8)
    self.assertEqual(state.trips.pipelined, 0)
    self.assertTrue((state.behind or '').startswith('held frame'), state.behind)

  def test_a_driven_away_preview_waits_for_its_own_reply_again(self):
    # joining clears `preview` the moment anything can be in control; the
    # pipeline reads it, so the wheel gets the frame loop it always had
    state, client = self.late_state()
    self.frames(state, 1)
    state.preview = False
    self.assertFalse(state.pipeline)
    self.frames(state, 2)
    self.assertEqual(state.trips.pipelined, 0)

  def test_the_silence_clock_is_the_hosts_answers_not_this_frames_reply(self):
    """A pipelined frame is never its own reply, so the rule cannot read
    'this frame was not answered in time' as silence: a host whose every frame
    is late answers every frame, and one that has stopped answering loses the
    picture at PREVIEW_SILENCE (5 s), as a preview always has. The clock steps
    0.5 s a read, so the sixteen frames below outlast it only where no answer
    ever comes back."""
    late, _ = self.late_state()
    with mock.patch.object(model_state, 'time', FakeClock()):
      self.frames(late, 16)
    self.assertIsNone(late.behind, 'a late answer is still an answer')

    client = FakeClient()
    client.keep = set(range(2, 40))          # nothing comes back at all
    silent = self.make(spec_for(STATEFUL), client, preview=True)
    with mock.patch.object(model_state, 'time', FakeClock()):
      self.frames(silent, 16)
    self.assertIn('no answer for', silent.behind or '')


class TestAPreviewFrameTheHostCannotAnswer(ModelStateTest):
  """A host that answers with a frame it could not compute is still there. A
  preview keeps its link and serves that frame from the small model, where the
  wheel hands the link back."""

  def make_preview(self, preview: bool):
    client = FakeClient()
    return super().make(spec_for(STATEFUL), client, preview=preview), client

  def test_a_preview_keeps_the_link_where_the_wheel_hands_it_back(self):
    state, client = self.make_preview(preview=True)
    self.assertTrue(client.tolerate_failed)
    client.failed.add(1)
    # nothing has come back this session, so there is no output to publish
    # again: the joining state serves this frame (joining._run_big)
    with self.assertRaises(model_state.FrameFailed):
      self.frames(state, 1)
    self.assertFalse(client.dead)

    driving, wheel = self.make_preview(preview=False)
    self.assertFalse(wheel.tolerate_failed)
    wheel.failed.add(1)
    with self.assertRaises(LinkError):
      self.frames(driving, 1)
    self.assertTrue(wheel.dead)

  def test_an_earlier_output_is_published_again_and_the_clock_restarts(self):
    state, client = self.make_preview(preview=True)
    first = self.frames(state, 1)[0]
    client.failed.add(2)
    with mock.patch.object(model_state, 'PREVIEW_SILENCE', 0.0):
      # an answer that carried no output is still an answer: a host that goes
      # on failing its frames is not a host that went quiet. A pipelined frame
      # publishes the newest answer, so the answer to frame 2 is read by frame
      # 3's prepare() -- and it is that reading, not frame 2's own wait, that
      # restarts the clock
      again = self.frames(state, 2)[0]
    self.assertIs(again, first)
    self.assertIsNone(state.behind)
    self.assertFalse(client.dead)

  def test_a_driven_away_preview_is_held_to_the_wheels_rules_again(self):
    state, client = self.make_preview(preview=True)
    self.frames(state, 1)
    state.preview = False
    self.assertFalse(client.tolerate_failed)


if __name__ == '__main__':
  unittest.main()
