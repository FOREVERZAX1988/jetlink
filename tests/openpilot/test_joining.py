"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The rules the late join has to keep.

Both model states are fakes; the joining state is plumbing around them. What
is pinned: modeld gets a working model immediately, a swap never lands on an
engaged frame, a large model that dies mid-drive falls back without losing the
frame, and a small-model failure still belongs to modeld.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jetlink.openpilot import joining
from jetlink.openpilot.joining import REJOIN_DELAY_QUICK, STABLE_SECONDS, JoiningModelState
from tests.openpilot.fakes import RecordingLog, isolate


class FakeModel:
  def __init__(self, name, chestnut=False, client=None):
    self.name = name
    self.chestnut = chestnut
    self.client = client
    self.lat_delay = 0.0
    self.vision_input_names = ['img', 'big_img']
    self.calls = 0
    self.raises = None
    self.closed = False
    self.warmed = False

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    self.calls += 1
    if self.raises is not None:
      raise self.raises
    return {'from': self.name}

  def warmup(self):
    self.warmed = True

  def close(self):
    self.closed = True


class JoiningBase(unittest.TestCase):
  """The joining state over fake models, and the helpers to drive it. No tests
  here, so each face below runs only its own."""

  def setUp(self):
    # a lost link reads the USB-C port's CC pin
    isolate(self, Path(tempfile.mkdtemp()))
    # The engagement watcher runs the adapter's poller, which EngagementTest
    # below covers. Drive the flag by hand instead
    patcher = mock.patch.object(JoiningModelState, '_watch_engagement', lambda self: None)
    patcher.start()
    self.addCleanup(patcher.stop)

    # The join reports what it waits on as progress; what it says is asserted here.
    self.progress = mock.Mock()
    self.log = RecordingLog()
    self.engaged = True

    self.small = FakeModel('small')
    self.big = FakeModel('big', chestnut=True, client=object())
    self.joined = threading.Event()
    self.connect_calls = 0
    self.connect_error = None

  def _connect(self, should_stop=None):
    self.connect_calls += 1
    if self.connect_error is not None:
      raise self.connect_error
    self.joined.set()
    # A link the joining state may have to close on its own, when it is torn
    # down holding a join that never found a disengaged frame to land on.
    return (mock.MagicMock(name='client'), 'spec')

  def _build(self, client, spec):
    return self.big

  def _engagement(self):
    def engaged(timeout_ms):
      time.sleep(timeout_ms / 1000)
      return self.engaged
    return engaged

  def _make(self, *args, **kwargs):
    return JoiningModelState(*args, progress=self.progress, engagement=self._engagement, log=self.log, **kwargs)

  def _state(self):
    s = self._make(self.small, self._connect, self._build)
    self.addCleanup(self._close, s)
    return s

  @staticmethod
  def _close(s):
    # the threads are joined while setUp's patches are still on
    s.close()
    for t in s._threads:
      t.join(5)

  def _wait_joined(self, s, timeout=5.0):
    # connect() returning is not publication: wait for the owner to hand off.
    deadline = time.monotonic() + timeout
    while s._joined is None and time.monotonic() < deadline:
      time.sleep(0.001)
    self.assertIsNotNone(s._joined)

  def _run(self, s):
    s._engagement_updated = time.monotonic()
    return s.run({}, {}, {})

  def _wait_reported(self, s, msg, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if any(c.args[:2] == ('connect', 0.0) and c.args[2].startswith(msg)
             for c in self.progress.report.call_args_list):
        return
      time.sleep(0.005)
    self.fail(f"never reported {msg!r}: {self.progress.report.call_args_list}")


class JoiningTest(JoiningBase):
  def test_stalled_watcher_cannot_leave_a_swap_window_open(self):
    s = self._state()
    s._engaged = False
    s._engagement_updated = time.monotonic() - 1.0
    self.assertFalse(s._window_open)

  def test_runs_the_small_model_immediately(self):
    s = self._state()
    # No waiting on a link: this is the whole point.
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertFalse(s.chestnut)
    self.assertIsNone(s.client)

  def test_does_not_swap_while_engaged(self):
    # stopped or moving: even stopped, longitudinal control can hold the brake
    # or request motion, so the window reads engagement and nothing else
    s = self._state()
    self._wait_joined(s)
    s._engaged = True
    for _ in range(3):
      self.assertEqual(self._run(s), {'from': 'small'})
    self.assertFalse(s.chestnut)
    self.assertFalse(self.big.warmed)

  def test_late_boot_announces_availability_without_switching(self):
    booted = threading.Event()
    connect = self._connect

    def after_boot(should_stop=None):
      if not booted.wait(5):
        raise RuntimeError('test boot timeout')
      return connect()

    self._connect = after_boot
    s = self._state()
    self.addCleanup(booted.set)
    self.assertFalse(s.big_model_available)
    self.assertEqual(s.big_model_state, 'joining', 'nothing to swap in yet')
    self.assertEqual(self._run(s), {'from': 'small'})
    booted.set()
    self._wait_joined(s)
    self.assertTrue(s.big_model_available)
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertEqual(s.big_model_state, 'ready')
    s._engaged = False
    self.assertEqual(self._run(s), {'from': 'big'})
    self.assertFalse(s.big_model_available)
    self.assertEqual(s.big_model_state, 'running')

  def test_availability_survives_ping_but_not_link_loss(self):
    pinging, release = threading.Event(), threading.Event()
    client = mock.MagicMock()

    def ping(**kwargs):
      pinging.set()
      release.wait(5)
      raise RuntimeError('link lost while waiting to switch')

    client.ping.side_effect = ping
    self._connect = lambda should_stop=None: (client, 'spec')
    with mock.patch.object(joining, 'KEEPALIVE_PERIOD', 0.01):
      s = self._state()
      self.addCleanup(release.set)
      self.assertTrue(pinging.wait(5))
      self.assertIsNone(s._joined)
      self.assertTrue(s.big_model_available)
      self.assertEqual(self._run(s), {'from': 'small'})
      release.set()
      deadline = time.monotonic() + 2
      while s.big_model_available and time.monotonic() < deadline:
        time.sleep(0.001)
      self.assertFalse(s.big_model_available)
      self.assertFalse(s.chestnut)

  def test_ping_taking_the_pending_link_during_swap_check_keeps_availability(self):
    s = self._state()
    self._wait_joined(s)
    joined = s._joined
    s._engaged = False
    # The frame sees _joined before locking, then the keepalive takes it.
    with mock.patch.object(s, '_lock') as lock:
      lock.__enter__.side_effect = lambda: setattr(s, '_joined', None)
      self.assertEqual(self._run(s), {'from': 'small'})
      self.assertTrue(s.big_model_available)
    s._joined = joined

  def test_close_with_pending_model_clears_availability(self):
    s = self._state()
    self._wait_joined(s)
    self.assertTrue(s.big_model_available)
    s.close()
    self.assertFalse(s.big_model_available)

  def test_swaps_on_a_disengaged_frame(self):
    s = self._state()
    self._wait_joined(s)
    s._engaged = False
    self.assertEqual(self._run(s), {'from': 'big'})
    self.assertTrue(s.chestnut)
    self.assertIs(s.client, self.big.client)
    # No warmup at the swap: the warp was prepared in __init__ and a frame
    # over the link here was two dropped camera frames on the car.
    self.assertFalse(self.big.warmed)

  def test_large_model_failure_demotes_and_keeps_the_frame(self):
    s = self._state()
    self._wait_joined(s)
    s._engaged = False
    self._run(s)
    self.assertTrue(s.chestnut)

    self.big.raises = RuntimeError("link gone")
    # modeld still gets an output for this frame, from the small model.
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertFalse(s.chestnut)
    for _ in range(100):
      if self.big.closed:
        break
      time.sleep(0.01)
    self.assertTrue(self.big.closed)
    # And it tries again rather than staying small for the rest of the drive.
    self.assertTrue(s._rejoin_at > time.monotonic() or self.connect_calls > 1)

  def test_failed_first_inference_never_announces_ready(self):
    s = self._state()
    self._wait_joined(s)
    s._engaged = False
    self.big.raises = RuntimeError('first inference failed')
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertEqual(s.big_model_state, 'retrying')
    self.assertFalse(s.big_model_available)
    self.progress.clear.assert_not_called()

  def test_fallback_resets_history_without_waiting_for_teardown(self):
    entered, release = threading.Event(), threading.Event()
    reset = mock.Mock()

    def close():
      entered.set()
      release.wait(2)

    self.big.close = close
    s = self._make(self.small, self._connect, self._build, reset_small=reset)
    self.addCleanup(self._close, s)
    self.addCleanup(release.set)
    self._wait_joined(s)
    s._engaged = False
    self._run(s)
    self.big.raises = RuntimeError('failed')
    run = self.small.run

    def small_run(*args):
      reset.assert_called_once()
      return run(*args)

    self.small.run = small_run
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertTrue(entered.wait(1))
    self.assertFalse(release.is_set())

  def test_ready_is_announced_only_after_inference_returns(self):
    s = self._state()
    self._wait_joined(s)
    s._engaged = False
    run = self.big.run

    def inspect(*args):
      # Swapped in, but not announced: the first frame has not returned yet.
      self.assertTrue(s._loading)
      self.progress.clear.assert_not_called()
      return run(*args)

    self.big.run = inspect
    self.assertEqual(self._run(s), {'from': 'big'})
    self.assertFalse(s._loading)
    self.progress.clear.assert_called_once()

  def test_prepare_failure_uses_startup_fallback_not_a_slow_swap(self):
    order = []
    self.connect_calls = 0

    def prepare():
      order.append('prepare')
      raise RuntimeError("no warp today")

    with self.assertRaisesRegex(RuntimeError, 'no warp today'):
      self._make(self.small, self._connect, self._build, prepare)
    # Ran, and ran before anything else: modeld's main thread is blocked for
    # exactly as long as the constructor takes, so this is the only place the
    # GPU work can go without costing a frame.
    self.assertEqual(order, ['prepare'])
    self.assertEqual(self.connect_calls, 0)
    self.assertFalse(self.big.warmed)

  def test_loading_has_no_deadline(self):
    # no 60 s edge: selfdrived gates on it only while nothing publishes modelV2,
    # so a Jetson that takes a whole drive stays "getting ready"
    self.connect_error = RuntimeError("jetson still booting")
    s = self._state()
    self._run(s)
    with mock.patch.object(joining.time, 'monotonic',
                    return_value=time.monotonic() + 600.0):
      self._run(s)
    self.assertFalse(s.chestnut)
    self.assertEqual(s.big_model_state, 'joining')
    # A join that succeeds later still swaps.
    self.connect_error = None
    s._rejoin.set()
    self._wait_joined(s, 10.0)
    s._engaged = False
    for _ in range(20):
      if self._run(s) == {'from': 'big'}:
        break
      time.sleep(0.05)
    self.assertTrue(s.chestnut)

  def test_state_travels_in_the_message_not_in_params(self):
    # a chestnut's load is over once; this never is, so selfdrived's edge is
    # modelV2.big turning true and the UI reads acceleratorState
    s = self._state()
    self.assertFalse(s.chestnut)

    self._wait_joined(s)
    # up and only a swap window away, which the icon draws steady rather than
    # pulsing "loading" for the rest of a drive with no stop in it
    self.assertEqual(s.big_model_state, 'ready')
    s._engaged = False
    self._run(s)
    self.assertTrue(s.chestnut)
    self.assertEqual(s.big_model_state, 'running')

    self.big.raises = RuntimeError("link gone")
    self._run(s)
    self.assertFalse(s.chestnut)
    self.assertEqual(s.big_model_state, 'retrying')
    # reported from the join thread, not the frame that lost the link
    self._wait_reported(s, 'lost the accelerator, reconnecting')

    s.close()
    self.assertEqual(s.big_model_state, 'unavailable')

  def test_a_lost_link_is_counted_and_repeated_drops_blame_the_cable(self):
    # every drop on the 2026-09-07 drives was the USB port letting go, and the
    # driver saw "Big Model Failed" six times with no hint of a cause
    s = self._state()
    self._wait_joined(s)
    s._engaged = False
    self._run(s)
    # held for a while: the rejoin is the quick one, so the test waits on it
    s._joined_at = time.monotonic() - (STABLE_SECONDS + 1)
    self.big.raises = RuntimeError("host dropped the gadget configuration (udc: not attached)")
    self._run(s)
    self._wait_reported(s, 'lost the accelerator, reconnecting')
    self.assertEqual(s._drops, 1)
    first = [c for c in self.progress.report.call_args_list if c.args[2].startswith('lost')]
    self.assertNotIn('cable', first[-1].args[2])
    # a second drop in the drive names the cable, on every status from then on
    self.big.raises = None
    self._wait_joined(s)
    self._run(s)
    self.assertTrue(s.chestnut)
    s._joined_at = time.monotonic() - (STABLE_SECONDS + 1)
    self.big.raises = RuntimeError("gadget write failed: [Errno 19] No such device (udc: default)")
    self._run(s)
    self._wait_reported(s, 'lost the accelerator, reconnecting; link dropped 2 times this drive, check the USB cable')
    self.assertEqual(s._drops, 2)
    self._wait_reported(s, 'waiting for the accelerator; link dropped 2 times this drive, check the USB cable')

  def test_the_frame_that_loses_the_link_does_not_report_or_read_the_port(self):
    # the frame thread is SCHED_FIFO on modeld's core; params and sysfs are
    # for the join thread. The drop is only counted here
    s = self._state()
    self._wait_joined(s)
    s._engaged = False
    self._run(s)
    self.big.raises = RuntimeError("link gone")
    # the join thread is held asleep, so whatever reported did so on the frame
    with mock.patch.object(s._rejoin, 'set'), mock.patch.object(s, '_note_link_loss') as note:
      self.assertEqual(self._run(s), {'from': 'small'})
      self.assertTrue(s._demoted)
      self.assertEqual(s._drops, 1)
      note.assert_not_called()
      calls = [c for c in self.progress.report.call_args_list if c.args[2].startswith('lost')]
      self.assertEqual(calls, [])

  def test_build_failure_backs_off(self):
    self._build = mock.Mock(side_effect=RuntimeError("no warp"))
    s = self._make(self.small, self._connect, self._build)
    self.addCleanup(self._close, s)
    self._wait_joined(s)
    s._engaged = False
    self.assertEqual(self._run(s), {'from': 'small'})
    # Not straight back onto the link: the next attempt waits REJOIN_DELAY.
    self.assertGreater(s._rejoin_at, time.monotonic() + 1.0)
    self.assertFalse(s.chestnut)
    self.assertEqual(s.big_model_state, 'retrying')

  def test_a_link_that_dies_before_the_swap_is_reopened(self):
    # a link waits in _joined for a window, which on a drive with no stop is the
    # whole drive; a Jetson that reboots in there must be caught before the swap
    clients = []

    def connect(should_stop=None):
      c = mock.MagicMock(name='client')
      clients.append(c)
      self.connect_calls += 1
      self.joined.set()
      return (c, 'spec')

    self._connect = connect
    with mock.patch.object(joining, 'KEEPALIVE_PERIOD', 0.05), \
         mock.patch.object(joining, 'REJOIN_DELAY', 0.05):
      s = self._state()
      self._wait_joined(s)
      # Never a window, so nothing consumes it; the ping is what notices.
      s._engaged = True
      clients[0].ping.side_effect = RuntimeError("jetson rebooted")
      for _ in range(100):
        if len(clients) > 1:
          break
        time.sleep(0.05)
      self.assertGreater(len(clients), 1, "a dead pending link was never reopened")
      clients[0].close.assert_called()
      # And the fresh one is what the window eventually gets.
      s._engaged = False
      for _ in range(40):
        if self._run(s) == {'from': 'big'}:
          break
        time.sleep(0.05)
      self.assertTrue(s.chestnut)

  def test_failures_back_off_and_a_stable_join_starts_over(self):
    # A link that dies on its first frame every time used to cost a swap, a
    # demote and a chime every REJOIN_DELAY for the drive.
    s = self._state()
    s._joined_at = 0.0
    delays = []
    for _ in range(5):
      t = time.monotonic()
      s._back_off()
      delays.append(round(s._rejoin_at - t))
    self.assertEqual(delays, [5, 10, 20, 40, 60])

    # A join that held is not that link, and must not inherit its delay: a
    # link that ran for minutes and then went is a USB drop, and the host
    # re-enumerates a rebound gadget in under a second.
    s._joined_at = time.monotonic() - (STABLE_SECONDS + 1)
    t = time.monotonic()
    s._back_off()
    self.assertEqual(round(s._rejoin_at - t), round(REJOIN_DELAY_QUICK))
    self.assertEqual(s._failures, 1)
    # and a failure on its heels is the second rung, not the first again
    s._back_off()
    self.assertEqual(round(s._rejoin_at - time.monotonic()), 10)

  def test_small_model_failure_is_modelds(self):
    s = self._state()
    self.small.raises = RuntimeError("vipc gone")
    with self.assertRaises(RuntimeError):
      self._run(s)

  def test_a_read_it_does_not_know_follows_the_model_that_drives(self):
    # modeld reads attributes off the model every frame; one a sync adds must
    # not be an AttributeError on the frame thread
    self.small.new_constant, self.big.new_constant = 'small', 'big'
    s = self._state()
    self.assertEqual(s.new_constant, 'small')
    self._wait_joined(s)
    s._engaged = False
    self._run(s)
    self.assertEqual(s.new_constant, 'big')
    self.big.raises = RuntimeError('link gone')
    self._run(s)
    self.assertEqual(s.new_constant, 'small')

  def test_a_name_neither_model_has_is_still_an_error(self):
    s = self._state()
    with self.assertRaises(AttributeError):
      _ = s.no_such_thing
    self.assertFalse(hasattr(s, 'no_such_thing'))

  def test_private_names_are_its_own(self):
    self.small._private = 1
    s = self._state()
    with self.assertRaises(AttributeError):
      _ = s._private

  def test_a_write_it_does_not_know_stays_on_it(self):
    # only reads are delegated: each write modeld makes has a setter that
    # reaches both models (lat_delay, PLANPLUS_CONTROL)
    s = self._state()
    s.something_new = 5
    self.assertFalse(hasattr(self.small, 'something_new'))

  def test_it_is_not_consulted_while_it_is_being_built(self):
    # a read before _active exists must not recurse
    s = JoiningModelState.__new__(JoiningModelState)
    with self.assertRaises(AttributeError):
      _ = s.anything

  def test_lat_delay_reaches_both_models(self):
    s = self._state()
    s.lat_delay = 0.25
    self.assertEqual(self.small.lat_delay, 0.25)
    self._wait_joined(s)
    s._engaged = False
    self._run(s)
    self.assertEqual(self.big.lat_delay, 0.25)


  def test_reports_joining_until_it_joins(self):
    # the UI reads this to tell "not up yet" from "failed"; modelV2.big is
    # false for the whole join
    s = self._state()
    self._run(s)
    self.assertIn(s.big_model_state, ('joining', 'ready'))

    self._wait_joined(s)
    s._engaged = False
    self._run(s)
    self.assertEqual(s.big_model_state, 'running')

    self.big.raises = RuntimeError("link gone")
    self._run(s)
    self.assertEqual(s.big_model_state, 'retrying')


class LagTest(JoiningBase):
  """A large model that answers, but late, is handed back as if it were lost.

  The frames' lengths are faked on the joining state's own clock, so nothing
  here waits them out and a busy machine cannot turn jitter into a fault.
  """

  def setUp(self):
    super().setUp()
    self.skew = 0.0
    real = time.monotonic
    patcher = mock.patch.object(joining, 'time', SimpleNamespace(monotonic=lambda: real() + self.skew))
    patcher.start()
    self.addCleanup(patcher.stop)
    self.reset = mock.Mock()
    self.took = 0.0
    run = self.big.run

    def slow_run(*args):
      self.skew += self.took
      return run(*args)
    self.big.run = slow_run
    self.small.new_constant, self.big.new_constant = 'small', 'big'
    self.s = self._make(self.small, self._connect, self._build, reset_small=self.reset)
    self.addCleanup(self._close, self.s)
    self._wait_joined(self.s)
    self.s._engaged = False
    # the swap frame carries the history reset and is never counted as slow
    self.took = 0.3
    self.assertEqual(self.frame(), {'from': 'big'})
    self.assertTrue(self.s.chestnut)

  def _run(self, s):
    s._engagement_updated = joining.time.monotonic()
    return s.run({}, {}, {})

  def frame(self, took=None):
    if took is not None:
      self.took = took
    result = self._run(self.s)
    self.took = 0.03
    return result

  def assert_big_drives(self):
    self.assertEqual(self.frame(), {'from': 'big'})
    self.assertTrue(self.s.chestnut)

  def test_a_late_frame_is_published_and_the_next_one_is_the_small_models(self):
    self.assert_big_drives()
    self.assertEqual(self.frame(took=joining.LATE_FRAME + 0.01), {'from': 'big'})
    # the late frame's output is the large model's, and so is everything modeld
    # reads off the model for it; only modelV2.big says the handover now, which
    # is what makes modeld forgive the stall instead of counting it as lag
    self.assertFalse(self.s.chestnut)
    self.assertEqual(self.s.new_constant, 'big')
    self.reset.assert_not_called()
    self.assertEqual(self.frame(), {'from': 'small'})
    self.reset.assert_called_once()
    self.assertFalse(self.s.chestnut)
    self.assertEqual(self.s.new_constant, 'small')
    # counted, backed off and reported like a loss, and the link let go
    self.assertEqual(self.s._drops, 1)
    self.assertEqual(self.s.big_model_state, 'retrying')
    self.assertGreater(self.s._rejoin_at, joining.time.monotonic())
    self._wait_reported(self.s, 'the accelerator fell behind, reconnecting')
    for _ in range(100):
      if self.big.closed:
        break
      time.sleep(0.01)
    self.assertTrue(self.big.closed)

  def test_one_slow_frame_is_jitter(self):
    self.assertEqual(self.frame(took=joining.SLOW_FRAME + 0.005), {'from': 'big'})
    self.assertTrue(self.s.chestnut)
    for _ in range(5):
      self.assert_big_drives()
    self.reset.assert_not_called()

  def test_a_second_slow_frame_within_the_window_hands_back(self):
    self.frame(took=joining.SLOW_FRAME + 0.005)
    self.assert_big_drives()
    self.skew += joining.LAG_WINDOW / 2
    self.assertEqual(self.frame(took=joining.SLOW_FRAME + 0.005), {'from': 'big'})
    self.assertFalse(self.s.chestnut)
    self.assertEqual(self.frame(), {'from': 'small'})
    self.assertEqual(self.s._drops, 1)

  def test_slow_frames_further_apart_are_not_a_pattern(self):
    self.frame(took=joining.SLOW_FRAME + 0.005)
    self.skew += joining.LAG_WINDOW + 1
    self.frame(took=joining.SLOW_FRAME + 0.005)
    self.assert_big_drives()
    self.skew += joining.LAG_WINDOW + 1
    self.frame(took=joining.SLOW_FRAME + 0.005)
    self.assert_big_drives()
    self.reset.assert_not_called()

  def test_the_next_large_model_starts_with_no_strike(self):
    self.frame(took=joining.SLOW_FRAME + 0.005)
    self.frame(took=joining.LATE_FRAME + 0.01)
    self.frame()
    self.assertFalse(self.s.chestnut)
    self.assertIsNone(self.s._slow_at)


class EngagementTest(unittest.TestCase):
  """The swap window reads the adapter's poller, made on the watcher's own
  thread, and closes when the answers stop coming. What the poller answers
  (the MADS rule over selfdriveState, carState and carControl) is the fork's
  adapter's, and tested there."""

  def setUp(self):
    isolate(self, Path(tempfile.mkdtemp()))
    self.answers = []
    self.made_on = []
    self.engaged = True
    self.progress = mock.Mock()
    self.log = RecordingLog()
    self.small = FakeModel('small')
    self.release = threading.Event()
    self.addCleanup(self.release.set)

  def engagement(self):
    self.made_on.append(threading.current_thread().name)

    def engaged(timeout_ms):
      self.answers.append(timeout_ms)
      self.release.wait(timeout_ms / 1000)
      return self.engaged
    return engaged

  def state(self):
    def never(should_stop=None):
      raise RuntimeError('no jetson in this test')
    s = JoiningModelState(self.small, never, None, progress=self.progress, engagement=self.engagement, log=self.log)
    self.addCleanup(lambda: (s.close(), [t.join(5) for t in s._threads]))
    return s

  def wait_for(self, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if predicate():
        return True
      time.sleep(0.005)
    return False

  def test_disengaged_and_fresh_opens_the_window(self):
    self.engaged = False
    s = self.state()
    self.assertTrue(self.wait_for(lambda: s._window_open))
    self.assertEqual(set(self.answers), {joining.ENGAGEMENT_POLL_MS})

  def test_engaged_or_unknown_keeps_it_shut(self):
    # the adapter answers True for "not known" too: missing, dead or invalid messages
    s = self.state()
    self.assertTrue(self.wait_for(lambda: self.answers))
    time.sleep(0.05)
    self.assertFalse(s._window_open)
    self.engaged = False
    self.assertTrue(self.wait_for(lambda: s._window_open))
    self.engaged = True
    self.assertTrue(self.wait_for(lambda: not s._window_open))

  def test_a_poller_that_stops_answering_shuts_it(self):
    # the frame thread expires the answer: a stalled watcher cannot leave it open
    self.engaged = False
    s = self.state()
    self.assertTrue(self.wait_for(lambda: s._window_open))
    s._engagement_updated = time.monotonic() - 1.0
    self.assertFalse(s._window_open)

  def test_both_threads_leave_modelds_realtime_core_first(self):
    # created after config_realtime_process(7, 54), they inherit SCHED_FIFO on
    # core 7; each drops it before anything else, the watcher before it makes
    # the poller
    events = []
    self.made_on = events
    with mock.patch.object(joining, 'background_thread', lambda: events.append(('off', threading.current_thread().name))):
      s = self.state()
      self.assertTrue(self.wait_for(lambda: len(events) >= 3))
    join_loop, watcher = (t.name for t in s._threads)
    self.assertIn(('off', join_loop), events)
    self.assertLess(events.index(('off', watcher)), events.index(watcher), 'made the poller on a realtime thread')

  def test_the_poller_is_made_on_the_watchers_thread(self):
    # a SubMaster's sockets belong to the thread that made them
    self.state()
    self.assertTrue(self.wait_for(lambda: self.made_on))
    self.assertNotEqual(self.made_on, [threading.current_thread().name])

class FakeV2Model(FakeModel):
  """A modeld_v2 ModelState: constants, smoothing and the action function are its own."""

  def __init__(self, name, chestnut=False, client=None, desire_key='desire', slots=('desire', 'lateral_control_params')):
    super().__init__(name, chestnut, client)
    self.constants = mock.Mock(name=f'{name}.constants', MODEL_FREQ=20, DESIRE_LEN=8)
    self.desire_key = desire_key
    self.numpy_inputs = dict.fromkeys(slots)
    self.LAT_SMOOTH_SECONDS = 0.1 if name == 'small' else 0.0
    self.LONG_SMOOTH_SECONDS = 0.2 if name == 'small' else 0.3
    self.PLANPLUS_CONTROL = 1.0
    self.seen_inputs = None

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    self.seen_inputs = inputs
    return super().run(bufs, transforms, inputs, after_enqueue)

  def get_action_from_model(self, *args):
    return (self.name, args)


class ModeldV2FaceTest(JoiningBase):
  """What sunnypilot's modeld_tinygrad reads off the model: per-model, following the
  model that is driving, and the frame it built for the small bundle handed to
  whichever model takes it."""

  def setUp(self):
    super().setUp()
    self.small = FakeV2Model('small')
    self.big = FakeV2Model('big', chestnut=True, client=object(), desire_key='desire_pulse', slots=('desire', 'action_t', 'traffic_convention'))

  def _run_with(self, s, inputs):
    s._engagement_updated = time.monotonic()
    return s.run({}, {}, inputs)

  def _swap(self, s):
    self._wait_joined(s)
    s._engaged = False
    self._run(s)
    self.assertIs(s._active, self.big)

  def test_the_face_follows_the_model_that_drives(self):
    s = self._state()
    self.assertIs(s.constants, self.small.constants)
    self.assertEqual((s.LAT_SMOOTH_SECONDS, s.LONG_SMOOTH_SECONDS), (0.1, 0.2))
    self.assertEqual(s.get_action_from_model('out', 'prev'), ('small', ('out', 'prev')))
    self._swap(s)
    self.assertIs(s.constants, self.big.constants)
    self.assertEqual((s.LAT_SMOOTH_SECONDS, s.LONG_SMOOTH_SECONDS), (0.0, 0.3))
    self.assertEqual(s.get_action_from_model('out', 'prev'), ('big', ('out', 'prev')))

  def test_what_the_loop_writes_lands_on_both(self):
    s = self._state()
    s.PLANPLUS_CONTROL = 0.5
    self.assertEqual(self.small.PLANPLUS_CONTROL, 0.5)
    self._swap(s)
    self.assertEqual(self.big.PLANPLUS_CONTROL, 1.0)  # not yet written since the join
    s.PLANPLUS_CONTROL = 0.7
    self.assertEqual((self.small.PLANPLUS_CONTROL, self.big.PLANPLUS_CONTROL), (0.7, 0.7))

  def test_the_frame_is_built_for_the_small_bundle_and_handed_over_as_is(self):
    # the loop keys the desire input and probes the slots by the small bundle,
    # before and after the join; the large model takes the frame as built
    s = self._state()
    self.assertEqual(s.desire_key, 'desire')
    self.assertIs(s.numpy_inputs, self.small.numpy_inputs)
    self._wait_joined(s)
    s._engaged = False
    inputs = {'desire': object(), 'action_t': 1}
    self.assertEqual(self._run_with(s, inputs), {'from': 'big'})
    self.assertIs(self.big.seen_inputs, inputs)
    self.assertEqual(s.desire_key, 'desire')
    self.assertIs(s.numpy_inputs, self.small.numpy_inputs)
    # and back on a demote, the same frame
    self.big.raises = RuntimeError('link died')
    self.assertEqual(self._run_with(s, inputs), {'from': 'small'})
    self.assertIs(self.small.seen_inputs, inputs)


if __name__ == '__main__':
  unittest.main()
