"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The facade a fork's hook lines call: every hook answers, whatever jetlink
does, and the answer on a failure is the one the link off gets.
"""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from jetlink.openpilot import STATES, Flags, Status, hooks
from jetlink.openpilot.hooks import COMPATIBLE, Hooks, Off
from jetlink.openpilot.testing import FakeJoinedModel, RecordingLog
from tests.openpilot.fakes import OpenpilotTest
from tests.openpilot.test_api import plain

# what a fork's hook lines call, as they call them. Additive only, as Jetlink's
SIGNATURES = {
  'should_run': '(started, params, CP)',
  'status': '()',
  'reason': '()',
  'prepare': '(chestnut)',
  'attach': '(model, small, cam_w, cam_h)',
  'handed_over': '(model)',
  'state': '(model)',
  'shutdown_step': "(should_shutdown, reason='')",
  'request_shutdown': "(reason='')",
  'shutdown_pending': '()',
  'model_state': '(ref)',
  'should_extend_catalog': '()',
  'extend_catalog': '(catalog)',
  'swap_events': '(tick_s)',
}
CATALOG = {'bundles': []}


def answers(h) -> dict:
  """Every hook's answer, two for shutdown_step: a power-off not wanted and
  wanted. modeld's own model, as it runs with the link off."""
  own = object()
  return {
    'should_run': h.should_run(False, None, None),
    'status': h.status(),
    'reason': h.reason(),
    'prepare': h.prepare(False),
    'attach': h.attach('model', object(), 1, 1),
    'handed_over': h.handed_over(own),
    'state': h.state(own),
    'shutdown_step': (h.shutdown_step(False), h.shutdown_step(True, 'test')),
    'request_shutdown': h.request_shutdown('test'),
    'shutdown_pending': h.shutdown_pending(),
    'model_state': h.model_state('f' * 40),
    'should_extend_catalog': h.should_extend_catalog(),
    'extend_catalog': h.extend_catalog(CATALOG) is CATALOG,
  }


# the link off: modeld keeps its model, the catalog stays as fetched, and a
# power-off that is wanted goes ahead
OFF = {'should_run': False, 'status': None, 'reason': None, 'prepare': False, 'attach': 'model',
       'handed_over': False, 'state': 'none', 'shutdown_step': (False, True), 'request_shutdown': False,
       'shutdown_pending': False, 'model_state': None, 'should_extend_catalog': False, 'extend_catalog': True}


class Broken:
  """A Jetlink whose every method raises."""

  def __getattr__(self, name):
    def fail(*args):
      raise RuntimeError(name)
    return fail


def over(jetlink, log=None) -> Hooks:
  """Hooks whose bind() hands back `jetlink`, over an adapter logging to `log`."""
  op = SimpleNamespace(log=log or RecordingLog())
  h = Hooks(3, lambda: op)
  p = mock.patch.object(hooks, 'bind', lambda adapter: jetlink)
  p.start()
  h._jetlink()
  p.stop()
  return h


class TestTheSurface(unittest.TestCase):
  def test_the_hooks_a_fork_calls(self):
    for cls in (Hooks, Off):
      self.assertEqual({n for n in dir(cls) if not n.startswith('_')}, set(SIGNATURES), cls.__name__)
      for name, expected in SIGNATURES.items():
        with self.subTest(f'{cls.__name__}.{name}'):
          self.assertEqual(plain(getattr(cls(3, object) if cls is Hooks else cls(), name)), expected)

  def test_the_table_covers_every_hook(self):
    # swap_events' answer is a tracker, TestSwapEvents'
    self.assertEqual(set(OFF), set(SIGNATURES) - {'swap_events'})

  def test_todays_api_is_accepted(self):
    from jetlink.openpilot import API
    self.assertIn(API, COMPATIBLE)

  def test_another_api_is_refused_at_once_and_says_why(self):
    factory = mock.Mock()
    for api in (2, 4, None):
      with self.subTest(api), self.assertRaises(ValueError) as e:
        Hooks(api, factory)
      self.assertEqual(str(e.exception), f"jetlink package API 3, this build expects {api}")
    factory.assert_not_called()


class TestOff(unittest.TestCase):
  def test_every_hook_is_the_link_off(self):
    self.assertEqual(answers(Off()), OFF)

  def test_the_reason_is_told_to_someone_who_turned_the_link_on(self):
    on = [True]
    off = Off("jetlink package API 4, this build expects 3", link_on=lambda: on[0])
    self.assertEqual(answers(off), {**OFF, 'reason': "jetlink package API 4, this build expects 3"})
    on[0] = False
    self.assertIsNone(off.reason())

  def test_without_a_way_to_read_the_setting_it_is_told(self):
    self.assertEqual(Off('why').reason(), 'why')

  def test_a_setting_that_cannot_be_read_tells_nobody(self):
    def fail():
      raise OSError('no params')
    self.assertIsNone(Off('why', link_on=fail).reason())


class TestAFailingJetlink(unittest.TestCase):
  def test_it_never_raises_into_its_caller(self):
    self.assertEqual(answers(over(Broken())), OFF)

  def test_a_joined_link_is_the_model_modeld_runs(self):
    joined, small = object(), object()
    jetlink = mock.Mock(**{'attach.return_value': joined})
    self.assertIs(over(jetlink).attach('model', small, 1928, 1208), joined)
    jetlink.attach.assert_called_once_with(small, 1928, 1208)
    jetlink.attach.return_value = None
    self.assertEqual(over(jetlink).attach('model', small, 1928, 1208), 'model')

  def test_each_failure_is_logged_once(self):
    log = RecordingLog()
    h = over(Broken(), log)
    for _ in range(3):
      answers(h)
    # all but handed_over and state, which read the model alone
    self.assertEqual(len(log.lines('exception')), len(OFF) - 2)
    self.assertTrue(log.has("status() failed: RuntimeError: status"))

  def test_a_failure_is_logged_again_once_it_changes_or_has_cleared(self):
    outcomes = iter([RuntimeError('a'), RuntimeError('a'), RuntimeError('b'), None, RuntimeError('b')])

    class Flaky:
      def status(self):
        if (e := next(outcomes)) is not None:
          raise e
        return 'a snapshot'
    log = RecordingLog()
    h = over(Flaky(), log)
    self.assertEqual([h.status() for _ in range(5)], [None, None, None, 'a snapshot', None])
    self.assertEqual(log.lines('exception'), ["jetlink: status() failed: RuntimeError: a",
                                              "jetlink: status() failed: RuntimeError: b",
                                              "jetlink: status() failed: RuntimeError: b"])

  def test_a_power_off_that_is_wanted_goes_ahead(self):
    # a broken jetlink must never keep a comma on to drain the car's battery
    h = over(Broken())
    self.assertTrue(h.shutdown_step(True))
    self.assertFalse(h.shutdown_step(False))


class TestBinding(unittest.TestCase):
  def test_nothing_is_built_until_a_hook_needs_it(self):
    # manager and the resident owner import the fork's hook module
    factory = mock.Mock(return_value=SimpleNamespace(log=RecordingLog()))
    with mock.patch.object(hooks, 'bind', return_value=SimpleNamespace(status=lambda: 'snapshot')) as bind:
      h = Hooks(3, factory)
      factory.assert_not_called()
      self.assertEqual([h.status(), h.status()], ['snapshot', 'snapshot'])
    factory.assert_called_once_with()
    bind.assert_called_once_with(factory.return_value)

  def test_a_chestnut_never_asks_jetlink(self):
    factory = mock.Mock()
    self.assertFalse(Hooks(3, factory).prepare(True))
    factory.assert_not_called()

  def test_threads_share_one_binding(self):
    made = []

    def factory():
      made.append(1)
      time.sleep(0.05)
      return SimpleNamespace(log=RecordingLog())
    h = Hooks(3, factory)
    with mock.patch.object(hooks, 'bind', return_value=SimpleNamespace(status=lambda: 'snapshot')):
      threads = [threading.Thread(target=h.status) for _ in range(8)]
      for t in threads:
        t.start()
      for t in threads:
        t.join()
    self.assertEqual(made, [1])

  def test_an_adapter_that_will_not_build_is_the_link_off_and_says_why(self):
    log, on = RecordingLog(), [True]

    def factory():
      raise RuntimeError('no params')
    h = Hooks(3, factory, link_on=lambda: on[0], log=log)
    why = "jetlink failed to start: RuntimeError: no params"
    self.assertEqual(answers(h), {**OFF, 'reason': why})
    answers(h)
    self.assertEqual(log.lines('exception'), [f"jetlink: {why}"])
    on[0] = False
    self.assertIsNone(h.reason())


class Unprintable(Exception):
  def __str__(self):
    raise RuntimeError('no str')


class RaisingLog:
  def __getattr__(self, name):
    def fail(*args, **kwargs):
      raise RuntimeError('the log is down')
    return fail


class TestNothingEscapes(unittest.TestCase):
  def test_a_log_that_raises(self):
    h = over(Broken(), RaisingLog())
    self.assertEqual(answers(h), OFF)
    self.assertEqual(h.swap_events(0).update(True, 'ready', False, True), hooks.NO_EVENTS)

  def test_an_error_whose_str_raises(self):
    log = RecordingLog()

    class Odd:
      def __getattr__(self, name):
        def fail(*args):
          raise Unprintable()
        return fail
    self.assertEqual(answers(over(Odd(), log)), OFF)
    self.assertTrue(log.has("status() failed: Unprintable"))

  def test_both_at_bind(self):
    def factory():
      raise Unprintable()
    h = Hooks(3, factory, log=RaisingLog(), link_on=lambda: True)
    self.assertEqual(answers(h), {**OFF, 'reason': "jetlink failed to start: Unprintable"})

  def test_a_model_whose_reads_raise(self):
    class Odd:
      @property
      def handovers(self):
        raise RuntimeError('no count')
      big_model_state = handovers
    for h in (Off(), Hooks(3, mock.Mock())):
      self.assertEqual((h.handed_over(Odd()), h.state(Odd())), (False, 'none'))

  def test_a_log_factory_is_asked_once_and_one_that_raises_is_logging(self):
    log, asked = RecordingLog(), []

    def factory():
      asked.append(1)
      return log

    def adapter():
      raise RuntimeError('no params')
    answers(Hooks(3, adapter, log=factory))
    self.assertEqual(asked, [1])
    self.assertEqual(log.lines('exception'), ["jetlink: jetlink failed to start: RuntimeError: no params"])

    def failing():
      raise RuntimeError('no cloudlog')
    with self.assertLogs('jetlink', 'ERROR'):
      answers(Hooks(3, adapter, log=failing))


class TestTheModelsReads(unittest.TestCase):
  def test_a_handover_is_the_count_moving_across_a_run_and_binds_nothing(self):
    factory = mock.Mock()
    for h in (Off(), Hooks(3, factory)):
      model = FakeJoinedModel()
      self.assertFalse(h.handed_over(model))
      model.hand_over()
      self.assertTrue(h.handed_over(model))
      self.assertFalse(h.handed_over(model))
      model.hand_over('ready')
      model.hand_over('running')
      self.assertTrue(h.handed_over(model))
    factory.assert_not_called()

  def test_the_state_is_the_models_and_none_for_one_without(self):
    for h in (Off(), Hooks(3, mock.Mock())):
      self.assertEqual(h.state(FakeJoinedModel(state='retrying')), 'retrying')
      self.assertEqual(h.state(object()), 'none')
      self.assertFalse(h.handed_over(object()))


class TestSwapEvents(unittest.TestCase):
  ARGS = (True, 'running', True, False)

  def test_off_is_no_event_at_all(self):
    events = Off('why').swap_events(0.01)
    self.assertEqual(events.update(*self.ARGS), hooks.NO_EVENTS)
    self.assertIsInstance(hooks.NO_EVENTS, Flags)
    self.assertFalse(any(hooks.NO_EVENTS))
    self.assertFalse(events.settling)

  def test_over_the_real_tracker(self):
    factory = mock.Mock()
    events = Hooks(3, factory).swap_events(0.01)
    # ready while in control: offered, for a while
    self.assertTrue(events.update(True, 'ready', False, True).offer)
    # the swap: a second of no entry, and settling
    flags = events.update(True, 'running', True, False)
    self.assertTrue(flags.no_entry and flags.settling)
    self.assertTrue(events.settling)
    for _ in range(100):
      flags = events.update(True, 'running', True, False)
    self.assertFalse(flags.no_entry or events.settling)
    factory.assert_not_called()

  def test_a_failing_update_is_no_event_logged_once_and_not_settling(self):
    log = RecordingLog()
    events = Hooks(3, mock.Mock(), log=log).swap_events(0.01)
    events.update(True, 'running', True, False)
    self.assertTrue(events.settling)
    outcomes = iter([RuntimeError('a'), RuntimeError('a'), Flags(True, False, False, False, False), RuntimeError('a')])

    def update(*args):
      if isinstance(answer := next(outcomes), Exception):
        raise answer
      return answer
    with mock.patch.object(events._events, 'update', update):
      self.assertEqual(events.update(*self.ARGS), hooks.NO_EVENTS)
      self.assertFalse(events.settling)
      self.assertEqual([events.update(*self.ARGS).offer for _ in range(3)], [False, True, False])
    self.assertEqual(log.lines('exception'), ["jetlink: swap_events.update() failed: RuntimeError: a"] * 2)

  def test_a_period_it_cannot_count_is_inert_and_said(self):
    log = RecordingLog()
    events = Hooks(3, mock.Mock(), log=log).swap_events(0)
    self.assertEqual((events.update(*self.ARGS), events.settling), (hooks.NO_EVENTS, False))
    self.assertTrue(log.has("swap_events() failed: ValueError: SwapEvents needs the loop period"))

  def test_a_settling_that_raises_is_false(self):
    events = Hooks(3, mock.Mock()).swap_events(0.01)
    with mock.patch.object(type(events._events), 'settling', property(lambda self: 1 / 0)):
      self.assertFalse(events.settling)


class TestOverTheRealJetlink(OpenpilotTest):
  def setUp(self):
    super().setUp()
    self.h = Hooks(3, lambda: self.op)

  def test_the_link_off_is_jetlinks_answer_not_the_nets(self):
    self.op.set_mode('off')
    found = answers(self.h)
    self.assertIsInstance(found.pop('status'), Status)
    found.pop('extend_catalog')
    # the catalog is hardware, not the setting
    expected = {**OFF, 'should_extend_catalog': True}
    del expected['status'], expected['extend_catalog']
    self.assertEqual(found, expected)
    self.assertEqual(self.op.log.lines('exception'), [])

  def test_jetlinks_own_reads_of_the_model(self):
    model = FakeJoinedModel(state='retrying')
    self.assertFalse(self.jl.handed_over(model))
    model.hand_over('running')
    self.assertEqual((self.jl.handed_over(model), self.jl.handed_over(model)), (True, False))
    self.assertEqual((self.jl.state(model), self.jl.state(object())), ('running', 'none'))
    self.assertIn(self.jl.state(object()), STATES)


if __name__ == '__main__':
  unittest.main()
