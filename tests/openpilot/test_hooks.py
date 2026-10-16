"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The facade a fork's hook lines call: every hook answers, whatever jetlink
does, and the answer on a failure is the one the link off gets.
"""
import sys
import threading
import time
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from jetlink.openpilot import Status, hooks
from jetlink.openpilot.hooks import COMPATIBLE, NO_EVENTS, Hooks, Off
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
  """Every hook's answer, two for shutdown_step: a power-off not wanted and wanted."""
  joined = FakeJoinedModel(state='running')
  joined.hand_over()
  return {
    'should_run': h.should_run(False, None, None),
    'status': h.status(),
    'reason': h.reason(),
    'prepare': h.prepare(False),
    'attach': h.attach('model', object(), 1, 1),
    'handed_over': h.handed_over(joined),
    'state': h.state(joined),
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

  def test_each_failure_is_logged_once(self):
    log = RecordingLog()
    h = over(Broken(), log)
    for _ in range(3):
      answers(h)
    self.assertEqual(len(log.lines('exception')), len(OFF))
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


def tracker(update=None, settling=False, fail_init=False):
  """A stand-in jetlink.openpilot.events whose SwapEvents answers `update`
  (an exception raises), for sys.modules."""
  class SwapEvents:
    def __init__(self, tick_s):
      if fail_init:
        raise RuntimeError('no tracker')
      self.tick_s = tick_s
      self.settling = settling

    def update(self, fresh, state, big, in_control):
      if isinstance(answer := update(), Exception):
        raise answer
      return answer
  events = ModuleType('jetlink.openpilot.events')
  events.SwapEvents = SwapEvents
  return mock.patch.dict(sys.modules, {'jetlink.openpilot.events': events})


class TestSwapEvents(unittest.TestCase):
  ARGS = (True, 'running', True, False)

  def test_off_is_no_event_at_all(self):
    events = Off('why').swap_events(0.01)
    self.assertEqual(events.update(*self.ARGS), NO_EVENTS)
    self.assertEqual(NO_EVENTS._fields, ('offer', 'link_lost', 'no_entry', 'chime', 'settling'))
    self.assertFalse(any(NO_EVENTS))
    self.assertFalse(events.settling)

  def test_the_trackers_answers_pass_through(self):
    flags = SimpleNamespace(offer=True)
    with tracker(lambda: flags, settling=True):
      events = Hooks(3, mock.Mock()).swap_events(0.01)
    self.assertEqual(events._events.tick_s, 0.01)
    self.assertIs(events.update(*self.ARGS), flags)
    self.assertTrue(events.settling)

  def test_a_failing_update_is_no_event_logged_once_and_not_settling(self):
    log = RecordingLog()
    outcomes = iter([RuntimeError('a'), RuntimeError('a'), 'flags', RuntimeError('a')])
    with tracker(lambda: next(outcomes), settling=True):
      events = Hooks(3, mock.Mock(), log=log).swap_events(0.01)
    self.assertEqual(events.update(*self.ARGS), NO_EVENTS)
    self.assertFalse(events.settling)
    self.assertEqual([events.update(*self.ARGS) for _ in range(3)], [NO_EVENTS, 'flags', NO_EVENTS])
    self.assertEqual(log.lines('exception'), ["jetlink: swap_events.update() failed: RuntimeError: a"] * 2)

  def test_a_tracker_that_will_not_build_is_inert(self):
    log = RecordingLog()
    with tracker(fail_init=True):
      events = Hooks(3, mock.Mock(), log=log).swap_events(0.01)
    self.assertEqual((events.update(*self.ARGS), events.settling), (NO_EVENTS, False))
    self.assertTrue(log.has("swap_events() failed: RuntimeError: no tracker"))

  def test_a_settling_that_raises_is_false(self):
    with tracker(lambda: 'flags'):
      events = Hooks(3, mock.Mock()).swap_events(0.01)
    type(events._events).settling = property(lambda self: 1 / 0)
    self.assertFalse(events.settling)

  def test_it_builds_nothing_of_the_adapters(self):
    factory = mock.Mock()
    with tracker(lambda: 'flags'):
      Hooks(3, factory).swap_events(0.01).update(*self.ARGS)
    factory.assert_not_called()


class TestOverTheRealJetlink(OpenpilotTest):
  def setUp(self):
    super().setUp()
    self.h = Hooks(3, lambda: self.op)

  def test_the_link_off_is_jetlinks_answer_not_the_nets(self):
    self.op.set_mode('off')
    found = answers(self.h)
    self.assertIsInstance(found.pop('status'), Status)
    found.pop('extend_catalog')
    # off the model it is handed, and the catalog is hardware, not the setting
    expected = {**OFF, 'handed_over': True, 'state': 'running', 'should_extend_catalog': True}
    del expected['status'], expected['extend_catalog']
    self.assertEqual(found, expected)
    self.assertEqual(self.op.log.lines('exception'), [])

  def test_a_handover_is_the_count_moving_across_a_run(self):
    model = FakeJoinedModel()
    self.assertFalse(self.h.handed_over(model))
    model.hand_over()
    self.assertTrue(self.h.handed_over(model))
    self.assertFalse(self.h.handed_over(model))
    model.hand_over('ready')
    model.hand_over('running')
    self.assertTrue(self.h.handed_over(model))

  def test_the_state_is_the_joining_models_and_none_for_any_other(self):
    from jetlink.openpilot import STATES
    self.assertEqual(self.h.state(FakeJoinedModel(state='retrying')), 'retrying')
    self.assertEqual(self.h.state(object()), 'none')
    self.assertFalse(self.h.handed_over(object()))
    self.assertIn(self.h.state(object()), STATES)


if __name__ == '__main__':
  unittest.main()
