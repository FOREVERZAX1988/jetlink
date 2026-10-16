"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The kit forks run in their own CI: it passes jetlink's own fake adapter and
both of zoompilot's modelds, and names what is wrong with a bad one.
"""
import importlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from jetlink.openpilot import interface
from jetlink.openpilot.hooks import Off
from jetlink.openpilot.stock import Stock
from jetlink.openpilot.testing import FakeJoinedModel, FakeSubMaster, Guard, check_adapter, check_fallback, check_seam
from tests.openpilot.test_hooks import SIGNATURES
from tests.openpilot import fakes

# zoompilot's modelds, where a checkout sits beside this one; never in CI
FORK = Path(__file__).resolve().parents[3] / 'sunnypilot-cross-fork' / 'openpilot'
MODELDS = (FORK / 'selfdrive' / 'modeld' / 'modeld.py', FORK / 'sunnypilot' / 'modeld_v2' / 'modeld.py')


# an owner_config() of jetlink's alone, which a module outside any fork's package may import
OWNER_CONFIG = """
def owner_config():
  from pathlib import Path
  from jetlink.openpilot.interface import Keys, OwnerConfig
  return OwnerConfig(Path('params'), Keys('link', 'offroad', 'progress', 'spec', 'pointers'), frozenset(), __name__,
                     Path('.'), {}, Path('owner.log'))
"""


class TestCheckAdapter(unittest.TestCase):
  def module(self, source: str, owner_config: str = OWNER_CONFIG):
    """An adapter module of `source`, importable here and in a fresh interpreter."""
    root = Path(tempfile.mkdtemp())
    name = f'adapter_{abs(hash(source))}'
    (root / f'{name}.py').write_text(textwrap.dedent(source) + owner_config)
    p = mock.patch.object(sys, 'path', [str(root), *sys.path])
    p.start()
    self.addCleanup(p.stop)
    self.addCleanup(sys.modules.pop, name, None)
    return importlib.import_module(name)

  def test_jetlinks_fake_adapter_passes(self):
    self.assertEqual(check_adapter(fakes), [])

  def test_a_heavy_top_level_and_a_member_that_is_not_a_method_are_named(self):
    m = self.module('''
      import numpy

      def adapter():
        from tests.openpilot import fakes

        class Adapter(fakes.FakeOpenpilot):
          in_control = None
        return Adapter()
    ''')
    self.assertEqual(check_adapter(m), [f"{m.__name__}'s top level imports ['numpy']: manager and the resident owner import it",
                                        "adapter: in_control: not callable"])

  def test_an_adapter_missing_a_side_is_named_by_member(self):
    m = self.module('''
      class Adapter:
        keys = None

      def adapter():
        return Adapter()
    ''')
    problems = check_adapter(m)
    self.assertIn("adapter: in_control: missing", problems)
    self.assertIn("adapter: warp_path: missing", problems)
    self.assertNotIn("adapter: small_reset: missing", problems)   # optional

  def test_a_heavy_owner_config_and_a_slow_answer_are_named(self):
    m = self.module('''
      import time

      def adapter():
        from tests.openpilot import fakes

        class Adapter(fakes.FakeOpenpilot):
          def in_control(self):
            time.sleep(0.06)
            return 1
        return Adapter()
    ''', owner_config='''
def owner_config():
  import numpy
  return {}
''')
    problems = check_adapter(m)
    self.assertEqual(problems[:3], ["owner_config() imports ['numpy']: the resident owner calls it",
                                    "owner_config(): not an OwnerConfig", "in_control() answered 1, not a bool"])
    self.assertTrue(problems[3].startswith("in_control() took"), problems)

  def test_a_forks_own_heavy_module_counts(self):
    m = self.module('''
      def adapter():
        from tests.openpilot import fakes
        return fakes.FakeOpenpilot()
    ''')
    self.assertEqual(check_adapter(m), [])
    self.assertEqual(check_adapter(m, heavy=('jetlink.openpilot.interface',)),
                     ["owner_config() imports ['jetlink.openpilot.interface']: the resident owner calls it"])

  def test_an_adapter_module_without_its_functions(self):
    m = self.module('''
      def adapter():
        return None
    ''', owner_config='')
    self.assertEqual(check_adapter(m), [f"{m.__name__}.owner_config(): missing"])

  def test_a_module_that_prints_as_it_imports(self):
    m = self.module('''
      print('hello from the fork')

      def adapter():
        from tests.openpilot import fakes
        return fakes.FakeOpenpilot()
    ''')
    self.assertEqual(check_adapter(m), [])

  def test_a_probe_that_hangs_is_a_problem(self):
    with mock.patch.object(subprocess, 'run', side_effect=subprocess.TimeoutExpired('python', 120)):
      self.assertEqual(check_adapter(fakes), [(f"{fakes.__name__} did not import and build its owner_config() in a fresh "
                                               "interpreter within 120 s")])

  def test_a_module_that_will_not_import_fresh(self):
    m = self.module('''
      import sys
      if sys.argv[0] == '-c':   # the fresh interpreter's
        raise ImportError('only here')
      def adapter():
        return None
    ''')
    self.assertIn("does not import in a fresh interpreter", check_adapter(m)[0])


GOOD = '''
def main(demo=False):
  CHESTNUT = chestnut_present()
  HOOKS.prepare(CHESTNUT)  # before going realtime
  config_realtime_process(7, 54)
  small_model = ModelState(1, 2, False) if CHESTNUT else None
  model = small_model
  model = HOOKS.attach(model, small_model, 1, 2)
  while True:
    model.frame_drop_ratio = 0.
    model.run(1, 2, 3)
    if HOOKS.handed_over(model):
      pass
    send.acceleratorState = HOOKS.state(model)
'''


class TestCheckSeam(unittest.TestCase):
  def seam(self, source: str) -> list[str]:
    path = Path(tempfile.mkdtemp()) / 'modeld.py'
    path.write_text(source)
    return check_seam(path)

  def test_a_good_seam(self):
    self.assertEqual(self.seam(GOOD), [])
    # zoompilot's shape: the fork's module rather than a Hooks
    self.assertEqual(self.seam(GOOD.replace('HOOKS', 'jetlink_adapter')), [])

  def test_prepare_after_going_realtime(self):
    source = GOOD.replace('  HOOKS.prepare(CHESTNUT)  # before going realtime\n', '').replace(
      '  config_realtime_process(7, 54)\n', '  config_realtime_process(7, 54)\n  HOOKS.prepare(CHESTNUT)\n')
    self.assertEqual(self.seam(source), [("modeld.py: prepare() comes after config_realtime_process(): the GPU's thread"
                                          " would inherit the realtime core")])

  def test_attach_before_the_small_model(self):
    source = GOOD.replace('  model = HOOKS.attach(model, small_model, 1, 2)\n', '').replace(
      '  small_model = ', '  model = HOOKS.attach(model, small_model, 1, 2)\n  small_model = ')
    self.assertEqual(self.seam(source), ["modeld.py: attach() comes before small_model is built"])

  def test_a_dropped_attach(self):
    source = GOOD.replace('model = HOOKS.attach(', 'HOOKS.attach(')
    self.assertEqual(self.seam(source), ["modeld.py: attach()'s result is dropped: modeld would never run the joined model"])

  def test_no_frame_drop_ratio_written(self):
    source = GOOD.replace('    model.frame_drop_ratio = 0.\n', '')
    self.assertEqual(self.seam(source), [("modeld.py: main() never writes model.frame_drop_ratio: a host that drops "
                                          "camera frames would never hand back")])

  def test_the_frame_loops_reads(self):
    source = GOOD.replace('HOOKS.handed_over(model)', 'False')
    self.assertEqual(self.seam(source), ["modeld.py: main() never calls HOOKS.handed_over(model)"])

  def test_state_is_the_forks_to_publish(self):
    # a fork with no message for the state reads it off Status.running instead
    self.assertEqual(self.seam(GOOD.replace('HOOKS.state(model)', "'none'")), [])

  def test_hooks_on_another_object_do_not_count(self):
    source = GOOD.replace('HOOKS.prepare(', 'other.prepare(')
    self.assertEqual(self.seam(source), ["modeld.py: main() never calls HOOKS.prepare()"])

  def test_a_body_in_one_try_keeps_its_order(self):
    # zoompilot's modeld_v2 shape: everything in main() inside one try/finally
    body = GOOD.split('\n', 2)[2]
    source = 'def main(demo=False):\n  try:\n' + textwrap.indent(body, '  ') + '  finally:\n    pass\n'
    self.assertEqual(self.seam(source), [])
    bad = source.replace('    HOOKS.prepare(CHESTNUT)  # before going realtime\n', '').replace(
      '    config_realtime_process(7, 54)\n', '    config_realtime_process(7, 54)\n    HOOKS.prepare(CHESTNUT)\n')
    self.assertEqual(len(self.seam(bad)), 1)

  def test_attach_by_keyword_and_annotated(self):
    source = GOOD.replace('model = HOOKS.attach(model, small_model, 1, 2)',
                          'model: object = HOOKS.attach(model, small=small_model, cam_w=1, cam_h=2)')
    self.assertEqual(self.seam(source), [])
    late = source.replace('  small_model = ModelState(1, 2, False) if CHESTNUT else None\n', '').replace(
      '  model = small_model\n', '  model = None\n') + '  small_model = None\n'
    self.assertEqual(self.seam(late), ["modeld.py: attach() comes before small_model is built"])

  def test_a_joined_model_the_loop_never_runs(self):
    source = GOOD.replace('model = HOOKS.attach(', 'joined = HOOKS.attach(')
    self.assertEqual(self.seam(source), [("modeld.py: attach()'s result, joined, is never run: modeld would keep driving "
                                          "the model it had")])

  def test_no_seam_at_all(self):
    self.assertEqual(self.seam('def main():\n  config_realtime_process(7, 54)\n'),
                     ["modeld.py: main() never calls attach(model, small, cam_w, cam_h)"])
    self.assertEqual(self.seam('x = 1\n'), ["modeld.py: no main()"])

  @unittest.skipUnless(all(p.is_file() for p in MODELDS), "no zoompilot checkout beside this one")
  def test_both_of_zoompilots_modelds(self):
    for path in MODELDS:
      with self.subTest(path.name):
        self.assertEqual(check_seam(path), [])


class NoJetlink:
  """A fork's fallback for no jetlink at all, as docs/porting.md's."""
  def should_run(self, started, params, CP): return False
  def status(self): return None
  def reason(self): return "jetlink is missing"
  def prepare(self, chestnut): return False
  def handed_over(self, model): return False
  def shutdown_step(self, should_shutdown, reason=''): return should_shutdown
  def runtime(self): return None


class TestCheckFallback(unittest.TestCase):
  def test_off_and_a_good_fallback_pass(self):
    self.assertEqual(check_fallback(Off('why', link_on=lambda: True)), [])
    self.assertEqual(check_fallback(NoJetlink()), [])
    self.assertEqual(check_fallback(NoJetlink(), link_on=lambda: True), [])

  def test_with_the_link_off_the_reason_is_none(self):
    self.assertEqual(check_fallback(Off('why', link_on=lambda: False), link_on=lambda: False), [])
    self.assertEqual(check_fallback(NoJetlink(), ['reason'], link_on=lambda: False),
                     ["reason() answered 'jetlink is missing', where the link off is None"])

    def fails():
      raise OSError('no params')
    self.assertEqual(check_fallback(NoJetlink(), ['reason'], link_on=fails), ["link_on() raised OSError: no params"])

  def test_it_knows_every_hook(self):
    self.assertEqual(check_fallback(Off(), SIGNATURES), [])

  def test_a_wrong_answer_is_named(self):
    class Wrong(NoJetlink):
      def prepare(self, chestnut): return chestnut
      def shutdown_step(self, should_shutdown, reason=''): return False
      def should_run(self, started, params, CP): return 0
      def attach(self, model, small, cam_w, cam_h): return small
    problems = check_fallback(Wrong())
    self.assertEqual(problems[:2] + problems[3:], ["should_run() answered 0, where the link off is False",
                                                   "prepare() answered True, where the link off is False",
                                                   "shutdown_step() answered False, where the link off is True"])
    self.assertRegex(problems[2], r"^attach\(\) answered <object .*>, where the link off is <object .*>$")

  def test_a_hook_the_fork_calls_must_be_there_and_answer(self):
    class Raising(NoJetlink):
      def status(self): raise RuntimeError('boom')
    Raising.nonsense = Raising.status
    self.assertEqual(check_fallback(Raising(), ['status', 'attach', 'nonsense']),
                     ["status() raised RuntimeError: boom", "attach(): missing", "nonsense(): not a hook"])

  def test_no_swap_events_or_inert_ones(self):
    flags = SimpleNamespace(offer=False, link_lost=False, no_entry=False, chime=False, settling=False)
    for answer, problems in ((None, 0), (SimpleNamespace(settling=False, update=lambda *a, **k: flags), 0),
                             (SimpleNamespace(settling=True, update=lambda *a, **k: flags), 1)):
      fallback = SimpleNamespace(swap_events=lambda tick_s, answer=answer: answer)
      self.assertEqual(len(check_fallback(fallback)), problems, answer)


class TestFakeSubMaster(unittest.TestCase):
  def test_stocks_in_control_over_it(self):
    idle = {'carState': SimpleNamespace(), 'carControl': SimpleNamespace(enabled=False, latActive=False, longActive=False),
            'selfdriveState': SimpleNamespace(enabled=False, active=False)}
    made = []

    def submaster(services):
      made.append(FakeSubMaster(services, **idle))
      return made[-1]
    messaging = ModuleType('openpilot.cereal.messaging')
    messaging.SubMaster = submaster
    cereal = ModuleType('openpilot.cereal')
    cereal.messaging = messaging
    with mock.patch.dict(sys.modules, {'openpilot': ModuleType('openpilot'), 'openpilot.cereal': cereal,
                                       'openpilot.cereal.messaging': messaging}):
      op = Stock()
      self.assertIs(op.in_control(), False)
      [sm] = made
      sm['carControl'].latActive = True
      self.assertIs(op.in_control(), True)
      sm['carControl'].latActive = False
      sm.valid['carState'] = False
      self.assertIs(op.in_control(), True)
    self.assertEqual(sm.timeouts, [0, 0, 0])

  def test_what_a_reader_sees(self):
    sm = FakeSubMaster(['modelV2', 'carState'], modelV2=SimpleNamespace(big=True))
    self.assertEqual((sm.seen, sm.alive, sm.valid), ({'modelV2': True, 'carState': False},) * 3)
    self.assertFalse(sm.all_alive() or sm.all_checks())
    self.assertTrue(sm.all_checks(['modelV2']))
    sm.update(0)
    self.assertEqual(sm.updated, {'modelV2': True, 'carState': False})
    sm.alive['modelV2'] = False
    sm.update(0)
    self.assertEqual((sm.frame, sm.recv_frame, sm.updated), (2, {'modelV2': 1, 'carState': 0},
                                                             {'modelV2': False, 'carState': False}))
    self.assertTrue(sm['modelV2'].big)

  def test_a_service_with_no_message_reads_blank(self):
    sm = FakeSubMaster(['carControl'])
    self.assertFalse(sm['carControl'].enabled or sm['carControl'].mads.enabled)
    with self.assertRaises(KeyError):
      sm['selfdriveState']


class TestTheFakes(unittest.TestCase):
  def test_the_kit_imports_no_test_framework_or_numpy(self):
    # a fork's conftest imports it before its own fixtures
    code = 'import sys, jetlink.openpilot.testing; print(sorted({"numpy", "pytest", "unittest"} & set(sys.modules)))'
    run = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2], capture_output=True,
                         text=True, timeout=120)
    self.assertEqual((run.returncode, run.stdout.strip()), (0, '[]'), run.stderr)

  def test_the_fake_joined_model_is_one(self):
    self.assertEqual(interface.conformance(FakeJoinedModel(), interface.JoinedModel), [])

  def test_it_drives_as_the_small_model_and_counts_handovers(self):
    small = fakes.SmallModel()
    joined = FakeJoinedModel(small)
    self.assertEqual(joined.run(None, None, None), {'from': 'small'})
    joined.hand_over('running')
    self.assertEqual((joined.handovers, joined.big_model_state, joined.runs), (1, 'running', 1))
    joined.close()
    self.assertTrue(joined.closed)

  def test_the_guard_records_only_while_started(self):
    guard = Guard(('/forbidden/',))
    guard('open', ('/forbidden/x',))
    guard.start()
    guard('open', ('/forbidden/x', 'r'))
    guard('open', ('/forbidden/x', 'r'))
    guard('open', ('/elsewhere/x', 'r'))
    guard('socket.connect', (None, '/forbidden/sock'))
    guard('subprocess.Popen', ('sudo', ['sudo', 'true'], None, None))
    self.assertEqual(guard.stop(), ['open /forbidden/x', 'socket.connect /forbidden/sock', 'subprocess.Popen sudo'])
    self.assertEqual(guard.stop(), [])

  def test_tinygrads_cpu_quota_read_is_no_reach(self):
    guard = Guard()
    guard.start()
    guard('open', ('/sys/fs/cgroup/cpu.max', 'r'))
    guard('open', ('/sys/class/power_supply/x', 'r'))
    self.assertEqual(guard.stop(), ['open /sys/class/power_supply/x'])


if __name__ == '__main__':
  unittest.main()
