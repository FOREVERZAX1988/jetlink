"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the fork calls, and the answers it gets.

modeld, manager, hardwared, the UI and the model manager reach jetlink
through bind() and the Jetlink it returns, so what is pinned here is the
surface itself and selection: what a device with the link off, on but not
ready, and ready gets told, and that a request that hangs costs the large
model and nothing else.
"""
import inspect
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jetlink.openpilot as jo
from jetlink.openpilot import interface, status
from jetlink.comma import gadget
from jetlink.openpilot import joining
from jetlink.spec import ModelSpec
from tests.openpilot import fakes
from tests.openpilot.fakes import OpenpilotTest

# what the fork calls on a Jetlink, as it calls it. Additive only: a changed
# or removed one is an API bump (see jetlink/openpilot/__init__.py)
JETLINK = {
  'enabled': '()',
  'status': '()',
  'reason': '()',
  'prepare': '()',
  'attach': '(small, cam_w, cam_h)',
  'shutdown': "(reason='', timeout=25.0)",
  'should_extend_catalog': '()',
  'extend_catalog': '(catalog)',
}


def plain(fn) -> str:
  sig = inspect.signature(fn)
  return str(sig.replace(parameters=[p.replace(annotation=p.empty) for p in sig.parameters.values()],
                         return_annotation=sig.empty))


# what the package exports: the contract, and nothing jetlink keeps to itself
EXPORTS = ['API', 'MODES', 'STATES', 'Jetlink', 'Keys', 'ModelFace', 'Openpilot', 'OwnerConfig', 'Status', 'bind',
           'conformance']


class TestTheSurface(OpenpilotTest):
  def test_the_methods_the_fork_calls(self):
    for name, expected in JETLINK.items():
      with self.subTest(name):
        self.assertEqual(plain(getattr(self.jl, name)), expected)

  def test_a_jetlink_shows_the_fork_nothing_else(self):
    # a part the fork could reach would become API without a bump noticing
    self.assertEqual({n for n in dir(self.jl) if not n.startswith('_')}, set(JETLINK))

  def test_the_package_exports_the_contract(self):
    self.assertEqual(sorted(jo.__all__), sorted(EXPORTS))
    for name in EXPORTS:
      self.assertTrue(hasattr(jo, name), name)

  def test_what_the_ui_calls_on_a_status(self):
    self.assertEqual(plain(jo.Status.icon), '(self, started, model_seen, running_big, state)')
    self.assertIsInstance(jo.Status.active_model, property)

  def test_the_entry_points_the_fork_names(self):
    from jetlink.openpilot import owner, provision, warp
    self.assertEqual(plain(owner.main), '(config)')
    for module in (provision, warp):
      self.assertEqual(plain(module.main), '(argv=None)', module)
    self.assertEqual(plain(jo.bind), '(op)')
    self.assertEqual(plain(interface.load_adapter), '(module)')

  def test_bind_points_the_comma_layers_log_at_the_adapters(self):
    # a heavy process's lines belong in the drive's log
    self.assertIs(gadget.log, self.op.log)
    self.assertIs(gadget.root.log, self.op.log)
    self.assertIsInstance(self.jl, jo.Jetlink)
    self.assertIs(self.parts.op, self.op)

  def test_an_adapter_module_makes_its_adapter(self):
    with mock.patch.dict('os.environ', {'JETLINK_FAKE_ROOT': str(self.tmp)}):
      op = interface.load_adapter('tests.openpilot.fakes')
    self.assertIsInstance(op, fakes.FakeOpenpilot)
    self.assertEqual(op.root, self.tmp)


class LenderFailureTest(OpenpilotTest):
  """Only the owner holds ep0. When its lender cannot listen it keeps the
  gadget, retries, and records why: that line is the offroad alert. modeld
  still prepares, so its join picks the link up once the lender listens."""

  def test_the_owners_lender_error_is_the_alert_not_a_no(self):
    built = self.tmp / 'jetlink-gadget'
    built.write_text('ok\n')
    self.op.set_mode('usb')
    self.patch(gadget, 'GADGET_STATUS', built)
    self.patch(gadget, 'LENDER_STATUS', self.tmp / 'jetlink-lender')
    self.patch(gadget, 'link_configured', return_value=True)
    self.patch(self.parts.warps, 'built', return_value=True)
    with mock.patch('jetlink.openpilot.warp.init_device'):
      gadget.note_lender_error('address in use')
      self.assertEqual(self.jl.status().reason, 'the lender could not listen: address in use')
      self.assertFalse(self.jl.status().ready)
      self.assertTrue(self.jl.prepare())
      # cleared once the lender listens again
      gadget.note_lender_error(None)
      self.assertIsNone(self.jl.status().reason)


class LoadTest(OpenpilotTest):
  """modeld's two calls: prepare() before it goes realtime, attach() once the camera is up."""

  def setUp(self):
    super().setUp()
    self.small = SimpleNamespace(name='small', client=None)
    self.init_device = self.patch(sys.modules['jetlink.openpilot.warp'], 'init_device')

  def prepared(self):
    """prepare() down its yes path, with nothing real behind it."""
    with mock.patch.object(self.jl, 'enabled', return_value=True), \
         mock.patch.object(gadget, 'link_configured', return_value=True), \
         mock.patch.object(self.parts.warps, 'built', return_value=True):
      self.assertTrue(self.jl.prepare())
    self.init_device.assert_called_once_with(self.op.log)

  def test_no_warp_is_no_before_the_gpu_comes_up(self):
    with mock.patch.object(self.jl, 'enabled', return_value=True), \
         mock.patch.object(gadget, 'link_configured', return_value=True), \
         mock.patch.object(self.parts.warps, 'built', return_value=False):
      self.assertFalse(self.jl.prepare())
    self.init_device.assert_not_called()
    self.assertTrue(self.op.log.has('no warp built for this camera, staying on the small model'))

  def test_no_usable_gadget_is_no_and_says_why(self):
    with mock.patch.object(self.jl, 'enabled', return_value=True), \
         mock.patch.object(gadget, 'link_configured', return_value=False), \
         mock.patch.object(gadget, 'gadget_error', return_value='no configfs'):
      self.assertFalse(self.jl.prepare())
    self.init_device.assert_not_called()
    self.assertTrue(self.op.log.has('no usable gadget (no configfs), staying on the small model'))

  def test_the_link_off_says_no_before_any_setup(self):
    with mock.patch.object(self.jl, 'enabled', return_value=False), \
         mock.patch.object(gadget, 'link_configured') as link_configured:
      self.assertFalse(self.jl.prepare())
    link_configured.assert_not_called()

  def test_nothing_joins_without_prepare(self):
    # the GPU's thread would start on modeld's realtime core
    with mock.patch.object(joining, 'join') as join:
      self.assertIsNone(self.jl.attach(self.small, 1928, 1208))
    join.assert_not_called()

  def test_a_later_no_takes_the_yes_back(self):
    self.prepared()
    with mock.patch.object(self.jl, 'enabled', return_value=False):
      self.assertFalse(self.jl.prepare())
    with mock.patch.object(joining, 'join') as join:
      self.assertIsNone(self.jl.attach(self.small, 1928, 1208))
    join.assert_not_called()

  def test_prepared_joins_modeld(self):
    self.prepared()
    joined = SimpleNamespace(client=object())
    with mock.patch.object(joining, 'join', return_value=joined) as join:
      model = self.jl.attach(self.small, 1928, 1208)
    join.assert_called_once_with(self.parts, 1928, 1208, self.small)
    self.assertIs(model, joined)

  def test_a_failed_build_drives_the_small_model_and_says_so(self):
    self.prepared()
    with mock.patch.object(joining, 'join', side_effect=RuntimeError('no warp')):
      model = self.jl.attach(self.small, 1928, 1208)
    self.assertEqual(self.op.log.lines('exception'), ["jetlink load failed"])
    self.assertIs(model, self.small)


def spec(model_hw=(128, 256)) -> ModelSpec:
  inputs = {'new_img': (2, 6, *model_hw), 'desire': (8,), 'traffic_convention': (1, 2), 'action_t': (1, 2)}
  return ModelSpec(sha256='a' * 64, nbytes=1, frame_skip=4, input_shapes=inputs, output_shapes={'outputs': (1, 16)},
                   output_slices={'plan': slice(0, 16)}, checkpoint=None)


class TestTheJoinFactory(OpenpilotTest):
  """What attach() builds: the joining model over the small one, with the warp
  loaded and warm before the frame loop exists."""

  def setUp(self):
    super().setUp()
    self.small = SimpleNamespace(name='small')
    p = mock.patch.dict(sys.modules, fakes.fake_tinygrad())
    p.start()
    self.addCleanup(p.stop)
    from jetlink.openpilot import link, warp
    self.present = self.patch(link, 'present_early')
    self.reset = self.patch(warp, 'prepare_reset')
    self.warm = self.patch(warp, 'warm')
    self.loaded = self.patch(self.parts.warps, 'load')
    # the join thread and the watcher are the joining state's; not here
    self.patch(joining.JoiningModelState, '_join_loop', lambda s: None)
    self.patch(joining.JoiningModelState, '_watch_engagement', lambda s: None)

  def join(self):
    s = joining.join(self.parts, 1928, 1208, self.small)
    self.addCleanup(s.close)
    return s

  def test_the_warp_is_sized_from_the_record(self):
    self.parts.spec.store(spec(model_hw=(64, 128)))
    self.join()
    self.loaded.assert_called_once_with(1928, 1208, 256, 128)
    self.warm.assert_called_once_with(self.loaded.return_value, fakes.frame_size(1928, 1208))
    self.reset.assert_called_once_with(self.small)
    self.present.assert_called_once()

  def test_the_early_present_leaves_modelds_realtime_core(self):
    from jetlink.transport.priority import background_thread
    self.join()
    link, background = self.present.call_args.args
    self.assertIs(background, background_thread)

  def test_without_a_record_it_is_this_devices_geometry(self):
    self.join()
    self.loaded.assert_called_once_with(1928, 1208, 512, 256)

  def test_a_warp_that_will_not_load_lets_the_link_go_and_raises(self):
    self.loaded.side_effect = RuntimeError('stale warp')
    with self.assertRaisesRegex(RuntimeError, 'stale warp'):
      joining.join(self.parts, 1928, 1208, self.small)

  def test_it_runs_the_adapters_face_and_reports_through_jetlink(self):
    s = self.join()
    self.assertIs(s._progress, self.parts.progress)
    self.assertEqual(s._engagement, self.op.engagement)
    client = mock.Mock()
    client.t.link_info.return_value = {'kind': 'usb'}
    big = s._build(client, spec())
    self.assertIs(big.face, self.op.face)
    self.assertIs(big.warp, self.loaded.return_value)

  def test_a_server_model_of_another_geometry_is_refused(self):
    s = self.join()
    with self.assertRaisesRegex(RuntimeError, 'no prepared warp'):
      s._build(mock.Mock(), spec(model_hw=(64, 128)))


class TestShutdown(OpenpilotTest):
  """hardwared's call at power-off: bounded, and it never raises."""

  def setUp(self):
    super().setUp()
    self.op.set_mode('usb')

  def test_disabled_costs_one_read_and_nothing_else(self):
    self.op.set_mode('off')
    with mock.patch.object(self.jl, '_request_shutdown') as request:
      self.jl.shutdown('car battery')
    request.assert_not_called()

  def test_beside_a_chestnut_nothing_is_asked(self):
    self.op.chestnut = True
    with mock.patch.object(self.jl, '_request_shutdown') as request:
      self.jl.shutdown('car battery')
    request.assert_not_called()

  def test_the_request_is_forwarded_with_the_bound(self):
    with mock.patch.object(self.jl, '_request_shutdown') as request:
      self.jl.shutdown('car battery', timeout=3.0)
    request.assert_called_once_with('car battery', 3.0)

  def test_a_request_that_hangs_cannot_hold_hardwared(self):
    release = threading.Event()
    self.addCleanup(release.set)
    with mock.patch.object(self.jl, '_request_shutdown', side_effect=lambda *a: release.wait(30)):
      t0 = time.monotonic()
      self.jl.shutdown('car battery', timeout=0.2)
      self.assertLess(time.monotonic() - t0, 2.0)
    self.assertEqual(self.op.log.lines('warning'),
                     ['jetlink: shutdown request still pending after 0 s, going on without it'])

  def test_a_request_that_raises_is_logged_not_propagated(self):
    with mock.patch.object(self.jl, '_request_shutdown', side_effect=RuntimeError('no')):
      self.jl.shutdown('car battery', timeout=1.0)
    self.assertEqual(self.op.log.lines('exception'), ['jetlink: shutdown request failed'])


class ShuttingTheJetsonDown(OpenpilotTest):
  """hardwared hands the request to the owner only when a Jetson is there to take it."""

  def shutdown(self, mode='usb', present=True, requested=True, taken=True):
    self.op.set_mode(mode)
    with mock.patch.object(status.Presence, 'present', return_value=present), \
         mock.patch.object(gadget, 'request_shutdown', return_value=requested) as request, \
         mock.patch.object(self.jl, '_await_shutdown', return_value=taken) as wait:
      self.jl._request_shutdown('car battery', 3.0)
    return request, wait

  def test_the_link_off_asks_nothing(self):
    request, wait = self.shutdown(mode='off')
    request.assert_not_called()
    wait.assert_not_called()

  def test_no_jetson_there_asks_nothing(self):
    request, wait = self.shutdown(present=False)
    request.assert_not_called()
    wait.assert_not_called()

  def test_a_jetson_there_is_asked_and_waited_for(self):
    request, wait = self.shutdown()
    request.assert_called_once_with('car battery')
    wait.assert_called_once_with(3.0)
    self.assertTrue(self.op.log.has('shutdown request handed to the jetson'))

  def test_a_request_that_could_not_be_written_is_not_waited_on(self):
    request, wait = self.shutdown(requested=False)
    request.assert_called_once_with('car battery')
    wait.assert_not_called()

  def test_a_jetson_that_just_left_is_not_waited_for(self):
    # hardwared's own readers keep this process's presence fresh; a host seen
    # moments before the power-off would have the run wait 20 s for nobody
    self.op.set_mode('usb')
    with mock.patch.object(gadget, 'host_attached', return_value=True):
      self.assertTrue(self.parts.presence.present())
    with mock.patch.object(gadget, 'host_attached', return_value=False), \
         mock.patch.object(gadget, 'dormant', return_value=False), \
         mock.patch.object(gadget, 'request_shutdown') as request:
      self.assertTrue(self.parts.presence.present(), 'the hold this test is about')
      self.jl._request_shutdown('car battery', 3.0)
    request.assert_not_called()

  def test_nobody_taking_it_is_logged(self):
    self.shutdown(taken=False)
    self.assertTrue(self.op.log.has('nobody took the shutdown request within 3 s'))


class TestAwaitingTheOwner(unittest.TestCase):
  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    p = mock.patch.object(gadget, 'SHUTDOWN_REQUEST', self.tmp / 'shutdown')
    p.start()
    self.addCleanup(p.stop)

  def test_it_gives_up_and_cleans_up(self):
    gadget.request_shutdown('car battery')
    self.assertFalse(jo.Jetlink._await_shutdown(0.3))
    self.assertIsNone(gadget.pending_shutdown())

  def test_it_returns_when_taken(self):
    gadget.request_shutdown('car battery')
    gadget.finish_shutdown()
    self.assertTrue(jo.Jetlink._await_shutdown(0.3))

  def test_the_request_carries_the_reason(self):
    gadget.request_shutdown('car battery')
    self.assertEqual(json.loads(gadget.SHUTDOWN_REQUEST.read_text()), {'reason': 'car battery'})


class TestExtendsCatalog(OpenpilotTest):
  """Hardware, not the link setting: the model manager drops a pick its catalog
  does not list, so a catalog that followed the setting lost one on a boot with
  it off."""

  def extends(self, chestnut=False, mode='off'):
    self.op.chestnut = chestnut
    self.op.set_mode(mode)
    self.parts._chestnut = None
    return self.jl.should_extend_catalog()

  def test_without_a_chestnut_it_is_extended_whatever_the_setting(self):
    self.assertTrue(self.extends(mode='off'))
    self.assertTrue(self.extends(mode='usb'))

  def test_a_chestnut_leaves_it_as_fetched(self):
    self.assertFalse(self.extends(chestnut=True, mode='usb'))
    self.assertFalse(self.extends(chestnut=True, mode='off'))


if __name__ == '__main__':
  unittest.main()
