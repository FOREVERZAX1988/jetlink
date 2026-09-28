"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's gadget: openpilot's params read as files, what carries the link,
and the gadget built and brought up through the root script.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from jetlink.comma import gadget, root

REPO = Path(__file__).resolve().parents[1]
# what the gadget owner must never end up importing. swaglog pulls numpy, capnp
# and zmq in to publish a log line and costs 28 MB; params imports swaglog.
# Measured on the comma: the owner plus the transport is 10.4 MB, against
# 47.5 MB for the daemon that imported the world
HEAVY = ('numpy', 'capnp', 'zmq', 'cereal', 'openpilot')


class TestNothingHeavyIsReachable(unittest.TestCase):
  """The owner is only small while this holds, so it is a test and not a note."""

  def imported_by(self, module: str) -> set[str]:
    """Top-level packages a fresh interpreter has after importing `module`."""
    roots = 'sorted({m.split(".")[0] for m in sys.modules})'
    code = f'import sys, json; __import__("{module}"); print(json.dumps({roots}))'
    env = {**os.environ, 'PYTHONPATH': str(REPO)}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env=env, cwd=str(REPO), timeout=120)
    self.assertEqual(out.returncode, 0, out.stderr)
    return set(json.loads(out.stdout))

  def test_the_gadget_core_stays_out_of_the_heavy_half(self):
    found = self.imported_by('jetlink.comma.gadget')
    self.assertEqual(sorted(found & set(HEAVY)), [],
                     'the gadget owner has to stay small; see the module docstring')

  def test_the_owner_itself_stays_out_of_the_heavy_half(self):
    found = self.imported_by('jetlink.comma.owner')
    self.assertEqual(sorted(found & set(HEAVY)), [],
                     'everything the owner imports runs for the whole drive')

  def test_the_transport_the_owner_opens_is_light_too(self):
    found = self.imported_by('jetlink.transport.ffs')
    self.assertEqual(sorted(found & set(HEAVY)), [])


class TestParamsOffTheFilesystem(unittest.TestCase):
  """params.cc writes a value to a temp file, fsyncs it, renames it over the key
  and fsyncs the directory, so a plain read never sees a torn value."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    (self.tmp / 'd').mkdir()
    self.enterContext(unittest.mock.patch.dict(os.environ, {'PARAMS_ROOT': str(self.tmp)}))
    os.environ.pop('OPENPILOT_PREFIX', None)

  def write(self, key: str, value: bytes) -> None:
    (self.tmp / 'd' / key).write_bytes(value)

  def test_the_path_follows_the_prefix(self):
    self.assertEqual(gadget.params_dir(), self.tmp / 'd')
    with unittest.mock.patch.dict(os.environ, {'OPENPILOT_PREFIX': 'abc123'}):
      self.assertEqual(gadget.params_dir(), self.tmp / 'abc123')

  def test_a_missing_param_is_not_a_false(self):
    # None and False are different answers: offroad treats an unwritten param
    # as parked, and enabled treats it as off
    self.assertIsNone(gadget.param_bool(gadget.P_OFFROAD))
    self.assertFalse(gadget.enabled())
    self.assertTrue(gadget.offroad())

  def test_a_bool_is_true_and_nothing_else(self):
    for raw, expected in ((b'1', True), (b'0', False), (b'', False), (b'true', True)):
      self.write(gadget.P_OFFROAD, raw)
      self.assertIs(gadget.param_bool(gadget.P_OFFROAD), expected, raw)
      self.assertIs(gadget.offroad(), expected, raw)

  def test_an_unreadable_store_is_not_an_error(self):
    with unittest.mock.patch.dict(os.environ, {'PARAMS_ROOT': '/nonexistent'}):
      self.assertIsNone(gadget.raw_param(gadget.P_LINK))
      self.assertFalse(gadget.enabled())


class TestLinkKind(unittest.TestCase):
  """What carries the link: the gadget the owner built, 'cable' for iOS and
  'usb' otherwise, with the setting standing in until the owner has said."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    for name, value in (('LINK', self.tmp / 'link'), ('UDC_PATH', self.tmp / 'udc'),
                        ('NET_STATUS', self.tmp / 'net'),
                        ('ios', unittest.mock.Mock(return_value=False))):
      p = unittest.mock.patch.object(gadget, name, value)
      self.addCleanup(p.stop)
      p.start()

  def test_nothing_recorded_is_usb(self):
    self.assertEqual(gadget.link_kind(), 'usb')
    self.assertIsNone(gadget.link_peer())

  def test_ios_is_the_cable_before_any_dial(self):
    gadget.ios.return_value = True
    self.assertEqual(gadget.link_kind(), 'cable')
    self.assertIsNone(gadget.link_peer())

  def test_the_owners_record_decides_over_the_setting(self):
    # a setting moved while somebody borrowed waits for the car to park; until
    # the owner rebuilds, the gadget is what it built
    gadget.ios.return_value = True
    gadget.note_link('usb')
    self.assertEqual(gadget.link_kind(), 'usb')
    gadget.ios.return_value = False
    gadget.note_link('cable', '192.168.60.3')
    self.assertEqual(gadget.link_kind(), 'cable')
    gadget.clear_link()
    self.assertEqual(gadget.link_kind(), 'usb', 'no owner yet: the setting')

  def test_a_dial_is_recorded_with_the_phone_and_cleared(self):
    gadget.note_link('cable', '192.168.60.3')
    self.assertEqual(gadget.link_peer(), '192.168.60.3')
    gadget.clear_link()
    self.assertIsNone(gadget.link_peer())
    gadget.clear_link()   # twice is not an error

  def test_on_the_cable_there_is_no_host_to_wait_for(self):
    # the connect already reached the phone; the UDC is configured by it, and
    # it is the dial that proved it
    with unittest.mock.patch.object(gadget, 'udc_state', return_value='powered'), \
         unittest.mock.patch.object(gadget.time, 'sleep', side_effect=AssertionError('waited')):
      gadget.ios.return_value = True
      self.assertTrue(gadget.wait_for_host(5.0, report=lambda: self.fail('reported a wait')))
      gadget.ios.return_value = False
      self.assertFalse(gadget.wait_for_host(0.0))

  def test_the_cable_needs_the_gadget_too(self):
    # a phone on the cable is on the gadget's own network interface
    with unittest.mock.patch.object(gadget, 'FFS_MOUNT', self.tmp / 'ffs'), \
         unittest.mock.patch.object(gadget, 'GADGET_STATUS', self.tmp / 'status'):
      gadget.note_link('cable', '192.168.60.3')
      self.assertFalse(gadget.link_configured())
      (self.tmp / 'ffs').mkdir()
      (self.tmp / 'ffs' / 'ep0').touch()
      self.assertTrue(gadget.link_configured())

  def test_the_bus_speed_is_read_off_the_bound_udc(self):
    with unittest.mock.patch.object(gadget, 'bound_udc', return_value=None):
      self.assertIsNone(gadget.usb_speed())
    with unittest.mock.patch.object(gadget, 'bound_udc', return_value='a600000.dwc3'):
      self.assertIsNone(gadget.usb_speed())
      (gadget.UDC_PATH / 'a600000.dwc3').mkdir(parents=True)
      (gadget.UDC_PATH / 'a600000.dwc3' / 'current_speed').write_text('super-speed\n')
      self.assertEqual(gadget.usb_speed(), 'super-speed')

  def test_the_network_status_is_the_scripts(self):
    self.assertIsNone(gadget.net_status())
    gadget.NET_STATUS.write_text('ok 192.168.60.1 usb1\n')
    self.assertEqual(gadget.net_status(), 'ok 192.168.60.1 usb1')


class TestTheTwoGadgets(unittest.TestCase):
  """The setting picks the gadget: the plain one for USB, the composite one
  with a network interface for iOS."""

  def test_setup_passes_the_setting(self):
    with unittest.mock.patch.object(root, 'run', return_value=True) as run, \
         unittest.mock.patch.object(gadget, 'link_configured', return_value=True):
      self.assertTrue(gadget.setup_gadget(True))
      run.assert_called_with('gadget', '--ios', timeout=root.GADGET_TIMEOUT)
      self.assertTrue(gadget.setup_gadget(False))
      run.assert_called_with('gadget', timeout=root.GADGET_TIMEOUT)

  def test_the_built_gadget_is_read_from_its_config(self):
    tmp = Path(tempfile.mkdtemp())
    config = tmp / 'configs' / 'c.1'
    config.mkdir(parents=True)
    (config / 'ffs.jetlink').touch()
    with unittest.mock.patch.object(gadget, 'GADGET_PATH', tmp):
      self.assertFalse(gadget.built_for_ios())
      (config / 'ncm.usb0').symlink_to(tmp / 'functions' / 'ncm.usb0')
      self.assertTrue(gadget.built_for_ios())


class TestGadgetSetup(unittest.TestCase):
  """The owner creates the gadget, and brings its network up after a bind,
  through the root script."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.script = self.tmp / 'jetlink-root.sh'
    self.script.write_text('#!/bin/sh\n')
    for module, name, value in ((root, 'SCRIPT', self.script), (gadget, 'AGNOS', True)):
      p = unittest.mock.patch.object(module, name, value)
      self.addCleanup(p.stop)
      p.start()

  def test_only_agnos_with_the_script_can_set_one_up(self):
    self.assertTrue(gadget.can_setup_gadget())
    with unittest.mock.patch.object(gadget, 'AGNOS', False):
      self.assertFalse(gadget.can_setup_gadget())
    self.script.unlink()
    self.assertFalse(gadget.can_setup_gadget())

  def test_the_scripts_checkout_is_the_one_the_owner_runs(self):
    self.assertEqual(root.SCRIPT, self.script)
    self.assertTrue((REPO / 'scripts' / 'comma' / 'jetlink-root.sh').is_file())

  def test_a_failed_setup_is_a_false_not_a_raise(self):
    # the script has already written the reason to the status file
    with unittest.mock.patch.object(root, 'run', return_value=False), \
         unittest.mock.patch.object(gadget, 'link_configured', side_effect=AssertionError('looked')):
      self.assertFalse(gadget.setup_gadget(False))

  def test_the_network_is_brought_up_by_the_same_script_and_judged_by_its_status(self):
    # the netdev exists only once the UDC is bound, so this runs after the
    # owner's bind rather than with the gadget
    status = self.tmp / 'net'
    with unittest.mock.patch.object(root, 'run', return_value=True) as run, \
         unittest.mock.patch.object(gadget, 'NET_STATUS', status):
      self.assertFalse(gadget.net_up())          # the script wrote nothing
      run.assert_called_once_with('net', timeout=root.GADGET_TIMEOUT)
      status.write_text('error: no netdev yet; it appears when the owner binds the UDC (then run net)\n')
      self.assertFalse(gadget.net_up())
      status.write_text('ok 192.168.60.1 usb1\n')
      self.assertTrue(gadget.net_up())
    with unittest.mock.patch.object(root, 'run', return_value=False), \
         unittest.mock.patch.object(gadget, 'NET_STATUS', status):
      self.assertFalse(gadget.net_up())
    with unittest.mock.patch.object(gadget, 'AGNOS', False), \
         unittest.mock.patch.object(root, 'run') as run:
      self.assertFalse(gadget.net_up())
      run.assert_not_called()


class TestTheLogger(unittest.TestCase):
  def test_the_root_scripts_failures_go_where_the_gadgets_lines_do(self):
    # the owner logs to its own file and the fork's processes to cloudlog;
    # a sudo that failed must land in the same place
    for module in (gadget, root):
      p = unittest.mock.patch.object(module, 'log', module.log)
      self.addCleanup(p.stop)
      p.start()
    logger = unittest.mock.Mock()
    gadget.set_logger(logger)
    self.assertIs(gadget.log, logger)
    self.assertIs(root.log, logger)


class TestLinkMode(unittest.TestCase):
  """Accelerator Link is one param, JetlinkLink: 0 off, 1 USB, 2 iOS."""

  def setUp(self):
    self.dir = Path(tempfile.mkdtemp())
    p = unittest.mock.patch.object(gadget, 'params_dir', return_value=self.dir)
    self.addCleanup(p.stop)
    p.start()

  def write(self, key: str, value: str) -> None:
    (self.dir / key).write_text(value)

  def test_the_three_modes(self):
    for raw, mode in (('0', 'off'), ('1', 'usb'), ('2', 'ios'), ('7', 'off'), ('x', 'off'), ('', 'off')):
      self.write('JetlinkLink', raw)
      self.assertEqual(gadget.link_mode(), mode, raw)
    self.write('JetlinkLink', '2')
    self.assertTrue(gadget.enabled() and gadget.ios())

  def test_unset_is_off_whatever_the_old_switch_says(self):
    # the fork's params migration carries JetlinkEnabled over, not this
    self.assertEqual(gadget.link_mode(), 'off')
    self.write('JetlinkEnabled', '1')
    self.assertEqual(gadget.link_mode(), 'off')
    self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ['JetlinkEnabled'], 'wrote a param')
