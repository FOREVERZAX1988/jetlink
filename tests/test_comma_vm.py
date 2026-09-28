"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The VM tuning the gadget needs, and how it is put back.

These numbers were measured: capping dirty memory and holding a free-memory
floor took the worst FunctionFS frame from 244 ms to 72 ms over 20 minutes.
The values and the record of the stock ones are jetlink-root.sh vm's, and
test_comma_root.py covers the awkward parts there: stock AGNOS runs the dirty
limits in ratio mode, and the kernel silently drops a 0 written back to a
*_bytes key. This is when the owner applies and restores, against the real
script on a fake /proc/sys.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget, owner, root, vm


class TestVm(unittest.TestCase):
  def test_apply_and_restore_are_the_root_scripts(self):
    with mock.patch.object(gadget, 'AGNOS', True), mock.patch.object(root, 'run', return_value=True) as run:
      self.assertTrue(vm.apply())
      run.assert_called_with('vm', 'apply', timeout=root.VM_TIMEOUT)
      self.assertTrue(vm.restore())
      run.assert_called_with('vm', 'restore', timeout=root.VM_TIMEOUT)

  def test_off_agnos_nothing_runs(self):
    with mock.patch.object(gadget, 'AGNOS', False), mock.patch.object(root, 'run') as run:
      self.assertFalse(vm.apply())
      self.assertFalse(vm.restore())
    run.assert_not_called()


class TestVmTuning(unittest.TestCase):
  """A device with the link off runs stock values, one that turns it off gets
  them back, and a plain exit keeps them for the drive that follows."""

  # Stock AGNOS: ratio mode, so both *_bytes read 0 and the ratios carry the limit.
  STOCK = {'vm.dirty_bytes': '0', 'vm.dirty_background_bytes': '0', 'vm.min_free_kbytes': '7274',
           'vm.dirty_ratio': '20', 'vm.dirty_background_ratio': '5'}
  TUNED = {'vm.dirty_bytes': '16777216', 'vm.dirty_background_bytes': '8388608', 'vm.min_free_kbytes': '131072'}

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.proc = self.tmp / 'proc'
    for key, value in self.STOCK.items():
      path = self.proc / key.replace('.', '/')
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(value + '\n')
    self.record = self.tmp / 'prev'
    self.calls: list[tuple[str, ...]] = []
    for p in (mock.patch.object(gadget, 'AGNOS', True),
              mock.patch.object(root, 'run', self.run_script),
              mock.patch.object(gadget, 'DORMANT', self.tmp / 'dormant'),
              mock.patch.object(gadget, 'link_endpoint', mock.Mock(return_value=None)),
              mock.patch.object(owner.port, 'Port', mock.Mock())):
      self.addCleanup(p.stop)
      p.start()

  def run_script(self, *args: str, timeout: float) -> bool:
    """root.run, without sudo, on the fake /proc/sys."""
    self.calls.append(args)
    env = {**os.environ, 'JETLINK_PROC_SYS': str(self.proc), 'JETLINK_SYSCTL_PREV': str(self.record)}
    return subprocess.run(['bash', str(root.SCRIPT), *args], env=env, capture_output=True, timeout=timeout).returncode == 0

  def read(self, key: str) -> str:
    return (self.proc / key.replace('.', '/')).read_text().strip()

  def owner(self, enabled=True):
    """An owner with the gadget already presented and nothing else to do."""
    o = owner.Owner()
    o.lender = mock.Mock(lent=False, listening=True)
    o.transport = mock.Mock(lendable=True)
    o.seen = dict.fromkeys(owner.WATCHED, 0)
    o.had_host = True
    for name in ('open_link', 'spawn_worker', 'settle', 'go_dormant'):
      p = mock.patch.object(o, name, mock.Mock(return_value=True))
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(gadget, 'enabled', return_value=enabled)
    self.addCleanup(p.stop)
    p.start()
    return o

  def test_applied_on_start_and_kept_on_exit(self):
    o = self.owner()
    o.step()
    assert self.calls == [('vm', 'apply')], "an exit is the ignition handoff; restoring here strips the drive of them"
    assert {k: self.read(k) for k in self.TUNED} == self.TUNED
    assert json.loads(self.record.read_text()) == self.STOCK, "the record is what a later disable restores to"

  def test_the_next_start_reapplies_without_touching_the_record(self):
    self.owner().step()
    self.owner().step()
    assert self.calls == [('vm', 'apply'), ('vm', 'apply')]
    assert json.loads(self.record.read_text()) == self.STOCK

  def test_applied_once_a_run(self):
    o = self.owner()
    o.step()
    o.step()
    assert self.calls == [('vm', 'apply')]

  def test_nothing_happens_when_disabled(self):
    self.owner(enabled=False).step()
    assert self.calls == []
    assert not self.record.exists()

  def test_disabling_mid_run_restores(self):
    o = self.owner()
    o.step()
    assert o.vm_tuned
    with mock.patch.object(gadget, 'enabled', return_value=False):
      o.step()
    assert not o.vm_tuned
    assert self.calls == [('vm', 'apply'), ('vm', 'restore')]
    assert not self.record.exists()
    assert self.read('vm.min_free_kbytes') == self.STOCK['vm.min_free_kbytes']
    # back to ratio mode through the ratio keys; the fake /proc cannot zero
    # the bytes keys as the kernel does
    assert self.read('vm.dirty_ratio') == '20' and self.read('vm.dirty_background_ratio') == '5'
