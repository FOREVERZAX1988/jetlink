"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's one root script and its wrapper. The wrapper against a fake
subprocess.run; the script's port and vm subcommands against a fake debugfs
and /proc/sys, run without sudo through its override variables. gadget, net,
check and teardown need a real configfs and are bench-only.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jetlink.comma import root

REPO = Path(__file__).resolve().parents[1]


class FakeRun:
  def __init__(self, returncode=0, stderr='', raises=None):
    self.returncode, self.stderr, self.raises = returncode, stderr, raises
    self.calls = []

  def __call__(self, argv, **kwargs):
    self.calls.append((argv, kwargs))
    if self.raises is not None:
      raise self.raises
    return subprocess.CompletedProcess(argv, self.returncode, None, self.stderr)


@pytest.fixture
def fake_run(monkeypatch):
  def install(**kwargs):
    fake = FakeRun(**kwargs)
    monkeypatch.setattr(root.subprocess, 'run', fake)
    return fake
  return install


# -- the wrapper ------------------------------------------------------------

def test_the_script_is_the_checkouts():
  assert root.SCRIPT == REPO / 'scripts' / 'comma' / 'jetlink-root.sh'
  assert root.SCRIPT.is_file()


def test_the_timeouts_are_the_forks():
  # gadget.GADGET_SETUP_TIMEOUT, usbport.SCRIPT_TIMEOUT and vmtune's sysctl call
  assert (root.GADGET_TIMEOUT, root.PORT_TIMEOUT, root.VM_TIMEOUT) == (30.0, 2.0, 5.0)


@pytest.mark.parametrize('args, timeout', [
  (('gadget',), root.GADGET_TIMEOUT),
  (('gadget', '--ios'), root.GADGET_TIMEOUT),
  (('net',), root.GADGET_TIMEOUT),
  (('teardown',), root.GADGET_TIMEOUT),
  (('port', 'hold'), root.PORT_TIMEOUT),
  (('vm', 'apply'), root.VM_TIMEOUT),
])
def test_run_is_sudo_bash_script_args(fake_run, args, timeout):
  fake = fake_run()
  assert root.run(*args, timeout=timeout) is True
  [(argv, kwargs)] = fake.calls
  assert argv == ['sudo', '-n', 'bash', str(root.SCRIPT), *args]
  assert kwargs['timeout'] == timeout
  assert kwargs['stdout'] == subprocess.DEVNULL
  assert kwargs['stderr'] == subprocess.PIPE


def test_a_failure_logs_the_last_stderr_line_once(fake_run, caplog):
  fake_run(returncode=3, stderr='a warning first\njetlink: no USB-C role lever; the port stays as it is\n')
  with caplog.at_level(logging.INFO, logger='jetlink.comma'):
    assert root.run('port', 'off', timeout=root.PORT_TIMEOUT) is False
  [record] = caplog.records
  assert record.levelno == logging.ERROR
  assert 'port off' in record.getMessage()
  assert 'exit 3' in record.getMessage()
  assert 'no USB-C role lever' in record.getMessage()
  assert 'a warning first' not in record.getMessage()


def test_a_failure_with_nothing_on_stderr_still_logs(fake_run, caplog):
  fake_run(returncode=1)
  with caplog.at_level(logging.INFO, logger='jetlink.comma'):
    assert root.run('vm', 'restore', timeout=root.VM_TIMEOUT) is False
  [record] = caplog.records
  assert 'no output' in record.getMessage()


@pytest.mark.parametrize('error, words', [
  (subprocess.TimeoutExpired(['sudo'], 2.0), 'timed out'),
  (FileNotFoundError(2, 'No such file or directory', 'sudo'), 'did not run'),
  (PermissionError(13, 'Permission denied'), 'did not run'),
])
def test_run_never_raises(fake_run, caplog, error, words):
  fake_run(raises=error)
  with caplog.at_level(logging.INFO, logger='jetlink.comma'):
    assert root.run('port', 'hold', timeout=root.PORT_TIMEOUT) is False
  [record] = caplog.records
  assert words in record.getMessage()


def test_the_wrapper_imports_nothing_heavy():
  # it lives in the comma's resident owner
  code = ("import sys, jetlink.comma.root; "
          "heavy = [m for m in ('numpy', 'openpilot', 'cereal') if m in sys.modules]; "
          "assert not heavy, heavy")
  subprocess.run([sys.executable, '-c', code], check=True, cwd=REPO,
                 env={**os.environ, 'PYTHONPATH': str(REPO)})


# -- the script -------------------------------------------------------------

def test_the_script_parses():
  subprocess.run(['bash', '-n', str(root.SCRIPT)], check=True)


@pytest.mark.skipif(shutil.which('shellcheck') is None, reason='needs shellcheck')
def test_the_script_passes_shellcheck():
  # at the default severity, as the CI job runs it
  subprocess.run(['shellcheck', str(root.SCRIPT)], check=True)


def script(tmp_path, *args):
  env = {
    **os.environ,
    'JETLINK_PROC_SYS': str(tmp_path / 'sys'),
    'JETLINK_SYSCTL_PREV': str(tmp_path / 'sysctl-prev'),
    'JETLINK_POWER_ROLE_VOTER': str(tmp_path / 'voter'),
  }
  return subprocess.run(['bash', str(root.SCRIPT), *args], env=env, capture_output=True, text=True)


# stock AGNOS: the dirty limits in ratio mode, so both *_bytes keys read 0
STOCK = {
  'vm.dirty_bytes': '0',
  'vm.dirty_background_bytes': '0',
  'vm.min_free_kbytes': '22528',
  'vm.dirty_ratio': '20',
  'vm.dirty_background_ratio': '10',
}
TUNED = {'vm.dirty_bytes': '16777216', 'vm.dirty_background_bytes': '8388608', 'vm.min_free_kbytes': '131072'}


def proc_sys(tmp_path, values):
  for key, value in values.items():
    f = tmp_path / 'sys' / key.replace('.', '/')
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(value + '\n')


def read_sys(tmp_path, key):
  return (tmp_path / 'sys' / key.replace('.', '/')).read_text().strip()


def test_vm_apply_records_the_stock_values_once_and_applies_ours(tmp_path):
  proc_sys(tmp_path, STOCK)
  assert script(tmp_path, 'vm', 'apply').returncode == 0
  # JSON, as the fork's vmtune.py wrote it, so either can read the other's
  assert json.loads((tmp_path / 'sysctl-prev').read_text()) == STOCK
  assert {k: read_sys(tmp_path, k) for k in TUNED} == TUNED
  # a second apply keeps the first record, not our own values
  assert script(tmp_path, 'vm', 'apply').returncode == 0
  assert json.loads((tmp_path / 'sysctl-prev').read_text()) == STOCK


def test_vm_restore_goes_back_to_ratio_mode_and_drops_the_record(tmp_path):
  proc_sys(tmp_path, STOCK)
  assert script(tmp_path, 'vm', 'apply').returncode == 0
  assert script(tmp_path, 'vm', 'restore').returncode == 0
  # the kernel drops a 0 written to a *_bytes key, so the ratios are written
  # instead; the fake /proc cannot zero the bytes keys as the kernel does
  assert read_sys(tmp_path, 'vm.dirty_ratio') == '20'
  assert read_sys(tmp_path, 'vm.dirty_background_ratio') == '10'
  assert read_sys(tmp_path, 'vm.min_free_kbytes') == '22528'
  assert not (tmp_path / 'sysctl-prev').exists()


def test_vm_restore_writes_bytes_that_were_set(tmp_path):
  proc_sys(tmp_path, {**TUNED, 'vm.dirty_ratio': '20', 'vm.dirty_background_ratio': '10'})
  (tmp_path / 'sysctl-prev').write_text(json.dumps({**STOCK, 'vm.dirty_bytes': '33554432'}))
  assert script(tmp_path, 'vm', 'restore').returncode == 0
  assert read_sys(tmp_path, 'vm.dirty_bytes') == '33554432'


def test_vm_restore_without_a_record_changes_nothing(tmp_path):
  proc_sys(tmp_path, TUNED)
  assert script(tmp_path, 'vm', 'restore').returncode == 0
  assert {k: read_sys(tmp_path, k) for k in TUNED} == TUNED


def test_an_unreadable_record_is_dropped_and_the_values_left(tmp_path):
  proc_sys(tmp_path, TUNED)
  (tmp_path / 'sysctl-prev').write_text('garbage\n')
  result = script(tmp_path, 'vm', 'restore')
  assert result.returncode == 1
  assert 'unreadable sysctl record' in result.stderr
  assert {k: read_sys(tmp_path, k) for k in TUNED} == TUNED
  assert not (tmp_path / 'sysctl-prev').exists()


def test_vm_apply_fails_when_a_key_will_not_take(tmp_path):
  proc_sys(tmp_path, STOCK)
  (tmp_path / 'sys' / 'vm' / 'min_free_kbytes').chmod(0o444)
  if os.access(tmp_path / 'sys' / 'vm' / 'min_free_kbytes', os.W_OK):
    pytest.skip('running as root')
  result = script(tmp_path, 'vm', 'apply')
  assert result.returncode == 1
  assert result.stderr.strip().splitlines()[-1] == 'jetlink: could not set vm.min_free_kbytes=131072'
  # the others still took
  assert read_sys(tmp_path, 'vm.dirty_bytes') == TUNED['vm.dirty_bytes']


def test_port_hold_forces_the_voter_and_off_lets_it_go(tmp_path):
  voter = tmp_path / 'voter'
  voter.mkdir()
  assert script(tmp_path, 'port', 'hold').returncode == 0
  assert (voter / 'force_val').read_text().strip() == '1'
  assert (voter / 'force_active').read_text().strip() == '1'
  assert script(tmp_path, 'port', 'off').returncode == 0
  assert (voter / 'force_active').read_text().strip() == '0'
  assert (voter / 'force_val').read_text().strip() == '0'


def test_port_without_the_lever_exits_3(tmp_path):
  result = script(tmp_path, 'port', 'hold')
  assert result.returncode == 3
  assert 'no USB-C role lever' in result.stderr


@pytest.mark.parametrize('args', [(), ('setup',), ('--ios',), ('gadget', '--net'), ('port',), ('port', 'on'), ('vm',), ('vm', 'undo')])
def test_anything_else_is_usage(tmp_path, args):
  result = script(tmp_path, *args)
  assert result.returncode == 2
  assert 'usage:' in result.stderr
