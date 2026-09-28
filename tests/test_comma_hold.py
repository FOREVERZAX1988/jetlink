"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

scripts/comma/jetlink_hold.py, which keeps the Jetson awake by holding a loan
of the gadget. Against a real Lender, as the owner runs it, on a socket of the
test's own.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import threading
from pathlib import Path
from unittest import mock

from jetlink.comma import lending
from tests.test_comma_lending import LendingTest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def script(path: Path):
  spec = importlib.util.spec_from_file_location(path.stem, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


hold = script(SCRIPTS / 'comma' / 'jetlink_hold.py')


class HoldTest(LendingTest):
  def setUp(self):
    super().setUp()
    for mod in (lending, hold):
      p = mock.patch.object(mod, 'POLL', 0.02)
      self.addCleanup(p.stop)
      p.start()
    self.said: list[str] = []
    p = mock.patch.object(hold, 'say', side_effect=self.said.append)
    self.addCleanup(p.stop)
    p.start()

  def run_hold(self, *argv: str) -> tuple[threading.Thread, list[int]]:
    rc: list[int] = []
    t = threading.Thread(target=lambda: rc.append(hold.main(list(argv), path=self.path)), daemon=True)
    t.start()
    self.addCleanup(t.join, 3.0)
    return t, rc


class Holding(HoldTest):
  def test_a_timed_hold_borrows_the_gadget_and_gives_it_back(self):
    lender = self.lender()
    t, rc = self.run_hold('-t', '0.5')
    assert self.until(lambda: lender.lent and lender.borrower == 'hold'), 'the hold never borrowed'
    t.join(3.0)
    assert rc == [0]
    assert self.said == ['holding the gadget: udc udc0, mount ' + str(lending.gadget.FFS_MOUNT), 'released']
    assert self.until(lambda: not lender.lent), 'the loan was not given back'

  def test_a_command_is_held_for_and_its_status_passed_on(self):
    lender = self.lender()
    seen = []
    t, rc = self.run_hold('--', sys.executable, '-c', 'import time, sys; time.sleep(0.3); sys.exit(3)')
    assert self.until(lambda: lender.lent), 'the hold never borrowed'
    seen.append(lender.borrower)
    t.join(3.0)
    assert rc == [3] and seen == ['hold']
    assert self.said[-1] == 'released'
    assert self.until(lambda: not lender.lent), 'the loan was not given back'

  def test_an_owner_that_goes_away_ends_the_hold(self):
    lender = self.lender()
    t, rc = self.run_hold()
    assert self.until(lambda: lender.lent), 'the hold never borrowed'
    lender.stop()
    t.join(3.0)
    assert rc == [1]
    assert 'the owner went away' in self.said[-2]


class NoLoan(HoldTest):
  def test_no_owner_listening_says_so(self):
    t, rc = self.run_hold('-t', '5')
    t.join(3.0)
    assert rc == [1]
    assert self.said and 'no gadget owner is listening' in self.said[0]
    assert 'released' not in self.said

  def test_an_owner_that_never_lends_says_so(self):
    self.free = False   # the owner never gets off the endpoints
    lender = self.lender()
    t, rc = self.run_hold('-t', '5', '--timeout', '0.3')
    t.join(3.0)
    assert rc == [1]
    assert 'no loan from the owner within 0.3 s' in self.said[0]
    assert self.until(lambda: not lender.lent)

  def test_a_bare_separator_is_refused(self):
    with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
      hold.main(['--'], path=self.path)
    assert refused.exception.code == 2

