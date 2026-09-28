"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Handing the endpoints over without handing the gadget over.

jetlinkd holds ep0 and the UDC bind for as long as the link is enabled, so a
drive starting or ending is no longer an unplug the Jetson has to recover from.
What still changes hands is the right to read the endpoint files, and this is
the handshake for it.
"""

import shutil
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import lending


class LendingTest(unittest.TestCase):
  def setUp(self):
    # a short path: an AF_UNIX address is about 100 bytes and a pytest tmp_path
    # spends most of that before the filename
    self.dir = Path(tempfile.mkdtemp(dir='/tmp'))
    self.addCleanup(shutil.rmtree, self.dir, True)
    self.path = self.dir / 's'
    self.free = True          # the daemon has nothing open on the endpoints
    self.bounced = 0
    p = mock.patch.object(lending, 'RETRY', 0.01)
    self.addCleanup(p.stop)
    p.start()
    p = mock.patch.object(lending.gadget, 'bound_udc', side_effect=lambda: self.udc)
    self.addCleanup(p.stop)
    self.udc = 'udc0'
    p.start()
    # a loopback stand-in for usb0
    p = mock.patch.object(lending.gadget, 'CABLE_ADDR', ('127.0.0.1', 0))
    self.addCleanup(p.stop)
    p.start()
    self.holding = False

  def lender(self, cable: lending.CableListener | None = None) -> lending.Lender:
    lender = lending.Lender(lambda: self.free, self.bounce, path=self.path,
                            holding=lambda: self.holding, cable=cable)
    assert lender.start()
    self.addCleanup(lender.stop)
    return lender

  def listener(self) -> lending.CableListener:
    listener = lending.CableListener()
    assert listener.open()
    self.addCleanup(listener.close)
    return listener

  def dial(self, listener: lending.CableListener) -> socket.socket:
    """A phone: connects, and the owner's step takes the dial."""
    phone = socket.create_connection(listener.bound[:2], timeout=3.0)
    self.addCleanup(phone.close)
    # the accept is non-blocking and the loopback handshake can still be
    # finishing when connect returns; the owner polls every step
    assert self.until(lambda: listener.poll() == '127.0.0.1'), 'the dial was never taken'
    return phone

  def take(self, **kw):
    loan = lending.borrow(path=self.path, **kw)
    if loan is not None:
      self.addCleanup(loan.close)
    return loan

  def refused(self, lender: lending.Lender, timeout: float = 0.3) -> None:
    """A borrow that gives up, and the lender marked lent while it asked.

    Watched from here while the borrower asks rather than read after: a
    borrower that gives up closes its connection, which ends the lease, and
    the lender's thread can notice that before the assert runs.
    """
    got = []
    t = threading.Thread(target=lambda: got.append(self.take(timeout=timeout)), daemon=True)
    t.start()
    assert self.until(lambda: lender.lent or bool(got)) and lender.lent, 'the daemon must still know somebody wants it'
    t.join(3.0)
    assert got == [None]

  def bounce(self) -> bool:
    self.bounced += 1
    return True

  @staticmethod
  def until(predicate, timeout=3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
      if predicate():
        return True
      time.sleep(0.01)
    return False


class Borrowing(LendingTest):
  def test_a_borrower_is_told_where_the_gadget_is(self):
    lender = self.lender()
    loan = self.take()
    assert loan is not None
    assert loan.udc == 'udc0'
    assert loan.mount == str(lending.gadget.FFS_MOUNT)
    assert lender.lent and lender.borrower == 'modeld'

  def test_nobody_listening_is_not_an_error(self):
    # the link was only just turned on, or the daemon died. The caller opens
    # the gadget itself, as it always did
    assert self.take(timeout=0.1) is None

  def test_the_daemon_is_told_to_get_off_the_endpoints_before_it_says_yes(self):
    # a borrow that lands while the daemon is mid-exchange: it hears about it
    # on the first ask, and answers once it has put the endpoints down
    self.free = False
    lender = self.lender()
    got = []
    import threading
    t = threading.Thread(target=lambda: got.append(self.take(timeout=3.0)), daemon=True)
    t.start()
    assert self.until(lambda: lender.lent), 'the daemon was never told to let go'
    assert not got, 'lent the endpoints while they were still in use'
    self.free = True
    t.join(3.0)
    assert got and got[0] is not None

  def test_during_the_hold_a_borrower_is_told_to_retry(self):
    # a phone may still dial; a hello over FunctionFS to a phone blocks 15 s
    self.holding = True
    lender = self.lender()
    self.refused(lender)
    self.holding = False
    loan = self.take(timeout=1.0)
    assert loan is not None and loan.sock is None

  def test_a_borrow_nobody_can_answer_gives_up_and_says_so(self):
    self.free = False
    lender = self.lender()
    self.refused(lender)

  def test_the_connection_is_the_lease(self):
    # modeld is stopped at every ignition-off and SIGKILLed if it lingers;
    # dying is how it hands the link back
    lender = self.lender()
    loan = self.take()
    assert lender.lent
    loan.close()
    assert self.until(lambda: not lender.lent), 'the link never came back'

  def test_a_gadget_that_is_not_bound_yet_is_waited_for(self):
    self.udc = None
    lender = self.lender()
    self.refused(lender)
    self.udc = 'udc0'
    assert self.take(timeout=1.0) is not None


class TheCable(LendingTest):
  """A phone dials the owner; the loan carries its socket, over the same unix
  socket, and the endpoint files stay where they are."""

  def test_a_loan_carries_the_phones_dial(self):
    listener = self.listener()
    phone = self.dial(listener)
    lender = self.lender(cable=listener)
    loan = self.take()
    assert loan is not None and loan.sock is not None
    assert lender.lent and lender.borrower == 'modeld'
    phone.sendall(b'hello')
    loan.sock.settimeout(3.0)
    assert loan.sock.recv(5) == b'hello'
    loan.sock.sendall(b'ready')
    assert phone.recv(5) == b'ready'

  def test_the_phone_dials_again_when_the_loan_ends(self):
    # the owner's copy of the dial goes with the loan, so a borrower that has
    # finished, or died, does not leave the phone talking to nobody
    listener = self.listener()
    phone = self.dial(listener)
    lender = self.lender(cable=listener)
    loan = self.take()
    loan.close()
    assert self.until(lambda: not lender.lent)
    assert self.until(lambda: not listener.held)
    assert phone.recv(1) == b''
    # and that redial is the same phone coming back, not a phone turning up
    assert listener.redial_expected
    self.dial(listener)
    assert not listener.news
    self.dial(listener)
    assert listener.news, 'a dial replacing a held one is a phone turning up'

  def test_a_dial_during_the_hold_ends_it_for_the_borrower(self):
    listener = self.listener()
    self.holding = True
    self.lender(cable=listener)
    got = []
    t = threading.Thread(target=lambda: got.append(self.take(timeout=3.0)), daemon=True)
    t.start()
    time.sleep(0.1)
    assert not got, 'lent inside the hold'
    self.dial(listener)
    t.join(3.0)
    assert got and got[0] is not None and got[0].sock is not None

  def test_a_usb_loan_leaves_a_later_dial_for_the_next_borrower(self):
    listener = self.listener()
    lender = self.lender(cable=listener)
    loan = self.take()
    assert loan.sock is None
    self.dial(listener)
    loan.close()
    assert self.until(lambda: not lender.lent)
    assert listener.held, 'let a phone go that nobody had borrowed'

  def test_a_dial_is_lent_whatever_the_endpoints_are_doing(self):
    # the endpoint files are not what a phone's borrower uses
    self.free = False
    self.udc = None
    listener = self.listener()
    self.dial(listener)
    self.lender(cable=listener)
    loan = self.take(timeout=1.0)
    assert loan is not None and loan.sock is not None


class Renewing(LendingTest):
  """The loan lasts the drive; which link it is for is asked again before
  every attempt at a join. On the bench a borrower that took the endpoint
  files once wrote a hello to a phone on every attempt, and each bounce that
  freed the write took the phone's network interface down before it dialed."""

  def test_a_phone_that_dialed_after_the_loan_is_what_a_renewal_gets(self):
    listener = self.listener()
    self.lender(cable=listener)
    loan = self.take()
    assert loan.sock is None
    self.dial(listener)
    assert loan.renew(timeout=3.0)
    assert loan.sock is not None

  def test_a_renewal_with_nothing_new_keeps_the_endpoint_files(self):
    self.lender()
    loan = self.take()
    assert loan.renew(timeout=1.0)
    assert loan.sock is None and loan.udc == 'udc0'

  def test_a_spent_dial_is_let_go_and_the_phones_next_one_lent(self):
    listener = self.listener()
    phone = self.dial(listener)
    self.lender(cable=listener)
    loan = self.take()
    first = loan.sock
    self.holding = True   # the owner, for a host that has dialed: a phone
    got = []
    t = threading.Thread(target=lambda: got.append(loan.renew(timeout=3.0)), daemon=True)
    t.start()
    # the session went with the attempt: the phone hears so, and dials again
    phone.settimeout(3.0)
    assert phone.recv(1) == b''
    assert not got, 'lent the endpoint files to a phone'
    self.dial(listener)
    t.join(3.0)
    assert got == [True] and loan.sock is not None and loan.sock is not first
    assert not listener.news, 'the phone coming back is not a phone turning up'

  def test_after_a_spent_dial_the_owners_hold_decides(self):
    # the lender keeps no clock of its own: with no hold, the endpoint files
    listener = self.listener()
    self.dial(listener)
    self.lender(cable=listener)
    loan = self.take()
    assert loan.renew(timeout=2.0)
    assert loan.sock is None

  def test_closing_a_loan_wakes_a_renewal_waiting_out_the_hold(self):
    # modeld shutting down must not sit behind a renewal for the whole hold
    self.lender()
    loan = self.take()
    self.holding = True
    got = []
    t = threading.Thread(target=lambda: got.append(loan.renew(timeout=5.0)), daemon=True)
    t.start()
    time.sleep(0.1)
    started = time.monotonic()
    loan.close()
    t.join(3.0)
    assert got == [False] and time.monotonic() - started < 1.0

  def test_a_renewal_during_the_hold_waits_and_keeps_the_loan(self):
    self.lender()
    loan = self.take()
    self.holding = True
    assert not loan.renew(timeout=0.2)
    assert not loan.closed

  def test_an_owner_that_is_gone_ends_the_loan(self):
    lender = self.lender()
    loan = self.take()
    lender.stop()
    assert not loan.renew(timeout=1.0)
    assert loan.closed


class TheListener(LendingTest):
  def test_a_newer_dial_replaces_an_older_one(self):
    listener = self.listener()
    first = self.dial(listener)
    self.dial(listener)
    assert listener.held
    assert first.recv(1) == b''

  def test_a_phone_that_hung_up_is_dropped(self):
    listener = self.listener()
    phone = self.dial(listener)
    phone.close()
    assert self.until(lambda: listener.poll() is None and not listener.held)

  def test_nothing_waiting_is_nothing(self):
    listener = self.listener()
    assert listener.poll() is None and not listener.held
    listener.release(expect_redial=True)   # nothing held is not an error, and nothing to expect
    assert not listener.redial_expected
    assert listener.news is False
    self.dial(listener)
    assert listener.news

  def test_an_address_that_is_not_ours_yet_is_tried_again_later(self):
    # usb0 has no address until jetlink-root.sh net has run
    listener = lending.CableListener()
    self.addCleanup(listener.close)
    with mock.patch.object(lending.gadget, 'CABLE_ADDR', ('192.0.2.1', 0)):
      assert not listener.open()
      assert not listener.listening
      assert listener.next_open > time.monotonic()
    assert not listener.open(), 'retried inside the backoff'
    listener.next_open = 0.0
    assert listener.open() and listener.listening


class Bouncing(LendingTest):
  def test_a_stuck_write_reaches_the_owner(self):
    # unbinding is the only thing that dequeues a FunctionFS write nobody is
    # reading, and the unbind belongs to whoever holds ep0
    self.lender()
    loan = self.take()
    assert loan.bounce() is True
    assert self.bounced == 1

  def test_a_bounce_after_the_loan_is_over_is_not_an_error(self):
    self.lender()
    loan = self.take()
    loan.close()
    assert loan.closed and loan.bounce() is False
    assert self.bounced == 0


class StaleSockets(LendingTest):
  def test_a_socket_a_dead_daemon_left_is_cleared(self):
    self.path.write_text('')          # anything at the address stops bind()
    lender = self.lender()
    assert lender.listening
    assert self.take() is not None
    assert lender.lent

  def test_a_live_daemon_keeps_its_socket(self):
    # two owners is a misconfiguration, and the second must not take the
    # gadget away from the one that owns it, even when it stops
    first = self.lender()
    second = lending.Lender(lambda: self.free, self.bounce, path=self.path)
    assert second.start() is False and not second.listening
    assert second.error
    second.stop()
    assert self.path.exists()
    # macOS refuses a connect while the second's probe still fills the backlog
    # of one; Linux queues it
    assert self.until(lambda: self.take() is not None)
    assert first.lent and not second.lent

