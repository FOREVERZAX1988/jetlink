"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The process that holds the gadget: what it keeps, what it lets go of, and when
it starts the heavy half.
"""
import json
import logging
import os
import select
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget, owner, root
from tests import comma_fakes


class OwnerTest(unittest.TestCase):
  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.params = self.tmp / 'params'
    self.params.mkdir()
    self.write('JetlinkLink', b'1')   # USB
    self.write('IsOffroad', b'1')
    for name, value in (('DORMANT', self.tmp / 'dormant'),
                        ('SHUTDOWN_REQUEST', self.tmp / 'shutdown'),
                        ('STATE', self.tmp / 'state'),
                        ('GADGET_STATUS', self.tmp / 'gadget-status'),
                        ('LENDER_STATUS', self.tmp / 'lender-status'),
                        ('params_dir', mock.Mock(return_value=self.params)),
                        ('link_configured', mock.Mock(return_value=True)),
                        ('host_attached', mock.Mock(return_value=True)),
                        ('udc_state', mock.Mock(return_value='configured')),
                        ('wait_for_host', mock.Mock(return_value=True)),
                        # the owner's own records, and a loopback stand-in for usb0
                        ('LINK', self.tmp / 'link'),
                        ('CABLE_ADDR', ('127.0.0.1', 0)),
                        ('net_up', mock.Mock(return_value=True)),
                        ('net_status', mock.Mock(return_value='ok 192.168.60.1')),
                        ('usb_speed', mock.Mock(return_value='super-speed'))):
      p = mock.patch.object(gadget, name, value)
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(owner, 'LOG', self.tmp / 'owner.log')
    self.addCleanup(p.stop)
    p.start()
    # every root step: the real one is sudo on a comma
    p = mock.patch.object(root, 'run', mock.Mock(return_value=True))
    self.addCleanup(p.stop)
    self.root_run = p.start()
    # the real one runs sudo on a comma, and these run there too
    p = mock.patch.object(owner.port, 'Port', mock.Mock())
    self.addCleanup(p.stop)
    p.start()

  def write(self, key: str, value: bytes) -> None:
    path = self.params / key
    path.write_bytes(value)
    # The owner sees a param move by its mtime, and Linux stamps files from a
    # clock that ticks every few ms: two writes in one tick look like none.
    self.stamp = max(getattr(self, 'stamp', 0), time.time_ns()) + 10_000_000
    os.utime(path, ns=(self.stamp, self.stamp))

  def vm_calls(self) -> list[str]:
    """What the owner asked jetlink-root.sh vm to do, in order."""
    return [c.args[1] for c in self.root_run.call_args_list if c.args[0] == 'vm']

  def note_state(self, **kw) -> None:
    gadget.STATE.write_text(json.dumps(kw))

  def owner(self, presented=True, lendable=False):
    o = owner.Owner()
    o.lender = mock.Mock(lent=False, listening=True)
    o.transport = mock.Mock(lendable=lendable) if presented else None
    for name in ('open_link', 'spawn_worker'):
      p = mock.patch.object(o, name, mock.Mock(return_value=True))
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(o, 'close_link', mock.Mock(side_effect=lambda: setattr(o, 'transport', None)))
    self.addCleanup(p.stop)
    p.start()
    self.addCleanup(o.cable.close)
    # a run has already reported, so nothing is outstanding and the far end sleeps
    self.note_state(sleep_after=1.0, unfinished=False)
    o.seen = o.marks()
    o.had_host = True
    return o

  def dial(self, o) -> socket.socket:
    """A phone: connects to the listener the owner opened."""
    self.assertTrue(o.cable.listening, 'the owner is not listening for a phone')
    phone = socket.create_connection(o.cable.bound[:2], timeout=3.0)
    self.addCleanup(phone.close)
    # the loopback handshake can still be finishing when connect returns; the
    # owner's next step takes the dial once the listener has it to accept
    readable, _, _ = select.select([o.cable._srv], [], [], 3.0)
    self.assertTrue(readable, 'the dial never reached the listener')
    return phone

  def hang_up(self, o, phone: socket.socket) -> None:
    """The phone closes its end, and the owner's next step can see it: the
    loopback FIN can still be on its way when close returns."""
    phone.close()
    readable, _, _ = select.select([o.cable._sock], [], [], 3.0)
    self.assertTrue(readable, 'the hang-up never reached the owner')


class TestOnroad(OwnerTest):
  """Once the car is moving the owner holds the gadget and stays off the bus."""

  def onroad(self) -> None:
    self.write('IsOffroad', b'0')

  def test_it_holds_the_gadget_and_does_nothing_else(self):
    o = self.owner(lendable=True)
    self.onroad()
    o.step()
    o.spawn_worker.assert_not_called()
    o.close_link.assert_not_called()
    o.transport.release_endpoints.assert_not_called()

  def test_a_run_still_going_is_stopped_so_modeld_can_borrow(self):
    # a build started while parked can still be running when the driver pulls
    # away; the lease it holds would keep modeld out for the whole drive
    o = self.owner(lendable=True)
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.onroad()
    o.step()
    worker.terminate.assert_called_once()

  def test_a_borrower_keeps_the_gadget_on_the_bus(self):
    o = self.owner(lendable=True)
    o.lender.lent = True
    o.step()
    o.close_link.assert_not_called()
    o.spawn_worker.assert_not_called()

  def test_endpoints_left_open_are_put_down_without_letting_go_of_ep0(self):
    o = self.owner(lendable=False)
    self.onroad()
    o.step()
    o.transport.release_endpoints.assert_called_once()
    o.close_link.assert_not_called()

  def test_a_borrower_wakes_a_dormant_owner(self):
    o = self.owner(presented=False)
    o.dormant = True
    o.lender.lent = True
    o.step()
    self.assertFalse(o.dormant)
    o.open_link.assert_called_once()


class TestParked(OwnerTest):
  """Letting the Jetson sleep, and taking the gadget back when there is work."""

  def test_the_gadget_is_held_until_the_hold_has_passed(self):
    o = self.owner()
    o.step()
    self.assertFalse(o.dormant)

  def test_it_releases_once_there_is_nothing_to_do_and_the_far_end_sleeps(self):
    o = self.owner()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertTrue(o.dormant)
    o.close_link.assert_called_once()
    self.assertTrue(gadget.dormant())

  def test_a_far_end_that_never_sleeps_keeps_the_gadget(self):
    # on ignition power the Jetson stays up, and letting go would leave a
    # powered awake box unenumerated for the whole parked period
    o = self.owner()
    self.note_state(sleep_after=0.0, unfinished=False)
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertFalse(o.dormant)
    o.close_link.assert_not_called()

  def test_a_far_end_too_old_to_say_keeps_the_release_it_always_had(self):
    gadget.STATE.unlink(missing_ok=True)
    o = self.owner()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertTrue(o.dormant)

  def test_a_run_that_wakes_the_jetson_gets_the_hold_before_letting_go(self):
    # bench 2026-09-10: a run finished 2.5 s after the wake, the owner released
    # the gadget 1 ms later, and the jetson was still enumerating. The hold used
    # to run from process start, which expires once and never applies again now
    # that this is not restarted at ignition
    o = self.owner()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.worker = mock.Mock(**{'poll.return_value': 0, 'returncode': 0})
    o.step()
    self.assertFalse(o.dormant, 'let the gadget go while the jetson was waking')
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertTrue(o.dormant)

  def test_a_borrower_holds_the_gadget_past_the_drive(self):
    o = self.owner()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.lender.lent = True
    o.step()
    o.lender.lent = False
    o.step()
    self.assertFalse(o.dormant, 'let the gadget go the moment the drive ended')


class TestStartingTheHeavyHalf(OwnerTest):
  """The owner cannot tell whether there is work: that needs the catalog, the
  spec and the Jetson. It notices what could have changed the answer."""

  def test_the_first_look_of_the_boot_always_runs(self):
    o = self.owner()
    o.seen = {}
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_new_pick_starts_a_run(self):
    o = self.owner()
    o.step()
    o.spawn_worker.assert_not_called()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "b" * 40}')
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_jetson_turning_up_starts_a_run(self):
    o = self.owner()
    o.had_host = False
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_shutdown_request_starts_a_run(self):
    o = self.owner()
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))
    o.step()
    o.spawn_worker.assert_called_once()

  def test_an_unfinished_run_is_tried_again_on_its_own_timer(self):
    o = self.owner()
    self.note_state(sleep_after=1.0, unfinished=True)
    o.next_worker = time.monotonic() + owner.WORKER_BACKOFF
    o.step()
    o.spawn_worker.assert_not_called()
    o.next_worker = 0.0
    o.step()
    o.spawn_worker.assert_called_once()

  def test_only_one_run_at_a_time(self):
    o = self.owner()
    o.seen = {}
    o.worker = mock.Mock(**{'poll.return_value': None})
    o.step()
    o.spawn_worker.assert_not_called()

  def test_a_dormant_owner_wakes_before_starting_one(self):
    o = self.owner(presented=False)
    o.dormant = True
    o.seen = {}
    o.step()
    self.assertFalse(o.dormant)
    o.spawn_worker.assert_called_once()

  def test_nothing_is_started_over_the_servers_teardown(self):
    # the borrower let go a moment ago and the server is still reopening the
    # gadget it lost; a hello inside that window costs a re-enumeration
    o = self.owner()
    o.lender.lent = True
    o.step()
    o.lender.lent = False
    o.seen = {}
    o.step()
    o.spawn_worker.assert_not_called()
    o.lease_settled = 0.0
    o.step()
    o.spawn_worker.assert_called_once()


class TestTheRunThatFinishes(OwnerTest):
  def finished(self, o):
    """A worker that has just exited."""
    o.worker = mock.Mock(**{'poll.return_value': 0, 'returncode': 0})

  def test_a_successful_provision_does_not_start_a_second_run(self):
    # a run writes JetlinkSpec itself, so a mark taken when it was spawned
    # always differs by the time it exits
    o = self.owner()
    self.finished(o)
    self.write('JetlinkSpec', b'{}')
    o.step()
    o.spawn_worker.assert_not_called()

  def test_a_pick_changed_while_the_run_was_going_is_still_seen(self):
    o = self.owner()
    o.worker = mock.Mock(**{'poll.return_value': None})
    o.step()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "c"}')
    o.worker.poll.return_value = 0
    o.step()
    # the mark is retaken when the run exits, so this looks unchanged...
    o.spawn_worker.assert_not_called()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "d"}')
    o.step()
    o.spawn_worker.assert_called_once()   # ...and a later pick still starts one


class TestShutdown(OwnerTest):
  """hardwared waits 25 s and a build takes minutes, so the request cannot
  queue behind a provisioning run."""

  def request(self) -> None:
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))

  def test_it_does_not_wait_for_a_run_in_flight(self):
    o = self.owner()
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.request()
    o.step()
    worker.terminate.assert_called_once()
    o.spawn_worker.assert_called_once()
    assert 'shut down' in o.spawn_worker.call_args.args[0]

  def test_the_run_it_starts_is_left_to_ask(self):
    # the run asking the jetson was stopped and started again every step, half
    # a second, less than it takes to start: on the bench nothing ever asked
    o = self.owner()
    self.request()
    o.step()
    o.spawn_worker.assert_called_once()
    asking = o.worker = mock.Mock(**{'poll.return_value': None})
    for _ in range(5):
      o.step()
    asking.terminate.assert_not_called()
    o.spawn_worker.assert_called_once()

  def test_a_run_that_exits_with_the_request_there_is_tried_again_shortly(self):
    o = self.owner()
    self.request()
    o.step()
    o.worker = mock.Mock(**{'poll.return_value': 0, 'returncode': 0})
    o.step()                                  # it finished, within the retry wait
    o.spawn_worker.assert_called_once()
    o.next_shutdown_run = 0.0                 # the wait is over
    o.step()
    assert o.spawn_worker.call_count == 2

  def test_a_borrower_is_left_alone(self):
    # modeld has the endpoints: this cannot talk over it, and hardwared only
    # shuts a parked car down anyway
    o = self.owner()
    o.lender.lent = True
    self.request()
    o.step()
    o.spawn_worker.assert_not_called()


class TestNobodyCanBorrow(OwnerTest):
  """Only the owner ever holds ep0. A lender that cannot listen leaves nobody
  a way to the link, so the owner keeps the gadget, says why where the panels
  look, and tries again."""

  REASON = 'the lender could not listen: [Errno 30] Read-only file system'

  def owner(self, **kw):
    o = super().owner(**kw)
    o.lender.listening = False
    o.lender.start.return_value = False
    o.lender.error = '[Errno 30] Read-only file system'
    return o

  def test_the_gadget_is_held_through_the_drive(self):
    o = self.owner(lendable=True)
    self.write('IsOffroad', b'0')
    o.step()
    o.close_link.assert_not_called()
    self.assertIsNotNone(o.transport)
    self.assertEqual(gadget.gadget_error(), self.REASON)
    self.assertEqual(gadget.LENDER_STATUS.read_text(), f'error: {self.REASON}\n')

  def test_it_is_not_a_build_failure(self):
    # the gadget exists; rebuilding it would not help, and would unplug the host
    o = self.owner()
    o.step()
    self.assertIsNone(gadget.build_error())

  def test_said_once_and_retried_on_a_backoff(self):
    o = self.owner()
    with mock.patch.object(gadget, 'log') as log:
      for _ in range(3):
        o.step()
      o.lender.start.assert_called_once()
      self.assertEqual(log.error.call_count, 1)
      o.next_lender = 0.0
      o.step()
      self.assertEqual(o.lender.start.call_count, 2)
      self.assertEqual(log.error.call_count, 1, 'said it again every retry')

  def test_listening_again_clears_the_error(self):
    o = self.owner()
    o.step()
    o.lender.start.return_value = True
    o.next_lender = 0.0
    o.step()
    self.assertIsNone(gadget.gadget_error())
    self.assertFalse(o.lender_failed)

  def test_a_stop_clears_the_error(self):
    # a chestnut turning up stops the owner with the link still on
    o = self.owner()
    o.step()
    o.stop = True
    o.run()
    self.assertIsNone(gadget.gadget_error())

  def test_parked_it_still_provisions(self):
    o = self.owner(lendable=True)
    o.seen = {}
    o.step()
    o.close_link.assert_not_called()
    o.spawn_worker.assert_called_once()

  def test_a_listening_lender_is_left_alone(self):
    o = super().owner()
    o.step()
    o.lender.start.assert_not_called()
    self.assertIsNone(gadget.gadget_error())


class TestTheToggle(OwnerTest):
  def test_turning_it_off_lets_everything_go(self):
    o = self.owner()
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.write('JetlinkLink', b'0')
    o.step()
    o.close_link.assert_called_once()
    worker.terminate.assert_called_once()

  def test_the_first_gadget_is_not_held_up_by_the_sysctls(self):
    self.write('IsOffroad', b'0')
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False):
      o.step()
    self.assertEqual([c.args[0] for c in self.root_run.call_args_list], ['gadget', 'vm'])

  def test_the_port_is_kept_a_device_while_the_link_is_on(self):
    o = self.owner()
    o.step()
    o.port.update.assert_called_once_with()

  def test_turning_it_off_gives_the_port_back(self):
    o = self.owner()
    self.write('JetlinkLink', b'0')
    o.step()
    o.port.off.assert_called_once()
    o.port.update.assert_not_called()

  def test_stopping_gives_the_port_back(self):
    o = self.owner()
    o.lender.start.return_value = True
    o.stop = True
    o.run()
    o.port.off.assert_called_once()


class TestVmTuning(OwnerTest):
  """When the owner applies and restores the VM tuning, against the real
  jetlink-root.sh vm on a fake /proc/sys. A device with the link off runs
  stock values, one that turns it off gets them back, and a plain exit keeps
  them for the drive that follows. The values and the ratio-mode restore are
  test_comma_root.py's."""

  def setUp(self):
    super().setUp()
    comma_fakes.proc_sys(self.tmp, comma_fakes.STOCK)
    self.root_run.side_effect = self.run_script

  def run_script(self, *args: str, timeout: float = root.TIMEOUT) -> bool:
    """root.run, without sudo, on the fake /proc/sys."""
    return comma_fakes.run_script(self.tmp, *args, timeout=timeout).returncode == 0

  def tuned(self) -> bool:
    return comma_fakes.read_all(self.tmp, comma_fakes.TUNED) == comma_fakes.TUNED

  def test_applied_once_on_start_and_kept_on_exit(self):
    o = self.owner()
    o.step()
    o.step()
    o.stop = True
    o.run()
    self.assertEqual(self.vm_calls(), ['apply'], "an exit is the ignition handoff; restoring there strips the drive of them")
    self.assertTrue(self.tuned())
    self.assertTrue(comma_fakes.record(self.tmp).exists(), "the record is what a later disable restores to")

  def test_the_next_start_reapplies_without_touching_the_record(self):
    self.owner().step()
    stock = comma_fakes.record(self.tmp).read_text()
    self.owner().step()
    self.assertEqual(self.vm_calls(), ['apply', 'apply'])
    self.assertEqual(comma_fakes.record(self.tmp).read_text(), stock)

  def test_nothing_happens_when_disabled(self):
    self.write('JetlinkLink', b'0')
    self.owner().step()
    self.assertEqual(self.vm_calls(), [])
    self.assertFalse(comma_fakes.record(self.tmp).exists())

  def test_disabling_mid_run_restores(self):
    o = self.owner()
    o.step()
    self.write('JetlinkLink', b'0')
    o.step()
    self.assertEqual(self.vm_calls(), ['apply', 'restore'])
    self.assertFalse(comma_fakes.record(self.tmp).exists(), 'the restore never ran')


class TestUsb(OwnerTest):
  """Accelerator Link USB: a Jetson or a Mac. The plain gadget, lent at once;
  nothing waits for a phone and nothing of the phone's runs."""

  def test_borrowers_are_never_held_off(self):
    o = self.owner()
    o.step()
    self.assertFalse(o.holding())
    self.assertEqual(gadget.link_kind(), 'usb')

  def test_no_network_and_no_listener(self):
    o = self.owner()
    o.step()
    o.step()
    gadget.net_up.assert_not_called()
    self.assertFalse(o.cable.listening)


class IosTest(OwnerTest):
  """Accelerator Link iOS: an iPhone on the gadget's network interface."""

  def setUp(self):
    super().setUp()
    self.write('JetlinkLink', b'2')   # iOS

  def owner(self, **kw):
    o = super().owner(**kw)
    o.built_ios = True
    return o


class TestCable(IosTest):
  """The phone dials the owner, and a borrower gets its socket; nothing is
  ever lent the endpoint files, which a phone does not read."""

  def test_borrowers_wait_for_the_phone(self):
    o = self.owner()
    o.step()
    self.assertTrue(o.holding())
    self.assertEqual(gadget.link_kind(), 'cable')

  def test_a_dial_is_the_link(self):
    o = self.owner()
    o.step()
    self.dial(o)
    o.step()
    self.assertEqual(gadget.link_peer(), '127.0.0.1')
    self.assertTrue(o.cable.held)
    o.spawn_worker.assert_called_once()
    self.assertIn('phone', o.spawn_worker.call_args.args[0])

  def test_the_phone_coming_back_after_a_run_is_not_another_run(self):
    # the owner hangs up when the borrower finishes and the phone dials
    # again: that must not start a run, which would end the same way, forever
    o = self.owner()
    o.step()
    o.cable.redial_expected = True
    self.dial(o)
    o.step()
    o.spawn_worker.assert_not_called()
    self.assertFalse(o.dialed)

  def test_a_dial_during_a_run_is_left_to_that_run(self):
    # the run spawned at the configured edge is still borrowing, and its
    # borrow takes the dial; a second run after it would be the same work
    o = self.owner()
    o.step()
    o.worker = mock.Mock(**{'poll.return_value': None})
    self.dial(o)
    o.step()
    self.assertFalse(o.dialed)
    o.worker.poll.return_value = 0
    o.worker.returncode = 0
    o.step()
    o.spawn_worker.assert_not_called()

  def test_the_owner_never_sleeps_or_settles(self):
    # every unbind takes the phone's network interface down with it
    o = self.owner(lendable=False)
    o.step()
    self.dial(o)
    o.step()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertFalse(o.dormant)
    o.close_link.assert_not_called()
    o.transport.release_endpoints.assert_not_called()

  def test_a_phone_that_hung_up_is_let_go_and_waited_for(self):
    o = self.owner()
    o.step()
    phone = self.dial(o)
    o.step()
    self.hang_up(o, phone)
    o.step()
    self.assertFalse(o.cable.held)
    self.assertIsNone(gadget.link_peer())
    self.assertTrue(o.holding(), 'lent the endpoint files to a phone between its dials')

  def test_a_phone_that_hung_up_does_not_send_the_owner_dormant(self):
    # the record says the far end sleeps (a run with nothing to do never
    # asks), but for iOS the gadget stays up: letting go would take the
    # network interface the phone dials back over
    o = self.owner()
    o.step()
    phone = self.dial(o)
    o.step()
    self.hang_up(o, phone)
    o.step()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertFalse(o.dormant)

  def test_the_host_going_away_clears_the_cable_link(self):
    o = self.owner()
    o.step()
    phone = self.dial(o)
    o.step()
    gadget.host_attached.return_value = False
    o.step()
    self.assertIsNone(gadget.link_peer())
    self.assertFalse(o.cable.held)
    self.assertEqual(phone.recv(1), b'', 'the phone was left talking to nobody')

  def test_a_newer_dial_replaces_an_older_one(self):
    # the app restarted: its old connection must not keep the new one out
    o = self.owner()
    o.step()
    first = self.dial(o)
    o.step()
    self.dial(o)
    o.step()
    self.assertTrue(o.cable.held)
    first.settimeout(3.0)
    self.assertEqual(first.recv(1), b'')

  def test_the_shutdown_path_still_presents_the_gadget(self):
    # the phone is on the gadget's network interface: no gadget, no phone
    o = self.owner(presented=False)
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))
    o.step()
    o.open_link.assert_called_once()
    o.spawn_worker.assert_called_once()

  def test_closing_the_link_takes_the_listener_and_the_record_with_it(self):
    o = owner.Owner()
    o.lender = mock.Mock(lent=False, listening=True)
    o.transport = mock.Mock()
    self.assertTrue(o.cable.open())
    gadget.note_link('cable', '192.168.60.3')
    o.close_link()
    self.assertFalse(o.cable.listening)
    self.assertIsNone(gadget.link_peer())
    o.transport = None


class TestTheGadgetNetwork(IosTest):
  """usb0 exists only once the UDC is bound, so the owner brings it up after
  its bind, and listens for a phone only once it is there."""

  def test_the_network_comes_up_once_per_bind(self):
    o = self.owner()
    o.step()
    o.step()
    gadget.net_up.assert_called_once()
    self.assertTrue(o.cable.listening)

  def test_a_failed_bring_up_is_retried_after_a_backoff_and_nothing_listens(self):
    gadget.net_up.return_value = False
    o = self.owner()
    for _ in range(3):
      o.step()
    gadget.net_up.assert_called_once()
    self.assertFalse(o.cable.listening, 'the bind to 192.168.60.1 fails without usb0')
    gadget.net_up.return_value = True
    o.next_net_attempt = 0.0
    o.step()
    self.assertEqual(gadget.net_up.call_count, 2)
    self.assertTrue(o.cable.listening)

  def test_nothing_presented_brings_nothing_up(self):
    o = self.owner(presented=False)
    o.step()
    gadget.net_up.assert_not_called()

  def test_a_bounce_brings_the_network_up_again(self):
    # the unbind took usb0 with it and the rebind made a bare one
    o = self.owner()
    o.step()
    o.transport.rebind.return_value = True
    self.assertTrue(o.bounce_gadget())
    o.step()
    self.assertEqual(gadget.net_up.call_count, 2)

  def test_an_address_that_is_not_there_yet_does_not_stop_the_owner(self):
    # the network said ok but the address is not local (a race with the
    # script): the bind fails, is noted, and is tried again later
    gadget.CABLE_ADDR = ('192.0.2.1', 0)
    o = self.owner()
    o.step()
    self.assertFalse(o.cable.listening)
    self.assertGreater(o.cable.next_open, time.monotonic())
    gadget.CABLE_ADDR = ('127.0.0.1', 0)
    o.step()
    self.assertFalse(o.cable.listening, 'retried inside the backoff')
    o.cable.next_open = 0.0
    o.step()
    self.assertTrue(o.cable.listening)


class TestSwitchingMode(OwnerTest):
  """USB and iOS are different gadgets; moving the setting rebuilds it, and
  only while parked with nobody on the link: the rebuild is an unplug."""

  def setUp(self):
    super().setUp()
    p = mock.patch.object(gadget, 'setup_gadget', mock.Mock(return_value=True))
    self.addCleanup(p.stop)
    self.setup_gadget = p.start()

  def switched(self, **kw):
    o = self.owner(**kw)
    o.built_ios = False
    self.write('JetlinkLink', b'2')   # iOS
    return o

  def test_parked_it_rebuilds_for_the_new_host(self):
    o = self.switched()
    o.step()
    o.close_link.assert_called_once()
    self.setup_gadget.assert_called_once()
    self.assertTrue(o.built_ios)
    o.step()
    self.setup_gadget.assert_called_once()

  def test_onroad_it_waits_for_the_car_to_park(self):
    self.write('IsOffroad', b'0')
    o = self.switched()
    o.step()
    self.setup_gadget.assert_not_called()
    self.write('IsOffroad', b'1')
    o.step()
    self.setup_gadget.assert_called_once()

  def test_a_borrower_on_the_link_is_not_unplugged(self):
    o = self.switched()
    o.lender.lent = True
    o.step()
    self.setup_gadget.assert_not_called()

  def test_what_is_built_is_learned_onroad_too(self):
    # an owner starting mid-drive on an iOS gadget must not take it for USB:
    # that would lend a phone the endpoint files
    self.write('IsOffroad', b'0')
    o = self.owner()
    o.built_ios = None
    with mock.patch.object(gadget, 'built_for_ios', return_value=True):
      o.step()
    self.assertTrue(o.built_ios)
    self.assertTrue(o.holding())
    self.assertEqual(gadget.link_kind(), 'cable')
    self.setup_gadget.assert_not_called()

  def test_a_switch_just_after_a_build_is_not_held_back(self):
    # the bench: a switch a minute after the last one waited out a backoff
    # that only a failed build should set, and logged every step meanwhile
    o = self.switched()
    o.step()
    self.assertTrue(o.built_ios)
    self.write('JetlinkLink', b'1')
    o.step()
    self.assertFalse(o.built_ios)
    self.assertEqual(self.setup_gadget.call_count, 2)

  def test_a_failed_rebuild_is_retried_after_a_backoff(self):
    self.setup_gadget.return_value = False
    o = self.switched()
    for _ in range(3):
      o.step()
    self.setup_gadget.assert_called_once()
    self.assertFalse(o.built_ios)

  def test_a_run_in_flight_is_not_unplugged(self):
    # the rebuild is an unplug, and a run may be mid-upload or mid-build
    o = self.switched()
    o.worker = mock.Mock(**{'poll.return_value': None})
    o.step()
    self.setup_gadget.assert_not_called()
    o.close_link.assert_not_called()
    o.worker.poll.return_value = 0
    o.worker.returncode = 0
    o.step()
    self.setup_gadget.assert_called_once()
    self.assertTrue(o.built_ios)


class TestTheLoop(OwnerTest):
  """What run() does around each step: nothing may escape it, since manager
  restarting the owner in a loop is worse than sitting out a cycle."""

  def run_steps(self, o, step) -> None:
    """run(), with `step` for the owner's step and no wait between cycles."""
    with mock.patch.object(o, 'step', side_effect=step), mock.patch.object(owner, 'POLL', 0.0):
      o.run()

  def test_an_error_lets_the_link_go_backs_off_and_carries_on(self):
    o = self.owner()
    seen = []

    def step():
      seen.append((o.next_attempt, o.close_link.call_count))
      if len(seen) == 1:
        raise RuntimeError('a step that fails')
      o.stop = True

    started = time.monotonic()
    with mock.patch.object(gadget, 'log') as log:
      self.run_steps(o, step)
    self.assertEqual(len(seen), 2, 'the loop ended at the error')
    next_attempt, closed = seen[1]
    self.assertEqual(closed, 1, 'the link was not let go after the error')
    self.assertGreaterEqual(next_attempt, started + owner.RECONNECT_BACKOFF)
    log.exception.assert_called_once()

  def test_a_previous_owners_record_is_cleared_before_the_first_step(self):
    # the record is this process's to write; one a killed owner left says
    # nothing about the gadget now
    gadget.note_link('cable', '192.168.60.3')
    o = self.owner()
    seen = []

    def step():
      seen.append(gadget.link_peer())
      o.stop = True

    self.run_steps(o, step)
    self.assertEqual(seen, [None])
    self.assertFalse(gadget.LINK.exists())


class TestSetup(OwnerTest):
  def test_the_owner_creates_the_gadget_when_there_is_none(self):
    # nothing sets it up at boot any more
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False), \
         mock.patch.object(gadget, 'setup_gadget', return_value=True) as setup:
      self.assertTrue(o.ensure_gadget(False))
      setup.assert_called_once()

  def test_a_failed_setup_is_not_retried_every_cycle(self):
    # off AGNOS too, where root.run is a False for everything
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False), \
         mock.patch.object(gadget, 'setup_gadget', return_value=False) as setup:
      for _ in range(3):
        self.assertFalse(o.ensure_gadget(False))
      setup.assert_called_once()


class TestTheWorker(OwnerTest):
  """The caller names the provisioning run; the owner knows no openpilot module."""

  def test_the_run_is_the_callers_argv_cwd_and_env(self):
    o = owner.Owner(['python3', '-m', 'the.worker'], cwd='/data/openpilot', env={'PYTHONPATH': '/data/openpilot'})
    with mock.patch.object(owner.subprocess, 'Popen') as popen:
      o.spawn_worker('nothing has been checked since boot')
    (argv,), kwargs = popen.call_args
    self.assertEqual(argv, ['python3', '-m', 'the.worker'])
    self.assertEqual(kwargs['cwd'], '/data/openpilot')
    self.assertEqual(kwargs['env'], {**os.environ, 'PYTHONPATH': '/data/openpilot'})
    self.assertIs(o.worker, popen.return_value)

  def test_a_run_that_will_not_start_is_not_an_error(self):
    o = owner.Owner()
    with mock.patch.object(owner.subprocess, 'Popen', side_effect=OSError('no such file')):
      o.spawn_worker('nothing has been checked since boot')
    self.assertIsNone(o.worker)

  def test_main_hands_the_worker_to_the_owner_and_logs_to_the_file(self):
    log = self.tmp / 'owner-main.log'
    with mock.patch.object(owner, 'Owner') as made, mock.patch.object(owner.signal, 'signal'), \
         mock.patch.object(gadget, 'set_logger') as set_logger:
      owner.main(['python3', '-m', 'the.worker'], cwd='/x', env={'A': 'b'}, log_file=log)
    made.assert_called_once_with(['python3', '-m', 'the.worker'], cwd='/x', env={'A': 'b'})
    made.return_value.run.assert_called_once()
    logger = set_logger.call_args.args[0]
    self.addCleanup(lambda: [logger.removeHandler(h) or h.close() for h in list(logger.handlers)])
    logger.warning('jetlink: a line for the file')
    self.assertIn('a line for the file', log.read_text())
    self.assertIsInstance(logger, logging.Logger)


class TestLending(OwnerTest):
  def test_lendable_only_once_the_endpoints_are_down(self):
    o = self.owner(lendable=False)
    self.assertFalse(o.lendable())
    o.transport.lendable = True
    self.assertTrue(o.lendable())
    o.transport = None
    self.assertFalse(o.lendable())

  def test_a_stuck_write_is_freed_by_the_owner(self):
    o = self.owner()
    o.transport.rebind.return_value = True
    self.assertTrue(o.bounce_gadget())
    o.transport.rebind.assert_called_once()

  def test_nothing_to_bounce_is_not_an_error(self):
    o = self.owner(presented=False)
    self.assertFalse(o.bounce_gadget())
    self.assertFalse(o.lendable())

  def test_a_bounce_that_raises_is_not_an_error(self):
    o = self.owner()
    o.transport.rebind.side_effect = OSError('no such device')
    self.assertFalse(o.bounce_gadget())

