"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's USB-C port, kept the device end of a USB link.

The port is dual role. It hosts a chestnut, and it is the device for a jetlink
host. An A-to-C cable settles which by construction: the A end only pulls CC
up, so the comma can only be the sink and the device. A C-to-C cable does not.
Both ends are dual role, and the comma can come out the source and the host:
facing a Mac, nothing enumerates; facing an iPhone or a Jetson's own USB-C
port, the far end enumerates as a device. Carrot's Jetson on C-to-C read
"Powered cable w/ sink" and enumerated as 0955:7020.

While the link is on, a chestnut is the only thing the comma should host on
this port. So once the comma has been the source for a few seconds with no
chestnut on the port, whatever is on the other end is a host that lost the
toss, and the port is held at sink until that cable comes out. Nothing happens
anywhere else: the comma as the device (every USB-A host, a C-to-C host that
won), a power supply and a chestnut all leave the port as AGNOS boots it.
Holding for the whole session would be simpler and would hide a chestnut
plugged in while the link is on, since chestnut_present() needs the comma to
host it before jetlink stands aside.

The lever is the charger's DISABLE_POWER_ROLE_SWITCH voter, forced from
debugfs. It is the one that holds. The charger puts the port back to dual role
on every unplug and refuses a role written through the power supply once
nothing is attached, so the policy engine's rev3_sink_only and dual_role/mode
last one plug at most; a forced voter gates all of those writes.

The voter only moves the power role, and a USB PD power role swap leaves the
data roles where they were. The comma's policy engine, as the source, accepts
a PR_Swap at any time, and its sink startup keeps the host role it had. So a
far end that came out the device and then asked for the source role, as a
phone or tablet that powers what it plugs into may, leaves the comma the sink
and still the host: no hold fires, and none would help. For that pair the
comma asks over USB PD for a data role swap (the dual_role class's data_role,
which sends a DR_Swap and moves nothing else), a few times, then once for a
hard reset, after which a sink is the device by the spec. Neither is ever sent
to a far end the comma is the device of.

USB PD is otherwise left alone; a host that comes back negotiates as over any
cable. The hold does not survive a reboot.

The right roles are not enough on their own: the comma also has to turn its
device side on. As a sink the policy engine does that only when the charger
detection read a USB port or floating data lines, and dwc3 turns it off again
when the lines read floating and no host has enumerated it in 10 s. Behind a
hub the lines are a USB port's. On a direct cable an Apple port connects them
only once USB PD is done, and a comma that boots with the phone plugged in has
no gadget bound for far longer than 10 s. So while the link is on both are
told to keep the device (jetlink-root.sh udc apply, which the owner runs with
the VM tuning), a sink that is the device with the device side still off a few
seconds into a plug has it turned on, and an empty port with it still on has it
turned off.

Every change of the port's roles is a line in the owner's log, so a plug that
did not connect can be read afterwards.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from jetlink.comma import gadget, root

USBPD = Path('/sys/class/usbpd/usbpd0')
POWER_ROLE = USBPD / 'current_pr'
DATA_ROLE = USBPD / 'current_dr'
CONTRACT = USBPD / 'contract'
USB_PSY = Path('/sys/class/power_supply/usb')
TYPEC_MODE = USB_PSY / 'typec_mode'
CHARGER = USB_PSY / 'real_type'
# dwc3's glue for usb0: 'peripheral' while the device side is on
UDC_MODE = Path('/sys/devices/platform/soc/a600000.ssusb/mode')
USB_DEVICES = Path('/sys/bus/usb/devices')
# how long the comma hosts the far end before judging it. A chestnut enumerates
# well inside this, and a far end that powers us and wants the host role has
# asked for it by then
SWAP_AFTER = 3.0
# DR_Swap requests to a far end that powers us and came out the device, a plug,
# this far apart; then one hard reset
SWAP_TRIES = 3
RETRY_AFTER = 2.0
# a device side turned on that went off again is turned on no sooner than
# this: dwc3 gives floating lines 10 s
RESTART_AFTER = 10.0
# how long the port reads empty before a hold is let go. The hold itself is an
# unplug and replug, so this has to outlast the replug
RELEASE_AFTER = 5.0
# how long the port reads empty before a plug counts as gone. A device let go
# by a hold reattaches in a DRP toggle and a CC debounce, well under this
UNPLUGGED = 1.5
ATTACHED = ('source', 'sink')


def power_role() -> str | None:
  """'source', 'sink' or 'none' from the policy engine, or None without one."""
  return gadget.read(POWER_ROLE) or None


def data_role() -> str | None:
  """'dfp' (the comma is the host), 'ufp' (the device) or 'none'."""
  return gadget.read(DATA_ROLE) or None


def chestnut_attached(chestnut_ids: frozenset[tuple[int, int]]) -> bool:
  """Is a chestnut enumerated, running or in its ROM? It can only be on this port."""
  try:
    names = os.listdir(USB_DEVICES)
  except OSError:
    return False
  for name in names:
    if ':' in name:
      continue   # an interface
    try:
      ids = tuple(int((USB_DEVICES / name / f).read_text(), 16) for f in ('idVendor', 'idProduct'))
    except (OSError, ValueError):
      continue   # a device going away
    if ids in chestnut_ids:
      return True
  return False


def run_script(command: str) -> bool:
  """jetlink-root.sh port hold|off|device|reset. Both commas have the levers,
  so a False is a failure, and root.run has logged why. Its timeout is short
  because the off in the owner's finally comes before the FunctionFS close,
  inside manager's 5 s."""
  return root.run('port', command, timeout=root.PORT_TIMEOUT)


def run_udc(command: str) -> bool:
  """jetlink-root.sh udc start|stop, on the same short timeout."""
  return root.run('udc', command, timeout=root.PORT_TIMEOUT)


class Port:
  """Called once a cycle by the owner. A cycle reads the power role, the data
  role while something is attached, and the device controller's mode while a
  host that powers us has not configured the gadget; sudo only runs on a
  change."""

  def __init__(self, chestnut_ids):
    # what a chestnut enumerates as, running or in its ROM: openpilot's
    # CHESTNUT_USB_IDS and CHESTNUT_ROM_USB_IDS, as the fork's adapter hands
    # them over (OwnerConfig.chestnut_ids), so a chestnut being flashed is
    # never taken for a host
    self.chestnut_ids = frozenset(chestnut_ids)
    self._reset()

  def _reset(self) -> None:
    self.cleared = False   # the port put back as AGNOS boots it, once a session
    self.deferred = False  # that waits for a host an owner before this one had
    self.held = False      # the voter is forced to sink
    # this plug has been judged, so leave it until it comes out. Also set for
    # the length of a hold until a host comes back
    self.settled = False
    self.role: str | None = None
    self.role_since = 0.0
    self.roles: tuple[str | None, str | None] = (None, None)   # power and data
    self.roles_since = 0.0
    # something is on the port, or was until less than UNPLUGGED ago. Set at
    # the start, so the first empty port is looked at too
    self.plugged = True
    self.asks = 0          # this plug's: SWAP_TRIES DR_Swaps, then a hard reset
    self.next_ask = 0.0
    self.next_start = 0.0

  def update(self, now: float | None = None, configured: bool = False) -> None:
    """configured: a host had the gadget configured at the last cycle, so the
    device side is on."""
    if not self.cleared:
      # whatever an owner killed mid-hold left behind. Not under a live link:
      # an owner started after one that died can find a borrower still on the
      # gadget the dead one presented, through a hold it made, and back at
      # dual role the port may toss the roles again. Once no host has us
      # configured the link has gone anyway, and the hold goes then
      if gadget.host_attached():
        if not self.deferred:
          gadget.log.warning("jetlink: a host is still on the gadget; leaving the USB-C port as it is until it goes")
          self.deferred = True
      else:
        run_script('off')
        self.cleared = True
    now = time.monotonic() if now is None else now
    role = power_role()
    data = data_role() if role in ATTACHED else None
    if (role, data) != self.roles:
      if role is not None:
        self._note(role, data)
      self.roles, self.roles_since = (role, data), now
    if role != self.role:
      self.role, self.role_since = role, now
    lasted = now - self.role_since
    if role in ATTACHED:
      self.plugged = True
    elif self.plugged and lasted >= UNPLUGGED:
      self._unplugged()
    if role == 'sink' and data == 'ufp' and not configured:
      self._keep_the_device_on(now)
    if self.held:
      if role == 'sink':
        self.settled = False   # a host came back
      elif role != 'source' and lasted >= RELEASE_AFTER:
        self._release()
        # the accessory reattaches in a moment; time the gap afresh
        self.role_since, self.plugged = now, True
    elif role == 'source' or data == 'dfp':
      # the comma is the host, which only a chestnut should make it. The
      # source is held at sink; a sink is asked over USB PD
      hosted = now - self.roles_since
      if not self.settled and hosted >= SWAP_AFTER and now >= self.next_ask:
        if self.asks == 0 and chestnut_attached(self.chestnut_ids):
          self.settled = True
        elif role == 'source':
          self._hold(hosted)
        else:
          self._ask_for_a_host(now)
    elif role == 'sink' or not self.plugged:
      self.settled = False     # a host, or the plug is gone

  def _note(self, role: str, data: str | None) -> None:
    gadget.log.info("jetlink: USB-C port %s, %s; USB PD contract %s; Type-C %s; charger detection %s",
                    role, {'dfp': 'host', 'ufp': 'device'}.get(data, 'no data role'),
                    gadget.read(CONTRACT) or 'unknown', gadget.read(TYPEC_MODE) or 'unknown',
                    gadget.read(CHARGER) or 'unknown')

  def _unplugged(self) -> None:
    self.plugged = False
    self.asks = 0            # the next plug is asked afresh
    # the policy engine turns off only a device side it turned on itself, so
    # one turned on here, by this owner or one before it, is turned off here
    if gadget.read(UDC_MODE) == 'peripheral':
      run_udc('stop')

  def _keep_the_device_on(self, now: float) -> None:
    """A host powers the port and the comma is its device, so the device side
    should be on; see the module's docstring for when it is not."""
    if now - self.roles_since < SWAP_AFTER or now < self.next_start:
      return
    mode = gadget.read(UDC_MODE)
    if mode in ('', 'peripheral'):
      return
    gadget.log.warning("jetlink: a host powers the USB-C port but the USB device controller is %s "
                       "(charger detection %s); turning its device side on", mode, gadget.read(CHARGER) or 'unknown')
    self.next_start = now + RESTART_AFTER
    run_udc('start')

  def _ask_for_a_host(self, now: float) -> None:
    """The far end powers the comma and is still its device; see the module's
    docstring for how that comes about."""
    self.asks += 1
    self.next_ask = now + RETRY_AFTER
    if self.asks <= SWAP_TRIES:
      gadget.log.warning("jetlink: the far end powers the USB-C port but came out the device; asking it over USB PD "
                         "to take the host role (%d of %d)", self.asks, SWAP_TRIES)
      run_script('device')
    elif self.asks == SWAP_TRIES + 1:
      gadget.log.warning("jetlink: the far end did not take the host role; resetting USB PD")
      run_script('reset')
    else:
      gadget.log.warning("jetlink: the far end still powers the USB-C port as the device; leaving it until the next "
                         "plug. A USB-C hub, or a USB-C to USB-A adapter and an A-to-C cable, settles the roles")
      self.settled = True

  def _hold(self, lasted: float) -> None:
    gadget.log.warning(f"jetlink: hosting something that is not a chestnut for {lasted:.0f} s on the USB-C port; holding it as a device")
    # settled stays set if the hold did not happen, so this plug is not tried every cycle
    self.held = run_script('hold')
    self.settled = True

  def _release(self) -> None:
    if self.settled:
      # a sink-only accessory: it comes back as a sink on dual role, and
      # holding again would only cycle it
      gadget.log.warning("jetlink: no host came back on the USB-C port; leaving it dual role until the next plug")
    else:
      gadget.log.warning("jetlink: the USB-C port is empty; back to dual role")
    self.held = False
    run_script('off')

  def off(self) -> None:
    """Undo a hold. Outside one the port is already as AGNOS boots it. A
    device side turned on here is left: this runs as a chestnut turns up, and
    turning it off would take the host role from under the chestnut; the next
    owner turns it off once the port is empty."""
    if self.held:
      run_script('off')
    self._reset()
