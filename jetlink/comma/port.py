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
this port. So once the comma has been the host for a few seconds with no
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

USB PD is left alone; a host that comes back negotiates as over any cable.
The hold does not survive a reboot.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from jetlink.comma import gadget, root

POWER_ROLE = Path('/sys/class/usbpd/usbpd0/current_pr')
USB_DEVICES = Path('/sys/bus/usb/devices')
# how long the comma hosts the far end before judging it. A chestnut enumerates
# well inside this
SWAP_AFTER = 3.0
# how long the port reads empty before a hold is let go. The hold itself is an
# unplug and replug, so this has to outlast the replug
RELEASE_AFTER = 5.0
# how long the port reads empty before a plug counts as gone. A device let go
# by a hold reattaches in a DRP toggle and a CC debounce, well under this
UNPLUGGED = 1.5


def power_role() -> str | None:
  """'source', 'sink' or 'none' from the policy engine, or None without one."""
  try:
    return POWER_ROLE.read_text().strip() or None
  except OSError:
    return None


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
  """jetlink-root.sh port hold|off. Both commas have the lever, so a False is
  a failure, and root.run has logged why. Its timeout is short because the
  off in the owner's finally comes before the FunctionFS close, inside
  manager's 5 s."""
  return root.run('port', command, timeout=root.PORT_TIMEOUT)


class Port:
  """Called once a cycle by the owner. A cycle reads one sysfs file; sudo only
  runs on a change."""

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

  def update(self, now: float | None = None) -> None:
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
    if role != self.role:
      self.role, self.role_since = role, now
    lasted = now - self.role_since
    if self.held:
      if role == 'sink':
        self.settled = False   # a host came back
      elif role != 'source' and lasted >= RELEASE_AFTER:
        self._release()
        self.role_since = now  # the accessory reattaches in a moment; time the gap afresh
    elif role == 'source':
      if not self.settled and lasted >= SWAP_AFTER:
        if chestnut_attached(self.chestnut_ids):
          self.settled = True
        else:
          self._hold(lasted)
    elif role == 'sink' or lasted >= UNPLUGGED:
      self.settled = False     # a host, or the plug is gone

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
    """Undo a hold. Outside one the port is already as AGNOS boots it."""
    if self.held:
      run_script('off')
    self._reset()
