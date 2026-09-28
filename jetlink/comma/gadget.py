"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's USB gadget, and how to look at it, using nothing but the
standard library and jetlink's own transport.

Kept apart from openpilot so the process that owns the gadget can be small.
Holding ep0 needs sysfs, a few params and a unix socket;
`openpilot.common.swaglog` costs 28 MB because it drags numpy, capnp and zmq
in to publish a log line, and `openpilot.common.params` imports swaglog, so a
module that touches either prices the owner out of being minimal. Measured on
the comma: python plus this plus the FunctionFS transport is 10.4 MB against
47.5 MB for the daemon that imported the world.

Nothing in jetlink.comma may import openpilot; openpilot's params are read
here as files, by name. tests/test_comma_gadget.py holds the line.

The fork's heavy processes import it directly, and its helpers module points
`log` at cloudlog (set_logger), so their lines still reach swaglog while the
owner's go to a file.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from jetlink.comma import root
from jetlink.transport import ffs
from jetlink.transport.base import UDC_SYSFS, udc_speed
from jetlink.transport.tcp import CABLE_ADDRESS, DEFAULT_PORT

# -- logging --------------------------------------------------------------
# a module-level indirection rather than an import: the owner has no swaglog
# and must not grow one, and every other process wants its lines in the drive.
log = logging.getLogger('jetlink.comma.gadget')


def set_logger(logger) -> None:
  """Send this module's lines somewhere else, and the root script's failures
  with them; the fork's helpers points it at cloudlog."""
  global log
  log = logger
  root.log = logger


# -- params ---------------------------------------------------------------
# openpilot's params, read straight off the filesystem. params.cc writes a
# value to a temp file, fsyncs it, renames it over the key and fsyncs the
# directory, so a plain read gets the old value or the new one and never a torn
# one. The path rule is params.cc's: PARAMS_ROOT or /data/params, plus "/" and
# OPENPILOT_PREFIX, which defaults to "d".
#
# Every key the comma layer reads is named here and nowhere else in it, and
# none is written. openpilot declares them all (params_keys.h).
P_READY = "JetlinkEngineReady"      # sha256 of the model the Jetson has built
P_SPEC = "JetlinkSpec"              # the spec a provisioning run recorded; the owner only stats it
P_LINK = "JetlinkLink"              # Accelerator Link, an index into LINK_MODES
P_OFFROAD = "IsOffroad"             # manager's: is the car parked
P_BIG_MODEL = "ModelManager_ActiveBundleChestnut"  # the model manager's big-model pick
LINK_MODES = ('off', 'usb', 'ios')  # off; a Jetson or a Mac on USB; an iPhone on the cable


_dirs: dict[tuple[str, str], Path] = {}


def params_dir() -> Path:
  """Where the params live, by params.cc's rule. Memoised on the two variables
  it depends on: this is on the path of every param read in the process."""
  prefix = os.environ.get('OPENPILOT_PREFIX', 'd')
  base = os.environ.get('PARAMS_ROOT', '')
  key = (base, prefix)
  found = _dirs.get(key)
  if found is None:
    # hw.h: PARAMS_ROOT, else /data/params on device. comma_home carries the
    # prefix off-device, so a bench under its own store lands where Params does
    home = base or ('/data/params' if root.AGNOS
                    else os.path.join(os.path.expanduser('~'),
                                      '.comma' + ('' if prefix == 'd' else prefix), 'params'))
    found = _dirs[key] = Path(home) / prefix
  return found


def raw_param(key: str) -> bytes | None:
  """A param's bytes, or None if it is unset or unreadable."""
  try:
    return (params_dir() / key).read_bytes()
  except OSError:
    return None


def param_bool(key: str) -> bool | None:
  """A param openpilot stores with put_bool. None when it is unset."""
  value = raw_param(key)
  if value is None:
    return None
  return value.strip() in (b'1', b'true', b'True')


def link_mode() -> str:
  """Accelerator Link: 'off', 'usb' or 'ios'. Unset or unreadable is 'off'.
  manager writes the default before anything runs, and the fork's params
  migration carries the old on/off switch over."""
  raw = raw_param(P_LINK)
  try:
    return LINK_MODES[int(raw)]
  except (TypeError, ValueError, IndexError):
    return 'off'


def enabled() -> bool:
  """Is the link on, for either host? Not "absent means auto": the gadget comes
  up at boot with the package installed, so auto turned installation into
  enablement."""
  return link_mode() != 'off'


def ios() -> bool:
  """Is the link set to iOS, an iPhone on the cable?"""
  return link_mode() == 'ios'



def offroad() -> bool:
  """Is the car parked?

  The owner runs onroad too, to keep hold of the gadget, and everything else
  jetlink does belongs to a parked car: a download, an upload, an engine build.
  A missing param is manager not having written one yet, which reads as parked.
  """
  value = param_bool(P_OFFROAD)
  return True if value is None else value


# -- what carries the link ------------------------------------------------
# The Accelerator Link setting names the host: USB (a Jetson or a Mac on the
# FunctionFS vendor interface) or iOS (an iPhone, which gives apps no USB
# access). For iOS the gadget is composite, with a CDC-NCM network interface
# whose comma end is 192.168.60.1 (jetlink-root.sh gadget --ios, and net, which
# also runs the DHCP server), and the phone dials CABLE_ADDR whenever its USB
# ethernet is up.
# The owner hands the accepted socket to whoever borrows the link; nothing
# writes to the endpoint files, which a phone never reads. The setting, not a
# guess, says which: a hello over FunctionFS to a phone blocks 15 s and bounces
# the gadget, and waiting to see whether a phone dials cost every Jetson
# reconnect 5 to 10 s.
LINK = Path("/dev/shm/jetlink-link")        # "cable <peer ip>" while a phone is dialed in
NET_STATUS = Path("/dev/shm/jetlink-net")   # jetlink-root.sh: "ok 192.168.60.1 <netdev>", "error: ...", "net: off" (USB)
CABLE_ADDR = (CABLE_ADDRESS, DEFAULT_PORT)


def _link_record() -> list[str]:
  try:
    return LINK.read_text().split()
  except OSError:
    return []


def link_kind() -> str:
  """The gadget the owner built and published: 'cable' for iOS (the phone's
  network interface on the gadget) or 'usb'. The setting stands in only until
  the owner has said: it may have moved and be waiting for the car to park."""
  record = _link_record()
  if record[:1] in (['cable'], ['usb']):
    return record[0]
  return 'cable' if ios() else 'usb'


def link_peer() -> str | None:
  """The phone's address, while a cable link is up."""
  record = _link_record()
  return record[1] if record[:1] == ['cable'] and len(record) > 1 else None


def note_link(kind: str, peer: str | None = None) -> None:
  """The owner's record of the gadget it built, 'usb' or 'cable', and on the
  cable the phone that dialed in; see link_kind and link_peer."""
  try:
    LINK.write_text(f"{kind} {peer}".strip() if peer else kind)
  except OSError:
    log.exception("jetlink: could not record the link")


def clear_link() -> None:
  try:
    LINK.unlink(missing_ok=True)
  except OSError:
    log.exception("jetlink: could not clear the link record")



def net_status() -> str | None:
  """What jetlink-root.sh said about the gadget's network interface, if it ran."""
  try:
    return NET_STATUS.read_text().strip() or None
  except OSError:
    return None


def usb_speed() -> str | None:
  """The bus speed the host enumerated us at: 458 KB a frame is ~1 ms on
  super-speed and ~11 ms on high-speed, so this is the first thing to read
  when the link is slow. None while unbound or where the UDC does not say."""
  udc = bound_udc()
  return None if udc is None else udc_speed(udc, str(UDC_PATH))


# -- where the gadget lives -----------------------------------------------
# the comma is the USB gadget and the Jetson the host, decided by the kernels:
# AGNOS has CONFIG_USB_F_FS built in, L4T images are often stripped of the
# gadget modules. See docs/transport.md
GADGET_PATH = Path(ffs.GADGET)
FFS_MOUNT = Path(ffs.MOUNT)
UDC_PATH = Path(UDC_SYSFS)
# the network function jetlink-root.sh gadget --ios adds after ffs.jetlink. Its
# netdev is not usb0, which the modem holds, but whatever the kernel names it
NET_FUNCTION = 'ncm.usb0'
# written by scripts/comma/jetlink-root.sh gadget, which the owner runs when the
# link is on and there is no gadget: "ok", or "error: <reason>"
GADGET_STATUS = Path("/dev/shm/jetlink-gadget")
CC_ORIENTATION = Path('/sys/class/power_supply/usb/typec_cc_orientation')
# the owner's pid while it has released the gadget on purpose so the Jetson can
# sleep. Presence comes from this, not the UDC; a marker whose writer is dead is
# a leftover from a kill
DORMANT = Path("/dev/shm/jetlink-dormant")
# hardwared's request to power the Jetson off; see backend.shutdown
SHUTDOWN_REQUEST = Path("/dev/shm/jetlink-shutdown")
# what a provisioning run leaves for the owner: whether the far end suspends
# when the gadget goes, and whether the run left anything undone. The owner
# never speaks the protocol, so it cannot learn either for itself
STATE = Path("/dev/shm/jetlink-owner-state")


def owner_state() -> dict:
  """What jetlinkd's runs left for the owner (see Jetlinkd.note_state), or {}."""
  try:
    value = json.loads(STATE.read_text())
  except (OSError, ValueError):
    return {}
  return value if isinstance(value, dict) else {}


def far_end_sleeps(state: dict | None = None) -> bool:
  """Does the far end suspend when the gadget goes, as the runs recorded it?
  No record means it does: letting go of one that does not only costs a rebind."""
  return (owner_state() if state is None else state).get('sleep_after', 1.0) > 0


def gadget_error() -> str | None:
  """Why the USB gadget is unavailable, if it is.

  The gadget is set up by root from the owner, nowhere a user would look. A
  missing file is not an error: the setup never ran.
  """
  try:
    reason = GADGET_STATUS.read_text().strip()
  except OSError:
    return None
  if not reason or reason == 'ok':
    return None
  return reason.removeprefix('error:').strip() or None


def bound_udc() -> str | None:
  """The device controller our gadget is attached to, if it is attached."""
  try:
    return (GADGET_PATH / "UDC").read_text().strip() or None
  except OSError:
    return None


def udc_state() -> str | None:
  """What the device controller says about the bus, or None if we are unbound.

  "configured" is a host that has us; "default" and "addressed" are one that
  reset the bus and stopped part way, which is what a Jetson that took the
  bind as a wake and did not finish waking looks like.
  """
  udc = bound_udc()
  if udc is None:
    return None
  try:
    return (UDC_PATH / udc / "state").read_text().strip() or None
  except OSError:
    return None


def host_attached() -> bool:
  """Has a host (the Jetson) enumerated and configured us?"""
  return udc_state() == "configured"


def port_has_host() -> bool:
  """Does the USB-C port controller see a host on the cable?

  The CC pin, so it is electrically true whether or not anything enumerated:
  0 is a port with nothing on it, 1 or 2 a cable with a live host. A legacy
  A-to-C cable's pull-up rides on the host's VBUS and reads the same.
  """
  try:
    return int(CC_ORIENTATION.read_text()) != 0
  except (OSError, ValueError):
    return False


# how long the UDC may sit half enumerated with a host on the cable before the
# gadget is bounced. A real enumeration is milliseconds; this only fires for a
# host that answered the bind with a bus reset and then stopped, which is what
# an unarmed hub does to a box asleep. See FfsTransport.rebind
STALLED_ENUMERATION = 20.0
STALLED_STATES = ('default', 'addressed')
HOST_POLL = 0.5


def wait_for_host(timeout: float, bounce=None, should_stop=None, report=None) -> bool:
  """Wait for the Jetson to enumerate us, bouncing a bus that stalled.

  The gadget stays bound throughout. An unbind is an unplug as the far end sees
  it, and while one boots it takes ~50 s a cycle: doing that on a timer landed
  an unplug on a box that was seconds from enumerating.

  The one case that needs an edge is a host that took the bind as a wake, reset
  the bus and stopped. The UDC then sits in default or addressed with the CC
  pin still showing a host, and only another connect moves it: that is what
  `bounce` is for, and it is spent once.

  On the cable there is nothing to wait for: the connect that made the client
  already reached the phone. The UDC is configured too, but by the phone, and
  it is the dial that proved it is there.
  """
  if link_kind() == 'cable':
    return True
  deadline = time.monotonic() + timeout
  stalled_since = None
  bounced = False
  reported = False
  while True:
    state = udc_state()
    if state == "configured":
      return True
    now = time.monotonic()
    if now >= deadline or (should_stop is not None and should_stop()):
      return False
    if report is not None and not reported:
      reported = True
      report()
    if state in STALLED_STATES and port_has_host():
      stalled_since = now if stalled_since is None else stalled_since
      if not bounced and bounce is not None and now - stalled_since > STALLED_ENUMERATION:
        bounced = True
        log.warning("jetlink: the bus has been half enumerated for %.0f s, bouncing the gadget",
                    STALLED_ENUMERATION)
        try:
          bounce()
        except Exception:
          log.exception("jetlink: could not bounce the gadget")
    else:
      stalled_since = None
    time.sleep(HOST_POLL)


def setup_gadget(ios: bool) -> bool:
  """Create the gadget, for USB or iOS: there is none yet, or the setting
  moved between the two. The script records "ok" or the reason itself, in
  GADGET_STATUS, so a failure here reaches the offroad alert. Off AGNOS
  there is nothing to create it with, and this is a False."""
  if not root.run('gadget', *(['--ios'] if ios else [])):
    return False
  log.warning("jetlink: gadget set up for %s", 'iOS' if ios else 'USB')
  return link_configured()


def built_for_ios() -> bool:
  """Does the gadget carry the network function, as jetlink-root.sh gadget --ios builds it?"""
  return os.path.lexists(GADGET_PATH / 'configs' / 'c.1' / NET_FUNCTION)


def net_up() -> bool:
  """Bring the gadget's network interface up, for a phone to dial over.

  The netdev does not exist until the first UDC bind (f_ncm registers it in
  its bind), and the gadget subcommand never binds, so the owner runs this
  after it binds. `net` is idempotent: nmcli unmanaged, 192.168.60.1/24, the
  DHCP server, and NET_STATUS written as "ok 192.168.60.1 <netdev>" or
  "error: <reason>".
  """
  if not root.run('net'):
    return False
  status = net_status() or ''
  log.warning("jetlink: gadget network: %s", status or 'no status written')
  return status.startswith('ok')


def link_configured() -> bool:
  """Can we even attempt a link? The gadget exists.

  Not host_attached(): the UDC only binds when something opens ep0, and nothing
  opens ep0 unless the link looks usable. Waiting for a host deadlocks.

  The cable too: a phone is on the gadget's own network interface, which
  exists only while ep0 is held.
  """
  if gadget_error() is not None:
    return False
  try:
    return (FFS_MOUNT / "ep0").exists()
  except OSError:
    # a root-only mount raises PermissionError from stat; unusable either way
    return False


def set_dormant(on: bool) -> None:
  try:
    if on:
      DORMANT.write_text(str(os.getpid()))
    else:
      DORMANT.unlink(missing_ok=True)
  except OSError:
    log.exception("jetlink: could not update the dormant marker")


def dormant() -> bool:
  """Has a live owner released the gadget on purpose?"""
  try:
    pid = int(DORMANT.read_text())
  except (OSError, ValueError):
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    pass  # alive, just not ours to signal
  return True


def request_shutdown(reason: str) -> bool:
  try:
    SHUTDOWN_REQUEST.write_text(json.dumps({'reason': reason}))
    return True
  except OSError:
    log.exception("jetlink: could not write the shutdown request")
    return False


def pending_shutdown() -> str | None:
  """The reason in a shutdown request that has not been dealt with, if any.

  Read twice a second for the life of the process and almost never there, so
  the miss is a stat rather than an open that raises.
  """
  if not SHUTDOWN_REQUEST.exists():
    return None
  try:
    return str(json.loads(SHUTDOWN_REQUEST.read_text()).get('reason', ''))
  except (OSError, ValueError):
    return None


def finish_shutdown() -> None:
  try:
    SHUTDOWN_REQUEST.unlink(missing_ok=True)
  except OSError:
    log.exception("jetlink: could not remove the shutdown request")
