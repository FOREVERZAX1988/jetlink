"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The link over Wi-Fi: the comma joins the device's hotspot (a phone's, a
Jetson's) and dials its default gateway, which is that device. Only the
gateway: nothing else on a network is ever dialed, so the hotspot's password,
which the person typed into the comma, is what the link trusts. No gadget is
built and the USB-C port is left alone, so ADB keeps working.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path

ROUTES = Path('/proc/net/route')
INTERFACE = 'wlan0'
# a hotspot answers a connect in milliseconds, and one with nothing listening
# refuses at once; this only bounds a gateway that drops the SYN
DIAL_TIMEOUT = 1.0
# between dials while nothing answers: the app may not be open yet
DIAL_DELAY = 2.0

_RTF_UP = 0x1
_RTF_GATEWAY = 0x2


def gateway(interface: str = INTERFACE, routes: Path = ROUTES) -> str | None:
  """The default gateway on `interface`, or None when the comma is on no
  Wi-Fi. The modem's default route is never it."""
  try:
    lines = routes.read_text().splitlines()[1:]
  except OSError:
    return None
  for line in lines:
    f = line.split()
    # Iface Destination Gateway Flags ..., addresses in host (little-endian) order
    if len(f) < 4 or f[0] != interface or f[1] != '00000000':
      continue
    flags = int(f[3], 16)
    if flags & _RTF_UP and flags & _RTF_GATEWAY:
      return socket.inet_ntoa(struct.pack('<I', int(f[2], 16)))
  return None


def band(interface: str = INTERFACE) -> str | None:
  """The band the comma is on, '2.4', '5' or '6' (GHz), from the frequency
  iwconfig reports; None when it cannot say. Read once a dial: 2.4 GHz is
  too slow for a big model's frames, and the apps say so."""
  tool = shutil.which('iwconfig') or '/usr/sbin/iwconfig'
  try:
    out = subprocess.run([tool, interface], capture_output=True, text=True, timeout=1.0).stdout
  except (OSError, subprocess.SubprocessError):
    return None
  m = re.search(r'Frequency[:=]\s*([\d.]+)\s*GHz', out)
  if m is None:
    return None
  ghz = float(m.group(1))
  return '2.4' if ghz < 3.0 else '5' if ghz < 5.925 else '6'


def link_info(interface: str = INTERFACE) -> dict:
  """What the comma's hello says of a Wi-Fi link (Transport.link_info)."""
  b = band(interface)
  return {'kind': 'wifi', **({'band': b} if b else {})}


# -- presence ----------------------------------------------------------------
#
# Wi-Fi is the one link the far end cannot see before it opens: the comma dials
# the hotspot's gateway, and until it does, the device that runs the app has
# nothing to go on but its own setting. A comma on another network is on its
# way in that reading and never arrives. So the comma says it is here: one
# datagram to the gateway every PRESENCE_PERIOD while the setting is Wi-Fi, and
# the app answers it (presence.PresenceMonitor in the Android app). The app
# calls the link connecting while a comma answers, and the comma knows the app
# is listening by that same answer: neither believes the other is there without
# having heard it.
#
# The gateway alone, never a broadcast: that is the device the comma joined the
# hotspot through, and an announcement that left the hotspot for another
# network is not this comma's link. Nothing here carries a frame, and a comma
# on no Wi-Fi sends nothing at all.
#
# The announcement names that gateway too (its `gw`), and the app reads the
# link as on its way only when the address is one of its own: a comma that
# joined another hotspot in the same subnet dials that one's device, not this
# phone, and is not the comma this phone is listening for.
PRESENCE_PORT = 5600
PRESENCE_PERIOD = 1.0
# how long an answer counts for. The app answers every announcement, so this
# covers only one that was dropped; the app holds a little longer still, since
# a comma that has stopped announcing is one it should stop saying yes to.
PRESENCE_HOLD = 5.0
PRESENCE_VERSION = 1
PRESENCE_KIND = 'presence'
# an announcement and the answer to it, well inside one datagram
PRESENCE_DATAGRAM = 512


class Presence:
  """The comma's end of the presence handshake (PRESENCE_PORT above).

  One socket for the life of a Wi-Fi setting, written and read without
  blocking, so the owner announces the comma in the step it already runs.
  Every announcement carries a fresh id and only an answer to the last one
  counts, so an answer that arrives late, or to an announcement from another
  comma, cannot hold the handshake up.
  """

  def __init__(self, port: int = PRESENCE_PORT, period: float = PRESENCE_PERIOD, clock=time.monotonic):
    self.port = port
    self.period = period
    self.clock = clock
    self.sock: socket.socket | None = None
    self.id = ''
    self.last_announce = 0.0
    self.answered_at = 0.0

  def handshake(self, gateway: str | None) -> bool:
    """Announce this comma to `gateway` and read back the app's answer.

    Called once a step: at most one announcement each `period`, and everything
    waiting on the socket read without blocking. True while the app has
    answered the last announcement within PRESENCE_HOLD; False with no gateway,
    which is the comma on no Wi-Fi, and then nothing is sent.
    """
    if gateway is None:
      self.close()
      return False
    if self.sock is None and not self._open():
      return False
    now = self.clock()
    if now - self.last_announce >= self.period:
      self.last_announce = now
      self.id = os.urandom(4).hex()
      self._announce(gateway)
    self._listen()
    return now - self.answered_at < PRESENCE_HOLD

  def close(self) -> None:
    """Let the socket go, and the answer with it: no Wi-Fi, no handshake."""
    sock, self.sock = self.sock, None
    self.id = ''
    self.answered_at = 0.0
    self.last_announce = 0.0
    if sock is not None:
      try:
        sock.close()
      except OSError:
        pass

  # -- the socket ------------------------------------------------------------

  def _open(self) -> bool:
    """A socket to send from, or nothing this step: the next one tries again."""
    try:
      sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      sock.setblocking(False)
    except OSError:
      return False
    self.sock = sock
    return True

  def _announce(self, gateway: str) -> None:
    """One datagram at the gateway, naming it: the app compares `gw` with its
    own addresses, and answers - reads the link as on its way - only when this
    comma dialed it. A send that cannot go takes the socket with it, and the
    next step opens another: the interface may come back."""
    try:
      self.sock.sendto(_message(id=self.id, gw=gateway), (gateway, self.port))
    except OSError:
      self.close()

  def _listen(self) -> None:
    """Every answer waiting on the socket, and nothing else."""
    while True:
      try:
        data, _ = self.sock.recvfrom(PRESENCE_DATAGRAM)
      except (BlockingIOError, InterruptedError):
        return
      except OSError:
        self.close()
        return
      if _answered(data, self.id):
        self.answered_at = self.clock()


def _message(**fields) -> bytes:
  """One presence datagram: what it is, its version, and the fields."""
  return json.dumps({'jetlink': PRESENCE_KIND, 'v': PRESENCE_VERSION, **fields},
                    separators=(',', ':')).encode()


def _answered(data: bytes, announced: str) -> bool:
  """Whether `data` is the app's answer to the announcement `announced`: the
  same id back with ack set. Nothing else counts as an answer."""
  try:
    message = json.loads(data)
  except (ValueError, UnicodeDecodeError):
    return False
  return (isinstance(message, dict) and message.get('jetlink') == PRESENCE_KIND
          and message.get('v') == PRESENCE_VERSION and message.get('ack') is True
          and bool(announced) and message.get('id') == announced)
