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

import socket
import struct
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
