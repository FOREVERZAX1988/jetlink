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

import re
import shutil
import socket
import struct
import subprocess
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
