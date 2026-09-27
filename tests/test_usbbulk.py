"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The host opener on the composite gadget: the link is the vendor-class
interface, wherever the descriptors put it, and not interface 0 by habit.
"""
import pytest

from jetlink.transport.base import LinkError
from jetlink.transport.usbbulk import _find_bulk_endpoints

BULK, INTERRUPT = 0x02, 0x03


class Endpoint:
  def __init__(self, address, attributes=BULK):
    self._address, self._attributes = address, attributes

  def getAddress(self):
    return self._address

  def getAttributes(self):
    return self._attributes


class Setting:
  def __init__(self, number, klass, subclass, protocol, endpoints):
    self._number, self._class, self._subclass, self._protocol = number, klass, subclass, protocol
    self._endpoints = endpoints

  def getNumber(self):
    return self._number

  def getClass(self):
    return self._class

  def getSubClass(self):
    return self._subclass

  def getProtocol(self):
    return self._protocol

  def __iter__(self):
    return iter(self._endpoints)


class Device:
  """Iterates like a usb1.USBDevice: configurations, interfaces, alternate settings."""

  def __init__(self, *interfaces):
    self._interfaces = interfaces

  def iterConfigurations(self):
    yield [list(settings) for settings in self._interfaces]


def ncm_interfaces():
  """f_ncm as the 4.9 kernel describes it: a CDC control interface with an
  interrupt endpoint, then a CDC Data interface whose alt 0 has no endpoints and
  alt 1 the bulk pair."""
  return (
    [Setting(0, 0x02, 0x0D, 0x00, [Endpoint(0x83, INTERRUPT)])],
    [Setting(1, 0x0A, 0x00, 0x01, []),
     Setting(1, 0x0A, 0x00, 0x01, [Endpoint(0x84), Endpoint(0x02)])],
  )


VENDOR = [Setting(2, 0xFF, 0xFF, 0xFF, [Endpoint(0x81), Endpoint(0x01)])]


def test_the_vendor_interface_is_chosen_over_the_network_ones():
  device = Device(*ncm_interfaces(), VENDOR)
  assert _find_bulk_endpoints(device) == (0x81, 0x01, 2)


def test_a_vendor_interface_at_zero_is_found_as_the_script_links_it():
  device = Device([Setting(0, 0xFF, 0xFF, 0xFF, [Endpoint(0x81), Endpoint(0x01)])], *ncm_interfaces())
  assert _find_bulk_endpoints(device) == (0x81, 0x01, 0)


def test_no_class_mark_falls_back_to_interface_zero():
  """A gadget from before the composite script, class 0 on a bare bulk pair."""
  device = Device([Setting(0, 0x00, 0x00, 0x00, [Endpoint(0x81), Endpoint(0x01)])])
  assert _find_bulk_endpoints(device) == (0x81, 0x01, 0)


def test_an_explicit_number_overrides_the_class_match():
  device = Device([Setting(0, 0x00, 0x00, 0x00, [Endpoint(0x85), Endpoint(0x05)])], *ncm_interfaces(),
                  [Setting(3, 0xFF, 0xFF, 0xFF, [Endpoint(0x81), Endpoint(0x01)])])
  assert _find_bulk_endpoints(device, 0) == (0x85, 0x05, 0)
  assert _find_bulk_endpoints(device, 3) == (0x81, 0x01, 3)


def test_a_data_interface_is_never_mistaken_for_the_link():
  """CDC Data has a bulk pair too; only the vendor class is the link."""
  device = Device(*ncm_interfaces())
  with pytest.raises(LinkError, match="no vendor interface"):
    _find_bulk_endpoints(device)
  with pytest.raises(LinkError, match="interface 5"):
    _find_bulk_endpoints(device, 5)
