"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Which link a comma is on, USB 3, USB 2 or TCP, as the apps show it: what the
comma's hello says, what each transport sees from its own end, and the link
event the server publishes.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.server.cache import EngineCache
from jetlink.server.session import EngineHost, Session
from jetlink.transport import base
from jetlink.transport.base import link_medium, medium_from_usb_speed, udc_speed
from jetlink.transport.ffs import FfsTransport
from jetlink.transport.tcp import CABLE_ADDRESS, TcpTransport
from jetlink.transport.usbbulk import UsbBulkTransport
from tests.fake_backend import FakeBackend


@pytest.fixture
def udc(tmp_path, monkeypatch):
  """A fake /sys/class/udc with one controller; returns a setter for its speed."""
  controller = tmp_path / 'a600000.dwc3'
  controller.mkdir()
  monkeypatch.setattr(base, 'UDC_SYSFS', str(tmp_path))

  def speed(value: str) -> None:
    (controller / 'current_speed').write_text(value + '\n')
  speed('UNKNOWN')
  return speed


@pytest.mark.parametrize('speed,medium', [('super-speed-plus', 'usb3'), ('super-speed', 'usb3'),
                                          ('high-speed', 'usb2'), ('full-speed', 'usb1'),
                                          ('UNKNOWN', 'usb'), (None, 'usb')])
def test_a_usb_speed_names_its_generation(speed, medium):
  assert medium_from_usb_speed(speed) == medium


def test_a_hello_names_the_medium_or_nothing():
  assert link_medium({'kind': 'usb', 'usb_speed': 'high-speed'}) == 'usb2'
  assert link_medium({'kind': 'cable', 'usb_speed': 'super-speed'}) == 'usb3'
  assert link_medium({'kind': 'cable'}) == 'usb'
  assert link_medium({'kind': 'tcp'}) == 'tcp'
  assert link_medium({'kind': 'carrier pigeon'}) is None
  assert link_medium(None) is None
  assert link_medium('usb') is None


def test_the_controller_speed_is_read_until_a_host_configures_it(udc):
  assert udc_speed() is None
  udc('super-speed')
  assert udc_speed() == 'super-speed'
  assert udc_speed('a600000.dwc3') == 'super-speed'
  assert udc_speed('no-such-controller') is None


def test_the_gadget_says_usb_and_its_speed(udc):
  udc('high-speed')
  t = FfsTransport.__new__(FfsTransport)
  t.bound_udc = 'a600000.dwc3'
  assert t.link_info() == {'kind': 'usb', 'usb_speed': 'high-speed'}
  assert t.medium == 'usb2'


def test_a_phones_dial_is_the_cable_and_a_lan_is_tcp(udc):
  udc('super-speed')
  cable = TcpTransport.__new__(TcpTransport)
  cable.sock = SimpleNamespace(getsockname=lambda: (CABLE_ADDRESS, 5599))
  assert cable.link_info() == {'kind': 'cable', 'usb_speed': 'super-speed'}
  lan = TcpTransport.__new__(TcpTransport)
  lan.sock = SimpleNamespace(getsockname=lambda: ('10.0.0.5', 40000))
  assert lan.link_info() == {'kind': 'tcp'}
  assert lan.medium == 'tcp'


def test_a_usb_host_reads_the_speed_libusb_negotiated():
  t = UsbBulkTransport.__new__(UsbBulkTransport)
  t.usb_speed = 'super-speed'
  assert t.medium == 'usb3'
  assert t.link_info() == {'kind': 'usb', 'usb_speed': 'super-speed'}


class FakeTransport:
  def __init__(self, link):
    self.link = link
    self.sent = []

  def link_info(self):
    if isinstance(self.link, Exception):
      raise self.link
    return self.link

  def send_json(self, msg_type, seq, obj, flags=0):
    self.sent.append(obj)


@pytest.mark.parametrize('link,expected', [({'kind': 'cable', 'usb_speed': 'super-speed'}, {'kind': 'cable', 'usb_speed': 'super-speed'}),
                                           ({}, None), (OSError('no sysfs'), None)])
def test_the_hello_carries_the_link(link, expected):
  t = FakeTransport(link)
  client = JetlinkClient(t, name='modeld')
  client._expect = lambda *a, **k: SimpleNamespace(payload=memoryview(b'{}'))
  client.hello()
  assert t.sent[0]['client'].get('link') == expected
  assert t.sent[0]['client']['name'] == 'modeld'


def test_the_server_names_the_medium_the_hello_gives(tmp_path):
  events = []
  host = EngineHost(EngineCache(tmp_path, FakeBackend()))
  host.subscribe(lambda kind, payload: events.append((kind, payload)))
  transport = SimpleNamespace(send=lambda *a: None, peer='192.168.60.4:50000', medium='tcp')
  session = Session(transport, host)
  assert session.link_event() == {'state': 'connected', 'detail': '', 'peer': '192.168.60.4:50000', 'medium': 'tcp'}

  hello = {'client': {'name': 'modeld', 'nonce': '1', 'link': {'kind': 'cable', 'usb_speed': 'high-speed'}}}
  session.handle(SimpleNamespace(msg_type=P.Msg.HELLO_REQ, seq=1, payload=memoryview(json.dumps(hello).encode())))
  links = [p for k, p in events if k == 'link']
  assert links == [{'state': 'connected', 'detail': '', 'peer': '192.168.60.4:50000', 'medium': 'usb2'}]

  # the next process on the same link says the same: no second event
  session.handle(SimpleNamespace(msg_type=P.Msg.HELLO_REQ, seq=1, payload=memoryview(json.dumps(hello).encode())))
  assert len([k for k, _ in events if k == 'link']) == 1


def test_an_old_comma_leaves_the_medium_to_this_end(tmp_path):
  events = []
  host = EngineHost(EngineCache(tmp_path, FakeBackend()))
  host.subscribe(lambda kind, payload: events.append((kind, payload)))
  session = Session(SimpleNamespace(send=lambda *a: None, peer='usb', medium='usb3'), host)
  hello = {'client': {'name': 'modeld', 'nonce': '1'}}
  session.handle(SimpleNamespace(msg_type=P.Msg.HELLO_REQ, seq=1, payload=memoryview(json.dumps(hello).encode())))
  assert not [k for k, _ in events if k == 'link']
  assert session.link_event()['medium'] == 'usb3'
