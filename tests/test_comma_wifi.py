"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The link over Wi-Fi: the comma dials its Wi-Fi gateway and nothing else, and
modeld's link opens over that dial with no gadget to wait for.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget, wifi
from jetlink.openpilot import link
from tests.openpilot import fakes

HEADER = 'Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n'
# the bench comma's: Wi-Fi at 192.168.1.254 (metric 600), the modem's default (metric 1000)
ROUTES = HEADER + ('wlan0\t00000000\tFE01A8C0\t0003\t0\t0\t600\t00000000\t0\t0\t0\n'
                   'ppp0\t00000000\t40404040\t0003\t0\t0\t1000\t00000000\t0\t0\t0\n'
                   'wlan0\t0001A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0\n')


class TestGateway(unittest.TestCase):
  def routes(self, text: str) -> Path:
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    path = Path(tmp.name) / 'route'
    path.write_text(text)
    return path

  def test_the_wifi_default_route_is_the_gateway(self):
    self.assertEqual(wifi.gateway(routes=self.routes(ROUTES)), '192.168.1.254')

  def test_the_modems_default_route_is_never_it(self):
    only_lte = HEADER + 'ppp0\t00000000\t40404040\t0003\t0\t0\t1000\t00000000\t0\t0\t0\n'
    self.assertIsNone(wifi.gateway(routes=self.routes(only_lte)))

  def test_a_route_that_is_down_or_has_no_gateway_is_not_it(self):
    down = HEADER + 'wlan0\t00000000\tFE01A8C0\t0002\t0\t0\t600\t00000000\t0\t0\t0\n'
    self.assertIsNone(wifi.gateway(routes=self.routes(down)))
    local = HEADER + 'wlan0\t0001A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0\n'
    self.assertIsNone(wifi.gateway(routes=self.routes(local)))

  def test_no_routing_table_is_no_wifi(self):
    self.assertIsNone(wifi.gateway(routes=Path('/nonexistent/route')))


class TestBand(unittest.TestCase):
  def band(self, out: str):
    done = mock.Mock(stdout=out)
    with mock.patch('subprocess.run', return_value=done):
      return wifi.band()

  def test_the_frequency_names_the_band(self):
    # the bench comma at home, as iwconfig prints it
    self.assertEqual(self.band('wlan0  IEEE 802.11  ESSID:"="\n  Mode:Managed  Frequency:2.437 GHz  Access Point: 48:41:7B'), '2.4')
    self.assertEqual(self.band('  Mode:Managed  Frequency:5.24 GHz  Access Point: 48:41:7B:F7:6F:28'), '5')
    self.assertEqual(self.band('  Mode:Managed  Frequency:6.115 GHz'), '6')

  def test_no_frequency_or_no_tool_is_no_band(self):
    self.assertIsNone(self.band('wlan0  unassociated'))
    with mock.patch('subprocess.run', side_effect=FileNotFoundError):
      self.assertIsNone(wifi.band())

  def test_the_hello_says_wifi_and_the_band_it_knows(self):
    with mock.patch.object(wifi, 'band', return_value='2.4'):
      self.assertEqual(wifi.link_info(), {'kind': 'wifi', 'band': '2.4'})
    with mock.patch.object(wifi, 'band', return_value=None):
      self.assertEqual(wifi.link_info(), {'kind': 'wifi'})

  def test_a_tcp_link_says_what_its_opener_set(self):
    import socket
    from jetlink.transport.tcp import TcpTransport
    srv = socket.create_server(('127.0.0.1', 0))
    self.addCleanup(srv.close)
    a = socket.create_connection(srv.getsockname())
    b, _ = srv.accept()
    self.addCleanup(a.close)
    self.addCleanup(b.close)
    t = TcpTransport(a)
    self.assertEqual(t.link_info(), {'kind': 'tcp'})
    t.link = {'kind': 'wifi', 'band': '5'}
    self.assertEqual(t.link_info(), {'kind': 'wifi', 'band': '5'})


class TestDial(fakes.OpenpilotTest):
  def test_a_wifi_link_dials_the_gateway_and_says_so(self):
    client = mock.Mock(dead=False)
    with mock.patch.object(wifi, 'gateway', return_value='172.20.10.1'), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp', return_value=client) as open_tcp, \
         mock.patch.object(gadget, 'note_link') as note:
      got = link.Link(self.parts.log, wifi=True).open()
    self.assertIs(got, client)
    self.assertEqual(open_tcp.call_args.args, ('172.20.10.1',))
    self.assertEqual(open_tcp.call_args.kwargs['timeout'], wifi.DIAL_TIMEOUT)
    note.assert_called_once_with('wifi', '172.20.10.1')
    self.assertEqual(client.t.link['kind'], 'wifi')

  def test_closing_a_wifi_link_clears_the_record_the_panels_read(self):
    client = mock.Mock(dead=False)
    with mock.patch.object(wifi, 'gateway', return_value='172.20.10.1'), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp', return_value=client):
      wifi_link = link.Link(self.parts.log, wifi=True)
      wifi_link.open()
      self.assertEqual(gadget.link_state(), ('wifi', '172.20.10.1'))
      wifi_link.close()
    client.close.assert_called_once()
    self.assertEqual(gadget.link_state(), (None, None))

  def test_the_panels_say_wifi_and_the_gateway(self):
    from jetlink.openpilot import status
    self.assertEqual(status.link_transport('wifi'), 'Wi-Fi')
    gadget.note_link('wifi', '172.20.10.1')
    self.assertEqual(status.link_transport('wifi'), 'Wi-Fi (172.20.10.1)')

  def test_off_wifi_there_is_nothing_to_dial(self):
    with mock.patch.object(wifi, 'gateway', return_value=None), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp') as open_tcp:
      with self.assertRaisesRegex(link.WifiWaiting, 'not on Wi-Fi') as caught:
        link.Link(self.parts.log, wifi=True).open()
    self.assertEqual(caught.exception.waiting, link.NOT_ON_WIFI)
    self.assertEqual(caught.exception.retry_after, wifi.DIAL_DELAY)
    open_tcp.assert_not_called()

  def test_nothing_listening_is_a_link_error_the_join_retries(self):
    with mock.patch.object(wifi, 'gateway', return_value='172.20.10.1'), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp', side_effect=ConnectionRefusedError(61, 'refused')):
      with self.assertRaisesRegex(link.WifiWaiting, 'nothing answered at 172.20.10.1') as caught:
        link.Link(self.parts.log, wifi=True).open()
    self.assertEqual(caught.exception.waiting, link.NO_ANSWER)

  def test_waiting_is_handed_to_the_join_at_once(self):
    # not retried for CONNECT_TIMEOUT here: the join reports why and asks again
    with mock.patch.object(wifi, 'gateway', return_value=None), mock.patch('time.sleep') as slept:
      with self.assertRaises(link.WifiWaiting):
        link.connect_patiently(link.Link(self.parts.log, wifi=True))
    slept.assert_not_called()

  def test_no_lease_is_asked_for_and_no_gadget_waited_on(self):
    client = mock.Mock(dead=False)
    with mock.patch.object(wifi, 'gateway', return_value='172.20.10.1'), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp', return_value=client), \
         mock.patch.object(gadget, 'note_link'), \
         mock.patch('jetlink.comma.lending.borrow') as borrow, \
         mock.patch.object(gadget, 'wait_for_host') as wait:
      self.assertIs(link.connect_patiently(link.Link(self.parts.log, wifi=True)), client)
    borrow.assert_not_called()
    wait.assert_not_called()


if __name__ == '__main__':
  unittest.main()
