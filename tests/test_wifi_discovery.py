"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Finding a Wi-Fi server: the announcement first, then the gateway, then a sweep
of the subnet, and never anything off the interface's own link.
"""
import ipaddress
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import wifi

HEADER = 'Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n'
# the bench comma's: Wi-Fi at 192.168.1.254 (metric 600), the modem's default (metric 1000)
ROUTES = HEADER + ('wlan0\t00000000\tFE01A8C0\t0003\t0\t0\t600\t00000000\t0\t0\t0\n'
                   'ppp0\t00000000\t40404040\t0003\t0\t0\t1000\t00000000\t0\t0\t0\n'
                   'wlan0\t0001A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0\n')

# avahi-browse -rtp, a resolved record and an unresolved one
BROWSE = ('+;wlan0;IPv4;comma-jetlink;_jetlink._tcp;local\n'
          '=;wlan0;IPv4;comma-jetlink;_jetlink._tcp;local;ipad.local;172.20.10.1;5599;"version=1"\n'
          '=;wlan0;IPv4;other;_airplay._tcp;local;x.local;172.20.10.9;7000;\n'
          '=;wlan0;IPv6;comma-jetlink;_jetlink._tcp;local;ipad.local;fe80::1;5599;\n'
          '=;wlan0;IPv4;pending;_jetlink._tcp;local;ipad.local;;5599;\n')


class TestParseBrowse(unittest.TestCase):
  def test_it_takes_resolved_ipv4_records_for_our_service_only(self):
    self.assertEqual(wifi.parse_browse(BROWSE), [('comma-jetlink', '172.20.10.1')])

  def test_another_service_is_not_ours(self):
    text = '=;wlan0;IPv4;x;_http._tcp;local;h;10.0.0.1;80;\n'
    self.assertEqual(wifi.parse_browse(text), [])

  def test_an_unresolved_or_ipv6_record_has_no_address_to_dial(self):
    # '+' is an announcement with no address yet; the IPv6 one is undialable
    self.assertEqual(wifi.parse_browse('+;wlan0;IPv4;comma-jetlink;_jetlink._tcp;local\n'), [])
    text = '=;wlan0;IPv6;comma-jetlink;_jetlink._tcp;local;h;fe80::1;5599;\n'
    self.assertEqual(wifi.parse_browse(text), [])

  def test_junk_lines_do_not_raise(self):
    self.assertEqual(wifi.parse_browse('not a record\n=;too;few\n\n'), [])

  def test_the_parse_is_bounded(self):
    one = '=;wlan0;IPv4;n%d;_jetlink._tcp;local;h;10.0.0.%d;5599;\n'
    text = ''.join(one % (i, i) for i in range(1, 200))
    self.assertEqual(len(wifi.parse_browse(text)), wifi.BROWSE_MAX)


class TestBrowse(unittest.TestCase):
  def test_an_absent_tool_is_an_empty_list_not_a_failure(self):
    with mock.patch('subprocess.run', side_effect=FileNotFoundError):
      self.assertEqual(wifi.browse(), [])

  def test_a_timeout_is_an_empty_list(self):
    import subprocess
    with mock.patch('subprocess.run', side_effect=subprocess.TimeoutExpired('avahi-browse', 1)):
      self.assertEqual(wifi.browse(), [])

  def test_the_daemon_is_asked_for_our_service_and_told_to_terminate(self):
    done = mock.Mock(stdout=BROWSE)
    with mock.patch('subprocess.run', return_value=done) as run:
      got = wifi.browse()
    self.assertEqual(got, [('comma-jetlink', '172.20.10.1')])
    argv = run.call_args.args[0]
    self.assertEqual(argv[0], wifi.BROWSE_BIN)
    self.assertIn('--terminate', argv)
    self.assertIn(wifi.MDNS_SERVICE, argv)
    self.assertEqual(run.call_args.kwargs['timeout'], wifi.BROWSE_TIMEOUT)


class TestDiscoverOrder(unittest.TestCase):
  """discover()'s order: the announcement, the gateway, then the sweep."""

  def setUp(self):
    # no remembered peer unless a test wants one
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.peer = Path(tmp.name) / 'peer'
    patcher = mock.patch.object(wifi, 'PEER_FILE', self.peer)
    patcher.start()
    self.addCleanup(patcher.stop)

  def routes(self, text: str) -> Path:
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    path = Path(tmp.name) / 'route'
    path.write_text(text)
    return path

  def test_an_announced_server_is_dialed_before_the_gateway(self):
    probed = []
    with mock.patch.object(wifi, '_probe', side_effect=lambda h, p, t: probed.append(h) or h == '172.20.10.1'), \
         mock.patch.object(wifi, 'gateway') as gw, \
         mock.patch.object(wifi, 'local_addr', return_value=('172.20.10.44', 24)):
      got = wifi.discover(mdnss=[('comma-jetlink', '172.20.10.1')])
    self.assertEqual(got, '172.20.10.1')
    gw.assert_not_called()             # the announcement answered first
    self.assertEqual(probed, ['172.20.10.1'])

  def test_a_stale_announcement_falls_through_to_the_gateway(self):
    with mock.patch.object(wifi, '_probe', side_effect=lambda h, p, t: h == '192.168.1.254'), \
         mock.patch.object(wifi, 'gateway', return_value='192.168.1.254'), \
         mock.patch.object(wifi, 'local_addr', return_value=('192.168.1.44', 24)):
      got = wifi.discover(mdnss=[('comma-jetlink', '172.20.10.1')])
    self.assertEqual(got, '192.168.1.254')

  def test_a_skipped_host_is_not_tried_again(self):
    with mock.patch.object(wifi, '_probe', side_effect=lambda h, p, t: h == '192.168.1.254'), \
         mock.patch.object(wifi, 'gateway', return_value='192.168.1.254'), \
         mock.patch.object(wifi, 'local_addr', return_value=('192.168.1.44', 24)):
      got = wifi.discover(mdnss=[('comma-jetlink', '192.168.1.254')], skip={'192.168.1.254'})
    self.assertIsNone(got)

  def test_a_remembered_peer_short_circuits_everything(self):
    with mock.patch.object(wifi, '_probe', return_value=True), \
         mock.patch.object(wifi, 'browse') as browse, \
         mock.patch.object(wifi, 'gateway') as gw:
      wifi._save_peer('10.0.0.9', 5599)
      got = wifi.discover()
    self.assertEqual(got, '10.0.0.9')
    browse.assert_not_called()
    gw.assert_not_called()

  def test_the_sweep_stays_inside_the_interfaces_own_subnet(self):
    probed = []

    def probe(host, port, timeout):
      probed.append(host)
      return host == '192.168.1.77'

    with mock.patch.object(wifi, '_probe', side_effect=probe), \
         mock.patch.object(wifi, 'gateway', return_value=None), \
         mock.patch.object(wifi, 'local_addr', return_value=('192.168.1.44', 24)):
      got = wifi.discover(mdnss=[])
    self.assertEqual(got, '192.168.1.77')
    net = ipaddress.ip_network('192.168.1.44/24', strict=False)
    for host in probed:
      self.assertIn(ipaddress.ip_address(host), net)
    self.assertNotIn('192.168.1.44', probed)      # never the comma itself

  def test_nothing_anywhere_is_none(self):
    with mock.patch.object(wifi, '_probe', return_value=False), \
         mock.patch.object(wifi, 'gateway', return_value='192.168.1.254'), \
         mock.patch.object(wifi, 'local_addr', return_value=('192.168.1.44', 30)):
      self.assertIsNone(wifi.discover(mdnss=[('x', '172.20.10.1')]))

  def test_off_wifi_with_no_address_there_is_nothing_to_sweep(self):
    with mock.patch.object(wifi, '_probe') as probe, \
         mock.patch.object(wifi, 'gateway', return_value=None), \
         mock.patch.object(wifi, 'local_addr', return_value=None):
      self.assertIsNone(wifi.discover(mdnss=[]))
    probe.assert_not_called()

  def test_a_found_host_is_remembered_for_the_next_join(self):
    with mock.patch.object(wifi, '_probe', side_effect=lambda h, p, t: h == '192.168.1.254'), \
         mock.patch.object(wifi, 'gateway', return_value='192.168.1.254'), \
         mock.patch.object(wifi, 'local_addr', return_value=('192.168.1.44', 24)):
      wifi.discover(mdnss=[])
    # the write is the point; _saved_peer would probe again, so read it here
    host, port, _ts = self.peer.read_text().strip().split(':')
    self.assertEqual((host, int(port)), ('192.168.1.254', 5599))


if __name__ == '__main__':
  unittest.main()
