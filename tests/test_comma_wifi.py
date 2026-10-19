"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The link over Wi-Fi: the comma dials its Wi-Fi gateway and nothing else, and
modeld's link opens over that dial with no gadget to wait for.
"""
import json
import socket
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

  def test_a_tcp_link_says_what_its_opener_set_and_undoes_it_on_close(self):
    from jetlink.transport.tcp import TcpTransport
    from tests.test_protocol import make_pair
    client, server = make_pair()
    self.addCleanup(server.close)
    self.assertEqual(client.link_info(), {'kind': 'tcp'})
    closed = mock.Mock()
    t = TcpTransport(client.sock, link={'kind': 'wifi', 'band': '5'}, on_close=closed)
    self.assertEqual(t.link_info(), {'kind': 'wifi', 'band': '5'})
    t.close()
    t.close()
    closed.assert_called_once()


class TestDial(fakes.OpenpilotTest):
  def setUp(self):
    super().setUp()
    # The mDNS browse is mocked here: a device that has avahi-browse would
    # really run it, which costs seconds and makes every test in this class
    # depend on what is on the network. The browse itself has its own tests.
    patcher = mock.patch.object(wifi, 'browse', return_value=[])
    patcher.start()
    self.addCleanup(patcher.stop)
    # The address on the interface, so a test does not depend on the machine
    # it runs on having a wlan0.
    patcher = mock.patch.object(wifi, 'local_addr', return_value=('192.168.7.100', 24))
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_a_wifi_link_dials_the_gateway_and_says_so(self):
    client = mock.Mock(dead=False)
    with mock.patch.object(wifi, 'gateway', return_value='172.20.10.1'), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp', return_value=client) as open_tcp, \
         mock.patch.object(gadget, 'note_link') as note:
      got = link.Link(self.parts.log, wifi=True).open()
    self.assertIs(got, client)
    self.assertEqual(open_tcp.call_args.args, ('172.20.10.1',))
    self.assertEqual(open_tcp.call_args.kwargs['timeout'], wifi.DIAL_TIMEOUT)
    self.assertEqual(open_tcp.call_args.kwargs['link']['kind'], 'wifi')
    # the panels' record of the dial goes with the link
    self.assertIs(open_tcp.call_args.kwargs['on_close'], gadget.clear_link)
    note.assert_called_once_with('wifi', '172.20.10.1')

  def test_the_panels_say_wifi_and_the_gateway(self):
    from jetlink.openpilot import status
    self.assertEqual(status.link_transport('wifi'), 'Wi-Fi')
    gadget.note_link('wifi', '172.20.10.1')
    self.assertEqual(status.link_transport('wifi'), 'Wi-Fi (172.20.10.1)')

  def test_off_wifi_there_is_nothing_to_dial(self):
    # no gateway and no address on wlan0: the comma is on no Wi-Fi at all, so
    # there is nothing to search either (a comma that does have an address
    # searches its subnet, which is another test's)
    with mock.patch.object(wifi, 'gateway', return_value=None), \
         mock.patch.object(wifi, 'local_addr', return_value=None), \
         mock.patch.object(wifi, 'discover') as discover, \
         mock.patch.object(wifi, 'browse') as browse, \
         mock.patch('jetlink.client.JetlinkClient.open_tcp') as open_tcp:
      with self.assertRaisesRegex(link.WifiWaiting, 'not on Wi-Fi') as caught:
        link.Link(self.parts.log, wifi=True).open()
    self.assertEqual(caught.exception.waiting, link.NOT_ON_WIFI)
    self.assertEqual(caught.exception.retry_after, wifi.DIAL_DELAY)
    open_tcp.assert_not_called()
    discover.assert_not_called()
    browse.assert_not_called()

  def test_off_wifi_nothing_is_browsed_for(self):
    # A browse costs avahi a couple of seconds, and a comma with no wlan0 and
    # no gateway has nothing to hear it on, so the check comes first.
    with mock.patch.object(wifi, 'gateway', return_value=None), \
         mock.patch.object(wifi, 'local_addr', return_value=None), \
         mock.patch.object(wifi, 'browse') as browse:
      with self.assertRaises(link.WifiWaiting):
        link.Link(self.parts.log, wifi=True).open()
    browse.assert_not_called()

  def test_an_announced_server_is_dialed_before_the_gateway(self):
    # The server that announces itself is the server: on a shared AP the
    # gateway is a router, and dialing it only spends a timeout. The gateway
    # is still read, because that read says whether there is Wi-Fi at all.
    client = mock.Mock(dead=False)
    with mock.patch.object(wifi, 'local_addr', return_value=('192.168.7.100', 24)), \
         mock.patch.object(wifi, 'browse', return_value=[('phone', '192.168.7.233')]), \
         mock.patch.object(wifi, 'gateway', return_value='192.168.7.1'), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp', return_value=client) as open_tcp:
      got = link.Link(self.parts.log, wifi=True).open()
    self.assertIs(got, client)
    self.assertEqual(open_tcp.call_args.args, ('192.168.7.233',))
    # the gateway was never dialed, only the server that announced itself
    open_tcp.assert_called_once()

  def test_nothing_listening_is_a_link_error_the_join_retries(self):
    with mock.patch.object(wifi, 'gateway', return_value='172.20.10.1'), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp', side_effect=ConnectionRefusedError(61, 'refused')):
      with self.assertRaisesRegex(link.WifiWaiting, 'nothing answered at 172.20.10.1') as caught:
        link.Link(self.parts.log, wifi=True).open()
    self.assertEqual(caught.exception.waiting, link.NO_ANSWER)

  def test_waiting_is_handed_to_the_join_at_once(self):
    # not retried for CONNECT_TIMEOUT here: the join reports why and asks again
    with mock.patch.object(wifi, 'gateway', return_value=None), \
         mock.patch.object(wifi, 'local_addr', return_value=None), \
         mock.patch('time.sleep') as slept:
      with self.assertRaises(link.WifiWaiting):
        link.connect_patiently(link.Link(self.parts.log, wifi=True))
    slept.assert_not_called()

  def test_no_lease_is_asked_for_and_no_gadget_waited_on(self):
    # the dial's own record says Wi-Fi, so the wait for a host returns at once
    client = mock.Mock(dead=False)
    with mock.patch.object(wifi, 'gateway', return_value='172.20.10.1'), \
         mock.patch('jetlink.client.JetlinkClient.open_tcp', return_value=client), \
         mock.patch('jetlink.comma.lending.borrow') as borrow, \
         mock.patch.object(gadget, 'udc_state') as udc:
      self.assertIs(link.connect_patiently(link.Link(self.parts.log, wifi=True)), client)
    borrow.assert_not_called()
    udc.assert_not_called()


class TestPresence(unittest.TestCase):
  """The handshake a comma on Wi-Fi sends to the device it joined, so that
  device's app reads the link as connecting only while there is a comma on the
  hotspot to connect to. The phone's half is presence.PresenceMonitor in the
  Android app; here a loopback socket is the phone."""

  def setUp(self):
    self.phone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.phone.bind(('127.0.0.1', 0))
    self.phone.settimeout(2.0)
    self.addCleanup(self.phone.close)
    # the clock the owner's step would move: nothing here waits on a real one
    self.now = [1000.0]
    # the loopback socket is the phone and its gateway at once: no interface of
    # this machine's is the network the handshake would broadcast on, so these
    # tests send to one target and nothing leaves the box
    self.presence = wifi.Presence(port=self.phone.getsockname()[1], clock=lambda: self.now[0], interface=None)
    self.addCleanup(self.presence.close)

  def announced(self) -> dict:
    """The announcement, as the phone read it, keeping where it came from: the
    answer goes back to the port the announcement came from, never to a port of
    its own."""
    data, self.caller = self.phone.recvfrom(wifi.PRESENCE_DATAGRAM)
    return json.loads(data)

  def answer(self, message: dict, **fields) -> None:
    """The app's answer: the announcement back with ack set."""
    self.phone.sendto(json.dumps({'jetlink': 'presence', 'v': 1, 'ack': True, **message, **fields}).encode(), self.caller)

  def test_the_announcement_names_the_protocol_and_carries_an_id(self):
    self.assertFalse(self.presence.handshake('127.0.0.1'))
    message = self.announced()
    self.assertEqual(message['jetlink'], 'presence')
    self.assertEqual(message['v'], wifi.PRESENCE_VERSION)
    self.assertTrue(message['id'])
    self.assertEqual(self.caller[0], '127.0.0.1')

  def test_the_announcement_names_the_gateway_it_dials(self):
    # the app reads the link as on its way only when this is one of its own
    self.presence.handshake('127.0.0.1')
    self.assertEqual(self.announced()['gw'], '127.0.0.1')

  def test_the_apps_answer_is_the_handshake(self):
    self.presence.handshake('127.0.0.1')
    self.answer(self.announced())
    # on the same step the answer arrived on, as the owner's next step is
    self.assertTrue(self.presence.handshake('127.0.0.1'))

  def test_only_the_same_id_answered_counts(self):
    self.presence.handshake('127.0.0.1')
    self.answer(self.announced(), id='deadbeef')
    self.assertFalse(self.presence.handshake('127.0.0.1'))

  def test_an_answer_stops_counting_after_the_hold(self):
    self.presence.handshake('127.0.0.1')
    self.answer(self.announced())
    self.assertTrue(self.presence.handshake('127.0.0.1'))
    self.now[0] += wifi.PRESENCE_HOLD - 0.1
    self.assertTrue(self.presence.handshake('127.0.0.1'), 'an answer inside the hold still counts')
    self.now[0] += wifi.PRESENCE_HOLD
    self.assertFalse(self.presence.handshake('127.0.0.1'))

  def test_one_announcement_a_period(self):
    self.presence.handshake('127.0.0.1')
    first = self.announced()
    self.now[0] += wifi.PRESENCE_PERIOD / 2
    self.presence.handshake('127.0.0.1')
    self.now[0] += wifi.PRESENCE_PERIOD / 2
    self.presence.handshake('127.0.0.1')
    self.assertNotEqual(first['id'], self.announced()['id'], 'a fresh id each announcement')
    self.phone.settimeout(0.1)
    with self.assertRaises(socket.timeout):
      self.phone.recvfrom(wifi.PRESENCE_DATAGRAM)

  def test_junk_is_no_answer(self):
    self.presence.handshake('127.0.0.1')
    self.announced()
    self.phone.sendto(b'not a datagram at all', self.caller)
    self.phone.sendto(b'{"jetlink": "presence", "v": 1, "ack": false, "id": "x"}', self.caller)
    self.assertFalse(self.presence.handshake('127.0.0.1'))

  def test_a_comma_on_no_wifi_announces_nothing(self):
    self.assertFalse(self.presence.handshake(None))
    self.assertIsNone(self.presence.sock)
    self.phone.settimeout(0.1)
    with self.assertRaises(socket.timeout):
      self.phone.recvfrom(wifi.PRESENCE_DATAGRAM)

  def test_the_answer_names_the_address_the_app_answered_from(self):
    # the source of the answer is the app's own address: the host this comma
    # will dial, and what a second app on the same LAN is read against
    self.presence.handshake('127.0.0.1')
    self.answer(self.announced())
    self.presence.handshake('127.0.0.1')
    self.assertEqual(self.presence.peer, '127.0.0.1')

  def test_a_peer_that_answered_is_named_in_the_announcements_that_follow(self):
    self.presence.handshake('127.0.0.1')
    first = self.announced()
    self.assertNotIn('to', first, 'nothing answered yet: the gateway is all it has')
    self.answer(first)
    # the answer is read inside the period it was announced in - the step runs
    # more often than the announcement - and that is where the peer comes from
    self.presence.handshake('127.0.0.1')
    self.assertEqual(self.presence.peer, '127.0.0.1')
    self.now[0] += wifi.PRESENCE_PERIOD
    self.presence.handshake('127.0.0.1')
    self.assertEqual(self.announced()['to'], '127.0.0.1')

  def test_a_comma_that_moved_networks_names_the_new_answer(self):
    # the peer is replaced by any later answer, never kept beside it
    self.presence.handshake('127.0.0.1')
    self.answer(self.announced())
    self.presence.handshake('127.0.0.1')
    self.now[0] += wifi.PRESENCE_PERIOD
    self.presence.handshake('127.0.0.1')
    self.assertEqual(self.announced()['to'], '127.0.0.1')
    # another app answers, from its own address
    other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    other.bind(('127.0.0.2', 0))
    self.addCleanup(other.close)
    other.sendto(json.dumps({'jetlink': 'presence', 'v': 1, 'ack': True, 'id': self.presence.id}).encode(),
                 self.caller)
    # the gateway is still the phone this comma dials: it is the app's own
    # address in the answer that moved
    self.presence.handshake('127.0.0.1')
    self.assertEqual(self.presence.peer, '127.0.0.2')
    self.now[0] += wifi.PRESENCE_PERIOD
    self.presence.handshake('127.0.0.1')
    self.assertEqual(self.announced()['to'], '127.0.0.2')

  def test_a_peer_that_answered_gets_the_announcement_unicast(self):
    # a phone that is a client of a router, in Wi-Fi power save, is handed the
    # unicast datagram addressed to it and drops the subnet's broadcast
    # (2026-10-20, on the bench's own LAN), so the address it answered from is
    # a target, and the one before the broadcast
    self.presence.handshake('127.0.0.1')
    self.answer(self.announced())
    self.presence.handshake('127.0.0.1')  # reads the answer, learns the peer
    self.assertEqual(self.presence.peer, '127.0.0.1')
    self.now[0] += wifi.PRESENCE_PERIOD
    with mock.patch.object(wifi, 'announce_targets', return_value=['127.0.0.1']) as targets:
      self.presence.handshake('127.0.0.1')
    self.assertEqual(targets.call_args.kwargs.get('peer'), '127.0.0.1')

  def test_every_target_gets_the_same_announcement(self):
    # one datagram per target: the gateway it dials, and - on a network of its
    # own - the broadcast a peer client hears it on
    second = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # the announcement's port is the app's, so a second target is another
    # address on that port: the phone, and a broadcast it would hear
    second.bind(('127.0.0.2', self.phone.getsockname()[1]))
    second.settimeout(2.0)
    self.addCleanup(second.close)
    with mock.patch.object(wifi, 'announce_targets', return_value=['127.0.0.1', '127.0.0.2']):
      self.presence.handshake('127.0.0.1')
    first_id = json.loads(self.phone.recvfrom(wifi.PRESENCE_DATAGRAM)[0])['id']
    self.assertEqual(json.loads(second.recvfrom(wifi.PRESENCE_DATAGRAM)[0])['id'], first_id)

  def test_a_broadcast_that_cannot_go_keeps_the_announcement_at_the_gateway(self):
    # the second target is the second chance, never the handshake's only leg
    with mock.patch.object(wifi, 'announce_targets', return_value=['127.0.0.1', '255.255.255.255']), \
         mock.patch.object(wifi.socket.socket, 'sendto', side_effect=[len(b''), OSError('EACCES')]):
      self.presence.handshake('127.0.0.1')
    self.assertIsNotNone(self.presence.sock, 'the gateway send went: the socket stays')
    self.assertFalse(self.presence.handshake('127.0.0.1'), 'and the next step announces again')

  def test_only_an_ack_of_the_same_id_is_an_answer(self):
    good = b'{"jetlink":"presence","v":1,"ack":true,"id":"ab"}'
    self.assertTrue(wifi._answered(good, 'ab'))
    self.assertFalse(wifi._answered(good, 'cd'))
    self.assertFalse(wifi._answered(b'{"jetlink":"presence","v":1,"ack":false,"id":"ab"}', 'ab'))
    self.assertFalse(wifi._answered(b'{"jetlink":"presence","v":2,"ack":true,"id":"ab"}', 'ab'))
    self.assertFalse(wifi._answered(b'{"jetlink":"presence","v":1,"ack":true,"id":"ab"}', ''))
    self.assertFalse(wifi._answered(b'nonsense', 'ab'))

class TestAnnounceTargets(unittest.TestCase):
  """Where an announcement goes: the phone's hotspot is the gateway, a phone on
  a router is a peer client the subnet's broadcast reaches. Which of the two
  this comma is on, it cannot know - the addresses say it, not the setting."""

  def test_a_gateway_is_always_one_target(self):
    self.assertEqual(wifi.announce_targets('192.168.43.1', interface=None), ['192.168.43.1'])

  def test_a_peer_that_answered_goes_before_the_broadcast(self):
    with mock.patch.object(wifi, 'local_addr', return_value=('192.168.10.79', 24)):
      self.assertEqual(wifi.announce_targets('192.168.10.251', peer='192.168.10.65'),
                       ['192.168.10.251', '192.168.10.65', '192.168.10.255'])

  def test_a_peer_on_no_interface_still_gets_a_datagram_of_its_own(self):
    self.assertEqual(wifi.announce_targets('10.0.0.1', interface=None, peer='10.0.0.2'),
                     ['10.0.0.1', '10.0.0.2'])

  def test_a_peer_that_is_the_gateway_is_sent_once(self):
    # the phone's own hotspot: the address the app answered from and the
    # gateway are one host, and one datagram reaches it
    self.assertEqual(wifi.announce_targets('192.168.43.1', interface=None, peer='192.168.43.1'),
                     ['192.168.43.1'])

  def test_the_subnets_broadcast_is_the_second_on_an_interface(self):
    with mock.patch.object(wifi, 'local_addr', return_value=('192.168.10.79', 24)):
      self.assertEqual(wifi.announce_targets('192.168.10.251'), ['192.168.10.251', '192.168.10.255'])

  def test_an_interface_with_no_address_is_the_gateway_alone(self):
    with mock.patch.object(wifi, 'local_addr', return_value=None):
      self.assertEqual(wifi.announce_targets('10.0.0.1'), ['10.0.0.1'])

  def test_a_gateway_that_is_the_broadcast_is_sent_once(self):
    with mock.patch.object(wifi, 'local_addr', return_value=('192.168.10.255', 24)):
      self.assertEqual(wifi.announce_targets('192.168.10.255'), ['192.168.10.255'])

  def test_a_prefix_with_no_hosts_has_no_broadcast(self):
    self.assertEqual(wifi.subnet_broadcast('10.0.0.1', 32), None)
    self.assertEqual(wifi.subnet_broadcast('10.0.0.1', 31), None)
    self.assertEqual(wifi.subnet_broadcast('10.0.0.1', 0), None)

  def test_the_broadcast_sets_the_host_bits_of_the_subnet(self):
    self.assertEqual(wifi.subnet_broadcast('192.168.10.79', 24), '192.168.10.255')
    self.assertEqual(wifi.subnet_broadcast('10.0.0.44', 16), '10.0.255.255')
    self.assertEqual(wifi.subnet_broadcast('172.20.10.5', 28), '172.20.10.15')
    self.assertEqual(wifi.subnet_broadcast('not an address', 24), None)


if __name__ == '__main__':
  unittest.main()
