#if canImport(Network)
  import Foundation
  import Testing

  @testable import JetlinkKit

  /// The announcement and the answer to it, as jetlink.comma.wifi.Presence sends
  /// and reads them, and what this device makes of the gateway a comma names.
  struct PresenceMonitorTests {
    @Test func anAnnouncementNamesItsProtocolAndItsId() {
      #expect(PresenceMonitor.announceId(#"{"jetlink":"presence","v":1,"id":"9b3f0c1d"}"#) == "9b3f0c1d")
    }

    @Test func anAnnouncementNamesTheGatewayItsCommaDials() {
      let announcement = #"{"jetlink":"presence","v":1,"id":"9b3f0c1d","gw":"192.168.43.1"}"#
      #expect(PresenceMonitor.announceGateway(announcement) == "192.168.43.1")
      // an older comma says nothing of one: the app reads it as before
      #expect(PresenceMonitor.announceGateway(#"{"jetlink":"presence","v":1,"id":"9b3f0c1d"}"#) == nil)
      #expect(PresenceMonitor.announceGateway(#"{"jetlink":"snapshot","v":1,"gw":"192.168.43.1"}"#) == nil)
    }

    @Test func nothingElseIsAnAnnouncement() {
      #expect(PresenceMonitor.announceId(#"{"jetlink":"snapshot","v":1,"id":"9b3f0c1d"}"#) == nil)
      #expect(PresenceMonitor.announceId(#"{"jetlink":"presence","v":2,"id":"9b3f0c1d"}"#) == nil)
      #expect(PresenceMonitor.announceId(#"{"jetlink":"presence","v":1}"#) == nil)
      #expect(PresenceMonitor.announceId("hello") == nil)
      #expect(PresenceMonitor.announceId("[]") == nil)
    }

    @Test func theGatewayIsThisDeviceOnlyWhenItIsOneOfItsOwn() {
      let hotspot: Set<LocalAddress> = [LocalAddress(address: "192.168.43.1", prefix: 24)]
      #expect(PresenceMonitor.verdict(target: nil, gateway: "192.168.43.1", sender: "192.168.43.42",
                                     addresses: hotspot) == .here)
      // the comma joined another network and dials that one's device, not this:
      // the announcement comes from a subnet this device is not on
      #expect(PresenceMonitor.verdict(target: nil, gateway: "192.168.10.1", sender: "192.168.10.77",
                                     addresses: hotspot) == .elsewhere)
      // a comma that names no gateway is read as before, as on this hotspot
      #expect(PresenceMonitor.verdict(target: nil, gateway: nil, sender: nil, addresses: hotspot) == .here)
    }

    @Test func aCommaOnThisDevicesOwnLanIsOnItsWay() {
      // the router is the gateway and this device is a client of it: discovery
      // finds the app and dials it, so that is this device's link too
      let lan: Set<LocalAddress> = [LocalAddress(address: "192.168.10.50", prefix: 24)]
      #expect(PresenceMonitor.verdict(target: nil, gateway: "192.168.10.251", sender: "192.168.10.79",
                                     addresses: lan) == .here)
    }

    @Test func theHostTheCommaNamesIsTheOneItIsReadAgainst() {
      // once an app answers, the comma names the host it dials, and that settles
      // whose link it is - on one LAN a second app would otherwise read every
      // announcement as its own
      let lan: Set<LocalAddress> = [LocalAddress(address: "192.168.10.50", prefix: 24)]
      #expect(PresenceMonitor.verdict(target: "192.168.10.50", gateway: "192.168.10.251",
                                     sender: "192.168.10.79", addresses: lan) == .here)
      #expect(PresenceMonitor.verdict(target: "192.168.10.65", gateway: "192.168.10.251",
                                     sender: "192.168.10.79", addresses: lan) == .elsewhere)
    }

    @Test func anAnnouncementNamesTheHostItsCommaDials() {
      let announcement = #"{"jetlink":"presence","v":1,"id":"9b3f0c1d","to":"192.168.10.50"}"#
      #expect(PresenceMonitor.announceTarget(announcement) == "192.168.10.50")
      #expect(PresenceMonitor.announceTarget(#"{"jetlink":"presence","v":1,"id":"9b3f0c1d"}"#) == nil)
      #expect(PresenceMonitor.announceTarget(#"{"jetlink":"snapshot","v":1,"to":"192.168.10.50"}"#) == nil)
    }

    @Test func aSubnetIsReadOfThisDevicesOwnPrefix() {
      let lan = LocalAddress(address: "192.168.10.50", prefix: 24)
      #expect(lan.sharesSubnet("192.168.10.251"))
      #expect(!lan.sharesSubnet("192.168.11.50"))
      #expect(LocalAddress(address: "10.0.0.44", prefix: 16).sharesSubnet("10.0.9.9"))
      #expect(!LocalAddress(address: "192.168.10.50", prefix: 32).sharesSubnet("192.168.10.50"))
      #expect(!LocalAddress(address: "10.0.0.1", prefix: 0).sharesSubnet("8.8.8.8"))
      #expect(!lan.sharesSubnet("not an address"))
    }

    @Test func theAnswerCarriesThatIdAndTheAck() throws {
      let data = try #require(PresenceMonitor.answer("9b3f0c1d").data(using: .utf8))
      let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
      #expect(object["jetlink"] as? String == "presence")
      #expect(object["v"] as? Int == 1)
      #expect(object["ack"] as? Bool == true)
      #expect(object["id"] as? String == "9b3f0c1d")
    }
  }
#endif
