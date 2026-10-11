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
      let ours: Set<String> = ["192.168.43.1"]
      #expect(PresenceMonitor.verdict(gateway: "192.168.43.1", addresses: ours) == .here)
      // the comma joined another hotspot in the same subnet and dials that one's
      // device, not this: not the link this device is listening for
      #expect(PresenceMonitor.verdict(gateway: "192.168.10.1", addresses: ours) == .elsewhere)
      // a comma that names no gateway is read as before, as on this hotspot
      #expect(PresenceMonitor.verdict(gateway: nil, addresses: ours) == .here)
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
