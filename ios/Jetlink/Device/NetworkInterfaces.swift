import Darwin
import Foundation
import Network
import Observation

/// The phone's IPv4 addresses, and which one the comma should be pointed at.
///
/// A comma reaches the phone over a USB-C Ethernet adapter, which iOS names
/// en1, en2 and so on; en0 is Wi-Fi and bridge100 the Personal Hotspot. Wired
/// is the one that meets the frame budget, so it comes first.
@MainActor
@Observable
final class NetworkInterfaces {
  struct Address: Identifiable, Equatable, Sendable {
    enum Kind: Int, Comparable, Sendable {
      case ethernet = 0, hotspot, wifi, other

      static func < (a: Kind, b: Kind) -> Bool { a.rawValue < b.rawValue }

      var title: String {
        switch self {
        case .ethernet: "Ethernet"
        case .hotspot: "Personal Hotspot"
        case .wifi: "Wi-Fi"
        case .other: "Other"
        }
      }

      var symbol: String {
        switch self {
        case .ethernet: "cable.connector"
        case .hotspot: "personalhotspot"
        case .wifi: "wifi"
        case .other: "network"
        }
      }
    }

    var id: String { "\(interface)-\(address)" }
    let interface: String
    let address: String
    let kind: Kind
  }

  private(set) var addresses: [Address] = []
  @ObservationIgnored private let monitor = NWPathMonitor()

  init() {
    refresh()
    monitor.pathUpdateHandler = { [weak self] _ in
      Task { @MainActor in self?.refresh() }
    }
    monitor.start(queue: DispatchQueue(label: "io.zoompilot.jetlink.network"))
  }

  /// The address to give the comma: wired first, then the hotspot, then Wi-Fi.
  var preferred: Address? {
    addresses.first
  }

  func refresh() {
    addresses = NetworkInterfaces.read()
  }

  static func read() -> [Address] {
    var head: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&head) == 0, let first = head else { return [] }
    defer { freeifaddrs(head) }
    var found: [Address] = []
    var cursor: UnsafeMutablePointer<ifaddrs>? = first
    while let entry = cursor {
      defer { cursor = entry.pointee.ifa_next }
      let flags = Int32(entry.pointee.ifa_flags)
      guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0,
        let socket = entry.pointee.ifa_addr, socket.pointee.sa_family == UInt8(AF_INET)
      else { continue }
      let name = String(cString: entry.pointee.ifa_name)
      let kind: Address.Kind
      if name == "en0" {
        kind = .wifi
      } else if name.hasPrefix("en") {
        kind = .ethernet
      } else if name.hasPrefix("bridge") {
        kind = .hotspot
      } else if name.hasPrefix("pdp_ip") || name.hasPrefix("utun") || name.hasPrefix("ipsec") || name.hasPrefix("awdl") || name.hasPrefix("llw") {
        continue  // cellular and tunnels: never where a comma is
      } else {
        kind = .other
      }
      var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
      guard getnameinfo(socket, socklen_t(socket.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 else { continue }
      let text = String(decoding: host.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
      found.append(Address(interface: name, address: text, kind: kind))
    }
    return found.sorted { ($0.kind, $0.interface) < ($1.kind, $1.interface) }
  }
}
