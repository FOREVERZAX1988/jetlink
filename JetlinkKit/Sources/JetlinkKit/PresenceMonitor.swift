#if canImport(Network)
  import Darwin
  import Foundation
  import Network
  import Observation

  #if canImport(os)
    import os
  #else
    import JetlinkLog
  #endif

  /// What a Wi-Fi presence announcement says of a comma, from this device.
  public enum Presence: Equatable, Sendable {
    /// No comma answered: the Wi-Fi Link setting alone is not a link.
    case none
    /// A comma on this device's hotspot: it dialed an address this device holds.
    case here
    /// A comma answered, but it dials another gateway: it joined a different hotspot.
    case elsewhere
  }

  /// The device's end of the Wi-Fi presence handshake (docs/link-protocol.md): a
  /// comma whose Jetlink setting is Wi-Fi announces itself once a second to its
  /// gateway, this device, and this answers it.
  ///
  /// What the app calls connecting over Wi-Fi is a comma answering, never the
  /// Wi-Fi Link setting alone. The setting says only that this device is
  /// listening: a comma joined to another network - the device's home Wi-Fi, a
  /// comma that is simply off - is not on its way, and reading the setting as one
  /// that is was what made the Wi-Fi link look connected with nothing on it.
  ///
  /// An announcement names the gateway its comma dials, and this device answers -
  /// and reads the comma as ``Presence/here`` - only when that address is one of
  /// its own: the two are on one hotspot when the comma dials this device. A
  /// comma reporting another gateway joined a different hotspot
  /// (``Presence/elsewhere``), and saying so beats the "connecting" a bare answer
  /// would give.
  @MainActor
  @Observable
  public final class PresenceMonitor {
    /// A comma on this device's hotspot, answered within the hold, or why not.
    public private(set) var state: Presence = .none

    /// Where the comma announces itself: PRESENCE_PORT in jetlink.comma.wifi.
    public static let port: UInt16 = 5600
    /// How long one announcement counts for. The comma announces every second.
    private static let hold: TimeInterval = 5

    @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "presence")
    @ObservationIgnored private var listener: NWListener?
    @ObservationIgnored private var timer: DispatchSourceTimer?
    /// When an announcement last arrived, so a comma that goes quiet stops counting.
    @ObservationIgnored private var lastHeard = Date.distantPast

    public init() {}

    /// Listening, once: the Wi-Fi Link setting is on.
    public func start() {
      guard listener == nil else { return }
      do {
        let listener = try NWListener(using: .udp, on: NWEndpoint.Port(rawValue: PresenceMonitor.port)!)
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        listener.start(queue: presenceQueue)
        self.listener = listener
      } catch {
        log.warning("presence could not listen on \(Self.port): \(error.localizedDescription, privacy: .public)")
        return
      }
      let timer = DispatchSource.makeTimerSource(queue: presenceQueue)
      timer.schedule(deadline: .now() + Self.hold, repeating: 1)
      timer.setEventHandler { [weak self] in
        Task { @MainActor in self?.expire() }
      }
      timer.resume()
      self.timer = timer
    }

    /// Not listening, and the answer with it.
    public func stop() {
      timer?.cancel()
      timer = nil
      listener?.cancel()
      listener = nil
      state = .none
    }

    // MARK: the socket

    nonisolated private func accept(_ connection: NWConnection) {
      connection.start(queue: presenceQueue)
      receive(connection)
    }

    nonisolated private func receive(_ connection: NWConnection) {
      connection.receiveMessage { [weak self] data, _, _, error in
        if let data, let text = String(data: data, encoding: .utf8) {
          self?.handle(text, on: connection)
        }
        guard error == nil else { return }
        self?.receive(connection)
      }
    }

    /// One announcement: what it means, and the answer when this device is the
    /// gateway it dialed. A comma that dialed another gateway is not this link,
    /// and an answer would only tell it an app is here.
    nonisolated private func handle(_ text: String, on connection: NWConnection) {
      guard let id = PresenceMonitor.announceId(text) else { return }
      let verdict = PresenceMonitor.verdict(
        gateway: PresenceMonitor.announceGateway(text),
        addresses: PresenceMonitor.localAddresses())
      Task { @MainActor in self.note(verdict) }
      guard verdict == .here, let data = PresenceMonitor.answer(id).data(using: .utf8) else { return }
      connection.send(content: data, completion: .contentProcessed { _ in })
    }

    // MARK: the reading

    private func note(_ verdict: Presence) {
      lastHeard = Date()
      state = verdict
    }

    private func expire() {
      if state != .none, Date().timeIntervalSince(lastHeard) > Self.hold { state = .none }
    }

    // MARK: the announcement

    /// The id of the announcement `text` is, or nil when it is not one this app
    /// answers: another kind, another version, or no announcement at all.
    nonisolated public static func announceId(_ text: String) -> String? { field(text, "id") }

    /// The gateway the comma that sent `text` dials, or nil when the announcement
    /// says nothing of one: an older comma, whose app reads it as on its way
    /// without checking, as it did before this field.
    nonisolated public static func announceGateway(_ text: String) -> String? { field(text, "gw") }

    /// What an announcement of the gateway `gateway` means to a device holding
    /// `addresses`: on its hotspot when the two are the same, on another one when
    /// not. An announcement that names no gateway is read as before, as one on
    /// this hotspot.
    nonisolated public static func verdict(gateway: String?, addresses: Set<String>) -> Presence {
      guard let gateway else { return .here }
      return addresses.contains(gateway) ? .here : .elsewhere
    }

    /// The answer to announcement `id`: the comma reads it as this app listening.
    nonisolated public static func answer(_ id: String) -> String {
      let object: [String: Any] = ["jetlink": "presence", "v": 1, "ack": true, "id": id]
      guard let data = try? JSONSerialization.data(withJSONObject: object) else { return "" }
      return String(decoding: data, as: UTF8.self)
    }

    /// This device's own IPv4 addresses, off the loopback: the addresses a comma
    /// on its hotspot dials.
    nonisolated public static func localAddresses() -> Set<String> {
      var found: Set<String> = []
      var head: UnsafeMutablePointer<ifaddrs>?
      guard getifaddrs(&head) == 0, let first = head else { return found }
      defer { freeifaddrs(head) }
      var cursor: UnsafeMutablePointer<ifaddrs>? = first
      while let entry = cursor {
        defer { cursor = entry.pointee.ifa_next }
        let flags = Int32(entry.pointee.ifa_flags)
        guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0,
          let socket = entry.pointee.ifa_addr, socket.pointee.sa_family == UInt8(AF_INET)
        else { continue }
        var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        guard getnameinfo(socket, socklen_t(socket.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 else { continue }
        found.insert(String(decoding: host.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self))
      }
      return found
    }

    /// One field of a presence announcement, or nil when `text` is not one.
    nonisolated private static func field(_ text: String, _ name: String) -> String? {
      guard let data = text.data(using: .utf8),
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        object["jetlink"] as? String == "presence",
        object["v"] as? Int == 1
      else { return nil }
      return object[name] as? String
    }
  }

  /// One queue for every presence listener's socket and timer.
  private let presenceQueue = DispatchQueue(label: "io.zoompilot.jetlink.presence")
#endif
