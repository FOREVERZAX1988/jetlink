#if canImport(Darwin) || canImport(Glibc)
  import Foundation
  import JetlinkKit
  import JetlinkLog
  import JetlinkServer
  #if canImport(Glibc)
    import Glibc
  #else
    import Darwin
  #endif

  /// The read-only status page, served by the daemon itself for a phone on
  /// the comma's hotspot, on three GET paths: `/` the page, `/events` a
  /// stream of Server-Sent Events, `/logs` the last lines this process
  /// logged. It observes the server's controller and never sends it a
  /// command. The server comes first: every thread here runs at the lowest
  /// priority, and the one lock a server thread could meet is behind
  /// `PageFeed`'s handoff queue, which never makes it wait.
  public final class PageServer: @unchecked Sendable {
    /// The request line and headers, with the blank line after them.
    static let headerBytes = 8 * 1024
    /// Connections at once of any kind; more are closed unanswered.
    static let connections = 32
    /// A page that stops reading is dropped after this long.
    static let writeTimeout: TimeInterval = 10
    static let logLines = 300

    /// Where it listens; the one asked for, or the one the system gave for 0.
    public let port: UInt16

    let feed: PageFeed
    /// A connection that has not sent its request by then is dropped.
    private let headerTimeout: TimeInterval
    /// Quiet before a comment on an event stream, so a phone or a proxy
    /// keeps it open.
    private let keepalive: TimeInterval
    private let page: Data
    private let logs: @Sendable () -> [String]
    private let listener: TCPListener
    private let condition = NSCondition()
    private var stopped = false
    private var threads = 0
    private var connections = 0
    private var detach: (() -> Void)?

    /// The page on every address at `port` (0: any free port), fed by
    /// `controller`. Throws `StatusPage.Unavailable` when the page's bundle
    /// is missing, or why it cannot listen.
    public static func start(port: Int, controller: ServerController, version: String, hardware: (any PageHardwareSource)?) throws -> PageServer {
      let server = try PageServer(port: port, page: StatusPage.page(), feed: PageFeed(hardware: hardware)) { LogRing.shared.lines() }
      server.watch(controller, version: version)
      return server
    }

    init(
      port: Int, page: Data, feed: PageFeed, headerTimeout: TimeInterval = 30, keepalive: TimeInterval = 15,
      logs: @escaping @Sendable () -> [String]
    ) throws {
      guard let port = UInt16(exactly: port) else { throw HostError.invalid("\(port) is not a port") }
      // IPv6 with IPv4 mapped in, since a phone may resolve the Jetson's name
      // to either
      listener = try TCPListener(port: port, dualStack: true)
      self.port = listener.port
      self.page = page
      self.feed = feed
      self.headerTimeout = headerTimeout
      self.keepalive = keepalive
      self.logs = logs
      threads = 1
      PageThread.start("jetlink-page") { [self] in acceptLoop() }
    }

    /// Relays `controller`'s events from now on, after the hello and what the
    /// controller knows already. Called on the daemon's thread: the state is
    /// read here, never from a page's thread, because it takes the host's
    /// lock, which a frame holds.
    func watch(_ controller: ServerController, version: String) {
      feed.publish(.hello(HelloEvent(version: version)))
      let id = controller.observe { [feed] in feed.publish($0) }
      feed.seed(controller.currentState())
      condition.lock()
      detach = { controller.stopObserving(id) }
      condition.unlock()
    }

    /// Tells the open pages the server is stopping, closes them and stops
    /// listening. Waits a second at most for the page's threads.
    public func stop() {
      condition.lock()
      let first = !stopped
      stopped = true
      let detach = self.detach
      self.detach = nil
      condition.unlock()
      guard first else { return }
      detach?()
      feed.stop()
      listener.close()
      let deadline = Date(timeIntervalSinceNow: 1)
      condition.lock()
      while threads > 0 && condition.wait(until: deadline) {}
      condition.unlock()
    }

    private var isStopped: Bool {
      condition.lock()
      defer { condition.unlock() }
      return stopped
    }

    private func threadEnded(connection: Bool) {
      condition.lock()
      threads -= 1
      if connection { connections -= 1 }
      condition.broadcast()
      condition.unlock()
    }

    // MARK: connections

    private func acceptLoop() {
      defer { threadEnded(connection: false) }
      while let (client, _) = listener.acceptConnection() {
        condition.lock()
        let admitted = !stopped && connections < PageServer.connections
        if admitted {
          connections += 1
          threads += 1
        }
        condition.unlock()
        guard admitted else {
          Sys.close(client)
          continue
        }
        PageThread.start("jetlink-page-client") { [self] in
          serve(client)
          Sys.close(client)
          threadEnded(connection: true)
        }
      }
    }

    private func serve(_ client: Int32) {
      PageServer.tune(client)
      let reply: Data
      switch readRequest(client) {
      case .request("GET", "/"):
        reply = PageResponse.whole(200, type: "text/html; charset=utf-8", body: page)
      case .request("GET", "/logs"):
        let lines = logs().suffix(PageServer.logLines)
        reply = PageResponse.text(200, lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n")
      case .request("GET", "/events"):
        stream(client)
        return
      case .request:
        reply = PageResponse.text(404, "not found\n")
      case .bad:
        reply = PageResponse.text(400, "bad request\n")
      case .tooLarge:
        reply = PageResponse.text(431, "request too large\n")
      case .incomplete:
        // Timed out, or the page's server is stopping.
        reply = PageResponse.text(408, "no request\n")
      case nil:
        return
      }
      if write(client, reply) {
        linger(client)
      }
    }

    /// Closing with unread bytes waiting resets the connection, and a reset
    /// can throw away the reply before the client reads it: the end of an
    /// oversized request, say. So stop sending, and read what is left for up
    /// to a second.
    private func linger(_ client: Int32) {
      _ = shutdown(client, Int32(SHUT_WR))
      let deadline = Date(timeIntervalSinceNow: 1)
      var sink = [UInt8](repeating: 0, count: 4096)
      var drained = 0
      while drained < 1 << 16 {
        let left = deadline.timeIntervalSinceNow
        if left <= 0 { return }
        var poller = pollfd(fd: client, events: Int16(POLLIN), revents: 0)
        if poll(&poller, 1, Int32(left * 1000) + 1) <= 0 { return }
        let n = sink.withUnsafeMutableBytes { recv(client, $0.baseAddress, $0.count, 0) }
        if n <= 0 { return }
        drained += n
      }
    }

    /// A page's socket: replies go out at once, and a phone that vanished
    /// without closing (it left the hotspot, its radio slept) frees its
    /// stream in about half a minute rather than once the send buffer
    /// fills. Keepalive probes after 15 s of silence, 5 s apart, 3 of them,
    /// find an idle one; on Linux, TCP_USER_TIMEOUT drops one whose events
    /// go unacknowledged for 30 s. Loopback never loses a packet, so the
    /// tests check the options, not a vanishing phone.
    static func tune(_ fd: Int32) {
      Sys.tune(fd, keepalive: (15, 5, 3), sendTimeout: writeTimeout)
      #if canImport(Glibc)
        Sys.set(fd, Int32(IPPROTO_TCP), TCP_USER_TIMEOUT, 30_000)
      #endif
    }

    /// The request, or `.incomplete` once the header timeout passes; nil
    /// when the client went away first.
    private func readRequest(_ client: Int32) -> PageRequest? {
      let deadline = Date(timeIntervalSinceNow: headerTimeout)
      var bytes: [UInt8] = []
      var chunk = [UInt8](repeating: 0, count: 2048)
      while !isStopped {
        let left = deadline.timeIntervalSinceNow
        if left <= 0 { return .incomplete }
        var poller = pollfd(fd: client, events: Int16(POLLIN), revents: 0)
        let ready = poll(&poller, 1, Int32(min(left, 0.25) * 1000) + 1)
        if ready < 0 && errno != EINTR { return nil }
        if ready <= 0 { continue }
        // Never more than the cap holds: past it the answer is 431 anyway.
        let room = min(chunk.count, max(1, PageServer.headerBytes - bytes.count))
        let n = chunk.withUnsafeMutableBytes { recv(client, $0.baseAddress, room, 0) }
        if n < 0 && (errno == EINTR || errno == EAGAIN) { continue }
        if n <= 0 { return nil }
        bytes += chunk[..<n]
        let request = PageRequest.parse(bytes, limit: PageServer.headerBytes)
        if request != .incomplete { return request }
      }
      return .incomplete
    }

    private func stream(_ client: Int32) {
      guard let stream = feed.open() else {
        _ = write(client, PageResponse.text(503, "too many pages open\n"))
        return
      }
      defer { feed.close(stream) }
      // retry: how soon a browser's own reconnect comes; the page backs off
      // on its own when that fails.
      guard write(client, PageResponse.head(200, type: "text/event-stream") + Data("retry: 2000\n\n".utf8)) else { return }
      while let frames = feed.next(stream, timeout: keepalive) {
        if peerClosed(client) { return }
        guard write(client, frames.isEmpty ? PageFeed.keepalive : Data(frames.joined())) else { return }
      }
    }

    /// A page that closed its end: a reload, a phone locking. Found before
    /// the next write rather than by it.
    private func peerClosed(_ client: Int32) -> Bool {
      var byte: UInt8 = 0
      let n = recv(client, &byte, 1, Int32(MSG_PEEK | MSG_DONTWAIT))
      return n == 0 || n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR
    }

    private func write(_ client: Int32, _ data: Data) -> Bool {
      data.withUnsafeBytes { raw in
        var offset = 0
        while offset < raw.count {
          let n = send(client, raw.baseAddress! + offset, raw.count - offset, Sys.sendFlags)
          if n < 0 && errno == EINTR { continue }
          // EAGAIN here is SO_SNDTIMEO: a page that stopped reading.
          if n <= 0 { return false }
          offset += n
        }
        return true
      }
    }
  }
#endif
