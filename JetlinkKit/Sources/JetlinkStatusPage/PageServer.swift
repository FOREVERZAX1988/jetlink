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
  /// the comma's hotspot: a small HTTP/1.1 server with three GET paths. `/`
  /// is the page, `/events` a stream of Server-Sent Events, `/logs` the last
  /// lines this process logged. It only looks: it observes the server's
  /// controller and never sends it a command.
  ///
  /// The server comes first. Every thread here runs at the lowest priority,
  /// nothing here runs on the frame path, and the one lock a server thread
  /// could meet is behind `PageFeed`'s handoff queue, which never makes it wait.
  public final class PageServer: @unchecked Sendable {
    public struct Limits: Sendable {
      /// Event streams at once; each holds a thread.
      public var streams = 8
      /// The request line and headers, with the blank line after them.
      public var headerBytes = 8 * 1024
      /// A connection that has not sent its request by then is dropped.
      public var headerTimeout: TimeInterval = 30
      /// Quiet before a comment on an event stream, so a phone or a proxy
      /// keeps it open.
      public var keepalive: TimeInterval = 15
      /// Stats kept for a page that opens mid-drive: the chart's two minutes.
      public var history: TimeInterval = 120
      /// Hardware sampling outlives the last page by this much: a reload, a
      /// phone waking.
      public var grace: TimeInterval = 60
      public var period: TimeInterval = 1
      public var logLines = 300
      /// Connections at once of any kind; more are closed unanswered.
      public var connections = 32
      /// A page that stops reading is dropped after this long.
      public var writeTimeout: TimeInterval = 10

      public init() {}

      var feed: PageFeed.Limits {
        PageFeed.Limits(streams: streams, history: history, grace: grace, period: period)
      }
    }

    /// Where it listens; the one asked for, or the one the system gave for 0.
    public let port: UInt16

    let feed: PageFeed
    let limits: Limits
    private let page: Data
    private let logs: @Sendable () -> [String]
    private let listener: Int32
    private let condition = NSCondition()
    private var stopped = false
    private var threads = 0
    private var connections = 0
    private var detach: (() -> Void)?

    /// The page on every address at `port` (0: any free port), fed by
    /// `controller`. Throws `StatusPage.Unavailable` when the page's bundle
    /// is missing, or why it cannot listen.
    public static func start(
      port: Int, controller: ServerController, version: String, hardware: (any PageHardwareSource)?, limits: Limits = Limits()
    ) throws -> PageServer {
      let feed = PageFeed(hardware: hardware, limits: limits.feed)
      let server = try PageServer(port: port, page: StatusPage.page(), feed: feed, limits: limits) { LogRing.shared.lines() }
      server.watch(controller, version: version)
      return server
    }

    init(port: Int, page: Data, feed: PageFeed, limits: Limits, logs: @escaping @Sendable () -> [String]) throws {
      guard (0...65535).contains(port) else { throw PageServerError("\(port) is not a port") }
      (listener, self.port) = try PageServer.listen(port: UInt16(port))
      self.page = page
      self.feed = feed
      self.limits = limits
      self.logs = logs
      threads = 1
      PageThread.start("jetlink-page") { [self] in acceptLoop() }
    }

    /// Relays `controller`'s events from now on, after the hello and what the
    /// controller knows already. Called on the daemon's thread: the state is
    /// read here, never from a page's thread, because it takes the host's
    /// lock, which a frame holds.
    func watch(_ controller: ServerController, version: String) {
      feed.publish(.hello(StatusPage.hello(controller.server.configuration, version: version)))
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
      // Takes the socket out of LISTEN and wakes Linux's poll on it; the
      // accept loop closes it.
      _ = shutdown(listener, Int32(SHUT_RDWR))
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
      defer {
        close(listener)
        threadEnded(connection: false)
      }
      while !isStopped {
        var poller = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
        let ready = poll(&poller, 1, 250)
        if ready < 0 && errno != EINTR { return }
        if ready <= 0 { continue }
        if poller.revents & Int16(POLLERR | POLLHUP | POLLNVAL) != 0 { return }
        let client = accept(listener, nil, nil)
        if client < 0 { continue }
        condition.lock()
        let admitted = !stopped && connections < limits.connections
        if admitted {
          connections += 1
          threads += 1
        }
        condition.unlock()
        guard admitted else {
          close(client)
          continue
        }
        PageThread.start("jetlink-page-client") { [self] in
          serve(client)
          close(client)
          threadEnded(connection: true)
        }
      }
    }

    private func serve(_ client: Int32) {
      PageServer.tune(client, writeTimeout: limits.writeTimeout)
      let reply: Data
      switch readRequest(client) {
      case .request("GET", "/"):
        reply = PageResponse.whole(200, type: "text/html; charset=utf-8", body: page)
      case .request("GET", "/logs"):
        let lines = logs().suffix(limits.logLines)
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
    /// find an idle one; on Linux, TCP_USER_TIMEOUT drops one whose
    /// events go unacknowledged for 30 s. Loopback never loses a packet,
    /// so the tests check the options, not a vanishing phone.
    static func tune(_ fd: Int32, writeTimeout: TimeInterval) {
      func set(_ level: Int32, _ name: Int32, _ value: Int32) {
        var value = value
        _ = setsockopt(fd, level, name, &value, socklen_t(MemoryLayout<Int32>.size))
      }
      #if canImport(Darwin)
        set(SOL_SOCKET, SO_NOSIGPIPE, 1)
        set(Int32(IPPROTO_TCP), TCP_KEEPALIVE, keepalive.idle)
      #else
        set(Int32(IPPROTO_TCP), TCP_KEEPIDLE, keepalive.idle)
        set(Int32(IPPROTO_TCP), TCP_USER_TIMEOUT, keepalive.userTimeoutMs)
      #endif
      set(Int32(IPPROTO_TCP), TCP_NODELAY, 1)
      set(SOL_SOCKET, SO_KEEPALIVE, 1)
      set(Int32(IPPROTO_TCP), TCP_KEEPINTVL, keepalive.interval)
      set(Int32(IPPROTO_TCP), TCP_KEEPCNT, keepalive.count)
      var timeout = timeval(tv_sec: .init(writeTimeout), tv_usec: 0)
      _ = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    }

    static let keepalive: (idle: Int32, interval: Int32, count: Int32, userTimeoutMs: Int32) = (15, 5, 3, 30_000)

    /// The request, or `.incomplete` once the header timeout passes; nil
    /// when the client went away first.
    private func readRequest(_ client: Int32) -> PageRequest? {
      let deadline = Date(timeIntervalSinceNow: limits.headerTimeout)
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
        let room = min(chunk.count, max(1, limits.headerBytes - bytes.count))
        let n = chunk.withUnsafeMutableBytes { recv(client, $0.baseAddress, room, 0) }
        if n < 0 && (errno == EINTR || errno == EAGAIN) { continue }
        if n <= 0 { return nil }
        bytes += chunk[..<n]
        let request = PageRequest.parse(bytes, limit: limits.headerBytes)
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
      while let items = feed.next(stream, timeout: limits.keepalive) {
        if peerClosed(client) { return }
        var out = Data()
        for item in items {
          out.append(PageFeed.encode(item))
        }
        guard write(client, items.isEmpty ? PageFeed.keepalive : out) else { return }
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
          let n = send(client, raw.baseAddress! + offset, raw.count - offset, PageServer.sendFlags)
          if n < 0 && errno == EINTR { continue }
          // EAGAIN here is SO_SNDTIMEO: a page that stopped reading.
          if n <= 0 { return false }
          offset += n
        }
        return true
      }
    }

    #if canImport(Glibc)
      static let sendFlags = Int32(MSG_NOSIGNAL)
      static let streamType = Int32(SOCK_STREAM.rawValue)
    #else
      static let sendFlags: Int32 = 0
      static let streamType = SOCK_STREAM
    #endif

    /// A listening socket on every address: IPv6 with IPv4 mapped in, since a
    /// phone may resolve the Jetson's name to either, else IPv4 alone.
    static func listen(port: UInt16) throws -> (fd: Int32, port: UInt16) {
      var yes: Int32 = 1
      var no: Int32 = 0
      let six = socket(AF_INET6, streamType, 0)
      if six >= 0 {
        _ = setsockopt(six, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        _ = setsockopt(six, Int32(IPPROTO_IPV6), IPV6_V6ONLY, &no, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in6()
        #if canImport(Darwin)
          address.sin6_len = UInt8(MemoryLayout<sockaddr_in6>.size)
        #endif
        address.sin6_family = sa_family_t(AF_INET6)
        address.sin6_port = port.bigEndian
        if bindAndListen(six, &address) {
          var bound = sockaddr_in6()
          boundAddress(six, &bound)
          return (six, UInt16(bigEndian: bound.sin6_port))
        }
        close(six)
      }
      let four = socket(AF_INET, streamType, 0)
      guard four >= 0 else { throw PageServerError("socket: \(String(cString: strerror(errno)))") }
      _ = setsockopt(four, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
      var address = sockaddr_in()
      #if canImport(Darwin)
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
      #endif
      address.sin_family = sa_family_t(AF_INET)
      address.sin_port = port.bigEndian
      guard bindAndListen(four, &address) else {
        let reason = String(cString: strerror(errno))
        close(four)
        throw PageServerError("cannot listen on port \(port): \(reason)")
      }
      var bound = sockaddr_in()
      boundAddress(four, &bound)
      return (four, UInt16(bigEndian: bound.sin_port))
    }

    private static func bindAndListen<Address>(_ fd: Int32, _ address: inout Address) -> Bool {
      let bound = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<Address>.size)) }
      }
      #if canImport(Glibc)
        return bound == 0 && Glibc.listen(fd, 16) == 0
      #else
        return bound == 0 && Darwin.listen(fd, 16) == 0
      #endif
    }

    private static func boundAddress<Address>(_ fd: Int32, _ address: inout Address) {
      var length = socklen_t(MemoryLayout<Address>.size)
      _ = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
      }
    }
  }

  public struct PageServerError: Error, CustomStringConvertible {
    public let description: String

    init(_ description: String) {
      self.description = description
    }
  }

  extension StatusPage {
    /// The hello the Python server's control socket opened with, from what
    /// this server was started with. The page shows its version.
    static func hello(_ configuration: Server.Configuration, version: String) -> HelloEvent {
      let transport = configuration.usb ? (configuration.listen ? "usb+tcp" : "usb") : "tcp"
      #if os(Linux)
        let platform = "linux"
      #else
        let platform = "darwin"
      #endif
      return HelloEvent(
        protocolVersion: 1, pid: getpid(), version: version, python: "", platform: platform,
        cache: configuration.cacheRoot.path, transport: transport, port: configuration.listen ? Int(configuration.port) : nil)
    }
  }
#endif
