import Darwin
import Foundation
import JetlinkKit

/// Where a server dials to serve: the comma's end of a USB network link,
/// which listens for the phone (192.168.60.1:5599 on the composite gadget).
public struct DialTarget: Sendable, Equatable, CustomStringConvertible {
  public let host: String
  public let port: UInt16

  public init(host: String, port: UInt16 = Wire.defaultPort) {
    self.host = host
    self.port = port
  }

  /// "host:port", or "host" for the default port.
  public init?(_ text: String) {
    let parts = text.split(separator: ":", maxSplits: 1).map(String.init)
    guard let host = parts.first, !host.isEmpty else { return nil }
    var port = Wire.defaultPort
    if parts.count == 2 {
      guard let parsed = UInt16(parts[1]) else { return nil }
      port = parsed
    }
    self.init(host: host, port: port)
  }

  public var description: String { "\(host):\(port)" }
}

/// The jetlink server over TCP: a listener, one comma at a time, and the
/// engine host they share. The Swift form of `server/main.py`'s `_serve`.
///
/// One difference from the Python: a new connection takes over from the one
/// being served instead of waiting behind it. A comma only ever has one
/// connection open, so a second one means the first is dead (a pulled cable
/// the keepalive has not noticed yet), and the reconnect must not wait for it.
///
/// A connection comes from the listener, or from dialing: over a USB network
/// link the comma listens and the phone dials it, so an accepted connection
/// on the comma is the proof of a phone. A dialed connection is served the
/// same way, and the listener stays open beside it for benches and a Mac on
/// the LAN.
public final class Server: @unchecked Sendable {
  public struct Configuration: Sendable {
    public var host: String
    public var port: UInt16
    public var cacheRoot: URL
    public var device: CoreMLBackend.Device
    /// Keep the GPU clocked up between frames (MetalKeepAlive).
    public var keepAlive: Bool
    /// Keep a CPU core busy between frames while a Neural Engine session
    /// runs (CPUKeepWarm).
    public var keepCPUWarm: Bool
    /// Start loading the engine that was loaded last, before a comma asks.
    public var preload: Bool
    /// Dial this end and serve the connection; `setDial` changes it later.
    public var dial: DialTarget?

    public init(
      host: String = "0.0.0.0", port: UInt16 = Wire.defaultPort, cacheRoot: URL, device: CoreMLBackend.Device = .ane, keepAlive: Bool = true,
      keepCPUWarm: Bool = true, preload: Bool = true, dial: DialTarget? = nil
    ) {
      self.host = host
      self.port = port
      self.cacheRoot = cacheRoot
      self.device = device
      self.keepAlive = keepAlive
      self.keepCPUWarm = keepCPUWarm
      self.preload = preload
      self.dial = dial
    }
  }

  public static let statsInterval: TimeInterval = 1.0
  /// A dial that takes longer has no comma behind it.
  public static let dialTimeout: TimeInterval = 1.0
  public static let dialInterval: TimeInterval = 0.5
  /// Between attempts to listen again after the socket went away.
  static let relistenBackoff: ClosedRange<TimeInterval> = 0.5...5.0

  public let configuration: Configuration
  public let host: EngineHost
  public let cache: EngineCache
  public let backend: CoreMLBackend
  /// The piggybacked telemetry: what the comma logs about its accelerator.
  public var telemetry: @Sendable () -> [String: Any] = { [:] }

  private let log = ServerLog(category: "server")
  private let lock = NSLock()
  /// Internal so a test can end it under the accept loop.
  var listener: TCPListener?
  private var current: Session?
  private var currentDone: Latch?
  private var link: LinkEvent = .waiting
  private var ticker: Ticker?
  private var started = false
  private var stopped = false
  private var dial: DialTarget?
  private var dialing = false

  public init(configuration: Configuration, preparer: any ModelPreparer) throws {
    self.configuration = configuration
    self.dial = configuration.dial
    backend = CoreMLBackend(
      device: configuration.device, preparer: preparer, keepAlive: configuration.keepAlive, keepCPUWarm: configuration.keepCPUWarm)
    cache = try EngineCache(root: configuration.cacheRoot, backend: backend)
    host = EngineHost(cache: cache)
    // A write to a socket the comma closed must be an error, not a signal
    // that kills the app.
    signal(SIGPIPE, SIG_IGN)
  }

  /// Listens, dials if configured, and serves until `stop`. Returns once listening.
  public func start() throws {
    try listen()
    lock.lock()
    started = true
    lock.unlock()
    if configuration.preload {
      host.preload()
    }
    ticker = Ticker(interval: Server.statsInterval) { [weak self] _ in self?.tick() }
    startDialing()
  }

  private func listen() throws {
    let listener = try TCPListener(host: configuration.host, port: configuration.port)
    lock.lock()
    if self.listener != nil {
      // Listened again from two sides at once; the first one stands.
      lock.unlock()
      listener.close()
      return
    }
    self.listener = listener
    lock.unlock()
    log.info("listening on \(configuration.host):\(listener.port)")
    if currentLink.state != .connected {
      setLink(LinkEvent(state: .waiting, detail: "listening on \(configuration.host):\(listener.port)", peer: nil))
    }
    let thread = Thread { [self] in
      acceptLoop(listener)
      listenerEnded(listener)
    }
    thread.name = "jetlink-accept"
    // The first receive of a new connection starts on this thread's priority.
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  /// Whether the listener is still accepting. iOS takes a suspended app's
  /// listening socket away, so the app asks this when it comes back.
  public var isListening: Bool {
    lock.lock()
    defer { lock.unlock() }
    return listener != nil
  }

  /// Listens again after the old socket was taken away. The engine stays loaded.
  public func reopenListener() throws {
    lock.lock()
    let old = listener
    let isStopped = stopped
    listener = nil
    lock.unlock()
    guard !isStopped else { return }
    old?.close()
    try listen()
  }

  /// The accept loop ended on its own: iOS reclaimed the socket, or accept
  /// failed. Listen again, backing off while the system refuses, until the
  /// server stops or something else has listened meanwhile.
  private func listenerEnded(_ ended: TCPListener) {
    lock.lock()
    let wasCurrent = listener === ended
    if wasCurrent { listener = nil }
    let isStopped = stopped
    lock.unlock()
    guard wasCurrent && !isStopped else { return }
    log.warning("the listening socket went away; listening again")
    ended.close()
    var backoff = Server.relistenBackoff.lowerBound
    while true {
      Thread.sleep(forTimeInterval: backoff)
      lock.lock()
      let idle = listener == nil && !stopped
      lock.unlock()
      guard idle else { return }
      do {
        try listen()
        return
      } catch {
        log.warning("cannot listen yet: \(String(describing: error))")
        backoff = min(backoff * 2, Server.relistenBackoff.upperBound)
      }
    }
  }

  public var port: UInt16? {
    lock.lock()
    defer { lock.unlock() }
    return listener?.port
  }

  public var currentLink: LinkEvent {
    lock.lock()
    defer { lock.unlock() }
    return link
  }

  /// Frames the connected comma has been served.
  public var framesServed: Int {
    lock.lock()
    defer { lock.unlock() }
    return current?.frames ?? 0
  }

  /// The frames served over the last `window` seconds, as one summary: the
  /// dashboard's headline, steadier than the once-a-second event.
  public func recentStats(window: TimeInterval) -> StatsEvent? {
    host.frameStats.summary(window: window, framesTotal: framesServed)
  }

  /// Stops listening, dialing and serving. The engine stays loaded, for a
  /// server that will be started again; `shutdown()` releases it.
  public func stop() {
    lock.lock()
    stopped = true
    let listener = self.listener
    let session = current
    self.listener = nil
    lock.unlock()
    ticker?.stop()
    listener?.close()
    session?.interrupt()
  }

  /// `stop()` and release the engine: the process is ending.
  public func shutdown() {
    stop()
    host.close()
  }

  // MARK: connections

  private func acceptLoop(_ listener: TCPListener) {
    while let transport = listener.accept() {
      _ = takeover(transport)
    }
  }

  /// Serves `transport` on a thread of its own, after the session being
  /// served, if any, has been interrupted and has ended. Returns the latch
  /// the new session's end releases.
  private func takeover(_ transport: TCPTransport) -> Latch {
    lock.lock()
    let previous = current
    let previousDone = currentDone
    lock.unlock()
    if let previous {
      log.info("a new connection from \(transport.peer) takes over from \(previous.peer)")
      previous.interrupt()
      previousDone?.wait()
    }
    let session = Session(transport: transport, host: host) { [weak self] in self?.telemetry() ?? [:] }
    let done = Latch()
    lock.lock()
    current = session
    currentDone = done
    lock.unlock()
    let thread = Thread { [self] in
      serve(session)
      done.release()
    }
    thread.name = "jetlink-session"
    thread.qualityOfService = .userInteractive
    thread.start()
    return done
  }

  private func serve(_ session: Session) {
    log.info("client connected from \(session.peer)")
    setLink(LinkEvent(state: .connected, detail: "", peer: session.peer))
    let reason = session.serveForever()
    session.close()
    lock.lock()
    let isCurrent = current === session
    if isCurrent {
      current = nil
      currentDone = nil
    }
    let isStopped = stopped
    lock.unlock()
    log.info("client disconnected: \(reason)")
    if isCurrent && !isStopped {
      setLink(LinkEvent(state: .disconnected, detail: reason, peer: nil))
    }
  }

  // MARK: dialing

  /// Where the server dials, if anywhere.
  public var dialTarget: DialTarget? {
    lock.lock()
    defer { lock.unlock() }
    return dial
  }

  /// Dial `target` from now on, or stop dialing with nil. A connection
  /// being served is left alone either way.
  public func setDial(_ target: DialTarget?) {
    lock.lock()
    let changed = dial != target
    dial = target
    lock.unlock()
    if changed, let target {
      log.info("dialing \(target.description)")
    } else if changed {
      log.info("no longer dialing")
    }
    startDialing()
  }

  /// Starts the dial thread if there is a target and none is running.
  private func startDialing() {
    lock.lock()
    guard started, !stopped, dial != nil, !dialing else {
      lock.unlock()
      return
    }
    dialing = true
    lock.unlock()
    let thread = Thread { [self] in dialLoop() }
    thread.name = "jetlink-dial"
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  /// Connects, serves, and dials again when the session ends; retries
  /// every `dialInterval` while the target stands and the server runs.
  private func dialLoop() {
    var failed: DialTarget?
    while true {
      lock.lock()
      guard !stopped, let target = dial else {
        dialing = false
        lock.unlock()
        return
      }
      lock.unlock()
      do {
        let transport = try TCPTransport.connect(host: target.host, port: target.port, timeout: Server.dialTimeout)
        failed = nil
        log.info("dialed \(transport.peer)")
        takeover(transport).wait()
      } catch {
        // Once per outage, not twice a second.
        if failed != target {
          failed = target
          log.info("cannot reach \(target.description) yet: \(String(describing: error))")
        }
      }
      Thread.sleep(forTimeInterval: Server.dialInterval)
    }
  }

  private func setLink(_ event: LinkEvent) {
    lock.lock()
    link = event
    lock.unlock()
    host.emit(.link(event))
  }

  private var lastTick = ProcessInfo.processInfo.systemUptime

  /// A frame summary a second while a comma is connected and sending. The
  /// window is the time since the last tick, so no frame falls between two.
  private func tick() {
    let now = ProcessInfo.processInfo.systemUptime
    let window = now - lastTick
    lastTick = now
    lock.lock()
    let connected = link.state == .connected
    let frames = current?.frames ?? 0
    lock.unlock()
    guard connected, let stats = host.frameStats.summary(window: window, framesTotal: frames) else { return }
    host.emit(.stats(stats))
  }
}

/// Released once, waited on by any number of threads: the accept loop and
/// the dial loop can both wait for the same session to end.
final class Latch: @unchecked Sendable {
  private let condition = NSCondition()
  private var released = false

  func release() {
    condition.lock()
    released = true
    condition.broadcast()
    condition.unlock()
  }

  func wait() {
    condition.lock()
    while !released {
      condition.wait()
    }
    condition.unlock()
  }
}
