import Darwin
import Foundation
import JetlinkKit
import os

/// The jetlink server over TCP: a listener, one comma at a time, and the
/// engine host they share. The Swift form of `server/main.py`'s `_serve`.
///
/// One difference from the Python: a new connection takes over from the one
/// being served instead of waiting behind it. A comma only ever has one
/// connection open, so a second one means the first is dead (a pulled cable
/// the keepalive has not noticed yet), and the reconnect must not wait for it.
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

    public init(
      host: String = "0.0.0.0", port: UInt16 = Wire.defaultPort, cacheRoot: URL, device: CoreMLBackend.Device = .ane, keepAlive: Bool = true,
      keepCPUWarm: Bool = true, preload: Bool = true
    ) {
      self.host = host
      self.port = port
      self.cacheRoot = cacheRoot
      self.device = device
      self.keepAlive = keepAlive
      self.keepCPUWarm = keepCPUWarm
      self.preload = preload
    }
  }

  public static let statsInterval: TimeInterval = 1.0

  public let configuration: Configuration
  public let host: EngineHost
  public let cache: EngineCache
  public let backend: CoreMLBackend
  /// The piggybacked telemetry: what the comma logs about its accelerator.
  public var telemetry: @Sendable () -> [String: Any] = { [:] }

  private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "server")
  private let lock = NSLock()
  private var listener: TCPListener?
  private var current: Session?
  private var currentDone: DispatchSemaphore?
  private var link: LinkEvent = .waiting
  private var ticker: Ticker?
  private var stopped = false

  public init(configuration: Configuration, preparer: any ModelPreparer) throws {
    self.configuration = configuration
    backend = CoreMLBackend(
      device: configuration.device, preparer: preparer, keepAlive: configuration.keepAlive, keepCPUWarm: configuration.keepCPUWarm)
    cache = try EngineCache(root: configuration.cacheRoot, backend: backend)
    host = EngineHost(cache: cache)
    // A write to a socket the comma closed must be an error, not a signal
    // that kills the app.
    signal(SIGPIPE, SIG_IGN)
  }

  /// Listens, and serves until `stop`. Returns once listening.
  public func start() throws {
    try listen()
    if configuration.preload {
      host.preload()
    }
    ticker = Ticker(interval: Server.statsInterval) { [weak self] _ in self?.tick() }
  }

  private func listen() throws {
    let listener = try TCPListener(host: configuration.host, port: configuration.port)
    lock.lock()
    self.listener = listener
    lock.unlock()
    log.info("listening on \(self.configuration.host, privacy: .public):\(listener.port)")
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

  private func listenerEnded(_ ended: TCPListener) {
    lock.lock()
    let wasCurrent = listener === ended
    if wasCurrent { listener = nil }
    let isStopped = stopped
    lock.unlock()
    if wasCurrent && !isStopped {
      log.warning("the listening socket went away")
      ended.close()
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
    host.close()
  }

  private func acceptLoop(_ listener: TCPListener) {
    while let transport = listener.accept() {
      lock.lock()
      let previous = current
      let previousDone = currentDone
      lock.unlock()
      if let previous {
        log.info("a new connection from \(transport.peer, privacy: .public) takes over from \(previous.peer, privacy: .public)")
        previous.interrupt()
        previousDone?.wait()
      }
      let session = Session(transport: transport, host: host) { [weak self] in self?.telemetry() ?? [:] }
      let done = DispatchSemaphore(value: 0)
      lock.lock()
      current = session
      currentDone = done
      lock.unlock()
      let thread = Thread { [self] in
        serve(session)
        done.signal()
      }
      thread.name = "jetlink-session"
      thread.qualityOfService = .userInteractive
      thread.start()
    }
  }

  private func serve(_ session: Session) {
    log.info("client connected from \(session.peer, privacy: .public)")
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
    log.info("client disconnected: \(reason, privacy: .public)")
    if isCurrent && !isStopped {
      setLink(LinkEvent(state: .disconnected, detail: reason, peer: nil))
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
