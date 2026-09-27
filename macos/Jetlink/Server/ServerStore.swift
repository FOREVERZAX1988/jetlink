import Foundation
import JetlinkKit
import JetlinkServer
import Observation
import os

/// What the running server says about itself, for Status and the menu bar.
struct ServerInfo: Equatable, Sendable {
  let version: String
  let backend: String
  let runtimeVersion: String
  let device: String
  let cache: String
  let transport: String
  let port: Int?

  init(version: String, backend: String, runtimeVersion: String, device: String, cache: String, transport: String, port: Int?) {
    self.version = version
    self.backend = backend
    self.runtimeVersion = runtimeVersion
    self.device = device
    self.cache = cache
    self.transport = transport
    self.port = port
  }
}

enum ServerStoreError: Error, LocalizedError, Equatable {
  case notRunning
  case failed(String)

  var errorDescription: String? {
    switch self {
    case .notRunning: return "The server is not running."
    case .failed(let detail): return detail
    }
  }
}

/// Owns the server, and everything the Status view shows. The server is the
/// Swift one the iPhone app runs, in this process through `EmbeddedServer`:
/// nothing to find or start, and the control channel's events without a
/// socket. It cannot crash apart from the app, so there is nothing to restart
/// behind the user's back.
@MainActor
@Observable
final class ServerStore: ServerControlling {
  private(set) var runState: ServerRunState = .stopped
  private(set) var info: ServerInfo?
  private(set) var link: LinkEvent = .waiting
  private(set) var engine: EngineEvent = .none
  /// The last two minutes of `stats` events, oldest first, while the comma stays connected.
  private(set) var statsHistory: [StatsSample] = []
  var stats: StatsEvent? { statsHistory.last?.stats }
  private(set) var startedAt: Date?
  /// The benchmark running or last run, if any.
  private(set) var benchmark: BenchmarkEvent?
  var lastFailure: String?

  let settings: AppSettings
  let logs: LogBuffer
  /// Every event a `ModelStore` cares about: inventory, catalog, download, import.
  let modelEvents: AsyncStream<ControlEvent>

  @ObservationIgnored private let modelEventsContinuation: AsyncStream<ControlEvent>.Continuation
  @ObservationIgnored private let logFile: LogFileWriter?
  @ObservationIgnored private let sleepAssertion: SleepAssertion
  @ObservationIgnored private let isLive: Bool
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "server")
  @ObservationIgnored private var embedded: EmbeddedServer?
  @ObservationIgnored private var consumeTask: Task<Void, Never>?

  init(settings: AppSettings, logs: LogBuffer, logFile: LogFileWriter? = LogFileWriter(), isLive: Bool = true) {
    self.settings = settings
    self.logs = logs
    self.logFile = logFile
    self.isLive = isLive
    self.sleepAssertion = SleepAssertion()
    let (stream, continuation) = AsyncStream<ControlEvent>.makeStream(bufferingPolicy: .unbounded)
    self.modelEvents = stream
    self.modelEventsContinuation = continuation
    if isLive {
      sleepAssertion.onPowerSourceChange = { [weak self] in self?.updateSleepAssertion() }
      sleepAssertion.startObservingPowerSource()
    }
  }

  // MARK: lifecycle

  func start() {
    guard isLive else { return }
    switch runState {
    case .stopped, .failed: break
    default: return
    }
    runState = .starting
    lastFailure = nil
    let buffer = logs
    let file = logFile
    Log.sink = { level, category, message in
      let line = EmbeddedServer.logLine(level, category, message)
      Task { @MainActor in buffer.append(line) }
      if let file { Task { await file.append(line) } }
    }
    let configuration = ServerStore.configuration(
      backend: settings.backend, transport: settings.transport, tcpPort: settings.tcpPort, cacheDirectory: settings.cacheDirectory)
    do {
      let embedded = try EmbeddedServer(configuration: configuration)
      self.embedded = embedded
      consumeTask = Task { [weak self] in
        for await event in embedded.events {
          self?.apply(event)
        }
      }
      let described = embedded.server.backend.describe()
      info = ServerInfo(
        version: ServerStore.appVersion,
        backend: described["backend"] ?? "",
        runtimeVersion: described["runtime_version"] ?? "",
        device: described["device"] ?? "",
        cache: settings.cacheDirectory.path(percentEncoded: false),
        transport: settings.transport.rawValue,
        port: configuration.listen ? Int(configuration.port) : nil)
      try embedded.start()
      runState = .serving
      startedAt = Date()
    } catch {
      tearDown()
      let detail = "The server could not start: \(error)"
      log.error("\(detail, privacy: .public)")
      lastFailure = detail
      runState = .failed(detail)
    }
    updateSleepAssertion()
  }

  func stop() {
    Task { await stopAndWait() }
  }

  /// Stops the server and lets the engine go, off the main thread: releasing
  /// a CoreML model can take a moment. The app awaits this when it quits.
  func stopAndWait() async {
    guard isLive else { return }
    switch runState {
    case .stopped, .stopping: return
    default: break
    }
    runState = .stopping
    if let embedded {
      await Task.detached { embedded.stop(releasingEngine: true) }.value
    }
    tearDown()
    runState = .stopped
    updateSleepAssertion()
  }

  private func tearDown() {
    embedded = nil
    consumeTask?.cancel()
    consumeTask = nil
    Log.sink = nil
    resetLiveState()
  }

  func restart() {
    Task {
      await stopAndWait()
      start()
    }
  }

  func send(_ command: ControlCommand) async throws -> ReplyEvent {
    guard let embedded else { throw ServerStoreError.notRunning }
    return await embedded.handle(command)
  }

  /// Starts the server if it is not serving, and waits until it is.
  func startIfNeeded() async throws {
    guard isLive else { throw ServerStoreError.notRunning }
    if case .serving = runState { return }
    switch runState {
    case .stopped, .failed: start()
    default: break
    }
    let deadline = Date().addingTimeInterval(30)
    while Date() < deadline {
      switch runState {
      case .serving: return
      case .failed(let detail): throw ServerStoreError.failed(detail)
      default: break
      }
      try? await Task.sleep(for: .milliseconds(100))
    }
    throw ServerStoreError.failed("The server did not start in time.")
  }

  // MARK: events

  func apply(_ event: ControlEvent) {
    switch event {
    case .server(let server):
      if let current = info {
        info = ServerInfo(
          version: current.version,
          backend: server.backend ?? current.backend,
          runtimeVersion: server.runtimeVersion ?? current.runtimeVersion,
          device: server.device ?? current.device,
          cache: current.cache,
          transport: current.transport,
          port: current.port)
      }
    case .link(let value):
      link = value
      if value.state != .connected { statsHistory = [] }
    case .engine(let value):
      engine = value
    case .stats(let value):
      statsHistory = StatsSample.appending(value, to: statsHistory)
    case .benchmark(let value):
      benchmark = value
    case .shutdownRequest(let value):
      log.warning("the comma asked this Mac to power off (\(value.reason, privacy: .public)); a Mac does not")
    case .hello, .reply:
      break
    default:
      modelEventsContinuation.yield(event)
    }
  }

  // MARK: configuration

  /// What the server is asked to be, from the settings.
  nonisolated static func configuration(backend: BackendChoice, transport: TransportChoice, tcpPort: Int, cacheDirectory: URL)
    -> Server.Configuration
  {
    let port = UInt16(clamping: tcpPort > 0 ? tcpPort : AppSettings.defaultTCPPort)
    return Server.Configuration(
      port: port, cacheRoot: cacheDirectory, device: backend.device, keepAlive: true, keepCPUWarm: true, preload: true,
      listen: transport == .tcp, usb: transport == .usb)
  }

  nonisolated static var appVersion: String {
    Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
  }

  private func resetLiveState() {
    link = .waiting
    engine = .none
    statsHistory = []
    startedAt = nil
  }

  private func updateSleepAssertion() {
    guard isLive else { return }
    let wanted: Bool
    if case .serving = runState {
      wanted = settings.keepAwakeWhileServing && sleepAssertion.isOnACPower
    } else {
      wanted = false
    }
    sleepAssertion.setActive(wanted)
  }

  /// Called by the settings view when keepAwakeWhileServing changes.
  func keepAwakeSettingChanged() {
    updateSleepAssertion()
  }
}

extension ServerStore {
  /// A store with fixed state and nothing behind it. Actions are no-ops.
  static func preview(
    runState: ServerRunState = .serving,
    info: ServerInfo? = nil,
    link: LinkEvent,
    engine: EngineEvent,
    stats: StatsEvent? = nil,
    statsHistory: [StatsSample] = []
  ) -> ServerStore {
    let store = ServerStore(settings: AppSettings.preview(), logs: LogBuffer(), logFile: nil, isLive: false)
    store.runState = runState
    store.info = info
    store.link = link
    store.engine = engine
    // One sample of `stats` unless a history is given, which ends with its own.
    store.statsHistory = statsHistory.isEmpty ? stats.map { [StatsSample(at: Date(), stats: $0)] } ?? [] : statsHistory
    store.startedAt = Date(timeIntervalSinceNow: -3600)
    return store
  }
}
