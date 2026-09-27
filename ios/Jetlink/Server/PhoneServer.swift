import Foundation
import JetlinkKit
import JetlinkRegistry
import JetlinkServer
import Observation
import os

/// The jetlink server inside the app, and everything the views read about it.
/// The iPhone's `ServerStore`: the same events and commands the Mac app has,
/// with the server in process instead of a Python child behind a socket.
@MainActor
@Observable
final class PhoneServer: ServerControlling {
  private(set) var runState: ServerRunState = .stopped
  private(set) var link: LinkEvent = .waiting
  private(set) var engine: EngineEvent = .none
  /// backend, runtime version and device, as the hello reports them.
  private(set) var info: ServerEvent?
  private(set) var port: UInt16?
  /// The last ten seconds of frames, refreshed once a second while serving.
  private(set) var recent: StatsEvent?
  /// The last two minutes of `stats`, oldest first, while the comma stays connected.
  private(set) var statsHistory: [StatsSample] = []
  private(set) var lastFailure: String?
  /// The benchmark running or last run, if any.
  private(set) var benchmark: BenchmarkEvent?
  let modelEvents: AsyncStream<ControlEvent>

  /// How far back the dashboard's headline reaches.
  static let recentWindow: TimeInterval = 10

  let settings: PhoneSettings
  @ObservationIgnored private let modelEventsContinuation: AsyncStream<ControlEvent>.Continuation
  @ObservationIgnored private var server: Server?
  @ObservationIgnored private var controller: ServerController?
  @ObservationIgnored private var consumeTask: Task<Void, Never>?
  @ObservationIgnored private var recentTask: Task<Void, Never>?
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "app")

  /// What the in-process server logs, for a Logs view.
  let logs = LogBuffer()

  init(settings: PhoneSettings) {
    self.settings = settings
    (modelEvents, modelEventsContinuation) = AsyncStream.makeStream(bufferingPolicy: .unbounded)
    let logs = self.logs
    let clock = ISO8601DateFormatter()
    clock.formatOptions = [.withFullTime, .withFractionalSeconds]
    Log.sink = { level, category, message in
      let line = "\(clock.string(from: Date())) \(level.rawValue.uppercased()) \(category): \(message)"
      Task { @MainActor in logs.append(line) }
    }
  }

  // MARK: lifecycle

  func start() {
    switch runState {
    case .stopped, .failed: break
    default: return
    }
    runState = .starting
    lastFailure = nil
    do {
      let root = try PhoneServer.prepareCacheDirectory()
      let server = try Server(
        configuration: Server.Configuration(
          port: settings.port, cacheRoot: root, device: settings.device, keepAlive: settings.keepGPUAwake),
        preparer: ONNXPreparer())
      let controller = ServerController(server: server, registry: Registry(layout: CacheLayout(root: root)))
      self.server = server
      self.controller = controller
      consumeTask = Task { [weak self] in
        for await event in controller.events {
          self?.apply(event)
        }
      }
      try server.start()
      port = server.port
      controller.publishInitialState()
      runState = .serving
      startRecentTicker()
      log.info("serving on port \(server.port ?? 0)")
    } catch {
      fail("Jetlink could not start its server: \(error)")
    }
  }

  /// Stops serving. The engine stays loaded in the server until it is let
  /// go; `shutdown()` is for the app's end.
  func stop() {
    tearDown(release: false)
  }

  /// The app is terminating: stop, and release the engine with it.
  func shutdown() {
    tearDown(release: true)
  }

  private func tearDown(release: Bool) {
    recentTask?.cancel()
    recentTask = nil
    controller?.finish()
    if release {
      server?.shutdown()
    } else {
      server?.stop()
    }
    consumeTask?.cancel()
    consumeTask = nil
    server = nil
    controller = nil
    runState = .stopped
    resetLiveState()
  }

  /// A change of port or compute device takes a new server; the engine is
  /// loaded again, or prepared again for a new device.
  func restart() {
    stop()
    start()
  }

  /// iOS takes a suspended app's listening socket; listen again on the way back.
  func becameActive() {
    guard runState == .serving, let server, !server.isListening else { return }
    do {
      try server.reopenListener()
      log.info("listening again after the app was suspended")
    } catch {
      fail("Jetlink could not listen again after being in the background: \(error)")
    }
  }

  private func fail(_ detail: String) {
    log.error("\(detail, privacy: .public)")
    lastFailure = detail
    runState = .failed(detail)
    server?.stop()
    server = nil
    controller = nil
    resetLiveState()
  }

  private static func prepareCacheDirectory() throws -> URL {
    var root = PhoneSettings.cacheDirectory
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    // Gigabytes of models and engines have no place in an iCloud backup.
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try? root.setResourceValues(values)
    return root
  }

  private func startRecentTicker() {
    recentTask?.cancel()
    recentTask = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(1))
        guard let self, let server = self.server else { return }
        self.recent = self.link.state == .connected ? server.recentStats(window: PhoneServer.recentWindow) : nil
      }
    }
  }

  // MARK: ServerControlling

  func send(_ command: ControlCommand) async throws -> ReplyEvent {
    guard let controller else { throw ServerUnavailable() }
    return await controller.handle(command)
  }

  func startIfNeeded() async throws {
    if runState != .serving {
      start()
    }
    if case .failed(let detail) = runState {
      throw ServerUnavailable(detail: detail)
    }
  }

  struct ServerUnavailable: LocalizedError {
    var detail = "The server is not running."
    var errorDescription: String? { detail }
  }

  // MARK: events

  func apply(_ event: ControlEvent) {
    switch event {
    case .server(let value):
      info = value
    case .link(let value):
      link = value
      if value.state != .connected {
        recent = nil
        statsHistory = []
      }
    case .engine(let value):
      engine = value
    case .stats(let value):
      statsHistory = StatsSample.appending(value, to: statsHistory)
    case .benchmark(let value):
      benchmark = value
    case .hello, .reply, .unknown:
      break
    case .inventory, .catalog, .download, .importEvent:
      modelEventsContinuation.yield(event)
    }
  }

  private func resetLiveState() {
    link = .waiting
    engine = .none
    recent = nil
    statsHistory = []
    port = nil
  }
}
