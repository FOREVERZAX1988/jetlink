import Foundation
import JetlinkKit
import JetlinkRegistry

/// The jetlink server in process, as an app runs it: the server, its control
/// semantics, and the model registry on the same cache. The iPhone's
/// `PhoneServer` and the Mac's `ServerStore` each drive one, so both apps run
/// the same code between their views and the comma.
public final class EmbeddedServer: @unchecked Sendable {
  public let server: Server
  public let controller: ServerController

  /// The control channel's events, as the Python server's socket sends them.
  public var events: AsyncStream<ControlEvent> { controller.events }

  /// Builds the server on `configuration`, creating its cache directory.
  public init(configuration: Server.Configuration) throws {
    try FileManager.default.createDirectory(at: configuration.cacheRoot, withIntermediateDirectories: true)
    server = try Server(configuration: configuration, preparer: ONNXPreparer())
    controller = ServerController(server: server, registry: Registry(layout: CacheLayout(root: configuration.cacheRoot)))
  }

  /// Serves, and publishes the state a client's first screen needs.
  /// Subscribe to `events` first: nothing published before is kept.
  public func start() throws {
    try server.start()
    controller.publishInitialState()
  }

  public func handle(_ command: ControlCommand) async -> ReplyEvent {
    await controller.handle(command)
  }

  /// Stops serving and ends `events`. With `releasingEngine` the engine is
  /// let go at once, as when the app ends; without it the engine goes with
  /// the last reference to this server.
  public func stop(releasingEngine: Bool) {
    controller.finish()
    if releasingEngine {
      server.shutdown()
    } else {
      server.stop()
    }
  }

  /// A log line as the Python server writes them,
  /// "%(asctime)s %(levelname)-7s %(name)s: %(message)s", so one Logs view
  /// and one log file read the same whichever server wrote them.
  public static func logLine(_ level: Log.Level, _ category: String, _ message: String, at date: Date = Date()) -> String {
    let name: String
    switch level {
    case .info: name = "INFO"
    case .warning: name = "WARNING"
    case .error: name = "ERROR"
    }
    return "\(timestamp(date)) \(name.padding(toLength: 7, withPad: " ", startingAt: 0)) jetlink.\(category): \(message)"
  }

  /// "2026-09-27 12:53:24,982" in local time, as Python's logging writes it.
  static func timestamp(_ date: Date) -> String {
    var time = time_t(date.timeIntervalSince1970.rounded(.down))
    var parts = tm()
    localtime_r(&time, &parts)
    let millis = Int((date.timeIntervalSince1970 - Double(time)) * 1000)
    return String(
      format: "%04d-%02d-%02d %02d:%02d:%02d,%03d", Int(parts.tm_year) + 1900, Int(parts.tm_mon) + 1, Int(parts.tm_mday), Int(parts.tm_hour),
      Int(parts.tm_min), Int(parts.tm_sec), min(millis, 999))
  }
}
