import Foundation

#if canImport(os)
  import os
#else
  import JetlinkLog
#endif

/// Where the server's log lines go besides the unified log: an app that
/// runs the server in process installs a sink and shows the lines itself.
public enum Log {
  public enum Level: String, Sendable {
    case info, warning, error
  }

  /// Called for every line, on the thread that logged it, with the level,
  /// the category (server, session, coreml, control) and the message.
  public static var sink: (@Sendable (Level, String, String) -> Void)? {
    get { lock.withLock { stored } }
    set { lock.withLock { stored = newValue } }
  }

  private static let lock = NSLock()
  nonisolated(unsafe) private static var stored: (@Sendable (Level, String, String) -> Void)?

  static func write(_ level: Level, _ category: String, _ message: String) {
    sink?(level, category, message)
  }
}

/// A category's logger: the unified log, and the sink when one is installed.
/// Messages are public: nothing the server logs is a person's.
struct ServerLog: Sendable {
  let category: String
  private let logger: Logger

  init(category: String) {
    self.category = category
    logger = Logger(subsystem: "io.zoompilot.jetlink", category: category)
  }

  func info(_ message: String) {
    logger.info("\(message, privacy: .public)")
    Log.write(.info, category, message)
  }

  func warning(_ message: String) {
    logger.warning("\(message, privacy: .public)")
    Log.write(.warning, category, message)
  }

  func error(_ message: String) {
    logger.error("\(message, privacy: .public)")
    Log.write(.error, category, message)
  }
}
