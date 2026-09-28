#if !canImport(os)
  import Foundation

  /// `os.Logger`'s shape where there is no unified log: Linux and Android. The
  /// same call sites compile, privacy arguments included, and write to
  /// standard error. On Apple platforms this file is empty and the modules
  /// import `os` itself.
  public struct Logger: Sendable {
    private let label: String

    public init(subsystem: String, category: String) {
      label = "\(subsystem).\(category)"
    }

    public func debug(_ message: LogMessage) {}
    public func info(_ message: LogMessage) { write("INFO", message) }
    public func notice(_ message: LogMessage) { write("NOTICE", message) }
    public func warning(_ message: LogMessage) { write("WARNING", message) }
    public func error(_ message: LogMessage) { write("ERROR", message) }

    private func write(_ level: String, _ message: LogMessage) {
      FileHandle.standardError.write(Data("\(level) \(label): \(message.text)\n".utf8))
    }
  }

  /// What `os.Logger` takes: a string whose interpolations may name a privacy.
  public struct LogMessage: ExpressibleByStringInterpolation, Sendable {
    public let text: String

    public init(stringLiteral value: String) {
      text = value
    }

    public init(stringInterpolation: Interpolation) {
      text = stringInterpolation.text
    }

    public struct Interpolation: StringInterpolationProtocol {
      var text = ""

      public init(literalCapacity: Int, interpolationCount: Int) {
        text.reserveCapacity(literalCapacity)
      }

      public mutating func appendLiteral(_ literal: String) {
        text += literal
      }

      public mutating func appendInterpolation<T>(_ value: T, privacy: LogPrivacy = .auto) {
        text += String(describing: value)
      }
    }
  }

  /// `OSLogPrivacy`'s cases, which mean nothing on standard error.
  public enum LogPrivacy: Sendable {
    case auto
    case `public`
    case `private`
  }
#endif
