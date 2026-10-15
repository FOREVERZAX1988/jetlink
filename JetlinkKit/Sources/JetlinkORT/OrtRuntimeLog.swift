import COrt
import Foundation
import JetlinkServer

/// The runtime's own words, for the `npu:` report: onnxruntime's logger, which
/// carries the execution provider's lines and QNN's beneath them, kept in
/// COrt's ring. An Android app's stderr has no reader, so a provider that took
/// no node of a graph said why to itself unless something reads this back --
/// and that reason is the first thing a report is for.
enum OrtRuntimeLog {
  /// Hands the ring to the report. JetlinkServer knows no onnxruntime: the
  /// lines arrive as this closure, and only where a backend that has one ran.
  static func install() {
    HostDiagnostics.runtimeLines = { OrtRuntimeLog.lines }
  }

  /// Oldest first, as the runtime wrote them, the last of the process so far.
  static var lines: [String] {
    String(cString: jl_log_lines()).split(separator: "\n").map(String.init)
  }
}
