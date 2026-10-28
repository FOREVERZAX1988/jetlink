import Foundation

/// What a machine has free, for the one step that can exhaust it.
///
/// Compiling a model for the phone's NPU is the peak of the whole build:
/// onnxruntime holds the weights while the QNN provider converts the graph, so
/// a phone with too little free memory does not fail the build -- the app is
/// killed under it, and the comma only sees the link go (reported 2026-10-17:
/// the app closes itself while a large model compiles). Answering before the
/// ask turns that into a sentence the app hands back instead.
package enum HostMemory {
  /// Bytes a process could have without swapping, or nil where the platform
  /// will not say. Android's /proc/meminfo is the one that does; iOS keeps it
  /// to itself, and there a build that runs out is killed the same way.
  package static func available() -> Int64? {
    guard let text = try? String(contentsOfFile: "/proc/meminfo", encoding: .utf8) else { return nil }
    for line in text.split(separator: "\n") where line.hasPrefix("MemAvailable:") {
      let fields = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
      guard fields.count >= 2, let kB = Int64(fields[1]) else { return nil }
      return kB * 1024
    }
    return nil
  }

  /// What a compile of `bytes` of model needs free: the weights are held twice
  /// (onnxruntime's copy and the provider's) with the converted graph beside
  /// them, so twice the model is the floor this refuses under, and never less
  /// than half a gigabyte.
  package static func needed(for bytes: Int64) -> Int64 { max(512 << 20, 2 * bytes) }

  /// Why a compile of `bytes` of model would not fit, or nil when it fits or
  /// when the platform will not say how much there is.
  package static func shortage(for bytes: Int64) -> String? {
    guard let free = available(), free < needed(for: bytes) else { return nil }
    return "compiling needs about \(needed(for: bytes) >> 20) MB free and this phone has \(free >> 20) MB: "
      + "close other apps and build again, or run this model on a phone with more memory"
  }
}
