import Foundation

/// What every backend keeps beside an artifact: the sidecar JSON that says how
/// it was built and how long it took, which the next build and load read for
/// their progress estimates.
package enum ArtifactSidecar {
  /// `<artifact>.json`, or empty when there is none.
  package static func read(_ artifact: URL) -> [String: Any] {
    let url = artifact.deletingPathExtension().appendingPathExtension("json")
    guard let data = try? Data(contentsOf: url) else { return [:] }
    return ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any]) ?? [:]
  }

  package static func write(_ artifact: URL, _ meta: [String: Any]) throws {
    let url = artifact.deletingPathExtension().appendingPathExtension("json")
    let data = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: url, options: .atomic)
  }

  /// The bytes of every regular file under `root`.
  package static func treeBytes(_ root: URL) -> Int64 {
    guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return 0 }
    var total: Int64 = 0
    for case let file as URL in walker {
      guard let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), values.isRegularFile == true else { continue }
      total += Int64(values.fileSize ?? 0)
    }
    return total
  }
}

/// Calls `body(elapsed)` every `interval` seconds on its own thread until stopped.
package final class Ticker: @unchecked Sendable {
  private let condition = NSCondition()
  private var stopped = false

  package init(interval: TimeInterval, _ body: @escaping @Sendable (TimeInterval) -> Void) {
    let started = Date()
    let thread = Thread { [self] in
      while true {
        condition.lock()
        if !stopped {
          _ = condition.wait(until: Date().addingTimeInterval(interval))
        }
        let done = stopped
        condition.unlock()
        if done { return }
        body(Date().timeIntervalSince(started))
      }
    }
    thread.name = "jetlink-progress"
    thread.start()
  }

  package func stop() {
    condition.lock()
    stopped = true
    condition.signal()
    condition.unlock()
  }
}

package func formatBytes(_ n: Int64) -> String {
  n >= 1_000_000_000 ? String(format: "%.1f GB", Double(n) / 1e9) : String(format: "%.0f MB", Double(n) / 1e6)
}

extension Ticker {
  /// Runs `body` while ticking every `interval`, and stops the ticks however
  /// it ends.
  package static func during<T>(interval: TimeInterval, _ tick: @escaping @Sendable (TimeInterval) -> Void, _ body: () throws -> T) rethrows -> T {
    let ticker = Ticker(interval: interval, tick)
    defer { ticker.stop() }
    return try body()
  }
}
