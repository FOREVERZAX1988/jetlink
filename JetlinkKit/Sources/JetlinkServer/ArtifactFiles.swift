import Foundation
import JetlinkKit

/// What every onnxruntime backend keeps beside an artifact: the sidecar JSON
/// that says how it was built and how long it took, which the next build and
/// load read for their progress estimates.
enum ArtifactSidecar {
  /// `<artifact>.json`, or empty when there is none.
  static func read(_ artifact: URL) -> [String: Any] {
    let url = artifact.deletingPathExtension().appendingPathExtension("json")
    guard let data = try? Data(contentsOf: url) else { return [:] }
    return ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any]) ?? [:]
  }

  static func write(_ artifact: URL, _ meta: [String: Any]) throws {
    let url = artifact.deletingPathExtension().appendingPathExtension("json")
    let data = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: url, options: .atomic)
  }

  /// The bytes of every regular file under `root`.
  static func treeBytes(_ root: URL) -> Int64 {
    guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return 0 }
    var total: Int64 = 0
    for case let file as URL in walker {
      guard let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), values.isRegularFile == true else { continue }
      total += Int64(values.fileSize ?? 0)
    }
    return total
  }
}

/// An onnxruntime artifact as both backends build and load it: a directory
/// holding each session's model and a `sessions.json` manifest naming them,
/// with the sidecar beside it (CoreMLBackend, QNNBackend).
enum OrtArtifact {
  static let manifestName = "sessions.json"

  /// Builds into a directory beside `artifact`, where the cache's sweep
  /// finds it if the build is killed, and moves it into place over any old
  /// one once `body` returns. The staging goes either way.
  static func build<T>(_ artifact: URL, _ body: (_ staged: URL) throws -> T) throws -> T {
    let fm = FileManager.default
    let parent = artifact.deletingLastPathComponent()
    try fm.createDirectory(at: parent, withIntermediateDirectories: true)
    let temp = parent.appending(path: "tmp\(UUID().uuidString.prefix(8))", directoryHint: .isDirectory)
    let staged = temp.appending(path: "artifact", directoryHint: .isDirectory)
    try fm.createDirectory(at: staged, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: temp) }
    let result = try body(staged)
    if fm.fileExists(atPath: artifact.path) {
      try fm.removeItem(at: artifact)
    }
    try fm.moveItem(at: staged, to: artifact)
    return result
  }

  static func writeManifest(_ manifest: [[String: Any]], in directory: URL) throws {
    try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted]).write(to: directory.appending(path: manifestName))
  }

  /// A built artifact's manifest and sidecar. Throws `ArtifactInvalid`,
  /// which rebuilds it, when the manifest is unreadable, the preparation was
  /// another version, a session's model is missing, or `check` says what is
  /// wrong with an entry.
  static func open(
    _ artifact: URL, prepareVersion: Int, builds: String, check: ([String: Any]) -> String? = { _ in nil }
  ) throws -> (manifest: [[String: Any]], meta: [String: Any]) {
    let name = artifact.lastPathComponent
    guard let data = try? Data(contentsOf: artifact.appending(path: manifestName)),
      let manifest = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]], !manifest.isEmpty
    else {
      throw ArtifactInvalid("\(name): no readable \(manifestName) inside")
    }
    let meta = ArtifactSidecar.read(artifact)
    let version = (meta["prepare"] as? NSNumber)?.intValue ?? 1
    if version != prepareVersion {
      throw ArtifactInvalid("\(name): prepared as version \(version), \(builds) builds are now at \(prepareVersion); rebuilding")
    }
    for entry in manifest {
      guard let model = entry["model"] as? String, FileManager.default.fileExists(atPath: artifact.appending(path: model).path) else {
        throw ArtifactInvalid("\(name): a session's model is missing")
      }
      if let problem = check(entry) {
        throw ArtifactInvalid("\(name): \(problem)")
      }
    }
    return (manifest, meta)
  }

  /// Loads with progress paced by the last load's time, then records this
  /// one's in the sidecar for the next.
  static func load(
    _ artifact: URL, meta: [String: Any], what: String, report: @escaping ProgressFn, _ body: () throws -> OrtEngine
  ) throws -> (engine: OrtEngine, seconds: TimeInterval) {
    let started = Date()
    let took = (meta["load_seconds"] as? NSNumber)?.doubleValue ?? 0
    report("load", 0, "loading \(what)")
    let engine = try Ticker.during(interval: 1, { elapsed in
      if took > 0 {
        report("load", min(0.95, elapsed / took), "loading \(what), \(Int(elapsed)) s of about \(Int(took.rounded())) s")
      } else {
        report("load", 0, "loading \(what), \(Int(elapsed)) s elapsed")
      }
    }, body)
    let seconds = Date().timeIntervalSince(started)
    report("load", 1, "loaded in \(Int(seconds.rounded())) s")
    if !meta.isEmpty {
      var updated = meta
      updated["load_seconds"] = pythonRound(seconds, 1)
      try? ArtifactSidecar.write(artifact, updated)
    }
    return (engine, seconds)
  }

  /// The sidecar keys every build writes, as the Python backend names them.
  static func meta(
    _ backend: any EngineBackend, manifest: [[String: Any]], providers: [String], model: URL, prepareVersion: Int, started: Date
  ) -> [String: Any] {
    [
      "backend": backend.name,
      "onnxruntime": backend.runtimeVersion,
      "device": backend.deviceTag(),
      "sessions": manifest,
      "providers": providers,
      "build_seconds": pythonRound(Date().timeIntervalSince(started), 1),
      "onnx": model.lastPathComponent,
      "prepare": prepareVersion,
      "preparer": "swift",
      "built_at": ISO8601DateFormatter().string(from: Date()),
    ]
  }
}

/// Calls `body(elapsed)` every `interval` seconds on its own thread until stopped.
final class Ticker: @unchecked Sendable {
  private let condition = NSCondition()
  private var stopped = false

  init(interval: TimeInterval, _ body: @escaping @Sendable (TimeInterval) -> Void) {
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

  func stop() {
    condition.lock()
    stopped = true
    condition.signal()
    condition.unlock()
  }
}

func formatBytes(_ n: Int64) -> String {
  n >= 1_000_000_000 ? String(format: "%.1f GB", Double(n) / 1e9) : String(format: "%.0f MB", Double(n) / 1e6)
}

extension Ticker {
  /// Runs `body` while ticking every `interval`, and stops the ticks however
  /// it ends.
  static func during<T>(interval: TimeInterval, _ tick: @escaping @Sendable (TimeInterval) -> Void, _ body: () throws -> T) rethrows -> T {
    let ticker = Ticker(interval: interval, tick)
    defer { ticker.stop() }
    return try body()
  }
}
