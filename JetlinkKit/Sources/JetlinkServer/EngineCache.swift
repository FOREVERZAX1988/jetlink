import Foundation

/// One artifact and its sidecar, as `server/cache.py`'s CacheEntry.
public struct CacheEntry: Sendable {
  /// The artifact: a directory for onnxruntime.
  public let path: URL
  public let metaPath: URL

  public var exists: Bool {
    FileManager.default.fileExists(atPath: path.path) && FileManager.default.fileExists(atPath: metaPath.path)
  }

  public func meta() throws -> [String: Any] {
    let data = try Data(contentsOf: metaPath)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return object
  }

  public func writeMeta(_ meta: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: metaPath, options: .atomic)
  }

  public func remove() {
    try? FileManager.default.removeItem(at: path)
    try? FileManager.default.removeItem(at: metaPath)
  }
}

/// Where built engines and models live, laid out as `server/cache.py` lays
/// them out:
///
///     <root>/engines/<sha16>.<tag><suffix>    the artifact
///     <root>/engines/<sha16>.<tag>.json       its sidecar: build facts and the model spec
///     <root>/models/<sha16>.onnx              the model as uploaded or downloaded
public final class EngineCache: @unchecked Sendable {
  /// One per registry entry: a rebuild costs a minute and 2 to 3 GB.
  public static let keepPlans = 6
  public static let lastLoadedName = "last-loaded.json"

  public let root: URL
  public let engines: URL
  public let models: URL
  public let backend: any EngineBackend

  public init(root: URL, backend: any EngineBackend) throws {
    self.root = root
    self.backend = backend
    engines = root.appending(path: "engines", directoryHint: .isDirectory)
    models = root.appending(path: "models", directoryHint: .isDirectory)
    for directory in [engines, models] {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
  }

  public static func isSHA256(_ value: String) -> Bool {
    value.count == 64 && value.allSatisfy { ("0"..."9").contains($0) || ("a"..."f").contains($0) }
  }

  public func key(_ sha256: String) -> String {
    "\(sha256.prefix(16)).\(backend.tag())"
  }

  public func entry(_ sha256: String) -> CacheEntry {
    let key = key(sha256)
    return CacheEntry(
      path: engines.appending(path: key + backend.suffix, directoryHint: .isDirectory),
      metaPath: engines.appending(path: key + ".json"))
  }

  public func modelPath(_ sha256: String) -> URL {
    models.appending(path: "\(sha256.prefix(16)).onnx")
  }

  /// Model identities with artifacts this backend can load on this device.
  public func inventory() -> [String] {
    let files = (try? FileManager.default.contentsOfDirectory(at: engines, includingPropertiesForKeys: nil)) ?? []
    var found = Set<String>()
    for meta in files where meta.pathExtension == "json" {
      guard let data = try? Data(contentsOf: meta),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let spec = object["spec"] as? [String: Any],
            let sha = spec["sha256"] as? String, EngineCache.isSHA256(sha)
      else { continue }
      let entry = entry(sha)
      if entry.metaPath.standardizedFileURL == meta.standardizedFileURL && entry.exists {
        found.insert(sha)
      }
    }
    return found.sorted()
  }

  /// What is loaded, for the next start to preload. frame_skip goes with it:
  /// the spec served is stamped with it.
  public func rememberLoaded(_ sha256: String, frameSkip: Int) {
    let object: [String: Any] = ["sha256": sha256, "frame_skip": frameSkip, "backend": backend.name]
    if let data = try? JSONSerialization.data(withJSONObject: object) {
      try? data.write(to: root.appending(path: EngineCache.lastLoadedName), options: .atomic)
    }
  }

  public func lastLoaded() -> (sha256: String, frameSkip: Int)? {
    guard let data = try? Data(contentsOf: root.appending(path: EngineCache.lastLoadedName)),
          let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let sha = object["sha256"] as? String, EngineCache.isSHA256(sha),
          let skip = (object["frame_skip"] as? NSNumber)?.intValue
    else { return nil }
    return (sha, skip)
  }

  public func forgetLastLoaded() {
    try? FileManager.default.removeItem(at: root.appending(path: EngineCache.lastLoadedName))
  }

  /// Keep the newest few artifacts of this backend's kind. `protect` is never
  /// pruned whatever its mtime says.
  public func prune(keep: Int = EngineCache.keepPlans, protect: URL? = nil) {
    let keys: [URLResourceKey] = [.contentModificationDateKey]
    let files = (try? FileManager.default.contentsOfDirectory(at: engines, includingPropertiesForKeys: keys)) ?? []
    var found = files.filter { $0.lastPathComponent.hasSuffix(backend.suffix) && $0.standardizedFileURL != protect?.standardizedFileURL }
    func modified(_ url: URL) -> Date {
      (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
    found.sort { modified($0) > modified($1) }
    let allowed = max(keep - (protect == nil ? 0 : 1), 0)
    for artifact in found.dropFirst(allowed) {
      let meta = artifact.deletingPathExtension().appendingPathExtension("json")
      CacheEntry(path: artifact, metaPath: meta).remove()
    }
  }

  /// Build directories a killed build left behind.
  public func sweepTemp(maxAge: TimeInterval = 6 * 3600) {
    let keys: [URLResourceKey] = [.contentModificationDateKey, .isDirectoryKey]
    let files = (try? FileManager.default.contentsOfDirectory(at: engines, includingPropertiesForKeys: keys)) ?? []
    for directory in files where directory.lastPathComponent.hasPrefix("tmp") {
      let values = try? directory.resourceValues(forKeys: Set(keys))
      guard values?.isDirectory == true else { continue }
      if let modified = values?.contentModificationDate, Date().timeIntervalSince(modified) < maxAge { continue }
      try? FileManager.default.removeItem(at: directory)
    }
  }
}
