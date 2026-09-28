import Foundation
import JetlinkRegistry

/// The server's cache: JetlinkRegistry's `EngineCache`, which the Python
/// conformance fixtures pin, for the backend this server runs. The backend
/// names its artifacts (tag, suffix, file or directory) once, here.
///
///     <root>/engines/<sha16>.<tag><suffix>    the artifact
///     <root>/engines/<sha16>.<tag>.json       its sidecar: build facts and the model spec
///     <root>/models/<sha16>.onnx              the model as uploaded or downloaded
public final class EngineCache: Sendable {
  /// One per registry entry on a Mac or a Jetson: a rebuild costs minutes
  /// and 1 to 2 GB. A phone's disk holds two.
  #if os(iOS) || os(Android)
    public static let keepPlans = 2
  #else
    public static let keepPlans = CacheLayout.keepArtifacts
  #endif

  public let layout: CacheLayout
  public let backend: any EngineBackend
  private let store: JetlinkRegistry.EngineCache

  public init(root: URL, backend: any EngineBackend) throws {
    layout = CacheLayout(root: root)
    self.backend = backend
    for directory in [layout.engines, layout.models] {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    store = JetlinkRegistry.EngineCache(
      layout: layout, tag: backend.tag(), suffix: backend.suffix, backend: backend.name, kind: backend.artifactKind)
  }

  /// The identity arrives from a peer and becomes a path, so anything but a
  /// digest is refused.
  public func entry(_ sha256: String) throws(RegistryError) -> CacheEntry {
    try store.entry(sha256)
  }

  public func modelPath(_ sha256: String) throws(RegistryError) -> URL {
    try layout.modelPath(sha256: sha256)
  }

  /// A request's identity was checked when it was made.
  func entry(_ request: Request) -> CacheEntry {
    try! store.entry(request.sha256)
  }

  func modelPath(_ request: Request) -> URL {
    try! layout.modelPath(sha256: request.sha256)
  }

  /// Model identities with artifacts this backend can load on this device.
  public func inventory() -> [String] {
    store.inventory()
  }

  /// What is loaded, for the next start to preload.
  public func rememberLoaded(_ sha256: String, frameSkip: Int) {
    store.rememberLoaded(sha256: sha256, frameSkip: frameSkip)
  }

  public func lastLoaded() -> (sha256: String, frameSkip: Int)? {
    layout.lastLoaded().map { ($0.sha256, $0.frameSkip) }
  }

  public func forgetLastLoaded() {
    try? FileManager.default.removeItem(at: layout.lastLoadedURL)
  }

  /// Keep the newest few artifacts of this backend's kind. `protect` is never
  /// pruned whatever its mtime says.
  public func prune(keep: Int = EngineCache.keepPlans, protect: URL? = nil) {
    store.prune(keep: keep, protect: protect)
  }

  /// Build directories a killed build left behind.
  public func sweepTemp(maxAge: TimeInterval = 6 * 3600) {
    store.sweepTemp(maxAge: maxAge)
  }
}

extension CacheEntry {
  /// The sidecar. Throws when it is missing or unreadable.
  func meta() throws -> [String: Any] {
    let data = try Data(contentsOf: metaPath)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return object
  }

  func writeMeta(_ meta: [String: Any]) throws {
    try Artifact.write(meta, to: metaPath)
  }
}
