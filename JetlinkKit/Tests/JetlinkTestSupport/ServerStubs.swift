import Foundation
import JetlinkRegistry
import JetlinkServer

/// A backend that is only its names: all a cache, a server and its
/// controller ask of one before any model. TensorRT's plan files, or
/// onnxruntime's directories.
public final class NamingBackend: EngineBackend {
  public let name: String
  public let suffix: String
  public let artifactKind: ArtifactKind
  public let runtimeVersion: String

  public init(kind: ArtifactKind = .directory) {
    artifactKind = kind
    (name, suffix, runtimeVersion) = kind == .file ? ("trt", ".plan", "10.3.0") : ("ort", ".ortcache", "1.29.0")
  }

  public func deviceTag() -> String { artifactKind == .file ? "Orin-sm87" : "cpu" }

  public func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    throw TestError("names only")
  }

  public func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    throw TestError("names only")
  }

  public func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    throw TestError("names only")
  }
}

extension EngineHost {
  /// Whether the job in hand ends, ready or failed, within `timeout`.
  public func settles(timeout: TimeInterval = 120) -> Bool {
    eventually(timeout: timeout) { snapshot().state == .ready || snapshot().state == .failed }
  }
}
