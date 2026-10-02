import Foundation
import JetlinkKit
import JetlinkServer

/// A LiteRT artifact as LiteRtBackend builds and loads it: a directory
/// holding the converted model, the GPU's compile cache and a `litert.json`
/// manifest naming them, with the sidecar beside it.
enum LiteRtArtifact {
  static let manifestName = "litert.json"

  struct Manifest {
    /// The .tflite, relative to the artifact.
    let model: String
    /// The GPU's compile cache: a directory relative to the artifact, and the
    /// key its files are named by.
    let cache: String
    let cacheKey: String

    var dictionary: [String: Any] { ["model": model, "cache": cache, "cache_key": cacheKey] }
  }

  static func write(_ manifest: Manifest, in directory: URL) throws {
    try JSONSerialization.data(withJSONObject: manifest.dictionary, options: [.prettyPrinted, .sortedKeys])
      .write(to: directory.appending(path: manifestName))
  }

  /// A built artifact's manifest and sidecar. Throws `ArtifactInvalid`,
  /// which rebuilds it, when the manifest is unreadable, the conversion was
  /// another version, or the model is missing.
  static func open(_ artifact: URL, version expected: Int) throws -> (manifest: Manifest, meta: [String: Any]) {
    let name = artifact.lastPathComponent
    guard let data = try? Data(contentsOf: artifact.appending(path: manifestName)),
      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let model = object["model"] as? String, let cache = object["cache"] as? String, let key = object["cache_key"] as? String
    else {
      throw ArtifactInvalid("\(name): no readable \(manifestName) inside")
    }
    let meta = Artifact.sidecar(artifact)
    let version = (meta["prepare"] as? NSNumber)?.intValue ?? 0
    if version != expected {
      throw ArtifactInvalid("\(name): converted as version \(version), LiteRT builds are now at \(expected); rebuilding")
    }
    guard FileManager.default.fileExists(atPath: artifact.appending(path: model).path) else {
      throw ArtifactInvalid("\(name): the converted model is missing")
    }
    return (Manifest(model: model, cache: cache, cacheKey: key), meta)
  }

  /// The sidecar keys every build writes; the runtime's release goes under
  /// `litert`, as onnxruntime's goes under `onnxruntime`.
  static func meta(_ backend: LiteRtBackend, manifest: Manifest, engine: LiteRtEngine, model: URL, started: Date) -> [String: Any] {
    var meta = Artifact.meta(backend, runtimeKey: "litert", model: model, started: started)
    meta["model"] = manifest.model
    meta["accelerator"] = engine.label
    meta["prepare"] = backend.converter.version
    meta["preparer"] = "swift"
    return meta
  }
}
