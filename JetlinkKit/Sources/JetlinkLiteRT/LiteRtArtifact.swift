import Foundation
import JetlinkKit
import JetlinkServer

/// A LiteRT artifact as LiteRtBackend builds and loads it: a directory
/// holding the converted model and the GPU's compile cache, with the sidecar
/// beside it.
enum LiteRtArtifact {
  /// The converted model, as LiteRTPreparation names it.
  static let model = "model.tflite"
  /// The GPU's compile cache, a directory.
  static let cache = "gpu-cache"

  /// What the GPU files its cache under: the artifact's name, which the
  /// build's staging directory does not have.
  static func cacheKey(_ artifact: URL) -> String {
    artifact.deletingPathExtension().lastPathComponent
  }

  /// A built artifact's sidecar. Throws `ArtifactInvalid`, which rebuilds
  /// it, when the conversion was another version or the model is missing.
  static func open(_ artifact: URL) throws -> [String: Any] {
    try Artifact.open(artifact, version: LiteRtBackend.conversionVersion, unversioned: 0, runtime: "LiteRT", files: [model])
  }

  /// The sidecar keys every build writes; the runtime's release goes under
  /// `litert`, as onnxruntime's goes under `onnxruntime`.
  static func meta(_ backend: LiteRtBackend, model: URL, started: Date) -> [String: Any] {
    var meta = Artifact.meta(backend, runtimeKey: "litert", model: model, started: started)
    meta["accelerator"] = backend.profile.label
    meta["prepare"] = LiteRtBackend.conversionVersion
    meta["preparer"] = "swift"
    return meta
  }
}
