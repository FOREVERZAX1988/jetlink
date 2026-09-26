import Foundation

/// The preparer the server uses. TEMP: a stand-in until JetlinkONNX lands;
/// it can serve an artifact that is already built, and nothing more.
public struct ONNXPreparer: ModelPreparer {
  public init() {}

  public func readSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    throw HostError.failed("reading ONNX metadata is not available in this build")
  }

  public func prepare(model: URL, into directory: URL, split: Bool, cacheKey: @escaping (String) -> String) throws -> PreparedModel {
    throw HostError.failed("preparing ONNX is not available in this build")
  }
}
