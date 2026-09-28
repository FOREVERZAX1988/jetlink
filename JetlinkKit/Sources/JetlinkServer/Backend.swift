import Foundation

/// Progress of a build or a load: stage, fraction, message.
public typealias ProgressFn = @Sendable (String, Double, String) -> Void

/// The artifact on disk is not one this backend can load. The host answers by
/// deleting it and rebuilding from the ONNX if it has one, as in Python.
public struct ArtifactInvalid: Error, CustomStringConvertible {
  public let description: String

  public init(_ description: String) {
    self.description = description
  }
}

/// A loaded model, ready to run a frame at a time. `OrtEngine` is the one the
/// server uses; the tests have one that needs no runtime.
public protocol Engine: AnyObject {
  var inputs: [String: TensorSpec] { get }
  var outputs: [String: TensorSpec] { get }
  var lastGpuUs: UInt32 { get }
  func hostInput(_ name: String) -> UnsafeMutableRawPointer?
  func output(_ name: String) -> UnsafeRawPointer?
  @discardableResult func loopState(_ pairs: [(input: String, output: String)]) throws -> Bool
  func resetState()
  func run() throws
  func warm() throws -> String
  func close()
}

/// Turns an ONNX file into an artifact it can load quickly, and loads one.
/// The Swift form of `server/backends/base.Backend`.
public protocol EngineBackend: AnyObject, Sendable {
  /// "ort", as the hello reports it.
  var name: String { get }
  /// The artifact's extension.
  var suffix: String { get }
  /// What an artifact is valid for: runtime version and device, sanitized.
  func tag() -> String
  /// The device part of the tag: "ane-Apple_M1_Pro", "htp-SM8650".
  func deviceTag() -> String
  /// backend, runtime_version, device: for the hello.
  func describe() -> [String: String]
  func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec
  func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws
  func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine
}

/// A version or device name as a filename component.
public func sanitize(_ s: String) -> String {
  String(s.map { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "_" || $0 == "-") ? $0 : "_" })
}
