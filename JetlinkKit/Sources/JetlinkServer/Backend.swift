import Foundation
import JetlinkRegistry

/// Progress of a build or a load: stage, fraction, message.
public typealias ProgressFn = @Sendable (String, Double, String) -> Void

/// Until TensorRT's backend stops naming it.
public typealias ArtifactKind = JetlinkRegistry.ArtifactKind

/// The artifact on disk is not one this backend can load: another runtime's
/// compiled cache, an older preparation, a plan from another TensorRT build.
/// Any backend's `load` throws it for what is wrong with the file itself, and
/// the host deletes the artifact and rebuilds it once from the ONNX when that
/// is on disk at the size asked for; without it the client uploads again.
/// Anything else a load throws (memory, a device gone) leaves the artifact.
public struct ArtifactInvalid: Error, CustomStringConvertible {
  public let description: String

  public init(_ description: String) {
    self.description = description
  }
}

/// An error an engine cannot come back from, such as CUDA's sticky errors
/// (an illegal address, a failed launch): the context is broken, and every
/// later frame and every rejoin would fail with the engine still "loaded".
/// The session answers the frame INFER_FAILED, then calls
/// `ServerHooks.fatal`. A type whose codes are only sometimes fatal
/// conforms and says which through `isFatal`.
public protocol FatalEngineError: Error {
  var isFatal: Bool { get }
}

extension FatalEngineError {
  public var isFatal: Bool { true }
}

/// A loaded model, ready to run a frame at a time: a backend's engine
/// (JetlinkORT's `OrtEngine`), or the tests' one that needs no runtime.
public protocol Engine: AnyObject {
  var inputs: [String: TensorSpec] { get }
  var outputs: [String: TensorSpec] { get }
  var lastGpuUs: UInt32 { get }
  /// What ran beside the model, for a benchmark report's build line: "CPU
  /// keep-warm on". Empty when there is nothing to say.
  var notes: String { get }
  /// Where the host writes an input. For every input but a looped pair's
  /// state_ one the pointer stays the same from load to close, as does
  /// `output`'s for every output the engine does not loop: staging looks
  /// them up once, at load.
  func hostInput(_ name: String) -> UnsafeMutableRawPointer?
  func output(_ name: String) -> UnsafeRawPointer?
  /// Feeds each next_state_ output back as its state_ input from the next
  /// run on, inside the engine. Throws for a pair it cannot loop.
  func loopState(_ pairs: [(input: String, output: String)]) throws
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
  /// The runtime's release, "1.29.0".
  var runtimeVersion: String { get }
  /// The device part of the tag: "ane-Apple_M1_Pro", "htp-SM8650".
  func deviceTag() -> String
  /// What an artifact is valid for: runtime version and device, sanitized.
  func tag() -> String
  /// backend, runtime_version, device: for the hello.
  func describe() -> [String: String]
  /// What this backend adds to the hello besides `describe()`: TensorRT's
  /// `trt_version`, which the comma logs.
  var helloFields: [String: Any] { get }
  func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec
  func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws
  func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine
}

extension Engine {
  public var notes: String { "" }
}

extension EngineBackend {
  public func tag() -> String {
    "\(name)\(sanitize(runtimeVersion)).\(deviceTag())"
  }

  public func describe() -> [String: String] {
    ["backend": name, "runtime_version": runtimeVersion, "device": deviceTag()]
  }

  public var helloFields: [String: Any] { [:] }
}

/// A version or device name as a filename component.
public func sanitize(_ s: String) -> String {
  String(s.map { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "_" || $0 == "-") ? $0 : "_" })
}
