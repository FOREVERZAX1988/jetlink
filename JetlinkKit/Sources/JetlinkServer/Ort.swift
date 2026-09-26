import COrt
import Foundation

/// onnxruntime, through the C shim in COrt.
public enum OrtRuntime {
  public static var version: String { String(cString: jl_version()) }
}

public struct OrtError: Error, CustomStringConvertible {
  public let description: String

  init(_ description: String) {
    self.description = description
  }

  /// Throws the shim's message when there is one, and frees it.
  static func check(_ message: UnsafeMutablePointer<CChar>?) throws {
    guard let message else { return }
    defer { jl_free(message) }
    throw OrtError(String(cString: message))
  }
}

/// ONNX element types the engine stages, with their sizes.
public enum ElementType: Int32, Sendable {
  case float = 1
  case uint8 = 2
  case int8 = 3
  case uint16 = 4
  case int16 = 5
  case int32 = 6
  case int64 = 7
  case bool = 9
  case float16 = 10
  case double = 11

  public var size: Int {
    switch self {
    case .uint8, .int8, .bool: 1
    case .uint16, .int16, .float16: 2
    case .float, .int32: 4
    case .int64, .double: 8
    }
  }

  public var name: String {
    switch self {
    case .float: "float32"
    case .uint8: "uint8"
    case .int8: "int8"
    case .uint16: "uint16"
    case .int16: "int16"
    case .int32: "int32"
    case .int64: "int64"
    case .bool: "bool"
    case .float16: "float16"
    case .double: "float64"
    }
  }
}

/// One input or output as a session declares it.
public struct TensorSpec: Sendable, Equatable {
  public let name: String
  public let type: ElementType
  public let shape: [Int]

  public var count: Int { shape.reduce(1, *) }
  public var byteCount: Int { count * type.size }
}

/// The process's one onnxruntime environment.
final class OrtEnvironment: @unchecked Sendable {
  let pointer: OpaquePointer

  private init() throws {
    var env: OpaquePointer?
    // 3: errors only
    try OrtError.check(jl_env_create(3, &env))
    guard let env else { throw OrtError("onnxruntime returned no environment") }
    pointer = env
  }

  private static let lock = NSLock()
  nonisolated(unsafe) private static var cached: OrtEnvironment?

  static func shared() throws -> OrtEnvironment {
    lock.lock()
    defer { lock.unlock() }
    if let cached { return cached }
    let env = try OrtEnvironment()
    cached = env
    return env
  }
}

/// One onnxruntime session: a model file, the CoreML provider with its options,
/// and the inputs and outputs it declares.
final class OrtSession: @unchecked Sendable {
  let pointer: OpaquePointer
  let inputs: [TensorSpec]
  let outputs: [TensorSpec]

  /// `coreML` nil runs the session on the CPU alone.
  init(model: URL, coreML options: [String: String]?) throws {
    let env = try OrtEnvironment.shared()
    var session: OpaquePointer?
    let pairs = (options ?? [:]).sorted { $0.key < $1.key }
    let keys = pairs.map { strdup($0.key) }
    let values = pairs.map { strdup($0.value) }
    defer {
      keys.forEach { free($0) }
      values.forEach { free($0) }
    }
    var entries = zip(keys, values).map { jl_option(key: $0, value: $1) }
    try entries.withUnsafeMutableBufferPointer { buffer in
      try model.path.withCString { path in
        if options != nil {
          try OrtError.check(jl_session_create(env.pointer, path, "CoreML", buffer.baseAddress, buffer.count, &session))
        } else {
          try OrtError.check(jl_session_create(env.pointer, path, nil, nil, 0, &session))
        }
      }
    }
    guard let session else { throw OrtError("onnxruntime returned no session for \(model.lastPathComponent)") }
    pointer = session
    do {
      inputs = try OrtSession.describe(session, output: false)
      outputs = try OrtSession.describe(session, output: true)
    } catch {
      jl_session_release(session)
      throw error
    }
  }

  deinit {
    jl_session_release(pointer)
  }

  private static func describe(_ session: OpaquePointer, output: Bool) throws -> [TensorSpec] {
    let count = jl_session_io_count(session, output ? 1 : 0)
    var specs: [TensorSpec] = []
    for index in 0..<count {
      var name = [CChar](repeating: 0, count: 512)
      var type: Int32 = 0
      var dims = [Int64](repeating: 0, count: 16)
      var rank = 0
      try OrtError.check(jl_session_io_info(session, output ? 1 : 0, index, &name, name.count, &type, &dims, dims.count, &rank))
      let tensorName = String(decoding: name.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
      guard let element = ElementType(rawValue: type) else {
        throw OrtError("\(tensorName) has ONNX element type \(type), which jetlink does not stage")
      }
      let shape = dims.prefix(rank).map { Int($0) }
      if shape.contains(where: { $0 <= 0 }) {
        throw OrtError("\(tensorName) has a dynamic shape \(shape); jetlink builds fixed-shape engines")
      }
      specs.append(TensorSpec(name: tensorName, type: element, shape: shape))
    }
    return specs
  }
}

/// Caller-owned buffers bound to one session's inputs and outputs.
final class OrtBinding {
  private let pointer: OpaquePointer

  init(session: OrtSession, inputs: [(TensorSpec, UnsafeMutableRawPointer)], outputs: [(TensorSpec, UnsafeMutableRawPointer)]) throws {
    var names: [UnsafeMutablePointer<CChar>?] = []
    var dims: [UnsafeMutablePointer<Int64>] = []
    defer {
      names.forEach { free($0) }
      dims.forEach { $0.deallocate() }
    }
    func tensor(_ spec: TensorSpec, _ data: UnsafeMutableRawPointer) -> jl_tensor {
      let name = strdup(spec.name)
      names.append(name)
      let shape = UnsafeMutablePointer<Int64>.allocate(capacity: max(spec.shape.count, 1))
      for (index, dim) in spec.shape.enumerated() { shape[index] = Int64(dim) }
      dims.append(shape)
      return jl_tensor(name: name, elem_type: spec.type.rawValue, dims: shape, rank: spec.shape.count, data: data, nbytes: spec.byteCount)
    }
    let ins = inputs.map { tensor($0.0, $0.1) }
    let outs = outputs.map { tensor($0.0, $0.1) }
    var binding: OpaquePointer?
    try OrtError.check(jl_binding_create(session.pointer, ins, ins.count, outs, outs.count, &binding))
    guard let binding else { throw OrtError("onnxruntime returned no binding") }
    pointer = binding
  }

  deinit {
    jl_binding_release(pointer)
  }

  func run() throws {
    try OrtError.check(jl_binding_run(pointer))
  }
}
