import CLiteRt
import Foundation
import JetlinkServer

/// LiteRT, through the C shim in CLiteRt.
public enum LiteRtRuntime {
  /// The release jetlink builds against and ships: CLiteRt's headers, the
  /// Mac's ai-edge-litert wheel and Android's AAR. LiteRT's C API reports no
  /// version of its own.
  public static let version = "2.2.0"

  /// Where the tests and the command line find LiteRT's libraries when the
  /// host names none: the ai-edge-litert wheel's package directory, which
  /// holds libLiteRt and its GPU accelerator side by side.
  public static let directoryVariable = "JETLINK_LITERT_DIR"

  /// Opens LiteRT and makes the process's environment, or throws why it
  /// cannot, in words an app can show. `directory` holds LiteRT's
  /// libraries, as the Android app's nativeLibraryDir does. Nil falls back to
  /// $JETLINK_LITERT_DIR, then to the loader's path. The first open that
  /// succeeds decides for the process; later calls return at once.
  public static func load(directory: URL? = nil) throws {
    let path = directory?.path ?? ProcessInfo.processInfo.environment[directoryVariable]
    try LiteRtError.check(jl_litert_open(path))
  }

  /// Whether LiteRT's GPU accelerator loaded, and the names of every
  /// accelerator that did ("GPU Metal, CpuAccelerator" on a Mac). LiteRT
  /// must be open.
  public static func accelerators() throws -> (gpu: Bool, names: String) {
    var hardware: Int32 = 0
    var names = [CChar](repeating: 0, count: 256)
    try LiteRtError.check(jl_litert_accelerators(&hardware, &names, names.count))
    let text = String(decoding: names.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    return (hardware & Int32(JL_LITERT_GPU) != 0, text)
  }
}

public struct LiteRtError: Error, CustomStringConvertible {
  public let description: String

  init(_ description: String) {
    self.description = description
  }

  /// Throws the shim's message when there is one, and frees it.
  static func check(_ message: UnsafeMutablePointer<CChar>?) throws {
    guard let message else { return }
    defer { jl_litert_free(message) }
    throw LiteRtError(String(cString: message))
  }
}

/// How a model is compiled: which accelerator runs it and with what.
enum LiteRtCompileOptions: Sendable {
  /// The GPU alone, in fp16, so an op it cannot run fails the compile
  /// rather than quietly running on the CPU. `cache` is where it keeps the
  /// programs it compiled, and the key it files them under.
  case gpu(cache: (directory: URL, key: String)?)
  /// XNNPACK on the CPU, with a pool of `threads`.
  case cpu(threads: Int)
}

/// One compiled model: a .tflite, compiled for an accelerator, and the
/// inputs and outputs of its first signature, in signature order.
final class LiteRtModel: @unchecked Sendable {
  let pointer: OpaquePointer
  let inputs: [TensorSpec]
  let outputs: [TensorSpec]

  /// LiteRT must be open (`LiteRtRuntime.load`).
  init(model: URL, options: LiteRtCompileOptions) throws {
    var shim = jl_litert_options()
    var cache: (directory: URL, key: String)?
    switch options {
    case .gpu(let programs):
      shim.gpu = 1
      cache = programs
    case .cpu(let threads):
      shim.cpu_threads = Int32(threads)
    }
    let cacheDirectory = cache.flatMap { strdup($0.directory.path) }
    let cacheKey = cache.flatMap { strdup($0.key) }
    defer {
      free(cacheDirectory)
      free(cacheKey)
    }
    shim.cache_dir = UnsafePointer(cacheDirectory)
    shim.cache_key = UnsafePointer(cacheKey)
    var compiled: OpaquePointer?
    try model.path.withCString { path in
      try LiteRtError.check(jl_litert_model_create(path, &shim, &compiled))
    }
    guard let compiled else { throw LiteRtError("LiteRT returned no compiled model for \(model.lastPathComponent)") }
    pointer = compiled
    do {
      inputs = try LiteRtModel.describe(compiled, output: false)
      outputs = try LiteRtModel.describe(compiled, output: true)
    } catch {
      jl_litert_model_release(compiled)
      throw error
    }
  }

  deinit {
    jl_litert_model_release(pointer)
  }

  /// Whether the accelerator asked for runs every op.
  var fullyAccelerated: Bool {
    get throws {
      var fully: Int32 = 0
      try LiteRtError.check(jl_litert_model_fully_accelerated(pointer, &fully))
      return fully != 0
    }
  }

  /// Whether the accelerator reads input `index` straight from host memory,
  /// as the CPU does, where a GPU wants its own.
  func readsHostMemory(input index: Int) throws -> Bool {
    var host: Int32 = 0
    try LiteRtError.check(jl_litert_model_input_host(pointer, index, &host))
    return host != 0
  }

  private static func describe(_ model: OpaquePointer, output: Bool) throws -> [TensorSpec] {
    try TensorSpec.described(count: jl_litert_model_io_count(model, output ? 1 : 0)) { index, name, type, dims, rank in
      try LiteRtError.check(jl_litert_model_io_info(model, output ? 1 : 0, index, &name, name.count, &type, &dims, dims.count, &rank))
    }
  }
}

/// One of LiteRT's tensor buffers: over memory the engine owns, or the
/// accelerator's own (GPU memory on a GPU).
final class LiteRtBuffer {
  let pointer: OpaquePointer

  /// A buffer for input or output `index` over `bytes` at `data`, 64-byte
  /// aligned, which must outlive it.
  init(_ model: LiteRtModel, output: Bool, index: Int, wrapping data: UnsafeMutableRawPointer, bytes: Int) throws {
    var buffer: OpaquePointer?
    try LiteRtError.check(jl_litert_buffer_wrap(model.pointer, output ? 1 : 0, index, data, bytes, &buffer))
    guard let buffer else { throw LiteRtError("LiteRT returned no buffer") }
    pointer = buffer
  }

  /// A zeroed buffer of the kind the accelerator works in for input `index`.
  init(_ model: LiteRtModel, input index: Int) throws {
    var buffer: OpaquePointer?
    try LiteRtError.check(jl_litert_buffer_create(model.pointer, 0, index, &buffer))
    guard let buffer else { throw LiteRtError("LiteRT returned no buffer") }
    pointer = buffer
  }

  deinit {
    jl_litert_buffer_release(pointer)
  }

  func zero() throws {
    try LiteRtError.check(jl_litert_buffer_write(pointer, nil, 0))
  }

  func read(into data: UnsafeMutableRawPointer, bytes: Int) throws {
    try LiteRtError.check(jl_litert_buffer_read(pointer, data, bytes))
  }
}

extension LiteRtModel {
  /// One run of the first signature, over buffers' pointers in signature
  /// order, which the caller keeps alive.
  func run(inputs: [OpaquePointer?], outputs: [OpaquePointer?]) throws {
    try LiteRtError.check(jl_litert_run(pointer, inputs, inputs.count, outputs, outputs.count))
  }
}
