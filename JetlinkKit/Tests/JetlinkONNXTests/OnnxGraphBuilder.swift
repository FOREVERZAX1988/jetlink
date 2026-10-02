import Foundation
import JetlinkTestSupport

@testable import JetlinkONNX

/// A small ONNX model written from Swift for the LiteRT lowering tests:
/// nodes, initializers as raw_data, and value infos with static shapes,
/// encoded as onnx.helper would.
struct OnnxGraphBuilder {
  enum Attr {
    case int(Int64)
    case ints([Int64])
    case float(Float)
    case string(String)
  }

  struct Value {
    let name: String
    let type: Int32
    let dims: [Int64]
  }

  var opset: Int64 = 17
  var inputs: [Value] = []
  var outputs: [Value] = []
  var valueInfo: [Value] = []
  private var nodes: [Encoded] = []
  private var initializers: [Encoded] = []

  init(opset: Int64 = 17) {
    self.opset = opset
  }

  mutating func input(_ name: String, _ type: Int32, _ dims: [Int64]) {
    inputs.append(Value(name: name, type: type, dims: dims))
  }

  mutating func output(_ name: String, _ type: Int32, _ dims: [Int64]) {
    outputs.append(Value(name: name, type: type, dims: dims))
  }

  mutating func node(_ op: String, _ inputs: [String], _ outputs: [String], _ attrs: [(String, Attr)] = []) {
    var e = Encoded()
    for i in inputs { e.stringField(1, i) }
    for o in outputs { e.stringField(2, o) }
    e.stringField(3, "\(op)_\(nodes.count)")
    e.stringField(4, op)
    for (name, value) in attrs {
      var a = Encoded()
      a.stringField(1, name)
      switch value {
      case .float(let f):
        a.tag(2, .fixed32)
        withUnsafeBytes(of: f.bitPattern.littleEndian) { a.bytes($0) }
        a.varintField(20, 1)
      case .int(let i):
        a.intField(3, i)
        a.varintField(20, 2)
      case .string(let s):
        a.stringField(4, s)
        a.varintField(20, 3)
      case .ints(let ints):
        for i in ints { a.intField(8, i) }
        a.varintField(20, 7)
      }
      e.message(5, a)
    }
    nodes.append(e)
  }

  mutating func initializer(_ name: String, _ type: Int32, _ dims: [Int64], _ raw: [UInt8]) {
    var e = Encoded()
    for d in dims { e.intField(1, d) }
    e.intField(2, Int64(type))
    e.stringField(8, name)
    e.bytesField(9, raw)
    initializers.append(e)
  }

  mutating func fp16(_ name: String, _ dims: [Int64], _ values: [Float]) {
    var raw: [UInt8] = []
    for v in values { withUnsafeBytes(of: Float16(v).bitPattern.littleEndian) { raw.append(contentsOf: $0) } }
    initializer(name, DataType.float16, dims, raw)
  }

  mutating func fp32(_ name: String, _ dims: [Int64], _ values: [Float]) {
    var raw: [UInt8] = []
    for v in values { withUnsafeBytes(of: v.bitPattern.littleEndian) { raw.append(contentsOf: $0) } }
    initializer(name, DataType.float, dims, raw)
  }

  mutating func int64(_ name: String, _ values: [Int64], dims: [Int64]? = nil) {
    var raw: [UInt8] = []
    for v in values { withUnsafeBytes(of: v.littleEndian) { raw.append(contentsOf: $0) } }
    initializer(name, DataType.int64, dims ?? [Int64(values.count)], raw)
  }

  mutating func bool(_ name: String, _ dims: [Int64], _ values: [Bool]) {
    initializer(name, DataType.bool, dims, values.map { $0 ? 1 : 0 })
  }

  private static func valueInfo(_ v: Value) -> Encoded {
    var dims = Encoded()
    for d in v.dims {
      var dim = Encoded()
      dim.intField(1, d)
      dims.message(1, dim)
    }
    var tensor = Encoded()
    tensor.intField(1, Int64(v.type))
    tensor.message(2, dims)
    var type = Encoded()
    type.message(1, tensor)
    var e = Encoded()
    e.stringField(1, v.name)
    e.message(2, type)
    return e
  }

  var bytes: [UInt8] {
    var g = Encoded()
    for n in nodes { g.message(1, n) }
    g.stringField(2, "test")
    for t in initializers { g.message(5, t) }
    for v in inputs { g.message(11, Self.valueInfo(v)) }
    for v in outputs { g.message(12, Self.valueInfo(v)) }
    for v in valueInfo { g.message(13, Self.valueInfo(v)) }
    var opsetImport = Encoded()
    opsetImport.stringField(1, "")
    opsetImport.intField(2, opset)
    var m = Encoded()
    m.intField(1, 8)
    m.message(7, g)
    m.message(8, opsetImport)
    return m.tail
  }

  /// The model converted by LiteRTPreparation, read back. Without
  /// `rewrites` the graph goes to the lowering as it is.
  func convert(rewrites: Bool = true) throws -> (file: TFLiteFile, report: LiteRTPreparation.Report) {
    let dir = try TemporaryDirectory()
    let out = dir.url.appendingPathComponent("out")
    let report: LiteRTPreparation.Report
    if rewrites {
      let source = dir.url.appendingPathComponent("model.onnx")
      try Data(bytes).write(to: source)
      report = try LiteRTPreparation.prepare(source: source, into: out)
    } else {
      report = try bytes.withUnsafeBytes { buf in
        let src = Source(bytes: buf)
        let model = try Decode.model(src)
        return try LiteRTPreparation.write(model.graph!, opsets: model.opsets, src, rewrites: [:], into: out, progress: nil)
      }
    }
    return (try TFLiteFile(report.url), report)
  }
}

/// Seeded values for tests, the same on every platform.
struct SeededValues {
  private var state: UInt64

  init(seed: UInt64) {
    state = seed &* 0x9E37_79B9_7F4A_7C15 | 1
  }

  /// Uniform in [-1, 1), on a grid fp16 holds exactly.
  mutating func next() -> Float {
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float(Int(state % 2048) - 1024) / 1024
  }

  mutating func callAsFunction(_ count: Int) -> [Float] {
    (0..<count).map { _ in next() }
  }
}

/// The worst absolute difference.
func maxError(_ a: [Float], _ b: [Float]) -> Float {
  precondition(a.count == b.count, "\(a.count) values against \(b.count)")
  return zip(a, b).map { abs($0 - $1) }.max() ?? 0
}
