import Foundation

@testable import JetlinkONNX

/// A graph made in memory for a rewrite to work on, every tensor's value info
/// recorded as an export records it.
struct GraphBuilder {
  var g = Graph()
  var opsets = [OpsetImport(raw: 0..<0, domain: "", version: 20)]

  mutating func input(_ name: String, _ type: Int32, _ dims: [Int64]) {
    g.inputs.append(.tensor(name, type, dims))
  }

  mutating func output(_ name: String, _ type: Int32, _ dims: [Int64]) {
    g.outputs.append(.tensor(name, type, dims))
  }

  /// A node, and the value info of its one output.
  mutating func node(
    _ op: String, _ inputs: [String], _ output: String, _ type: Int32, _ dims: [Int64], _ attributes: [Attribute] = []
  ) {
    nodes(op, inputs, [output], attributes)
    if !g.outputs.contains(where: { $0.key == output }) {
      g.valueInfo.append(.tensor(output, type, dims))
    }
  }

  /// A node with no value infos of its own.
  mutating func nodes(_ op: String, _ inputs: [String], _ outputs: [String], _ attributes: [Attribute] = []) {
    g.nodes.append(Node(inputs: inputs, outputs: outputs, name: "n\(g.nodes.count)", opType: op, attributes: attributes))
  }

  mutating func ints(_ name: String, _ values: [Int64], dims: [Int64]? = nil) {
    var t = Patches.int64Tensor(values, name)
    t.dims = dims ?? [Int64(values.count)]
    g.initializers.append(t)
  }

  mutating func floats(_ name: String, _ values: [Float], dims: [Int64], type: Int32 = DataType.float) {
    var bytes: [UInt8] = []
    for v in values {
      if type == DataType.float16 {
        withUnsafeBytes(of: Float16(v).bitPattern.littleEndian) { bytes += $0 }
      } else {
        withUnsafeBytes(of: v.bitPattern.littleEndian) { bytes += $0 }
      }
    }
    g.initializers.append(Tensor(name: name, dims: dims, dataType: type, raw: .owned(bytes)))
  }
}

/// A source with no bytes, for graphs whose constants are all owned.
var emptySource: Source { Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0)) }

/// A tensor's dims and values, in row-major order, as doubles.
struct Values: Equatable {
  var dims: [Int]
  var data: [Double]

  var count: Int { dims.reduce(1, *) }

  func strides() -> [Int] {
    var s = Array(repeating: 1, count: dims.count)
    for i in stride(from: dims.count - 2, through: 0, by: -1) {
      s[i] = s[i + 1] * dims[i + 1]
    }
    return s
  }

  /// The multi-index of flat position `p`.
  func index(_ p: Int) -> [Int] {
    var rest = p
    return strides().map { s in
      defer { rest %= s }
      return rest / s
    }
  }
}

/// Runs a graph on doubles, op by op as onnx defines them, for the ops the
/// LiteRT rewrites make and remove. Exact for the ones that move data, so a
/// layout rewrite is held to the same values bit for bit; the arithmetic is
/// double, so a rewrite that is exact in real arithmetic lands within
/// rounding of the original.
struct GraphEvaluator {
  let g: Graph
  let opset: Int64

  func run(_ inputs: [String: Values]) throws -> [String: Values] {
    var env = inputs
    for t in g.initializers {
      env[t.key] = try Self.constant(t)
    }
    for n in g.nodes {
      let args = n.inputs.map { $0.isEmpty ? nil : env[$0] }
      let outs = try evaluate(n, args)
      for (name, v) in zip(n.outputs, outs) {
        env[name] = v
      }
    }
    return env
  }

  static func constant(_ t: Tensor) throws -> Values {
    let dims = t.dims.map(Int.init)
    if DataType.isInteger(t.elementType) {
      return Values(dims: dims, data: try Elements.integers(t, emptySource).map(Double.init))
    }
    let bytes = try Elements.littleEndian(t, emptySource)
    var data: [Double] = []
    switch t.elementType {
    case DataType.float16:
      for i in stride(from: 0, to: bytes.count, by: 2) {
        data.append(Double(Float16(bitPattern: UInt16(bytes[i]) | UInt16(bytes[i + 1]) << 8)))
      }
    case DataType.float:
      for i in stride(from: 0, to: bytes.count, by: 4) {
        var bits: UInt32 = 0
        for b in 0..<4 { bits |= UInt32(bytes[i + b]) << UInt32(8 * b) }
        data.append(Double(Float(bitPattern: bits)))
      }
    case DataType.bool:
      data = bytes.map { Double($0) }
    default:
      throw OnnxError("the evaluator reads no element type \(t.elementType)")
    }
    return Values(dims: dims, data: data)
  }

  /// Integer values; the INT64_MAX and INT64_MIN that mean "to the end" are
  /// no longer exact as doubles, so they come back clamped.
  private func ints(_ v: Values?) -> [Int] {
    (v?.data ?? []).map { $0 >= 0x1p63 ? .max : ($0 <= -0x1p63 ? .min : Int($0)) }
  }

  private func axes(_ n: Node, _ args: [Values?], rank: Int) -> [Int]? {
    let raw: [Int]
    if args.count > 1, let a = args[1] {
      raw = ints(a)
    } else if let a = n.attribute("axes") {
      raw = a.ints.map(Int.init)
    } else {
      return nil
    }
    return raw.map { $0 < 0 ? $0 + rank : $0 }
  }

  private func evaluate(_ n: Node, _ args: [Values?]) throws -> [Values] {
    let x = args.first ?? nil
    switch n.op {
    case "Identity", "Cast":
      return [x!]
    case "Reshape":
      var shape = ints(args[1])
      let known = shape.filter { $0 > 0 }.reduce(1, *)
      for i in shape.indices where shape[i] == 0 { shape[i] = x!.dims[i] }
      for i in shape.indices where shape[i] == -1 { shape[i] = x!.count / known }
      return [Values(dims: shape, data: x!.data)]
    case "Squeeze":
      let rank = x!.dims.count
      let drop = axes(n, args, rank: rank) ?? x!.dims.indices.filter { x!.dims[$0] == 1 }
      return [Values(dims: x!.dims.enumerated().filter { !drop.contains($0.offset) }.map(\.element), data: x!.data)]
    case "Unsqueeze":
      let rank = x!.dims.count + (axes(n, args, rank: 0)?.count ?? 0)
      let add = axes(n, args, rank: rank)!
      var rest = x!.dims.makeIterator()
      return [Values(dims: (0..<rank).map { add.contains($0) ? 1 : rest.next()! }, data: x!.data)]
    case "Transpose":
      let perm = n.attribute("perm")?.ints.map(Int.init) ?? Array(x!.dims.indices.reversed())
      return [gather(x!, dims: perm.map { x!.dims[$0] }) { out in perm.indices.map { i in out[perm.firstIndex(of: i)!] } }]
    case "Slice":
      let starts = ints(args[1])
      let ends = ints(args[2])
      let axesList = args.count > 3 && args[3] != nil ? ints(args[3]).map { $0 < 0 ? $0 + x!.dims.count : $0 } : Array(starts.indices)
      let steps = args.count > 4 && args[4] != nil ? ints(args[4]) : Array(repeating: 1, count: starts.count)
      var picks = x!.dims.map { Array(0..<$0) }
      for (i, a) in axesList.enumerated() {
        picks[a] = Patches.sliceIndices(size: Int64(x!.dims[a]), start: Int64(starts[i]), end: Int64(ends[i]), step: Int64(steps[i])).map {
          Int($0)
        }
      }
      return [gather(x!, dims: picks.map(\.count)) { out in out.enumerated().map { picks[$0.offset][$0.element] } }]
    case "Concat":
      let parts = args.compactMap { $0 }
      let axis = Int(n.attribute("axis")!.i) < 0 ? Int(n.attribute("axis")!.i) + parts[0].dims.count : Int(n.attribute("axis")!.i)
      var dims = parts[0].dims
      dims[axis] = parts.map { $0.dims[axis] }.reduce(0, +)
      var offsets: [Int] = []
      var o = 0
      for p in parts {
        offsets.append(o)
        o += p.dims[axis]
      }
      let out = Values(dims: dims, data: Array(repeating: 0, count: dims.reduce(1, *)))
      var data = out.data
      for p in 0..<out.count {
        var index = out.index(p)
        let part = offsets.lastIndex { $0 <= index[axis] }!
        index[axis] -= offsets[part]
        data[p] = parts[part].data[zip(index, parts[part].strides()).map(*).reduce(0, +)]
      }
      return [Values(dims: dims, data: data)]
    case "Gather":
      let index = args[1]!
      let rank = x!.dims.count
      let raw = Int(n.attribute("axis")?.i ?? 0)
      let axis = raw < 0 ? raw + rank : raw
      let dims = Array(x!.dims[..<axis]) + index.dims + x!.dims[(axis + 1)...]
      return [
        gather(x!, dims: dims) { out in
          let at = Array(out[axis..<(axis + index.dims.count)])
          var i = Int(index.data[zip(at, index.strides()).map(*).reduce(0, +)])
          if i < 0 { i += x!.dims[axis] }
          return Array(out[..<axis]) + [i] + out[(axis + index.dims.count)...]
        }
      ]
    case "GatherND":
      let index = args[1]!
      let depth = index.dims.last!
      let lead = Array(index.dims.dropLast())
      let dims = lead + x!.dims[depth...]
      return [
        gather(x!, dims: dims) { out in
          let row = zip(out[..<lead.count], index.strides().dropLast()).map(*).reduce(0, +)
          let tuple = (0..<depth).map { j -> Int in
            let i = Int(index.data[row + j])
            return i < 0 ? i + x!.dims[j] : i
          }
          return tuple + out[lead.count...]
        }
      ]
    case "Split":
      let rank = x!.dims.count
      let raw = Int(n.attribute("axis")?.i ?? 0)
      let axis = raw < 0 ? raw + rank : raw
      let size = x!.dims[axis] / n.outputs.count
      return (0..<n.outputs.count).map { k in
        var dims = x!.dims
        dims[axis] = size
        return gather(x!, dims: dims) { out in
          var i = out
          i[axis] += k * size
          return i
        }
      }
    case "ReduceMean", "ReduceMax":
      let rank = x!.dims.count
      let over = axes(n, args, rank: rank) ?? Array(0..<rank)
      var dims = x!.dims
      for a in over { dims[a] = 1 }
      var groups = Array(repeating: [Double](), count: dims.reduce(1, *))
      let out = Values(dims: dims, data: [])
      let outStrides = out.strides()
      for p in 0..<x!.count {
        var index = x!.index(p)
        for a in over { index[a] = 0 }
        groups[zip(index, outStrides).map(*).reduce(0, +)].append(x!.data[p])
      }
      let data = groups.map { n.op == "ReduceMax" ? $0.max()! : $0.reduce(0, +) / Double($0.count) }
      let keep = (n.attribute("keepdims")?.i ?? 1) == 1
      return [Values(dims: keep ? dims : x!.dims.enumerated().filter { !over.contains($0.offset) }.map(\.element), data: data)]
    case "Abs": return [Values(dims: x!.dims, data: x!.data.map { abs($0) })]
    case "Sqrt": return [Values(dims: x!.dims, data: x!.data.map { $0.squareRoot() })]
    case "Reciprocal": return [Values(dims: x!.dims, data: x!.data.map { 1 / $0 })]
    case "Not": return [Values(dims: x!.dims, data: x!.data.map { $0 == 0 ? 1 : 0 })]
    case "Add": return [broadcast(args[0]!, args[1]!, +)]
    case "Sub": return [broadcast(args[0]!, args[1]!, -)]
    case "Mul": return [broadcast(args[0]!, args[1]!, *)]
    case "Div": return [broadcast(args[0]!, args[1]!, /)]
    case "Max": return [broadcast(args[0]!, args[1]!, max)]
    case "LayerNormalization":
      let rank = x!.dims.count
      let raw = Int(n.attribute("axis")?.i ?? -1)
      let axis = raw < 0 ? raw + rank : raw
      let epsilon = Double(n.attribute("epsilon")?.f ?? 1e-5)
      let inner = x!.dims[axis...].reduce(1, *)
      var data = x!.data
      for start in stride(from: 0, to: x!.count, by: inner) {
        let row = x!.data[start..<(start + inner)]
        let mean = row.reduce(0, +) / Double(inner)
        let variance = row.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(inner)
        for i in 0..<inner {
          data[start + i] = (row[start + i] - mean) / (variance + epsilon).squareRoot()
        }
      }
      var y = Values(dims: x!.dims, data: data)
      y = broadcast(y, args[1]!, *)
      if args.count > 2, let b = args[2] { y = broadcast(y, b, +) }
      return [y]
    default:
      throw OnnxError("the evaluator runs no \(n.op)")
    }
  }

  /// Each output element read from the input element `from` names.
  private func gather(_ x: Values, dims: [Int], from: ([Int]) -> [Int]) -> Values {
    let out = Values(dims: dims, data: [])
    let strides = x.strides()
    let data = (0..<out.count).map { p in x.data[zip(from(out.index(p)), strides).map(*).reduce(0, +)] }
    return Values(dims: dims, data: data)
  }

  private func broadcast(_ a: Values, _ b: Values, _ f: (Double, Double) -> Double) -> Values {
    let rank = max(a.dims.count, b.dims.count)
    let ad = Array(repeating: 1, count: rank - a.dims.count) + a.dims
    let bd = Array(repeating: 1, count: rank - b.dims.count) + b.dims
    let dims = zip(ad, bd).map { max($0, $1) }
    let out = Values(dims: dims, data: [])
    let sa = Values(dims: ad, data: []).strides()
    let sb = Values(dims: bd, data: []).strides()
    let data = (0..<out.count).map { p -> Double in
      let i = out.index(p)
      let ia = zip(zip(i, ad), sa).map { ($0.1 == 1 ? 0 : $0.0) * $1 }.reduce(0, +)
      let ib = zip(zip(i, bd), sb).map { ($0.1 == 1 ? 0 : $0.0) * $1 }.reduce(0, +)
      return f(a.data[ia], b.data[ib])
    }
    return Values(dims: dims, data: data)
  }
}
