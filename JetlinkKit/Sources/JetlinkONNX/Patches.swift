import Foundation

/// The graph rewrites of jetlink/onnx_patch.py on the `simplify` branch, on
/// the decoded model. Each one follows its Python function step by step,
/// including the order it appends initializers in and the errors it raises,
/// so the file it leads to is the one Python writes. The reasons for each
/// rewrite are in the Python docstrings; the comments here are about the port.
///
/// There is no shape inferrer in Swift. Where the Python asks onnx's shape
/// inferrer for a shape the file does not record, the rewrite that needed it
/// is skipped here, as the Python skips it when the inferrer finds none. The
/// driving models' exports record every tensor's shape, so on them nothing
/// is skipped and the output is the Python's byte for byte.
enum Patches {
  /// Where the vision trunk starts (onnx_patch.IMG_INPUTS).
  static let imageInputs = ["img", "big_img", "new_img", "state_img_q"]
  static let imageInputsRepr = "('img', 'big_img', 'new_img', 'state_img_q')"
  /// Ops that only move image bytes around (onnx_patch.LAYOUT_OPS).
  static let layoutOps: Set<String> = [
    "Concat", "Slice", "Gather", "Reshape", "Unsqueeze", "Squeeze", "Transpose", "Flatten", "Identity", "Expand",
  ]
  static let tinygradDomain = "org.tinygrad"
  static let passthroughOps: Set<String> = ["Contiguous"]
  /// A weight smaller than this stays as it is (onnx_patch.BLOB_MIN_ELEMENTS).
  static let blobMinElements: Int64 = 1024

  // MARK: strip_tinygrad_ops

  /// Bypasses tinygrad's layout-hint nodes and returns how many went.
  static func stripTinygradOps(_ g: inout Graph, _ opsets: inout [OpsetImport]) throws -> Int {
    let graphOutputs = Set(g.outputs.map(\.key))
    var removed = 0
    var k = 0
    while k < g.nodes.count {
      let node = g.nodes[k]
      guard node.domainName == tinygradDomain else {
        k += 1
        continue
      }
      guard passthroughOps.contains(node.op) else {
        throw OnnxError("unknown \(tinygradDomain) op \(pyRepr(node.op)); it may not be a no-op, so dropping it is not safe")
      }
      guard node.inputs.count == 1, node.outputs.count == 1, node.attributes.isEmpty else {
        throw OnnxError("\(node.op) is not a plain one-in one-out passthrough")
      }
      let source = node.inputs[0]
      let produced = node.outputs[0]
      if graphOutputs.contains(produced) {
        // The output keeps its name, so its producer takes the name over.
        rename(&g, outputs: source, to: produced)
        rename(&g, inputs: source, to: produced)
      } else {
        rename(&g, inputs: produced, to: source)
      }
      // Nothing before it moved, so it is still at k; the next node is now there.
      g.nodes.remove(at: k)
      removed += 1
    }

    if removed > 0 {
      if let i = opsets.firstIndex(where: { $0.domain == tinygradDomain }) {
        opsets.remove(at: i)
      }
      var live = Set<String>()
      for node in g.nodes {
        live.formUnion(node.inputs)
        live.formUnion(node.outputs)
      }
      g.valueInfo.removeAll { !live.contains($0.key) }
    }
    return removed
  }

  // MARK: patch_uint8_inputs

  static func needsPatch(_ g: Graph) -> Bool {
    g.inputs.contains { $0.elemType == DataType.uint8 }
  }

  /// Retypes the uint8 inputs, the images, to fp16 and drops the head Cast.
  static func patchUint8Inputs(_ g: inout Graph) throws {
    let sources = g.inputs.filter { $0.elemType == DataType.uint8 }.map(\.key)
    guard !sources.isEmpty else {
      throw OnnxError("model has no uint8 inputs; already patched?")
    }

    // _image_dataflow: nodes are in topological order, so one pass sees every
    // producer before its consumers.
    var retyped = Set(sources)
    var casts: [Int] = []
    for (k, node) in g.nodes.enumerated() where node.inputs.contains(where: retyped.contains) {
      if node.op == "Cast" {
        casts.append(k)
      } else if layoutOps.contains(node.op) {
        retyped.formUnion(node.outputs)
      } else {
        throw OnnxError("\(node.op) \(pyRepr(node.displayName)) reads the uint8 images; only layout ops "
          + "and the head Cast are expected there")
      }
    }
    guard !casts.isEmpty else {
      throw OnnxError("could not find the head Cast: no Cast ends the uint8 image chain")
    }
    let graphOutputs = Set(g.outputs.map(\.key))
    for k in casts {
      let cast = g.nodes[k]
      guard let to = cast.attribute("to") else {
        throw OnnxError("head Cast \(cast.displayName) has no 'to' attribute")
      }
      guard to.i == Int64(DataType.float16) else {
        throw OnnxError("head Cast targets \(to.i), expected FLOAT16 (\(DataType.float16))")
      }
      if let out = cast.outputs.first, graphOutputs.contains(out) {
        throw OnnxError("head Cast \(cast.displayName) feeds a graph output; dropping it would rename one")
      }
    }

    // Everything downstream reads each cast's source directly now. Python
    // removes each cast right after its renames; removing them all after all
    // the renames ends the same, since a rename that reaches a cast only
    // touches a node that is dropped anyway.
    for k in casts {
      let cast = g.nodes[k]
      if let source = cast.inputs.first, let produced = cast.outputs.first {
        rename(&g, inputs: produced, to: source)
      }
    }
    for k in casts.reversed() {
      g.nodes.remove(at: k)
    }

    retype(&g.inputs, retyped)
    retype(&g.valueInfo, retyped)
    retype(&g.outputs, retyped)
  }

  private static func retype(_ values: inout [ValueInfo], _ names: Set<String>) {
    for i in values.indices where names.contains(values[i].key) {
      var type = values[i].type ?? TypeInfo()
      if type.tensor == nil {
        // Setting tensor_type in Python selects it in TypeProto's oneof,
        // which clears whichever other member was set.
        type.extras.removeAll { [4, 5, 8, 9].contains($0.number) }
        type.tensor = TensorType()
      }
      type.tensor!.elemType = DataType.float16
      values[i].type = type
    }
  }

  // MARK: normalize_gather_indices

  /// Rewrites Gathers whose constant index is negative to the positive
  /// equivalent, each with an index initializer of its own.
  static func normalizeGatherIndices(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    let gathers = g.nodes.indices.filter { k in
      let n = g.nodes[k]
      return n.op == "Gather" && n.inputs.count > 1 && initializers[n.inputs[1]] != nil
    }
    let dims = staticDims(g)
    var rewritten = 0
    for k in gathers {
      let node = g.nodes[k]
      let index = g.initializers[initializers[node.inputs[1]]!]
      guard DataType.isInteger(index.elementType) else { continue }
      let values = try Elements.integers(index, src)
      guard let lowest = values.min(), lowest < 0 else { continue }
      let axis = node.attribute("axis")?.i ?? 0
      guard let shape = dims[node.inputs[0]], axis < Int64(shape.count) else { continue }
      // Python indexes the shape tuple with the axis, so a negative one counts from the end.
      let position = axis < 0 ? Int(axis) + shape.count : Int(axis)
      guard position >= 0 else {
        throw OnnxError("\(node.displayName): Gather axis \(axis) is out of range for a rank-\(shape.count) input")
      }
      let size = shape[position]
      guard size > 0 else { continue }
      let fixed = values.map { $0 < 0 ? $0 + size : $0 }
      if fixed.contains(where: { $0 < 0 || $0 >= size }) {
        throw OnnxError("\(node.displayName): Gather index \(pyList(values, index.dims)) out of range "
          + "for axis \(axis) of size \(size)")
      }
      let name = "\(node.outputs.first ?? "")__index"
      g.initializers.append(Tensor(name: name, dims: index.dims, dataType: index.dataType,
                                   raw: .owned(try Elements.encode(fixed, as: index.elementType, for: name))))
      g.nodes[k].inputs[1] = name
      rewritten += 1
    }
    return rewritten
  }

  // MARK: expand_to_tile

  /// Rewrites an Expand with a constant shape of the input's rank, which only
  /// repeats size-1 axes, as the equivalent Tile.
  static func expandToTile(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    let dims = staticDims(g)
    var done = 0
    for k in g.nodes.indices {
      let n = g.nodes[k]
      guard n.op == "Expand", n.inputs.count == 2, let shapeIndex = initializers[n.inputs[1]] else { continue }
      let shape = dims[n.inputs[0]]
      let target = try Elements.integers(g.initializers[shapeIndex], src)
      guard let shape, shape.count == target.count, !shape.contains(where: { $0 <= 0 }) else { continue }
      var repeats: [Int64] = []
      var repeatsOnly = true
      for (have, want) in zip(shape, target) {
        if want == 1 || want == have {
          repeats.append(1)
        } else if have == 1 {
          repeats.append(want)
        } else {
          repeatsOnly = false
          break
        }
      }
      guard repeatsOnly else { continue }
      let name = "\(n.outputs.first ?? "")__repeats"
      g.initializers.append(int64Tensor(repeats, name))
      g.nodes[k].opType = "Tile"
      g.nodes[k].inputs[1] = name
      done += 1
    }
    return done
  }

  // MARK: gemm_with_transposed_weight

  /// Rewrites `MatMul(x, W)` followed by `Add(b)` as `Gemm(x, W.T, b,
  /// transB=1)`, with a Reshape either side when x has more than two axes.
  ///
  /// The transposed weights are not made here. Each new initializer is a
  /// recipe over the original's bytes in the mapped file, carried out block
  /// by block when the part is written, so no two are ever in memory at once.
  static func gemmWithTransposedWeight(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    var consumers: [String: [Int]] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in node.inputs {
        consumers[name, default: []].append(k)
      }
    }
    let outputs = Set(g.outputs.map(\.key))
    let candidates = g.nodes.indices.filter { k in
      let n = g.nodes[k]
      return n.op == "MatMul" && n.inputs.count == 2 && initializers[n.inputs[1]] != nil
        && !(n.outputs.first.map(outputs.contains) ?? true)
    }
    let dims = staticDims(g)

    var replacements: [Int: [Node]] = [:]
    var drop = Set<Int>()
    var rewritten = 0
    for k in candidates {
      let node = g.nodes[k]
      let weight = g.initializers[initializers[node.inputs[1]]!]
      guard weight.dims.count == 2, weight.dims[0] * weight.dims[1] >= blobMinElements else { continue }
      let after = consumers[node.outputs[0]] ?? []
      guard after.count == 1, g.nodes[after[0]].op == "Add" else { continue }
      let add = g.nodes[after[0]]
      // The bias has to broadcast over the output's last axis; anything else
      // is a real elementwise Add and not a Gemm's C.
      guard let bias = add.inputs.first(where: { initializers[$0] != nil }),
            g.initializers[initializers[bias]!].dims == [weight.dims[1]] else { continue }
      guard let shape = dims[node.inputs[0]], shape.count >= 2, shape.last == weight.dims[0],
            !shape.contains(where: { $0 <= 0 }) else { continue }

      let stem = node.outputs[0]
      let transposed = "\(stem)__wt"
      g.initializers.append(try transposedTensor(weight, transposed, src))

      var new: [Node] = []
      var a = node.inputs[0]
      if shape.count > 2 {
        let flat = "\(stem)__flat_shape"
        g.initializers.append(int64Tensor([-1, weight.dims[0]], flat))
        a = "\(stem)__flat"
        new.append(Node(inputs: [node.inputs[0], flat], outputs: [a], name: "\(stem)__reshape_in", opType: "Reshape"))
      }
      let gemmOut = shape.count == 2 ? add.outputs[0] : "\(stem)__gemm"
      new.append(Node(inputs: [a, transposed, bias], outputs: [gemmOut], name: "\(stem)__gemm", opType: "Gemm",
                      attributes: [.int("transB", 1)]))
      if shape.count > 2 {
        let back = "\(stem)__out_shape"
        g.initializers.append(int64Tensor(Array(shape.dropLast()) + [weight.dims[1]], back))
        new.append(Node(inputs: [gemmOut, back], outputs: [add.outputs[0]], name: "\(stem)__reshape_out",
                        opType: "Reshape"))
      }
      replacements[k] = new
      drop.insert(after[0])
      rewritten += 1
    }

    guard rewritten > 0 else { return 0 }

    var rebuilt: [Node] = []
    rebuilt.reserveCapacity(g.nodes.count + 2 * rewritten)
    for (k, node) in g.nodes.enumerated() where !drop.contains(k) {
      if let new = replacements[k] {
        rebuilt.append(contentsOf: new)
      } else {
        rebuilt.append(node)
      }
    }
    g.nodes = rebuilt
    dropUnusedInitializers(&g)
    return rewritten
  }

  /// The weights the rewrite left behind (_drop_unused_initializers).
  static func dropUnusedInitializers(_ g: inout Graph) {
    var used = Set<String>()
    for node in g.nodes {
      used.formUnion(node.inputs)
    }
    g.initializers.removeAll { !used.contains($0.key) }
  }

  /// `numpy_helper.from_array(np.ascontiguousarray(W.T), name)`: the same
  /// element type, the dims swapped, the data as raw_data.
  private static func transposedTensor(_ weight: Tensor, _ name: String, _ src: Source) throws -> Tensor {
    let type = weight.elementType
    guard let size = DataType.size(type) else {
      throw OnnxError("\(weight.key): cannot transpose a weight of element type \(type)")
    }
    let rows = Int(weight.dims[0])
    let cols = Int(weight.dims[1])
    let expected = rows * cols * size
    let elements: Transpose.Elements
    switch weight.raw {
    case .source(let r):
      guard r.count == expected else { throw sizeMismatch(weight, r.count, expected) }
      elements = .source(r)
    case .owned(let bytes):
      guard bytes.count == expected else { throw sizeMismatch(weight, bytes.count, expected) }
      elements = .owned(bytes)
    case .transposed:
      throw OnnxError("\(weight.key): a transposed weight cannot be transposed again")
    case nil:
      // float_data and double_data, packed in one run, are already the raw
      // layout. Anything else is decoded when it is written.
      let field = Elements.typedField(type)
      if let field, field == 4 || field == 10, type == DataType.float || type == DataType.double,
         let ranges = weight.typed[field], ranges.count == 1, ranges[0].count == expected {
        elements = .source(ranges[0])
      } else {
        let count = try Elements.typedCount(weight, src)
        guard count == rows * cols else { throw sizeMismatch(weight, count * size, expected) }
        elements = .typed(weight)
      }
    }
    let transpose = Transpose(elements: elements, rows: rows, cols: cols, elementSize: size)
    return Tensor(name: name, dims: [weight.dims[1], weight.dims[0]], dataType: weight.dataType,
                  raw: .transposed(transpose))
  }

  private static func sizeMismatch(_ t: Tensor, _ have: Int, _ want: Int) -> OnnxError {
    OnnxError("initializer \(t.key) has \(have) bytes of data where its dims \(t.dims) need \(want)")
  }

  // MARK: helpers

  /// Static shapes from what the file carries (_static_dims), inputs first,
  /// then value_info, then outputs, a later entry replacing an earlier one.
  /// A dimension without a value is -1.
  static func staticDims(_ g: Graph) -> [String: [Int64]] {
    var dims: [String: [Int64]] = [:]
    for vi in g.inputs + g.valueInfo + g.outputs {
      if let shape = vi.shape {
        dims[vi.key] = shape.dims.map { $0.value ?? -1 }
      }
    }
    return dims
  }

  /// `{t.name: t for t in g.initializer}` as positions: a repeated name maps
  /// to its last tensor, as a dict built that way does.
  static func lastIndexByName(_ tensors: [Tensor]) -> [String: Int] {
    var map: [String: Int] = [:]
    for (i, t) in tensors.enumerated() {
      map[t.key] = i
    }
    return map
  }

  static func int64Tensor(_ values: [Int64], _ name: String) -> Tensor {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(values.count * 8)
    for v in values {
      withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) }
    }
    return Tensor(name: name, dims: [Int64(values.count)], dataType: DataType.int64, raw: .owned(bytes))
  }

  private static func rename(_ g: inout Graph, inputs from: String, to: String) {
    for k in g.nodes.indices {
      for i in g.nodes[k].inputs.indices where g.nodes[k].inputs[i] == from {
        g.nodes[k].inputs[i] = to
      }
    }
  }

  private static func rename(_ g: inout Graph, outputs from: String, to: String) {
    for k in g.nodes.indices {
      for i in g.nodes[k].outputs.indices where g.nodes[k].outputs[i] == from {
        g.nodes[k].outputs[i] = to
      }
    }
  }
}

// MARK: Python's formatting, for error messages that match

/// Python's repr of a str.
func pyRepr(_ s: String) -> String {
  let quote: Character = (s.contains("'") && !s.contains("\"")) ? "\"" : "'"
  var out = String(quote)
  for scalar in s.unicodeScalars {
    switch scalar {
    case "\\": out += "\\\\"
    case "\n": out += "\\n"
    case "\r": out += "\\r"
    case "\t": out += "\\t"
    default:
      if Character(scalar) == quote {
        out += "\\\(quote)"
      } else if scalar.value < 0x20 || scalar.value == 0x7f {
        out += String(format: "\\x%02x", scalar.value)
      } else {
        out.unicodeScalars.append(scalar)
      }
    }
  }
  out.append(quote)
  return out
}

/// numpy's `tolist()` of integers, as Python prints it.
func pyList(_ values: [Int64], _ dims: [Int64]) -> String {
  guard let first = dims.first else { return values.first.map(String.init) ?? "[]" }
  let rest = Array(dims.dropFirst())
  let stride = max(1, rest.reduce(1) { $0 * Int($1) })
  let items = (0..<Int(first)).map { i -> String in
    if rest.isEmpty { return String(values[i]) }
    return pyList(Array(values[(i * stride)..<((i + 1) * stride)]), rest)
  }
  return "[" + items.joined(separator: ", ") + "]"
}

/// Python's sort order for str: by code point, with no normalisation.
func pyLess(_ a: String, _ b: String) -> Bool {
  a.unicodeScalars.lexicographicallyPrecedes(b.unicodeScalars) { $0.value < $1.value }
}
