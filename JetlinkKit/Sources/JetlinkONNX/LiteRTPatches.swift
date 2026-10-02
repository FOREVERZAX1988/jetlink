import Foundation

/// What `Patches.forLiteRT` did, rewrite by rewrite. A rewrite that finds
/// nothing it matches counts 0 and leaves the graph as it was.
public struct LiteRTRewrites: Sendable, Equatable {
  /// tinygrad layout ops bypassed.
  public internal(set) var stripped = 0
  /// Gathers whose negative constant index was rewritten.
  public internal(set) var gathers = 0
  /// GatherNDs whose negative constant indices were rewritten.
  public internal(set) var gatherNDs = 0
  /// 5-D attention reshapes made 4-D, each with what split it into heads.
  public internal(set) var attention = 0
  /// rank-5 frame queues carried as their 4-D view.
  public internal(set) var frameQueues = 0
  /// Constant-index GatherNDs made Slices.
  public internal(set) var gatherNDSlices = 0
  /// Constant-index Gathers made Slices.
  public internal(set) var gatherSlices = 0
  /// LayerNormalizations written out in the fp16-safe scaled form.
  public internal(set) var layerNorms = 0
  /// Squeezes and Unsqueezes made static Reshapes.
  public internal(set) var reshapes = 0
}

/// The rewrites that make a driving model something LiteRT's GPU runs whole,
/// ported from the LiteRT spike's prep_onnx.py and prep_gpu.py --all
/// (plans/litert/spike). Each is exact in real arithmetic, each is a pattern
/// match that leaves the graph alone when the pattern is absent, and every
/// tensor one makes gets a value info with its element type and static
/// shape, because the TFLite writer that reads the graph sizes every tensor
/// from those. uint8 inputs stay uint8: LiteRT takes the frames as bytes.
///
/// No weight is read or copied: a rewrite reads only small index and shape
/// constants and adds only small ones of its own, so the graph still streams
/// from the mapped source the way the CoreML preparation's does.
extension Patches {
  /// The model as the LiteRT path reads it: decoded from the mapped source
  /// and rewritten by `forLiteRT`. Both the `.liteRT` layout and the TFLite
  /// writer start here, so they see the same graph.
  static func liteRTModel(_ src: Source) throws -> (model: Model, rewrites: LiteRTRewrites) {
    var model = try Decode.model(src)
    guard var g = model.graph else { throw OnnxError("the model has no graph") }
    if let t = g.initializers.first(where: \.isExternal) {
      throw OnnxError("initializer \(t.key) keeps its data in an external file, which the preparation does not read")
    }
    let rewrites = try forLiteRT(&g, &model.opsets, src)
    model.graph = g
    return (model, rewrites)
  }

  /// Every LiteRT rewrite, in the order prep_gpu.py --all applies them. The
  /// opsets come in because stripping tinygrad's ops drops its opset, and
  /// because ReduceMean and ReduceMax take their axes as an input from opset
  /// 18 and as an attribute before it.
  ///
  /// prep_gpu.py decides the attention, Gather and queue rewrites in one
  /// pass, its Gather rewrite skipping the queue's Gathers; here the queue
  /// goes before the Gathers, which comes to the same.
  static func forLiteRT(_ g: inout Graph, _ opsets: inout [OpsetImport], _ src: Source) throws -> LiteRTRewrites {
    var r = LiteRTRewrites()
    r.stripped = try stripTinygradOps(&g, &opsets)
    let opset = opsets.first { $0.domain == "" || $0.domain == "ai.onnx" }?.version ?? 0
    r.gathers = try normalizeGatherIndices(&g, src)
    r.gatherNDs = try normalizeGatherNDIndices(&g, src)
    r.attention = try attentionIn4D(&g, src, opset: opset)
    r.frameQueues = try frameQueuesIn4D(&g, src, opset: opset)
    r.gatherNDSlices = try gatherNDsAsSlices(&g, src)
    r.gatherSlices = try gathersAsSlices(&g, src)
    r.layerNorms = try scaledLayerNorms(&g, opset: opset)
    r.reshapes = try squeezesAsReshapes(&g, src, opset: opset)
    dropUnusedInitializers(&g)
    dropStaleValueInfo(&g)
    return r
  }

  // MARK: GatherND indices (prep_onnx.py)

  /// Rewrites GatherNDs whose constant indices are negative to the positive
  /// equivalent, each with an index initializer of its own. onnx2tf fixed
  /// Gather's but not GatherND's, and LiteRT's GATHER_ND refuses negatives
  /// when it runs.
  static func normalizeGatherNDIndices(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    let dims = staticDims(g)
    var rewritten = 0
    for k in g.nodes.indices {
      let node = g.nodes[k]
      guard node.op == "GatherND", node.inputs.count == 2, let at = initializers[node.inputs[1]] else { continue }
      let index = g.initializers[at]
      guard DataType.isInteger(index.elementType), let depth = index.dims.last.map(Int.init), depth > 0 else { continue }
      let values = try Elements.integers(index, src)
      guard let lowest = values.min(), lowest < 0 else { continue }
      let batch = Int(node.attribute("batch_dims")?.i ?? 0)
      guard let shape = dims[node.inputs[0]] ?? initializers[node.inputs[0]].map({ g.initializers[$0].dims }),
        shape.count >= batch + depth
      else { continue }
      let sizes = shape[batch..<(batch + depth)]
      guard !sizes.contains(where: { $0 <= 0 }) else { continue }
      let fixed = values.enumerated().map { i, v in v < 0 ? v + sizes[batch + i % depth] : v }
      if let bad = fixed.enumerated().first(where: { i, v in v < 0 || v >= sizes[batch + i % depth] }) {
        throw OnnxError("\(node.displayName): GatherND index \(values[bad.offset]) out of range for an axis of size \(sizes[batch + bad.offset % depth])")
      }
      let name = "\(node.outputs.first ?? "")__index"
      g.initializers.append(
        Tensor(name: name, dims: index.dims, dataType: index.dataType, raw: .owned(try Elements.encode(fixed, as: index.elementType, for: name))))
      g.nodes[k].inputs[1] = name
      rewritten += 1
    }
    return rewritten
  }

  // MARK: attention in 4-D (prep_gpu.py --attn4d)

  /// Rewrites an attention block's 5-D split into heads, `Reshape [B,T,K*H*D]
  /// -> [B,T,K,H,D]`, an optional Transpose, then a Split, Slices or Gathers
  /// that take the K chunks apart and the Squeezes that drop the size-1 axis,
  /// as one `Slice` of the last axis, a `Reshape [B,T,H,D]` and, if the heads
  /// were transposed, a 4-D `Transpose` per chunk. LiteRT's GPU has no rank-5
  /// tensors. Matched by shape and op, not by node names: Cinque Terre's
  /// transposed Split and the older models' unbind both fit.
  static func attentionIn4D(_ g: inout Graph, _ src: Source, opset: Int64) throws -> Int {
    var rw = GraphRewrite(g, src)
    var done = 0
    for (at, reshape) in g.nodes.enumerated() where reshape.op == "Reshape" && reshape.inputs.count == 2 {
      guard let x = reshape.inputs.first, let wide = reshape.outputs.first, let type = rw.types[x],
        let xd = rw.dims(x), let vd = rw.dims(wide), xd.count == 3, vd.count == 5,
        xd[0] == vd[0], xd[1] == vd[1], vd[2] * vd[3] * vd[4] == xd[2], !rw.outputs.contains(wide)
      else { continue }
      let (b, t, chunks, heads, depth) = (vd[0], vd[1], vd[2], vd[3], vd[4])
      var perm = [0, 1, 2, 3, 4]
      var split = wide
      var gone: [Int] = []
      let readers = rw.consumers[wide] ?? []
      if readers.count == 1, g.nodes[readers[0]].op == "Transpose" {
        let transpose = g.nodes[readers[0]]
        guard let p = transpose.attribute("perm")?.ints.map({ Int($0) }), p.sorted() == perm,
          let out = transpose.outputs.first, !rw.outputs.contains(out)
        else { continue }
        perm = p
        split = out
        gone.append(readers[0])
      }
      let axis = perm.firstIndex(of: 2)!
      guard let picks = try headPicks(g, rw, split, axis: axis, chunks: chunks, opset: opset) else { continue }

      // The heads come out in the transpose's order with the chunk axis gone.
      let order = perm.filter { $0 != 2 }.map { $0 > 2 ? $0 - 1 : $0 }
      let flat = [b, t, heads, depth]
      let expected = order.map { flat[$0] }
      guard picks.outputs.allSatisfy({ rw.dims($0.output).map { $0 == expected } ?? true }) else { continue }

      var new: [Node] = []
      let width = heads * depth
      for (chunk, output) in picks.outputs {
        let slice = "\(output)__qkv"
        new.append(
          rw.node(
            "Slice", [x, rw.ints([chunk * width]), rw.ints([(chunk + 1) * width]), rw.ints([2])], slice,
            type: type, dims: [b, t, width]))
        if order == [0, 1, 2, 3] {
          new.append(rw.node("Reshape", [slice, rw.ints(flat)], output, type: type, dims: flat))
        } else {
          let unflat = "\(output)__heads"
          new.append(rw.node("Reshape", [slice, rw.ints(flat)], unflat, type: type, dims: flat))
          new.append(
            rw.node("Transpose", [unflat], output, type: type, dims: expected, attributes: [.ints("perm", order.map(Int64.init))]))
        }
      }
      rw.replace(at, with: new)
      for k in gone + picks.nodes {
        rw.drop(k)
      }
      done += 1
    }
    rw.apply(&g)
    return done
  }

  /// What takes the 5-D heads apart along `axis`: each chunk's 4-D output
  /// with the chunk it holds, and the nodes that made them, or nil if
  /// anything else reads the 5-D tensor.
  private static func headPicks(
    _ g: Graph, _ rw: GraphRewrite, _ wide: String, axis: Int, chunks: Int64, opset: Int64
  ) throws -> (outputs: [(chunk: Int64, output: String)], nodes: [Int])? {
    var outputs: [(chunk: Int64, output: String)] = []
    var nodes: [Int] = []
    /// The Squeezes of `one`, a size-1 chunk, that drop the chunk axis.
    func squeezed(_ one: String, _ chunk: Int64) throws -> Bool {
      guard !rw.outputs.contains(one) else { return false }
      for k in rw.consumers[one] ?? [] {
        let n = g.nodes[k]
        guard n.op == "Squeeze", let out = n.outputs.first, try rw.axes(n, rank: 5, opset: opset) == [axis] else { return false }
        outputs.append((chunk, out))
        nodes.append(k)
      }
      return true
    }
    for k in rw.consumers[wide] ?? [] {
      let n = g.nodes[k]
      nodes.append(k)
      switch n.op {
      case "Gather":
        guard n.inputs.count == 2, normalized(n.attribute("axis")?.i ?? 0, 5) == axis,
          rw.initializerDims(n.inputs[1]) == [], let index = try rw.integers(n.inputs[1])?.first,
          let out = n.outputs.first
        else { return nil }
        let chunk = index < 0 ? index + chunks : index
        guard (0..<chunks).contains(chunk) else { return nil }
        outputs.append((chunk, out))
      case "Slice":
        guard let picked = try rw.slice(n), picked.count == 1, let only = picked[axis], only.count == 1,
          let out = n.outputs.first, try squeezed(out, only[0])
        else { return nil }
      case "Split":
        guard normalized(n.attribute("axis")?.i ?? 0, 5) == axis, n.outputs.count == Int(chunks) else { return nil }
        if n.inputs.count > 1, !n.inputs[1].isEmpty {
          guard try rw.integers(n.inputs[1]) == Array(repeating: 1, count: Int(chunks)) else { return nil }
        } else if let sizes = n.attribute("split")?.ints, !sizes.isEmpty {
          guard sizes == Array(repeating: 1, count: Int(chunks)) else { return nil }
        }
        for (chunk, out) in n.outputs.enumerated() {
          guard try squeezed(out, Int64(chunk)) else { return nil }
        }
      default:
        return nil
      }
    }
    // a guard, not a ternary: the Android toolchain's type checker gives up on nil beside a tuple
    guard !outputs.isEmpty else { return nil }
    return (outputs, nodes)
  }

  // MARK: the frame queue in 4-D (prep_gpu.py --img4d)

  /// Carries a rank-5 frame queue, a graph input `[B,F,C,H,W]` shifted into
  /// a graph output of the same shape by `Concat(Slice(queue, frames),
  /// Unsqueeze(new frame))`, as its 4-D view `[B,F*C,H,W]`: the same bytes,
  /// so a server that checks element counts feeds it unchanged, but the
  /// model's input and output shapes change. The Gathers that pick a camera
  /// out of the queue become Slices of the view; a camera's frames picked by
  /// a Slice after one become channel Slices of the view, concatenated, which
  /// is what prep_gpu.py made of Cinque Terre's every-fourth-frame read.
  static func frameQueuesIn4D(_ g: inout Graph, _ src: Source, opset: Int64) throws -> Int {
    var rw = GraphRewrite(g, src)
    var done = 0
    var reshaped: [(name: String, dims: [Int64])] = []
    for input in g.inputs {
      guard let queue = try frameQueue(g, rw, input.key, opset: opset) else { continue }
      let (b, f, c, h, w) = (queue.dims[0], queue.dims[1], queue.dims[2], queue.dims[3], queue.dims[4])
      let view = [b, f * c, h, w]
      var parts: [String] = []
      for part in queue.parts {
        switch part {
        case .frames(let k, let range):
          let out = "\(g.nodes[k].outputs[0])__4d"
          let slice = rw.node(
            "Slice", [input.key, rw.ints([range.lowerBound * c]), rw.ints([range.upperBound * c]), rw.ints([1])], out,
            type: queue.type, dims: [b, Int64(range.count) * c, h, w])
          rw.replace(k, with: [slice])
          parts.append(out)
        case .frame(let k, let frame):
          rw.drop(k)
          parts.append(frame)
        }
      }
      rw.replace(queue.concat, with: [rw.node("Concat", parts, queue.shifted, type: queue.type, dims: view, attributes: [.int("axis", 1)])])
      for (k, from) in queue.readers {
        try readCamera(g, &rw, k, from, view: view, frames: f, type: queue.type)
      }
      reshaped += [(input.key, view), (queue.shifted, view)]
      done += 1
    }
    rw.apply(&g)
    for (name, dims) in reshaped {
      setDims(&g, name, dims)
    }
    return done
  }

  /// A frame queue as `frameQueuesIn4D` finds it.
  private struct FrameQueue {
    enum Part {
      /// A Slice of the queue's frames, at node k.
      case frames(Int, Range<Int64>)
      /// A new frame unsqueezed by node k.
      case frame(Int, String)
    }

    let dims: [Int64]
    let type: Int32
    /// The Concat that shifts the queue, its output, and its inputs in order.
    let concat: Int
    let shifted: String
    let parts: [Part]
    /// The Gathers that read a camera, with the queue tensor they read.
    let readers: [(node: Int, from: String)]
  }

  /// `name` as a frame queue, or nil if it is not one or anything reads it
  /// or its shifted copy other than the shift and the camera Gathers.
  private static func frameQueue(_ g: Graph, _ rw: GraphRewrite, _ name: String, opset: Int64) throws -> FrameQueue? {
    guard let dims = rw.dims(name), dims.count == 5, let type = rw.types[name] else { return nil }
    var slices: [String: (node: Int, frames: Range<Int64>)] = [:]
    var readers: [(node: Int, from: String)] = []
    var concat: Int?
    for k in rw.consumers[name] ?? [] {
      let n = g.nodes[k]
      if let picked = try rw.slice(n), picked.count == 1, let frames = picked[1], let first = frames.first,
        frames == Array(first..<(first + Int64(frames.count))), let out = n.outputs.first, !rw.outputs.contains(out),
        let after = rw.consumers[out], after.count == 1, concat == nil || concat == after[0]
      {
        slices[out] = (k, first..<(first + Int64(frames.count)))
        concat = after[0]
      } else if try camera(n, rw, cameras: dims[0]) != nil {
        readers.append((k, name))
      } else {
        return nil
      }
    }
    guard let concat else { return nil }
    let cat = g.nodes[concat]
    guard cat.op == "Concat", normalized(cat.attribute("axis")?.i ?? 0, 5) == 1, let shifted = cat.outputs.first,
      rw.outputs.contains(shifted), rw.dims(shifted) == dims
    else { return nil }

    // The Concat's inputs: the queue's Slices, and new frames unsqueezed.
    var parts: [FrameQueue.Part] = []
    var frames: Int64 = 0
    for part in cat.inputs {
      if let slice = slices[part] {
        parts.append(.frames(slice.node, slice.frames))
        frames += Int64(slice.frames.count)
      } else if let k = rw.producers[part], g.nodes[k].op == "Unsqueeze", let frame = g.nodes[k].inputs.first,
        rw.dims(frame) == [dims[0]] + dims[2...], try rw.axes(g.nodes[k], rank: 5, opset: opset) == [1],
        rw.consumers[part]?.count == 1, !rw.outputs.contains(part)
      {
        parts.append(.frame(k, frame))
        frames += 1
      } else {
        return nil
      }
    }
    guard frames == dims[1] else { return nil }
    for k in rw.consumers[shifted] ?? [] {
      guard try camera(g.nodes[k], rw, cameras: dims[0]) != nil else { return nil }
      readers.append((k, shifted))
    }
    return FrameQueue(dims: dims, type: type, concat: concat, shifted: shifted, parts: parts, readers: readers)
  }

  /// The camera a `Gather(queue, b, axis=0)` with a constant scalar index
  /// picks, or nil if the node is not one.
  private static func camera(_ n: Node, _ rw: GraphRewrite, cameras: Int64) throws -> Int64? {
    guard n.op == "Gather", n.inputs.count == 2, normalized(n.attribute("axis")?.i ?? 0, 5) == 0,
      rw.initializerDims(n.inputs[1]) == [], let index = try rw.integers(n.inputs[1])?.first
    else { return nil }
    let camera = index < 0 ? index + cameras : index
    return (0..<cameras).contains(camera) ? camera : nil
  }

  /// One camera's read of the queue, the Gather at `k`, from the queue's 4-D
  /// view `from`.
  private static func readCamera(
    _ g: Graph, _ rw: inout GraphRewrite, _ k: Int, _ from: String, view: [Int64], frames: Int64, type: Int32
  ) throws {
    let camera = try camera(g.nodes[k], rw, cameras: view[0])!
    let picked = g.nodes[k].outputs[0]
    let (c, h, w) = (view[1] / frames, view[2], view[3])
    let after = rw.consumers[picked] ?? []
    guard after.count == 1, !rw.outputs.contains(picked), g.nodes[after[0]].inputs.first == picked,
      let spec = try rw.slice(g.nodes[after[0]]), spec.count == 1, let chosen = spec[0], !chosen.isEmpty
    else {
      let slice = "\(picked)__4d"
      rw.replace(
        k,
        with: [
          rw.node("Slice", [from, rw.ints([camera]), rw.ints([camera + 1]), rw.ints([0])], slice, type: type, dims: [1] + view.dropFirst()),
          rw.node("Reshape", [slice, rw.ints([frames, c, h, w])], picked, type: type, dims: [frames, c, h, w]),
        ])
      return
    }

    // A Slice of the camera's frames reads the view's channels directly, and
    // a Reshape after it to the channels' own shape is what they already are.
    let sliced = g.nodes[after[0]].outputs[0]
    let count = Int64(chosen.count)
    let flat = [1, count * c, h, w]
    var reshape: Int?
    if !rw.outputs.contains(sliced), let next = rw.consumers[sliced], next.count == 1, g.nodes[next[0]].op == "Reshape",
      let out = g.nodes[next[0]].outputs.first, rw.dims(out) == flat
    {
      reshape = next[0]
    }
    let target = reshape.map { g.nodes[$0].outputs[0] } ?? sliced
    let joined = reshape == nil ? "\(sliced)__frames" : target
    let ranges = runs(chosen)
    var new: [Node] = []
    var parts: [String] = []
    for run in ranges {
      let name = ranges.count == 1 ? joined : "\(target)__f\(run.lowerBound)"
      new.append(
        rw.node(
          "Slice", [from, rw.ints([camera, run.lowerBound * c]), rw.ints([camera + 1, run.upperBound * c]), rw.ints([0, 1])],
          name, type: type, dims: [1, Int64(run.count) * c, h, w]))
      parts.append(name)
    }
    if ranges.count > 1 {
      new.append(rw.node("Concat", parts, joined, type: type, dims: flat, attributes: [.int("axis", 1)]))
    }
    if reshape == nil {
      new.append(rw.node("Reshape", [joined, rw.ints([count, c, h, w])], sliced, type: type, dims: [count, c, h, w]))
    }
    rw.replace(k, with: new)
    rw.drop(after[0])
    if let reshape { rw.drop(reshape) }
  }

  // MARK: constant gathers as Slices (prep_gpu.py --gathernd, --gather)

  /// Rewrites a GatherND with constant indices into the first axis
  /// (batch_dims 0, one index per tuple) as `gathersAsSlices` does a Gather:
  /// it is the same Gather on axis 0.
  static func gatherNDsAsSlices(_ g: inout Graph, _ src: Source) throws -> Int {
    var rw = GraphRewrite(g, src)
    var done = 0
    for (k, n) in g.nodes.enumerated() where n.op == "GatherND" && n.inputs.count == 2 {
      guard (n.attribute("batch_dims")?.i ?? 0) == 0, let index = rw.initializerDims(n.inputs[1]), index.last == 1,
        let rows = try rw.integers(n.inputs[1]), let data = rw.dims(n.inputs[0]), !data.isEmpty, let type = rw.types[n.inputs[0]],
        let out = n.outputs.first
      else { continue }
      let shape = Array(index.dropLast()) + data.dropFirst()
      guard let new = rw.slices(n.inputs[0], data, axis: 0, rows, out, shape, type: type) else { continue }
      rw.replace(k, with: new)
      done += 1
    }
    rw.apply(&g)
    return done
  }

  /// Rewrites a Gather with constant indices as Slices of its runs of
  /// consecutive indices, concatenated, then reshaped to the Gather's shape
  /// where that differs. LiteRT's GPU cannot build the 225-index desire
  /// Gather. prep_gpu.py left a Gather of an initializer for onnx2tf to fold;
  /// with no folding after this, those go too, and one that takes the whole
  /// initializer in order, as the position embeddings do, is the initializer.
  static func gathersAsSlices(_ g: inout Graph, _ src: Source) throws -> Int {
    var rw = GraphRewrite(g, src)
    var done = 0
    for (k, n) in g.nodes.enumerated() where n.op == "Gather" && n.inputs.count == 2 {
      guard let index = rw.initializerDims(n.inputs[1]), let values = try rw.integers(n.inputs[1]),
        let data = rw.dims(n.inputs[0]), !data.isEmpty, let type = rw.types[n.inputs[0]], let out = n.outputs.first
      else { continue }
      let axis = normalized(n.attribute("axis")?.i ?? 0, data.count)
      guard (0..<data.count).contains(axis) else { continue }
      let shape = Array(data[..<axis]) + index + data[(axis + 1)...]
      guard let new = rw.slices(n.inputs[0], data, axis: axis, values, out, shape, type: type) else { continue }
      rw.replace(k, with: new)
      done += 1
    }
    rw.apply(&g)
    return done
  }

  // MARK: LayerNormalization (prep_gpu.py --ln)

  /// Writes each LayerNormalization out as
  ///
  ///     d = x - mean(x);  s = max(max|d|, 1e-2);  y = (d/s) / sqrt(mean((d/s)^2) + eps/s^2)
  ///
  /// over the normalized axes, then scale and bias. It is LayerNorm exactly in
  /// real arithmetic, but nothing in it exceeds 1 before the square root, where
  /// the plain form squares deviations up to 5.5e6 in Cinque Terre's norms:
  /// past fp16's 65504, which a GPU computing in fp16 turns into wrong outputs.
  ///
  /// The statistics are computed in stash_type, as LayerNormalization's own
  /// definition does: when that is float and x is not, x is cast up first and
  /// the normalized value cast back before the scale. prep_gpu.py worked in
  /// x's type throughout; the TFLite writer drops fp16 to fp32 Casts, so it
  /// reads the two the same. A norm whose mean or inverse deviation output is
  /// used is left alone.
  static func scaledLayerNorms(_ g: inout Graph, opset: Int64) throws -> Int {
    var rw = GraphRewrite(g, nil)
    var done = 0
    for (k, n) in g.nodes.enumerated() where n.op == "LayerNormalization" && n.inputs.count >= 2 {
      guard let x = n.inputs.first, let y = n.outputs.first,
        !n.outputs.dropFirst().contains(where: { rw.consumers[$0] != nil || rw.outputs.contains($0) }),
        let dims = rw.dims(x), !dims.isEmpty, let type = rw.types[x], DataType.isFloat(type)
      else { continue }
      let axis = normalized(n.attribute("axis")?.i ?? -1, dims.count)
      guard (0..<dims.count).contains(axis) else { continue }
      let epsilon = Double(n.attribute("epsilon")?.f ?? 1e-5)
      let stash = Int32(n.attribute("stash_type")?.i ?? Int64(DataType.float))
      guard DataType.isFloat(stash) else { continue }
      let math = stash == DataType.float ? stash : type
      let axes = (axis..<dims.count).map(Int64.init)
      var reduced = dims
      for a in axis..<dims.count { reduced[a] = 1 }

      var new: [Node] = []
      func op(_ name: String, _ inputs: [String], _ suffix: String, _ dims: [Int64], _ t: Int32 = math, _ attributes: [Attribute] = []) -> String {
        let out = suffix.isEmpty ? y : "\(y)__ln_\(suffix)"
        new.append(rw.node(name, inputs, out, type: t, dims: dims, attributes: attributes))
        return out
      }
      func reduce(_ name: String, _ input: String, _ suffix: String) -> String {
        if opset >= 18 {
          return op(name, [input, rw.ints(axes)], suffix, reduced, math, [.int("keepdims", 1)])
        }
        return op(name, [input], suffix, reduced, math, [.ints("axes", axes), .int("keepdims", 1)])
      }
      let xs = math == type ? x : op("Cast", [x], "x", dims, math, [.int("to", Int64(math))])
      let mean = reduce("ReduceMean", xs, "mean")
      let d = op("Sub", [xs, mean], "d", dims)
      let magnitude = op("Abs", [d], "abs", dims)
      let peak = reduce("ReduceMax", magnitude, "peak")
      let s = op("Max", [peak, rw.scalar(1e-2, math)], "s", reduced)
      let inverse = op("Reciprocal", [s], "r", reduced)
      let dn = op("Mul", [d, inverse], "dn", dims)
      let square = op("Mul", [dn, dn], "sq", dims)
      let variance = reduce("ReduceMean", square, "var")
      let inverse2 = op("Mul", [inverse, inverse], "r2", reduced)
      let scaledEpsilon = op("Mul", [inverse2, rw.scalar(epsilon, math)], "eps", reduced)
      let sum = op("Add", [variance, scaledEpsilon], "ve", reduced)
      let deviation = op("Sqrt", [sum], "sd", reduced)
      let inverseDeviation = op("Reciprocal", [deviation], "rs", reduced)
      let normal = op("Mul", [dn, inverseDeviation], math == type ? "y" : "y32", dims)
      let yt = math == type ? normal : op("Cast", [normal], "y", dims, type, [.int("to", Int64(type))])
      if n.inputs.count > 2, !n.inputs[2].isEmpty {
        let scaled = op("Mul", [yt, n.inputs[1]], "scaled", dims, type)
        _ = op("Add", [scaled, n.inputs[2]], "", dims, type)
      } else {
        _ = op("Mul", [yt, n.inputs[1]], "", dims, type)
      }
      rw.replace(k, with: new)
      done += 1
    }
    rw.apply(&g)
    return done
  }

  // MARK: Squeeze and Unsqueeze (prep_gpu.py --squeeze)

  /// Rewrites each Squeeze and Unsqueeze as a Reshape to its static output
  /// shape. Left as they are, the converter computes those shapes at run
  /// time (SHAPE, GATHER, RESHAPE): 264 dynamic tensors in Cinque Terre.
  static func squeezesAsReshapes(_ g: inout Graph, _ src: Source, opset: Int64) throws -> Int {
    var rw = GraphRewrite(g, src)
    var done = 0
    for k in g.nodes.indices {
      let n = g.nodes[k]
      guard n.op == "Squeeze" || n.op == "Unsqueeze", let x = n.inputs.first, let out = n.outputs.first else { continue }
      var shape = rw.dims(out)
      if shape == nil, let input = rw.dims(x) {
        shape = try reshaped(n, input, rw, opset: opset)
      }
      guard let shape, !shape.contains(where: { $0 <= 0 }) else { continue }
      g.nodes[k].opType = "Reshape"
      g.nodes[k].inputs = [x, rw.ints(shape)]
      g.nodes[k].attributes = []
      if rw.dims(out) == nil, let type = rw.types[x] {
        rw.info(out, type, shape)
      }
      done += 1
    }
    rw.apply(&g)
    return done
  }

  /// A Squeeze's or Unsqueeze's output shape from its input's, for one the
  /// export records no shape for.
  private static func reshaped(_ n: Node, _ input: [Int64], _ rw: GraphRewrite, opset: Int64) throws -> [Int64]? {
    if n.op == "Squeeze" {
      guard let axes = try rw.axes(n, rank: input.count, opset: opset) else {
        return input.filter { $0 != 1 }
      }
      guard axes.allSatisfy({ (0..<input.count).contains($0) && input[$0] == 1 }) else { return nil }
      return input.enumerated().filter { !axes.contains($0.offset) }.map(\.element)
    }
    let rank = input.count + ((try rw.axes(n, rank: 0, opset: opset))?.count ?? 0)
    guard let axes = try rw.axes(n, rank: rank, opset: opset), Set(axes).count == axes.count,
      axes.allSatisfy({ (0..<rank).contains($0) })
    else { return nil }
    var rest = input.makeIterator()
    return (0..<rank).map { axes.contains($0) ? 1 : rest.next()! }
  }

  // MARK: helpers

  /// A negative axis counted from the end of a rank-`rank` shape.
  static func normalized(_ axis: Int64, _ rank: Int) -> Int {
    axis < 0 ? Int(axis) + rank : Int(axis)
  }

  /// The indices onnx's Slice keeps along an axis of `size`, from start to
  /// end by step, after counting negatives from the end and clamping.
  static func sliceIndices(size: Int64, start: Int64, end: Int64, step: Int64) -> [Int64] {
    guard step != 0 else { return [] }
    var from = start < 0 ? start &+ size : start
    var to = end < 0 ? end &+ size : end
    if step > 0 {
      from = min(max(from, 0), size)
      to = min(max(to, 0), size)
      return Array(stride(from: from, to: to, by: Int(step)))
    }
    from = min(max(from, 0), size - 1)
    to = min(max(to, -1), size - 1)
    return Array(stride(from: from, to: to, by: Int(step)))
  }

  /// Ascending runs of consecutive values, as half-open ranges.
  static func runs(_ values: [Int64]) -> [Range<Int64>] {
    guard !values.isEmpty else { return [] }
    var out: [Range<Int64>] = []
    var start = 0
    for i in 1...values.count where i == values.count || values[i] != values[i - 1] + 1 {
      out.append(values[start]..<(values[i - 1] + 1))
      start = i
    }
    return out
  }

  /// Sets a tensor's shape wherever the graph records one: its input or
  /// output and its value info.
  static func setDims(_ g: inout Graph, _ name: String, _ dims: [Int64]) {
    func set(_ values: inout [ValueInfo]) {
      for i in values.indices where values[i].key == name {
        values[i].type?.tensor?.shape = .of(dims)
      }
    }
    set(&g.inputs)
    set(&g.outputs)
    set(&g.valueInfo)
  }

  /// The value infos of tensors no node reads or writes any more.
  static func dropStaleValueInfo(_ g: inout Graph) {
    var live = Set(g.inputs.map(\.key) + g.outputs.map(\.key))
    for node in g.nodes {
      live.formUnion(node.inputs)
      live.formUnion(node.outputs)
    }
    g.valueInfo.removeAll { !live.contains($0.key) }
  }
}

/// One rewrite's view of the graph and what it changes: the shapes and types
/// the file records, who reads and writes each tensor, the nodes to replace
/// or drop, and the constants and value infos to add. The graph is changed
/// once, by `apply`, so the indices stay valid while the rewrite looks.
struct GraphRewrite {
  private let src: Source?
  private let initializers: [String: Tensor]
  private var shapes: [String: [Int64]]
  private(set) var types: [String: Int32]
  let outputs: Set<String>
  let consumers: [String: [Int]]
  let producers: [String: Int]
  private var replacements: [Int: [Node]] = [:]
  private var dropped = Set<Int>()
  private var aliases: [String: String] = [:]
  private var constants: [Tensor] = []
  private var made = Set<String>()
  private var infos: [ValueInfo] = []

  /// `src` is where the constants the rewrite reads live; nil for one that
  /// reads none.
  init(_ g: Graph, _ src: Source?) {
    self.src = src
    var initializers: [String: Tensor] = [:]
    for t in g.initializers {
      initializers[t.key] = t
    }
    self.initializers = initializers
    shapes = Patches.staticDims(g)
    types = Patches.staticTypes(g)
    for t in g.initializers {
      shapes[t.key] = t.dims
      types[t.key] = t.elementType
    }
    outputs = Set(g.outputs.map(\.key))
    var consumers: [String: [Int]] = [:]
    var producers: [String: Int] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in Set(node.inputs) where !name.isEmpty {
        consumers[name, default: []].append(k)
      }
      for name in node.outputs where !name.isEmpty {
        producers[name] = k
      }
    }
    self.consumers = consumers
    self.producers = producers
  }

  /// A tensor's shape, if every dimension of it is known.
  func dims(_ name: String) -> [Int64]? {
    guard let d = shapes[name], !d.contains(where: { $0 < 0 }) else { return nil }
    return d
  }

  /// An initializer's dims, or nil if the name is not one.
  func initializerDims(_ name: String) -> [Int64]? {
    initializers[name]?.dims
  }

  /// An integer initializer's values, or nil if the name is not one.
  func integers(_ name: String) throws -> [Int64]? {
    guard let t = initializers[name], DataType.isInteger(t.elementType), let src else { return nil }
    return try Elements.integers(t, src)
  }

  /// For a Slice whose bounds are all constants, the indices it keeps along
  /// each axis it names; nil for anything else.
  func slice(_ n: Node) throws -> [Int: [Int64]]? {
    guard n.op == "Slice", n.inputs.count >= 3, let data = dims(n.inputs[0]),
      let starts = try integers(n.inputs[1]), let ends = try integers(n.inputs[2]), starts.count == ends.count
    else { return nil }
    let axes = n.inputs.count > 3 && !n.inputs[3].isEmpty ? try integers(n.inputs[3]) : starts.indices.map(Int64.init)
    let steps = n.inputs.count > 4 && !n.inputs[4].isEmpty ? try integers(n.inputs[4]) : Array(repeating: 1, count: starts.count)
    guard let axes, let steps, axes.count == starts.count, steps.count == starts.count else { return nil }
    var picked: [Int: [Int64]] = [:]
    for i in starts.indices {
      let axis = Patches.normalized(axes[i], data.count)
      guard (0..<data.count).contains(axis), picked[axis] == nil else { return nil }
      picked[axis] = Patches.sliceIndices(size: data[axis], start: starts[i], end: ends[i], step: steps[i])
    }
    return picked
  }

  /// A Squeeze's or Unsqueeze's axes, counted from the front of a rank-`rank`
  /// shape: the second input from opset 13, the attribute before; nil when
  /// it has none.
  func axes(_ n: Node, rank: Int, opset: Int64) throws -> [Int]? {
    let values: [Int64]?
    if n.inputs.count > 1, !n.inputs[1].isEmpty {
      values = try integers(n.inputs[1])
      guard values != nil else { return nil }
    } else if opset < 13, let attribute = n.attribute("axes") {
      values = attribute.ints
    } else {
      values = nil
    }
    return values.map { $0.map { Patches.normalized($0, rank) } }
  }

  /// A 1-D int64 constant, made once per value.
  mutating func ints(_ values: [Int64]) -> String {
    let name = "__litert_i64_" + values.map { $0 < 0 ? "m\(-$0)" : "\($0)" }.joined(separator: "_")
    if initializers[name] == nil, made.insert(name).inserted {
      constants.append(Patches.int64Tensor(values, name))
    }
    return name
  }

  /// A scalar of a float type, made once per value.
  mutating func scalar(_ value: Double, _ type: Int32) -> String {
    let bytes: [UInt8]
    switch type {
    case DataType.float16: bytes = Self.littleEndian(Float16(value).bitPattern)
    case DataType.double: bytes = Self.littleEndian(value.bitPattern)
    case DataType.bfloat16:
      // Round to nearest even, as numpy's ml_dtypes does.
      let bits = Float(value).bitPattern
      bytes = Self.littleEndian(UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16))
    default: bytes = Self.littleEndian(Float(value).bitPattern)
    }
    let name = "__litert_f\(type)_\(value)"
    if initializers[name] == nil, made.insert(name).inserted {
      constants.append(Tensor(name: name, dims: [], dataType: type, raw: .owned(bytes)))
    }
    return name
  }

  private static func littleEndian<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
    withUnsafeBytes(of: value.littleEndian) { Array($0) }
  }

  /// A new node named after its output, and the output's value info.
  mutating func node(_ op: String, _ inputs: [String], _ output: String, type: Int32, dims: [Int64], attributes: [Attribute] = []) -> Node {
    info(output, type, dims)
    return Node(inputs: inputs, outputs: [output], name: "\(output)__\(op.lowercased())", opType: op, attributes: attributes)
  }

  /// Records a tensor's value info, added unless the graph has one: a tensor
  /// a rewrite makes again under its old name keeps the one it had.
  mutating func info(_ name: String, _ type: Int32, _ dims: [Int64]) {
    shapes[name] = dims
    types[name] = type
    infos.append(.tensor(name, type, dims))
  }

  mutating func replace(_ k: Int, with nodes: [Node]) {
    replacements[k] = nodes
  }

  mutating func drop(_ k: Int) {
    dropped.insert(k)
  }

  /// `data` gathered along `axis` at `values`, as Slices of the runs of
  /// consecutive values, a Concat if there is more than one, and a Reshape to
  /// `shape` if that is not already the result's. A gather of the whole axis
  /// in order to the same shape is `data` itself: the readers read it, and
  /// nothing is left. nil if an index is out of range or the output's
  /// recorded shape is not `shape`.
  mutating func slices(
    _ data: String, _ dataDims: [Int64], axis: Int, _ values: [Int64], _ output: String, _ shape: [Int64], type: Int32
  ) -> [Node]? {
    let size = dataDims[axis]
    let fixed = values.map { $0 < 0 ? $0 + size : $0 }
    guard !fixed.isEmpty, fixed.allSatisfy({ (0..<size).contains($0) }), dims(output).map({ $0 == shape }) ?? true else { return nil }
    let runs = Patches.runs(fixed)
    if runs == [0..<size], shape == dataDims, !outputs.contains(output) {
      aliases[output] = data
      return []
    }
    var new: [Node] = []
    var parts: [String] = []
    for run in runs {
      var partDims = dataDims
      partDims[axis] = Int64(run.count)
      let name = runs.count == 1 ? "\(output)__slice" : "\(output)__\(run.lowerBound)"
      new.append(
        node("Slice", [data, ints([run.lowerBound]), ints([run.upperBound]), ints([Int64(axis)])], name, type: type, dims: partDims))
      parts.append(name)
    }
    var result = parts[0]
    var resultDims = dataDims
    resultDims[axis] = Int64(fixed.count)
    if parts.count > 1 {
      result = "\(output)__concat"
      new.append(node("Concat", parts, result, type: type, dims: resultDims, attributes: [.int("axis", Int64(axis))]))
    }
    if resultDims == shape {
      // The last node writes the Gather's output itself.
      new[new.count - 1].outputs = [output]
      new[new.count - 1].name = "\(output)__\(new[new.count - 1].op.lowercased())"
      infos.removeAll { $0.key == result }
      info(output, type, shape)
    } else {
      new.append(node("Reshape", [result, ints(shape)], output, type: type, dims: shape))
    }
    return new
  }

  /// Changes the graph: the replacements where the nodes they replace were,
  /// the dropped nodes gone, readers of an aliased tensor reading what it
  /// aliases, and the constants and value infos added.
  func apply(_ g: inout Graph) {
    var nodes: [Node] = []
    nodes.reserveCapacity(g.nodes.count)
    for (k, node) in g.nodes.enumerated() {
      if let new = replacements[k] {
        nodes.append(contentsOf: new)
      } else if !dropped.contains(k) {
        nodes.append(node)
      }
    }
    if !aliases.isEmpty {
      for k in nodes.indices {
        for i in nodes[k].inputs.indices {
          if let to = aliases[nodes[k].inputs[i]] { nodes[k].inputs[i] = to }
        }
      }
    }
    g.nodes = nodes
    g.initializers.append(contentsOf: constants)
    // The last word on each tensor, unless the graph already has one.
    let recorded = Set((g.inputs + g.outputs + g.valueInfo).map(\.key))
    var last: [String: Int] = [:]
    for (i, vi) in infos.enumerated() {
      last[vi.key] = i
    }
    for (i, vi) in infos.enumerated() where last[vi.key] == i && !recorded.contains(vi.key) {
      g.valueInfo.append(vi)
    }
  }
}

extension DataType {
  /// The floating-point types LayerNormalization takes.
  static func isFloat(_ type: Int32) -> Bool {
    [float, float16, double, bfloat16].contains(type)
  }
}
