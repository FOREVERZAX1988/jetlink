import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkONNX

/// The LiteRT rewrites on graphs made for each one: the graph they make, its
/// value infos, the same values as the graph they replace (run by
/// GraphEvaluator), and nothing done where the pattern is absent.
@Suite struct LiteRTPatchesTests {
  static let u8 = DataType.uint8
  static let f32 = DataType.float
  static let f16 = DataType.float16

  /// Inputs counting up from 1, so a layout rewrite that moves one element
  /// to the wrong place changes the outputs.
  static func counting(_ g: Graph) -> [String: Values] {
    var inputs: [String: Values] = [:]
    for (k, vi) in g.inputs.enumerated() {
      let dims = vi.shape!.dims.map { Int($0.value!) }
      let count = dims.reduce(1, *)
      inputs[vi.key] = Values(dims: dims, data: (0..<count).map { Double(k * 100_000 + $0 + 1) })
    }
    return inputs
  }

  /// Runs both graphs on the same inputs, viewed as each graph declares
  /// them, and returns each one's graph outputs in order.
  static func outputs(_ before: Graph, _ after: Graph, _ inputs: [String: Values], opset: Int64 = 20) throws -> ([Values], [Values]) {
    var viewed = inputs
    for vi in after.inputs {
      viewed[vi.key]!.dims = vi.shape!.dims.map { Int($0.value!) }
    }
    let a = try GraphEvaluator(g: before, opset: opset).run(inputs)
    let b = try GraphEvaluator(g: after, opset: opset).run(viewed)
    return (before.outputs.map { a[$0.key]! }, after.outputs.map { b[$0.key]! })
  }

  /// Every tensor a node writes has a value info with a static shape and an
  /// element type, and a rank-5 tensor is nowhere.
  static func expectStaticAndFlat(_ g: Graph, sourceLocation: SourceLocation = #_sourceLocation) {
    let infos = Dictionary((g.inputs + g.valueInfo + g.outputs).map { ($0.key, $0) }, uniquingKeysWith: { $1 })
    for n in g.nodes {
      for out in n.outputs {
        let vi = infos[out]
        #expect(vi != nil && vi!.elemType != 0, "\(out) has no value info", sourceLocation: sourceLocation)
        let dims = vi?.shape?.dims.map { $0.value ?? -1 } ?? []
        #expect(!dims.contains { $0 <= 0 } && vi?.shape != nil, "\(out) is not static: \(dims)", sourceLocation: sourceLocation)
        #expect(dims.count < 5, "\(out) is rank 5", sourceLocation: sourceLocation)
      }
    }
  }

  static func ops(_ g: Graph) -> [String] { g.nodes.map(\.op) }

  // MARK: GatherND indices

  @Test func gatherNDIndicesCountFromTheFront() throws {
    var b = GraphBuilder()
    b.input("x", Self.f32, [33, 1, 4])
    b.ints("rows", Array(-9..<0), dims: [9, 1])
    b.node("GatherND", ["x", "rows"], "y", Self.f32, [9, 1, 4])
    b.g.outputs = [.tensor("y", Self.f32, [9, 1, 4])]
    var g = b.g
    #expect(try Patches.normalizeGatherNDIndices(&g, emptySource) == 1)
    let index = try #require(g.initializers.first { $0.key == "y__index" })
    #expect(try Elements.integers(index, emptySource) == Array(24..<33) && index.dims == [9, 1])
    #expect(g.nodes[0].inputs == ["x", "y__index"])
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want == got)

    // Already positive: nothing to do.
    #expect(try Patches.normalizeGatherNDIndices(&g, emptySource) == 0)
  }

  // MARK: attention

  /// Cinque Terre's form: Reshape to [B,T,3,H,D], Transpose [2,0,3,1,4],
  /// Split into three, Squeeze each.
  static func transposedAttention(opset: Int64 = 20) -> GraphBuilder {
    var b = GraphBuilder()
    b.opsets = [OpsetImport(raw: 0..<0, domain: "", version: opset)]
    b.input("x", f32, [1, 6, 24])
    b.ints("shape", [1, 6, 3, 2, 4])
    b.ints("zero", [0])
    b.node("Reshape", ["x", "shape"], "v", f32, [1, 6, 3, 2, 4])
    b.node("Transpose", ["v"], "p", f32, [3, 1, 2, 6, 4], [.ints("perm", [2, 0, 3, 1, 4])])
    b.nodes("Split", ["p"], ["s0", "s1", "s2"], [.int("axis", 0), .int("num_outputs", 3)])
    for k in 0..<3 {
      b.g.valueInfo.append(.tensor("s\(k)", f32, [1, 1, 2, 6, 4]))
      b.node("Squeeze", ["s\(k)", "zero"], "q\(k)", f32, [1, 2, 6, 4])
      b.output("q\(k)", f32, [1, 2, 6, 4])
    }
    return b
  }

  @Test func transposedAttentionIsFourD() throws {
    let b = Self.transposedAttention()
    var g = b.g
    #expect(try Patches.attentionIn4D(&g, emptySource, opset: 20) == 1)
    #expect(Self.ops(g) == Array(repeating: ["Slice", "Reshape", "Transpose"], count: 3).flatMap { $0 })
    #expect(g.nodes[2].attribute("perm")?.ints == [0, 2, 1, 3])
    #expect(g.nodes.map(\.outputs[0]).filter { !$0.contains("__") } == ["q0", "q1", "q2"])
    Patches.dropStaleValueInfo(&g)
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want == got)
  }

  /// The older models' unbind: Slices of the chunk axis, no Transpose.
  @Test func unboundAttentionIsFourD() throws {
    var b = GraphBuilder()
    b.input("x", Self.f32, [1, 9, 24])
    b.ints("shape", [1, 9, 3, 2, 4])
    b.ints("two", [2])
    b.node("Reshape", ["x", "shape"], "v", Self.f32, [1, 9, 3, 2, 4])
    for k in 0..<3 {
      b.ints("s\(k)", [Int64(k)])
      b.ints("e\(k)", [Int64(k + 1)])
      b.node("Slice", ["v", "s\(k)", "e\(k)", "two"], "c\(k)", Self.f32, [1, 9, 1, 2, 4])
      b.node("Squeeze", ["c\(k)", "two"], "q\(k)", Self.f32, [1, 9, 2, 4])
      b.output("q\(k)", Self.f32, [1, 9, 2, 4])
    }
    var g = b.g
    #expect(try Patches.attentionIn4D(&g, emptySource, opset: 20) == 1)
    #expect(Self.ops(g) == Array(repeating: ["Slice", "Reshape"], count: 3).flatMap { $0 })
    Patches.dropStaleValueInfo(&g)
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want == got)
  }

  /// A Gather of one chunk drops the axis itself.
  @Test func gatheredChunkIsFourD() throws {
    var b = GraphBuilder()
    b.input("x", Self.f32, [1, 5, 12])
    b.ints("shape", [1, 5, 3, 2, 2])
    b.ints("last", [-1], dims: [])
    b.node("Reshape", ["x", "shape"], "v", Self.f32, [1, 5, 3, 2, 2])
    b.node("Gather", ["v", "last"], "q", Self.f32, [1, 5, 2, 2], [.int("axis", 2)])
    b.output("q", Self.f32, [1, 5, 2, 2])
    var g = b.g
    #expect(try Patches.attentionIn4D(&g, emptySource, opset: 20) == 1)
    #expect(Self.ops(g) == ["Slice", "Reshape"])
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want == got)
  }

  /// Anything else reading the 5-D tensor leaves the block as it is.
  @Test func attentionWithAnotherReaderIsLeft() throws {
    var b = Self.transposedAttention()
    b.ints("axes", [4])
    b.node("ReduceMean", ["v", "axes"], "m", Self.f32, [1, 6, 3, 2, 1])
    var g = b.g
    #expect(try Patches.attentionIn4D(&g, emptySource, opset: 20) == 0)
    #expect(Self.ops(g) == Self.ops(b.g))
  }

  // MARK: the frame queue

  /// stateful.onnx's queue, Cinque Terre's in small: two cameras of five
  /// 6-channel frames, shifted by one, and frames 0 and 4 of each read.
  static func frameQueue() -> GraphBuilder {
    var b = GraphBuilder()
    b.input("new_img", u8, [2, 6, 2, 3])
    b.input("state_img_q", u8, [2, 5, 6, 2, 3])
    b.output("next_state_img_q", u8, [2, 5, 6, 2, 3])
    b.output("imgs", f16, [1, 24, 2, 3])
    b.ints("one", [1])
    b.ints("zero", [0])
    b.ints("end", [Int64.max])
    b.ints("four", [4])
    b.ints("i0", [0], dims: [])
    b.ints("i1", [-1], dims: [])
    b.ints("cam", [1, 12, 2, 3])
    b.node("Slice", ["state_img_q", "one", "end", "one", "one"], "tail", u8, [2, 4, 6, 2, 3])
    b.node("Unsqueeze", ["new_img", "one"], "frame", u8, [2, 1, 6, 2, 3])
    b.node("Concat", ["tail", "frame"], "next_state_img_q", u8, [2, 5, 6, 2, 3], [.int("axis", 1)])
    for (k, index) in ["i0", "i1"].enumerated() {
      b.node("Gather", ["next_state_img_q", index], "cam\(k)", u8, [5, 6, 2, 3], [.int("axis", 0)])
      b.node("Slice", ["cam\(k)", "zero", "end", "zero", "four"], "pair\(k)", u8, [2, 6, 2, 3])
      b.node("Reshape", ["pair\(k)", "cam"], "flat\(k)", u8, [1, 12, 2, 3])
    }
    b.node("Concat", ["flat0", "flat1"], "both", u8, [1, 24, 2, 3], [.int("axis", 1)])
    b.node("Cast", ["both"], "imgs", f16, [1, 24, 2, 3], [.int("to", Int64(f16))])
    return b
  }

  @Test func frameQueueIsItsFourDView() throws {
    let b = Self.frameQueue()
    var g = b.g
    #expect(try Patches.frameQueuesIn4D(&g, emptySource, opset: 20) == 1)
    #expect(g.inputs[1].shape?.dims.map(\.value) == [2, 30, 2, 3])
    #expect(g.outputs[0].shape?.dims.map(\.value) == [2, 30, 2, 3])
    #expect(g.inputs[1].elemType == Self.u8 && g.outputs[0].elemType == Self.u8)
    // The shift, then per camera frames 0 and 4 as two channel Slices and a Concat.
    #expect(Self.ops(g) == ["Slice", "Concat", "Slice", "Slice", "Concat", "Slice", "Slice", "Concat", "Concat", "Cast"])
    #expect(g.nodes[1].inputs == ["tail__4d", "new_img"])
    #expect(g.nodes[4].outputs == ["flat0"] && g.nodes[7].outputs == ["flat1"])
    Patches.dropStaleValueInfo(&g)
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want.map(\.data) == got.map(\.data))
  }

  /// A camera read some other way is a Slice of the view reshaped to the frames.
  @Test func plainCameraReadIsASlice() throws {
    var b = Self.frameQueue()
    b.g.nodes.removeAll { $0.inputs.first == "cam1" || $0.outputs.first == "flat1" }
    b.g.nodes.removeAll { ["both", "imgs"].contains($0.outputs.first) }
    b.ints("axes", [1])
    b.node("ReduceMean", ["cam1", "axes"], "mean1", Self.u8, [5, 1, 2, 3])
    b.g.outputs = [b.g.outputs[0], .tensor("flat0", Self.u8, [1, 12, 2, 3]), .tensor("mean1", Self.u8, [5, 1, 2, 3])]
    var g = b.g
    #expect(try Patches.frameQueuesIn4D(&g, emptySource, opset: 20) == 1)
    #expect(Self.ops(g) == ["Slice", "Concat", "Slice", "Slice", "Concat", "Slice", "Reshape", "ReduceMean"])
    Patches.dropStaleValueInfo(&g)
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want.map(\.data) == got.map(\.data))
  }

  /// A queue that something else reads stays rank 5.
  @Test func frameQueueWithAnotherReaderIsLeft() throws {
    var b = Self.frameQueue()
    b.node("Identity", ["next_state_img_q"], "copy", Self.u8, [2, 5, 6, 2, 3])
    var g = b.g
    #expect(try Patches.frameQueuesIn4D(&g, emptySource, opset: 20) == 0)
    #expect(Self.ops(g) == Self.ops(b.g))
    #expect(g.inputs[1].shape?.dims.count == 5)
  }

  // MARK: gathers as Slices

  @Test func slidingWindowsAreSlicesAndAConcat() throws {
    var b = GraphBuilder()
    b.input("desire", Self.f32, [1, 6, 2])
    b.ints("windows", [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5])
    b.node("Gather", ["desire", "windows"], "picked", Self.f32, [1, 12, 2], [.int("axis", 1)])
    b.output("picked", Self.f32, [1, 12, 2])
    var g = b.g
    #expect(try Patches.gathersAsSlices(&g, emptySource) == 1)
    #expect(Self.ops(g) == ["Slice", "Slice", "Slice", "Concat"])
    #expect(g.nodes.last?.outputs == ["picked"])
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want == got)
  }

  @Test(arguments: [
    ([-1], [Int64](), ["Slice", "Reshape"]),
    ([2, 0, 1], [3], ["Slice", "Slice", "Concat"]),
    ([0, 1, 3, 2], [2, 2], ["Slice", "Slice", "Slice", "Concat", "Reshape"]),
  ])
  func constantGathersAreSlices(_ indices: [Int64], _ dims: [Int64], _ expected: [String]) throws {
    var b = GraphBuilder()
    b.input("x", Self.f32, [2, 4, 3])
    b.ints("i", indices, dims: dims)
    let shape: [Int64] = [2] + dims + [3]
    b.node("Gather", ["x", "i"], "y", Self.f32, shape, [.int("axis", 1)])
    b.output("y", Self.f32, shape)
    var g = b.g
    #expect(try Patches.gathersAsSlices(&g, emptySource) == 1)
    #expect(Self.ops(g) == expected)
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want == got)
  }

  /// A gather of a whole initializer in order, as the position embeddings
  /// are read, is the initializer.
  @Test func wholeInitializerGatherIsTheInitializer() throws {
    var b = GraphBuilder()
    b.floats("table", (0..<8).map(Float.init), dims: [4, 2])
    b.ints("all", [0, 1, 2, 3])
    b.input("x", Self.f32, [4, 2])
    b.node("Gather", ["table", "all"], "rows", Self.f32, [4, 2], [.int("axis", 0)])
    b.node("Add", ["x", "rows"], "y", Self.f32, [4, 2])
    b.output("y", Self.f32, [4, 2])
    var g = b.g
    #expect(try Patches.gathersAsSlices(&g, emptySource) == 1)
    #expect(Self.ops(g) == ["Add"])
    #expect(g.nodes[0].inputs == ["x", "table"])
  }

  @Test func gatherNDRowsAreASlice() throws {
    var b = GraphBuilder()
    b.input("x", Self.f32, [33, 1, 4])
    b.ints("rows", Array(24..<33), dims: [9, 1])
    b.node("GatherND", ["x", "rows"], "y", Self.f32, [9, 1, 4])
    b.output("y", Self.f32, [9, 1, 4])
    var g = b.g
    #expect(try Patches.gatherNDsAsSlices(&g, emptySource) == 1)
    #expect(Self.ops(g) == ["Slice"])
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want == got)
  }

  /// Gathers whose indices are not constant, or a GatherND of index pairs,
  /// stay as they are.
  @Test func gathersThatAreNotConstantAreLeft() throws {
    var b = GraphBuilder()
    b.input("x", Self.f32, [4, 4])
    b.input("i", DataType.int64, [2])
    b.ints("pairs", [0, 1, 2, 3], dims: [2, 2])
    b.node("Gather", ["x", "i"], "y", Self.f32, [2, 4])
    b.node("GatherND", ["x", "pairs"], "z", Self.f32, [2])
    var g = b.g
    #expect(try Patches.gathersAsSlices(&g, emptySource) == 0)
    #expect(try Patches.gatherNDsAsSlices(&g, emptySource) == 0)
    #expect(Self.ops(g) == ["Gather", "GatherND"])
  }

  // MARK: LayerNormalization

  static func layerNorm(type: Int32, axis: Int64 = -1, stash: Int64 = 1, bias: Bool = true) -> GraphBuilder {
    var b = GraphBuilder()
    b.input("x", type, [2, 3, 8])
    let inner: [Int64] = axis == -1 ? [8] : [3, 8]
    let count = Int(inner.reduce(1, *))
    b.floats("w", (0..<count).map { 0.5 + Float($0) / 16 }, dims: inner, type: type)
    var inputs = ["x", "w"]
    if bias {
      b.floats("b", (0..<count).map { Float($0) / 8 - 1 }, dims: inner, type: type)
      inputs.append("b")
    }
    b.node(
      "LayerNormalization", inputs, "y", type, [2, 3, 8],
      [.int("axis", axis), .int("stash_type", stash)])
    b.output("y", type, [2, 3, 8])
    return b
  }

  /// Inputs whose rows spread widely, as the norms' inputs do in Cinque Terre.
  static func spread(_ g: Graph) -> [String: Values] {
    let data = (0..<48).map { (i: Int) -> Double in
      let step = Double((i * 37) % 48) * 300
      return step - 7000 + (i % 3 == 0 ? 2500 : 0)
    }
    return ["x": Values(dims: [2, 3, 8], data: data)]
  }

  @Test(arguments: [(Int64(20), Int64(-1)), (20, 1), (17, -1)])
  func layerNormIsScaled(_ opset: Int64, _ axis: Int64) throws {
    let b = Self.layerNorm(type: Self.f32, axis: axis)
    var g = b.g
    #expect(try Patches.scaledLayerNorms(&g, opset: opset) == 1)
    #expect(!Self.ops(g).contains("LayerNormalization") && !Self.ops(g).contains("Cast"))
    #expect(Self.ops(g).filter { $0 == "ReduceMean" }.count == 2 && Self.ops(g).filter { $0 == "Reciprocal" }.count == 2)
    let reduce = try #require(g.nodes.first { $0.op == "ReduceMean" })
    let axes: [Int64] = axis == -1 ? [2] : [1, 2]
    if opset >= 18 {
      #expect(reduce.inputs.count == 2 && reduce.attribute("axes") == nil)
      #expect(try Elements.integers(g.initializers.first { $0.key == reduce.inputs[1] }!, emptySource) == axes)
    } else {
      #expect(reduce.inputs.count == 1 && reduce.attribute("axes")?.ints == axes)
    }
    #expect(g.nodes.last?.outputs == ["y"] && g.nodes.last?.op == "Add")
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.spread(b.g), opset: opset)
    for (w, o) in zip(want[0].data, got[0].data) {
      #expect(abs(w - o) < 1e-9, "\(w) against \(o)")
    }
  }

  /// fp16 with the statistics stashed in fp32: cast up, normalized, cast back,
  /// the constants fp32.
  @Test func fp16LayerNormStashesInFP32() throws {
    let b = Self.layerNorm(type: Self.f16, bias: false)
    var g = b.g
    #expect(try Patches.scaledLayerNorms(&g, opset: 20) == 1)
    let casts = g.nodes.filter { $0.op == "Cast" }
    #expect(casts.map { $0.attribute("to")?.i } == [Int64(Self.f32), Int64(Self.f16)])
    #expect(g.nodes.first?.op == "Cast" && g.nodes.last?.op == "Mul" && g.nodes.last?.outputs == ["y"])
    let floor = try #require(g.initializers.first { $0.key.hasPrefix("__litert_f1_0.01") })
    #expect(floor.dims == [] && floor.elementType == Self.f32)
    let infos = Dictionary(g.valueInfo.map { ($0.key, $0.elemType) }, uniquingKeysWith: { $1 })
    #expect(infos["y__ln_d"] == Self.f32 && infos["y__ln_y"] == Self.f16)
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.spread(b.g))
    for (w, o) in zip(want[0].data, got[0].data) {
      #expect(abs(w - o) < 1e-9)
    }
  }

  /// stash_type 10 keeps the arithmetic in fp16, as prep_gpu.py did.
  @Test func fp16LayerNormWithoutAStash() throws {
    var g = Self.layerNorm(type: Self.f16, stash: Int64(Self.f16)).g
    #expect(try Patches.scaledLayerNorms(&g, opset: 20) == 1)
    #expect(!Self.ops(g).contains("Cast"))
    #expect(g.initializers.contains { $0.key.hasPrefix("__litert_f10_") && $0.elementType == Self.f16 })
  }

  /// A norm whose mean output something reads is left.
  @Test func layerNormWithItsMeanReadIsLeft() throws {
    var b = Self.layerNorm(type: Self.f32)
    b.g.nodes[0].outputs = ["y", "mean"]
    b.node("Identity", ["mean"], "m", Self.f32, [2, 3, 1])
    var g = b.g
    #expect(try Patches.scaledLayerNorms(&g, opset: 20) == 0)
    #expect(Self.ops(g) == ["LayerNormalization", "Identity"])
  }

  // MARK: Squeeze and Unsqueeze

  @Test func squeezesAreStaticReshapes() throws {
    var b = GraphBuilder()
    b.input("x", Self.f32, [1, 3, 1, 4])
    b.ints("axes", [0])
    b.ints("ax", [-1])
    b.node("Squeeze", ["x", "axes"], "a", Self.f32, [3, 1, 4])
    b.nodes("Unsqueeze", ["a", "ax"], ["c"])  // no value info: worked out
    b.node("Squeeze", ["c"], "d", Self.f32, [3, 4])
    b.output("d", Self.f32, [3, 4])
    var g = b.g
    #expect(try Patches.squeezesAsReshapes(&g, emptySource, opset: 20) == 3)
    #expect(Self.ops(g) == ["Reshape", "Reshape", "Reshape"])
    let shapes = try g.nodes.map { n in try Elements.integers(g.initializers.first { $0.key == n.inputs[1] }!, emptySource) }
    #expect(shapes == [[3, 1, 4], [3, 1, 4, 1], [3, 4]])
    Self.expectStaticAndFlat(g)
    let (want, got) = try Self.outputs(b.g, g, Self.counting(b.g))
    #expect(want == got)
  }

  /// Before opset 13 the axes are an attribute.
  @Test func squeezeAxesFromTheAttribute() throws {
    var b = GraphBuilder()
    b.input("x", Self.f32, [2, 3])
    b.nodes("Unsqueeze", ["x"], ["y"], [.ints("axes", [0, 3])])
    var g = b.g
    #expect(try Patches.squeezesAsReshapes(&g, emptySource, opset: 11) == 1)
    #expect(g.nodes[0].attributes.isEmpty && g.nodes[0].op == "Reshape")
    #expect(g.valueInfo.first { $0.key == "y" }?.shape?.dims.map(\.value) == [1, 2, 3, 1])
  }

  // MARK: the whole pass

  @Test func everyRewriteOnTheFrameQueue() throws {
    let b = Self.frameQueue()
    var g = b.g
    var opsets = b.opsets
    let r = try Patches.forLiteRT(&g, &opsets, emptySource)
    // The wide camera's Gather index -1 is written from the front first.
    #expect(r.frameQueues == 1 && r.gathers == 1 && r.gatherSlices == 0 && r.reshapes == 0)
    Self.expectStaticAndFlat(g)
    // Run again, it finds nothing left to do.
    var again = g
    #expect(try Patches.forLiteRT(&again, &opsets, emptySource) == LiteRTRewrites())
  }

  @Test func sliceIndicesFollowOnnx() {
    #expect(Patches.sliceIndices(size: 5, start: 1, end: .max, step: 1) == [1, 2, 3, 4])
    #expect(Patches.sliceIndices(size: 5, start: 0, end: .max, step: 4) == [0, 4])
    #expect(Patches.sliceIndices(size: 5, start: -2, end: 10, step: 1) == [3, 4])
    #expect(Patches.sliceIndices(size: 5, start: -1, end: .min, step: -2) == [4, 2, 0])
    #expect(Patches.sliceIndices(size: 5, start: 3, end: 3, step: 1) == [])
    #expect(Patches.runs([0, 1, 2, 4, 3, 4]) == [0..<3, 4..<5, 3..<5])
  }
}
