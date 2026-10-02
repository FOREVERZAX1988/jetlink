import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkONNX

/// The `.liteRT` layout on the test models, and on a real one on request.
@Suite struct LiteRTPreparationTests {
  static let graphs = ["stateful", "queued", "variants", "nocut", "noentry", "noshape", "notype", "unrecorded"]
  /// What stays because the export records no shape or type for its input,
  /// and Swift has no shape inferrer: a rewrite skips it, as the CoreML
  /// patches skip what they cannot size.
  static let leftovers: [String: Set<String>] = [
    "unrecorded.onnx": ["Gather"], "notype.onnx": ["LayerNormalization"], "noentry.onnx": ["LayerNormalization"],
  ]
  static let fixtures: [URL] = graphs.map { Fixtures.url("\($0).onnx") } + [TinyModel.stateful, TinyModel.queued]

  /// The test models prepare for LiteRT: no rank-5 tensor, Gather, GatherND,
  /// Squeeze or Unsqueeze left, every new tensor with a static value info, the
  /// uint8 inputs uint8, and the frame queue 4-D with as many bytes.
  @Test(arguments: fixtures)
  func fixturePrepares(_ url: URL) throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(source: url, into: out.url, layout: .liteRT, cacheKey: { _ in "unused" })
    let r = try #require(report.liteRT)
    #expect(report.parts.map(\.name) == ["model"] && !report.retypedImages)
    let source = try OnnxMeta.read(contentsOf: url)
    let meta = try OnnxMeta.read(contentsOf: report.parts[0].url)
    #expect(meta.props == source.props)
    #expect(meta.inputs.map(\.name) == source.inputs.map(\.name) && meta.outputs.map(\.name) == source.outputs.map(\.name))
    for (a, b) in zip(meta.inputs + meta.outputs, source.inputs + source.outputs) {
      #expect(a.elemType == b.elemType && a.dims.reduce(1, *) == b.dims.reduce(1, *), "\(a.name)")
    }
    let stateful = source.inputs.contains { $0.name == "state_img_q" }
    #expect(r.frameQueues == (stateful ? 1 : 0))
    if stateful {
      #expect(meta.inputs.first { $0.name == "state_img_q" }?.dims == [2, 30, 8, 16])
    }

    let data = try Data(contentsOf: report.parts[0].url)
    let before = try Data(contentsOf: url).withUnsafeBytes { try Decode.model(Source(bytes: $0)).graph! }
    let g = try data.withUnsafeBytes { try Decode.model(Source(bytes: $0)).graph! }
    let ops = Set(g.nodes.map(\.op))
    let left = ops.intersection(["Gather", "GatherND", "Squeeze", "Unsqueeze", "LayerNormalization"])
    #expect(left == Self.leftovers[url.lastPathComponent] ?? [], "\(ops)")
    #expect(!ops.contains("Contiguous"))
    // What the export recorded, and every tensor a rewrite made, has a static shape.
    let unrecorded = Set(before.nodes.flatMap(\.outputs)).subtracting((before.inputs + before.valueInfo + before.outputs).map(\.key))
    let infos = Dictionary((g.inputs + g.valueInfo + g.outputs).map { ($0.key, $0) }, uniquingKeysWith: { $1 })
    for name in g.nodes.flatMap(\.outputs) where !unrecorded.contains(name) {
      let dims = infos[name]?.shape?.dims.map { $0.value ?? -1 }
      #expect(dims != nil && !dims!.contains { $0 <= 0 } && dims!.count < 5, "\(name): \(String(describing: dims))")
    }
  }

  /// A real model prepared on request, to check against the original with
  /// onnxruntime or to measure: JETLINK_PREPARE=<layout>:<model.onnx>:<directory>,
  /// the layout liteRT, whole, ane-whole, split or plain.
  @Test(.enabled(if: ProcessInfo.processInfo.environment["JETLINK_PREPARE"] != nil))
  func preparesARealModel() throws {
    let spec = ProcessInfo.processInfo.environment["JETLINK_PREPARE"]!.split(separator: ":", maxSplits: 2).map(String.init)
    let layouts: [String: CoreMLPreparation.Layout] = [
      "liteRT": .liteRT, "whole": .whole, "ane-whole": .aneWhole, "split": .split, "plain": .plain,
    ]
    let layout = try #require(layouts[spec[0]])
    let start = Date()
    let report = try CoreMLPreparation.prepare(
      source: URL(fileURLWithPath: spec[1]), into: URL(fileURLWithPath: spec[2]), layout: layout, cacheKey: { "jetlink\($0)" })
    print("prepared in \(Date().timeIntervalSince(start)) s: \(report.liteRT.map { "\($0)" } ?? "\(report)")")
  }
}
