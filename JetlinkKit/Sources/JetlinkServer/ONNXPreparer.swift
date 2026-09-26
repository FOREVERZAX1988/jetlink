import Foundation
import JetlinkONNX

/// The server's reading and preparing of an ONNX, through JetlinkONNX: the
/// spec from the graph's inputs, outputs and output_slices, and the CoreML
/// preparation the Python backend does, streamed from a mapped file.
public struct ONNXPreparer: ModelPreparer {
  public init() {}

  public func readSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    let meta = try OnnxMeta.read(contentsOf: model)
    return ModelSpec(
      sha256: sha256,
      nbytes: nbytes,
      frameSkip: frameSkip,
      inputShapes: meta.inputs.map { NamedShape($0.name, $0.dims.map { Int($0) }) },
      outputShapes: meta.outputs.map { NamedShape($0.name, $0.dims.map { Int($0) }) },
      outputSlices: try meta.outputSlices().map { NamedRange($0.name, $0.range) },
      checkpoint: meta.modelCheckpoint)
  }

  public func prepare(model: URL, into directory: URL, split: Bool, cacheKey: @escaping (String) -> String) throws -> PreparedModel {
    let report = try CoreMLPreparation.prepare(source: model, into: directory, layout: split ? .split : .whole, cacheKey: cacheKey)
    let summary =
      "stripped \(report.stripped) tinygrad op(s), \(report.retypedImages ? "images retyped to fp16" : "inputs left as declared"), "
      + "\(report.gathers) negative Gather index(es) normalized, \(report.gemms) MatMul+Add rewritten as Gemm(transB=1), \(report.tiles) Expand(s) as Tile"
    return PreparedModel(
      parts: report.parts.map { PreparedModel.Part(name: $0.name, file: $0.url.lastPathComponent, weightBytes: $0.weightBytes) },
      summary: summary)
  }
}
