import Foundation
import JetlinkONNX

// jetlink-onnx: the Swift ONNX preparation on its own, for checking it against
// the Python (JetlinkKit/Scripts/check_onnx_prep.py) and for looking at a model.
//
//   jetlink-onnx meta <file>
//   jetlink-onnx slices <file>
//   jetlink-onnx prepare <src> <outdir> [--whole] [--key-prefix P]

let usage = """
  usage:
    jetlink-onnx meta <file>                   inputs, outputs and metadata_props
    jetlink-onnx slices <file>                 openpilot's output_slices
    jetlink-onnx prepare <src> <outdir> [--whole] [--key-prefix P]
                                               prepare for CoreML: vision.onnx and policy.onnx,
                                               or model.onnx with --whole. Each part's
                                               COREML_CACHE_KEY is the server's _cache_key with
                                               P (default: the source's file name stem) as the stem.
  """

func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data((message + "\n").utf8))
  exit(1)
}

func shape(_ dims: [Int64]) -> String {
  // A Python tuple, as onnx_meta.py prints the shapes.
  dims.count == 1 ? "(\(dims[0]),)" : "(" + dims.map(String.init).joined(separator: ", ") + ")"
}

func padded(_ s: String, _ width: Int) -> String {
  s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
}

func meta(_ path: String) throws {
  let m = try OnnxMeta.read(contentsOf: URL(fileURLWithPath: path))
  print("inputs:")
  for t in m.inputs {
    print("  \(padded(t.name, 20)) \(padded(shape(t.dims), 24)) \(t.typeName)")
  }
  print("outputs:")
  for t in m.outputs {
    print("  \(padded(t.name, 20)) \(padded(shape(t.dims), 24)) \(t.typeName)")
  }
  print("metadata_props:")
  for p in m.props {
    let value = p.value.count > 72 ? String(p.value.prefix(72)) + "... (\(p.value.count) chars)" : p.value
    print("  \(p.key): \(value.replacingOccurrences(of: "\n", with: "\\n"))")
  }
}

func slices(_ path: String) throws {
  let m = try OnnxMeta.read(contentsOf: URL(fileURLWithPath: path))
  for s in try m.outputSlices() {
    print("  slice \(padded(s.name, 24)) slice(\(s.start), \(s.stop), None)")
  }
}

func prepare(_ args: [String]) throws {
  var positional: [String] = []
  var whole = false
  var prefix: String?
  var i = 0
  while i < args.count {
    switch args[i] {
    case "--whole":
      whole = true
    case "--key-prefix":
      guard i + 1 < args.count else { fail("--key-prefix needs a value") }
      prefix = args[i + 1]
      i += 1
    default:
      if args[i].hasPrefix("--") { fail("unknown option \(args[i])\n\(usage)") }
      positional.append(args[i])
    }
    i += 1
  }
  guard positional.count == 2 else { fail(usage) }
  let source = URL(fileURLWithPath: positional[0])
  let out = URL(fileURLWithPath: positional[1])
  let stem = prefix ?? source.deletingPathExtension().lastPathComponent

  let clock = ContinuousClock()
  let start = clock.now
  let report = try CoreMLPreparation.prepare(
    source: source, into: out, layout: whole ? .whole : .split,
    cacheKey: { CoreMLPreparation.cacheKey(stem: stem, part: $0) })
  let elapsed = clock.now - start

  print(
    "stripped \(report.stripped) tinygrad op(s), "
      + (report.retypedImages ? "images retyped to fp16" : "inputs left as declared")
      + ", \(report.gathers) negative Gather index(es) normalized, \(report.gemms) MatMul+Add rewritten as "
      + "Gemm(transB=1), \(report.tiles) Expand(s) as Tile")
  for part in report.parts {
    let size = (try? FileManager.default.attributesOfItem(atPath: part.url.path)[.size] as? Int64) ?? 0
    print(
      "  \(padded(part.name, 7)) \(part.url.path)  \(size) bytes, weights \(part.weightBytes) bytes, "
        + "key \(CoreMLPreparation.cacheKey(stem: stem, part: part.name))")
  }
  let ms = Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
  print(String(format: "prepared in %.0f ms", ms))
}

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { fail(usage) }
do {
  switch command {
  case "meta":
    guard args.count == 2 else { fail(usage) }
    try meta(args[1])
  case "slices":
    guard args.count == 2 else { fail(usage) }
    try slices(args[1])
  case "prepare":
    try prepare(Array(args.dropFirst()))
  default:
    fail(usage)
  }
} catch {
  fail("error: \(error)")
}
