import Foundation

/// A driving model prepared for onnxruntime's CoreML provider, as the
/// `simplify` branch's ORT backend prepares it (`_prepared_model(for_coreml=
/// True)`, `_stage` and `_with_cache_key` in jetlink/server/backends/ort):
///
/// 1. strip tinygrad's `org.tinygrad` layout ops;
/// 2. retype uint8 image inputs to fp16 and drop the head Cast, if any input is uint8;
/// 3. write negative constant Gather indices from the front;
/// 4. rewrite MatMul+Add as Gemm with a transposed weight (transB=1);
/// 5. rewrite repeat-only Expands as Tiles;
/// 6. cut the graph where the vision trunk ends (`.split`, writing
///    vision.onnx and policy.onnx), or keep it whole (`.whole`, model.onnx);
/// 7. give each file a COREML_CACHE_KEY in its metadata_props.
///
/// The files are what Python's onnx.save writes for the same model: field
/// for field, and byte for byte on a model Python wrote.
///
/// Memory is the point, because an iPhone runs this. The source is
/// memory-mapped and only the graph's structure is decoded; each weight stays
/// a range of the mapped file until it is copied to the output. The
/// transposed Gemm weights, about 671 MB on the big models, are made a few
/// MB at a time while they are written. Every message's length is known
/// before it is written, so the output streams to disk in one pass.
public enum CoreMLPreparation {
  public enum Layout: Sendable {
    /// The vision trunk and the rest as two models: vision.onnx and policy.onnx.
    case split
    /// One model: model.onnx.
    case whole
  }

  public struct Part: Sendable, Equatable {
    /// "vision", "policy" or "model"; the file is `<name>.onnx`.
    public let name: String
    public let url: URL
    /// The initializers' raw_data bytes in this part, which is what
    /// onnxruntime writes out as the CoreML weight file. Python sums
    /// `len(t.raw_data)` the same way, so a tensor kept in the typed fields
    /// counts as 0.
    public let weightBytes: Int64
  }

  public struct Report: Sendable, Equatable {
    /// tinygrad layout ops bypassed.
    public let stripped: Int
    /// Whether uint8 image inputs were retyped to fp16.
    public let retypedImages: Bool
    /// Gathers whose negative index was rewritten.
    public let gathers: Int
    /// MatMul+Add pairs rewritten as Gemm.
    public let gemms: Int
    /// Expands rewritten as Tile.
    public let tiles: Int
    public let parts: [Part]
  }

  /// The metadata_props key onnxruntime's CoreML provider keys its
  /// compiled-model cache by.
  public static let cacheKeyProp = "COREML_CACHE_KEY"

  /// The Python server's `_cache_key(out_path, part)` with `stem` for the
  /// path's stem: ASCII letters and digits only, at most 63 characters.
  public static func cacheKey(stem: String, part: String) -> String {
    let kept = (stem + part).unicodeScalars.filter {
      ("A"..."Z").contains($0) || ("a"..."z").contains($0) || ("0"..."9").contains($0)
    }
    return String(String.UnicodeScalarView(kept.prefix(63)))
  }

  public static func prepare(
    source: URL, into directory: URL, layout: Layout,
    cacheKey: (String) -> String,
    progress: ((Double) -> Void)? = nil
  ) throws -> Report {
    let data = try Data(contentsOf: source, options: .alwaysMapped)
    return try data.withUnsafeBytes { buf in
      try prepare(Source(bytes: buf), into: directory, layout: layout, cacheKey: cacheKey, progress: progress)
    }
  }

  private static func prepare(
    _ src: Source, into directory: URL, layout: Layout,
    cacheKey: (String) -> String,
    progress: ((Double) -> Void)?
  ) throws -> Report {
    var model = try Decode.model(src)
    guard var g = model.graph else { throw OnnxError("the model has no graph") }
    if let t = g.initializers.first(where: \.isExternal) {
      throw OnnxError("initializer \(t.key) keeps its data in an external file, which the preparation does not read")
    }

    let stripped = try Patches.stripTinygradOps(&g, &model.opsets)
    let retyped = Patches.needsPatch(g)
    if retyped {
      try Patches.patchUint8Inputs(&g)
    }
    let gathers = try Patches.normalizeGatherIndices(&g, src)
    let gemms = try Patches.gemmWithTransposedWeight(&g, src)
    let tiles = try Patches.expandToTile(&g, src)
    model.graph = g

    var parts: [(name: String, model: Model)]
    switch layout {
    case .split:
      let (vision, policy) = try Split.visionPolicy(model)
      parts = [("vision", vision), ("policy", policy)]
    case .whole:
      parts = [("model", model)]
    }
    for i in parts.indices {
      // _with_cache_key: any key already there goes, the part's own is added last.
      parts[i].model.props.removeAll { $0.key == cacheKeyProp || $0.key == "CACHE_KEY" }
      parts[i].model.props.append(Prop(raw: nil, key: cacheKeyProp, value: cacheKey(parts[i].name)))
    }

    let encoded = parts.map { part in
      (
        name: part.name, bytes: Encode.model(part.model, src),
        weights: part.model.graph!.initializers.reduce(Int64(0)) { $0 + Int64($1.raw?.count ?? 0) }
      )
    }
    let total = max(1, encoded.reduce(0) { $0 + $1.bytes.count })

    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    var done = 0
    var written: [URL] = []
    var reported: [Part] = []
    do {
      for part in encoded {
        let url = directory.appendingPathComponent("\(part.name).onnx")
        written.append(url)
        let base = done
        try PartWriter.write(part.bytes, src, to: url) { bytes in
          progress?(Double(base + bytes) / Double(total))
        }
        done += part.bytes.count
        reported.append(Part(name: part.name, url: url, weightBytes: part.weights))
      }
    } catch {
      for url in written {
        try? FileManager.default.removeItem(at: url)
      }
      throw error
    }
    progress?(1.0)
    return Report(
      stripped: stripped, retypedImages: retyped, gathers: gathers, gemms: gemms, tiles: tiles,
      parts: reported)
  }
}

// MARK: writing a part

/// Streams an encoded model to a file: owned bytes through a 1 MB buffer,
/// weights straight from the mapped source, transposed weights a block at a
/// time.
final class PartWriter {
  private static let bufferSize = 1 << 20
  /// Weights go to the file in pieces of this size, so progress moves.
  private static let chunkSize = 8 << 20
  /// A transposed weight is produced this many bytes at a time, at most.
  private static let blockSize = 4 << 20

  private let handle: FileHandle
  private var buffer: [UInt8] = []
  private var written = 0
  private let onWrite: (Int) -> Void

  private init(url: URL, onWrite: @escaping (Int) -> Void) throws {
    guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
      throw OnnxError("could not create \(url.path)")
    }
    handle = try FileHandle(forWritingTo: url)
    buffer.reserveCapacity(PartWriter.bufferSize)
    self.onWrite = onWrite
  }

  static func write(_ encoded: Encoded, _ src: Source, to url: URL, onWrite: @escaping (Int) -> Void) throws {
    let writer = try PartWriter(url: url, onWrite: onWrite)
    do {
      for piece in encoded.allPieces {
        switch piece {
        case .bytes(let b): try b.withUnsafeBytes { try writer.write($0) }
        case .source(let r): try writer.write(src.slice(r))
        case .transposed(let t): try writer.transpose(t, src)
        }
      }
      try writer.flush()
      try writer.handle.close()
    } catch {
      try? writer.handle.close()
      throw error
    }
    guard writer.written == encoded.count else {
      throw OnnxError("wrote \(writer.written) bytes of \(url.lastPathComponent) where \(encoded.count) were planned")
    }
  }

  private func write(_ bytes: UnsafeRawBufferPointer) throws {
    if buffer.count + bytes.count <= PartWriter.bufferSize {
      buffer.append(contentsOf: bytes)
      return
    }
    try flush()
    if bytes.count < PartWriter.bufferSize {
      buffer.append(contentsOf: bytes)
      return
    }
    var offset = 0
    while offset < bytes.count {
      let n = min(PartWriter.chunkSize, bytes.count - offset)
      try handle.write(contentsOf: UnsafeRawBufferPointer(rebasing: bytes[offset..<(offset + n)]))
      offset += n
      written += n
      onWrite(written)
    }
  }

  private func flush() throws {
    guard !buffer.isEmpty else { return }
    try handle.write(contentsOf: buffer)
    written += buffer.count
    buffer.removeAll(keepingCapacity: true)
    onWrite(written)
  }

  private func transpose(_ t: Transpose, _ src: Source) throws {
    switch t.elements {
    case .source(let r):
      try transpose(src.slice(r), t)
    case .owned(let bytes):
      try bytes.withUnsafeBytes { try transpose($0, t) }
    case .typed(let tensor):
      // Only this one weight is decoded, and it goes when this returns.
      let bytes = try Elements.littleEndian(tensor, src)
      try bytes.withUnsafeBytes { try transpose($0, t) }
    }
  }

  /// Writes the [cols, rows] transpose of a row-major [rows, cols] matrix, a
  /// block of output rows at a time. Each block reads a narrow column band
  /// of every source row, so the reads stay sequential within a row.
  private func transpose(_ elements: UnsafeRawBufferPointer, _ t: Transpose) throws {
    let rows = t.rows
    let cols = t.cols
    let size = t.elementSize
    guard elements.count == t.byteCount else {
      throw OnnxError("a weight has \(elements.count) bytes where [\(rows), \(cols)] needs \(t.byteCount)")
    }
    guard rows > 0, cols > 0, let from = elements.baseAddress else { return }
    let rowBytes = rows * size
    let band = max(1, min(cols, PartWriter.blockSize / rowBytes))
    var scratch = [UInt8](repeating: 0, count: band * rowBytes)
    try flush()
    try scratch.withUnsafeMutableBytes { scratchBytes in
      let to = scratchBytes.baseAddress!
      var j0 = 0
      while j0 < cols {
        let j1 = min(cols, j0 + band)
        switch size {
        case 1: PartWriter.transposeBand(UInt8.self, from, to, rows, cols, j0, j1)
        case 2: PartWriter.transposeBand(UInt16.self, from, to, rows, cols, j0, j1)
        case 4: PartWriter.transposeBand(UInt32.self, from, to, rows, cols, j0, j1)
        case 8: PartWriter.transposeBand(UInt64.self, from, to, rows, cols, j0, j1)
        default:
          for i in 0..<rows {
            for j in j0..<j1 {
              (to + ((j - j0) * rows + i) * size).copyMemory(from: from + (i * cols + j) * size, byteCount: size)
            }
          }
        }
        let n = (j1 - j0) * rowBytes
        try handle.write(contentsOf: UnsafeRawBufferPointer(start: to, count: n))
        written += n
        onWrite(written)
        j0 = j1
      }
    }
  }

  @inline(__always)
  private static func transposeBand<T: FixedWidthInteger>(
    _: T.Type, _ from: UnsafeRawPointer,
    _ to: UnsafeMutableRawPointer,
    _ rows: Int, _ cols: Int, _ j0: Int, _ j1: Int
  ) {
    let size = MemoryLayout<T>.size
    // The source may sit at any offset in the file, so its loads are unaligned.
    for i in 0..<rows {
      let row = from + i * cols * size
      for j in j0..<j1 {
        let value = row.loadUnaligned(fromByteOffset: j * size, as: T.self)
        to.storeBytes(of: value, toByteOffset: ((j - j0) * rows + i) * size, as: T.self)
      }
    }
  }
}
