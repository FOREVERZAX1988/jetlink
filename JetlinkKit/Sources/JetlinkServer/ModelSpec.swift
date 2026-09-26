import Foundation

/// openpilot's ModelConstants, as `jetlink/spec.py` duplicates them.
public enum ModelConstants {
  public static let runFrequency = 20
  public static let contextFrequency = 5
  public static let defaultFrameSkip = runFrequency / contextFrequency  // 4
  /// The upload chunk the comma sends a model in.
  public static let chunk = 4 << 20
  /// The driving output every layout has, 18452 floats in openpilot's layout.
  public static let drivingOutput = "outputs"
  /// The input only a stateful graph (openpilot #38916) has.
  public static let statefulFrame = "new_img"
  public static let stateOutputPrefix = "next_"
}

public struct NamedShape: Sendable, Equatable {
  public let name: String
  public let shape: [Int]

  public init(_ name: String, _ shape: [Int]) {
    self.name = name
    self.shape = shape
  }

  public var count: Int { shape.reduce(1, *) }
}

public struct NamedRange: Sendable, Equatable {
  public let name: String
  public let range: Range<Int>

  public init(_ name: String, _ range: Range<Int>) {
    self.name = name
    self.range = range
  }
}

/// Everything both ends need to agree on, derived from the ONNX. The Swift
/// twin of `jetlink/spec.py`'s ModelSpec: the same properties, the same wire
/// sizes, and the same dict on the wire.
public struct ModelSpec: Sendable, Equatable {
  public let sha256: String
  public let nbytes: Int64
  public let frameSkip: Int
  public let inputShapes: [NamedShape]
  public let outputShapes: [NamedShape]
  public let outputSlices: [NamedRange]
  public let checkpoint: String?

  public init(
    sha256: String, nbytes: Int64, frameSkip: Int, inputShapes: [NamedShape], outputShapes: [NamedShape], outputSlices: [NamedRange], checkpoint: String?
  ) {
    self.sha256 = sha256
    self.nbytes = nbytes
    self.frameSkip = frameSkip
    self.inputShapes = inputShapes
    self.outputShapes = outputShapes
    self.outputSlices = outputSlices
    self.checkpoint = checkpoint
  }

  public func input(_ name: String) -> [Int]? {
    inputShapes.first { $0.name == name }?.shape
  }

  public func output(_ name: String) -> [Int]? {
    outputShapes.first { $0.name == name }?.shape
  }

  public func withFrameSkip(_ frameSkip: Int) -> ModelSpec {
    ModelSpec(
      sha256: sha256, nbytes: nbytes, frameSkip: frameSkip, inputShapes: inputShapes, outputShapes: outputShapes, outputSlices: outputSlices,
      checkpoint: checkpoint)
  }

  // MARK: layout

  /// The graph keeps its own history (openpilot #38916).
  public var stateful: Bool { input(ModelConstants.statefulFrame) != nil }

  /// state_<q> input to the next_state_<q> output that feeds it next frame.
  /// Empty for a queued graph. Same rule as openpilot's ModelState.
  public var statePairs: [(input: String, output: String)] {
    inputShapes.compactMap { entry in
      let next = ModelConstants.stateOutputPrefix + entry.name
      return output(next) != nil ? (entry.name, next) : nil
    }
  }

  // MARK: vision

  /// (1, 12, H, W); queued graphs only.
  public var imgShape: [Int] { input("img") ?? [] }

  public var nFrames: Int { (imgShape.count > 1 ? imgShape[1] : 0) / 6 }

  public var modelHW: (h: Int, w: Int) {
    // new_img is (2, 6, H, W) and img (1, 12, H, W): the last two either way
    let shape = stateful ? (input(ModelConstants.statefulFrame) ?? []) : imgShape
    guard shape.count >= 2 else { return (0, 0) }
    return (shape[shape.count - 2], shape[shape.count - 1])
  }

  public var imgBufShape: [Int] {
    let (h, w) = modelHW
    return [frameSkip * (nFrames - 1) + 1, 6, h, w]
  }

  /// What `warp` on the comma produces: narrow and wide stacked.
  public var warpedShape: [Int] {
    let (h, w) = modelHW
    return [2, 6, h, w]
  }

  public var warpedBytes: Int { warpedShape.reduce(1, *) }

  // MARK: recurrent and scalar inputs

  /// Flattened per-frame feature size. (1,32,32,512) -> 16384.
  public var featDim: Int {
    guard let fb = input("features_buffer"), fb.count > 2 else { return 0 }
    return fb[2...].reduce(1, *)
  }

  public var packedShapes: [NamedShape] {
    let scalars = [
      NamedShape("traffic_convention", input("traffic_convention") ?? []),
      NamedShape("action_t", input("action_t") ?? []),
    ]
    if stateful {
      // the pulse is the graph's own input and the hidden state stays inside it
      return [NamedShape("desire", [(input("desire") ?? []).reduce(1, *)])] + scalars
    }
    let fb = input("features_buffer") ?? [0]
    let dp = input("desire_pulse") ?? [0, 0, 0]
    return [NamedShape("desire", [dp.count > 2 ? dp[2] : 0])] + scalars + [NamedShape("prev_feat", [fb[0], featDim])]
  }

  /// Where each of packedShapes sits in the flat floats.
  public var packedLayout: [(name: String, range: Range<Int>, shape: [Int])] {
    var out: [(String, Range<Int>, [Int])] = []
    var offset = 0
    for entry in packedShapes {
      out.append((entry.name, offset..<(offset + entry.count), entry.shape))
      offset += entry.count
    }
    return out
  }

  public var packedCount: Int { packedShapes.reduce(0) { $0 + $1.count } }
  public var packedBytes: Int { packedCount * 4 }

  public var featQShape: [Int] {
    let fb = input("features_buffer") ?? [0, 0]
    return [frameSkip * fb[1], fb[0], featDim]
  }

  public var desireQShape: [Int] {
    let dp = input("desire_pulse") ?? [0, 0, 0]
    return [frameSkip * dp[1], dp[0], dp[2]]
  }

  // MARK: output

  public var outputCount: Int { (output(ModelConstants.drivingOutput) ?? []).reduce(1, *) }
  /// float32 on the wire, as openpilot's JIT returns.
  public var outputBytes: Int { outputCount * 4 }

  // MARK: wire sizes

  public var inferReqBytes: Int { Wire.inferReqSize + warpedBytes + packedBytes }
  public var inferRespBytes: Int { Wire.inferRespSize + outputBytes }

  // MARK: the wire form, spec.py's to_dict and from_dict

  public func dictionary() -> [String: Any] {
    [
      "sha256": sha256,
      "nbytes": nbytes,
      "frame_skip": frameSkip,
      "checkpoint": checkpoint ?? NSNull(),
      "input_shapes": Dictionary(uniqueKeysWithValues: inputShapes.map { ($0.name, $0.shape) }),
      "output_shapes": Dictionary(uniqueKeysWithValues: outputShapes.map { ($0.name, $0.shape) }),
      "output_slices": Dictionary(uniqueKeysWithValues: outputSlices.map { ($0.name, [$0.range.lowerBound, $0.range.upperBound]) }),
    ]
  }

  public enum DecodeError: Error, CustomStringConvertible {
    case missing(String)

    public var description: String {
      switch self {
      case .missing(let field): return "model spec has no valid \(field)"
      }
    }
  }

  public static func from(_ d: [String: Any]) throws -> ModelSpec {
    guard let sha = d["sha256"] as? String else { throw DecodeError.missing("sha256") }
    guard let nbytes = (d["nbytes"] as? NSNumber)?.int64Value else { throw DecodeError.missing("nbytes") }
    let frameSkip = (d["frame_skip"] as? NSNumber)?.intValue ?? ModelConstants.defaultFrameSkip
    func shapes(_ key: String) throws -> [NamedShape] {
      guard let raw = d[key] as? [String: Any] else { throw DecodeError.missing(key) }
      return try raw.keys.sorted().map { name in
        guard let dims = raw[name] as? [NSNumber] else { throw DecodeError.missing("\(key).\(name)") }
        return NamedShape(name, dims.map(\.intValue))
      }
    }
    guard let rawSlices = d["output_slices"] as? [String: Any] else { throw DecodeError.missing("output_slices") }
    let slices = try rawSlices.map { name, value -> NamedRange in
      guard let bounds = value as? [NSNumber], bounds.count == 2 else { throw DecodeError.missing("output_slices.\(name)") }
      return NamedRange(name, bounds[0].intValue..<bounds[1].intValue)
    }.sorted { $0.range.lowerBound < $1.range.lowerBound }
    return ModelSpec(
      sha256: sha, nbytes: nbytes, frameSkip: frameSkip,
      inputShapes: try shapes("input_shapes"), outputShapes: try shapes("output_shapes"),
      outputSlices: slices, checkpoint: d["checkpoint"] as? String)
  }
}
