import Foundation

/// Copies with the casts the model's inputs need, into whatever type the engine
/// declared. The Swift form of `queues.store`, the bulk casts through Convert.
enum Stage {
  static func store(u8 source: UnsafeRawPointer, count: Int, into dest: UnsafeMutableRawPointer, as type: ElementType) {
    switch type {
    case .float16:
      Convert.u8ToF16(source, dest, count: count)
    case .float:
      let src = source.assumingMemoryBound(to: UInt8.self)
      let out = dest.assumingMemoryBound(to: Float.self)
      for i in 0..<count { out[i] = Float(src[i]) }
    case .uint8:
      dest.copyMemory(from: source, byteCount: count)
    default:
      preconditionFailure("images staged as \(type.name)")
    }
  }

  /// float32 off the wire (possibly unaligned) into a float32 or float16 input.
  static func store(f32 source: UnsafeRawPointer, count: Int, into dest: UnsafeMutableRawPointer, as type: ElementType) {
    switch type {
    case .float:
      dest.copyMemory(from: source, byteCount: count * 4)
    case .float16:
      Convert.f32ToF16(source, dest, count: count)
    default:
      preconditionFailure("scalars staged as \(type.name)")
    }
  }

  /// float16 bits from a queue into a float16 or float32 input.
  static func store(f16 source: UnsafePointer<UInt16>, count: Int, into dest: UnsafeMutableRawPointer, as type: ElementType) {
    switch type {
    case .float16:
      dest.copyMemory(from: source, byteCount: count * 2)
    case .float:
      Convert.f16ToF32(source, dest, count: count)
    default:
      preconditionFailure("queue staged as \(type.name)")
    }
  }
}

/// A fixed-length FIFO of float16 rows over a preallocated buffer. Logical row
/// i (0 = oldest) lives at physical row (head + i) % n. Ring buffers rather
/// than openpilot's rolling `cat(buf[1:], new)`, which copies 8.4 MB a frame.
final class RingQueue {
  let rows: Int
  let rowCount: Int
  private let buffer: UnsafeMutablePointer<UInt16>
  private var head = 0

  init(shape: [Int]) {
    rows = shape[0]
    rowCount = shape.dropFirst().reduce(1, *)
    buffer = .allocate(capacity: rows * rowCount)
    buffer.initialize(repeating: 0, count: rows * rowCount)
  }

  deinit {
    buffer.deallocate()
  }

  func reset() {
    buffer.update(repeating: 0, count: rows * rowCount)
    head = 0
  }

  /// The slot the oldest row occupies becomes the newest once head moves.
  func push(u8 source: UnsafeRawPointer) {
    Stage.store(u8: source, count: rowCount, into: UnsafeMutableRawPointer(buffer + head * rowCount), as: .float16)
    head = (head + 1) % rows
  }

  func push(f32 source: UnsafeRawPointer) {
    Stage.store(f32: source, count: rowCount, into: UnsafeMutableRawPointer(buffer + head * rowCount), as: .float16)
    head = (head + 1) % rows
  }

  func row(_ logical: Int) -> UnsafePointer<UInt16> {
    UnsafePointer(buffer + ((head + logical) % rows) * rowCount)
  }

  /// openpilot's `buf[::frame_skip]`: logical rows 0, step, 2*step, ... oldest
  /// first, written contiguously into `dest`.
  func gather(step: Int, into dest: UnsafeMutableRawPointer, as type: ElementType) {
    var out = 0
    var logical = 0
    while logical < rows {
      Stage.store(f16: row(logical), count: rowCount, into: dest + out * rowCount * type.size, as: type)
      out += 1
      logical += step
    }
  }
}

/// A model's server-side history. The Swift form of `queues.PolicyQueues` and
/// `queues.StateLoop`: the comma sends only the newest warped frame and the
/// packed scalars, and this turns them into the model's inputs.
protocol FrameStaging: AnyObject {
  /// The frame's sizes, worked out once at load.
  var layout: FrameLayout { get }
  func reset()
  /// Writes one frame's inputs into the engine's buffers. `packed` is read
  /// where the request put it, aligned or not.
  func stage(warped: UnsafeRawPointer, packed: UnsafeRawPointer) throws
}

/// A model's INFER_REQ and reply as the session handles them, from the spec
/// and the engine once at load: the spec derives each size through arrays,
/// which a frame should not build.
struct FrameLayout {
  /// INFER_REQ's payload: the request head, the warped frame, the packed floats.
  let requestBytes: Int
  let warpedBytes: Int
  /// The driving output, float32 on the wire whatever the engine gives.
  let outputCount: Int
  let outputType: ElementType?

  init(spec: ModelSpec, engine: any Engine) {
    requestBytes = spec.inferReqBytes
    warpedBytes = spec.warpedBytes
    outputCount = spec.outputCount
    outputType = engine.outputs[ModelConstants.drivingOutput]?.type
  }
}

enum StagingError: Error, CustomStringConvertible {
  case missingInput(String)
  case noStatePairs

  var description: String {
    switch self {
    case .missingInput(let name): return "the engine has no input \(name) to stage"
    case .noStatePairs: return "the graph takes new_img but returns no next_state_ outputs"
    }
  }
}

/// An engine input's host buffer and type, looked up once at load so a frame
/// looks nothing up. The buffers an engine gives for the inputs the host
/// writes stay put from load to close: EngineCore allocates them once, and
/// only a looped pair's state_ input is ever double-buffered or released,
/// which staging never writes.
struct StagingTarget {
  let name: String
  let pointer: UnsafeMutableRawPointer
  let type: ElementType

  init(_ engine: any Engine, _ name: String) throws {
    guard let io = engine.inputs[name], let pointer = engine.hostInput(name) else { throw StagingError.missingInput(name) }
    self.name = name
    self.pointer = pointer
    self.type = io.type
  }

  /// For the debug build's check that the engine kept its word.
  func isCurrent(in engine: any Engine) -> Bool {
    engine.hostInput(name) == pointer
  }
}

/// A queued graph (V1, V2): the image, feature and desire queues live here.
final class PolicyQueues: FrameStaging {
  let layout: FrameLayout
  private let engine: any Engine
  private let frameSkip: Int
  /// One camera's share of the warped frame.
  private let cameraBytes: Int
  // The queues hold float16, as Python's do, so the cast is paid once per
  // row. An engine that took the images as uint8 and cast them itself (a
  // plan keeping the graph's Cast) would want uint8 image queues here,
  // chosen from `img`'s type as the targets below are, and a gather that
  // copies bytes.
  private let imgQueue: RingQueue
  private let bigImgQueue: RingQueue
  private let featQueue: RingQueue
  private let desireQueue: RingQueue
  private let gathers: [(queue: RingQueue, target: StagingTarget)]
  private let desire: StagingTarget
  private let scalars: [(offset: Int, count: Int, target: StagingTarget)]
  private let desireOffset: Int
  private let prevFeatOffset: Int

  init(spec: ModelSpec, engine: any Engine) throws {
    self.engine = engine
    layout = FrameLayout(spec: spec, engine: engine)
    frameSkip = spec.frameSkip
    cameraBytes = spec.warpedBytes / 2
    imgQueue = RingQueue(shape: spec.imgBufShape)
    bigImgQueue = RingQueue(shape: spec.imgBufShape)
    featQueue = RingQueue(shape: spec.featQShape)
    desireQueue = RingQueue(shape: spec.desireQShape)
    let packed = Dictionary(uniqueKeysWithValues: spec.packedLayout.map { ($0.name, $0.range) })
    gathers = try [(imgQueue, "img"), (bigImgQueue, "big_img"), (featQueue, "features_buffer")].map { ($0, try StagingTarget(engine, $1)) }
    desire = try StagingTarget(engine, "desire_pulse")
    scalars = try ["traffic_convention", "action_t"].map { name in
      let target = try StagingTarget(engine, name)
      return (packed[name]!.lowerBound * 4, engine.inputs[name]!.count, target)
    }
    desireOffset = packed["desire"]!.lowerBound * 4
    prevFeatOffset = packed["prev_feat"]!.lowerBound * 4
  }

  func reset() {
    for queue in [imgQueue, bigImgQueue, featQueue, desireQueue] { queue.reset() }
  }

  func stage(warped: UnsafeRawPointer, packed: UnsafeRawPointer) throws {
    assert(
      gathers.allSatisfy { $0.target.isCurrent(in: engine) } && desire.isCurrent(in: engine)
        && scalars.allSatisfy { $0.target.isCurrent(in: engine) })
    // push casts into the queue's float16, so the cast costs one row here
    // rather than the whole sampled window later
    imgQueue.push(u8: warped)
    bigImgQueue.push(u8: warped + cameraBytes)
    desireQueue.push(f32: packed + desireOffset)
    featQueue.push(f32: packed + prevFeatOffset)

    for gather in gathers {
      gather.queue.gather(step: frameSkip, into: gather.target.pointer, as: gather.target.type)
    }
    sampleDesire(into: desire.pointer, as: desire.type)
    for scalar in scalars {
      Stage.store(f32: packed + scalar.offset, count: scalar.count, into: scalar.target.pointer, as: scalar.target.type)
    }
  }

  /// openpilot: `buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1)`, the
  /// strongest desire in each group of frame_skip frames.
  private func sampleDesire(into dest: UnsafeMutableRawPointer, as type: ElementType) {
    let skip = frameSkip
    let width = desireQueue.rowCount
    let groups = desireQueue.rows / skip
    for group in 0..<groups {
      for column in 0..<width {
        var best = -Float.infinity
        for k in 0..<skip {
          let value = Float(Float16(bitPattern: desireQueue.row(group * skip + k)[column]))
          best = max(best, value)
        }
        let index = group * width + column
        switch type {
        case .float16: dest.assumingMemoryBound(to: UInt16.self)[index] = Float16(best).bitPattern
        case .float: dest.assumingMemoryBound(to: Float.self)[index] = best
        default: preconditionFailure("desire_pulse staged as \(type.name)")
        }
      }
    }
  }
}

/// A stateful graph (openpilot #38916): the frame goes into new_img and the
/// scalars into their inputs as they are; the engine loops the queues itself.
/// An engine that declines the loop (a TensorRT pair whose size or type
/// differs) has each next_state_ output copied into its state_ input here,
/// as `queues.StateLoop.after_run` did, when the next frame is staged.
final class StateLoop: FrameStaging {
  /// A pair the engine left to the host: the state_ input, and the
  /// next_state_ output that feeds it, both resolved once as the targets are.
  private struct HostPair {
    let input: StagingTarget
    let count: Int
    let output: String
    let source: UnsafeRawPointer
    let sourceType: ElementType
  }

  let layout: FrameLayout
  private let engine: any Engine
  /// new_img's type picks the conversion: uint8 would be a plain copy, for
  /// an engine that casts on the device.
  private let frame: StagingTarget
  private let scalars: [(offset: Int, count: Int, target: StagingTarget)]
  private let hostPairs: [HostPair]
  /// Whether a run since the last reset left next_state_ outputs to carry over.
  private var carry = false

  init(spec: ModelSpec, engine: any Engine) throws {
    self.engine = engine
    layout = FrameLayout(spec: spec, engine: engine)
    let pairs = spec.statePairs
    guard !pairs.isEmpty else { throw StagingError.noStatePairs }
    for name in [ModelConstants.statefulFrame] + spec.packedLayout.map(\.name) where engine.inputs[name] == nil {
      throw StagingError.missingInput(name)
    }
    // Looping first: that is when an engine settles which buffers it keeps.
    let looped = try engine.loopState(pairs)
    frame = try StagingTarget(engine, ModelConstants.statefulFrame)
    scalars = try spec.packedLayout.map { ($0.range.lowerBound * 4, $0.range.count, try StagingTarget(engine, $0.name)) }
    hostPairs =
      looped
      ? []
      : try pairs.map { pair in
        guard let input = engine.inputs[pair.input], let output = engine.outputs[pair.output], input.count == output.count,
          input.type == output.type || Set([input.type, output.type]).isSubset(of: [.float, .float16]),
          let source = engine.output(pair.output)
        else { throw HostError.failed("state \(pair.input) cannot be fed from \(pair.output)") }
        return HostPair(input: try StagingTarget(engine, pair.input), count: input.count, output: pair.output, source: source, sourceType: output.type)
      }
  }

  func reset() {
    guard !hostPairs.isEmpty else {
      engine.resetState()
      return
    }
    for pair in hostPairs {
      pair.input.pointer.initializeMemory(as: UInt8.self, repeating: 0, count: pair.count * pair.input.type.size)
    }
    carry = false
  }

  func stage(warped: UnsafeRawPointer, packed: UnsafeRawPointer) throws {
    assert(
      frame.isCurrent(in: engine) && scalars.allSatisfy { $0.target.isCurrent(in: engine) }
        && hostPairs.allSatisfy { $0.input.isCurrent(in: engine) && engine.output($0.output) == $0.source })
    if carry {
      for pair in hostPairs {
        let dest = pair.input.pointer
        switch pair.sourceType {
        case .float16: Stage.store(f16: pair.source.assumingMemoryBound(to: UInt16.self), count: pair.count, into: dest, as: pair.input.type)
        case .float: Stage.store(f32: pair.source, count: pair.count, into: dest, as: pair.input.type)
        default: dest.copyMemory(from: pair.source, byteCount: pair.count * pair.input.type.size)
        }
      }
    }
    carry = !hostPairs.isEmpty
    Stage.store(u8: warped, count: layout.warpedBytes, into: frame.pointer, as: frame.type)
    for scalar in scalars {
      Stage.store(f32: packed + scalar.offset, count: scalar.count, into: scalar.target.pointer, as: scalar.target.type)
    }
  }
}

enum Staging {
  /// Queues for a queued graph, the loop for a stateful one.
  static func forModel(_ spec: ModelSpec, engine: any Engine) throws -> any FrameStaging {
    spec.stateful ? try StateLoop(spec: spec, engine: engine) : try PolicyQueues(spec: spec, engine: engine)
  }
}
