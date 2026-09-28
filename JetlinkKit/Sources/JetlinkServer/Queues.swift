import Foundation

/// Copies with the casts the model's inputs need, into whatever type the engine
/// declared. The Swift form of `queues.store`, the bulk casts through vImage.
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
  func reset()
  /// Writes one frame's inputs into the engine's buffers.
  func stage(warped: UnsafeRawPointer, packed: UnsafeRawPointer) throws
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

/// A queued graph (V1, V2): the image, feature and desire queues live here.
final class PolicyQueues: FrameStaging {
  private let spec: ModelSpec
  private let engine: any Engine
  private let imgQueue: RingQueue
  private let bigImgQueue: RingQueue
  private let featQueue: RingQueue
  private let desireQueue: RingQueue
  private let layout: [String: Range<Int>]
  /// Resolved once: the queues gathered into inputs, and the scalars copied
  /// straight in, so a frame builds no lists.
  private let gathers: [(queue: RingQueue, name: String, type: ElementType)]
  private let scalars: [(name: String, offset: Int, count: Int, type: ElementType)]
  private let desireType: ElementType
  private let desireOffset: Int
  private let prevFeatOffset: Int

  init(spec: ModelSpec, engine: any Engine) throws {
    self.spec = spec
    self.engine = engine
    imgQueue = RingQueue(shape: spec.imgBufShape)
    bigImgQueue = RingQueue(shape: spec.imgBufShape)
    featQueue = RingQueue(shape: spec.featQShape)
    desireQueue = RingQueue(shape: spec.desireQShape)
    let layout = Dictionary(uniqueKeysWithValues: spec.packedLayout.map { ($0.name, $0.range) })
    self.layout = layout
    for name in ["img", "big_img", "features_buffer", "desire_pulse", "traffic_convention", "action_t"] where engine.inputs[name] == nil {
      throw StagingError.missingInput(name)
    }
    let queues = [("img", imgQueue), ("big_img", bigImgQueue), ("features_buffer", featQueue)]
    gathers = queues.map { ($1, $0, engine.inputs[$0]!.type) }
    scalars = ["traffic_convention", "action_t"].map { name in
      let io = engine.inputs[name]!
      return (name, layout[name]!.lowerBound * 4, io.count, io.type)
    }
    desireType = engine.inputs["desire_pulse"]!.type
    desireOffset = layout["desire"]!.lowerBound * 4
    prevFeatOffset = layout["prev_feat"]!.lowerBound * 4
  }

  func reset() {
    for queue in [imgQueue, bigImgQueue, featQueue, desireQueue] { queue.reset() }
  }

  func stage(warped: UnsafeRawPointer, packed: UnsafeRawPointer) throws {
    let half = spec.warpedBytes / 2
    // push casts into the queue's float16, so the cast costs one row here
    // rather than the whole sampled window later
    imgQueue.push(u8: warped)
    bigImgQueue.push(u8: warped + half)
    desireQueue.push(f32: packed + desireOffset)
    featQueue.push(f32: packed + prevFeatOffset)

    let skip = spec.frameSkip
    for gather in gathers {
      gather.queue.gather(step: skip, into: engine.hostInput(gather.name)!, as: gather.type)
    }
    sampleDesire(into: engine.hostInput("desire_pulse")!, as: desireType)
    for scalar in scalars {
      Stage.store(f32: packed + scalar.offset, count: scalar.count, into: engine.hostInput(scalar.name)!, as: scalar.type)
    }
  }

  /// openpilot: `buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1)`, the
  /// strongest desire in each group of frame_skip frames.
  private func sampleDesire(into dest: UnsafeMutableRawPointer, as type: ElementType) {
    let skip = spec.frameSkip
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
  private let spec: ModelSpec
  private let engine: any Engine
  private let scalars: [(name: String, range: Range<Int>)]
  /// The pairs the engine left to the host, and whether a run since the
  /// last reset left next_state_ outputs to carry over.
  private let hostPairs: [(input: TensorSpec, output: TensorSpec)]
  private var carry = false

  init(spec: ModelSpec, engine: any Engine) throws {
    self.spec = spec
    self.engine = engine
    let pairs = spec.statePairs
    guard !pairs.isEmpty else { throw StagingError.noStatePairs }
    scalars = spec.packedLayout.map { ($0.name, $0.range) }
    for name in [ModelConstants.statefulFrame] + scalars.map(\.name) where engine.inputs[name] == nil {
      throw StagingError.missingInput(name)
    }
    if try engine.loopState(pairs) {
      hostPairs = []
      return
    }
    hostPairs = try pairs.map { pair in
      guard let input = engine.inputs[pair.input], let output = engine.outputs[pair.output], input.count == output.count,
        input.type == output.type || (Set([input.type, output.type]).isSubset(of: [.float, .float16]))
      else { throw HostError.failed("state \(pair.input) cannot be fed from \(pair.output)") }
      return (input, output)
    }
  }

  func reset() {
    guard !hostPairs.isEmpty else {
      engine.resetState()
      return
    }
    for pair in hostPairs {
      engine.hostInput(pair.input.name)?.initializeMemory(as: UInt8.self, repeating: 0, count: pair.input.byteCount)
    }
    carry = false
  }

  func stage(warped: UnsafeRawPointer, packed: UnsafeRawPointer) throws {
    if carry {
      for (input, output) in hostPairs {
        let dest = engine.hostInput(input.name)!
        let source = engine.output(output.name)!
        switch output.type {
        case .float16: Stage.store(f16: source.assumingMemoryBound(to: UInt16.self), count: input.count, into: dest, as: input.type)
        case .float: Stage.store(f32: source, count: input.count, into: dest, as: input.type)
        default: dest.copyMemory(from: source, byteCount: input.byteCount)
        }
      }
    }
    carry = !hostPairs.isEmpty
    let frame = engine.inputs[ModelConstants.statefulFrame]!
    Stage.store(u8: warped, count: spec.warpedBytes, into: engine.hostInput(frame.name)!, as: frame.type)
    for scalar in scalars {
      let io = engine.inputs[scalar.name]!
      Stage.store(f32: packed + scalar.range.lowerBound * 4, count: scalar.range.count, into: engine.hostInput(scalar.name)!, as: io.type)
    }
  }
}

enum Staging {
  /// Queues for a queued graph, the loop for a stateful one.
  static func forModel(_ spec: ModelSpec, engine: any Engine) throws -> any FrameStaging {
    spec.stateful ? try StateLoop(spec: spec, engine: engine) : try PolicyQueues(spec: spec, engine: engine)
  }
}
