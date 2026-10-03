import Foundation
import JetlinkServer

/// A loaded model: one LiteRT compiled model run over buffers the engine
/// owns, the LiteRT counterpart of `OrtEngine`.
///
/// The host writes a frame's inputs into `hostInput(_:)`, calls `run()`, and
/// reads `output(_:)`. LiteRT reads and writes those host buffers in place:
/// the CPU directly, a GPU through a copy of its own each run. Buffers of
/// the GPU's own, made once and written and read each frame, were no faster
/// on Metal: Cinque Terre V3 ran 39.8 and 40.3 ms a frame that way against
/// 39.7 and 40.1 ms through the host's buffers, outputs bit for bit the
/// same. An NPU gets buffers of its own, which the engine copies the host's
/// into and out of each run: LiteRT's NPU path runs on AHardwareBuffers, and
/// its own fallback for host memory is the CPU's, untried with an NPU. After
/// `loopState`, a stateful graph's next_state_ outputs feed its state_ inputs
/// on the next run without crossing to the host. On a GPU or an NPU each pair
/// has two buffers of the accelerator's that swap roles every run; on the CPU
/// two host buffers do, as onnxruntime's do.
public final class LiteRtEngine: EngineCore, @unchecked Sendable {
  public let device: String
  /// What runs the model, for the log and the benchmark's report:
  /// "GPU(fp16)", "CPU(4 threads)".
  public let label: String
  /// What runs the model: on an NPU compile, the NPU, or the GPU where the
  /// NPU's compiler could not take it.
  public let hardware: LiteRtHardware
  /// Whether the accelerator runs every op, none left to LiteRT's own CPU
  /// kernels.
  public let fullyAccelerated: Bool
  /// Whether the looped state stays in the accelerator's memory.
  public var stateOnDevice: Bool { !deviceState.isEmpty }
  public override var notes: String { stateOnDevice ? "state on the \(hardware == .npu ? "NPU" : "GPU")" : "" }

  private var model: LiteRtModel?
  /// Every buffer a run reads or writes, alive as long as the sets below.
  private var buffers: [LiteRtBuffer] = []
  /// [set][signature index]: the buffers' pointers, one set without a loop,
  /// two with one.
  private var inputSets: [[OpaquePointer?]] = []
  private var outputSets: [[OpaquePointer?]] = []
  /// A looped pair's two GPU buffers, by its state_ input's name.
  private var deviceState: [String: [LiteRtBuffer]] = [:]
  /// Which of the GPU buffers the next run reads.
  private var phase = 0
  /// On an NPU, per set: the host buffers copied into the NPU's own before a
  /// run, and out of them after.
  private var copiesIn: [[Copy]] = []
  private var copiesOut: [[Copy]] = []

  private struct Copy {
    let buffer: LiteRtBuffer
    let host: UnsafeMutableRawPointer
    let bytes: Int
  }

  /// Loads and compiles `model`. LiteRT must be open (`LiteRtRuntime.load`).
  /// `label` names what runs it, given what does.
  init(model: URL, options: LiteRtCompileOptions, device: String, label: (LiteRtHardware) -> String) throws {
    let compiled = try LiteRtModel(model: model, options: options)
    self.model = compiled
    self.device = device
    hardware = try compiled.hardware
    self.label = label(hardware)
    fullyAccelerated = try compiled.fullyAccelerated
    try super.init(
      inputs: Dictionary(uniqueKeysWithValues: compiled.inputs.map { ($0.name, $0) }),
      outputs: Dictionary(uniqueKeysWithValues: compiled.outputs.map { ($0.name, $0) }))
    // A throw from here on closes through deinit.
    try rebind([])
  }

  deinit {
    close()
  }

  /// Loops each pair where the accelerator works: in GPU memory when it
  /// does not read the host's, else in two host buffers. The host's buffers
  /// for a pair looped on the GPU, and for every looped output, are freed.
  public override func bindLoop(_ pairs: [(input: String, output: String)]) throws {
    guard let model else { throw HostError.failed("engine is closed") }
    for pair in pairs {
      guard let input = inputs[pair.input], let output = outputs[pair.output] else {
        throw HostError.failed("no tensors \(pair.input) and \(pair.output) to loop")
      }
      guard input.byteCount == output.byteCount, input.type == output.type else {
        throw HostError.failed("\(pair.output) \(output.shape) cannot feed \(pair.input) \(input.shape)")
      }
    }
    let indices = Dictionary(uniqueKeysWithValues: model.inputs.enumerated().map { ($1.name, $0) })
    let onDevice = try pairs.contains { try !model.readsHostMemory(input: indices[$0.input]!) }
    if onDevice {
      deviceState = try Dictionary(
        uniqueKeysWithValues: pairs.map { pair in
          let index = indices[pair.input]!
          return (pair.input, [try LiteRtBuffer(model, index: index), try LiteRtBuffer(model, index: index)])
        })
      try rebind(pairs)
      releaseHostBuffers(pairs.flatMap { [$0.input, $0.output] })
    } else {
      try doubleBuffer(pairs.map(\.input))
      try rebind(pairs)
      releaseHostBuffers(pairs.map(\.output))
    }
  }

  /// One set of buffers, or two with a loop: set 0 reads state_ from the
  /// first buffer and writes next_state_ into the second, set 1 the other
  /// way round. Everything else is the host's buffer, wrapped, or on an NPU
  /// a buffer of the NPU's the host's is copied through.
  private func rebind(_ pairs: [(input: String, output: String)]) throws {
    guard let model else { return }
    // The old wraps point into host memory that may be about to go.
    inputSets = []
    outputSets = []
    copiesIn = []
    copiesOut = []
    buffers = []
    let inputOf = Dictionary(uniqueKeysWithValues: pairs.map { ($0.output, $0.input) })
    var own: [String: LiteRtBuffer] = [:]
    var copiesIn: [[Copy]] = []
    var copiesOut: [[Copy]] = []
    func host(_ spec: TensorSpec, output: Bool, index: Int, _ memory: UnsafeMutableRawPointer) throws -> OpaquePointer? {
      guard hardware == .npu else {
        let wrapped = try LiteRtBuffer(model, output: output, index: index, wrapping: memory, bytes: spec.byteCount)
        buffers.append(wrapped)
        return wrapped.pointer
      }
      // one of the NPU's per tensor, whichever host buffer a set copies
      let key = "\(output ? "out" : "in") \(spec.name)"
      let buffer = try own[key] ?? LiteRtBuffer(model, output: output, index: index)
      if own[key] == nil {
        own[key] = buffer
        buffers.append(buffer)
      }
      let copy = Copy(buffer: buffer, host: memory, bytes: spec.byteCount)
      if output {
        copiesOut[copiesOut.count - 1].append(copy)
      } else {
        copiesIn[copiesIn.count - 1].append(copy)
      }
      return buffer.pointer
    }
    var inputSets: [[OpaquePointer?]] = []
    var outputSets: [[OpaquePointer?]] = []
    for set in 0..<(pairs.isEmpty ? 1 : 2) {
      copiesIn.append([])
      copiesOut.append([])
      var ins: [OpaquePointer?] = []
      for (index, spec) in model.inputs.enumerated() {
        if let state = deviceState[spec.name] {
          ins.append(state[set].pointer)
        } else {
          ins.append(try host(spec, output: false, index: index, buffer(spec.name, parity: set)!))
        }
      }
      var outs: [OpaquePointer?] = []
      for (index, spec) in model.outputs.enumerated() {
        if let input = inputOf[spec.name] {
          // writes what the state_ input reads next run
          if let state = deviceState[input] {
            outs.append(state[set ^ 1].pointer)
          } else {
            outs.append(try host(spec, output: true, index: index, buffer(input, parity: set ^ 1)!))
          }
        } else {
          outs.append(try host(spec, output: true, index: index, buffer(spec.name, parity: set)!))
        }
      }
      inputSets.append(ins)
      outputSets.append(outs)
    }
    self.inputSets = inputSets
    self.outputSets = outputSets
    self.copiesIn = copiesIn
    self.copiesOut = copiesOut
  }

  public override func execute() throws {
    guard let model else { throw HostError.failed("engine is closed") }
    let set = stateOnDevice ? phase : parity
    for copy in copiesIn[set] {
      try copy.buffer.write(from: copy.host, bytes: copy.bytes)
    }
    try model.run(inputs: inputSets[set], outputs: outputSets[set])
    for copy in copiesOut[set] {
      try copy.buffer.read(into: copy.host, bytes: copy.bytes)
    }
    if stateOnDevice {
      phase ^= 1
    }
  }

  /// The GPU's copies of the state, emptied as the host's are.
  public override func stateDidReset() {
    for state in deviceState.values {
      for buffer in state {
        try? buffer.zero()
      }
    }
    phase = 0
  }

  /// What a looped state_ input holds for the next run, copied out: for
  /// tests, which cannot see GPU memory otherwise.
  func state(_ name: String, into data: UnsafeMutableRawPointer) throws {
    guard let spec = inputs[name] else { throw HostError.failed("no input \(name)") }
    if let state = deviceState[name] {
      try state[phase].read(into: data, bytes: spec.byteCount)
    } else if let host = buffer(name, parity: parity) {
      data.copyMemory(from: host, byteCount: spec.byteCount)
    }
  }

  /// The first run loads the GPU's programs and weights; the second is the
  /// steady state.
  public override func warm() throws -> String {
    _ = try super.warm()
    return "LiteRT \(LiteRtRuntime.version) on \(device) in process, \(label)"
  }

  public override func close() {
    guard !isClosed else { return }
    // They wrap the buffers the core frees.
    inputSets = []
    outputSets = []
    copiesIn = []
    copiesOut = []
    buffers = []
    deviceState = [:]
    model = nil
    super.close()
  }
}
