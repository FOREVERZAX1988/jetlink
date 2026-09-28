import Foundation

/// One session of a chain: its model file, the execution provider to run it
/// on, and that provider's options.
public struct SessionPlan: Sendable, Equatable {
  public let model: URL
  /// onnxruntime's name for the provider, "CoreML" or "QNN"; nil for the CPU
  /// provider alone.
  public let provider: String?
  public let options: [String: String]
  /// Session config entries on top of jetlink's own, as the QNN provider's
  /// context cache takes them.
  public let config: [String: String]
  /// The CPU provider's intra-op pool: 1 where an accelerator does the work.
  public let threads: Int
  /// The session as the log and the hello name it: "CoreML(CPUAndGPU)",
  /// "QNN(htp)", "CPU".
  public let label: String
  /// Runs on the GPU, so the Metal keep-alive helps it.
  public let usesGPU: Bool
  /// Runs on the Neural Engine, so the CPU keep-warm helps it.
  public let usesNeuralEngine: Bool

  public init(
    model: URL, provider: String?, options: [String: String] = [:], config: [String: String] = [:], threads: Int = 1, label: String,
    usesGPU: Bool = false, usesNeuralEngine: Bool = false
  ) {
    self.model = model
    self.provider = provider
    self.options = options
    self.config = config
    self.threads = threads
    self.label = label
    self.usesGPU = usesGPU
    self.usesNeuralEngine = usesNeuralEngine
  }

  /// A CoreML session on `computeUnits` ("CPUAndNeuralEngine", "CPUAndGPU",
  /// "ALL", or nil for the CPU provider alone), with CoreML's compiled model
  /// kept in `cacheDirectory` so a load does not compile again. The provider
  /// options are the ones the Python backend passes (backends/ort/__init__.py).
  public init(model: URL, computeUnits: String?, cacheDirectory: URL?) {
    guard let computeUnits else {
      self.init(model: model, provider: nil, label: "CPU")
      return
    }
    let neuralEngine = computeUnits == "CPUAndNeuralEngine" || computeUnits == "ALL"
    var options = [
      "ModelFormat": "MLProgram",
      "MLComputeUnits": computeUnits,
    ]
    if neuralEngine {
      // Apple's hint for a model that is predicted many times. Only where
      // the Neural Engine is in play: on the GPU it changed nothing.
      options["SpecializationStrategy"] = "FastPrediction"
    }
    if let cacheDirectory {
      options["ModelCacheDirectory"] = cacheDirectory.path
    }
    self.init(
      model: model, provider: "CoreML", options: options, label: "CoreML(\(computeUnits))",
      usesGPU: computeUnits == "CPUAndGPU" || computeUnits == "ALL", usesNeuralEngine: neuralEngine)
  }
}

/// A loaded model: a chain of onnxruntime sessions run back to back over
/// buffers the engine owns, the Swift form of the Python ort worker.
///
/// The host writes a frame's inputs into `hostInput(_:)`, calls `run()`, and
/// reads `output(_:)`. The trunk's outputs feed the policy by name without
/// leaving the engine, and after `loopState` a stateful graph's next_state_
/// outputs feed its state_ inputs on the next run by swapping two buffers:
/// the 12 MB of queues never cross to the host.
public final class OrtEngine: @unchecked Sendable {
  public let device: String
  /// What the host stages: every session's inputs no earlier session produces.
  public let inputs: [String: TensorSpec]
  /// What the host can read: every session's outputs no later session reads.
  public let outputs: [String: TensorSpec]
  /// How long the last `run()` took, the whole chain, in microseconds.
  public private(set) var lastGpuUs: UInt32 = 0
  public let providers: [String]
  /// Whether a CPU keep-warm thread runs beside this engine.
  public var keepsCPUWarm: Bool { keepWarm != nil }

  private let chain: [OrtSession]
  private var buffers: [String: UnsafeMutableRawPointer] = [:]
  /// state_ input -> the buffer its next_state_ output writes, swapped each run.
  private var spare: [String: UnsafeMutableRawPointer] = [:]
  private var looped: [(input: String, output: String)] = []
  private var bindings: [[OrtBinding]] = []  // [parity][session]
  private var parity = 0
  #if canImport(Metal)
    private let keepAlive: MetalKeepAlive?
  #endif
  private let keepWarm: CPUKeepWarm?
  private var closed = false

  /// `keepAlive` keeps the GPU clocked up between frames, `keepCPUWarm` the
  /// CPU; each only where a plan runs on that unit.
  public init(plans: [SessionPlan], device: String, keepAlive: Bool = true, keepCPUWarm: Bool = true) throws {
    self.device = device
    var chain: [OrtSession] = []
    for plan in plans {
      chain.append(try OrtSession(model: plan.model, provider: plan.provider, options: plan.options, config: plan.config, threads: plan.threads))
    }
    self.chain = chain
    self.providers = plans.map(\.label)

    var produced = Set<String>()
    var inputs: [String: TensorSpec] = [:]
    for session in chain {
      for spec in session.inputs where !produced.contains(spec.name) && inputs[spec.name] == nil {
        inputs[spec.name] = spec
      }
      produced.formUnion(session.outputs.map(\.name))
    }
    var outputs: [String: TensorSpec] = [:]
    for (index, session) in chain.enumerated() {
      let readLater = Set(chain[(index + 1)...].flatMap { $0.inputs.map(\.name) })
      for spec in session.outputs where !readLater.contains(spec.name) {
        outputs[spec.name] = spec
      }
    }
    self.inputs = inputs
    self.outputs = outputs

    // One buffer per tensor name, zeroed: inputs, hand-offs and outputs alike.
    var sizes: [String: Int] = [:]
    for session in chain {
      for spec in session.inputs + session.outputs {
        sizes[spec.name] = max(sizes[spec.name] ?? 0, spec.byteCount)
      }
    }
    for (name, size) in sizes {
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: max(size, 1), alignment: 64)
      buffer.initializeMemory(as: UInt8.self, repeating: 0, count: max(size, 1))
      buffers[name] = buffer
    }
    #if canImport(Metal)
      self.keepAlive = keepAlive && plans.contains(where: \.usesGPU) ? MetalKeepAlive.make() : nil
    #endif
    self.keepWarm = keepCPUWarm && plans.contains(where: \.usesNeuralEngine) ? CPUKeepWarm() : nil
    do {
      try rebind()
    } catch {
      release()
      throw error
    }
  }

  deinit {
    close()
  }

  public func hostInput(_ name: String) -> UnsafeMutableRawPointer? {
    guard inputs[name] != nil else { return nil }
    return current(name)
  }

  public func output(_ name: String) -> UnsafeRawPointer? {
    guard outputs[name] != nil else { return nil }
    return UnsafeRawPointer(current(name))
  }

  /// Where a tensor is this run: a looped state input alternates between two buffers.
  private func current(_ name: String) -> UnsafeMutableRawPointer {
    if parity == 1, let other = spare[name] {
      return other
    }
    return buffers[name]!
  }

  /// Keep a stateful graph's queues here, each next_state_ output fed back as
  /// its state_ input on the next run. Always true: this engine can.
  @discardableResult
  public func loopState(_ pairs: [(input: String, output: String)]) throws -> Bool {
    looped = pairs
    for pair in pairs where spare[pair.input] == nil {
      guard let spec = inputs[pair.input] else { continue }
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: max(spec.byteCount, 1), alignment: 64)
      spare[pair.input] = buffer
    }
    try rebind()
    resetState()
    return true
  }

  /// Empty queues, as openpilot's warmup leaves them.
  public func resetState() {
    for pair in looped {
      guard let spec = inputs[pair.input] else { continue }
      buffers[pair.input]?.initializeMemory(as: UInt8.self, repeating: 0, count: spec.byteCount)
      spare[pair.input]?.initializeMemory(as: UInt8.self, repeating: 0, count: spec.byteCount)
    }
    parity = 0
  }

  /// One binding per session, or two with a loop: parity 0 reads state_ from
  /// the first buffer and writes next_state_ into the second, parity 1 the
  /// other way round. The next_state_ output has no buffer of its own.
  private func rebind() throws {
    let parities = looped.isEmpty ? 1 : 2
    var sets: [[OrtBinding]] = []
    let outputOf = Dictionary(uniqueKeysWithValues: looped.map { ($0.output, $0.input) })
    for p in 0..<parities {
      var set: [OrtBinding] = []
      for session in chain {
        let ins = session.inputs.map { spec -> (TensorSpec, UnsafeMutableRawPointer) in
          if p == 1, let other = spare[spec.name] { return (spec, other) }
          return (spec, buffers[spec.name]!)
        }
        let outs = session.outputs.map { spec -> (TensorSpec, UnsafeMutableRawPointer) in
          if let input = outputOf[spec.name] {
            // writes the buffer the state_ input reads next run
            return (spec, p == 0 ? spare[input]! : buffers[input]!)
          }
          return (spec, buffers[spec.name]!)
        }
        set.append(try OrtBinding(session: session, inputs: ins, outputs: outs))
      }
      sets.append(set)
    }
    bindings = sets
    parity = 0
  }

  public func run() throws {
    guard !closed else { throw OrtError("engine is closed") }
    #if canImport(Metal)
      keepAlive?.pulse()
    #endif
    keepWarm?.pulse()
    let started = DispatchTime.now().uptimeNanoseconds
    do {
      for binding in bindings[parity] {
        try binding.run()
      }
    } catch {
      #if canImport(Metal)
        keepAlive?.pause()
      #endif
      throw error
    }
    if !looped.isEmpty {
      parity ^= 1
    }
    lastGpuUs = UInt32(min(UInt64(UInt32.max), (DispatchTime.now().uptimeNanoseconds - started) / 1000))
  }

  /// CoreML and QNN allocate their working set on the first run and the
  /// second is the steady state.
  public func warm() throws -> String {
    try run()
    try run()
    return "onnxruntime \(OrtRuntime.version) on \(device) in process, sessions \(providers.joined(separator: " then "))"
  }

  public func close() {
    guard !closed else { return }
    closed = true
    #if canImport(Metal)
      keepAlive?.close()
    #endif
    keepWarm?.close()
    release()
  }

  private func release() {
    bindings = []
    for buffer in buffers.values { buffer.deallocate() }
    for buffer in spare.values { buffer.deallocate() }
    buffers = [:]
    spare = [:]
  }
}

extension OrtEngine: Engine {}
