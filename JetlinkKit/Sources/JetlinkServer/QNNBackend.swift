import Foundation
import JetlinkKit
import JetlinkONNX

/// onnxruntime's QNN provider in process, the Android backend: Qualcomm's
/// Hexagon NPU ("HTP") and Adreno GPU on a Snapdragon phone. The Android
/// counterpart of `CoreMLBackend`, on the same preparation.
///
/// `htp` is the Mac's `ane` split: the vision trunk on the NPU in fp16, the
/// policy and heads after it on the GPU. `htp-whole` is the iPhone's
/// `ane-whole` layout, the whole graph on the NPU with the policy's norms
/// prescaled; the NPU computes in fp16 whatever the graph says, so its fp32
/// heads are fp16 there too, and only the parity check can say whether that
/// is close enough.
///
/// The artifact is a directory like CoreML's: a `sessions.json` manifest and
/// each session's model. An NPU session is compiled once, at build, into
/// onnxruntime's EP context (`<name>_ctx.onnx` and the QNN context binary
/// beside it), which then replaces its prepared ONNX: a load reads the
/// compiled graph and does not finalize it again. A GPU or CPU session keeps
/// its prepared ONNX.
///
/// Nothing here has run on a Snapdragon yet: the provider options follow
/// onnxruntime 1.29's QNN documentation and source, and the `cpu` device is
/// what the emulator and the tests run.
public final class QNNBackend: EngineBackend {
  public enum Device: String, Sendable, CaseIterable {
    /// The vision trunk on the NPU, everything after it on the GPU.
    case htp
    /// The whole graph on the NPU, prepared as the iPhone's `ane-whole`.
    case htpWhole = "htp-whole"
    /// The whole graph on the Adreno GPU.
    case gpu
    /// onnxruntime's CPU provider and no QNN at all: for the emulator and
    /// tests, and nowhere near the frame budget with a real model.
    case cpu
  }

  /// What runs one session.
  enum Unit: String {
    case htp, gpu, cpu
  }

  /// The same preparation as CoreML's, so the same version.
  public static let prepareVersion = Pinned.prepareVersion
  /// A first NPU compile with no earlier build to go by. The QNN graph
  /// finalization of a big model is minutes on a phone (unmeasured).
  static let expectedCompileSeconds = 180.0

  public let name = "ort"
  public let suffix = ".ortcache"
  public let device: Device
  /// Hold the NPU in burst mode between frames rather than let it settle.
  public let keepAlive: Bool
  public let keepCPUWarm: Bool
  private let preparer: any ModelPreparer
  private let chip: String
  private let log = ServerLog(category: "qnn")

  /// `chip` is the SoC's model, "SM8650" (Build.SOC_MODEL), which is what a
  /// QNN context is valid for; the Android app passes it.
  public init(device: Device = .htp, preparer: any ModelPreparer, keepAlive: Bool = true, keepCPUWarm: Bool = false, chip: String = "cpu") {
    self.device = device
    self.preparer = preparer
    self.keepAlive = keepAlive
    self.keepCPUWarm = keepCPUWarm
    self.chip = chip.isEmpty ? "unknown" : chip
  }

  public var runtimeVersion: String { OrtRuntime.version }

  public func deviceTag() -> String {
    sanitize("\(device.rawValue)-\(chip)")
  }

  /// (session name, unit) in run order.
  var sessions: [(name: String, unit: Unit)] {
    switch device {
    case .htp: [("vision", .htp), ("policy", .gpu)]
    case .htpWhole: [("model", .htp)]
    case .gpu: [("model", .gpu)]
    case .cpu: [("model", .cpu)]
    }
  }

  var layout: CoreMLPreparation.Layout {
    switch device {
    case .htp: .split
    case .htpWhole: .aneWhole
    case .gpu, .cpu: .whole
    }
  }

  /// The QNN provider's options for one unit, as onnxruntime 1.29 names them.
  func providerOptions(_ unit: Unit) -> [String: String] {
    switch unit {
    case .htp:
      [
        "backend_type": "htp",
        // burst holds the clocks up between frames, the counterpart of the
        // Mac's GPU keep-alive; sustained trades speed for less heat.
        "htp_performance_mode": keepAlive ? "burst" : "sustained_high_performance",
        // the slowest finalization and the fastest graph: it runs once, at build
        "htp_graph_finalization_optimization_mode": "3",
        "enable_htp_fp16_precision": "1",
        "qnn_context_priority": "high",
      ]
    case .gpu:
      ["backend_type": "gpu"]
    case .cpu:
      [:]
    }
  }

  func plan(_ model: URL, _ unit: Unit) -> SessionPlan {
    switch unit {
    case .cpu:
      SessionPlan(model: model, provider: nil, threads: QNNBackend.cpuThreads, label: "CPU")
    case .htp, .gpu:
      SessionPlan(model: model, provider: "QNN", options: providerOptions(unit), label: "QNN(\(unit.rawValue))")
    }
  }

  /// The CPU provider's pool when the CPU runs the whole model: half the
  /// cores, leaving the rest to the link and the app.
  static var cpuThreads: Int { max(1, ProcessInfo.processInfo.activeProcessorCount / 2) }

  public func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try preparer.readSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  // MARK: build

  public func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    let started = Date()
    let expect = ArtifactSidecar.read(artifact)
    var meta = try OrtArtifact.build(artifact) { staged in
      report("patch", 0, "preparing the model")
      let prepared = try preparer.prepare(model: model, into: staged, layout: layout) {
        CoreMLPreparation.cacheKey(stem: artifact.deletingPathExtension().lastPathComponent, part: $0)
      }
      log.info("prepared \(model.lastPathComponent): \(prepared.summary)")
      guard prepared.parts.map(\.name) == sessions.map(\.name) else {
        throw HostError.failed("the preparation wrote \(prepared.parts.map(\.name)), expected \(sessions.map(\.name))")
      }
      report("patch", 1, "prepared")
      let (manifest, compileSeconds) = try compile(prepared, in: staged, expect: expect, report: report)
      try OrtArtifact.writeManifest(manifest, in: staged)

      // Prove it runs before calling it built: minutes on a CPU, so it ticks.
      let providers = try Ticker.during(interval: 1, { report("load", 0, "loading the model to check it runs, \(Int($0)) s elapsed") }) {
        let engine = try OrtEngine(plans: plans(staged, manifest), device: deviceTag(), keepAlive: false, keepCPUWarm: false)
        defer { engine.close() }
        try engine.run()
        return engine.providers
      }
      var meta = OrtArtifact.meta(
        self, manifest: manifest, providers: providers, model: model, prepareVersion: QNNBackend.prepareVersion, started: started)
      meta["compile_seconds"] = pythonRound(compileSeconds, 1)
      meta["artifact_bytes"] = ArtifactSidecar.treeBytes(staged)
      return meta
    }
    for (key, value) in metaExtra { meta[key] = value }
    try ArtifactSidecar.write(artifact, meta)
    report("build", 1, "done in \(meta["build_seconds"]!)s")
  }

  /// Compiles each NPU session into its EP context, which then replaces its
  /// prepared model: the context carries the compiled graph and any nodes
  /// left to the CPU with their weights. Returns the manifest, what each
  /// session loads, and how long the compiles took.
  private func compile(_ prepared: PreparedModel, in staged: URL, expect: [String: Any], report: @escaping ProgressFn) throws -> (
    manifest: [[String: Any]], seconds: TimeInterval
  ) {
    let took = (expect["compile_seconds"] as? NSNumber)?.doubleValue ?? QNNBackend.expectedCompileSeconds
    let started = Date()
    var manifest: [[String: Any]] = []
    for (session, part) in zip(sessions, prepared.parts) {
      var file = part.file
      if session.unit == .htp {
        let source = staged.appending(path: part.file)
        let context = staged.appending(path: "\(session.name)_ctx.onnx")
        report("compile", 0, "compiling \(session.name) for the NPU")
        try Ticker.during(interval: 2, { elapsed in
          report("compile", min(0.95, elapsed / took), "compiling \(session.name) for the NPU, \(Int(elapsed)) s of about \(Int(took.rounded())) s")
        }) {
          let config = [
            "ep.context_enable": "1",
            "ep.context_file_path": context.path,
            // the context binary in a file of its own beside the model, not base64 inside it
            "ep.context_embed_mode": "0",
          ]
          _ = try OrtSession(model: source, provider: "QNN", options: providerOptions(.htp), config: config)
        }
        if FileManager.default.fileExists(atPath: context.path) {
          try? FileManager.default.removeItem(at: source)
          file = context.lastPathComponent
        } else {
          log.warning("onnxruntime wrote no EP context for \(session.name); each load will compile it again")
        }
      }
      manifest.append(["model": file, "unit": session.unit.rawValue])
    }
    let seconds = Date().timeIntervalSince(started)
    if sessions.contains(where: { $0.unit == .htp }) {
      report("compile", 1, "compiled in \(Int(seconds.rounded())) s")
    }
    return (manifest, seconds)
  }

  // MARK: load

  public func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    let (manifest, meta) = try OrtArtifact.open(artifact, prepareVersion: QNNBackend.prepareVersion, builds: "QNN") { entry in
      (entry["unit"] as? String).flatMap(Unit.init(rawValue:)) == nil ? "a session names no unit" : nil
    }
    let (engine, seconds) = try OrtArtifact.load(artifact, meta: meta, what: "the model", report: report) {
      try OrtEngine(plans: plans(artifact, manifest), device: deviceTag(), keepAlive: keepAlive, keepCPUWarm: keepCPUWarm)
    }
    log.info("onnxruntime sessions on \(device.rawValue) in \(String(format: "%.1f", seconds)) s: \(engine.providers.joined(separator: " then "))")
    return engine
  }

  /// The manifest's sessions as plans, with today's provider options: the
  /// performance mode is a setting, not part of the artifact.
  private func plans(_ artifact: URL, _ manifest: [[String: Any]]) -> [SessionPlan] {
    manifest.map { entry in
      plan(artifact.appending(path: entry["model"] as! String), Unit(rawValue: entry["unit"] as! String)!)
    }
  }
}
