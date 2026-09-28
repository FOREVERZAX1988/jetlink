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
  static let manifestName = "sessions.json"
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

  public init(device: Device = .htp, preparer: any ModelPreparer, keepAlive: Bool = true, keepCPUWarm: Bool = false) {
    self.device = device
    self.preparer = preparer
    self.keepAlive = keepAlive
    self.keepCPUWarm = keepCPUWarm
    self.chip = QNNBackend.chipName()
  }

  /// The SoC's model, "SM8650", which is what a QNN context is valid for.
  /// The Android app reports Build.SOC_MODEL before it starts the server;
  /// "cpu" off Android.
  public static func chipName() -> String {
    chipLock.lock()
    defer { chipLock.unlock() }
    return reportedChip ?? {
      #if os(Android)
        "unknown"
      #else
        "cpu"
      #endif
    }()
  }

  public static func reportChip(_ name: String) {
    chipLock.lock()
    reportedChip = name.isEmpty ? nil : name
    chipLock.unlock()
  }

  private static let chipLock = NSLock()
  nonisolated(unsafe) private static var reportedChip: String?

  public var runtimeVersion: String { OrtRuntime.version }

  public func deviceTag() -> String {
    sanitize("\(device.rawValue)-\(chip)")
  }

  public func tag() -> String {
    "ort\(sanitize(runtimeVersion)).\(deviceTag())"
  }

  public func describe() -> [String: String] {
    ["backend": name, "runtime_version": runtimeVersion, "device": deviceTag()]
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

  func plan(_ model: URL, _ unit: Unit, config: [String: String] = [:]) -> SessionPlan {
    switch unit {
    case .cpu:
      SessionPlan(model: model, provider: nil, threads: QNNBackend.cpuThreads, label: "CPU")
    case .htp, .gpu:
      SessionPlan(model: model, provider: "QNN", options: providerOptions(unit), config: config, label: "QNN(\(unit.rawValue))")
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
    let fm = FileManager.default
    let parent = artifact.deletingLastPathComponent()
    try fm.createDirectory(at: parent, withIntermediateDirectories: true)
    // Staged beside the final path, where the cache's sweep finds it if a
    // build is killed, and renamed into place only once it has run.
    let temp = parent.appending(path: "tmp\(UUID().uuidString.prefix(8))", directoryHint: .isDirectory)
    let staged = temp.appending(path: "artifact", directoryHint: .isDirectory)
    try fm.createDirectory(at: staged, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: temp) }

    report("patch", 0, "preparing the model")
    let sessions = self.sessions
    let prepared = try preparer.prepare(model: model, into: staged, layout: layout) {
      CoreMLPreparation.cacheKey(stem: artifact.deletingPathExtension().lastPathComponent, part: $0)
    }
    log.info("prepared \(model.lastPathComponent): \(prepared.summary)")
    guard prepared.parts.map(\.name) == sessions.map(\.name) else {
      throw HostError.failed("the preparation wrote \(prepared.parts.map(\.name)), expected \(sessions.map(\.name))")
    }
    report("patch", 1, "prepared")

    // Compile each NPU session into its EP context.
    let expect = ArtifactSidecar.read(artifact)
    let took = (expect["compile_seconds"] as? NSNumber)?.doubleValue ?? QNNBackend.expectedCompileSeconds
    var manifest: [[String: Any]] = []
    var contextBytes: Int64 = 0
    let compileStarted = Date()
    for (session, part) in zip(sessions, prepared.parts) {
      guard session.unit == .htp else {
        manifest.append(["model": part.file, "unit": session.unit.rawValue])
        continue
      }
      let source = staged.appending(path: part.file)
      let context = staged.appending(path: "\(session.name)_ctx.onnx")
      report("compile", 0, "compiling \(session.name) for the NPU")
      let ticker = Ticker(interval: 2) { elapsed in
        report(
          "compile", min(0.95, elapsed / took),
          "compiling \(session.name) for the NPU, \(Int(elapsed)) s of about \(Int(took.rounded())) s")
      }
      do {
        let config = [
          "ep.context_enable": "1",
          "ep.context_file_path": context.path,
          // the context binary in a file of its own beside the model, not
          // base64 inside it
          "ep.context_embed_mode": "0",
        ]
        _ = try OrtSession(model: source, provider: "QNN", options: providerOptions(.htp), config: config)
      } catch {
        ticker.stop()
        throw error
      }
      ticker.stop()
      if fm.fileExists(atPath: context.path) {
        // The context carries the compiled graph and any nodes left to the
        // CPU with their weights; the prepared model is dead weight beside it.
        try? fm.removeItem(at: source)
        manifest.append(["model": context.lastPathComponent, "unit": session.unit.rawValue])
      } else {
        log.warning("onnxruntime wrote no EP context for \(session.name); each load will compile it again")
        manifest.append(["model": part.file, "unit": session.unit.rawValue])
      }
    }
    let compileSeconds = Date().timeIntervalSince(compileStarted)
    contextBytes = ArtifactSidecar.treeBytes(staged)
    try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted]).write(to: staged.appending(path: QNNBackend.manifestName))
    report("compile", 1, "compiled in \(Int(compileSeconds.rounded())) s")

    // Prove it runs before calling it built.
    report("load", 0, "loading the compiled model")
    let engine = try OrtEngine(plans: plans(staged, manifest), device: deviceTag(), keepAlive: false, keepCPUWarm: false)
    try engine.run()
    let providers = engine.providers
    engine.close()

    if fm.fileExists(atPath: artifact.path) {
      try fm.removeItem(at: artifact)
    }
    try fm.moveItem(at: staged, to: artifact)

    var meta: [String: Any] = [
      "backend": name,
      "onnxruntime": runtimeVersion,
      "device": deviceTag(),
      "sessions": manifest,
      "providers": providers,
      "build_seconds": pythonRound(Date().timeIntervalSince(started), 1),
      "onnx": model.lastPathComponent,
      "prepare": QNNBackend.prepareVersion,
      "preparer": "swift",
      "built_at": ISO8601DateFormatter().string(from: Date()),
      "compile_seconds": pythonRound(compileSeconds, 1),
      "artifact_bytes": contextBytes,
    ]
    for (key, value) in metaExtra { meta[key] = value }
    try ArtifactSidecar.write(artifact, meta)
    report("build", 1, "done in \(meta["build_seconds"]!)s")
  }

  // MARK: load

  public func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    let manifestURL = artifact.appending(path: QNNBackend.manifestName)
    guard let data = try? Data(contentsOf: manifestURL),
      let manifest = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]], !manifest.isEmpty
    else {
      throw ArtifactInvalid("\(artifact.lastPathComponent): no readable \(QNNBackend.manifestName) inside")
    }
    let meta = ArtifactSidecar.read(artifact)
    let version = (meta["prepare"] as? NSNumber)?.intValue ?? 1
    if version != QNNBackend.prepareVersion {
      throw ArtifactInvalid(
        "\(artifact.lastPathComponent): prepared as version \(version), builds are now at \(QNNBackend.prepareVersion); rebuilding")
    }
    for entry in manifest {
      guard let model = entry["model"] as? String, FileManager.default.fileExists(atPath: artifact.appending(path: model).path),
        (entry["unit"] as? String).flatMap(Unit.init(rawValue:)) != nil
      else {
        throw ArtifactInvalid("\(artifact.lastPathComponent): a session's model is missing")
      }
    }
    let started = Date()
    let took = (meta["load_seconds"] as? NSNumber)?.doubleValue ?? 0
    report("load", 0, "loading the model")
    let ticker = Ticker(interval: 1) { elapsed in
      if took > 0 {
        report("load", min(0.95, elapsed / took), "loading the model, \(Int(elapsed)) s of about \(Int(took.rounded())) s")
      } else {
        report("load", 0, "loading the model, \(Int(elapsed)) s elapsed")
      }
    }
    defer { ticker.stop() }
    let engine = try OrtEngine(plans: plans(artifact, manifest), device: deviceTag(), keepAlive: keepAlive, keepCPUWarm: keepCPUWarm)
    let seconds = Date().timeIntervalSince(started)
    log.info("onnxruntime sessions on \(device.rawValue) in \(String(format: "%.1f", seconds)) s: \(engine.providers.joined(separator: " then "))")
    report("load", 1, "loaded in \(Int(seconds.rounded())) s")
    if !meta.isEmpty {
      var updated = meta
      updated["load_seconds"] = pythonRound(seconds, 1)
      try? ArtifactSidecar.write(artifact, updated)
    }
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
