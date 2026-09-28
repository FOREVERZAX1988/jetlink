import Foundation
import JetlinkKit
import JetlinkONNX
import Metal

/// onnxruntime's CoreML provider in process: the Swift form of the Python ort
/// backend on `--device ane` (the vision trunk on the Neural Engine, the rest
/// on the GPU) or `--device coreml` (the whole graph on the GPU).
///
/// The artifact is the directory the Python backend writes: the prepared
/// ONNX, a `sessions.json` manifest naming the sessions, and onnxruntime's
/// CoreML cache per session. A load needs nothing else on disk.
public final class CoreMLBackend: EngineBackend {
  public enum Device: String, Sendable {
    /// The trunk on the Neural Engine, everything after it on the GPU.
    case ane
    /// The whole graph on the GPU, for when something else holds the Neural Engine.
    case coreml
    /// The whole graph as one CoreML program with every compute unit allowed,
    /// prepared for the Neural Engine (the policy's norms prescaled, the vision
    /// heads in fp32): what a phone, whose GPU is far weaker than its Neural
    /// Engine, wants.
    case aneWhole = "ane-whole"
    /// onnxruntime's CPU provider and no CoreML at all: for tests, and nowhere
    /// near the frame budget with a real model.
    case cpu
  }

  /// What a CoreML build writes, the Python's PREPARE_VERSION: 5 is every
  /// graph split on `ane` and Expand as Tile on both. `ane-whole` is a layout
  /// under a device tag of its own, so adding it changed no artifact and took
  /// no bump. The same number on both sides is what lets one Mac cache serve
  /// both servers; `Pinned` carries it from the Python.
  public static let prepareVersion = Pinned.prepareVersion

  public let name = "ort"
  public let suffix = ".ortcache"
  public let device: Device
  public let keepAlive: Bool
  public let keepCPUWarm: Bool
  private let preparer: any ModelPreparer
  private let chip: String
  private let log = ServerLog(category: "coreml")

  public init(device: Device = .ane, preparer: any ModelPreparer, keepAlive: Bool = true, keepCPUWarm: Bool = true) {
    self.device = device
    self.preparer = preparer
    self.keepAlive = keepAlive
    self.keepCPUWarm = keepCPUWarm
    self.chip = CoreMLBackend.chipName()
  }

  /// The SoC's name, which is the GPU: "Apple M1 Pro", "Apple A17 Pro". On a
  /// Mac the CPU brand string, as the Python's gpu_name reads it, so the two
  /// agree on a cache key.
  static func chipName() -> String {
    #if os(macOS)
      var size = 0
      if sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 1 {
        var bytes = [CChar](repeating: 0, count: size)
        if sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 {
          return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
      }
    #endif
    let name = MTLCreateSystemDefaultDevice()?.name ?? "unknown"
    return name.hasSuffix(" GPU") ? String(name.dropLast(4)) : name
  }

  public var runtimeVersion: String { OrtRuntime.version }

  public func deviceTag() -> String {
    sanitize("\(device.rawValue)-\(chip)")
  }

  /// (session name, compute units) in run order.
  var sessions: [(name: String, units: String?)] {
    switch device {
    case .ane: [("vision", "CPUAndNeuralEngine"), ("policy", "CPUAndGPU")]
    case .coreml: [("model", "CPUAndGPU")]
    case .aneWhole: [("model", "ALL")]
    case .cpu: [("model", nil)]
    }
  }

  /// How the preparation lays the graph out for the device's sessions.
  var layout: CoreMLPreparation.Layout {
    switch device {
    case .ane: .split
    case .coreml, .cpu: .whole
    case .aneWhole: .aneWhole
    }
  }

  /// The Python backend's key for one session's compiled model: onnxruntime
  /// wants it alphanumeric and under 64 characters.
  static func cacheKey(artifact: URL, part: String) -> String {
    CoreMLPreparation.cacheKey(stem: artifact.deletingPathExtension().lastPathComponent, part: part)
  }

  public func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try preparer.readSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  // MARK: build

  public func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    let started = Date()
    let expect = ArtifactSidecar.read(artifact)
    var meta = try OrtArtifact.build(artifact) { staged in
      report("patch", 0, "preparing the model for CoreML")
      let prepared = try preparer.prepare(model: model, into: staged, layout: layout) {
        CoreMLBackend.cacheKey(artifact: artifact, part: $0)
      }
      log.info("prepared \(model.lastPathComponent): \(prepared.summary)")
      guard prepared.parts.map(\.name) == sessions.map(\.name) else {
        throw HostError.failed("the preparation wrote \(prepared.parts.map(\.name)), expected \(sessions.map(\.name))")
      }
      var manifest: [[String: Any]] = []
      for (session, part) in zip(sessions, prepared.parts) {
        guard let units = session.units else {
          manifest.append(["model": part.file, "units": NSNull(), "cache": NSNull()])
          continue
        }
        let cache = "coreml-\(session.name)"
        try FileManager.default.createDirectory(at: staged.appending(path: cache, directoryHint: .isDirectory), withIntermediateDirectories: true)
        manifest.append(["model": part.file, "units": units, "cache": cache])
      }
      try OrtArtifact.writeManifest(manifest, in: staged)
      report("patch", 1, "prepared")

      let weights = prepared.parts.reduce(0) { $0 + $1.weightBytes }
      let caches = manifest.compactMap { ($0["cache"] as? String).map { staged.appending(path: $0, directoryHint: .isDirectory) } }
      let progress = CoreMLProgress(caches: caches, weightBytes: weights, expect: expect)
      report("convert", 0, "converting for CoreML")
      let engineStarted = Date()
      let engine = try Ticker.during(interval: 2, { elapsed in
        let (stage, frac, msg) = progress.tick(elapsed: elapsed)
        report(stage, frac, msg)
      }) {
        try OrtEngine(plans: plans(staged, manifest), device: deviceTag(), keepAlive: false, keepCPUWarm: false)
      }
      defer { engine.close() }
      let (converted, compiled) = progress.measure()
      let compileSeconds = Date().timeIntervalSince(engineStarted)
      report("convert", 1, "converted \(formatBytes(converted))")
      report("compile", 1, "compiled in \(Int(compileSeconds.rounded())) s")
      // Prove it runs before calling it built.
      try engine.run()
      engine.close()
      // A load reads only the compiled model, so the MLProgram onnxruntime
      // converted each session to is dead weight beside it: on an M1 Pro with
      // 1.29.0 the artifact went from 2.2 GB to 1.4 GB, loading in 0.5 s with
      // outputs bit for bit the same. A phone has no room for both.
      let freed = CoreMLBackend.dropConvertedModels(caches)
      if freed > 0 {
        log.info("removed \(formatBytes(freed)) of converted model the compiled one replaces")
      }
      var meta = OrtArtifact.meta(
        self, manifest: manifest, providers: engine.providers, model: model, prepareVersion: CoreMLBackend.prepareVersion, started: started)
      meta["convert_bytes"] = converted
      meta["compile_bytes"] = compiled
      meta["compile_seconds"] = pythonRound(compileSeconds, 1)
      meta["freed_bytes"] = freed
      return meta
    }
    for (key, value) in metaExtra { meta[key] = value }
    try ArtifactSidecar.write(artifact, meta)
    report("build", 1, "done in \(meta["build_seconds"]!)s")
  }

  /// Deletes each session's MLProgram `Data` directory once CoreML's
  /// compile of it exists beside it. Returns the bytes freed.
  static func dropConvertedModels(_ caches: [URL]) -> Int64 {
    let fm = FileManager.default
    var freed: Int64 = 0
    for cache in caches {
      guard let walker = fm.enumerator(at: cache, includingPropertiesForKeys: [.isDirectoryKey]) else { continue }
      var converted: [URL] = []
      for case let url as URL in walker where url.lastPathComponent == "Data" {
        guard (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
        let compiled = url.deletingLastPathComponent().appending(path: CoreMLProgress.compiledDir, directoryHint: .isDirectory)
        if fm.fileExists(atPath: compiled.path) {
          converted.append(url)
        }
        walker.skipDescendants()
      }
      for url in converted {
        freed += ArtifactSidecar.treeBytes(url)
        try? fm.removeItem(at: url)
      }
    }
    return freed
  }

  // MARK: load

  public func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    let (manifest, meta) = try OrtArtifact.open(artifact, prepareVersion: CoreMLBackend.prepareVersion, builds: "CoreML") { entry in
      // Without its compile, onnxruntime would recompile under a "loading"
      // that never moves. Rebuild instead, which reports progress. Only the
      // compile is looked for: the converted MLProgram beside it is dropped
      // after the build (dropConvertedModels).
      guard let cache = entry["cache"] as? String else { return nil }
      let contents = (try? FileManager.default.contentsOfDirectory(atPath: artifact.appending(path: cache, directoryHint: .isDirectory).path)) ?? []
      return contents.isEmpty ? "the CoreML cache for \(entry["model"] as? String ?? cache) is empty" : nil
    }
    let (engine, seconds) = try OrtArtifact.load(artifact, meta: meta, what: "the CoreML model", report: report) {
      try OrtEngine(plans: plans(artifact, manifest), device: deviceTag(), keepAlive: keepAlive, keepCPUWarm: keepCPUWarm)
    }
    log.info("onnxruntime sessions on \(device.rawValue) in \(String(format: "%.1f", seconds)) s")
    return engine
  }

  private func plans(_ artifact: URL, _ manifest: [[String: Any]]) -> [SessionPlan] {
    manifest.map { entry in
      SessionPlan(
        model: artifact.appending(path: entry["model"] as! String),
        computeUnits: entry["units"] as? String,
        cacheDirectory: (entry["cache"] as? String).map { artifact.appending(path: $0, directoryHint: .isDirectory) })
    }
  }
}

/// What a CoreML session creation is doing, read off its cache directory:
/// onnxruntime writes the converted MLProgram first, then CoreML compiles it
/// into `compiled_model.mlmodelc`. Which one is growing is the stage.
final class CoreMLProgress: @unchecked Sendable {
  static let compiledDir = "compiled_model.mlmodelc"
  /// The last resort for the compile's fraction, only on a first build.
  static let expectedCompileSeconds = 10.0

  let caches: [URL]
  let weightBytes: Int64
  let expect: [String: Any]

  init(caches: [URL], weightBytes: Int64, expect: [String: Any]) {
    self.caches = caches
    self.weightBytes = weightBytes
    self.expect = expect
  }

  func measure() -> (converted: Int64, compiled: Int64) {
    var converted: Int64 = 0
    var compiled: Int64 = 0
    for cache in caches {
      guard let walker = FileManager.default.enumerator(at: cache, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { continue }
      for case let file as URL in walker {
        guard let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), values.isRegularFile == true else { continue }
        let size = Int64(values.fileSize ?? 0)
        if file.pathComponents.contains(CoreMLProgress.compiledDir) {
          compiled += size
        } else {
          converted += size
        }
      }
    }
    return (converted, compiled)
  }

  func tick(elapsed: TimeInterval) -> (String, Double, String) {
    let (converted, compiled) = measure()
    if compiled == 0 {
      let total = weightBytes > 0 ? weightBytes : ((expect["convert_bytes"] as? NSNumber)?.int64Value ?? 0)
      let frac = total > 0 ? min(0.95, Double(converted) / Double(total)) : 0
      let of = total > 0 ? " of \(formatBytes(total))" : ""
      return ("convert", frac, "converting for CoreML, \(formatBytes(converted))\(of) written")
    }
    if let total = (expect["compile_bytes"] as? NSNumber)?.int64Value, total > 0 {
      return ("compile", min(0.95, Double(compiled) / Double(total)), "compiling for CoreML, \(formatBytes(compiled)) of \(formatBytes(total)) written")
    }
    let took = (expect["compile_seconds"] as? NSNumber)?.doubleValue ?? CoreMLProgress.expectedCompileSeconds
    return ("compile", min(0.95, elapsed / took), "compiling for CoreML, \(formatBytes(compiled)) written, \(Int(elapsed)) s elapsed")
  }
}
