import Foundation
import JetlinkKit
import JetlinkONNX
import JetlinkRegistry
import JetlinkServer

/// Where LiteRT runs a model. The raw value is the device the Android app's
/// start config and `--device` name beside the backend, "litert", and the
/// device part of the artifact's tag.
public enum LiteRtProfile: String, CaseIterable, Sendable {
  /// The whole graph on the GPU in fp16: OpenCL on Android, Metal on a Mac.
  /// A model the GPU cannot run whole fails to build rather than run partly
  /// on the CPU at a fraction of the speed.
  case gpu
  /// XNNPACK on the CPU: for tests, the emulator, and phones LiteRT drives no
  /// GPU on. Nowhere near the frame budget with a real model.
  case cpu

  /// What the log and the benchmark's report call it.
  var label: String {
    switch self {
    case .gpu: "GPU(fp16)"
    case .cpu: "CPU(\(HostChip.cpuThreads) threads)"
    }
  }

  /// How the profile compiles a model; on the GPU, its programs cached in
  /// `cache` when there is one.
  func options(cache: (directory: URL, key: String)? = nil) -> LiteRtCompileOptions {
    switch self {
    case .gpu: .gpu(cache: cache)
    case .cpu: .cpu(threads: HostChip.cpuThreads)
    }
  }
}

/// LiteRT in process, on the GPU or the CPU: the backend for phones
/// onnxruntime's QNN does not drive. The artifact is a directory holding the
/// converted model and the GPU's compile cache (LiteRtArtifact).
///
/// Nothing here has run on a phone yet: the GPU path is Metal's on a Mac,
/// and Android's OpenCL one follows LiteRT 2.2.0's documentation and source.
public final class LiteRtBackend: EngineBackend {
  /// A first GPU compile with no earlier build to go by: 7 to 12 s for
  /// Cinque Terre V3 on an M1 Pro's Metal; a phone's OpenCL compile is
  /// unmeasured.
  static let expectedCompileSeconds = 60.0
  /// What JetlinkONNX's LiteRTPreparation writes now: an artifact converted
  /// under another version rebuilds.
  static let conversionVersion = 1

  public let name = "litert"
  /// A directory: the converted model and the GPU's compile cache.
  public let suffix = ".litertcache"
  public let profile: LiteRtProfile
  /// Where LiteRT's libraries are; nil for $JETLINK_LITERT_DIR or the
  /// loader's path (`LiteRtRuntime.load`).
  public let libraries: URL?
  private let preparer: any ModelPreparer
  private let chip: String
  private let log = ServerLog(category: "litert")

  /// `libraries` is the directory holding libLiteRt and its GPU
  /// accelerator, the Android app's nativeLibraryDir; nil reads
  /// $JETLINK_LITERT_DIR. `chip` names what the artifacts are valid for: on
  /// Android the SoC's model, "Tensor G5" (Build.SOC_MODEL), which a GPU's
  /// compiled programs are for; nil is HostChip's name for this machine.
  public init(profile: LiteRtProfile, preparer: any ModelPreparer, libraries: URL? = nil, chip: String? = nil) {
    self.profile = profile
    self.preparer = preparer
    self.libraries = libraries
    self.chip = HostChip.resolve(chip)
  }

  /// The GPU accelerator's library, which LiteRT opens from beside its own.
  static var gpuLibrary: String {
    #if os(macOS) || os(iOS)
      "libLiteRtMetalAccelerator.dylib"
    #else
      "libLiteRtClGlAccelerator.so"
    #endif
  }

  /// Opens LiteRT from `libraries`, and for the GPU checks its accelerator
  /// loaded: both are what an app shows when they fail. Whoever makes the
  /// backend calls it, before any build or load: the app as its server
  /// starts, so a phone LiteRT cannot run on says so at once rather than
  /// after converting a model.
  public func open() throws {
    try LiteRtRuntime.load(directory: libraries)
    if profile == .gpu {
      let (gpu, names) = try LiteRtRuntime.accelerators()
      guard gpu else {
        throw LiteRtError(
          "LiteRT's GPU accelerator did not load (\(LiteRtBackend.gpuLibrary) beside libLiteRt), so only the CPU can run LiteRT here; "
            + "it has \(names.isEmpty ? "no accelerators" : names)")
      }
    }
  }

  public var runtimeVersion: String { LiteRtRuntime.version }

  public func deviceTag() -> String {
    sanitize("\(profile.rawValue)-\(chip)")
  }

  public func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try preparer.readSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  /// The model in `directory`, an artifact or its staging, compiled for the
  /// profile. On the GPU every op must be the GPU's: a model that runs
  /// partly on the CPU is a build that failed.
  func engine(_ directory: URL, cacheKey: String) throws -> LiteRtEngine {
    let engine: LiteRtEngine
    do {
      engine = try LiteRtEngine(
        model: directory.appending(path: LiteRtArtifact.model),
        options: profile.options(cache: (directory.appending(path: LiteRtArtifact.cache, directoryHint: .isDirectory), cacheKey)),
        device: deviceTag(), label: profile.label)
    } catch let error as LiteRtError where profile == .gpu {
      #if os(Android)
        if !LiteRtRuntime.hasOpenCL {
          throw LiteRtError(
            "no OpenCL: LiteRT's GPU runs on OpenCL here, and \(chip) gives apps none of "
              + "\(LiteRtRuntime.openCLLibraries.joined(separator: ", ")) (\(error.description)); the CPU profile runs every op")
        }
      #endif
      // A GPU-only compile fails outright on an op the GPU cannot run.
      throw LiteRtError(
        "LiteRT could not compile the model for the GPU on \(chip) (\(error.description)); LiteRT's log names any op the GPU cannot run, "
          + "and the CPU profile runs every op")
    }
    if profile == .gpu && !engine.fullyAccelerated {
      engine.close()
      throw LiteRtError("LiteRT's GPU on \(chip) cannot run every op of this model; LiteRT's log names them, and the CPU profile runs every op")
    }
    return engine
  }

  // MARK: build

  public func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    let started = Date()
    let expect = Artifact.sidecar(artifact)
    try Artifact.build(artifact, metaExtra: metaExtra, report: report) { staged in
      try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: true)
      report("convert", 0, "converting the model for LiteRT")
      let convertStarted = Date()
      let convertTook = (expect["convert_seconds"] as? NSNumber)?.doubleValue ?? 0
      let converted = try Ticker.during(interval: 1, Ticker.paced("convert", "converting the model for LiteRT", took: convertTook, report: report)) {
        try LiteRTPreparation.prepare(source: model, into: staged)
      }
      let convertSeconds = Date().timeIntervalSince(convertStarted)
      log.info("converted \(model.lastPathComponent): \(converted.summary)")
      report("convert", 1, "converted in \(Int(convertSeconds.rounded())) s")

      guard converted.url.lastPathComponent == LiteRtArtifact.model else {
        throw HostError.failed("the conversion wrote \(converted.url.lastPathComponent), expected \(LiteRtArtifact.model)")
      }
      try FileManager.default.createDirectory(
        at: staged.appending(path: LiteRtArtifact.cache, directoryHint: .isDirectory), withIntermediateDirectories: true)

      // One compile on the profile's accelerator proves the model runs there
      // whole, and leaves the GPU's cache for every load after. No run: the
      // compile writes the cache (Metal's whole-graph one appears before any
      // run), and the host's warm-up runs the loaded model anyway.
      let what = profile == .gpu ? "compiling for the GPU" : "loading the model to check it runs"
      let took = (expect["compile_seconds"] as? NSNumber)?.doubleValue ?? (profile == .gpu ? LiteRtBackend.expectedCompileSeconds : 0)
      report("compile", 0, what)
      let compileStarted = Date()
      try Ticker.during(interval: 1, Ticker.paced("compile", what, took: took, report: report)) {
        try self.engine(staged, cacheKey: LiteRtArtifact.cacheKey(artifact)).close()
      }
      let compileSeconds = Date().timeIntervalSince(compileStarted)
      report("compile", 1, "compiled in \(Int(compileSeconds.rounded())) s")
      var meta = LiteRtArtifact.meta(self, model: model, started: started)
      meta["convert_seconds"] = pythonRound(convertSeconds, 1)
      meta["compile_seconds"] = pythonRound(compileSeconds, 1)
      meta["artifact_bytes"] = Files.size(of: staged)
      return meta
    }
  }

  // MARK: load

  public func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    let meta = try LiteRtArtifact.open(artifact)
    let (engine, seconds) = try Artifact.load(artifact, meta: meta, what: "the model", report: report) {
      try self.engine(artifact, cacheKey: LiteRtArtifact.cacheKey(artifact))
    }
    log.info("LiteRT on \(profile.rawValue) in \(String(format: "%.1f", seconds)) s: \(engine.label)")
    return engine
  }
}
