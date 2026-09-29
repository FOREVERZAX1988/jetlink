import CTrt
import Foundation
import JetlinkONNX
import JetlinkServer

#if canImport(Android)
  import Android
#endif

/// TensorRT on an NVIDIA GPU: the Jetson's backend, and a PC's. The Swift
/// form of the Python server's trt/__init__.py and trt/build.py.
///
/// Artifacts are plans, `engines/<sha16>.<tag>.plan`, with the tag Python
/// wrote (`trt<version>.<device>-sm<cc>`) so the plans already on a Jetson
/// load without a rebuild.
///
/// Two TensorRT generations build from this one file, as in Python. JetPack
/// ships 10.x, and the car was validated on its weakly typed network with the
/// FP16 flag. PCs run 11.x, which dropped weak typing and the precision flags
/// with it: precision follows the ONNX, which is fp16 end to end.
///
/// Two environment variables, for the Jetson acceptance runs (plan section
/// 7): JETLINK_TRT_GPU_TIMING=1 times every launch with CUDA events and logs
/// the spread (the engine's `gpuTiming`), and JETLINK_FAULT_CUDA_AFTER=N makes
/// every frame after the first N throw a sticky CUDA error, to exercise the
/// fatal exit and restart (H6, D15).
public final class TrtBackend: EngineBackend {
  public let name = "trt"
  public let suffix = ".plan"
  public let artifactKind = ArtifactKind.file
  public let trt: TensorRT
  let gpuTiming: Bool
  let faultAfter: Int?
  /// MemAvailable, which the workspace is sized from.
  let available: @Sendable () -> Int
  private let log = ServerLog(category: "builder")

  /// A ceiling, not an allocation: TensorRT picks tactics that fit inside
  /// it. A flat 4 GB on an 8 GB Orin met the OOM killer on a 1.76 GB model,
  /// so it is sized from what is free.
  static let maxWorkspace = 4 << 30
  static let minWorkspace = 256 << 20
  static let workspaceFraction = 0.4
  static let optimizationLevel = 3

  /// TensorRT on CUDA device `device`, or `TensorRTUnavailable`.
  public convenience init(device: Int = 0) throws {
    let env = ProcessInfo.processInfo.environment
    self.init(
      trt: try TensorRT(device: device), gpuTiming: env["JETLINK_TRT_GPU_TIMING"] == "1",
      faultAfter: env["JETLINK_FAULT_CUDA_AFTER"].flatMap { Int($0) })
  }

  package init(
    trt: TensorRT, gpuTiming: Bool = false, faultAfter: Int? = nil, available: @escaping @Sendable () -> Int = TrtBackend.memAvailable
  ) {
    self.trt = trt
    self.gpuTiming = gpuTiming
    self.faultAfter = faultAfter
    self.available = available
  }

  public var runtimeVersion: String { trt.version }

  /// The hardware a plan is valid for. The compute capability is what
  /// matters; the name makes the file readable.
  public func deviceTag() -> String {
    sanitize("\(trt.deviceName)-sm\(trt.computeCapability.major)\(trt.computeCapability.minor)")
  }

  /// The comma logs it.
  public var helloFields: [String: Any] { ["trt_version": trt.fullVersion] }

  /// `engines/timing.<tag>.cache`: kernel timings mostly do not depend on
  /// the model (a warm cache cut a Lebowski build from 254 s to 173 s), but
  /// a timing from another TensorRT build or chip is not one.
  func timingCache(beside artifact: URL) -> URL {
    artifact.deletingLastPathComponent().appending(path: "timing.\(tag()).cache")
  }

  public func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try ONNXPreparer().readSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  static func workspaceBytes(available: Int) -> Int {
    guard available > 0 else { return maxWorkspace }
    return max(minWorkspace, min(maxWorkspace, Int(Double(available) * workspaceFraction)))
  }

  /// MemAvailable, free plus what the kernel would reclaim, or 0 for "no
  /// idea, use the cap". Swap does not count: on Tegra the GPU's memory is
  /// pinned system RAM that cannot page out.
  static func memAvailable() -> Int {
    #if os(Linux)
      guard let text = try? String(contentsOfFile: "/proc/meminfo", encoding: .utf8) else { return 0 }
      for line in text.split(separator: "\n") where line.hasPrefix("MemAvailable:") {
        return (Int(line.split(separator: " ").dropFirst().first ?? "") ?? 0) * 1024
      }
      return 0
    #else
      return 0
    #endif
  }

  // MARK: build

  /// Patch, parse and build, with Python's progress stages and messages;
  /// the plan is staged and moved into place, the sidecar beside it.
  public func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    let started = Date()
    let free = available()
    let workspace = Self.workspaceBytes(available: free)
    log.info("building with a \(workspace >> 20) MB workspace (\(free >> 20) MB available)")
    let cacheURL = timingCache(beside: artifact)
    try Artifact.build(artifact, kind: artifactKind, metaExtra: metaExtra, report: report) { staged in
      report("patch", 0, "retyping uint8 image inputs to fp16")
      let prepared = try CoreMLPreparation.prepare(source: model, into: staged.deletingLastPathComponent(), layout: .trt, cacheKey: { _ in "" })
      report("patch", 1, "patched")

      var build: OpaquePointer?
      try trt.check { jl_trt_build_create(trt.handle, &build, $0, $1) }
      defer { jl_trt_build_destroy(build) }
      report("parse", 0, "parsing onnx")
      do {
        try trt.check(capacity: 16 << 10) { jl_trt_build_parse(build, prepared.parts[0].url.path, $0, $1) }
      } catch let error as TrtError where !error.isFatal {
        throw TrtError(code: error.code, "onnx parse failed:\n\(error.description)")
      }
      report("parse", 1, "\(jl_trt_build_layers(build)) layers")

      let precision: String
      if trt.stronglyTyped {
        precision = "strongly typed network; precision follows the ONNX"
      } else {
        try trt.check { jl_trt_build_set_fp16(build, $0, $1) }
        precision = "fp16 enabled"
      }
      log.info("tensorrt \(runtimeVersion): \(precision)")
      jl_trt_build_set_optimization_level(build, Int32(Self.optimizationLevel))
      jl_trt_build_set_workspace(build, workspace)
      let monitor = BuildMonitor(report)
      let cached = attachTimingCache(build!, cacheURL)

      report("build", 0, "building engine")
      try withExtendedLifetime(monitor) {
        jl_trt_build_set_progress(build, forwardProgress, Unmanaged.passUnretained(monitor).toOpaque())
        defer { jl_trt_build_set_progress(build, nil, nil) }
        do {
          try trt.check { jl_trt_build_write_plan(build, staged.path, $0, $1) }
        } catch let error as TrtError where !error.isFatal && !error.description.hasPrefix("cannot write") {
          log.error("\(error.description)")
          throw TrtError(code: error.code, "TensorRT returned no engine; see the build log")
        }
      }
      if cached {
        saveTimingCache(build!, cacheURL)
      }

      var meta = Artifact.meta(self, runtimeKey: "trt_version", model: model, started: started)
      meta["fp16"] = true
      meta["strongly_typed"] = trt.stronglyTyped
      meta["optimization_level"] = Self.optimizationLevel
      return meta
    }
  }

  /// Seeds the tactic timings from the last build. Fails open: another
  /// build's cache or a killed build's truncated one is replaced by an empty
  /// one, which this build fills and saves. (Python kept it detached, so
  /// every later build stayed cold.) False when not even that attaches.
  private func attachTimingCache(_ build: OpaquePointer, _ url: URL) -> Bool {
    let data = (try? Data(contentsOf: url)) ?? Data()
    do {
      try data.withUnsafeBytes { bytes in
        try trt.check { jl_trt_build_set_timing_cache(build, bytes.baseAddress, bytes.count, $0, $1) }
      }
      return true
    } catch {
      log.warning("timing cache unusable (\(error)), building cold")
    }
    do {
      try trt.check { jl_trt_build_set_timing_cache(build, nil, 0, $0, $1) }
      return true
    } catch {
      log.warning("no timing cache for this build: \(error)")
      return false
    }
  }

  /// Atomic like the plan: a build killed mid-write would leave a truncated
  /// cache for the next one to read.
  private func saveTimingCache(_ build: OpaquePointer, _ url: URL) {
    let temp = url.deletingLastPathComponent().appending(path: url.lastPathComponent + ".tmp")
    do {
      try trt.check { jl_trt_build_write_timing_cache(build, temp.path, $0, $1) }
      guard rename(temp.path, url.path) == 0 else {
        throw TrtError("cannot rename \(temp.lastPathComponent): \(String(cString: strerror(errno)))")
      }
    } catch {
      try? FileManager.default.removeItem(at: temp)
      log.warning("could not write the timing cache: \(error)")
    }
  }

  // MARK: load

  /// The host reports "deserializing engine" before and "ready" after, as
  /// Python's did; a plan TensorRT refuses is `ArtifactInvalid`.
  public func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    try TrtEngine(plan: artifact, trt: trt, gpuTiming: gpuTiming, faultAfter: faultAfter)
  }
}

/// TensorRT's build phases as one fraction, as trt/build.py's _Monitor made
/// them: the root phase's step over its step count, named by the phase.
/// Never stops the build.
final class BuildMonitor {
  private let report: ProgressFn
  private var phases: [String: (step: Int, total: Int)] = [:]
  private var root: String?

  init(_ report: @escaping ProgressFn) {
    self.report = report
  }

  func event(_ kind: Int32, phase: String, parent: String?, value: Int) {
    switch kind {
    case Int32(JL_TRT_PHASE_START):
      if parent == nil {
        root = phase
      }
      phases[phase] = (0, value)
      emit()
    case Int32(JL_TRT_PHASE_STEP):
      phases[phase] = (value, phases[phase]?.total ?? 0)
      emit()
    default:
      phases[phase] = nil
      if phase == root {
        root = nil
      }
    }
  }

  private func emit() {
    guard let root, let (step, total) = phases[root] else { return }
    let frac = total != 0 ? Double(step) / Double(total) : 0
    report("build", min(max(frac, 0), 1), root)
  }
}

/// The shim's progress calls, one at a time, into the build's monitor.
private func forwardProgress(
  _ ctx: UnsafeMutableRawPointer?, _ event: Int32, _ phase: UnsafePointer<CChar>?, _ parent: UnsafePointer<CChar>?, _ value: Int32
) {
  guard let ctx, let phase else { return }
  Unmanaged<BuildMonitor>.fromOpaque(ctx).takeUnretainedValue()
    .event(event, phase: String(cString: phase), parent: parent.map { String(cString: $0) }, value: Int(value))
}
