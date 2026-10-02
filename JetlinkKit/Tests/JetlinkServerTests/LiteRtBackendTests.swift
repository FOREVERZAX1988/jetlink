import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkLiteRT
@testable import JetlinkServer

#if canImport(Android)
  import Android
#endif

/// LiteRT's libraries, where $JETLINK_LITERT_DIR names them: the
/// ai-edge-litert 2.2.0 wheel's package directory on a Mac. Without them the
/// LiteRT suite skips, so a machine without LiteRT stays green.
enum LiteRtLibraries {
  static var available: Bool {
    !(ProcessInfo.processInfo.environment[LiteRtRuntime.directoryVariable] ?? "").isEmpty
  }

  /// The profiles a test can run here. On Android the GPU needs OpenCL:
  /// without it LiteRT falls back to OpenGL, which in a test runner (a shell
  /// process, no app, no EGL context) dies inside LiteRT's accelerator on a
  /// null string, as on the emulator, where the app's server gets an error.
  static var profiles: [LiteRtProfile] {
    #if os(Android)
      dlopen("libOpenCL.so", RTLD_NOW) != nil ? [.cpu, .gpu] : [.cpu]
    #else
      [.cpu, .gpu]
    #endif
  }

  /// The bench's directory, $JETLINK_LITERT_BENCH (LiteRtBenchTests).
  static var bench: URL? {
    ProcessInfo.processInfo.environment["JETLINK_LITERT_BENCH"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
  }
}

/// The on-device conversion's stand-in: the .tflite that
/// Scripts/make_litert_fixtures.py made from the same ONNX, which it finds
/// by the uploaded model's bytes.
struct FixtureConversion: LiteRTConverter {
  /// The fixture handed over for each source graph.
  var fixtures = ["tiny_queued": "tiny_queued", "tiny_stateful": "tiny_stateful"]
  var version = 1

  func convert(model: URL, into directory: URL) throws -> PreparedModel {
    let bytes = try Data(contentsOf: model)
    guard let fixture = fixtures.first(where: { (try? Fixture.data("\($0.key).onnx")) == bytes })?.value else {
      throw TestError("no LiteRT fixture for \(model.lastPathComponent)")
    }
    try FileManager.default.copyItem(at: Fixture.url("\(fixture).tflite"), to: directory.appending(path: "model.tflite"))
    return PreparedModel(
      parts: [PreparedModel.Part(name: "model", file: "model.tflite", weightBytes: 0)], summary: "\(fixture).tflite in place of a conversion")
  }
}

/// LiteRT behind the whole server: the build through the conversion's seam,
/// the compile, the load, the state loop on the GPU's memory or the CPU's,
/// and the replies against Python's outputs. On a Mac the GPU is Metal's.
@Suite(
  "LiteRT backend", .serialized,
  .enabled(if: LiteRtLibraries.available, "LiteRT's libraries are not in $\(LiteRtRuntime.directoryVariable)"))
struct LiteRtBackendTests {
  func backend(_ profile: LiteRtProfile, _ conversion: FixtureConversion = FixtureConversion()) -> LiteRtBackend {
    LiteRtBackend(profile: profile, preparer: ONNXPreparer(), converter: conversion)
  }

  @Test("A comma is served Python's outputs from LiteRT", arguments: LiteRtLibraries.profiles, ["tiny_queued", "tiny_stateful"])
  func servesGoldenFrames(_ profile: LiteRtProfile, _ name: String) throws {
    let golden = try Golden(name)
    try serve(backend: backend(profile)) { server, client in
      // The weights are fp16 and the GPU computes in fp16: held to what
      // verify_parity asks of a phone, not to Python's bits.
      let (hello, count) = try client.replay(golden, exact: false)
      #expect(hello["backend"] as? String == "litert")
      #expect(hello["runtime_version"] as? String == LiteRtRuntime.version)
      #expect((hello["device"] as? String)?.hasPrefix("\(profile.rawValue)-") == true)
      #expect(count == 8)
      #expect(eventually { server.framesServed == count })
      let engine = server.host.lock.withLock { server.host.loaded?.engine } as? LiteRtEngine
      #expect(engine?.fullyAccelerated == true || profile == .cpu)
      // The GPU keeps the queues in its own memory; the CPU reads the host's.
      #expect(engine?.stateOnDevice == (profile == .gpu && name == "tiny_stateful"))

      // The artifact: the model, its manifest, the GPU's cache directory, a sidecar.
      let engines = server.cache.layout.engines
      let artifacts = try FileManager.default.contentsOfDirectory(atPath: engines.path).filter { $0.hasSuffix(".litertcache") }
      #expect(artifacts.count == 1)
      let artifact = engines.appending(path: artifacts[0])
      let manifest = try LiteRtArtifact.open(artifact, version: 1).manifest
      #expect(manifest.model == "model.tflite")
      #expect(FileManager.default.fileExists(atPath: artifact.appending(path: manifest.cache).path))
      let meta = Artifact.sidecar(artifact)
      #expect(meta["backend"] as? String == "litert")
      #expect(meta["litert"] as? String == LiteRtRuntime.version)
      #expect(meta["accelerator"] as? String == profile.label)
      #expect(meta["prepare"] as? Int == 1)
      #expect(((meta["artifact_bytes"] as? NSNumber)?.int64Value ?? 0) > 0)
    }
  }

  @Test("Reset empties the looped state, in the GPU's memory as in the host's", arguments: LiteRtLibraries.profiles)
  func resetsState(_ profile: LiteRtProfile) throws {
    let temp = try TemporaryDirectory()
    let backend = backend(profile)
    let artifact = temp.url.appending(path: "tiny.litertcache")
    try backend.build(model: TinyModel.stateful, artifact: artifact, report: { _, _, _ in }, metaExtra: [:])
    let engine = try #require(try backend.load(artifact: artifact, report: { _, _, _ in }) as? LiteRtEngine)
    defer { engine.close() }
    let pairs = ["img_q", "desire_q", "feat_q"].map { (input: "state_\($0)", output: "next_state_\($0)") }
    try engine.loopState(pairs)
    #expect(engine.stateOnDevice == (profile == .gpu))
    #expect(pairs.allSatisfy { engine.hostInput($0.input) == nil || !engine.stateOnDevice })

    // The same frame three times: the queues fill, and the output moves with them.
    for name in engine.hostInputs {
      let spec = engine.inputs[name]!
      let input = engine.hostInput(name)!
      if spec.type == .uint8 {
        input.initializeMemory(as: UInt8.self, repeating: 200, count: spec.count)
      } else {
        input.initializeMemory(as: Float.self, repeating: 0.5, count: spec.count)
      }
    }
    let count = engine.outputs["outputs"]!.count
    func run() throws -> [Float] {
      try engine.run()
      return Array(UnsafeBufferPointer(start: engine.output("outputs")!.assumingMemoryBound(to: Float.self), count: count))
    }
    let first = try run()
    _ = try run()
    #expect(try run() != first)
    let queue = engine.inputs["state_img_q"]!
    var bytes = [UInt8](repeating: 0, count: queue.byteCount)
    try engine.state("state_img_q", into: &bytes)
    #expect(bytes.contains(200))

    engine.resetState()
    try engine.state("state_img_q", into: &bytes)
    #expect(!bytes.contains { $0 != 0 })
    let again = try run()
    #expect(Golden.correlation(Data(bytes: again, count: count * 4), Data(bytes: first, count: count * 4)) > 0.99999)
  }

  @Test(
    "A model the GPU cannot run whole fails to build, and says so; the CPU builds it",
    .enabled(if: LiteRtLibraries.profiles.contains(.gpu), "no OpenCL for LiteRT's GPU here"))
  func gpuRunsEveryOpOrNothing() throws {
    let temp = try TemporaryDirectory()
    let conversion = FixtureConversion(fixtures: ["tiny_stateful": "tiny_stateful_5d"])
    let artifact = temp.url.appending(path: "tiny.litertcache")
    #expect {
      try backend(.gpu, conversion).build(model: TinyModel.stateful, artifact: artifact, report: { _, _, _ in }, metaExtra: [:])
    } throws: { error in
      (error as? LiteRtError)?.description.contains("could not compile the model for the GPU") == true
    }
    #expect(!FileManager.default.fileExists(atPath: artifact.path))
    try backend(.cpu, conversion).build(model: TinyModel.stateful, artifact: artifact, report: { _, _, _ in }, metaExtra: [:])
    #expect(FileManager.default.fileExists(atPath: artifact.appending(path: "model.tflite").path))
  }

  @Test("An artifact from another conversion is rebuilt")
  func conversionVersion() throws {
    let temp = try TemporaryDirectory()
    let artifact = temp.url.appending(path: "tiny.litertcache")
    try backend(.cpu).build(model: TinyModel.queued, artifact: artifact, report: { _, _, _ in }, metaExtra: [:])
    var newer = FixtureConversion()
    newer.version = 2
    #expect(throws: ArtifactInvalid.self) { try backend(.cpu, newer).load(artifact: artifact, report: { _, _, _ in }) }
    #expect(throws: ArtifactInvalid.self) { try backend(.cpu).load(artifact: temp.url.appending(path: "missing.litertcache"), report: { _, _, _ in }) }
  }
}

/// What needs no library: the profiles and the tags.
@Suite("LiteRT profiles")
struct LiteRtProfileTests {
  @Test("The profiles are the apps' device names, and the tag names the chip")
  func profiles() {
    #expect(LiteRtProfile.allCases.map(\.rawValue) == ["litert-gpu", "litert-cpu"])
    let backend = LiteRtBackend(profile: .gpu, preparer: ONNXPreparer(), chip: "Tensor G5")
    #expect(backend.name == "litert" && backend.suffix == ".litertcache")
    #expect(backend.deviceTag() == "litert-gpu-Tensor_G5")
    #expect(backend.tag() == "litert2.2.0.litert-gpu-Tensor_G5")
    #expect(LiteRtBackend(profile: .cpu, preparer: ONNXPreparer()).deviceTag() == sanitize("litert-cpu-\(HostChip.name())"))
  }
}

/// A real model on LiteRT in a closed loop, where $JETLINK_LITERT_BENCH names
/// a directory holding model.tflite and, for each input the host writes,
/// `<name>.bin`: its frames back to back. Logs the compile, the first frame
/// and the frame time ($JETLINK_LITERT_BENCH_FRAMES of them, 200 by default),
/// and writes the driving output of each recorded frame to
/// outputs.litert.bin for a parity check against onnxruntime's.
/// $JETLINK_LITERT_BENCH_DEVICE is litert-gpu (the default) or litert-cpu.
@Suite(
  "LiteRT bench", .serialized,
  .enabled(if: LiteRtLibraries.available && LiteRtLibraries.bench != nil, "no LiteRT, or no model in $JETLINK_LITERT_BENCH"))
struct LiteRtBenchTests {
  @Test("A model in a closed loop: compile, first frame, frame time, outputs")
  func closedLoop() throws {
    let environment = ProcessInfo.processInfo.environment
    let directory = try #require(LiteRtLibraries.bench)
    let profile = LiteRtProfile(rawValue: environment["JETLINK_LITERT_BENCH_DEVICE"] ?? "") ?? .gpu
    let total = Int(environment["JETLINK_LITERT_BENCH_FRAMES"] ?? "") ?? 200
    try LiteRtRuntime.load()
    let compileStarted = DispatchTime.now()
    let options =
      profile == .gpu ? LiteRtCompileOptions(gpu: true, gpuFP16: true) : LiteRtCompileOptions(gpu: false, cpuThreads: LiteRtBackend.cpuThreads)
    let engine = try LiteRtEngine(model: directory.appending(path: "model.tflite"), options: options, device: "bench", label: profile.label)
    defer { engine.close() }
    let compileMs = Double(DispatchTime.now().uptimeNanoseconds - compileStarted.uptimeNanoseconds) / 1e6
    #expect(engine.fullyAccelerated || profile == .cpu)
    try engine.loopState(
      engine.inputs.keys.filter { $0.hasPrefix("state_") && engine.outputs["next_\($0)"] != nil }.map { (input: $0, output: "next_\($0)") })

    let frames = try Dictionary(uniqueKeysWithValues: engine.hostInputs.map { ($0, try Data(contentsOf: directory.appending(path: "\($0).bin"))) })
    let recorded = engine.hostInputs.map { frames[$0]!.count / engine.inputs[$0]!.byteCount }.min() ?? 0
    #expect(recorded > 0)
    let output = try #require(engine.outputs["outputs"])
    var outputs = Data(capacity: recorded * output.byteCount)
    var scratch = Data(count: output.byteCount)
    var times: [Double] = []
    for i in 0..<max(total, recorded) {
      let frame = i % recorded
      let started = DispatchTime.now()
      for name in engine.hostInputs {
        let bytes = engine.inputs[name]!.byteCount
        frames[name]!.withUnsafeBytes { engine.hostInput(name)!.copyMemory(from: $0.baseAddress! + frame * bytes, byteCount: bytes) }
      }
      try engine.run()
      scratch.withUnsafeMutableBytes { $0.baseAddress!.copyMemory(from: engine.output("outputs")!, byteCount: output.byteCount) }
      times.append(Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1e6)
      if i < recorded { outputs.append(scratch) }
    }
    try outputs.write(to: directory.appending(path: "outputs.litert.bin"))
    let steady = times.dropFirst(2).sorted()
    let at = { (q: Double) in steady[min(steady.count - 1, Int(q * Double(steady.count)))] }
    print(
      String(
        format: "LiteRT bench %@ (%@): compile %.0f ms, first frame %.1f ms, then p50 %.2f ms p99 %.2f ms max %.2f ms over %d frames; %d recorded",
        profile.rawValue, engine.notes, compileMs, times[0], at(0.5), at(0.99), steady.last ?? 0, steady.count, recorded))
  }
}
