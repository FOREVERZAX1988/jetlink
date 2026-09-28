import Foundation
import JetlinkKit
import Testing

@testable import JetlinkORT
@testable import JetlinkServer

/// onnxruntime's backend: the build, artifact and load on the CPU profile,
/// which runs anywhere and serves the golden frames; the other profiles'
/// sessions, layouts and options, which need a Neural Engine or a Snapdragon
/// to run, are only checked.
@Suite("onnxruntime backend", .serialized)
struct OrtBackendTests {
  @Test("A comma is served the Python server's outputs through the CPU profile", arguments: ["tiny_queued", "tiny_stateful"])
  func servesGoldenFrames(_ name: String) throws {
    let golden = try Golden(name)
    let cache = try TemporaryDirectory()
    let configuration = Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false)
    let server = try Server(configuration: configuration, backend: OrtBackend(profile: .cpu, preparer: ONNXPreparer(), keepAlive: false))
    try server.start()
    defer { server.stop() }
    let client = try TestClient(port: server.port!)
    defer { client.close() }
    let (hello, count) = try client.replay(golden)
    #expect(hello["backend"] as? String == "ort")
    #expect((hello["device"] as? String)?.hasPrefix("cpu-") == true)
    #expect(count == 8)

    // The artifact: a manifest naming the session and its unit, and a sidecar.
    let engines = cache.url.appending(path: "engines")
    let artifacts = try FileManager.default.contentsOfDirectory(atPath: engines.path).filter { $0.hasSuffix(".ortcache") }
    #expect(artifacts.count == 1)
    let artifact = engines.appending(path: artifacts[0])
    let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: artifact.appending(path: "sessions.json"))) as? [[String: Any]]
    #expect(manifest?.map { $0["unit"] as? String } == ["cpu"])
    let meta = Artifact.sidecar(artifact)
    #expect(meta["prepare"] as? Int == OrtBackend.prepareVersion)
    #expect(meta["preparer"] as? String == "swift")
    #expect(((meta["artifact_bytes"] as? NSNumber)?.int64Value ?? 0) > 0)
  }

  @Test("Each profile's sessions and layout")
  func profiles() {
    let table: [(OrtProfile, [String], [OrtUnit], String)] = [
      (.ane, ["vision", "policy"], [.coreML("CPUAndNeuralEngine"), .coreML("CPUAndGPU")], "split"),
      (.coreml, ["model"], [.coreML("CPUAndGPU")], "whole"),
      (.aneWhole, ["model"], [.coreML("ALL")], "aneWhole"),
      (.htp, ["vision", "policy"], [.htp, .qnnGPU], "split"),
      (.htpWhole, ["model"], [.htp], "aneWhole"),
      (.gpu, ["model"], [.qnnGPU], "whole"),
      (.cpu, ["model"], [.cpu], "whole"),
    ]
    #expect(table.map(\.0) == OrtProfile.allCases)
    for (profile, names, units, layout) in table {
      #expect(profile.sessions.map(\.name) == names, "\(profile)")
      #expect(profile.sessions.map(\.unit) == units, "\(profile)")
      #expect("\(profile.layout)" == layout, "\(profile)")
    }
    #expect(OrtProfile(rawValue: "ane-whole") == .aneWhole && OrtProfile(rawValue: "htp-whole") == .htpWhole)
    #if canImport(Metal)
      #expect(OrtProfile.available == [.ane, .aneWhole, .coreml, .cpu])
    #elseif os(Android)
      #expect(OrtProfile.available == [.htp, .htpWhole, .gpu, .cpu])
    #else
      #expect(OrtProfile.available == [.cpu])
    #endif
  }

  @Test("Each unit's session plan and provider options")
  func plans() {
    let backend = OrtBackend(profile: .htp, preparer: ONNXPreparer())
    #expect(backend.providerOptions(.htp)["backend_type"] == "htp")
    #expect(backend.providerOptions(.htp)["htp_performance_mode"] == "burst")
    #expect(backend.providerOptions(.qnnGPU)["backend_type"] == "gpu")
    let sustained = OrtBackend(profile: .htp, preparer: ONNXPreparer(), keepAlive: false)
    #expect(sustained.providerOptions(.htp)["htp_performance_mode"] == "sustained_high_performance")

    let model = URL(fileURLWithPath: "/tmp/model.onnx")
    let npu = backend.plan(.htp, model: model, cache: nil)
    #expect(npu.provider == "QNN" && npu.label == "QNN(htp)" && npu.usesNeuralEngine && !npu.usesGPU)
    let gpu = backend.plan(.qnnGPU, model: model, cache: nil)
    #expect(gpu.provider == "QNN" && gpu.label == "QNN(gpu)" && gpu.usesGPU)
    let cpu = backend.plan(.cpu, model: model, cache: nil)
    #expect(cpu.provider == nil && cpu.label == "CPU" && cpu.threads == OrtBackend.cpuThreads)
    let coreML = backend.plan(.coreML("ALL"), model: model, cache: URL(fileURLWithPath: "/tmp/coreml-model"))
    #expect(coreML.provider == "CoreML" && coreML.label == "CoreML(ALL)" && coreML.usesGPU && coreML.usesNeuralEngine)
    #expect(coreML.options["ModelCacheDirectory"] == "/tmp/coreml-model")
    #expect(coreML.options["SpecializationStrategy"] == "FastPrediction")
    #expect(backend.plan(.coreML("CPUAndGPU"), model: model, cache: nil).options["SpecializationStrategy"] == nil)
  }

  /// An entry names its unit as the backend that wrote it did: QNN's and the
  /// CPU's by unit, CoreML's by compute units, the CPU as null units in Apple
  /// artifacts from before the backends were one.
  @Test("Manifest entries, old and new, read back to their units")
  func manifestEntries() {
    for unit in [OrtUnit.coreML("CPUAndNeuralEngine"), .coreML("ALL"), .htp, .qnnGPU, .cpu] {
      #expect(OrtUnit(entry: unit.entry(model: "model.onnx", session: "model")) == unit, "\(unit)")
    }
    #expect(OrtUnit.coreML("CPUAndGPU").entry(model: "policy.onnx", session: "policy")["cache"] as? String == "coreml-policy")
    #expect(OrtUnit(entry: ["model": "model.onnx", "units": NSNull(), "cache": NSNull()]) == .cpu)
    #expect(OrtUnit(entry: ["model": "model.onnx"]) == nil)
    #expect(OrtUnit(entry: ["model": "model.onnx", "unit": "npu"]) == nil)
  }

  @Test("The chip names the artifacts")
  func chip() {
    let backend = OrtBackend(profile: .htp, preparer: ONNXPreparer(), chip: "SM8650")
    #expect(backend.deviceTag() == "htp-SM8650")
    #expect(backend.tag() == "ort\(sanitize(OrtRuntime.version)).htp-SM8650")
    #expect(OrtBackend(profile: .htp, preparer: ONNXPreparer(), chip: "").deviceTag() == "htp-unknown")
    #if canImport(Metal)
      #expect(OrtBackend(profile: .cpu, preparer: ONNXPreparer()).deviceTag() == sanitize("cpu-\(OrtBackend.chipName())"))
    #else
      #expect(OrtBackend(profile: .cpu, preparer: ONNXPreparer()).deviceTag() == "cpu-cpu")
    #endif
  }
}
