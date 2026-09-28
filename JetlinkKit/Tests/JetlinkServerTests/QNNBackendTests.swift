import Foundation
import JetlinkKit
import Testing

@testable import JetlinkORT
@testable import JetlinkServer

/// The Android backend's build, artifact and load, on its CPU device, which
/// is onnxruntime's CPU provider and so runs anywhere: the same golden frames
/// the CoreML backend's CPU device serves, bit for bit. The NPU and GPU
/// devices need a Snapdragon; here only their sessions and options are checked.
@Suite("QNN backend", .serialized)
struct QNNBackendTests {
  @Test("A comma is served the Python server's outputs through the QNN backend's CPU device", arguments: ["tiny_queued", "tiny_stateful"])
  func servesGoldenFrames(_ name: String) throws {
    let golden = try Golden(name)
    let cache = try TemporaryDirectory()
    let configuration = Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false)
    let backend = QNNBackend(device: .cpu, preparer: ONNXPreparer(), keepAlive: false)
    let server = try Server(configuration: configuration, backend: backend)
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
    let meta = ArtifactSidecar.read(artifact)
    #expect(meta["prepare"] as? Int == QNNBackend.prepareVersion)
    #expect(meta["preparer"] as? String == "swift")
  }

  @Test("Each device's sessions, layout and provider options")
  func devices() {
    let preparer = ONNXPreparer()
    let split = QNNBackend(device: .htp, preparer: preparer)
    #expect(split.sessions.map(\.name) == ["vision", "policy"])
    #expect(split.sessions.map(\.unit) == [.htp, .gpu])
    #expect(split.layout == .split)
    #expect(split.providerOptions(.htp)["backend_type"] == "htp")
    #expect(split.providerOptions(.htp)["htp_performance_mode"] == "burst")
    #expect(split.providerOptions(.gpu)["backend_type"] == "gpu")
    let sustained = QNNBackend(device: .htp, preparer: preparer, keepAlive: false)
    #expect(sustained.providerOptions(.htp)["htp_performance_mode"] == "sustained_high_performance")
    let whole = QNNBackend(device: .htpWhole, preparer: preparer)
    #expect(whole.sessions.map(\.unit) == [.htp])
    #expect(whole.layout == .aneWhole)
    #expect(QNNBackend(device: .gpu, preparer: preparer).layout == .whole)
    let plan = split.plan(URL(fileURLWithPath: "/tmp/vision_ctx.onnx"), .htp)
    #expect(plan.provider == "QNN")
    #expect(plan.label == "QNN(htp)")
    #expect(split.plan(URL(fileURLWithPath: "/tmp/model.onnx"), .cpu).provider == nil)
  }

  @Test("The chip the app reports names the artifacts")
  func chip() {
    let backend = QNNBackend(device: .htp, preparer: ONNXPreparer(), chip: "SM8650")
    #expect(backend.deviceTag() == "htp-SM8650")
    #expect(QNNBackend(device: .htp, preparer: ONNXPreparer(), chip: "").deviceTag() == "htp-unknown")
    #expect(backend.tag() == "ort\(sanitize(OrtRuntime.version)).htp-SM8650")
  }
}
