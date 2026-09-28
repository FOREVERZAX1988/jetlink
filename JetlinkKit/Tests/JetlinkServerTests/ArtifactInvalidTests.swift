import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

/// The CPU backend, except that the next `failLoads` loads throw
/// ArtifactInvalid, as a plan from another TensorRT build does.
final class FlakyBackend: EngineBackend, @unchecked Sendable {
  private let inner = cpuBackend()
  private let lock = NSLock()
  private var failing = 0
  private var built = 0
  private var loaded = 0

  var name: String { inner.name }
  var suffix: String { inner.suffix }
  var artifactKind: ArtifactKind { inner.artifactKind }
  var runtimeVersion: String { inner.runtimeVersion }
  func deviceTag() -> String { inner.deviceTag() }

  func failNextLoads(_ count: Int) {
    lock.withLock {
      failing = count
      built = 0
      loaded = 0
    }
  }

  var counts: (builds: Int, loads: Int) { lock.withLock { (built, loaded) } }

  func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try inner.deriveSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    lock.withLock { built += 1 }
    try inner.build(model: model, artifact: artifact, report: report, metaExtra: metaExtra)
  }

  func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    let fail = lock.withLock {
      loaded += 1
      guard failing > 0 else { return false }
      failing -= 1
      return true
    }
    if fail {
      throw ArtifactInvalid("\(artifact.lastPathComponent): built by another runtime")
    }
    return try inner.load(artifact: artifact, report: report)
  }
}

/// A cached artifact that will not load is deleted and rebuilt once from
/// the ONNX, for any backend.
@Suite("Invalid artifacts", .serialized)
struct ArtifactInvalidTests {
  func ready(_ host: EngineHost) -> Bool {
    eventually(timeout: 120) { host.snapshot().state == .ready || host.snapshot().state == .failed }
  }

  /// A host with `golden`'s model built and cached, then unloaded.
  func cached(_ golden: Golden, in tmp: TemporaryDirectory, backend: FlakyBackend) throws -> (EngineCache, Request) {
    let cache = try EngineCache(root: tmp.url, backend: backend)
    let model = try Data(contentsOf: golden.model)
    let request = try Request(sha256: golden.sha256, nbytes: Int64(model.count), frameSkip: 4)
    try model.write(to: cache.modelPath(request))
    let host = EngineHost(cache: cache)
    _ = host.request(request, session: nil)
    #expect(ready(host) && host.snapshot().state == .ready)
    host.close()
    #expect(cache.entry(request).exists)
    return (cache, request)
  }

  @Test("An artifact that will not load is rebuilt once from the ONNX on disk")
  func rebuildsOnce() throws {
    let golden = try Golden("tiny_stateful")
    let tmp = try TemporaryDirectory()
    let backend = FlakyBackend()
    let (cache, request) = try cached(golden, in: tmp, backend: backend)

    backend.failNextLoads(1)
    let host = EngineHost(cache: cache)
    defer { host.close() }
    _ = host.request(request, session: nil)
    #expect(ready(host))
    #expect(host.snapshot().state == .ready)
    #expect(backend.counts == (builds: 1, loads: 2))
    #expect(cache.entry(request).exists)
  }

  @Test("Without the ONNX the artifact still goes, and the client is asked to upload")
  func withoutTheModel() throws {
    let golden = try Golden("tiny_stateful")
    let tmp = try TemporaryDirectory()
    let backend = FlakyBackend()
    let (cache, request) = try cached(golden, in: tmp, backend: backend)
    try FileManager.default.removeItem(at: cache.modelPath(request))

    backend.failNextLoads(1)
    let host = EngineHost(cache: cache)
    defer { host.close() }
    _ = host.request(request, session: nil)
    #expect(ready(host))
    let event = host.snapshot()
    #expect(event.state == .failed)
    #expect(event.detail.contains("artifact invalid and the model is not on disk"))
    #expect(backend.counts == (builds: 0, loads: 1))
    #expect(!cache.entry(request).exists)
    #expect(host.request(request, session: nil)["state"] as? String == "need_upload")
  }
}
