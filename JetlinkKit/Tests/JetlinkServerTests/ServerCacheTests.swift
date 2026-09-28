import Foundation
import Testing

@testable import JetlinkServer

/// A backend that is only its names: all the cache asks of one.
final class NamingBackend: EngineBackend {
  let name: String
  let suffix: String
  let artifactKind: ArtifactKind
  let runtimeVersion = "10.3.0"

  init(kind: ArtifactKind) {
    artifactKind = kind
    (name, suffix) = kind == .file ? ("trt", ".plan") : ("ort", ".ortcache")
  }

  func deviceTag() -> String { "Orin-sm87" }

  func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    throw HostError.failed("names only")
  }

  func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    throw HostError.failed("names only")
  }

  func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    throw HostError.failed("names only")
  }
}

@Suite("Server cache")
struct ServerCacheTests {
  let sha = String(repeating: "c", count: 64)

  /// One artifact of `kind` and its sidecar, stamped at `mtime`.
  func artifact(_ url: URL, kind: ArtifactKind, mtime: TimeInterval) throws {
    if kind == .directory {
      try FileManager.default.createDirectory(at: url.appending(path: "coreml-model"), withIntermediateDirectories: true)
    } else {
      try Data("plan".utf8).write(to: url)
    }
    try Data("{}".utf8).write(to: Artifact.sidecarURL(url))
    try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: mtime)], ofItemAtPath: url.path)
  }

  @Test("The artifact is a file or a directory as the backend says, named as before")
  func naming() throws {
    let tmp = try TemporaryDirectory()
    let plans = try ServerCache(root: tmp.url, backend: NamingBackend(kind: .file))
    let plan = try plans.entry(sha)
    #expect(plan.path.lastPathComponent == "cccccccccccccccc.trt10.3.0.Orin-sm87.plan")
    #expect(!plan.path.hasDirectoryPath)
    #expect(plan.metaPath.lastPathComponent == "cccccccccccccccc.trt10.3.0.Orin-sm87.json")
    let ort = try ServerCache(root: tmp.url, backend: NamingBackend(kind: .directory)).entry(sha)
    #expect(ort.path.hasDirectoryPath)
    #expect(Artifact.sidecarURL(ort.path) == ort.metaPath)
    #expect(Artifact.sidecarURL(plan.path) == plan.metaPath)
    #expect(try plans.modelPath(sha).lastPathComponent == "cccccccccccccccc.onnx")
    #expect(throws: (any Error).self) { try plans.entry("../../etc") }
  }

  /// A Jetson boots at 1970 without NTP, so the plan it just built can be the
  /// oldest file on disk. A plan is one file, and its entry must still match
  /// the listing prune walks.
  @Test("The artifact just built survives pruning though its mtime is the oldest", arguments: [ArtifactKind.file, .directory])
  func pruneProtects(kind: ArtifactKind) throws {
    let tmp = try TemporaryDirectory()
    let backend = NamingBackend(kind: kind)
    let cache = try ServerCache(root: tmp.url, backend: backend)
    for i in 0..<3 {
      try artifact(cache.layout.engines.appending(path: "old\(i)\(backend.suffix)"), kind: kind, mtime: 2_000_000 + Double(i))
    }
    let fresh = try cache.entry(sha)
    try artifact(fresh.path, kind: kind, mtime: 1)

    cache.prune(keep: 2, protect: fresh.path)

    #expect(fresh.exists)
    let left = try FileManager.default.contentsOfDirectory(atPath: cache.layout.engines.path).filter { $0.hasSuffix(backend.suffix) }
    #expect(left.sorted() == ["cccccccccccccccc.\(backend.tag())\(backend.suffix)", "old2\(backend.suffix)"])
  }

  @Test("sweepTemp drops build directories older than six hours, and only those")
  func sweep() throws {
    let tmp = try TemporaryDirectory()
    let cache = try ServerCache(root: tmp.url, backend: NamingBackend(kind: .file))
    let stale = cache.layout.engines.appending(path: "tmpstale")
    let fresh = cache.layout.engines.appending(path: "tmpfresh")
    for directory in [stale, fresh] {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-7 * 3600)], ofItemAtPath: stale.path)
    cache.sweepTemp()
    #expect(!FileManager.default.fileExists(atPath: stale.path))
    #expect(FileManager.default.fileExists(atPath: fresh.path))
  }
}

@Suite("Artifacts")
struct ArtifactTests {
  final class Reports: @unchecked Sendable {
    private let lock = NSLock()
    private var seen: [String] = []
    var all: [String] { lock.withLock { seen } }
    var fn: ProgressFn { { [self] stage, frac, msg in lock.withLock { seen.append("\(stage) \(frac) \(msg)") } } }
  }

  @Test("A file artifact is staged beside the cache, moved into place, and described")
  func buildsAFile() throws {
    let tmp = try TemporaryDirectory()
    let backend = NamingBackend(kind: .file)
    let plan = tmp.url.appending(path: "engines/m.trt10.3.0.Orin-sm87.plan")
    let reports = Reports()
    try Artifact.build(plan, kind: .file, metaExtra: ["spec": ["sha256": "x"]], report: reports.fn) { staged in
      #expect(staged.deletingLastPathComponent().lastPathComponent.hasPrefix("tmp"))
      #expect(!FileManager.default.fileExists(atPath: staged.path))
      try Data("plan".utf8).write(to: staged)
      var meta = Artifact.meta(backend, runtimeKey: "trt_version", model: URL(filePath: "/models/m.onnx"), started: Date())
      meta["fp16"] = true
      return meta
    }
    #expect(try Data(contentsOf: plan) == Data("plan".utf8))
    let meta = Artifact.sidecar(plan)
    #expect(Set(meta.keys) == ["backend", "trt_version", "device", "build_seconds", "onnx", "built_at", "fp16", "spec"])
    #expect(meta["backend"] as? String == "trt" && meta["trt_version"] as? String == "10.3.0" && meta["onnx"] as? String == "m.onnx")
    #expect((meta["built_at"] as? String)?.wholeMatch(of: /\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ/) != nil)
    #expect(reports.all == ["build 1.0 done in 0.0s"])
    let left = try FileManager.default.contentsOfDirectory(atPath: plan.deletingLastPathComponent().path)
    #expect(left.sorted() == ["m.trt10.3.0.Orin-sm87.json", "m.trt10.3.0.Orin-sm87.plan"])
  }

  @Test("A failed build leaves the old artifact and no staging")
  func failedBuild() throws {
    let tmp = try TemporaryDirectory()
    let artifact = tmp.url.appending(path: "m.ort1.ortcache", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: artifact, withIntermediateDirectories: true)
    #expect(throws: HostError.self) {
      try Artifact.build(artifact, kind: .directory, metaExtra: [:], report: { _, _, _ in }) { staged in
        #expect(FileManager.default.fileExists(atPath: staged.path))
        throw HostError.failed("no")
      }
    }
    #expect(try FileManager.default.contentsOfDirectory(atPath: tmp.url.path) == ["m.ort1.ortcache"])
  }
}
