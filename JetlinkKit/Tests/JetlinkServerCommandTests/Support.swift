#if os(macOS) || os(Linux)
  import Foundation
  import JetlinkRegistry
  import JetlinkServer
  import JetlinkTestSupport
  import Testing

  @testable import jetlink_server

  #if canImport(FoundationNetworking)
    import FoundationNetworking
  #endif

  /// The tiny graphs the server's tests serve, and the specs Python derived
  /// for them, in place in the checkout.
  enum Tiny {
    static let directory = SourceTree.root().appending(path: "JetlinkKit/Tests/JetlinkServerTests/Fixtures")
    static let queued = directory.appending(path: "tiny_queued.onnx")
    static let stateful = directory.appending(path: "tiny_stateful.onnx")

    static func spec(_ model: URL) throws -> [String: Any] {
      let data = try Data(contentsOf: model.deletingPathExtension().appendingPathExtension("spec.json"))
      return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    static func sha256(_ model: URL) throws -> String {
      try #require(try spec(model)["sha256"] as? String)
    }
  }

  /// The registry suite's fixtures: tests/fixtures.
  enum RegistryFixture {
    static let directory = SourceTree.root().appending(path: "tests/fixtures")
    static let ref = "f877d7a0ccc3cce943c76e285214c020cd65c899"
    static let oid = "a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525"
    static let size: Int64 = 765_953_504
    static let newestRef = "37bfa1413edcdc2e8844984b83727c33f81d8f46"
    static let smallRef = String(repeating: "a", count: 40)

    static func data(_ name: String) throws -> Data {
      try Data(contentsOf: directory.appending(path: name))
    }

    static func catalogRoutes() throws -> [String: MockNet.Reply] {
      [Catalog.url: .body(try data("catalog_chestnut_v25.json"))]
    }

    static func pointerText(oid: String, size: Int64) -> Data {
      Data("version https://git-lfs.github.com/spec/v1\noid sha256:\(oid)\nsize \(size)\n".utf8)
    }

    /// The batch response, retargeted at `oid` and served from `href`.
    static func batch(oid: String, size: Int64, href: String?) throws -> Data {
      var payload = try #require(
        try JSONSerialization.jsonObject(with: data(href == nil ? "lfs_batch_missing.json" : "lfs_batch_response.json")) as? [String: Any])
      var objects = try #require(payload["objects"] as? [[String: Any]])
      objects[0]["oid"] = oid
      objects[0]["size"] = size
      if let href {
        objects[0]["actions"] = ["download": ["href": href, "header": [String: String]()]]
      }
      payload["objects"] = objects
      return try JSONSerialization.data(withJSONObject: payload)
    }

    /// `smallRef` points at `body` under `oid`; the first LFS server has
    /// nothing, the second serves it from `href`.
    static func smallRoutes(_ body: Data, oid: String, href: String = "https://blob.example/object") throws -> [String: MockNet.Reply] {
      let size = Int64(body.count)
      var routes: [String: MockNet.Reply] = [
        LFS.pointerURL(ref: smallRef): .body(pointerText(oid: oid, size: size)),
        "\(LFS.endpoints[0])/objects/batch": .body(try batch(oid: oid, size: size, href: nil)),
        "\(LFS.endpoints[1])/objects/batch": .body(try batch(oid: oid, size: size, href: href)),
      ]
      if !href.hasPrefix("http://127.0.0.1") {
        routes[href] = .body(body)
      }
      return routes
    }
  }

  /// A directory per test, removed by the test's `defer`.
  struct TempDir {
    let url: URL

    init() throws {
      url = FileManager.default.temporaryDirectory
        .appending(path: "jetlink-command-tests", directoryHint: .isDirectory)
        .appending(path: UUID().uuidString, directoryHint: .isDirectory)
      try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    var path: String { url.path }

    func remove() {
      try? FileManager.default.removeItem(at: url)
    }
  }

  /// What a models command wrote, for a test to read.
  final class Captured: Sendable {
    private let text = Locked<(out: String, err: String)>(("", ""))

    var console: Console {
      Console(
        out: { line in self.text.withLock { $0.out += line + "\n" } },
        err: { chunk in self.text.withLock { $0.err += chunk } })
    }

    var out: String { text.withLock { $0.out } }
    var err: String { text.withLock { $0.err } }
  }

  /// A models command as the command line gives it, run against `net`.
  func models<Command: ModelsCommand>(
    _ type: Command.Type, _ arguments: [String], cache: TempDir, net: MockNet = MockNet()
  ) async throws -> (code: Int32, out: String, err: String) {
    let command = try Command.parse(arguments + ["--cache", cache.path])
    let captured = Captured()
    let code = await command.execute(Registry(layout: CacheLayout(root: cache.url), session: net.session), captured.console)
    return (code, captured.out, captured.err)
  }
#endif
