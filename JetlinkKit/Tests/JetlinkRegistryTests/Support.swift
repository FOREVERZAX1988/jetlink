import Foundation
import JetlinkTestSupport
import Synchronization
import Testing

@testable import JetlinkRegistry

#if canImport(CryptoKit)
  import CryptoKit
#else
  import Crypto
#endif
#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

// MARK: - fixtures

/// The Python suite's fixtures, read from the repo so both suites test against
/// the same bytes: <repo>/tests/fixtures.
enum Fixture {
  static let directory = SourceTree.root().appending(path: "tests/fixtures")

  static func data(_ name: String) -> Data {
    do {
      return try Data(contentsOf: directory.appending(path: name))
    } catch {
      Issue.record("missing fixture \(name): \(error)")
      return Data()
    }
  }

  static func json(_ name: String) -> JSON {
    (try? JSON.parse(data(name))) ?? .null
  }
}

let fixtureRef = "f877d7a0ccc3cce943c76e285214c020cd65c899"
let fixtureOID = "a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525"
let fixtureSize: Int64 = 765_953_504
let newestRef = "37bfa1413edcdc2e8844984b83727c33f81d8f46"
let fixtureBlob = Data(String(repeating: "onnx", count: 1024).utf8)
let fixtureBlobSHA = sha256Hex(fixtureBlob)
let smallRef = String(repeating: "a", count: 40)

func sha256Hex(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func catalogURL(_ version: Int) -> String { Catalog.url(version: version) }

/// One of the batch fixtures, retargeted at another object.
func batch(oid: String, size: Int64, href: String?) -> Data {
  guard var payload = Fixture.json(href == nil ? "lfs_batch_missing.json" : "lfs_batch_response.json").object,
    var objects = payload["objects"]?.array, var first = objects.first?.object
  else { return Data() }
  first["oid"] = .string(oid)
  first["size"] = .int(size)
  if let href, var actions = first["actions"]?.object, var download = actions["download"]?.object {
    download["href"] = .string(href)
    actions["download"] = .object(download)
    first["actions"] = .object(actions)
  }
  objects[0] = .object(first)
  payload["objects"] = .array(objects)
  return JSON.object(payload).data()
}

func smallPointerText(oid: String = fixtureBlobSHA, size: Int64 = Int64(fixtureBlob.count)) -> Data {
  Data("version https://git-lfs.github.com/spec/v1\noid sha256:\(oid)\nsize \(size)\n".utf8)
}

/// The first endpoint has nothing, the second serves the object.
func smallRoutes(oid: String = fixtureBlobSHA, size: Int64 = Int64(fixtureBlob.count), blob: Data = fixtureBlob) -> [String: MockNet.Reply] {
  [
    LFS.pointerURL(ref: smallRef): .body(smallPointerText(oid: oid, size: size)),
    "\(LFS.endpoints[0])/objects/batch": .body(batch(oid: oid, size: size, href: nil)),
    "\(LFS.endpoints[1])/objects/batch": .body(batch(oid: oid, size: size, href: "https://blob.example/object")),
    "https://blob.example/object": .body(blob),
  ]
}

func catalogRoutes(_ extra: [String: MockNet.Reply] = [:]) -> [String: MockNet.Reply] {
  var routes: [String: MockNet.Reply] = [Catalog.url: .body(Fixture.data("catalog_chestnut_v25.json"))]
  routes.merge(extra) { _, new in new }
  return routes
}

func makeSpec(_ sha256: String) -> JSON {
  [
    "sha256": .string(sha256), "nbytes": 1234, "frame_skip": 4, "checkpoint": "b9facbcc",
    "input_shapes": ["img": [1, 12, 128, 256]],
    "output_shapes": ["outputs": [1, 18452]],
    "output_slices": ["hidden_state": [2066, 18450]],
  ]
}

// MARK: - a scratch cache

/// A fresh directory per test, removed by the test's `defer`. Not a deinit:
/// Swift may release an object before the end of its scope.
struct TempDir {
  let url: URL

  init() throws {
    url = FileManager.default.temporaryDirectory
      .appending(path: "jetlink-registry-tests", directoryHint: .isDirectory)
      .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  }

  var layout: CacheLayout { CacheLayout(root: url) }

  func remove() {
    try? FileManager.default.removeItem(at: url)
  }

  func names(_ sub: String) -> [String] {
    (try? FileManager.default.contentsOfDirectory(atPath: url.appending(path: sub).path(percentEncoded: false)))?.sorted() ?? []
  }
}

// MARK: - the network

// MockNet and LocalServer are JetlinkTestSupport's, which the command's tests
// share.

#if canImport(Darwin) || canImport(Glibc)
  extension LocalServer {
    /// The SHA-256 of what is served, without holding it.
    var sha256: String {
      var hasher = SHA256()
      var left = total
      while left > 0 {
        let n = Int(min(Int64(pattern.count), left))
        hasher.update(data: pattern.prefix(n))
        left -= Int64(n)
      }
      return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
  }
#endif

/// Collects progress calls from whatever thread makes them.
final class ProgressLog: Sendable {
  private let values = Mutex<[Double]>([])

  var callback: @Sendable (Double) -> Void {
    { [self] value in values.withLock { $0.append(value) } }
  }

  var all: [Double] { values.withLock { $0 } }
}

/// A shouldStop that answers from a list, then with its last answer.
final class StopScript: Sendable {
  private let answers: Mutex<[Bool]>

  init(_ answers: [Bool]) {
    self.answers = Mutex(answers)
  }

  var callback: @Sendable () -> Bool {
    { [self] in
      answers.withLock { list in
        list.count > 1 ? list.removeFirst() : (list.first ?? false)
      }
    }
  }
}
