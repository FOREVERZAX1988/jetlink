import Foundation
import Synchronization
import Testing

@testable import JetlinkRegistry

#if canImport(CryptoKit)
  import CryptoKit
#else
  import Crypto
#endif
#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#endif
#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

// MARK: - fixtures

/// The Python suite's fixtures, read from the repo so both suites test against
/// the same bytes: <repo>/tests/fixtures.
enum Fixture {
  static let directory = URL(filePath: #filePath)
    .deletingLastPathComponent()  // JetlinkRegistryTests
    .deletingLastPathComponent()  // Tests
    .deletingLastPathComponent()  // JetlinkKit
    .deletingLastPathComponent()  // the repo
    .appending(path: "tests/fixtures")

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

/// A URLSession whose requests are answered from a table, and refused when
/// they are not in it, as the Python suite's FakeOpener does. Each MockNet
/// has its own table, found by a header its session adds, so tests running in
/// parallel do not see each other's routes. A table outlives its MockNet: a
/// test that keeps only the session still has its routes.
final class MockNet: Sendable {
  enum Reply: Sendable {
    case body(Data)
    /// An HTTP error status with a short body.
    case status(Int)
    /// A transport failure, as urllib's URLError.
    case failure
    /// A body delivered in pieces, for progress.
    case chunks([Data])
  }

  struct Call: Sendable {
    let url: String
    let method: String
    let body: Data?
  }

  static let header = "X-Jetlink-Test-Net"
  fileprivate static let tables = Mutex<[String: Table]>([:])

  fileprivate struct Table {
    var routes: [String: Reply]
    var calls: [Call] = []
  }

  let id = UUID().uuidString
  let session: URLSession

  init(_ routes: [String: Reply] = [:]) {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockProtocol.self]
    configuration.httpAdditionalHeaders = [MockNet.header: id]
    session = URLSession(configuration: configuration)
    MockNet.tables.withLock { $0[id] = Table(routes: routes) }
  }

  subscript(url: String) -> Reply? {
    get { MockNet.tables.withLock { $0[id]?.routes[url] } }
    set { MockNet.tables.withLock { $0[id]?.routes[url] = newValue } }
  }

  var calls: [Call] { MockNet.tables.withLock { $0[id]?.calls ?? [] } }
  var urls: [String] { calls.map(\.url) }

  func count(_ url: String) -> Int { urls.filter { $0 == url }.count }
}

final class MockProtocol: URLProtocol, @unchecked Sendable {
  override class func canInit(with request: URLRequest) -> Bool {
    // The local server's requests go to the real socket.
    request.value(forHTTPHeaderField: MockNet.header) != nil && request.url?.host != "127.0.0.1"
  }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  override func startLoading() {
    guard let client, let url = request.url, let id = request.value(forHTTPHeaderField: MockNet.header) else { return }
    let body = request.httpBody ?? request.httpBodyStream.map(MockProtocol.drain)
    let call = MockNet.Call(url: url.absoluteString, method: request.httpMethod ?? "GET", body: body)
    let reply = MockNet.tables.withLock { tables -> MockNet.Reply? in
      tables[id]?.calls.append(call)
      return tables[id]?.routes[url.absoluteString]
    }
    func respond(_ status: Int, _ pieces: [Data]) {
      let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
      client.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
      for piece in pieces { client.urlProtocol(self, didLoad: piece) }
      client.urlProtocolDidFinishLoading(self)
    }
    switch reply {
    case .body(let data): respond(200, [data])
    case .chunks(let pieces): respond(200, pieces)
    case .status(let status): respond(status, [Data("error \(status)".utf8)])
    case .failure, nil:
      client.urlProtocol(self, didFailWithError: URLError(.cannotFindHost, userInfo: [NSURLErrorFailingURLStringErrorKey: url.absoluteString]))
    }
  }

  override func stopLoading() {}

  private static func drain(_ stream: InputStream) -> Data {
    stream.open()
    defer { stream.close() }
    var out = Data()
    var buffer = [UInt8](repeating: 0, count: 4096)
    while true {
      let n = stream.read(&buffer, maxLength: buffer.count)
      if n <= 0 { break }
      out.append(buffer, count: n)
    }
    return out
  }
}

// FetchTests' loopback server speaks Darwin sockets; the Linux build
// (docs/conformance.md) runs the registry's conformance tests without it.
#if canImport(Darwin)
  // MARK: - a real HTTP server on loopback

  /// Serves `total` bytes of a repeating pattern to every GET, over a real
  /// socket, so a download goes through URLSession's own HTTP stack. Nothing is
  /// read from disk on the serving side.
  final class LocalServer: Sendable {
    let port: UInt16
    let total: Int64
    let pattern: Data
    private let listener: Int32

    init(total: Int64, patternBytes: Int = 1 << 20) throws {
      self.total = total
      var generator = SystemRandomNumberGenerator()
      pattern = Data((0..<patternBytes).map { _ in UInt8.random(in: 0...255, using: &generator) })

      let fd = socket(AF_INET, SOCK_STREAM, 0)
      guard fd >= 0 else { throw POSIXError(.EIO) }
      var yes: Int32 = 1
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
      var address = sockaddr_in()
      address.sin_family = sa_family_t(AF_INET)
      address.sin_addr.s_addr = inet_addr("127.0.0.1")
      address.sin_port = 0
      let bound = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
      }
      guard bound == 0, listen(fd, 8) == 0 else {
        close(fd)
        throw POSIXError(.EADDRINUSE)
      }
      var length = socklen_t(MemoryLayout<sockaddr_in>.size)
      _ = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
      }
      port = UInt16(bigEndian: address.sin_port)
      listener = fd

      let pattern = self.pattern
      Thread.detachNewThread {
        while true {
          let client = accept(fd, nil, nil)
          if client < 0 { return }  // the listener was closed
          Thread.detachNewThread { LocalServer.serve(client, total: total, pattern: pattern) }
        }
      }
    }

    var url: String { "http://127.0.0.1:\(port)/object" }

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

    func stop() {
      shutdown(listener, SHUT_RDWR)
      close(listener)
    }

    private static func serve(_ client: Int32, total: Int64, pattern: Data) {
      defer { close(client) }
      var yes: Int32 = 1
      // A client that hangs up mid-body must not take the test process with SIGPIPE.
      setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
      var request = Data()
      var buffer = [UInt8](repeating: 0, count: 4096)
      while request.range(of: Data("\r\n\r\n".utf8)) == nil {
        let n = read(client, &buffer, buffer.count)
        if n <= 0 { return }
        request.append(buffer, count: n)
      }
      let header = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: \(total)\r\nConnection: close\r\n\r\n"
      guard send(client, Data(header.utf8)) else { return }
      var left = total
      while left > 0 {
        let n = Int(min(Int64(pattern.count), left))
        guard send(client, pattern.prefix(n)) else { return }
        left -= Int64(n)
      }
    }

    private static func send(_ fd: Int32, _ data: Data) -> Bool {
      data.withUnsafeBytes { raw -> Bool in
        guard var base = raw.baseAddress else { return true }
        var left = raw.count
        while left > 0 {
          let n = write(fd, base, left)
          if n <= 0 { return false }
          base += n
          left -= n
        }
        return true
      }
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
