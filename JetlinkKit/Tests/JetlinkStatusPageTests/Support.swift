import Foundation
import JetlinkKit
import JetlinkServer
import JetlinkTestSupport

@testable import JetlinkStatusPage

#if canImport(Glibc)
  import Glibc
#elseif canImport(Darwin)
  import Darwin
#endif

/// Hardware that counts what the page asks of it.
final class FakeHardware: PageHardwareSource, @unchecked Sendable {
  private let lock = NSLock()
  private var counts = (host: 0, samples: 0, resets: 0)

  var hosts: Int { lock.withLock { counts.host } }
  var samples: Int { lock.withLock { counts.samples } }
  var resets: Int { lock.withLock { counts.resets } }

  func host() -> [String: Any] {
    lock.withLock { counts.host += 1 }
    return ["hostname": "bench", "jetson": true]
  }

  func sample() -> [String: Any] {
    let n = lock.withLock {
      counts.samples += 1
      return counts.samples
    }
    return ["cpu": [["online": true, "load": 5.0]], "n": n]
  }

  func reset() {
    lock.withLock { counts.resets += 1 }
  }
}

/// A registry that counts every call: the page may read the inventory, and
/// must never reach the catalog, a download or an import.
final class CountingRegistry: ModelRegistry, @unchecked Sendable {
  private let lock = NSLock()
  private var calls: [String: Int] = [:]

  func count(_ name: String) -> Int { lock.withLock { calls[name] ?? 0 } }
  private func called(_ name: String) { lock.withLock { calls[name, default: 0] += 1 } }

  /// Everything but reading the inventory: what touches the network or the disk's models.
  var reachedOut: Int {
    lock.withLock { calls.filter { $0.key != "inventory" }.values.reduce(0, +) }
  }

  func cachedCatalog() -> CatalogEvent? {
    called("cachedCatalog")
    return nil
  }

  func catalog(refresh: Bool, maxAge: TimeInterval) async -> CatalogEvent {
    called("catalog")
    return CatalogEvent(fetchedAt: nil, url: "", defaultRef: "", error: nil, models: [])
  }

  func resolveMissingPointers(_ refs: [String]) async { called("resolveMissingPointers") }

  func resolvePointer(ref: String) async throws -> (sha256: String, size: Int64) {
    called("resolvePointer")
    throw TestError("no network here")
  }

  func ref(for sha256: String) -> String? {
    called("ref")
    return nil
  }

  func fetch(_ refOrSHA256: String, progress: @escaping @Sendable (Double) -> Void, shouldStop: @escaping @Sendable () -> Bool) async throws -> URL {
    called("fetch")
    throw TestError("no network here")
  }

  func importModelFile(at url: URL, name: String?, progress: @escaping @Sendable (Double) -> Void) async throws -> String {
    called("import")
    throw TestError("no imports here")
  }

  func inventory(artifactTag: String?, artifactSuffix: String, loaded: String?) -> InventoryEvent {
    called("inventory")
    return InventoryEvent(
      loaded: loaded, lastLoaded: nil, models: [], artifacts: [], disk: InventoryDisk(modelsBytes: 0, enginesBytes: 0, freeBytes: 1_000_000_000))
  }

  func remove(sha256: String, artifacts: Bool, model: Bool) throws { called("remove") }
}

// Where PageServer is: Linux and macOS, not Android.
#if canImport(Darwin) || canImport(Glibc)
  /// A page server on a free port of its own, stopped with the value.
  final class RunningPage {
    let feed: PageFeed
    let server: PageServer

    init(headerTimeout: TimeInterval = 30, keepalive: TimeInterval = 15, logs: @escaping @Sendable () -> [String] = { [] }) throws {
      feed = PageFeed(hardware: nil)
      server = try PageServer(
        port: 0, page: Data("<!doctype html><p>page</p>".utf8), feed: feed, headerTimeout: headerTimeout, keepalive: keepalive, logs: logs)
    }

    var port: UInt16 { server.port }

    deinit {
      server.stop()
    }
  }

  /// A raw HTTP client, so a test sees exactly the bytes the page sends.
  final class Client {
    let fd: Int32
    private(set) var received = Data()
    private(set) var closed = false

    init(port: UInt16) throws {
      fd = socket(AF_INET, Sys.stream, 0)
      guard fd >= 0 else { throw TestError("socket failed") }
      #if canImport(Darwin)
        Sys.set(fd, SOL_SOCKET, SO_NOSIGPIPE, 1)
      #endif
      var address = sockaddr_in()
      #if canImport(Darwin)
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
      #endif
      address.sin_family = sa_family_t(AF_INET)
      address.sin_port = port.bigEndian
      address.sin_addr.s_addr = inet_addr("127.0.0.1")
      let connected = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
      }
      guard connected == 0 else {
        close(fd)
        throw TestError("cannot connect to port \(port): \(String(cString: strerror(errno)))")
      }
    }

    deinit {
      close(fd)
    }

    var text: String { String(decoding: received, as: UTF8.self) }

    func send(_ text: String) {
      let bytes = Array(text.utf8)
      var offset = 0
      while offset < bytes.count {
        #if canImport(Glibc)
          let n = bytes[offset...].withUnsafeBytes { Glibc.send(fd, $0.baseAddress, $0.count, Sys.sendFlags) }
        #else
          let n = bytes[offset...].withUnsafeBytes { Darwin.send(fd, $0.baseAddress, $0.count, Sys.sendFlags) }
        #endif
        if n <= 0 { return }
        offset += n
      }
    }

    /// Reads until `done` holds for all received so far, the page closes the
    /// connection, or `timeout` passes. Returns everything received.
    @discardableResult
    func read(timeout: TimeInterval = 5, until done: (String) -> Bool = { _ in false }) -> String {
      let deadline = Date(timeIntervalSinceNow: timeout)
      var chunk = [UInt8](repeating: 0, count: 65536)
      while !closed && !done(text) {
        let left = deadline.timeIntervalSinceNow
        if left <= 0 { break }
        var poller = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
        if poll(&poller, 1, Int32(left * 1000) + 1) <= 0 { continue }
        let n = chunk.withUnsafeMutableBytes { recv(fd, $0.baseAddress, $0.count, 0) }
        if n <= 0 {
          closed = true
        } else {
          received.append(contentsOf: chunk[..<n])
        }
      }
      return text
    }

    /// One whole request and its whole reply, read until the page closes.
    static func get(_ port: UInt16, _ path: String, method: String = "GET") throws -> String {
      let client = try Client(port: port)
      client.send("\(method) \(path) HTTP/1.1\r\nHost: jetlink.local\r\n\r\n")
      return client.read()
    }
  }
#endif

/// The JSON objects in a stream's `data:` lines, in order.
func dataEvents(_ text: String) -> [[String: Any]] {
  text.components(separatedBy: "\n\n").compactMap { block in
    guard let line = block.split(separator: "\n").last(where: { $0.hasPrefix("data: ") }) else { return nil }
    return (try? JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8))) as? [String: Any]
  }
}

func names(_ text: String) -> [String] {
  dataEvents(text).compactMap { $0["event"] as? String }
}

func stats(frames: Int, p99: Double = 30) -> ControlEvent {
  .stats(
    StatsEvent(
      frames: frames, fps: 20, servedMs: .init(mean: 25, p99: p99, max: p99 + 1), stagesMs: .init(queue: 1, gpu: 20, other: 2, send: 2), slow: 0,
      windowS: 1, totalMs: .init(mean: 23, p99: p99 - 2, max: p99), gpuMs: .init(mean: 20)))
}
