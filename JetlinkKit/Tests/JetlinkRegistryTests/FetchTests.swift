import Foundation
import Testing

@testable import JetlinkRegistry

/// tests/test_registry.py's fetch section, plus what the Swift download adds:
/// progress by whole percent, a real socket, and no .part left on any failure.
struct FetchTests {
  @Test func fallsThroughToTheServerThatHasIt() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let net = MockNet(smallRoutes())
    let registry = Registry(layout: tmp.layout, session: net.session)
    let seen = ProgressLog()

    let path = try await registry.fetch(smallRef, progress: seen.callback)

    #expect(path == (try registry.modelPath(sha256: fixtureBlobSHA)))
    #expect(try Data(contentsOf: path) == fixtureBlob)
    #expect(tmp.names("models") == ["\(fixtureBlobSHA.prefix(16)).onnx"])
    #expect(seen.all.last == 1.0)
    #expect(net.urls.contains("\(LFS.endpoints[0])/objects/batch"))
    #expect(!net.urls.contains("\(LFS.endpoints[2])/objects/batch"), "the first server with the object serves it")
  }

  @Test func aModelAlreadyOnDiskTouchesNoNetwork() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    _ = try await Registry(layout: tmp.layout, session: MockNet(smallRoutes()).session).fetch(smallRef)
    let offline = MockNet([LFS.pointerURL(ref: smallRef): .body(smallPointerText())])
    let path = try await Registry(layout: tmp.layout, session: offline.session).fetch(smallRef)
    #expect(try Data(contentsOf: path) == fixtureBlob)
    // the pointer was kept, so even that is not asked for again
    #expect(offline.calls.isEmpty)
  }

  @Test func aWrongHashLeavesNothingBehind() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let net = MockNet(smallRoutes(oid: String(repeating: "b", count: 64)))
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: net.session).fetch(smallRef)
    }
    #expect(error?.kind == .verify)
    #expect(error?.message.contains("hash") == true)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func aShortDownloadLeavesNothingBehind() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let net = MockNet(smallRoutes(size: Int64(fixtureBlob.count) + 99))
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: net.session).fetch(smallRef)
    }
    #expect(error?.kind == .verify)
    #expect(error?.message.contains("bytes") == true)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func aCancelledDownloadLeavesNothingBehind() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let stops = StopScript([false, true, true])
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet(smallRoutes()).session).fetch(smallRef, shouldStop: stops.callback)
    }
    #expect(error?.kind == .cancelled)
    #expect(error?.message.contains("cancelled") == true)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func aCancelledTaskStopsTheDownload() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let server = try LocalServer(total: 64 << 20)
    defer { server.stop() }
    let pointer = Pointer(oid: String(repeating: "c", count: 64), size: server.total)
    let dest = tmp.url.appending(path: "models/cccccccccccccccc.onnx")
    let task = Task {
      try await LFS.download(href: server.url, pointer: pointer, dest: dest, session: .shared, progress: { _ in }, shouldStop: { false })
    }
    try await Task.sleep(for: .milliseconds(20))
    task.cancel()
    let result = await task.result
    #expect(throws: RegistryError.self) { try result.get() }
    #expect(tmp.names("models").isEmpty)
  }

  @Test func noServerHasIt() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    var routes = smallRoutes()
    routes["\(LFS.endpoints[1])/objects/batch"] = .body(batch(oid: fixtureBlobSHA, size: Int64(fixtureBlob.count), href: nil))
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet(routes).session).fetch(smallRef)
    }
    #expect(error?.isNetwork == true)
    #expect(error?.message.contains("no LFS server") == true)
  }

  @Test func anHrefThatFailsIsANetworkErrorAndLeavesNothing() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    var routes = smallRoutes()
    routes["https://blob.example/object"] = .status(403)
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet(routes).session).fetch(smallRef)
    }
    #expect(error?.kind == .network)
    #expect(error?.message.hasPrefix("could not download \(fixtureBlobSHA.prefix(16)): HTTP Error 403") == true)
    #expect(tmp.names("models").isEmpty)

    routes["https://blob.example/object"] = .failure
    let dropped = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet(routes).session).fetch(smallRef)
    }
    #expect(dropped?.kind == .network)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func fetchingBySHA256NeedsAKnownSize() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let registry = Registry(layout: tmp.layout, session: MockNet().session)
    let unknown = await #expect(throws: RegistryError.self) { try await registry.fetch(fixtureBlobSHA) }
    #expect(unknown?.message.contains("fetch by catalog ref") == true)
    let neither = await #expect(throws: RegistryError.self) { try await registry.fetch("nothex") }
    #expect(neither?.message == "'nothex' is neither a 40 character ref nor a 64 character sha256")
  }

  @Test func aKnownSHA256IsFetchedByItsPointer() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let net = MockNet(smallRoutes())
    let registry = Registry(layout: tmp.layout, session: net.session)
    _ = try await registry.resolve(ref: smallRef)
    let path = try await registry.fetch(fixtureBlobSHA)
    #expect(try Data(contentsOf: path) == fixtureBlob)
  }

  @Test func tooLittleDiskIsRefusedBeforeAnyBytes() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let huge = Pointer(oid: fixtureBlobSHA, size: 1 << 60)
    let error = await #expect(throws: RegistryError.self) {
      try await LFS.download(
        href: "https://blob.example/object", pointer: huge, dest: tmp.url.appending(path: "models/x.onnx"),
        session: MockNet().session, progress: { _ in }, shouldStop: { false })
    }
    #expect(error?.message.hasPrefix("need \(Int64(1 << 60) >> 20) MB for the model") == true)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func progressIsReportedByWholePercent() async throws {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let blob = Data((0..<(1 << 20)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
    let pieces = stride(from: 0, to: blob.count, by: 1000).map { blob.subdata(in: $0..<min($0 + 1000, blob.count)) }
    var routes = smallRoutes(oid: sha256Hex(blob), size: Int64(blob.count), blob: blob)
    routes["https://blob.example/object"] = .chunks(pieces)
    let seen = ProgressLog()

    _ = try await Registry(layout: tmp.layout, session: MockNet(routes).session).fetch(smallRef, progress: seen.callback)

    let values = seen.all
    #expect(values.last == 1.0)
    #expect(values == values.sorted())
    let percents = values.dropLast().map { Int($0 * 100) }
    #expect(Set(percents).count == percents.count, "one call per whole percent")
    #expect(values.count <= 102)
  }
}

/// The download through URLSession's real HTTP stack, from a server on loopback.
struct LocalDownloadTests {
  static let benchmark = ProcessInfo.processInfo.environment["JETLINK_BENCH"] == "1"

  private func download(megabytes: Int64, session: URLSession) async throws -> (seconds: Double, progress: [Double]) {
    let tmp = try TempDir()
    defer { tmp.remove() }
    let server = try LocalServer(total: megabytes << 20)
    defer { server.stop() }
    let pointer = Pointer(oid: server.sha256, size: server.total)
    let dest = try tmp.layout.modelPath(sha256: pointer.oid)
    let seen = ProgressLog()

    let clock = ContinuousClock()
    let start = clock.now
    let path = try await LFS.download(href: server.url, pointer: pointer, dest: dest, session: session, progress: seen.callback, shouldStop: { false })
    let elapsed = clock.now - start

    #expect(path == dest)
    #expect(Files.status(dest)?.size == server.total)
    #expect(tmp.names("models") == [dest.lastPathComponent])
    let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    return (seconds, seen.all)
  }

  @Test func streamsHashesAndVerifiesOverARealSocket() async throws {
    let (seconds, progress) = try await download(megabytes: 16, session: .shared)
    #expect(progress.last == 1.0)
    #expect(progress.count >= 2 && progress.count <= 102)
    print("local download: 16 MB in \(String(format: "%.3f", seconds)) s, \(String(format: "%.0f", 16 / seconds)) MB/s")
  }

  /// JETLINK_BENCH=1 swift test --filter throughput. JETLINK_BENCH_MB sets the
  /// size (256 by default); it is written to the temp dir and deleted.
  @Test(.enabled(if: benchmark)) func throughput() async throws {
    let megabytes = Int64(ProcessInfo.processInfo.environment["JETLINK_BENCH_MB"] ?? "") ?? 256
    let configuration = URLSessionConfiguration.ephemeral
    for (label, session) in [("shared", URLSession.shared), ("ephemeral", URLSession(configuration: configuration))] {
      let (seconds, _) = try await download(megabytes: megabytes, session: session)
      let rate = Double(megabytes) / seconds
      print("local download (\(label) session): \(megabytes) MB in \(String(format: "%.3f", seconds)) s = \(String(format: "%.0f", rate)) MB/s")
      #expect(rate >= 100, "the download must not be per-byte slow")
    }
  }
}
